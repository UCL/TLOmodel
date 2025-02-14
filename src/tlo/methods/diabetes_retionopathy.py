from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Simulation, Parameter, Types, Property, Population, logging
from tlo.events import RegularEvent, PopulationScopeEventMixin, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import Metadata
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Diabetes_Retinopathy(Module):
    """ This is Diabetes Retinopathy module. It seeks to skeleton of blindness due to diabetes. """

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    # define a dictionary of parameters this module will use
    PARAMETERS = {
        'rate_onset_to_early_dr': Parameter(Types.REAL,
                                            'Probability of people who get diagnosed with early diabetic retinopathy'),
        'rate_progression_to_dr': Parameter(Types.REAL,
                                            'Probability of people who get diagnosed with late diabetic retinopathy'),
        'prob_fast_dr': Parameter(Types.REAL,
                                  'Probability of people who get diagnosed from none phase to late diabetic retinopathy stage'),
        'init_prob_any_dr': Parameter(Types.REAL, 'Initial probability of anyone with diabetic retinopathy'),
        'init_prob_late_dr': Parameter(Types.REAL,
                                       'Initial probability of people with diabetic retinopathy in the late stage'),
        'p_medication': Parameter(Types.REAL, 'Diabetic retinopathy treatment/medication effectiveness'),
    }

    # define a dictionary of properties this module will use
    PROPERTIES = {
        "dr_status": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "early",
                "late",
            ],
            description="dr status",
        ),
        'dr_on_treatment': Property(
            Types.BOOL, 'Whether this person is on diabetic retinopathy treatment',
        ),
        'dr_date_treatment': Property(
            Types.DATE,
            'date of first receiving diabetic retinopathy treatment (pd.NaT if never started treatment)'
        ),
        'dr_early_diagnosed': Property(
            Types.BOOL, 'Whether this person will die during a current severe exacerbation'
        ),
        'dr_late_diagnosed': Property(
            Types.BOOL, 'Whether this person will die during a current severe exacerbation'
        ),
        'dr_diagnosed': Property(
            Types.BOOL, 'Whether this person has been diagnosed with any diabetic retinopathy'
        ),

    }

    def __init__(self):
        # this method is included to define all things that should be initialised first
        super().__init__()

    def read_parameters(self, data_folder: str | Path) -> None:
        """ initialise module parameters. Here we are assigning values to all parameters defined at the beginning of
        this module. For this demo module, we will manually assign values to parameters but in the
        Thanzi model we do this by reading from an Excel file containing parameter names and values

        :param data_folder: Path to the folder containing parameter values

        """
        self.parameters['rate_onset_to_early_dr'] = 0.5
        self.parameters['rate_progression_to_dr'] = 0.5
        self.parameters['prob_fast_dr'] = 0.5
        self.parameters['init_prob_any_dr'] = 0.5
        self.parameters['init_prob_late_dr'] = 0.5
        self.parameters['p_medication'] = 0.4

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='blindness_partial'),
            Symptom(name='blindness_full')
        )

    def initialise_population(self, population: Population) -> None:
        """ set the initial state of the population. The state will be update over time

        :param population: all individuals in the model

        """
        df = population.props
        p = self.parameters

        alive_diabetes_idx = df.loc[df.is_alive & df.nc_diabetes].index

        any_dr_idx = alive_diabetes_idx[
            self.rng.random_sample(size=len(alive_diabetes_idx)) < self.parameters['init_prob_any_dr']
            ]
        no_dr_idx = set(alive_diabetes_idx) - set(any_dr_idx)

        late_dr_idx = any_dr_idx[
            self.rng.random_sample(size=len(any_dr_idx)) < self.parameters['init_prob_late_dr']
            ]

        early_dr_idx = set(any_dr_idx) - set(late_dr_idx)

        # write to property:
        df.loc[df.is_alive & ~df.nc_diabetes, 'dr_status'] = 'none'
        df.loc[list(no_dr_idx), 'dr_status'] = 'none'
        df.loc[list(early_dr_idx), "dr_status"] = "early"
        df.loc[list(late_dr_idx), "dr_status"] = "late"
        df.loc[list(alive_diabetes_idx), "dr_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dr_diagnosed"] = False
        df.loc[list(alive_diabetes_idx), "dr_date_treatment"] = pd.NaT

    def initialise_simulation(self, sim: Simulation) -> None:
        """ This is where you should include all things you want to be happening during simulation

        :param sim: simulation object

        """
        sim.schedule_event(DrPollEvent(self), date=sim.date)
        sim.schedule_event(DiabeticRetinopathyLoggingEvent(self), sim.date + DateOffset(months=1))

        self.make_the_linear_models()

    def report_daly_values(self) -> pd.Series:
        return pd.Series(index=self.sim.population.props.index, data=0.0)

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """ set properties of a child when they are born. """
        self.sim.population.props.at[child_id, 'dr_status'] = 'none'
        self.sim.population.props.at[child_id, 'dr_on_treatment'] = False
        self.sim.population.props.at[child_id, 'dr_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_diagnosed'] = False
        self.sim.population.props.at[child_id, 'dr_diagnosis_date'] = pd.NaT

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Here is where we make and save LinearModels that will be used when the module is running"""

        self.lm = dict()

        self.lm['onset_early_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_onset_to_early_dr']
        )

        self.lm['onset_late_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_progression_to_dr']
        )

        self.lm['onset_fast_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['prob_fast_dr']
        )


class DrPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retionpathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        """

        """
        df = population.props

        diabetes_and_alive_nodr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'none')]
        diabetes_and_alive_earlydr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'early')]

        will_progress = self.module.lm['onset_early_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # will_progress_idx = will_progress[will_progress].index
        # print(f'Will Progress: {will_progress_idx}')
        # will_progress_idx = df.index[np.where(will_progress)[0]]
        will_progress_idx = df.index[np.where(will_progress)[0]]
        print(f'New Will Progress: {will_progress_idx}')
        old_will_progress_idx = will_progress[will_progress].index
        print(f'Old Will Progress: {old_will_progress_idx}')
        df.loc[will_progress_idx, 'dr_status'] = 'early'

        early_to_late = self.module.lm['onset_late_dr'].predict(diabetes_and_alive_earlydr, self.module.rng)
        # early_to_late_idx = early_to_late[early_to_late].index
        early_to_late_idx = df.index[np.where(early_to_late)[0]]
        df.loc[early_to_late_idx, 'dr_status'] = 'late'

        fast_dr = self.module.lm['onset_fast_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # fast_dr_idx = fast_dr[fast_dr].index
        fast_dr_idx = df.index[np.where(fast_dr)[0]]
        df.loc[fast_dr_idx, 'dr_status'] = 'late'

        if len(will_progress_idx):
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=will_progress_idx,
                symptom_string='blindness_partial',
                add_or_remove='+',
                disease_module=self.module,
            )

        if len(early_to_late_idx):
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=early_to_late_idx,
                symptom_string='blindness_full',
                add_or_remove='+',
                disease_module=self.module,
            )

    def do_at_generic_first_appt_emergency(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        # Example for mockitis
        if "blindness_full" in symptoms or "blindness_partial" in symptoms:
            event = HSI_Dr_StartTreatment(
                module=self,
                person_id=person_id,
            )
            schedule_hsi_event(event, priority=1, topen=self.sim.date)


class HSI_Dr_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is inititaed.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Diabetes_Retinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Treatment_Initiation'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Dr_StartTreatment: initiating treatment for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        treatment_slows_progression_to_late = self.module.rng.rand() < self.module.parameters['p_medication']

        #TODO Add consumables in codition below
        if treatment_slows_progression_to_late:
            df.at[person_id, 'dr_on_treatment'] = True
            df.at[person_id, 'dr_date_treatment'] = self.sim.date
        #     df.at[person_id, 'dr_status'] = 'early'
        # else:
        #     df.at[person_id, 'dr_on_treatment'] = True
        #     df.at[person_id, 'dr_status'] = 'late'  #checkout cervical cancer on via after some years


class DiabeticRetinopathyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 month
        """
        self.repeat = 30
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        df = population.props

        # CURRENT STATUS COUNTS
        # Create dictionary for each subset, adding prefix to key name, and adding to make a flat dict for logging.
        out = {}

        # Current counts, total
        out.update({
            f'total_{k}': v for k, v in df.loc[df.is_alive].dr_status.value_counts().items()})

        # Current counts, off treatment
        out.update({f'off_treatment_{k}': v for k, v in df.loc[df.is_alive].loc[
            pd.isnull(df.dr_date_treatment), 'dr_status'].value_counts().items()})

        # Current counts, on treatment (excl. palliative care)
        out.update({f'treatment_{k}': v for k, v in df.loc[df.is_alive].loc[(~pd.isnull(
            df.dr_date_treatment)), 'dr_status'].value_counts().items()})


        logger.info(key='summary_stats', data=out)
