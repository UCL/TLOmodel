from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import Metadata
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiabeticRetinopathy(Module):
    """ This is Diabetic Retinopathy module. It seeks to model of blindness due to diabetes. """

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    PARAMETERS = {
        'rate_onset_to_early_dr': Parameter(Types.REAL,
                                            'Probability of people who get diagnosed with early diabetic retinopathy'),
        'rate_progression_to_dr': Parameter(Types.REAL,
                                            'Probability of people who get diagnosed with late diabetic retinopathy'),
        'prob_fast_dr': Parameter(Types.REAL,
                                  'Probability of people who get diagnosed from none phase to late diabetic '
                                  'retinopathy stage'),
        'init_prob_any_dr': Parameter(Types.REAL, 'Initial probability of anyone with diabetic retinopathy'),
        'init_prob_late_dr': Parameter(Types.REAL,
                                       'Initial probability of people with diabetic retinopathy in the late stage'),
        'p_medication': Parameter(Types.REAL, 'Diabetic retinopathy treatment/medication effectiveness'),
    }

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
            Types.BOOL, 'Whether this person has been diagnosed with early diabetic retinopathy'
        ),
        'dr_late_diagnosed': Property(
            Types.BOOL, 'Whether this person has been diagnosed with late diabetic retinopathy'
        ),
        'dr_diagnosed': Property(
            Types.BOOL, 'Whether this person has been diagnosed with any diabetic retinopathy'
        ),
        "dr_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of diabetic retinopathy (pd.NaT if never diagnosed)"
        ),
        "dr_blindness_investigated": Property(
            Types.BOOL,
            "whether blindness has been investigated, and diabetic retinopathy missed"
        ),

    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)

    def read_parameters(self, data_folder: str | Path) -> None:
        """ initialise module parameters. Here we are assigning values to all parameters defined at the beginning of
        this module.

        :param data_folder: Path to the folder containing parameter values

        """
        #TODO Read from resourcefile
        self.parameters['rate_onset_to_early_dr'] = 0.29
        self.parameters['rate_progression_to_dr'] = 0.07
        self.parameters['prob_fast_dr'] = 0.5
        self.parameters['init_prob_any_dr'] = 0.36
        self.parameters['init_prob_late_dr'] = 0.09
        self.parameters['p_medication'] = 0.8

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='blindness_partial'),
            Symptom(name='blindness_full')
        )

    def initialise_population(self, population: Population) -> None:
        """ Set property values for the initial population

        :param population: all individuals in the model

        """
        df = population.props
        # p = self.parameters

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
        df.loc[list(alive_diabetes_idx), "dr_date_diagnosis"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_blindness_investigated"] = False

    def initialise_simulation(self, sim: Simulation) -> None:
        """ This is where you should include all things you want to be happening during simulation
        * Schedule the main polling event
        * Schedule the main logging event
        * Call the LinearModels
        """
        sim.schedule_event(DrPollEvent(self), date=sim.date)
        sim.schedule_event(DiabeticRetinopathyLoggingEvent(self), sim.date + DateOffset(months=1))
        self.make_the_linear_models()
        self.look_up_consumable_item_codes()

    def report_daly_values(self) -> pd.Series:
        return pd.Series(index=self.sim.population.props.index, data=0.0)

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """ set properties of a child when they are born.
        :param child_id: the new child
        """
        self.sim.population.props.at[child_id, 'dr_status'] = 'none'
        self.sim.population.props.at[child_id, 'dr_on_treatment'] = False
        self.sim.population.props.at[child_id, 'dr_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_diagnosed'] = False
        self.sim.population.props.at[child_id, 'dr_date_diagnosis'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_blindness_investigated'] = False

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""

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

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()
        # self.cons_item_codes['laser_photocoagulation'] = {
        #         get_item_codes("Anesthetic Eye drops, 15ml"): 3,
        #         get_item_codes('Disposables gloves, powder free, 100 pieces per box'): 1,
        #     }
        self.cons_item_codes['eye_injection'] = {
                get_item_codes("Anesthetic Eye drops, 15ml"): 1,
                get_item_codes('Aflibercept, 2mg'): 3,
            }



class DrPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retinopathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        df = population.props

        diabetes_and_alive_nodr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'none')]
        diabetes_and_alive_earlydr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'early')]

        will_progress = self.module.lm['onset_early_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # will_progress_idx = will_progress[will_progress].index
        # will_progress_idx = df.index[np.where(will_progress)[0]]
        will_progress_idx = df.index[np.where(will_progress)[0]]
        # old_will_progress_idx = will_progress[will_progress].index
        # Count new early cases for incidence tracking
        new_early_cases = len(will_progress_idx)
        df.loc[will_progress_idx, 'dr_status'] = 'early'

        early_to_late = self.module.lm['onset_late_dr'].predict(diabetes_and_alive_earlydr, self.module.rng)
        # early_to_late_idx = early_to_late[early_to_late].index
        early_to_late_idx = df.index[np.where(early_to_late)[0]]
        # Count new late cases for incidence tracking
        new_late_cases = len(early_to_late_idx)
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

        if "blindness_full" in symptoms or "blindness_partial" in symptoms:
            event = HSI_Dr_TestingFollowingSymptoms(
                module=self,
                person_id=person_id,
            )
            schedule_hsi_event(event, priority=1, topen=self.sim.date)



class HSI_Dr_TestingFollowingSymptoms(HSI_Event, IndividualScopeEventMixin):
    """
        This event is scheduled by do_at_generic_first_appt_emergency following presentation for care with the symptoms
        of partial and full blindness.
        This event begins the investigation that may result in diagnosis of Diabetic Retinopathy and the scheduling of
        treatment.
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "DiabeticRetinopathy_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Check that this event has been called for someone with the symptom blindness_partial
        assert 'blindness_partial' in self.sim.modules['SymptomManager'].has_what(person_id=person_id)
        # Check that this event has been called for someone with the symptom blindness_full
        assert 'blindness_full' in self.sim.modules['SymptomManager'].has_what(person_id=person_id)

        # If the person is already diagnosed, then take no action:
        if not pd.isnull(df.at[person_id, "dr_date_diagnosis"]):
            return hs.get_blank_appt_footprint()

        df.at[person_id, 'dr_blindness_investigated'] = True

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['eye_injection']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='dilated_eye_exam_dr_blindness',
            hsi_event=self
        )
        # TODO Those in early DR must not go to start treatment since late DR Work can be managed with good blood
        #  sugar control to slow the progression.
        if dx_result and is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'dr_date_diagnosis'] = self.sim.date
            # If consumables are available, add equipment used and run dx_test
            self.add_equipment({'Ophthalmoscope'})

            hs.schedule_hsi_event(
                hsi_event=HSI_Dr_StartTreatment(
                    module=self.module,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )




#TODO HSI_Dr_StartTreatment should be called by HSI_Dr_TestingFollowingSymptoms
class HSI_Dr_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event initiates the treatment of DR.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

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

        randomly_sampled = self.module.rng.rand()
        treatment_slows_progression_to_late = randomly_sampled < self.module.parameters['p_medication']

        #TODO Add consumables in codition below
        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['eye_injection']
        )

        if treatment_slows_progression_to_late:
            df.at[person_id, 'dr_on_treatment'] = True
            df.at[person_id, 'dr_date_treatment'] = self.sim.date

            # Reduce probability of progression to "late" DR
            progression_chance = self.module.parameters['rate_progression_to_dr'] * (
                1 - self.module.parameters['p_medication'])

            # Determine if person will still progress
            if self.module.rng.rand() < progression_chance:
                df.at[person_id, 'dr_status'] = 'late'
            else:
                df.at[person_id, 'dr_status'] = 'early'  # Stays in early stage due to medication

        else:
            # If medication is not effective, progression happens as usual
            df.at[person_id, 'dr_on_treatment'] = True
            df.at[person_id, 'dr_status'] = 'late'


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

        # Count new cases
        new_early_cases = len(df.loc[df.is_alive & (df.dr_status == 'early') & (df.dr_date_treatment == self.sim.date)])
        new_late_cases = len(df.loc[df.is_alive & (df.dr_status == 'late') & (df.dr_date_treatment == self.sim.date)])

        # Log incidence counts
        out.update({
            'new_early': new_early_cases,
            'new_late': new_late_cases
        })

        logger.info(key='summary_stats', data=out)
