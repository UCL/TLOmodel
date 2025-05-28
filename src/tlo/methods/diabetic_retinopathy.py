from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, cardio_metabolic_disorders
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.methods.symptommanager import Symptom
from tlo.population import IndividualProperties
from tlo.util import transition_states

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
        "rate_onset_to_mild_dr": Parameter(Types.REAL,
                                           "Probability of people who get diagnosed with mild diabetic retinopathy"),
        "rate_mild_to_moderate": Parameter(Types.REAL,
                                           "Probability of people who get diagnosed with moderate diabetic retinopathy"),
        "rate_moderate_to_severe": Parameter(Types.REAL,
                                             "Probability of people who get diagnosed with severe diabetic retinopathy"),
        "rate_severe_to_proliferative": Parameter(Types.REAL,
                                                  "Probability of people who get diagnosed with proliferative "
                                                  "diabetic retinopathy"),
        'prob_fast_dr': Parameter(Types.REAL,
                                  "Probability of people who get diagnosed from none phase to proliferative diabetic "
                                  "retinopathy stage"),

        "init_prob_any_dr": Parameter(Types.LIST, "Initial probability of anyone with diabetic retinopathy"),
        "prob_any_dmo": Parameter(Types.LIST, "Probability of anyone with diabetic retinopathy having Diabetic "
                                              "Macular Oedema (DMO)"),
        # "init_prob_proliferative_dr": Parameter(Types.REAL, "Initial probability of people with diabetic
        # retinopathy in the proliferative stage"),
        "p_medication": Parameter(Types.REAL, "Diabetic retinopathy treatment/medication effectiveness"),
        "init_prob_ever_diet_mgmt_if_diagnosed": Parameter(
            Types.REAL, "Initial probability of ever having had a diet management session if ever diagnosed "
                        "with diabetic retinopathy"
        ),
        "prob_reg_eye_exam": Parameter(
            Types.REAL, "Probability of people with Diabetes Mellitus selected for regular eye exam"
        ),
        "rp_dr_tobacco": Parameter(
            Types.REAL, "relative prevalence at baseline of diabetic retinopathy if tobacco"
        ),
        "rp_dr_ex_alc": Parameter(
            Types.REAL, "relative prevalence at baseline of diabetic retinopathy if excessive alcohol"
        ),
        "rp_dr_high_sugar": Parameter(
            Types.REAL, "relative prevalence at baseline of diabetic retinopathy if high sugar"
        ),
        "rp_dr_low_ex": Parameter(
            Types.REAL, "relative prevalence at baseline of diabetic retinopathy if low exercise"
        ),
        "rp_dr_urban": Parameter(
            Types.REAL, "relative prevalence at baseline of diabetic retinopathy if urban"
        ),
        'effectiveness_of_laser_photocoagulation_in_severe_regression': Parameter(
            Types.REAL,
            'Probability of severe diabetic retinopathy regressing to moderate.'),
        "probs_for_dmo_when_dr_status_mild": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has mild Diabetic Retinopathy "),
        "probs_for_dmo_when_dr_status_moderate": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has mild Diabetic Retinopathy "),
        "probs_for_dmo_when_dr_status_severe": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has severe Diabetic Retinopathy "),
        "probs_for_dmo_when_dr_status_proliferative": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has proliferative Diabetic Retinopathy "),
    }

    PROPERTIES = {
        "dr_status": Property(
            Types.CATEGORICAL,
            "DR status",
            categories=["none", "mild", "moderate", "severe", "proliferative"],
        ),
        "dmo_status": Property(
            Types.CATEGORICAL,
            "DMO status. Only occurs to people with any type of Diabetic Retinopathy.",
            categories=["none", "clinically_significant", "non_clinically_significant"],
        ),
        "dr_on_treatment": Property(
            Types.BOOL, "Whether this person is on diabetic retinopathy treatment",
        ),
        "dr_date_treatment": Property(
            Types.DATE,
            "date of first receiving diabetic retinopathy treatment (pd.NaT if never started treatment)"
        ),
        "dr_mild_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with mild diabetic retinopathy"
        ),
        "dr_proliferative_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with proliferative diabetic retinopathy"
        ),
        "dr_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with any diabetic retinopathy"
        ),
        "dr_date_diagnosis": Property(
            Types.DATE,
            "The date of diagnosis of diabetic retinopathy (pd.NaT if never diagnosed)"
        ),
        "dr_blindness_investigated": Property(
            Types.BOOL,
            "Whether blindness has been investigated, and diabetic retinopathy missed"
        ),
        "dr_ever_diet_mgmt": Property(Types.BOOL,
                                      "Whether this person has ever had a diabetic retinopathy diet management"
                                      "session in the diabetic clinic"),

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
        self.parameters['rate_onset_to_mild_dr'] = 0.29

        self.parameters['rate_mild_to_moderate'] = 0.4
        self.parameters['rate_moderate_to_severe'] = 0.5

        self.parameters['rate_severe_to_proliferative'] = 0.07

        self.parameters['prob_fast_dr'] = 0.5
        self.parameters['init_prob_any_dr'] = [0.2, 0.3, 0.3, 0.2]
        self.parameters['prob_any_dmo'] = [0.1, 0.2, 0.3, 0.4]

        self.parameters['probs_for_dmo_when_dr_status_mild'] = [0.7, 0.1, 0.2]
        self.parameters['probs_for_dmo_when_dr_status_moderate'] = [0.5, 0.3, 0.2]
        self.parameters['probs_for_dmo_when_dr_status_severe'] = [0.3, 0.5, 0.2]
        self.parameters['probs_for_dmo_when_dr_status_proliferative'] = [0.1, 0.7, 0.2]

        # self.parameters['init_prob_any_dr'] = [0.2, 0.3, 0.3, 0.15, 0.05]
        # self.parameters['init_prob_proliferative_dr'] = 0.09
        self.parameters['p_medication'] = 0.8
        self.parameters['effectiveness_of_laser_photocoagulation_in_severe_regression'] = 0.21
        self.parameters['init_prob_ever_diet_mgmt_if_diagnosed'] = 0.1
        self.parameters['prob_reg_eye_exam'] = 0.05

        self.parameters['rp_dr_ex_alc'] = 1.1
        self.parameters['rp_dr_tobacco'] = 1.3
        self.parameters['rp_dr_high_sugar'] = 1.2
        self.parameters['rp_dr_low_ex'] = 1.3
        self.parameters['rp_dr_urban'] = 1.4

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

        # any_dr_idx = alive_diabetes_idx[
        #     self.rng.random_sample(size=len(alive_diabetes_idx)) < self.parameters['init_prob_any_dr']
        #     ]
        # no_dr_idx = set(alive_diabetes_idx) - set(any_dr_idx)
        #
        # proliferative_dr_idx = any_dr_idx[
        #     self.rng.random_sample(size=len(any_dr_idx)) < self.parameters['init_prob_proliferative_dr']
        #     ]
        #
        # mild_dr_idx = set(any_dr_idx) - set(proliferative_dr_idx)

        # write to property:
        df.loc[df.is_alive & ~df.nc_diabetes, 'dr_status'] = 'none'
        df.loc[df.is_alive & ~df.nc_diabetes, 'dmo_status'] = 'none'

        df.loc[list(alive_diabetes_idx), "dr_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dr_diagnosed"] = False
        df.loc[list(alive_diabetes_idx), "dr_date_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_date_diagnosis"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_blindness_investigated"] = False
        df.loc[list(alive_diabetes_idx), "dr_ever_diet_mgmt"] = False

        # -------------------- dr_status -----------
        # Determine who has diabetic retinopathy at all stages:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(self.parameters['init_prob_any_dr']) <= 1.0

        lm_init_dr_status_any_dr = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(self.parameters['init_prob_any_dr']),
            Predictor('li_ex_alc').when(True, self.parameters['rp_dr_ex_alc']),
            Predictor('li_tob').when(True, self.parameters['rp_dr_tobacco']),
            Predictor('li_high_sugar').when(True, self.parameters['rp_dr_high_sugar']),
            Predictor('li_low_ex').when(True, self.parameters['rp_dr_low_ex']),
            Predictor('li_urban').when(True, self.parameters['rp_dr_urban']),
        )

        # any_dr = \
        #     lm_init_dr_status_any_dr.predict(df.loc[df.is_alive & df.nc_diabetes], self.rng)

        # Get boolean Series of who has DR
        has_dr = lm_init_dr_status_any_dr.predict(df.loc[df.is_alive & df.nc_diabetes], self.rng)

        # Get indices of those with DR
        dr_idx = has_dr[has_dr].index if has_dr.any() else pd.Index([])

        if not dr_idx.empty:
            # Get non-none categories
            categories = [cat for cat in df.dr_status.cat.categories if cat != 'none']

            # Verify probabilities match categories
            assert len(categories) == len(self.parameters['init_prob_any_dr'])

            # Normalize probabilities
            total_prob = sum(self.parameters['init_prob_any_dr'])
            probs = [p / total_prob for p in self.parameters['init_prob_any_dr']]

            # Assign DR stages
            df.loc[dr_idx, 'dr_status'] = self.rng.choice(
                categories,
                size=len(dr_idx),
                p=probs
            )

        # dr_stage_probs = self.parameters["init_prob_any_dr"]
        # Determine the stage of DR for those who have DM:
        # if any_dr.sum():
        #     categories = [cat for cat in df.dr_status.cat.categories if cat != 'none']
        #
        #     # Make sure we have the right number of probabilities
        #     assert len(categories) == len(self.parameters['init_prob_any_dr'])
        #
        #     # Normalize probabilities
        #     sum_probs = sum(self.parameters['init_prob_any_dr'])
        #     prob_by_stage_of_dr_if_dr = [p / sum_probs for p in self.parameters['init_prob_any_dr']]
        #
        #     # Assign statuses
        #     df.loc[any_dr, "dr_status"] = self.rng.choice(
        #         categories,
        #         size=any_dr.sum(),
        #         p=prob_by_stage_of_dr_if_dr
        #     )

        # sum_probs = sum(self.parameters['init_prob_any_dr'])
        # if sum_probs > 0:
        #     prob_by_stage_of_dr_if_dr = [i / sum_probs for i in self.parameters['init_prob_any_dr']]
        #     assert (sum(prob_by_stage_of_dr_if_dr) - 1.0) < 1e-10
        #     # df.loc[any_dr, "dr_status"] = self.rng.choice(
        #     #     dr_stage_probs[1:],  # exclude "none"
        #     #     size=len(df.loc[any_dr]),
        #     #     p=dr_stage_probs
        #     # )
        #     df.loc[any_dr, "dr_status"] = self.rng.choice(
        #         [val for val in df.dr_status.cat.categories if val != 'none'],
        #         size=any_dr.sum(),
        #         p=prob_by_stage_of_dr_if_dr
        #     )

        # df.loc[~any_dr, "dr_status"] = "none"

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
        self.sim.population.props.at[child_id, 'dmo_status'] = 'none'
        self.sim.population.props.at[child_id, 'dr_on_treatment'] = False
        self.sim.population.props.at[child_id, 'dr_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_diagnosed'] = False
        self.sim.population.props.at[child_id, 'dr_date_diagnosis'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_blindness_investigated'] = False
        self.sim.population.props.at[child_id, 'dr_ever_diet_mgmt'] = False

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        self.lm = dict()

        self.lm['onset_mild_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_onset_to_mild_dr']
        )

        self.lm['mild_moderate_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_mild_to_moderate']
        )

        self.lm['moderate_severe_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_moderate_to_severe']
        )

        self.lm['severe_proliferative_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['rate_severe_to_proliferative']
        )

        self.lm['ever_diet_mgmt_initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=self.parameters['init_prob_ever_diet_mgmt_if_diagnosed']
        )

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()
        self.cons_item_codes['laser_photocoagulation'] = {
            get_item_codes("Anesthetic Eye drops, 15ml"): 1,
            get_item_codes('Gloves, exam, latex, disposable, pair'): 4,
            get_item_codes('Contact lens'): 7
        }
        self.cons_item_codes['eye_injection'] = {
            get_item_codes("Anesthetic Eye drops, 15ml"): 1,
            get_item_codes('Aflibercept, 2mg'): 3
        }

    def do_recovery(self, idx: Union[list, pd.Index]):
        """Represent the recovery from diabetic retinopathy for the person_id given in `idx`.
        Recovery causes the person to move from severe to moderate"""
        df = self.sim.population.props

        # Getting those with severe and updating to moderate
        mask = df.loc[idx, 'dr_status'] == 'severe'
        df.loc[idx[mask], 'dr_status'] = 'moderate'
        df.loc[idx[mask], 'dr_date_treatment'] = self.sim.date
        df.loc[idx[mask], 'dr_on_treatment'] = True

    def do_treatment(self, person_id, prob_success):
        """For treatment of individuals with Severe DR status. If treatment is successful, regress to moderate."""
        if prob_success > self.rng.random_sample():
            self.do_recovery([person_id])

    def update_dmo_status(self):
        """Update DMO status for people with diabetic retinopathy.
        Ensures dmo_status is none when dr_status is none/nan."""
        df = self.sim.population.props

        # First reset dmo_status to 'none' for anyone without DR
        no_dr_mask = (df.dr_status == 'none') | df.dr_status.isna()
        df.loc[no_dr_mask, 'dmo_status'] = 'none'

        # Now only process people with valid DR status
        valid_dr_statuses = ['mild', 'moderate', 'severe', 'proliferative']
        dr_idx = df.loc[df.is_alive & df.dr_status.isin(valid_dr_statuses)].index

        if not dr_idx.empty:
            for person in dr_idx:
                dr_stage = df.at[person, 'dr_status']

                if dr_stage == 'mild':
                    probs = self.parameters['probs_for_dmo_when_dr_status_mild']
                elif dr_stage == 'moderate':
                    probs = self.parameters['probs_for_dmo_when_dr_status_moderate']
                elif dr_stage == 'severe':
                    probs = self.parameters['probs_for_dmo_when_dr_status_severe']
                elif dr_stage == 'proliferative':
                    probs = self.parameters['probs_for_dmo_when_dr_status_proliferative']

                df.at[person, 'dmo_status'] = self.rng.choice(
                    ['none', 'clinically_significant', 'non_clinically_significant'],
                    p=probs
                )

        invalid_cases = df[
            ((df.dr_status == 'none') | df.dr_status.isna()) &
            (df.dmo_status.isin(['clinically_significant', 'non_clinically_significant']))
            ]
        assert len(invalid_cases) == 0, (
            f"Found {len(invalid_cases)} cases where people with no DR "
            f"have DMO status: {invalid_cases[['dr_status', 'dmo_status']].to_dict()}"
        )


class DrPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retinopathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        df = population.props

        diabetes_and_alive_nodr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'none')]
        diabetes_and_alive_milddr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'mild')]
        diabetes_and_alive_moderatedr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'moderate')]
        diabetes_and_alive_severedr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'severe')]

        will_progress = self.module.lm['onset_mild_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # will_progress_idx = will_progress[will_progress].index
        will_progress_idx = df.index[np.where(will_progress)[0]]
        df.loc[will_progress_idx, 'dr_status'] = 'mild'

        mild_to_moderate = self.module.lm['mild_moderate_dr'].predict(diabetes_and_alive_milddr, self.module.rng)
        # mild_to_moderate_idx = mild_to_moderate[mild_to_moderate].index
        mild_to_moderate_idx = df.index[np.where(mild_to_moderate)[0]]
        df.loc[mild_to_moderate_idx, 'dr_status'] = 'moderate'

        moderate_to_severe = self.module.lm['moderate_severe_dr'].predict(diabetes_and_alive_moderatedr,
                                                                          self.module.rng)
        # moderate_to_severe_idx = moderate_to_severe[moderate_to_severe].index
        moderate_to_severe_idx = df.index[np.where(moderate_to_severe)[0]]
        df.loc[moderate_to_severe_idx, 'dr_status'] = 'severe'

        severe_to_proliferative = self.module.lm['severe_proliferative_dr'].predict(diabetes_and_alive_severedr,
                                                                                    self.module.rng)
        # severe_to_proliferative_idx = mild_to_moderate[severe_to_proliferative].index
        severe_to_proliferative_idx = df.index[np.where(severe_to_proliferative)[0]]
        df.loc[severe_to_proliferative_idx, 'dr_status'] = 'proliferative'

        # Update DMO status
        self.module.update_dmo_status()

        # fast_dr = self.module.lm['onset_fast_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # # fast_dr_idx = fast_dr[fast_dr].index
        # fast_dr_idx = df.index[np.where(fast_dr)[0]]
        # df.loc[fast_dr_idx, 'dr_status'] = 'late'

        # eligible_for_ns_threatening = df.loc[df.is_alive & df.nc_diabetes & (df.age_years >= 40)
        #                                      & (df.dr_status == 'none')]
        #
        # df.loc[eligible_for_ns_threatening, 'selected_for_regular_eye_exam'] = (
        #     np.random.random_sample(size=len(df[eligible_for_ns_threatening]))
        #     < self.module.parameters['prob_reg_eye_exam'])
        #
        # # Schedule HSI event for selected individuals
        # selected_for_exam = df.index[df['selected_for_regular_eye_exam']]
        # for person_id in selected_for_exam:
        #     self.sim.modules['HealthSystem'].schedule_hsi_event(
        #         hsi_event=HSI_Regular_Eye_Exam(module=self.module, person_id=person_id),
        #         priority=0,
        #         topen=self.sim.date,
        #         tclose=None
        #     )

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:

        # get the clinical states
        dr_stage = individual_properties['dr_status']

        # No interventions if well
        if dr_stage == 'none':
            return

        # Interventions for MAM
        elif dr_stage == 'mild' or dr_stage == 'moderate':
            # schedule HSI for mild and moderate
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=cardio_metabolic_disorders.HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                    person_id=person_id,
                    module=self.sim.modules["CardioMetabolicDisorders"],
                    condition='diabetes',
                ),
                priority=0,
                topen=self.sim.date
            )


        elif dr_stage == 'severe' or dr_stage == 'proliferative':
            # Interventions for severe and proliferative
            schedule_hsi_event(
                hsi_event=HSI_Dr_StartTreatment(module=self, person_id=person_id),
                priority=0, topen=self.sim.date)


class HSI_Dr_DietManagement(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event in which a person receives a session of diet management in the
    diabetes clinic. It is one of a course of 5 sessions (at months 0, 6, 12, 18, 24). If one of these HSI does not happen
    then no further sessions occur. Sessions after the first have no direct effect, as the only property affected is
    reflects ever having had one session of talking therapy."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Dr_DietManagement'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.num_of_sessions_had = 0  # A counter for the number of diet management sessions had

    def apply(self, person_id, squeeze_factor):
        """Set the property `dr_ever_diet_mgmt` to be True and schedule the next session in the course if the person
        has not yet had 5 sessions."""

        self.num_of_sessions_had += 1

        df = self.sim.population.props
        if not df.at[person_id, 'dr_ever_diet_mgmt']:
            df.at[person_id, 'dr_ever_diet_mgmt'] = True

        if self.num_of_sessions_had < 5:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=self,
                topen=self.sim.date + pd.DateOffset(months=6),
                tclose=None,
                priority=1
            )


class HSI_Dr_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event initiates the treatment of DR for severe and proliferative stages.
    This event is scheduled by HSI_GenericFirstAppt.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Treatment_Initiation'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Dr_StartTreatment: initiating treatment for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # randomly_sampled = self.module.rng.rand()
        # treatment_slows_progression_to_proliferative = randomly_sampled < self.module.parameters['p_medication']

        df.at[person_id, 'dr_blindness_investigated'] = True

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['laser_photocoagulation']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='dilated_eye_exam_dr',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'dr_date_diagnosis'] = self.sim.date
            # If consumables are available, add equipment used and run dx_test
            self.add_equipment({'Ophthalmoscope'})

            if person.dr_status == 'severe':
                #determine_effectiveness
                self.module.do_treatment(person_id, prob_success=self.module.parameters[
                    'effectiveness_of_laser_photocoagulation_in_severe_regression'])

            # if treatment_slows_progression_to_proliferative:
            #     df.at[person_id, 'dr_on_treatment'] = True
            #     df.at[person_id, 'dr_date_treatment'] = self.sim.date
            #
            #     # Reduce probability of progression to "proliferative" DR
            #     progression_chance = self.module.parameters['rate_severe_to_proliferative'] * (
            #         1 - self.module.parameters['p_medication'])
            #
            #     # Determine if person will still progress
            #     if self.module.rng.rand() < progression_chance:
            #         df.at[person_id, 'dr_status'] = 'proliferative'
            #     else:
            #         df.at[person_id, 'dr_status'] = 'mild'  # Stays in mild stage due to medication
            #
            # else:
            #     # If medication is not effective, progression happens as usual
            #     df.at[person_id, 'dr_on_treatment'] = True
            #     df.at[person_id, 'dr_status'] = 'proliferative'


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

        date_now = self.sim.date
        date_lastlog = self.sim.date - pd.DateOffset(months=self.repeat)

        n_any_dr = (df.is_alive & ((df.dr_status == "proliferative") | (df.dr_status == "mild"))).sum()
        n_ever_diet_mgmt = (
            df.dr_ever_diet_mgmt & df.is_alive & ((df.dr_status == "proliferative") | (df.dr_status == "mild"))).sum()

        def zero_out_nan(x):
            return x if not np.isnan(x) else 0.0

        def safe_divide(x, y):
            return float(x / y) if y > 0.0 else 0.0

        out.update({
            'diagnosed_since_last_log': df.dr_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.dr_date_treatment.between(date_lastlog, date_now).sum(),
            'prop_ever_diet_mgmt_if_any_dr': zero_out_nan(safe_divide(n_ever_diet_mgmt, n_any_dr)),
        })

        logger.info(key='summary_stats', data=out)
