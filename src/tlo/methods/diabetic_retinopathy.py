from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, cardio_metabolic_disorders
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.population import IndividualProperties
from tlo.util import read_csv_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiabeticRetinopathy(Module):
    """ This is Diabetic Retinopathy (DR) module. It seeks to model DR effects. """

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    PARAMETERS = {
        "rate_onset_to_mild_or_moderate_dr": Parameter(Types.REAL,
                                                       "Probability of people who get diagnosed with non-proliferative "
                                                       "mild/moderate diabetic retinopathy"),
        "rate_mild_or_moderate_to_severe": Parameter(Types.REAL,
                                                     "Probability of people who get diagnosed with severe "
                                                     "diabetic retinopathy"),
        "rate_severe_to_proliferative": Parameter(Types.REAL,
                                                  "Probability of people who get diagnosed with "
                                                  "proliferative diabetic retinopathy"),

        "init_prob_any_dr": Parameter(Types.LIST, "Initial probability of anyone with diabetic retinopathy"),
        "prob_any_dmo": Parameter(Types.LIST, "Probability of anyone with diabetic retinopathy having "
                                              "Diabetic Macular Oedema (DMO)"),
        "p_laser_success": Parameter(Types.REAL, "Diabetic retinopathy treatment/medication effectiveness"),
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
        "prob_diabetes_controlled": Parameter(
            Types.REAL,
            "Probability that a person with mild DR has controlled diabetes"
        ),
        "prob_mild_to_none_if_controlled_diabetes": Parameter(
            Types.REAL,
            "Probability that people with mild DR and controlled diabetes regress to 'none'"
        ),
        "probs_for_dmo_when_dr_status_mild_or_moderate": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has non-proliferative mild/moderate "
                        "Diabetic Retinopathy "),
        "probs_for_dmo_when_dr_status_severe": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has severe Diabetic Retinopathy "),
        "probs_for_dmo_when_dr_status_proliferative": Parameter(
            Types.LIST, "probability of having a DMO state when an individual has "
                        "proliferative Diabetic Retinopathy "),
        "prob_eye_screening": Parameter(
            Types.REAL, "probability of going for an eye exam/screening"
        ),
        "prob_repeat_laser": Parameter(
            Types.REAL,
            "Probability that a patient who remains proliferative at follow-up/review will "
            "require a repeat of HSI_Dr_Laser_Pan_Retinal_Coagulation"
        ),
        "next_screening_if_mild_or_moderate": Parameter(
            Types.INT,
            "Number of months until next eye screening if DR status is mild or moderate"
        ),
        "next_screening_if_severe": Parameter(
            Types.INT,
            "Number of months until next eye screening if DR status is severe"
        ),
    }

    PROPERTIES = {
        "dr_status": Property(
            Types.CATEGORICAL,
            "DR status",
            categories=["none", "mild_or_moderate", "severe", "proliferative"],
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
        "dmo_on_treatment": Property(
            Types.BOOL, "Whether this person is on diabetic macular oedema treatment",
        ),
        "dmo_date_treatment": Property(
            Types.DATE,
            "date of first receiving diabetic macular oedema treatment (pd.NaT if never started treatment)"
        ),
        "dr_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "The DR stage at which treatment was given (used to apply stage-specific treatment effect)",
            categories=["none", "mild_or_moderate", "severe", "proliferative"]
        ),
        "dr_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with many form of diabetic retinopathy"
        ),
        "dr_mild_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with mild/moderate non-proliferative diabetic "
                        "retinopathy"
        ),
        "dr_proliferative_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with proliferative diabetic retinopathy"
        ),
        "dr_dmo_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with any diabetic retinopathy or diabetic macular "
                        "oedema"
        ),
        "dr_date_diagnosis": Property(
            Types.DATE,
            "The date of diagnosis of diabetic retinopathy (pd.NaT if never diagnosed)"
        ),
        "selected_for_eye_screening": Property(
            Types.BOOL, "selected for screening this period"
        ),
        "dmo_antivegf_count": Property(
            Types.INT,
            "Number of anti-VEGF injections received in the current treatment cycle",
        ),
    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)

    def read_parameters(self, resourcefilepath: Optional[Path]=None):
        """ Read all parameters and define symptoms (if any)"""
        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / "ResourceFile_Diabetic_Retinopathy",
                           files="parameter_values")
        )

    def initialise_population(self, population: Population) -> None:
        """ Set property values for the initial population

        :param population: all individuals in the model

        """
        df = population.props
        p = self.parameters

        alive_diabetes_idx = df.loc[df.is_alive & df.nc_diabetes].index

        # any_dr_idx = alive_diabetes_idx[
        #     self.rng.random_sample(size=len(alive_diabetes_idx)) < p['init_prob_any_dr']
        #     ]
        # no_dr_idx = set(alive_diabetes_idx) - set(any_dr_idx)
        #
        # proliferative_dr_idx = any_dr_idx[
        #     self.rng.random_sample(size=len(any_dr_idx)) < p['init_prob_proliferative_dr']
        #     ]
        #
        # mild_dr_idx = set(any_dr_idx) - set(proliferative_dr_idx)

        # write to property:
        df.loc[df.is_alive & ~df.nc_diabetes, 'dr_status'] = 'none'
        df.loc[df.is_alive & ~df.nc_diabetes, 'dmo_status'] = 'none'

        df.loc[list(alive_diabetes_idx), "dr_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dmo_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dr_diagnosed"] = False
        df.loc[list(alive_diabetes_idx), "selected_for_eye_screening"] = False
        df.loc[list(alive_diabetes_idx), "dr_date_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dmo_date_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_stage_at_which_treatment_given"] = "none"
        df.loc[list(alive_diabetes_idx), "dr_date_diagnosis"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dmo_antivegf_count"] = 0

        # -------------------- dr_status -----------
        # Determine who has diabetic retinopathy at all stages:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prob_any_dr']) <= 1.0

        lm_init_dr_status_any_dr = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prob_any_dr']),
            Predictor('li_ex_alc').when(True, p['rp_dr_ex_alc']),
            Predictor('li_tob').when(True, p['rp_dr_tobacco']),
            Predictor('li_high_sugar').when(True, p['rp_dr_high_sugar']),
            Predictor('li_low_ex').when(True, p['rp_dr_low_ex']),
            Predictor('li_urban').when(True, p['rp_dr_urban']),
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
            assert len(categories) == len(p['init_prob_any_dr'])

            # Normalize probabilities
            total_prob = sum(p['init_prob_any_dr'])
            probs = [p / total_prob for p in p['init_prob_any_dr']]

            # Assign DR stages
            df.loc[dr_idx, 'dr_status'] = self.rng.choice(
                categories,
                size=len(dr_idx),
                p=probs
            )

        # dr_stage_probs = p["init_prob_any_dr"]
        # Determine the stage of DR for those who have DM:
        # if any_dr.sum():
        #     categories = [cat for cat in df.dr_status.cat.categories if cat != 'none']
        #
        #     # Make sure we have the right number of probabilities
        #     assert len(categories) == len(p['init_prob_any_dr'])
        #
        #     # Normalize probabilities
        #     sum_probs = sum(p['init_prob_any_dr'])
        #     prob_by_stage_of_dr_if_dr = [p / sum_probs for p in p['init_prob_any_dr']]
        #
        #     # Assign statuses
        #     df.loc[any_dr, "dr_status"] = self.rng.choice(
        #         categories,
        #         size=any_dr.sum(),
        #         p=prob_by_stage_of_dr_if_dr
        #     )

        # sum_probs = sum(p['init_prob_any_dr'])
        # if sum_probs > 0:
        #     prob_by_stage_of_dr_if_dr = [i / sum_probs for i in p['init_prob_any_dr']]
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
        """ Set properties of a child when they are born.
        :param child_id: the new child
        """
        self.sim.population.props.at[child_id, 'dr_status'] = 'none'
        self.sim.population.props.at[child_id, 'dmo_status'] = 'none'
        self.sim.population.props.at[child_id, 'dr_on_treatment'] = False
        self.sim.population.props.at[child_id, 'dr_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dmo_on_treatment'] = False
        self.sim.population.props.at[child_id, 'dmo_date_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_stage_at_which_treatment_given'] = 'none'
        self.sim.population.props.at[child_id, 'dr_diagnosed'] = False
        self.sim.population.props.at[child_id, 'selected_for_eye_screening'] = False
        self.sim.population.props.at[child_id, 'dr_date_diagnosis'] = pd.NaT
        self.sim.population.props.at[child_id, "dmo_antivegf_count"] = 0

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        self.lm = dict()
        p =self.parameters

        self.lm['onset_mild_or_moderate_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=p['rate_onset_to_mild_or_moderate_dr']
        )

        self.lm['mildmoderate_severe_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['rate_mild_or_moderate_to_severe'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, 0.0).otherwise(1.0)
        )

        self.lm['severe_proliferative_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['rate_severe_to_proliferative'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, 0.0).otherwise(1.0)
        )

    def look_up_consumable_item_codes(self):
        """Look up the item codes used in the HSI of this module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()
        self.cons_item_codes['laser_pan_retinal_photocoagulation'] = {
            get_item_codes("Anesthetic Eye drops, 15ml"): 1,
            get_item_codes("Mydriatic/Dilation Drops, 15ml"): 1,
            get_item_codes("Methylcellulose"): 1,
            get_item_codes('Alcohol wipes for lens'): 4
        }
        self.cons_item_codes['anti_vegf_injection'] = {
            get_item_codes("Anesthetic Eye drops, 15ml"): 1,
            get_item_codes('Aflibercept, 2mg'): 3,
            get_item_codes("Antiseptic solution, 15ml"): 1,
            get_item_codes("Syringe, 5ml, disposable, hypoluer with 21g needle_each_CMST"): 1,
            get_item_codes("30G needle"): 1,
            get_item_codes("Speculum"): 1,
            get_item_codes("Sterile gloves and drapes"): 1
        }

        self.cons_item_codes['focal_laser'] = {
            get_item_codes("Anesthetic Eye drops, 15ml"): 1,
            get_item_codes("Mydriatic/Dilation Drops, 15ml"): 1,
            get_item_codes("Methylcellulose"): 1,
            get_item_codes('Alcohol wipes for lens'): 4
        }

        self.cons_item_codes['eye_screening'] = {
            get_item_codes("Mydriatic/Dilation Drops, 15ml"): 1
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
        p = self.parameters

        # First reset dmo_status to 'none' for anyone without DR
        no_dr_mask = (df.dr_status == 'none') | df.dr_status.isna()
        df.loc[no_dr_mask, 'dmo_status'] = 'none'

        # Now only process people with valid DR status
        valid_dr_statuses = ['mild_or_moderate', 'severe', 'proliferative']
        dr_idx = df.loc[df.is_alive & df.dr_status.isin(valid_dr_statuses)].index

        if not dr_idx.empty:
            for person in dr_idx:
                dr_stage = df.at[person, 'dr_status']

                if dr_stage == 'mild_or_moderate':
                    probs = p['probs_for_dmo_when_dr_status_mild_or_moderate']
                elif dr_stage == 'severe':
                    probs = p['probs_for_dmo_when_dr_status_severe']
                elif dr_stage == 'proliferative':
                    probs = p['probs_for_dmo_when_dr_status_proliferative']

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

        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.dr_date_treatment) & \
            (df.dr_status == df.dr_stage_at_which_treatment_given)

        diabetes_and_alive_nodr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'none')]
        diabetes_and_alive_mild_moderate_dr = df.loc[
            df.is_alive & df.nc_diabetes & (df.dr_status == 'mild_or_moderate')]
        diabetes_and_alive_severedr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'severe')]

        will_progress = self.module.lm['onset_mild_or_moderate_dr'].predict(diabetes_and_alive_nodr, self.module.rng)
        # will_progress_idx = will_progress[will_progress].index
        will_progress_idx = df.index[np.where(will_progress)[0]]
        df.loc[will_progress_idx, 'dr_status'] = 'mild_or_moderate'

        mildmoderate_to_severe = self.module.lm['mildmoderate_severe_dr'].predict(
            diabetes_and_alive_mild_moderate_dr,
            self.module.rng,
            had_treatment_during_this_stage=had_treatment_during_this_stage)
        # moderate_to_severe_idx = moderate_to_severe[moderate_to_severe].index
        mildmoderate_to_severe_idx = df.index[np.where(mildmoderate_to_severe)[0]]
        df.loc[mildmoderate_to_severe_idx, 'dr_status'] = 'severe'

        severe_to_proliferative = self.module.lm['severe_proliferative_dr'].predict(
            diabetes_and_alive_severedr,
            self.module.rng,
            had_treatment_during_this_stage=had_treatment_during_this_stage)
        # severe_to_proliferative_idx = mild_to_moderate[severe_to_proliferative].index
        severe_to_proliferative_idx = df.index[np.where(severe_to_proliferative)[0]]
        df.loc[severe_to_proliferative_idx, 'dr_status'] = 'proliferative'

        # Update DMO status
        self.module.update_dmo_status()

        mild_dr_individuals = diabetes_and_alive_mild_moderate_dr
        # Get those who are currently on diabetes weight loss medication from cardiometabolicdisorders
        mild_dr_individuals_eligible = mild_dr_individuals[mild_dr_individuals.nc_diabetes_on_medication]
        # Get those with controlled diabetes among those with mild dr_status
        selected_individuals_with_controlled_and_mild = (
            self.module.rng.random_sample(len(mild_dr_individuals_eligible))
            < self.module.parameters['prob_diabetes_controlled'])
        controlled_and_mild_idx = mild_dr_individuals_eligible.index[selected_individuals_with_controlled_and_mild]

        # Get those who will regress to none dr_status among those with mild dr_status and controlled diabetes
        selected_to_regress_to_none = (self.module.rng.random_sample(len(controlled_and_mild_idx))
                                       < self.module.parameters['prob_mild_to_none_if_controlled_diabetes'])
        regress_to_none_idx = controlled_and_mild_idx[selected_to_regress_to_none]

        df.loc[regress_to_none_idx, "dr_status"] = "none"
        # df.loc[regress_to_none_idx, "dmo_status"] = "none"

        # ------------------------SELECTING INDIVIDUALS FOR FOR DR OR DMO EYE EXAM/SCREENING---------------------
        df.selected_for_eye_screening = False

        eligible_population_for_eye_screening = (
            (df.is_alive & df.nc_diabetes) &  #todo add condition for people not to be selected again witin 1 year
            (df.dr_status == 'none') &
            (df.dmo_status == 'none') &
            (df.age_years >= 20) &
            (pd.isna(df.dr_date_diagnosis))
        )

        df.loc[eligible_population_for_eye_screening, 'selected_for_eye_screening'] = (
            np.random.random_sample(size=len(df[eligible_population_for_eye_screening]))
            < self.module.parameters['prob_eye_screening']
        )

        for idx in df.index[df.selected_for_eye_screening]:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Dr_Dmo_Screening(module=self.module, person_id=idx),
                priority=0,
                topen=self.sim.date,
                tclose=None)

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        pass


class HSI_Dr_Dmo_Screening(HSI_Event, IndividualScopeEventMixin):
    """This is the eye examination/screening done to individuals selected for screening
    for individuals with any complication"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Dmo_Eye_Examination'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Dr_Dmo_Screening: investigating the condition of {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        p = self.parameters

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        assert pd.isnull(df.at[person_id, 'dr_date_treatment'])

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['eye_screening']
        )

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='dilated_eye_exam_dr_dmo',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'dr_date_diagnosis'] = self.sim.date
            df.at[person_id, 'dr_date_treatment'] = self.sim.date
            df.at[person_id, 'dr_stage_at_which_treatment_given'] = df.at[person_id, 'dr_status']
            # If consumables are available, add equipment used
            self.add_equipment({'Visual acuity chart', 'Direct ophthalmoscope',
                                'Fundus camera'})

            if person.dr_status == 'mild_or_moderate' or person.dmo_status == 'non_clinically_significant':
                # schedule HSI_CardioMetabolicDisorders_StartWeightLossAndMedication
                # and repeat HSI_DR_Dmo_Screening in 1 year
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    hsi_event=cardio_metabolic_disorders.HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                        person_id=person_id,
                        module=self.sim.modules["CardioMetabolicDisorders"],
                        condition='diabetes',
                    ),
                    priority=0,
                    topen=self.sim.date
                )

                # Repeat eye screening in 1 year if dr_status is mild_or_moderate or dmo_status is
                # non_clinically_significant
                next_screening_if_mild_or_moderate =\
                    self.sim.date + pd.DateOffset(months=p['next_screening_if_mild_or_moderate'])
                self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                                    topen=next_screening_if_mild_or_moderate,
                                                                    tclose=None,
                                                                    priority=1)

            elif person.dr_status == 'severe':
                next_screening_if_severe = self.sim.date + pd.DateOffset(months=p['next_screening_if_severe'])
                self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                                    topen=next_screening_if_severe,
                                                                    tclose=None,
                                                                    priority=1)

            elif person.dr_status == 'proliferative':
                # If this is their FIRST diagnosis (no prior treatment yet), send them
                # for HSI_Dr_Laser_Pan_Retinal_Coagulation
                if pd.isna(person.dr_date_treatment):
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Dr_Laser_Pan_Retinal_Coagulation(self.module, person_id, session=1),
                        topen=self.sim.date,
                        priority=0
                    )
                else:
                    # This is a scheduled follow-up or review (2 months / 3,6,9,12 months after
                    # HSI_Dr_Laser_Pan_Retinal_Coagulation). Since HSI_Dr_Laser_Pan_Retinal_Coagulation is
                    # scheduled for 2 weeks, this will only execute after at least 2 months and hence after session 2
                    if self.module.rng.random_sample() < self.module.parameters['prob_repeat_laser']:
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            hsi_event=HSI_Dr_Laser_Pan_Retinal_Coagulation(self.module, person_id, session=1),
                            topen=self.sim.date,
                            priority=0
                        )

            if person.dmo_status == "clinically_significant" and person.dr_status != "none":
                # Randomise between AntiVEGF or Focal Laser
                treatment_for_cs_dmo = HSI_Dr_Dmo_AntiVEGF \
                    if self.module.rng.random_sample() < 0.5 else HSI_Dr_Dmo_Focal_Laser
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=treatment_for_cs_dmo(module=self.module, person_id=person_id),
                    topen=self.sim.date,
                    priority=0
                )


class HSI_Dr_Dmo_Focal_Laser(HSI_Event, IndividualScopeEventMixin):
    """This is the Laser treatment for individuals with CSMO."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Dmo_Focal_Laser_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Dr_Focal_Laser: initiating laser treatment for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        # hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        assert pd.isnull(df.at[person_id, 'dr_date_treatment'])

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['focal_laser']
        )

        if is_cons_available:
            # record date of diagnosis
            df.at[person_id, 'dmo_date_treatment'] = self.sim.date
            df.at[person_id, 'dmo_on_treatment'] = True
            # df.at[person_id, 'dr_stage_at_which_treatment_given'] = df.at[person_id, 'dr_status']
            # If consumables are available, add equipment used and run dx_test
            self.add_equipment({'Visual acuity chart', 'Opthalmic laser system',
                                'Silt lamp laser delivery system', 'Laser macular contact lens'})

            # Schedule follow-up checkup after 3 months #todo add chance that person needs after 3 months
            follow_up_date = self.sim.date + pd.DateOffset(months=3)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Dr_Dmo_Screening(self.module, person_id),
                topen=follow_up_date,
                priority=1
            )


class HSI_Dr_Dmo_AntiVEGF(HSI_Event, IndividualScopeEventMixin):
    """This is the Anti-VEGF treatment for individuals with CSMO."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Dmo_AntiVEGF_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Dr_AntiVEGF for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # If person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['anti_vegf_injection']
        )

        if is_cons_available:
            self.add_equipment({'Visual acuity chart', 'Silt lamp', 'Optical coherence tomography device',
                                'Fundus camera', 'Tonometre'})

            df.at[person_id, 'dmo_date_treatment'] = self.sim.date
            df.at[person_id, 'dmo_on_treatment'] = True
            # Increment the injection count
            df.at[person_id, 'dmo_antivegf_count'] = df.at[person_id, 'dmo_antivegf_count'] + 1

            # Check if more injections are needed. Must not exceed 8
            if df.at[person_id, 'dmo_antivegf_count'] < 8:
                next_session = self.sim.date + pd.DateOffset(months=1)
                self.sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                                    topen=next_session,
                                                                    priority=0
                                                                    )
            else:
                # Reset individual injections after getting 8 of them
                df.at[person_id, 'dmo_antivegf_count'] = 0
                df.at[person_id, 'dmo_on_treatment'] = False


class HSI_Dr_Laser_Pan_Retinal_Coagulation(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event in which a person receives a dialysis session 2 times a week
    adding up to 8 times a month."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Dr_Laser_Pan_Retinal_Coagulation_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        self.add_equipment({'Visual acuity chart', 'Opthalmic laser system',
                            'Silt lamp laser delivery system',
                            'Indirect laser delivery system', 'Laser wide-field contact lens'})

        next_session_date = self.sim.date + pd.DateOffset(weeks=1)
        self.sim.modules['HealthSystem'].schedule_hsi_event(self,
            topen=next_session_date,
            tclose=next_session_date + pd.DateOffset(days=3),
            priority=1
        )


class HSI_Dr_Laser_Pan_Retinal_Coagulation(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI event given to individuals with proliferative diabetic retinopathy after undergoing an eye
    exam/screening
    """

    def __init__(self, module, person_id, session: int = 1):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Laser_Pan_Retinal_Coagulation_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []
        self.session = int(session)

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Laser_Pan_Retinal_Coagulation for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # if person already on treatment or not yet diagnosed, do nothing
        if person["dr_on_treatment"] or not person["dr_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['laser_pan_retinal_photocoagulation']
        )

        if is_cons_available:
            self.add_equipment({'Visual acuity chart', 'Opthalmic laser system',
                                'Silt lamp laser delivery system',
                                'Indirect laser delivery system', 'Laser wide-field contact lens'})
            if self.session == 1:
                # schedule the second session in 1 week
                hs.schedule_hsi_event(
                    hsi_event=HSI_Dr_Laser_Pan_Retinal_Coagulation(self.module, person_id, session=2),
                    topen=self.sim.date + pd.DateOffset(weeks=1),
                    priority=0
                )
            elif self.session == 2:
                # should complete treatment??
                df.at[person_id, 'dr_on_treatment'] = False

                # schedule follow-up at 2 months
                follow_up = self.sim.date + pd.DateOffset(months=2)
                hs.schedule_hsi_event(
                    hsi_event=HSI_Dr_Dmo_Screening(self.module, person_id),
                    topen=follow_up,
                    priority=1
                )

                # schedule reviews at 3, 6, 9, 12 months
                for m in [3, 6, 9, 12]:
                    review_date = self.sim.date + pd.DateOffset(months=m)
                    hs.schedule_hsi_event(
                        hsi_event=HSI_Dr_Dmo_Screening(self.module, person_id),
                        topen=review_date,
                        priority=1
                    )


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

        out.update({
            'diagnosed_since_last_log': df.dr_date_diagnosis.between(date_lastlog, date_now).sum(),
            'treated_since_last_log': df.dr_date_treatment.between(date_lastlog, date_now).sum(),
        })

        logger.info(key='summary_stats', data=out)
