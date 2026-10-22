import copy
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Population, Property, Simulation, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, cardio_metabolic_disorders
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
from tlo.methods.symptommanager import Symptom
from tlo.population import IndividualProperties
from tlo.util import read_csv_files, transition_states

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DiabeticRetinopathy(Module):
    """ This is Diabetic Retinopathy (DR) module. It seeks to model DR effects. """

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle', 'HealthSystem', 'CardioMetabolicDisorders', 'SymptomManager'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    CAUSES_OF_DEATH = {
        'diabetes': Cause(
            gbd_causes='Diabetes mellitus', label='Diabetes'), }
    CAUSES_OF_DISABILITY = {
        'diabetes': Cause(
            gbd_causes='Diabetes mellitus', label='Diabetes'), }

    PARAMETERS = {
        "prob_onset_to_mild_or_moderate_dr": Parameter(Types.REAL,
                                                       "Probability of people who get diagnosed with non-proliferative "
                                                       "mild/moderate diabetic retinopathy"),
        "prob_mild_or_moderate_to_severe": Parameter(Types.REAL,
                                                     "Probability of people who get diagnosed with severe "
                                                     "diabetic retinopathy"),
        "prob_severe_to_proliferative": Parameter(Types.REAL,
                                                  "Probability of people who get diagnosed with "
                                                  "proliferative diabetic retinopathy"),

        "init_prob_any_dr": Parameter(Types.LIST, "Initial probability of anyone with diabetic retinopathy"),
        "prob_any_dmo": Parameter(Types.LIST, "Probability of anyone with diabetic retinopathy having "
                                              "Diabetic Macular Oedema (DMO)"),
        "p_laser_success": Parameter(Types.REAL, "Diabetic retinopathy treatment/medication effectiveness"),
        "rp_nc_hypertension": Parameter(
            Types.REAL, "relative risk at baseline of diabetic retinopathy if person has hypertension"),
        # "prob_diabetes_controlled": Parameter(
        #     Types.REAL,
        #     "Probability that a person with mild DR has controlled diabetes"
        # ),
        # "prob_mild_to_none_if_controlled_diabetes": Parameter(
        #     Types.REAL,
        #     "Probability that people with mild DR and controlled diabetes regress to 'none'"
        # ),
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
        "prob_repeat_dmo_treatment": Parameter(
            Types.REAL,
            "Probability that clinically significant DMO requires repeat treatment at review"
        ),
        "prob_antivegf_for_clinically_significant_dmo": Parameter(
            Types.REAL,
            "Probability that Anti-VEGF is selected rather than focal laser for clinically significant DMO"
        ),
        "next_screening_if_mild_or_moderate": Parameter(
            Types.INT,
            "Number of months until next eye screening if DR status is mild or moderate"
        ),
        "next_screening_if_severe": Parameter(
            Types.INT,
            "Number of months until next eye screening if DR status is severe"
        ),
        "rr_diabetes_duration_greater_than_15": Parameter(
            Types.REAL,
            "Relative risk multiplier for diabetic retinopathy when diabetes duration > 15 years"
        ),
        "prob_antivegf_success_clinically_significant": Parameter(
            Types.REAL,
            "Probability that anti-VEGF treatment improves vision (transition from clinically_significant to "
            "non_clinically_significant)"
        ),
        "prob_focal_laser_success_clinically_significant": Parameter(
            Types.REAL,
            "Probability that focal laser treatment improves vision (transition from clinically_significant to "
            "non_clinically_significant)"
        ),
        "prob_laser_prp_success_proliferative": Parameter(
            Types.REAL,
            "Probability that laser pan-retinal photocoagulation successfully preserves the "
            "person's current vision by preventing further vision deterioration from proliferative DR"
        ),
        "prob_antivegf_moderate_to_normal": Parameter(
            Types.REAL,
            "Probability that successful anti-VEGF treatment improves vision from "
            "moderate_vision_impairment to normal",
        ),
        "prob_antivegf_severe_to_moderate": Parameter(
            Types.REAL,
            "Probability that successful anti-VEGF treatment improves vision from "
            "severe_vision_impairment to moderate_vision_impairment",
        ),
        "prob_antivegf_severe_to_normal": Parameter(
            Types.REAL,
            "Probability that successful anti-VEGF treatment improves vision from "
            "severe_vision_impairment directly to normal",
        ),
        "prob_focal_laser_moderate_to_normal": Parameter(
            Types.REAL,
            "Probability that successful focal laser treatment improves vision from "
            "moderate_vision_impairment to normal",
        ),
        "prob_focal_laser_severe_to_moderate": Parameter(
            Types.REAL,
            "Probability that successful focal laser treatment improves vision from "
            "severe_vision_impairment to moderate_vision_impairment",
        ),
        "prob_focal_laser_severe_to_normal": Parameter(
            Types.REAL,
            "Probability that successful focal laser treatment improves vision from "
            "severe_vision_impairment directly to normal",
        ),
        "rr_progression_after_treatment": Parameter(
            Types.REAL,
            "Relative risk of progression to more severe stages after receiving treatment (applied multiplicatively)"
        ),
        "vision_transition_matrix_no_dr": Parameter(
            Types.LIST,
            "Probability matrix for vision status transitions for people without sight-threatening DR"
        ),
        "vision_transition_matrix_sight_threatening": Parameter(
            Types.LIST,
            "Probability matrix for vision status transitions for people with sight-threatening DR"
        ),
        "sensitivity_of_dilated_eye_exam_dr_dmo": Parameter(
            Types.REAL, "sensitivity of dilated eye exam/test for DR and DMO"
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
        "dmo_on_treatment": Property(
            Types.BOOL, "Whether this person is on diabetic macular oedema treatment",
        ),
        "dr_date_treatment": Property(
            Types.DATE,
            "date of first receiving diabetic retinopathy treatment (pd.NaT if never started treatment)"
        ),
        "dmo_date_treatment": Property(
            Types.DATE,
            "date of first receiving diabetic macular oedema treatment (pd.NaT if never started treatment)"
        ),
        "dr_date_last_screening": Property(
            Types.DATE,
            "Date of last eye screening (pd.NaT if never screened)"
        ),
        "dr_date_prp_treatment": Property(
            Types.DATE,
            "date of first receiving Laser Pan Retinal Photo Coagulation treatment "
            "(pd.NaT if never started treatment)"
        ),
        "dr_date_diagnosis": Property(
            Types.DATE,
            "The date of diagnosis of diabetic retinopathy (pd.NaT if never diagnosed)"
        ),
        "dr_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "The DR stage at which treatment was given (used to apply stage-specific treatment effect)",
            categories=["none", "mild_or_moderate", "severe", "proliferative"]
        ),
        "dr_diagnosed": Property(
            Types.BOOL, "Whether this person has been diagnosed with many form of diabetic retinopathy"
        ),
        "selected_for_eye_screening": Property(
            Types.BOOL, "selected for screening this period"
        ),
        "dmo_antivegf_count": Property(
            Types.INT,
            "Number of anti-VEGF injections received in the current treatment cycle",
        ),
        "total_laser_pan_retinal_photocoagulation_sessions": Property(
            Types.INT, "Number of Laser Pan Retinal Photo Coagulation sessions received",
        ),
        "on_laser_pan_retinal_photocoagulation_treatment": Property(
            Types.BOOL, "Whether this person is on Laser Pan Retinal Coagulation treatment",
        ),
        "dr_prp_vision_preserved": Property(
            Types.BOOL,
            "Whether successful laser pan-retinal photocoagulation treatment is "
            "preserving the person's current vision status",
        ),
        "vision_status": Property(
            Types.CATEGORICAL,
            "Visual acuity status of an individual",
            categories=["normal", "moderate_vision_impairment", "severe_vision_impairment", "blindness"],
            ordered=True
        ),
        "vision_loss_due_to_dr": Property(
            Types.BOOL,
            "Whether current vision impairment is due to diabetic retinopathy"
        ),

        "dmo_treatment_success": Property(
            Types.BOOL,
            "Whether the most recent DMO treatment assessment was successful",
        ),
        "dmo_vision_improved": Property(
            Types.BOOL,
            "Whether vision improved during the most recent DMO treatment assessment",
        ),
        "dr_dmo_diagnosis_pathway": Property(
            Types.CATEGORICAL,
            "Pathway by which DR/DMO was first diagnosed",
            categories=["none", "screening", "symptomatic"],
        ),

    }

    def __init__(self):
        super().__init__()
        self.cons_item_codes = None  # (Will store consumable item codes)
        self.daly_wts = dict()

        import copy

        # DR incidence tracker
        self.dr_incident_case_tracker = {
            'mild_or_moderate': [],
            'severe': [],
            'proliferative': []
        }
        self.dr_incident_case_tracker_blank = copy.deepcopy(self.dr_incident_case_tracker)

        # DMO incidence tracker
        self.dmo_incident_case_tracker = {
            'clinically_significant': [],
            'non_clinically_significant': []
        }
        self.dmo_incident_case_tracker_blank = copy.deepcopy(self.dmo_incident_case_tracker)

        # Vision impairment incidence tracker
        self.vision_incident_case_tracker = {
            'moderate_vision_impairment': [],
            'severe_vision_impairment': [],
            'blindness': []
        }

        self.vision_incident_case_tracker_blank = copy.deepcopy(self.vision_incident_case_tracker)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """ Read all parameters and define symptoms (if any)"""
        self.load_parameters_from_dataframe(
            read_csv_files(resourcefilepath / "ResourceFile_Diabetic_Retinopathy",
                           files="parameter_values")
        )
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='reduced_vision',
                odds_ratio_health_seeking_in_adults=3.0,  # to adjust based on literature/assumptions
            )
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

        df.loc[df.is_alive & df.nc_diabetes, 'dr_status'] = 'none'
        df.loc[df.is_alive & df.nc_diabetes, 'dmo_status'] = 'none'

        df.loc[list(alive_diabetes_idx), "dr_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dmo_on_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dr_diagnosed"] = False
        df.loc[list(alive_diabetes_idx), "selected_for_eye_screening"] = False
        df.loc[list(alive_diabetes_idx), "dr_date_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dmo_date_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_date_last_screening"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_stage_at_which_treatment_given"] = "none"
        df.loc[list(alive_diabetes_idx), "dr_date_diagnosis"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dmo_antivegf_count"] = 0
        df.loc[list(alive_diabetes_idx), "total_laser_pan_retinal_photocoagulation_sessions"] = 0
        df.loc[list(alive_diabetes_idx), "on_laser_pan_retinal_photocoagulation_treatment"] = False
        df.loc[list(alive_diabetes_idx), "dr_date_prp_treatment"] = pd.NaT
        df.loc[list(alive_diabetes_idx), "dr_prp_vision_preserved"] = False
        df.loc[list(alive_diabetes_idx), "vision_status"] = "normal"
        df.loc[list(alive_diabetes_idx), "vision_loss_due_to_dr"] = False
        df.loc[list(alive_diabetes_idx), "dmo_treatment_success"] = False
        df.loc[list(alive_diabetes_idx), "dmo_vision_improved"] = False
        df.loc[list(alive_diabetes_idx), "dr_dmo_diagnosis_pathway"] = "none"

        # -------------------- dr_status -----------
        # Determine who has diabetic retinopathy at all stages:
        # check parameters are sensible: probability of having any cancer stage cannot exceed 1.0
        assert sum(p['init_prob_any_dr']) <= 1.0

        lm_init_dr_status_any_dr = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            sum(p['init_prob_any_dr']),
            Predictor('nc_hypertension').when(True, p['rp_nc_hypertension']),
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

        # Assign DMO for those with DR at baseline
        self.update_dmo_status()

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
        self.reset_incidence_trackers()

        # ----- DX TESTS -----
        # Create the diagnostic test representing dilated eye exam for DR and DMO
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            dilated_eye_exam_dr_dmo=DxTest(
                property='dr_status',
                sensitivity=self.parameters['sensitivity_of_dilated_eye_exam_dr_dmo'],
                target_categories=["none", "mild_or_moderate", "severe", "proliferative"]
            )
        )

        # ----- DISABILITY-WEIGHTS -----
        if "HealthBurden" in self.sim.modules:
            # For those with moderate vision impairment due to Diabetes Mellitus
            self.daly_wts["moderate_vision_impairment_due_to_dm"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=967
            )
            # For those with severe vision impairment due to Diabetes Mellitus
            self.daly_wts["severe_vision_impairment_due_to_dm"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=969
            )
            # For those who are blind due to Diabetes Mellitus
            self.daly_wts["blindness_due_to_dm"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=974
            )

        # Convert vision transition matrices
        vision_states = sim.population.props.vision_status.cat.categories

        for key in ['vision_transition_matrix_no_dr', 'vision_transition_matrix_sight_threatening']:
            mat = pd.DataFrame(
                self.parameters[key],
                index=vision_states,
                columns=vision_states
            )

            # CSV is row-stochastic → model expects column-stochastic
            mat = mat.T

            # Validate (columns must sum to 1)
            col_sums = mat.sum(axis=0)
            if not np.allclose(col_sums, 1.0):
                raise ValueError(
                    f"{key} has columns that do not sum to 1:\n{col_sums}"
                )

            if (mat < 0).any().any():
                raise ValueError(f"{key} contains negative probabilities")

            self.parameters[key] = mat

    def report_daly_values(self) -> pd.Series:
        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with moderate vision impairment due to DM
        disability_series_for_alive_persons.loc[
            (df.vision_status == "moderate_vision_impairment") & df.vision_loss_due_to_dr
            ] = self.daly_wts['moderate_vision_impairment_due_to_dm']

        # Assign daly_wt to those with severe vision impairment due to DM
        disability_series_for_alive_persons.loc[
            (df.vision_status == "severe_vision_impairment") & df.vision_loss_due_to_dr
            ] = self.daly_wts['severe_vision_impairment_due_to_dm']

        # Assign daly_wt to those who are blind due to DM
        disability_series_for_alive_persons.loc[
            (df.vision_status == "blindness") & df.vision_loss_due_to_dr
            ] = self.daly_wts['blindness_due_to_dm']

        return disability_series_for_alive_persons

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
        self.sim.population.props.at[child_id, "total_laser_pan_retinal_photocoagulation_sessions"] = 0
        self.sim.population.props.at[child_id, 'on_laser_pan_retinal_photocoagulation_treatment'] = False
        self.sim.population.props.at[child_id, 'dr_date_prp_treatment'] = pd.NaT
        self.sim.population.props.at[child_id, 'dr_prp_vision_preserved'] = False
        self.sim.population.props.at[child_id, "dmo_treatment_success"] = False
        self.sim.population.props.at[child_id, "dmo_vision_improved"] = False
        self.sim.population.props.at[child_id, 'dr_dmo_diagnosis_pathway'] = 'none'

    def on_simulation_end(self) -> None:
        pass

    def make_the_linear_models(self) -> None:
        """Make and save LinearModels that will be used when the module is running"""
        self.lm = dict()
        p = self.parameters

        self.lm['onset_mild_or_moderate_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=p['prob_onset_to_mild_or_moderate_dr']
        )

        self.lm['mildmoderate_severe_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_mild_or_moderate_to_severe'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, p['rr_progression_after_treatment']),
            Predictor('diabetes_duration_greater_than_15_years', external=True)
            .when(True, p['rr_diabetes_duration_greater_than_15'])
        )

        self.lm['severe_proliferative_dr'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_severe_to_proliferative'],
            Predictor('had_treatment_during_this_stage', external=True)
            .when(True, p['rr_progression_after_treatment']),
            Predictor('diabetes_duration_greater_than_15_years', external=True)
            .when(True, p['rr_diabetes_duration_greater_than_15'])
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

    def do_treatment_success_dmo(self, person_id: int, treatment_type: str) -> None:
        """Apply treatment success for DMO (clinically_significant) and treatment-specific vision outcomes."""
        df = self.sim.population.props
        p = self.parameters

        if df.at[person_id, 'dmo_status'] != 'clinically_significant':
            return

        if treatment_type == 'antivegf':
            success_prob = p['prob_antivegf_success_clinically_significant']
            prob_moderate_to_normal = p['prob_antivegf_moderate_to_normal']
            prob_severe_to_moderate = p['prob_antivegf_severe_to_moderate']
            prob_severe_to_normal = p['prob_antivegf_severe_to_normal']

        elif treatment_type == 'focal_laser':
            success_prob = p['prob_focal_laser_success_clinically_significant']
            prob_moderate_to_normal = p['prob_focal_laser_moderate_to_normal']
            prob_severe_to_moderate = p['prob_focal_laser_severe_to_moderate']
            prob_severe_to_normal = p['prob_focal_laser_severe_to_normal']

        else:
            raise ValueError(f"Unknown DMO treatment type: {treatment_type}")

        if self.rng.random_sample() >= success_prob:
            return

        vision_before = df.at[person_id, 'vision_status']

        self.apply_dmo_vision_outcome(
            person_id=person_id,
            prob_moderate_to_normal=prob_moderate_to_normal,
            prob_severe_to_moderate=prob_severe_to_moderate,
            prob_severe_to_normal=prob_severe_to_normal,
        )

        vision_after = df.at[person_id, 'vision_status']

        # If using these outcome properties, update them accordingly.
        df.at[person_id, 'dmo_treatment_success'] = True
        df.at[person_id, 'dmo_vision_improved'] = (
            vision_after != vision_before
        )

    def apply_dmo_vision_outcome(
        self,
        person_id: int,
        prob_moderate_to_normal: float,
        prob_severe_to_moderate: float,
        prob_severe_to_normal: float,
    ) -> None:
        """Apply vision regression conditional on successful DMO treatment.
        Any probability not assigned to an improvement represents successful
        treatment with no change in vision.
        """
        df = self.sim.population.props

        if not df.at[person_id, 'vision_loss_due_to_dr']:
            return

        current_vision = df.at[person_id, 'vision_status']
        random_draw = self.rng.random_sample()

        if current_vision == 'moderate_vision_impairment':
            if random_draw < prob_moderate_to_normal:
                df.at[person_id, 'vision_status'] = 'normal'

        elif current_vision == 'severe_vision_impairment':
            if random_draw < prob_severe_to_normal:
                # Direct improvement from severe impairment to normal.
                df.at[person_id, 'vision_status'] = 'normal'

            elif random_draw < prob_severe_to_normal + prob_severe_to_moderate:
                # Improvement from severe to moderate impairment.
                df.at[person_id, 'vision_status'] = 'moderate_vision_impairment'

        # Once vision is normal, vision loss is no longer attributed to DR.
        if df.at[person_id, 'vision_status'] == 'normal':
            df.at[person_id, 'vision_loss_due_to_dr'] = False
            # Clear the symptom
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=[person_id],
                symptom_string='reduced_vision',
                add_or_remove='-',
                disease_module=self,
            )


    def improve_vision(self, person_id: int) -> None:
        """Improve vision status by one category after receiving DMO treatment"""
        df = self.sim.population.props

        if not df.at[person_id, 'vision_loss_due_to_dr']:
            return

        vision_categories = df.vision_status.cat.categories
        current_vision = df.at[person_id, 'vision_status']
        current_idx = list(vision_categories).index(current_vision)

        if current_idx > 0:  # Not already at best
            df.at[person_id, 'vision_status'] = vision_categories[current_idx - 1]

            # If vision becomes normal, vision loss is no longer due to DR
            if df.at[person_id, 'vision_status'] == 'normal':
                df.at[person_id, 'vision_loss_due_to_dr'] = False

    def do_treatment_success_proliferative(self, person_id: int) -> None:
        """Apply treatment success for proliferative DR
        Successful PRP does not improve existing vision impairment.
        Instead, it preserves the person's current vision status and
        prevents further vision deterioration due to proliferative DR.
        """
        df = self.sim.population.props
        p = self.parameters

        if df.at[person_id, 'dr_status'] != 'proliferative':
            return

        success_prob = p['prob_laser_prp_success_proliferative']

        if self.rng.random_sample() < success_prob:
            # Successful PRP preserves vision status the individual has
            # at the time of successful treatment.
            df.at[person_id, 'dr_prp_vision_preserved'] = True

    def update_dmo_status(self):
        """Update DMO status for people with diabetic retinopathy.
        Ensures dmo_status is none when dr_status is none/nan."""
        df = self.sim.population.props
        p = self.parameters
        old_dmo = df['dmo_status'].copy()

        # First reset dmo_status to 'none' for anyone without DR
        no_dr_mask = (df.dr_status == 'none') | df.dr_status.isna()
        df.loc[no_dr_mask, 'dmo_status'] = 'none'

        # Now only process people with valid DR status
        valid_dr_statuses = ['mild_or_moderate', 'severe', 'proliferative']
        dr_idx = df.loc[df.is_alive & df.dr_status.isin(valid_dr_statuses) & (df.dmo_status == 'none')].index
        # dr_idx = df.loc[df.is_alive & df.dr_status.isin(valid_dr_statuses)].index

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

        new_dmo = df['dmo_status']
        # changed = old_dmo != new_dmo
        changed = (old_dmo == 'none') & (new_dmo != 'none')

        # Counts new cases (excludes transitions to 'none')
        for status, count in new_dmo[changed].value_counts().items():
            if status != 'none':
                self.dmo_incident_case_tracker[status].extend(
                    [self.sim.date] * count
                )

        invalid_cases = df[
            ((df.dr_status == 'none') | df.dr_status.isna()) &
            (df.dmo_status.isin(['clinically_significant', 'non_clinically_significant']))
            ]
        assert len(invalid_cases) == 0, (
            f"Found {len(invalid_cases)} cases where people with no DR "
            f"have DMO status: {invalid_cases[['dr_status', 'dmo_status']].to_dict()}"
        )

    def update_vision_status(self):
        """Update vision status for people living with diabetes"""
        df = self.sim.population.props
        p = self.parameters
        rng = self.rng
        old_vision = df['vision_status'].copy()

        # alive_diabetes = df.is_alive & df.nc_diabetes
        alive_diabetes = (
            df.is_alive &
            df.nc_diabetes &
            (
                (df.dr_status != "none") |
                (df.dmo_status != "none")
            )
        )

        vision_preserved = (
            df.dr_status.eq('proliferative') &
            df.dr_prp_vision_preserved
        )

        if not alive_diabetes.any():
            return

        sight_threatening = (
                df.dr_status.isin(['severe', 'proliferative']) |
                (df.dmo_status == 'clinically_significant')
        )

        # Get all people on treatment
        on_treatment = df.dr_on_treatment | df.dmo_on_treatment

        # Apply natural-history transitions only to people who are not
        # currently receiving any treatment and whose vision is not protected
        # by successful PRP.

        untreated = alive_diabetes & ~vision_preserved & ~on_treatment

        for idx, base_matrix in [
            (
                untreated & ~sight_threatening,
                p['vision_transition_matrix_no_dr']
            ),
            (
                untreated & sight_threatening,
                p['vision_transition_matrix_sight_threatening']
            ),
        ]:
            if not idx.any():
                continue

            current = df.loc[idx, 'vision_status']

            if not np.allclose(base_matrix.sum(axis=0), 1.0):
                raise ValueError(
                    "Vision transition matrix columns do not sum to 1"
                )

            proposed = transition_states(current, base_matrix, rng)
            df.loc[idx, 'vision_status'] = proposed

        # Update cause only when vision impaired and diabetes-related eye disease
        impaired = df.vision_status.isin(['moderate_vision_impairment', 'severe_vision_impairment', 'blindness'])

        df.loc[impaired & df.nc_diabetes, 'vision_loss_due_to_dr'] = True

        new_vision = df['vision_status']
        changed = old_vision != new_vision

        # Exclude 'normal'
        for status, count in new_vision[changed].value_counts().items():
            if status != 'normal':
                self.vision_incident_case_tracker[status].extend(
                    [self.sim.date] * count
                )

        # Add symptom when vision deteriorates
        newly_impaired = (
            df.is_alive & df.nc_diabetes & ~df.dr_diagnosed
            & df.vision_loss_due_to_dr
            & df.vision_status.isin(['moderate_vision_impairment',
                                     'severe_vision_impairment', 'blindness'])
        )
        if newly_impaired.any():
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=df.index[newly_impaired].tolist(),
                symptom_string='reduced_vision',
                add_or_remove='+',
                disease_module=self,
            )


    def reset_incidence_trackers(self):
        self.dr_incident_case_tracker = copy.deepcopy(self.dr_incident_case_tracker_blank)
        self.dmo_incident_case_tracker = copy.deepcopy(self.dmo_incident_case_tracker_blank)
        self.vision_incident_case_tracker = copy.deepcopy(self.vision_incident_case_tracker_blank)


class DrPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the development process of Diabetes Retinopathy (DR) and logs current states. DR diagnosis
    begins at least after 3 years of being infected with Diabetes Mellitus."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population: Population) -> None:
        df = population.props

        # Getting all those with diabetes from CMD module
        alive_with_diabetes = df.is_alive & df.nc_diabetes

        # Access the diabetes_onset_dates from CMD module
        cmd_module = self.sim.modules['CardioMetabolicDisorders']
        diabetes_onset_series = pd.Series(cmd_module.diabetes_onset_dates)

        # Convert to pandas datetime format
        diabetes_onset_series = pd.to_datetime(diabetes_onset_series)

        # Compute diabetes duration (years) from biological onset
        diabetes_duration_years = pd.Series(0.0, index=df.index)

        # Calculate only for those who have onset dates recorded
        has_onset_date = alive_with_diabetes & df.index.isin(diabetes_onset_series.index)
        diabetes_duration_years.loc[has_onset_date] = (
                (self.sim.date - diabetes_onset_series.loc[has_onset_date]).dt.days / 365.25
        )

        # Boolean for >15 years
        diabetes_duration_greater_than_15_years = diabetes_duration_years >= 15

        had_treatment_during_this_stage = \
            df.is_alive & ~pd.isnull(df.dr_date_treatment) & \
            (df.dr_status == df.dr_stage_at_which_treatment_given)

        diabetes_and_alive_nodr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'none')]
        diabetes_and_alive_mild_moderate_dr = df.loc[
            df.is_alive & df.nc_diabetes & (df.dr_status == 'mild_or_moderate')]
        diabetes_and_alive_severedr = df.loc[df.is_alive & df.nc_diabetes & (df.dr_status == 'severe')]

        will_progress = self.module.lm['onset_mild_or_moderate_dr'].predict(
            diabetes_and_alive_nodr,
            self.module.rng,
            squeeze_single_row_output=False)
        will_progress_idx = will_progress[will_progress].index
        df.loc[will_progress_idx, 'dr_status'] = 'mild_or_moderate'
        self.module.dr_incident_case_tracker['mild_or_moderate'].extend(
            [self.sim.date] * len(will_progress_idx)
        )

        mildmoderate_to_severe = self.module.lm['mildmoderate_severe_dr'].predict(
            diabetes_and_alive_mild_moderate_dr,
            self.module.rng,
            squeeze_single_row_output=False,
            had_treatment_during_this_stage=had_treatment_during_this_stage,
            diabetes_duration_greater_than_15_years=diabetes_duration_greater_than_15_years)
        mildmoderate_to_severe_idx = mildmoderate_to_severe[mildmoderate_to_severe].index
        df.loc[mildmoderate_to_severe_idx, 'dr_status'] = 'severe'
        self.module.dr_incident_case_tracker['severe'].extend(
            [self.sim.date] * len(mildmoderate_to_severe_idx)
        )

        severe_to_proliferative = self.module.lm['severe_proliferative_dr'].predict(
            diabetes_and_alive_severedr,
            self.module.rng,
            squeeze_single_row_output=False,
            had_treatment_during_this_stage=had_treatment_during_this_stage,
            diabetes_duration_greater_than_15_years=diabetes_duration_greater_than_15_years)
        severe_to_proliferative_idx = severe_to_proliferative[severe_to_proliferative].index
        df.loc[severe_to_proliferative_idx, 'dr_status'] = 'proliferative'
        self.module.dr_incident_case_tracker['proliferative'].extend(
            [self.sim.date] * len(severe_to_proliferative_idx)
        )

        # Update DMO status
        self.module.update_dmo_status()

        # mild_dr_individuals = diabetes_and_alive_mild_moderate_dr
        # # Get those who are currently on diabetes weight loss medication from cardiometabolicdisorders
        # mild_dr_individuals_eligible = mild_dr_individuals[mild_dr_individuals.nc_diabetes_on_medication]
        # # Get those with controlled diabetes among those with mild dr_status
        # selected_individuals_with_controlled_and_mild = (
        #     self.module.rng.random_sample(len(mild_dr_individuals_eligible))
        #     < self.module.parameters['prob_diabetes_controlled'])
        # controlled_and_mild_idx = mild_dr_individuals_eligible.index[selected_individuals_with_controlled_and_mild]
        #
        # # Get those who will regress to none dr_status among those with mild dr_status and controlled diabetes
        # selected_to_regress_to_none = (self.module.rng.random_sample(len(controlled_and_mild_idx))
        #                                < self.module.parameters['prob_mild_to_none_if_controlled_diabetes'])
        # regress_to_none_idx = controlled_and_mild_idx[selected_to_regress_to_none]
        #
        # df.loc[regress_to_none_idx, "dr_status"] = "none"
        # df.loc[regress_to_none_idx, "dmo_status"] = "none"

        self.module.update_vision_status()

        # ------------------------SELECTING INDIVIDUALS FOR FOR DR OR DMO EYE EXAM/SCREENING---------------------
        df.selected_for_eye_screening = False

        eligible_population_for_eye_screening = (
                (df.is_alive & df.nc_diabetes) &  # todo add condition for people not to be selected again witin 1 year
                # (df.dr_status == 'none') &
                # (df.dmo_status == 'none') &
                # (df["dr_dmo_diagnosis_pathway"] == "none") &
                (df.dr_dmo_diagnosis_pathway == 'none') &
                (df.age_years >= 20) &
                (pd.isna(df.dr_date_diagnosis)) &
                # Add time since last screening condition
                ((df.dr_date_last_screening.isna()) |
                 (df.dr_date_last_screening < self.sim.date - pd.DateOffset(years=1)))
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
            symptoms: List[str],
            schedule_hsi_event: HSIEventScheduler,
            **kwargs,
    ) -> None:
        if ('reduced_vision' in symptoms
            and individual_properties['nc_diabetes']
            and not individual_properties['dr_diagnosed']):
            schedule_hsi_event(
                HSI_Dr_Dmo_Symptomatic_Presentation(self, person_id),
                priority=0,
                topen=self.sim.date,
            )


class HSI_Dr_Dmo_Symptomatic_Presentation(HSI_Event, IndividualScopeEventMixin):
    """Symptom-driven (reduced_vision) eye examination. Triggered by the SymptomManager.
    People entering this pathway may present more advanced disease than those selected through screening"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Dmo_Symptomatic_Presentation'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2' #todo should also include level 3?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        # if person["dr_on_treatment"] and not pd.isna(person["dr_date_last_screening"]):
        #     return hs.get_blank_appt_footprint()
        if person["dr_on_treatment"] or person["dmo_on_treatment"]:
            return hs.get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['eye_screening']
        )
        # todo should I create another test so that the second one is for dmo?
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='dilated_eye_exam_dr_dmo',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            df.at[person_id, 'dr_date_diagnosis'] = self.sim.date
            df.at[person_id, 'dr_date_treatment'] = self.sim.date
            df.at[person_id, 'dr_date_last_screening'] = self.sim.date
            df.at[person_id, 'dr_diagnosed'] = True
            df.at[person_id, 'dr_stage_at_which_treatment_given'] = df.at[person_id, 'dr_status']
            df.at[person_id, 'dr_dmo_diagnosis_pathway'] = 'symptomatic'

            # Set diabetes ever diagnosed to true if not already set
            if not df.at[person_id, 'nc_diabetes_ever_diagnosed']:
                df.at[person_id, 'nc_diabetes_ever_diagnosed'] = True

            # If consumables are available, add equipment used
            self.add_equipment({'Visual acuity chart', 'Direct ophthalmoscope',
                                'Fundus camera'})

            if person.dr_status == 'mild_or_moderate' or person.dmo_status == 'non_clinically_significant':
                # schedule HSI_CardioMetabolicDisorders_StartWeightLossAndMedication
                # and repeat HSI_DR_Dmo_Screening in 1 year
                hs.schedule_hsi_event(
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
                next_screening_if_mild_or_moderate = \
                    self.sim.date + pd.DateOffset(months=p['next_screening_if_mild_or_moderate'])
                hs.schedule_hsi_event(self,
                                      topen=next_screening_if_mild_or_moderate,
                                      tclose=None,
                                      priority=1)

            elif person.dr_status == 'severe':
                next_screening_if_severe = self.sim.date + pd.DateOffset(months=p['next_screening_if_severe'])
                hs.schedule_hsi_event(self,
                                      topen=next_screening_if_severe,
                                      tclose=None,
                                      priority=1)

            if person.dr_status == 'proliferative':
                hs.schedule_hsi_event(
                    hsi_event=HSI_Dr_Laser_Pan_Retinal_Coagulation(self.module, person_id),
                    topen=self.sim.date,
                    priority=0
                )

            if person.dmo_status == 'clinically_significant':
                # Randomise between AntiVEGF or Focal Laser
                treatment_for_cs_dmo = HSI_Dr_Dmo_AntiVEGF \
                    if self.module.rng.random_sample() < p[
                    'prob_antivegf_for_clinically_significant_dmo'] else HSI_Dr_Dmo_Focal_Laser
                hs.schedule_hsi_event(
                    hsi_event=treatment_for_cs_dmo(module=self.module, person_id=person_id),
                    topen=self.sim.date,
                    priority=0
                )


class HSI_Dr_Dmo_Screening(HSI_Event, IndividualScopeEventMixin):
    """This is the eye examination/screening done to individuals selected for screening
    for individuals with any complication"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Dmo_Eye_Examination'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3' #todo should also include level 2
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        # if person["dr_on_treatment"] and not pd.isna(person["dr_date_last_screening"]):
        #     return hs.get_blank_appt_footprint()
        if person["dr_on_treatment"] or person["dmo_on_treatment"]:
            return hs.get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['eye_screening']
        )
        # todo should I create another test so that the second one is for dmo?
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='dilated_eye_exam_dr_dmo',
            hsi_event=self
        )

        if dx_result and is_cons_available:
            df.at[person_id, 'dr_date_diagnosis'] = self.sim.date
            df.at[person_id, 'dr_date_treatment'] = self.sim.date
            df.at[person_id, 'dr_date_last_screening'] = self.sim.date
            df.at[person_id, 'dr_diagnosed'] = True
            df.at[person_id, 'dr_stage_at_which_treatment_given'] = df.at[person_id, 'dr_status']
            df.at[person_id, 'dr_dmo_diagnosis_pathway'] = 'screening'

            # Set diabetes ever diagnosed to true if not already set
            if not df.at[person_id, 'nc_diabetes_ever_diagnosed']:
                df.at[person_id, 'nc_diabetes_ever_diagnosed'] = True

            # If consumables are available, add equipment used
            self.add_equipment({'Visual acuity chart', 'Direct ophthalmoscope',
                                'Fundus camera'})

            if person.dr_status == 'mild_or_moderate' or person.dmo_status == 'non_clinically_significant':
                # schedule HSI_CardioMetabolicDisorders_StartWeightLossAndMedication
                # and repeat HSI_DR_Dmo_Screening in 1 year
                hs.schedule_hsi_event(
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
                next_screening_if_mild_or_moderate = \
                    self.sim.date + pd.DateOffset(months=p['next_screening_if_mild_or_moderate'])
                hs.schedule_hsi_event(self,
                                      topen=next_screening_if_mild_or_moderate,
                                      tclose=None,
                                      priority=1)

            elif person.dr_status == 'severe':
                next_screening_if_severe = self.sim.date + pd.DateOffset(months=p['next_screening_if_severe'])
                hs.schedule_hsi_event(self,
                                      topen=next_screening_if_severe,
                                      tclose=None,
                                      priority=1)

            if person.dr_status == 'proliferative':
                hs.schedule_hsi_event(
                    hsi_event=HSI_Dr_Laser_Pan_Retinal_Coagulation(self.module, person_id),
                    topen=self.sim.date,
                    priority=0
                )

            if person.dmo_status == 'clinically_significant':
                # Randomise between AntiVEGF or Focal Laser
                treatment_for_cs_dmo = HSI_Dr_Dmo_AntiVEGF \
                    if self.module.rng.random_sample() < p[
                    'prob_antivegf_for_clinically_significant_dmo'] else HSI_Dr_Dmo_Focal_Laser
                hs.schedule_hsi_event(
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        # if not already diagnosed, do nothing
        if not person["dr_diagnosed"]:
            return hs.get_blank_appt_footprint()

        # assert pd.isnull(df.at[person_id, 'dr_date_treatment'])

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

            # Apply treatment success
            self.module.do_treatment_success_dmo(person_id, 'focal_laser')

            # todo find out if this is the case?
            # Treatment is complete after this procedure, regardless of success while the
            # person remains eligible for natural-history transitions
            df.at[person_id, 'dmo_on_treatment'] = False

            # Schedule follow-up checkup after 3 months #todo add chance that person needs after 3 months
            follow_up_date = self.sim.date + pd.DateOffset(months=3)
            hs.schedule_hsi_event(
                hsi_event=HSI_Dr_Dmo_Review(self.module, person_id),
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        # If not already diagnosed, do nothing
        if not person["dr_diagnosed"]:
            return hs.get_blank_appt_footprint()

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


            # Continue monthly injections until the course reaches 4 injections.
            if df.at[person_id, 'dmo_antivegf_count'] < 4:
                next_session = self.sim.date + pd.DateOffset(months=1)
                hs.schedule_hsi_event(self, topen=next_session, priority=0,)

            else:
                # Assess treatment success once, at the end of the course.
                self.module.do_treatment_success_dmo(person_id, 'antivegf')

                # Reset the counter and end the treatment course.
                df.at[person_id, 'dmo_antivegf_count'] = 0
                df.at[person_id, 'dmo_on_treatment'] = False


class HSI_Dr_Laser_Pan_Retinal_Coagulation(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI event given to individuals with proliferative diabetic retinopathy after undergoing an eye
    exam/screening
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, DiabeticRetinopathy)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dr_Laser_Pan_Retinal_Coagulation_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Laser_Pan_Retinal_Coagulation for person {person_id}')
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        # if person not yet diagnosed, do nothing
        if (
            not person["dr_diagnosed"]
            or df.at[person_id, 'dr_status'] != 'proliferative'
        ):
            return hs.get_blank_appt_footprint()

        is_cons_available = self.get_consumables(
            self.module.cons_item_codes['laser_pan_retinal_photocoagulation']
        )

        if is_cons_available:
            self.add_equipment({'Visual acuity chart', 'Opthalmic laser system',
                                'Silt lamp laser delivery system',
                                'Indirect laser delivery system', 'Laser wide-field contact lens'})
            df.at[person_id, 'dr_on_treatment'] = True
            # Increment the per-person session count
            df.at[person_id, 'total_laser_pan_retinal_photocoagulation_sessions'] += 1
            sessions = df.at[person_id, 'total_laser_pan_retinal_photocoagulation_sessions']

            # Marks treatment started on first session of each cycle
            # Odd-numbered session means session 1 of cycle
            if sessions % 2 == 1:
                df.at[person_id, 'on_laser_pan_retinal_photocoagulation_treatment'] = True
                df.at[person_id, 'dr_date_prp_treatment'] = self.sim.date
                df.at[person_id, 'dr_stage_at_which_treatment_given'] = df.at[person_id, 'dr_status']

                # Schedule session 2 in 1 week
                next_session_date = self.sim.date + pd.DateOffset(weeks=1)
                hs.schedule_hsi_event(
                    self,
                    topen=next_session_date,
                    tclose=next_session_date + pd.DateOffset(days=3),
                    priority=1
                )
            else:
                self.module.do_treatment_success_proliferative(person_id)
                df.at[person_id, 'on_laser_pan_retinal_photocoagulation_treatment'] = False

                # Schedule follow-up and/or review at 2 months
                follow_up = self.sim.date + pd.DateOffset(months=2)
                hs.schedule_hsi_event(
                    HSI_Dr_Dmo_Review(self.module, person_id),
                    topen=follow_up,
                    priority=1
                )

                # Schedule reviews at 3, 6, 9, 12 months
                # for m in [3, 6, 9, 12]:
                #     hs.schedule_hsi_event(
                #         self,
                #         topen=self.sim.date + pd.DateOffset(months=m),
                #         priority=1
                #     )


class HSI_Dr_Dmo_Review(HSI_Event, IndividualScopeEventMixin):
    """Review appointment after pan-retinal laser"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'Dr_Dmo_Review'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        p = self.module.parameters
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return hs.get_blank_appt_footprint()

        if df.at[person_id, 'dr_status'] == 'proliferative':
            if self.module.rng.random_sample() < p['prob_repeat_laser']:
                # Schedule new treatment cycle
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Dr_Laser_Pan_Retinal_Coagulation(module=self.module, person_id=person_id),
                    topen=self.sim.date,
                    priority=0
                )
        if df.at[person_id, 'dmo_status'] == 'clinically_significant':
            if self.module.rng.random_sample() < p['prob_repeat_dmo_treatment']:
                treatment_for_cs_dmo = HSI_Dr_Dmo_AntiVEGF \
                    if self.module.rng.random_sample() < p[
                        'prob_antivegf_for_clinically_significant_dmo'] else HSI_Dr_Dmo_Focal_Laser
                hs.schedule_hsi_event(
                    hsi_event=treatment_for_cs_dmo(module=self.module, person_id=person_id),
                    topen=self.sim.date,
                    priority=0
                )


class DiabeticRetinopathyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 month"""
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        df = population.props

        # CURRENT STATUS COUNTS
        # Create dictionary for each subset, adding prefix to key name, and adding to make a flat dict for logging.
        out = {}

        # # Vision-related stats
        # vision_out = {
        #     'total_normal_vision': (df.vision_status == 'normal').sum(),
        #     'total_moderate_vision_impairment': (df.vision_status == 'moderate_vision_impairment').sum(),
        #     'total_severe_vision_impairment': (df.vision_status == 'severe_vision_impairment').sum(),
        #     'total_blindness': (df.vision_status == 'blindness').sum(),
        #     'vision_loss_due_to_dr': df.vision_loss_due_to_dr.sum(),
        #     'total_diagnosed': df.dr_diagnosed.sum(),
        #     'total_on_treatment': df.dr_on_treatment.sum(),
        #     'treatment_success_rate': df[df.dr_on_treatment & (df.vision_status == 'normal')].shape[0] /
        #                               max(1, df.dr_on_treatment.sum())
        # }
        #
        # out.update(vision_out)

        # DR prevalence
        # out.update({
        #     f'total_dr_{k}': v
        #     for k, v in df.loc[df.is_alive].dr_status.value_counts().items()
        # })
        # # DMO prevalence
        # out.update({
        #     f'total_dmo_{k}': v
        #     for k, v in df.loc[df.is_alive].dmo_status.value_counts().items()
        # })
        # # Vision impairment prevalence
        # out.update({
        #     f'total_vision_{k}': v
        #     for k, v in df.loc[df.is_alive].vision_status.value_counts().items()
        # })

        #Prevalence
        alive = df.loc[df.is_alive]

        for state in df.dr_status.cat.categories:
            # out[f'total_dr_{state}'] = (alive.dr_status == state).sum()
            out[f'total_dr_{state}'] = int((alive.dr_status == state).sum())

        for state in df.dmo_status.cat.categories:
            # out[f'total_dmo_{state}'] = (alive.dmo_status == state).sum()
            out[f'total_dmo_{state}'] = int((alive.dmo_status == state).sum())

        for state in df.vision_status.cat.categories:
            # out[f'total_vision_{state}'] = (alive.vision_status == state).sum()
            out[f'total_vision_{state}'] = int((alive.vision_status == state).sum())

        for pathway in ['screening', 'symptomatic']:
            mask = alive.dr_dmo_diagnosis_pathway == pathway
            out[f'n_diagnosed_{pathway}'] = int(mask.sum())
            out[f'n_on_treatment_{pathway}'] = int((mask & alive.dr_on_treatment).sum())
            for state in df.dr_status.cat.categories:
                out[f'total_dr_{state}_{pathway}'] = int((mask & (alive.dr_status == state)).sum())
            for state in df.vision_status.cat.categories:
                out[f'total_vision_{state}_{pathway}'] = int((mask & (alive.vision_status == state)).sum())

        # To calculate prevalence among total alive diabetics
        out['total_alive_diabetics'] = int(alive.nc_diabetes.sum())

        # DR incidence
        dr_inc = {
            f'inc_dr_{k}': len(v)
            for k, v in self.module.dr_incident_case_tracker.items()
        }

        # DMO incidence
        dmo_inc = {
            f'inc_dmo_{k}': len(v)
            for k, v in self.module.dmo_incident_case_tracker.items()
        }

        # Vision incidence
        vision_inc = {
            f'inc_vision_{k}': len(v)
            for k, v in self.module.vision_incident_case_tracker.items()
        }

        out.update(dr_inc)
        out.update(dmo_inc)
        out.update(vision_inc)

        logger.info(key='summary_stats', data=out)

        self.module.reset_incidence_trackers()
