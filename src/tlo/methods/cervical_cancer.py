"""
Cervical Cancer Disease Module

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.cancer_consumables import get_consumable_item_codes_cancers
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
    from tlo.population import IndividualProperties

from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CervicalCancer(Module, GenericFirstAppointmentsMixin):
    """Cervical Cancer Disease Module"""

    INIT_DEPENDENCIES = {
        'Demography', 'HealthSystem', 'Lifestyle', 'SymptomManager', 'Hiv',
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden', 'HealthSeekingBehaviour'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'CervicalCancer': Cause(gbd_causes='Cervical cancer', label='Cancer (Cervix)'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'CervicalCancer': Cause(gbd_causes='Cervical cancer', label='Cancer (Cervix)'),
    }

    PARAMETERS = {
        "init_prev_cin_hpv_cc_stage_hiv": Parameter(
            Types.LIST,
            "initial proportions in hpv cancer categories in women with hiv"
        ),
        "init_prev_cin_hpv_cc_stage_nhiv": Parameter(
            Types.LIST,
            "initial proportions in hpv cancer categories in women without hiv"
        ),
        "r_hpv": Parameter(
            Types.REAL,
            "probability per month of oncogenic hpv infection",
        ),
        "r_cin1_hpv": Parameter(
            Types.REAL,
            "probability per month of incident cin1 amongst people with hpv",
        ),
        "prob_revert_from_cin1": Parameter(
            Types.REAL,
            "probability of reverting from cin1 to none",
        ),
        "r_cin2_cin1": Parameter(
            Types.REAL,
            "probability per month of incident cin2 amongst people with cin1",
        ),
        "r_cin3_cin2": Parameter(
            Types.REAL,
            "probability per month of incident cin3 amongst people with cin2",
        ),
        "r_stage1_cin3": Parameter(
            Types.REAL,
            "probability per month of incident stage1 cervical cancer amongst people with cin3",
        ),
        "r_stage2a_stage1": Parameter(
            Types.REAL,
            "probability per month of incident stage2a cervical cancer amongst people with stage1",
        ),
        "r_stage2b_stage2a": Parameter(
            Types.REAL,
            "probability per month of incident stage2b cervical cancer amongst people with stage2a",
        ),
        "r_stage3_stage2b": Parameter(
            Types.REAL,
            "probability per month of incident stage3 cervical cancer amongst people with stage2b",
        ),
        "r_stage4_stage3": Parameter(
            Types.REAL,
            "probability per month of incident stage4 cervical cancer amongst people with stage3",
        ),
        "rr_progress_cc_hiv": Parameter(
            Types.REAL, "risk ratio for progressing through cin and cervical cancer stages if have unsuppressed hiv"
        ),
        "rr_hpv_vaccinated": Parameter(
            Types.REAL,
            "risk ratio for hpv if vaccinated - this is combined effect of probability the hpv is "
            "vaccine-preventable and vaccine efficacy against vaccine-preventable hpv ",
        ),
        "rr_hpv_age50plus": Parameter(
            Types.REAL,
            "risk ratio for hpv if age 50 plus"
        ),
        "prob_cure_stage1": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 1 cervical cancer",
        ),
        "prob_cure_stage2a": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 2a cervical cancer",
        ),
        "prob_cure_stage2b": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 2b cervical cancer",
        ),
        "prob_cure_stage3": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 3 cervical cancer",
        ),
        "r_death_cervical_cancer": Parameter(
            Types.REAL,
            "probability per month of death from cervical cancer amongst people with stage 4 cervical cancer",
        ),
        "p_vaginal_bleeding_if_no_cc": Parameter(
            Types.REAL,
            "Probability of vaginal bleeding onset if not in stage 1/2/3/4"
        ),
        "r_vaginal_bleeding_cc_stage1": Parameter(
            Types.REAL, "probability of vaginal bleeding if have stage 1 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2a": Parameter(
            Types.REAL, "risk ratio for vaginal bleeding if have stage 2a cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2b": Parameter(
            Types.REAL, "risk ratio for vaginal bleeding if have stage 2b cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage3": Parameter(
            Types.REAL, "risk ratio for vaginal bleeding if have stage 3 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage4": Parameter(
            Types.REAL, "risk ratio for vaginal bleeding if have stage 4 cervical cancer"
        ),
        "prob_referral_biopsy_given_vaginal_bleeding": Parameter(
            Types.REAL, "probability of being referred for a biopsy if presenting with vaginal bleeding"
        ),
        "sensitivity_of_biopsy_for_cervical_cancer": Parameter(
            Types.REAL, "sensitivity of biopsy for diagnosis of cervical cancer"
        ),
        "sensitivity_of_xpert_for_hpv_cin_cc": Parameter(
            Types.REAL, "sensitivity of xpert for presence of hpv, cin or cervical cancer"
        ),
        "sensitivity_of_via_for_cin_cc": Parameter(
            Types.REAL, "sensitivity of via for cin and cervical cancer"
        ),
        "prob_xpert_screen": Parameter(
            Types.REAL, "probability of xpert screening"
        ),
        "prob_via_screen": Parameter(
            Types.REAL, "probability of via screening"
        ),
        "prob_thermoabl_successful": Parameter(
            Types.REAL, "probability of thermoablation treatment successful in removing CIN (ce_hpv_cc_status set to "
                        "none)"
        ),
        "prob_cryotherapy_successful": Parameter(
            Types.REAL, "probability of cryotherapy treatment successful in removing CIN (ce_hpv_cc_status set to none)"
        ),
        "transition_screening_method_year": Parameter(
            Types.REAL, "year screening method switches from VIA to Xpert"
        ),
        "transition_to_thermo_year": Parameter(
            Types.REAL, "year CIN removal method switches from Cryo to Thermo"
        ),
        "screening_min_age_hv_neg": Parameter(
            Types.REAL, "minimum age individual to be screened if HIV negative"
        ),
        "screening_max_age_hv_neg": Parameter(
            Types.REAL, "maximum age individual to be screened if HIV negative"
        ),
        "screening_min_age_hv_pos": Parameter(
            Types.REAL, "minimum age individual to be screened if HIV positive"
        ),
        "screening_max_age_hv_pos": Parameter(
            Types.REAL, "maximum age individual to be screened if HIV positive"
        ),
        "yrs_between_screen_hv_pos": Parameter(
            Types.REAL, "minimum years between screening if HIV positive"
        ),
        "yrs_between_screen_hv_neg": Parameter(
            Types.REAL, "minimum years between screening if HIV negative"
        ),
        "palliative_care_bed_days": Parameter(
            Types.REAL, "palliative_care_bed_days"
        ),
        "stage_1_3_daly_wt": Parameter(
            Types.REAL, "GBD sequalae code corresponding to stage_1_3_daly_wt"
        ),
        "stage_1_3_treated_daly_wt": Parameter(
            Types.REAL, "GBD sequalae code corresponding to stage_1_3_treated_daly_wt"
        ),
        "stage4_daly_wt": Parameter(
            Types.REAL, "GBD sequalae code corresponding tostage4_daly_wt"
        ),
        "min_yrs_between_screening_if_cin_treated": Parameter(
            Types.REAL, "minimum years between screening if individual has been treated for CIN previously"
        )
    }

    PROPERTIES = {
        "ce_hpv_cc_status": Property(
            Types.CATEGORICAL,
            "Current hpv / cervical cancer status - note that hpv means persistent hpv",
            categories=["none", "hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_date_diagnosis": Property(
            Types.DATE,
            "the date of diagnosis of cervical cancer stage (pd.NaT if never diagnosed)"
        ),
        "ce_stage_at_diagnosis": Property(
            Types.CATEGORICAL,
            "the cancer stage at which cancer diagnosis was made",
            categories=[ "none", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_date_cin_removal": Property(
            Types.DATE,
            "the date of last cin removal (pd.NaT if never had cin removal)"
        ),
        "ce_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),
        "ce_ever_screened": Property(
            Types.BOOL,
            "whether ever been screened"
        ),
        "ce_ever_treated": Property(
            Types.BOOL,
            "ever been treated for cc"
        ),
        "ce_cured_date_cc": Property(
            Types.DATE,
            "ever cured of cervical cancer date"
        ),
        "ce_cc_ever": Property(
            Types.BOOL,
            "ever had cc"
        ),
        "ce_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment was given (because the treatment only has an effect during the stage"
            "at which it is given). (Note treatment only initiated in stage1/2a/2b/3",
            categories=["none", "hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "ce_ever_diagnosed": Property(
            Types.BOOL,
            "ever diagnosed with cervical cancer (even if now cured)"
        ),
        "ce_xpert_hpv_ever_pos": Property(
            Types.BOOL,
            "hpv positive on xpert test ever"
        ),
        "ce_via_cin_ever_detected": Property(
            Types.BOOL,
        "cin ever_detected on via"
        ),
        "ce_date_last_screened": Property(
          Types.DATE,
          "date of last screening"
        ),
        "ce_date_thermoabl": Property(
            Types.DATE,
        "date of thermoablation for CIN"
        ),
        "ce_date_cryotherapy": Property(
            Types.DATE,
            "date of cryotherapy for CIN"
        ),
        "ce_date_via": Property(
            Types.DATE,
            "date of VIA screening"
        ),
        "ce_date_xpert": Property(
            Types.DATE,
            "date of XPERT screening"
        ),
        "ce_current_cc_diagnosed": Property(
            Types.BOOL,
            "currently has diagnosed cervical cancer (which until now has not been cured)"
        ),
        "ce_biopsy": Property(
            Types.BOOL,
            "ce biopsy done ever"
        )
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.polling_event = None  # Will store the regular polling event

        self.hpv_cin_options = ["hpv", "cin1", "cin2", "cin3"]
        self.hpv_stage_options = ["stage1", "stage2a", "stage2b", "stage3", "stage4"]

        self.linear_models_for_progression_of_hpv_cc_status = dict()
        self.lm_onset_vaginal_bleeding = None
        self.daly_wts = dict()
        self.item_codes_cervical_can = dict()


    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            read_csv_files(Path(resourcefilepath) / "ResourceFile_Cervical_Cancer",
                           files="parameter_values")
        )

        # Declare symptom (with typical healthcare seeking)
        self.sim.modules['SymptomManager'].register_symptom(Symptom(name='vaginal_bleeding'))

    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props
        rng = self.rng
        p = self.parameters

        # defaults
        df.loc[df.is_alive, "ce_hpv_cc_status"] = "none"
        df.loc[df.is_alive, "ce_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "ce_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "ce_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "ce_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "ce_date_cin_removal"] = pd.NaT
        df.loc[df.is_alive, "ce_stage_at_diagnosis"] = "none"
        df.loc[df.is_alive, "ce_ever_treated"] = False
        df.loc[df.is_alive, "ce_cc_ever"] = False
        df.loc[df.is_alive, "ce_xpert_hpv_ever_pos"] = False
        df.loc[df.is_alive, "ce_via_cin_ever_detected"] = False
        df.loc[df.is_alive, "ce_date_thermoabl"] = pd.NaT
        df.loc[df.is_alive, "ce_date_cryotherapy"] = pd.NaT
        df.loc[df.is_alive, "ce_date_via"] = pd.NaT
        df.loc[df.is_alive, "ce_date_xpert"] = pd.NaT
        df.loc[df.is_alive, 'ce_current_cc_diagnosed'] = False
        df.loc[df.is_alive, "ce_biopsy"] = False
        df.loc[df.is_alive, "ce_ever_screened"] = False
        df.loc[df.is_alive, "ce_ever_diagnosed"] = False
        df.loc[df.is_alive, "ce_cured_date_cc"] = pd.NaT
        df.loc[df.is_alive, "ce_date_last_screened"] = pd.NaT

        # ------------------- SET INITIAL CE_HPV_CC_STATUS -------------------------------------------------------------
        women_over_15_nhiv_idx = df.index[(df["age_years"] > 15) & (df["sex"] == 'F') & ~df["hv_inf"]]

        df.loc[women_over_15_nhiv_idx, 'ce_hpv_cc_status'] = rng.choice(
            ['none', 'hpv', 'cin1', 'cin2', 'cin3', 'stage1', 'stage2a', 'stage2b', 'stage3', 'stage4'],
            size=len(women_over_15_nhiv_idx), p=p['init_prev_cin_hpv_cc_stage_nhiv']
        )

        women_15plus_hiv_idx = df.index[(df["age_years"] >= 15) & (df["sex"] == 'F') & df["hv_inf"]]

        df.loc[women_15plus_hiv_idx, 'ce_hpv_cc_status'] = rng.choice(
            ['none', 'hpv', 'cin1', 'cin2', 'cin3', 'stage1', 'stage2a', 'stage2b', 'stage3', 'stage4'],
            size=len(women_15plus_hiv_idx), p=p['init_prev_cin_hpv_cc_stage_hiv']
        )

        # -------------------- symptoms, diagnosis, treatment  -----------
        # For simplicity we assume all these are null at baseline - we don't think this will influence population
        # status in the present to any significant degree

    def initialise_simulation(self, sim):
        """
        * Schedule the main polling event
        * Schedule the main logging event
        * Define the LinearModels
        * Define the Diagnostic used
        * Define the Disability-weights
        * Schedule the palliative care appointments for those that are on palliative care at initiation
        """

        p = self.parameters
        sim = self.sim

        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        self.polling_event = CervicalCancerMainPollingEvent(self)
        sim.schedule_event(self.polling_event, sim.date)

        # ----- SCHEDULE LOGGING EVENTS -----
        # Schedule logging event to happen immediately
        sim.schedule_event(CervicalCancerLoggingEvent(self), sim.date)

        # Look-up consumable item codes
        self.item_codes_cervical_can = get_consumable_item_codes_cancers(self)

        # Build linear models
        self.build_linear_models()

        # ----- DX TESTS -----
        # Create the diagnostic test representing screening and the use of a biopsy
        # in future could add different sensitivity according to target category

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_cervical_cancer=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_cervical_cancer'],
                target_categories=["stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            screening_with_xpert_for_cin_and_cervical_cancer =DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_xpert_for_hpv_cin_cc'],
                target_categories=["hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            screening_with_via_for_cin_and_cervical_cancer=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_via_for_cin_cc'],
                target_categories=["cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        # ----- DISABILITY-WEIGHT -----
        if "HealthBurden" in self.sim.modules:
            # For those with cancer (any stage prior to stage 4) and never treated
            self.daly_wts["stage_1_3"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=p['stage_1_3_daly_wt']
                # "Diagnosis and primary therapy phase of cervical cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to stage 4) and has been treated
            self.daly_wts["stage_1_3_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=p['stage_1_3_treated_daly_wt']
                # "Controlled phase of cervical cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in stage 4: no palliative care
            self.daly_wts["stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code = p['stage4_daly_wt']
                # "Metastatic phase of cervical cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in stage 4: with palliative care
            self.daly_wts["stage4_palliative_care"] = self.daly_wts["stage_1_3"]
            # By assumption, we say that the weight for those in stage 4 with palliative care is the same as
            # that for those with stage 1-3 cancers.


    def build_linear_models(self) -> None:
        """Build linear models"""
        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 1 month period
        # NB. The effect being produced is that treatment only has the effect in the stage at which the
        # treatment was received.

        df = self.sim.population.props
        p = self.parameters
        lm = self.linear_models_for_progression_of_hpv_cc_status

        lm['hpv'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_hpv'],
            Predictor('va_hpv')
            .when(1, p['rr_hpv_vaccinated'])
            .when(2, p['rr_hpv_vaccinated']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(0,15)', 0.0)
            .when('>50', p['rr_hpv_age50plus']),
            Predictor('sex').when('M', 0.0),
            Predictor('ce_hpv_cc_status').when('none', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['cin1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin1_hpv'],
            Predictor('ce_hpv_cc_status').when('hpv', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
            # Assume no progression if ever has had treatment
            Predictor('ce_ever_treated', conditions_are_exhaustive=True, conditions_are_mutually_exclusive=True).when(
                True, 0.0).when(False, 1.0),

        )

        lm['cin2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin2_cin1'],
            Predictor('ce_hpv_cc_status').when('cin1', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['cin3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin3_cin2'],
            Predictor('ce_hpv_cc_status').when('cin2', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage1_cin3'],
            Predictor('ce_hpv_cc_status').when('cin3', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['stage2a'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2a_stage1'],
            Predictor('ce_hpv_cc_status').when('stage1', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['stage2b'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2b_stage2a'],
            Predictor('ce_hpv_cc_status').when('stage2a', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage3_stage2b'],
            Predictor('ce_hpv_cc_status').when('stage2b', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        lm['stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage4_stage3'],
            Predictor('ce_hpv_cc_status').when('stage3', 1.0).otherwise(0.0),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "on_not_vl_suppressed")',
                (p['rr_progress_cc_hiv'])),
            Predictor().when(
                'hv_inf & '
                '(hv_art == "not")',
                (p['rr_progress_cc_hiv'])),
        )

        # Check that the dict labels are correct as these are used to set the value of ce_hpv_cc_status
        if not set(lm).union({'none'}) == set(df.ce_hpv_cc_status.cat.categories):
            logger.warning(key="warning", data="Label dict are not correct.")

        # Linear Model for the onset of vaginal bleeding, in each 1 month period
        # Create variables for used to predict the onset of vaginal bleeding at
        # various stages of the disease


        # Build linear model of vaginal bleeding
        p_vaginal_bleeding_if_no_cc = p['p_vaginal_bleeding_if_no_cc']
        stage1 = p['r_vaginal_bleeding_cc_stage1']
        stage2a = p['rr_vaginal_bleeding_cc_stage2a'] * p['r_vaginal_bleeding_cc_stage1']
        stage2b = p['rr_vaginal_bleeding_cc_stage2b'] * p['r_vaginal_bleeding_cc_stage1']
        stage3 = p['rr_vaginal_bleeding_cc_stage3'] * p['r_vaginal_bleeding_cc_stage1']
        stage4 = p['rr_vaginal_bleeding_cc_stage4'] * p['r_vaginal_bleeding_cc_stage1']

        self.lm_onset_vaginal_bleeding = LinearModel.multiplicative(
            Predictor('sex', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True).when('M', 0.0)
                                                                                                    .when('F', 1.0),
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            ).when('none', p_vaginal_bleeding_if_no_cc)
             .when('cin1', p_vaginal_bleeding_if_no_cc)
             .when('cin2', p_vaginal_bleeding_if_no_cc)
             .when('cin3', p_vaginal_bleeding_if_no_cc)
             .when('stage1', stage1)
             .when('stage2a', stage2a)
             .when('stage2b', stage2b)
             .when('stage3', stage3)
             .when('stage4', stage4)
        )

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, "ce_hpv_cc_status"] = "none"
        df.at[child_id, "ce_date_treatment"] = pd.NaT
        df.at[child_id, "ce_stage_at_which_treatment_given"] = "none"
        df.at[child_id, "ce_date_diagnosis"] = pd.NaT
        df.at[child_id, "ce_date_palliative_care"] = pd.NaT
        df.at[child_id, "ce_date_cin_removal"] = pd.NaT
        df.at[child_id, "ce_stage_at_diagnosis"] = 'none'
        df.at[child_id, "ce_ever_treated"] = False
        df.at[child_id, "ce_cc_ever"] = False
        df.at[child_id, "ce_xpert_hpv_ever_pos"] = False
        df.at[child_id, "ce_via_cin_ever_detected"] = False
        df.at[child_id, "ce_date_thermoabl"] = pd.NaT
        df.loc[child_id, "ce_date_cryotherapy"] = pd.NaT
        df.loc[child_id, "ce_date_via"] = pd.NaT
        df.loc[child_id, "ce_date_xpert"] = pd.NaT
        df.at[child_id, "ce_current_cc_diagnosed"] = False
        df.at[child_id, "ce_biopsy"] = False
        df.at[child_id, "ce_ever_screened"] = False
        df.at[child_id, "ce_ever_diagnosed"] = False
        df.at[child_id, "ce_cured_date_cc"] = pd.NaT
        df.at[child_id, "ce_date_last_screened"] = pd.NaT

    def report_daly_values(self):

        # This must send back a dataframe that reports on the HealthStates for all individuals over the past month

        df = self.sim.population.props  # shortcut to population properties dataframe for alive persons

        disability_series_for_alive_persons = pd.Series(index=df.index[df.is_alive], data=0.0)

        # Assign daly_wt to those with cancer stages before stage4 and have either never been treated or are no longer
        # in the stage in which they were treated
        disability_series_for_alive_persons.loc[
            df.ce_hpv_cc_status.isin(['stage1', 'stage2a', 'stage2b', 'stage3'])
        ] = self.daly_wts['stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated and who are still in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                pd.notnull(df.ce_date_treatment)
                & df.ce_hpv_cc_status.isin(['stage1', 'stage2a', 'stage2b', 'stage3'])
                & (df.ce_hpv_cc_status == df.ce_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['stage_1_3_treated']

        # Assign daly_wt to those in stage4 cancer (who have not had palliative care)
        disability_series_for_alive_persons.loc[
            (df.ce_hpv_cc_status == "stage4") &
            (pd.isnull(df.ce_date_palliative_care))
            ] = self.daly_wts['stage4']

        # Assign daly_wt to those in stage4 cancer, who have had palliative care
        disability_series_for_alive_persons.loc[
            (df.ce_hpv_cc_status == "stage4") &
            (~pd.isnull(df.ce_date_palliative_care))
            ] = self.daly_wts['stage4_palliative_care']

        return disability_series_for_alive_persons

    def do_at_generic_first_appt(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        if 'vaginal_bleeding' in symptoms:
            schedule_hsi_event(
                HSI_CervicalCancer_Presentation_WithVaginalBleeding(
                    person_id=person_id,
                    module=self
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None)

    def choose_cin_procedure_and_schedule_it(self, person_id):
        """Function to decide treatment for individuals with CIN based on year and to schedule the relevant HSI.
        If year is >= transition-year then Thermoablation, else  Cryotherapy
        :param hsi_event: HSI Event (required to pass in order to register equipment)
        :param person_id: person of interest
        """
        year = self.sim.date.year
        p = self.parameters
        hs = self.sim.modules["HealthSystem"]
        treatment_methods = {
            'Thermoablation': HSI_CervicalCancer_CINRemoval_Thermoablation,
            'Cryotherapy': HSI_CervicalCancer_CINRemoval_Cryotherapy,
        }

        selected_method = 'Thermoablation' if year >= p['transition_to_thermo_year'] else 'Cryotherapy'

        # Schedule HSI event
        hs.schedule_hsi_event(
            hsi_event=treatment_methods[selected_method](module=self, person_id=person_id),
            priority=0,
            topen=self.sim.date,
            tclose=None
        )

# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class CervicalCancerMainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all cervical cancer properties for population:
    * Acquisition and progression of hpv, cin, cervical cancer
    * Symptom Development according to stage of cervical Cancer
    * Deaths from cervical cancer for those in stage4
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # scheduled to run every 1 month: do not change as this is hard-wired into the values of all the parameters.
        self.selected_for_screening_last_month = {}  # Will store number screened for logging purposes.

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        year = self.sim.date.year
        m = self.module
        rng = m.rng
        p = m.parameters

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (ce_hpv_cc_status) -------------------------------

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.
        for stage, lm in reversed(list(self.module.linear_models_for_progression_of_hpv_cc_status.items())):
            # Note (1): terms in the LinearModels make the progression only applicable in the "origin" class.
            # Note (2): stages are iterated in reverse order to prevent a person progressing through all stages in one
            # time-step
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng)
            idx_gets_new_stage = gets_new_stage[gets_new_stage].index

            df.loc[idx_gets_new_stage, 'ce_hpv_cc_status'] = stage

        # Apply the reversion probability to change some 'cin1' to 'none'
        has_cin1 = (        # Identify rows where the status is 'cin1'
            df.is_alive &
            (df.sex == 'F') &
            (df.ce_hpv_cc_status == 'cin1')
        )
        df.loc[has_cin1, 'ce_hpv_cc_status'] = np.where(
            self.module.rng.random(size=len(df[has_cin1])) < p['prob_revert_from_cin1'],
            'none',
            df.loc[has_cin1, 'ce_hpv_cc_status']
        )

        # Update the ce_cc_ever property
        df.loc[
            (df['is_alive']) & (~df['ce_cc_ever']),  # Apply only if is_alive is True and ce_cc_ever is not True
            'ce_cc_ever'
        ] = (
            (df['ce_hpv_cc_status'].isin(['stage1', 'stage2a', 'stage2b', 'stage3', 'stage4']))
            | df['ce_ever_treated']
        )

        # -------------------------------- SCREENING FOR CERVICAL CANCER USING XPERT HPV TESTING AND VIA---------------
        # A subset of women will receive a screening test. Age of eligibility for screening depending on HIV status

        # Define screening age and interval criteria based on HIV status
        # Individuals with HIV are recommended for screening earlier and with more frequency
        # Individuals who have been treated for CIN previously are recommended for screening with more frequency
        age_min_to_be_eligible_for_screening = np.where(df.hv_diagnosed,
                                                        p['screening_min_age_hv_pos'],
                                                        p['screening_min_age_hv_neg'])
        age_max_to_be_eligible_for_screening = np.where(df.hv_diagnosed,
                                                        p['screening_max_age_hv_pos'],
                                                        p['screening_max_age_hv_neg'])
        screening_interval_in_days = (np.where(df.hv_diagnosed,
                                               p['yrs_between_screen_hv_pos'],
                                               p['yrs_between_screen_hv_neg']
                                               ) * DAYS_IN_YEAR
                                      ).astype(int)
        days_since_last_screen = (self.sim.date - df.ce_date_last_screened).dt.days

        # Define the eligible population for screening
        eligible_population = (
                df.is_alive
                & (df.sex == 'F')
                & (~df.ce_current_cc_diagnosed)
                & (df.age_years.between(age_min_to_be_eligible_for_screening, age_max_to_be_eligible_for_screening,
                                      inclusive="left"))
                & (
                        df.ce_date_last_screened.isnull()   # Never screened
                        | (days_since_last_screen > screening_interval_in_days)  # ... or, due a screen
                        | (                                                        # .... time since cin treatment
                            (df["ce_date_cryotherapy"].notnull() | df["ce_date_thermoabl"].notnull())
                             & (days_since_last_screen > (p['min_yrs_between_screening_if_cin_treated'] * DAYS_IN_YEAR))
                          )
                )
        )
        eligible_idx = eligible_population[eligible_population].index

        # Screen eligible population
        screening_methods = {
            'VIA': {
                'prob': 'prob_via_screen',
                'event_class': HSI_CervicalCancer_Screening_VIA,
            },
            'Xpert': {
                'prob': 'prob_xpert_screen',
                'event_class': HSI_CervicalCancer_Screening_Xpert,
            }
        }
        selected_screening_method = 'VIA' if year < p['transition_screening_method_year'] else 'Xpert'
        method_info = screening_methods[selected_screening_method]

        # Randomly select for screening
        selected_for_screening = eligible_idx[rng.random(size=len(eligible_idx)) < p[method_info['prob']]]

        # Schedule HSI events
        self.sim.modules['HealthSystem'].schedule_batch_of_individual_hsi_events(
            hsi_event_class=method_info['event_class'],
            person_ids=selected_for_screening,
            priority=0,
            topen=self.sim.date,
            tclose=None,
            module=self.module,
        )

        # store number of those screened on this Event, for logging purposes.
        self.selected_for_screening_last_month = {selected_screening_method: len(selected_for_screening)}

        # -------------------- UPDATING OF SYMPTOM OF vaginal bleeding OVER TIME --------------------------------
        # Each time this event is called (every month) individuals with cervical cancer may develop the symptom of
        # vaginal bleeding.  Once the symptom is developed it never resolves naturally. It may trigger
        # health-care-seeking behaviour.
        already_has_vaginal_bleeding = self.sim.modules['SymptomManager'].who_has('vaginal_bleeding')
        onset_vaginal_bleeding = self.module.lm_onset_vaginal_bleeding.predict(
            df.loc[df.is_alive & ~df.index.isin(already_has_vaginal_bleeding)],
            rng
        )
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_vaginal_bleeding[onset_vaginal_bleeding].index.tolist(),
            symptom_string='vaginal_bleeding',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM cervical CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death date is spread across 90d interval.
        stage4_idx = df.index[df.is_alive & (df.ce_hpv_cc_status == "stage4")]

        selected_to_die = stage4_idx[
            rng.random_sample(size=len(stage4_idx)) < self.module.parameters['r_death_cervical_cancer']]

        days_spread = 90
        date_min = self.sim.date
        date_max = self.sim.date + pd.DateOffset(days=days_spread)
        for person_id in selected_to_die:
            random_death_date = random_date(date_min, date_max, rng)
            self.sim.schedule_event(
                CervicalCancer_DeathInStage4(self.module, person_id), random_death_date
            )

# ---------------------------------------------------------------------------------------------------------
#   NATURAL HISTORY EVENTS
# ---------------------------------------------------------------------------------------------------------

class CervicalCancer_DeathInStage4(Event, IndividualScopeEventMixin):
    """Does a Cervical Cancer death for a person in Stage 4."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        person = self.sim.population.props.loc[person_id, ['is_alive', 'ce_hpv_cc_status']]

        # Only kill the person is they are alive and still in stage4
        if not (person['is_alive'] and (person['ce_hpv_cc_status'] == 'stage4')):
            return

        self.sim.modules['Demography'].do_death(individual_id=person_id,
                                                originating_module=self.module,
                                                cause='CervicalCancer')

# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_CervicalCancer_Screening_VIA(HSI_Event, IndividualScopeEventMixin):
    """
    This event is triggered if individual in eligible population is selected for screening based using this method.

    CIN HSI is scheduled if individual is diagnosed with CIN2 or CIN3
    Biopsy HSI is scheuduled if individual is believed to have severe cervical dysplasia (stage 1 to 4) based on
    observation of lesions in screening

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Screening_VIA"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.add_equipment({'Cusco’s/ bivalved Speculum (small, medium, large)'})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Check consumables are available
        cons_avail = self.get_consumables(
              item_codes=self.module.item_codes_cervical_can['cervical_cancer_screening_via'],
              optional_item_codes=self.module.item_codes_cervical_can['cervical_cancer_screening_via_optional'])

        if cons_avail:

            # Run a test to diagnose whether the person has condition:
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='screening_with_via_for_cin_and_cervical_cancer',
                hsi_event=self
            )

            # Update properties
            df.loc[person_id,
            ["ce_date_last_screened", "ce_date_via", "ce_ever_screened"]] = [self.sim.date, self.sim.date, True]

            if dx_result:
                df.at[person_id, 'ce_via_cin_ever_detected'] = True

                current_status = df.at[person_id, 'ce_hpv_cc_status']
                # CIN removal if suspected CIN2 or CIN3
                if current_status in ['cin2', 'cin3']:
                    self.module.choose_cin_procedure_and_schedule_it(person_id)

                # Biopsy if suspected Stage 1 to Stage 4
                elif current_status in ['stage1', 'stage2a', 'stage2b', 'stage3', 'stage4']:
                    hs.schedule_hsi_event(
                        hsi_event=HSI_CervicalCancer_Biopsy(
                            module=self.module,
                            person_id=person_id
                        ),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                )

class HSI_CervicalCancer_Screening_Xpert(HSI_Event, IndividualScopeEventMixin):
    """
    This event is triggered if individual in eligible population is selected for screening based on xpert screening
    probability

     Care recommendation depends on HIV status.
     If individual does not have HIV, proceed to VIA screening for confirmation.
     If individual has HIV, then send to CIN treatment regardless of severity (stage of cancer is not as readily
     detectable in xpert screening, so this step is required). In the CIN treatment appointment, if it is deemed to be
     severe, then biopsy will occur at this point.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Screening_Xpert"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.add_equipment({'Cusco’s/ bivalved Speculum (small, medium, large)', 'Conventional PCR Equipment set'})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Check consumables are available
        cons_avail = self.get_consumables(
            item_codes=self.module.item_codes_cervical_can['cervical_cancer_screening_xpert'],
            optional_item_codes = self.module.item_codes_cervical_can['cervical_cancer_screening_xpert_optional'])

        if cons_avail:

            # Run a test to diagnose whether the person has condition:
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='screening_with_xpert_for_cin_and_cervical_cancer',
                hsi_event=self
            )

            # Update properties
            df.loc[person_id, [
                "ce_date_last_screened", "ce_date_xpert", "ce_ever_screened"]] = [self.sim.date, self.sim.date, True]

            if dx_result:
                df.at[person_id, 'ce_xpert_hpv_ever_pos'] = True

                # If HIV negative, do VIA to confirm diagnosis and next steps
                if not df.loc[person_id, 'hv_diagnosed']:
                    if dx_result and (df.at[person_id, 'ce_hpv_cc_status'] != 'none'):
                        hs.schedule_hsi_event(
                            hsi_event=HSI_CervicalCancer_Screening_VIA(
                                module=self.module,
                                person_id=person_id
                                   ),
                            priority=0,
                            topen=self.sim.date,
                            tclose=None
                                   )

                # IF HIV positive, send for CIN treatment; Biopsy will occur within CIN treatment if required based on
                # severity of cancer
                else:
                    self.module.choose_cin_procedure_and_schedule_it(person_id)

class HSI_CervicalCancer_Presentation_WithVaginalBleeding(HSI_Event, IndividualScopeEventMixin):
    """
    This event is triggered if individual presents symptom of vaginal bleeding

    Patient is sent for follow-up biopsy based on prob_referral_biopsy_given_vaginal_bleeding
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Presentation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        hs = self.sim.modules["HealthSystem"]
        m = self.module
        p = m.parameters

        rng = m.rng
        random_value = rng.random()

        if random_value <= p['prob_referral_biopsy_given_vaginal_bleeding']:
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_Biopsy(
                    module=m,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )

class HSI_CervicalCancer_CINRemoval_Cryotherapy(HSI_Event, IndividualScopeEventMixin):
    """
    This event is triggered if individual requires CIN Treatment.

    Success of treatment is defined by indivdidual's ce_hpv_cc_status and prob_cryotherapy_successful
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_CINRemoval_Cryotherapy"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.add_equipment('Cryotherapy unit')  # Essential Equipment

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        m = self.module
        p = m.parameters
        rng = m.rng

        cons_avail = self.get_consumables(
            item_codes=self.module.item_codes_cervical_can['cervical_cancer_cryotherapy'],
            optional_item_codes= self.module.item_codes_cervical_can['cervical_cancer_cryotherapy_optional'])

        if cons_avail:
            # Record date and stage of starting treatment
            df.at[person_id, "ce_date_cryotherapy"] = self.sim.date

            # If individual has CIN, there is a chance of prob_cryotherapy_successful that CIN treatment is successful
            if df.at[person_id, "ce_hpv_cc_status"] in self.module.hpv_cin_options:
                if rng.random_sample() <= p['prob_cryotherapy_successful']:
                    df.at[person_id, "ce_date_cin_removal"] = self.sim.date
                    df.at[person_id, "ce_hpv_cc_status"] = 'none'

            # If individual has ce_hpv_cc_status stage1+, CIN treatment cannot be successful and individual will be
            # sent for biopsy if biopsy has not been performed previously
            elif (
                (df.at[person_id, "ce_hpv_cc_status"] in self.module.hpv_stage_options)
                & (not df.at[person_id, "ce_biopsy"])
            ):
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_Biopsy(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

class HSI_CervicalCancer_CINRemoval_Thermoablation(HSI_Event, IndividualScopeEventMixin):
    """
    This event is triggered if individual requires CIN Treatment and year is 2024 or after

    Success of treatment is defined by indivdidual's ce_hpv_cc_status and prob_thermoabl_successful
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = "CervicalCancer_CINRemoval_Thermoablation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.add_equipment({'LLETZ Machines',
                            'Cusco’s/ bivalved Speculum (small, medium, large)'})  # Essential Equipment

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters
        rng = self.module.rng

        # Check consumables are available
        cons_avail = self.get_consumables(
            optional_item_codes=self.module.item_codes_cervical_can['cervical_cancer_thermoablation_optional'])

        if cons_avail:

            # Record date and stage of starting treatment
            df.at[person_id, "ce_date_thermoabl"] = self.sim.date

            # If individual has CIN, there is a chance of prob_thermoabl_successful that CIN treatment is successful
            if df.at[person_id, "ce_hpv_cc_status"] in self.module.hpv_cin_options:
                if rng.random_sample() <= p['prob_thermoabl_successful']:
                    df.at[person_id, "ce_date_cin_removal"] = self.sim.date
                    df.at[person_id, "ce_hpv_cc_status"] = 'none'

            # If individual has ce_hpv_cc_status stage1+, CIN treatment cannot be successful and individual will be
            # sent for biopsy if biopsy has not been performed previously
            elif (
                df.at[person_id, "ce_hpv_cc_status"] in self.module.hpv_stage_options
                and (not df.at[person_id, "ce_biopsy"])
            ):
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_Biopsy(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

class HSI_CervicalCancer_Biopsy(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_Screening_*, HSI_CervicalCancer_Presentation_WithVaginalBleeding,
    HSI_CervicalCancer_CIN_Cryotherapy, or HSI_CervicalCancer_CIN_Thermoablation

    This event begins the investigation that may result in diagnosis of cervical cancer and the scheduling of
    palliative care if diagnosis is stage 4
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Biopsy"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.add_equipment({'Ultrasound scanning machine',
                            'Ordinary Microscope',
                            'Cusco’s/ bivalved Speculum (small, medium, large)'})  # Essential equipment

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        cons_avail = self.get_consumables(
            item_codes=self.module.item_codes_cervical_can['screening_biopsy_core'],
            optional_item_codes=self.module.item_codes_cervical_can['screening_biopsy_endoscopy_cystoscopy_optional'])

        if cons_avail:
            # Use a biopsy to diagnose whether the person has cervical cancer
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='biopsy_for_cervical_cancer',
                hsi_event=self
            )
            df.at[person_id, "ce_biopsy"] = True

            # If biopsy confirms that individual does not have cervical cancer but CIN is detected, then individual is
            # sent for CIN treatment
            if (not dx_result) and (df.at[person_id, 'ce_hpv_cc_status'] in self.module.hpv_cin_options):
                self.module.choose_cin_procedure_and_schedule_it(person_id)

            # If biopsy confirms that individual has cervical cancer, register diagnosis and either refer to treatment
            # or palliative care
            elif dx_result and (df.at[person_id, 'ce_hpv_cc_status'] in self.module.hpv_stage_options):
                # Record date of diagnosis:
                df.at[person_id, 'ce_date_diagnosis'] = self.sim.date
                df.at[person_id, 'ce_stage_at_diagnosis'] = df.at[person_id, 'ce_hpv_cc_status']
                df.at[person_id, 'ce_current_cc_diagnosed'] = True
                df.at[person_id, 'ce_ever_diagnosed'] = True

                # Check if is in stage4:
                in_stage4 = df.at[person_id, 'ce_hpv_cc_status'] == 'stage4'
                # Assume accurate detection of staege4
                if not in_stage4:
                    # start treatment:
                    hs.schedule_hsi_event(
                        hsi_event=HSI_CervicalCancer_Treatment_Start(
                            module=self.module,
                            person_id=person_id
                        ),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )
                else:
                    # person is in_stage4:
                    # start palliative care:
                    hs.schedule_hsi_event(
                        hsi_event=HSI_CervicalCancer_PalliativeCare(
                            module=self.module,
                            person_id=person_id
                        ),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )

class HSI_CervicalCancer_Treatment_Start(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_Biopsy following a diagnosis of
    cervical Cancer. It initiates the treatment of cervical Cancer.
    It is only for persons with a cancer that is not in stage4 and who have been diagnosed.
    Successful treatment, put people back in 'hpv' stage (from which a progression is not possible)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Treatment_Start"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MajorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters
        rng = self.module.rng
        self.add_equipment(self.healthcare_system.equipment.from_pkg_names('Major Surgery'))

        # If the status is already in `stage4`, start palliative care (instead of treatment)
        if df.at[person_id, "ce_hpv_cc_status"] == 'stage4':
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(
                     module=self.module,
                     person_id=person_id,
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )
            return self.make_appt_footprint({})

        # Check that the person starting treatment has diagnosis date
        if pd.isnull(df.at[person_id, "ce_date_diagnosis"]):
            logger.warning(key="warning", data="Person treated for cervical cancer does not have diagnosis date")

        # Check that consumables are available
        cons_available = self.get_consumables(
            item_codes=self.module.item_codes_cervical_can['treatment_surgery_core'],
            optional_item_codes=self.module.item_codes_cervical_can['treatment_surgery_optional'],
        )

        if cons_available:
            # If consumables are available and the treatment will go ahead - add the used equipment

            # Log the use of adjuvant chemotherapy: try cisplatin first, if not available try fluorouracil
            # Currently just documenting chemo consumbale, treatment not dependent on availability
            chemo_cons_available = self.get_consumables(
                item_codes=self.module.item_codes_cervical_can['cervical_cancer_treatment_chemotherapy_cisplatin'],
                optional_item_codes=self.module.item_codes_cervical_can['iv_drug_cons']
            )

            if not chemo_cons_available:
                # attempt to get second_lines (if they are not available, there will be no treatment
                chemo_cons_available_second_line = self.get_consumables(
                    item_codes=self.module.item_codes_cervical_can[
                        'cervical_cancer_treatment_chemotherapy_fluorouracil'],
                    optional_item_codes=self.module.item_codes_cervical_can['iv_drug_cons']
                )
                if not chemo_cons_available_second_line:
                    return

            # Record date and stage of starting treatment
            df.at[person_id, "ce_date_treatment"] = self.sim.date
            df.at[person_id, "ce_ever_treated"] = True
            df.at[person_id, "ce_stage_at_which_treatment_given"] = df.at[person_id, "ce_hpv_cc_status"]

            # stop vaginal bleeding
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string='vaginal_bleeding',
                add_or_remove='-',
                disease_module=self.module
                )

            # cure individual based on corresponding probabilities
            current_stage = df.at[person_id, "ce_hpv_cc_status"]
            if current_stage == "stage1":
                prob_cure_at_this_stage = p['prob_cure_stage1']
            elif current_stage == "stage2a":
                prob_cure_at_this_stage = p['prob_cure_stage2a']
            elif current_stage == "stage2b":
                prob_cure_at_this_stage = p['prob_cure_stage2b']
            elif current_stage == "stage3":
                prob_cure_at_this_stage = p['prob_cure_stage3']
            else:
                prob_cure_at_this_stage = 1.0

            treatment_will_be_successful = rng.random() <= prob_cure_at_this_stage

            if treatment_will_be_successful:
                # Reset properties for this person at this episode is, in effect, over.
                df.at[person_id, "ce_hpv_cc_status"] = 'hpv'
                                    # <-- resets to 'hpv' stage (and no re-progression possible; see linear_models)
                df.at[person_id, 'ce_current_cc_diagnosed'] = False
                df.at[person_id, 'ce_cured_date_cc'] = self.sim.date

            # Schedule a post-treatment check for 3 months:
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_Treatment_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id,
                ),
                topen=self.sim.date + DateOffset(months=3),
                tclose=None,
                priority=0
            )

class HSI_CervicalCancer_Treatment_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for cervical Cancer.
    If the person has developed cancer to stage4, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Treatment_PostTreatmentCheck"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if df.at[person_id, 'ce_hpv_cc_status'] == 'stage4':
            # If person has progressed to stage4, then start Palliative Care immediately:
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(
                    module=self.module,
                    person_id=person_id
                ),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        else:
            months_since_treatment = int(
                (self.sim.date - df.at[person_id, 'ce_date_treatment']) // pd.Timedelta(days=30))

            if months_since_treatment < 12:
                months_to_next_appt = 3
            elif 12 <= months_since_treatment < 36:
                months_to_next_appt = 6
            elif 36 <= months_since_treatment < 60:
                months_to_next_appt = 12
            else:
                return  # no more follow-up appointments

            hs.schedule_hsi_event(
                self,
                topen=self.sim.date + DateOffset(months=months_to_next_appt),
                tclose=None,
                priority=0
            )


class HSI_CervicalCancer_PalliativeCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is the event for palliative care. It does not affect the patients progress but does affect the disability
     weight and takes resources from the healthsystem.
    This event is scheduled by either:
    * HSI_CervicalCancer_Biopsy following a diagnosis of cervical Cancer at stage4.
    * HSI_CervicalCancer_PostTreatmentCheck following progression to stage4 during treatment.
    * Itself for the continuance of care.
    It is only for persons with a cancer in stage4.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        self.TREATMENT_ID = "CervicalCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': int(p['palliative_care_bed_days'])})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        cons_item_codes = self.module.item_codes_cervical_can['palliation']

        # Check that the person is in stage4
        if not (df.at[person_id, "ce_hpv_cc_status"] == 'stage4'):
            logger.warning(key="warning", data="Person with palliative care not in stage 4.")

        # Check consumables are available
        cons_available = self.get_consumables(item_codes=cons_item_codes)

        if cons_available:
            # If consumables are available and the treatment will go ahead - add the used equipment
            self.add_equipment({'Infusion pump', 'Drip stand'})

            # Record the start of palliative care if this is first appointment
            if pd.isnull(df.at[person_id, "ce_date_palliative_care"]):
                df.at[person_id, "ce_date_palliative_care"] = self.sim.date

            # Schedule another instance of the event for one month
            hs.schedule_hsi_event(
                self,
                topen=self.sim.date + DateOffset(months=1),
                tclose=None,
                priority=0
            )

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class CervicalCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    def __init__(self, module):
        """schedule logging to repeat every 1 month
        """
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Compute statistics regarding the current status of persons and output to the logger
        """
        df = population.props
        p = self.module.parameters

        # CURRENT STATUS COUNTS
        # Create dictionary for each subset, adding prefix to key name, and adding to make a flat dict for logging.
        out = {}
        # Current counts, total
        df_alive_females = df.loc[df.is_alive & (df['sex'] == 'F') & (df['age_years'] > 15)]

        out.update({
            f'total_{k}': v for k, v in df_alive_females.ce_hpv_cc_status.value_counts().items()})

        # Current counts, total hiv negative
        out.update({
            f'total_hivneg_{k}': v for k, v
            in df_alive_females.loc[~df_alive_females['hv_inf']].ce_hpv_cc_status.value_counts().items()})

        # Current counts, total hiv positive
        out.update({
            f'total_hivpos_{k}': v for k, v
            in df_alive_females.loc[df_alive_females['hv_inf']].ce_hpv_cc_status.value_counts().items()})

        out.update({
            'total_males': len(df[df.is_alive & (df['sex'] == 'M')])})
        out.update({
            'total_dead': (~df['is_alive']).sum()})
        out.update({
            'total_overall': len(df)})

        # Get the day of the year
        day_of_year = self.sim.date.timetuple().tm_yday

        # Calculate the decimal year
        decimal_year = self.sim.date.year + (day_of_year - 1) / 365.25
        rounded_decimal_year = round(decimal_year, 2)
        date_1_year_ago = self.sim.date - pd.DateOffset(years=1)
        n_deaths_past_year = (
            df.date_of_death.between(date_1_year_ago, self.sim.date) &
            (df.cause_of_death == "CervicalCancer")
        ).sum()

        n_deaths_cc_hivneg_past_year = (
            (~df['hv_inf']) &
            df.date_of_death.between(date_1_year_ago, self.sim.date) &
            (df.cause_of_death == "CervicalCancer")
        ).sum()

        n_deaths_cc_hivpos_past_year = (
            (df['hv_inf']) &
            df.date_of_death.between(date_1_year_ago, self.sim.date) &
            (df.cause_of_death == "CervicalCancer")
        ).sum()
        n_treated_past_year = df.ce_date_treatment.between(date_1_year_ago, self.sim.date).sum()
        n_cured_past_year = df.ce_cured_date_cc.between(date_1_year_ago, self.sim.date).sum()
        n_thermoabl_past_year = df.ce_date_thermoabl.between(date_1_year_ago, self.sim.date).sum()
        n_cryotherapy_past_year = df.ce_date_cryotherapy.between(date_1_year_ago, self.sim.date).sum()
        n_via_past_year = df.ce_date_via.between(date_1_year_ago, self.sim.date).sum()
        n_xpert_past_year = df.ce_date_xpert.between(date_1_year_ago, self.sim.date).sum()


        date_1p25_years_ago = self.sim.date - pd.DateOffset(days=456)
        date_0p75_years_ago = self.sim.date - pd.DateOffset(days=274)

        cc = (df.is_alive & ((df.ce_hpv_cc_status == 'stage1') | (df.ce_hpv_cc_status == 'stage2a')
                             | (df.ce_hpv_cc_status == 'stage2b') | (df.ce_hpv_cc_status == 'stage3')
                             | (df.ce_hpv_cc_status == 'stage4'))).sum()
        cc_hiv = (df.is_alive & df.hv_inf & ((df.ce_hpv_cc_status == 'stage1') | (df.ce_hpv_cc_status == 'stage2a')
                             | (df.ce_hpv_cc_status == 'stage2b') | (df.ce_hpv_cc_status == 'stage3')
                             | (df.ce_hpv_cc_status == 'stage4'))).sum()
        if cc > 0:
            prop_cc_hiv = cc_hiv / cc
        else:
            prop_cc_hiv = np.nan


        screened_last_month = self.module.polling_event.selected_for_screening_last_month
        n_screened_via_this_month = screened_last_month.get("VIA", 0)
        n_screened_xpert_this_month = screened_last_month.get("Xpert", 0)

        n_ever_screened = (
            (df['is_alive']) &
            (df['ce_ever_screened']) &
            (
                (
                    (df['age_years'] > p['screening_min_age_hv_neg']) &
                    (df['age_years'] < p['screening_max_age_hv_neg']) &
                    (~df['hv_diagnosed'])
                ) |
                (
                    (df['age_years'] > p['screening_min_age_hv_pos']) &
                    (df['age_years'] < p['screening_max_age_hv_pos']) &
                    (~df['hv_diagnosed'])
                )
            )
        ).sum()

        n_vaginal_bleeding_stage1 = (df.is_alive & (df.sy_vaginal_bleeding == 2) &
                                     (df.ce_hpv_cc_status == 'stage1')).sum()
        n_vaginal_bleeding_stage2a = (df.is_alive & (df.sy_vaginal_bleeding == 2) &
                                     (df.ce_hpv_cc_status == 'stage2a')).sum()
        n_vaginal_bleeding_stage2b = (df.is_alive & (df.sy_vaginal_bleeding == 2) &
                                     (df.ce_hpv_cc_status == 'stage2b')).sum()
        n_vaginal_bleeding_stage3 = (df.is_alive & (df.sy_vaginal_bleeding == 2) &
                                     (df.ce_hpv_cc_status == 'stage3')).sum()
        n_vaginal_bleeding_stage4 = (df.is_alive & (df.sy_vaginal_bleeding == 2) &
                                     (df.ce_hpv_cc_status == 'stage4')).sum()

        n_diagnosed_1_year_ago = df.ce_date_diagnosis.between(date_1p25_years_ago, date_0p75_years_ago).sum()
        n_diagnosed_1_year_ago_died = (df.ce_date_diagnosis.between(date_1p25_years_ago, date_0p75_years_ago)
                                       & ~df.is_alive).sum()

        n_diagnosed_past_year_stage1 = \
            (df.ce_date_diagnosis.between(date_1_year_ago, self.sim.date) &
             (df.ce_stage_at_diagnosis == 'stage1')).sum()
        n_diagnosed_past_year_stage2a = \
            (df.ce_date_diagnosis.between(date_1_year_ago, self.sim.date) &
             (df.ce_stage_at_diagnosis == 'stage2a')).sum()
        n_diagnosed_past_year_stage2b = \
            (df.ce_date_diagnosis.between(date_1_year_ago, self.sim.date) &
             (df.ce_stage_at_diagnosis == 'stage2b')).sum()
        n_diagnosed_past_year_stage3 = \
            (df.ce_date_diagnosis.between(date_1_year_ago, self.sim.date) &
             (df.ce_stage_at_diagnosis == 'stage3')).sum()
        n_diagnosed_past_year_stage4 = \
            (df.ce_date_diagnosis.between(date_1_year_ago, self.sim.date) &
             (df.ce_stage_at_diagnosis == 'stage4')).sum()

        n_diagnosed_past_year = (df['ce_date_diagnosis'].between(date_1_year_ago, self.sim.date)).sum()

        n_ever_diagnosed = ((df['is_alive']) & (df['ce_ever_diagnosed'])).sum()

        n_women_alive = ((df['is_alive']) & (df['sex'] == 'F')).sum()
        n_women_alive_1549 = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] >= 15)
                              & (df['age_years'] < 50)).sum()

        n_women_vaccinated = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] >= 15)
                              & df['va_hpv']).sum()

        n_women_hiv_unsuppressed = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] >= 15)
                                    & (((df['hv_art'] == 'on_not_vl_suppressed') | (df['hv_art'] == 'not'))
                                       & (df['hv_inf']))).sum()

        n_via_cin_ever_detected = ((df['is_alive']) & (df['ce_via_cin_ever_detected'])).sum()

        n_women_hivneg = ((df['is_alive']) &
                          (df['sex'] == 'F') &
                          (df['age_years'] > 15) &
                          (~df['hv_inf'])).sum()

        n_women_hivpos = ((df['is_alive']) &
                          (df['sex'] == 'F') &
                          (df['age_years'] > 15) &
                          (df['hv_inf'])).sum()

        rate_diagnosed_cc = n_diagnosed_past_year / n_women_alive

        n_women_living_with_diagnosed_cc = \
            (df['ce_date_diagnosis'].notnull()).sum()

        n_women_living_with_diagnosed_cc_age_lt_30 = \
            (df['ce_date_diagnosis'].notnull() & (df['age_years'] < 30)).sum()
        n_women_living_with_diagnosed_cc_age_3050 = \
            (df['ce_date_diagnosis'].notnull() & (df['age_years'] > 29) & (df['age_years'] < 50)).sum()
        n_women_living_with_diagnosed_cc_age_gt_50 = \
            (df['ce_date_diagnosis'].notnull() & (df['age_years'] > 49)).sum()

        out.update({"rounded_decimal_year": rounded_decimal_year})
        out.update({"n_deaths_past_year": n_deaths_past_year})
        out.update({"n_deaths_cc_hivneg_past_year": n_deaths_cc_hivneg_past_year})
        out.update({"n_deaths_cc_hivpos_past_year": n_deaths_cc_hivpos_past_year})
        out.update({"n_treated_past_year": n_treated_past_year})
        out.update({"n_cured_past_year": n_cured_past_year})
        out.update({"prop_cc_hiv": prop_cc_hiv})
        out.update({"n_diagnosed_past_year_stage1": n_diagnosed_past_year_stage1})
        out.update({"n_diagnosed_past_year_stage2a": n_diagnosed_past_year_stage2a})
        out.update({"n_diagnosed_past_year_stage2b": n_diagnosed_past_year_stage2b})
        out.update({"n_diagnosed_past_year_stage3": n_diagnosed_past_year_stage3})
        out.update({"n_diagnosed_past_year_stage4": n_diagnosed_past_year_stage4})
        out.update({"n_ever_diagnosed": n_ever_diagnosed})
        out.update({"n_screened_xpert_this_month": n_screened_xpert_this_month})
        out.update({"n_screened_via_this_month": n_screened_via_this_month})
        out.update({"n_women_alive": n_women_alive})
        out.update({"n_women_alive_1549": n_women_alive_1549})
        out.update({"n_ever_screened": n_ever_screened})
        out.update({"n_women_vaccinated": n_women_vaccinated})
        out.update({"n_vaginal_bleeding_stage1": n_vaginal_bleeding_stage1})
        out.update({"n_vaginal_bleeding_stage2a": n_vaginal_bleeding_stage2a})
        out.update({"n_vaginal_bleeding_stage2b": n_vaginal_bleeding_stage2b})
        out.update({"n_vaginal_bleeding_stage3": n_vaginal_bleeding_stage3})
        out.update({"n_vaginal_bleeding_stage4": n_vaginal_bleeding_stage4})
        out.update({"n_diagnosed_past_year": n_diagnosed_past_year})
        out.update({"n_women_alive": n_women_alive})
        out.update({"rate_diagnosed_cc": rate_diagnosed_cc})
        out.update({"cc": cc})
        out.update({"n_women_living_with_diagnosed_cc": n_women_living_with_diagnosed_cc })
        out.update({"n_women_living_with_diagnosed_cc_age_lt_30": n_women_living_with_diagnosed_cc_age_lt_30})
        out.update({"n_women_living_with_diagnosed_cc_age_3050": n_women_living_with_diagnosed_cc_age_3050})
        out.update({"n_women_living_with_diagnosed_cc_age_gt_50": n_women_living_with_diagnosed_cc_age_gt_50})
        out.update({"n_diagnosed_1_year_ago": n_diagnosed_1_year_ago})
        out.update({"n_diagnosed_1_year_ago_died": n_diagnosed_1_year_ago_died})
        out.update({"n_women_hiv_unsuppressed": n_women_hiv_unsuppressed})
        out.update({"n_women_hivneg": n_women_hivneg})
        out.update({"n_women_hivpos": n_women_hivpos})
        out.update({"n_thermoabl_past_year": n_thermoabl_past_year})
        out.update({"n_cryotherapy_past_year": n_cryotherapy_past_year})
        out.update({"n_via_past_year": n_via_past_year})
        out.update({"n_xpert_past_year": n_xpert_past_year})
        out.update({"n_via_cin_ever_detected": n_via_cin_ever_detected})

        pop = len(df.is_alive)
        count_summary = {
            "population": pop,
            "n_deaths_past_year": n_deaths_past_year,
            "n_women_alive": n_women_alive,
            "n_women_living_with_diagnosed_cc": n_women_living_with_diagnosed_cc,
        }

        logger.info(key="deaths",
                    data=count_summary,
                    description="summary of deaths")

        logger.info(key="all",
                    data=out,
                    description="all_data")
