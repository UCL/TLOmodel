
"""
Cervical Cancer Disease Module

Limitations to note:
* Footprints of HSI -- pending input from expert on resources required.
at some point we may need to specify the treatment eg total hysterectomy plus or minus chemotherapy
but we agree not now
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

import math
from typing import TYPE_CHECKING, List

import pandas as pd
import json
import numpy as np
import csv

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.methods import Metadata
from tlo.methods.cancer_consumables import get_consumable_item_codes_cancers

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
    from tlo.population import IndividualProperties

from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin

# Set parameters
screening_min_age = 25
screening_max_age = 50
screening_min_age_hv_neg = 30
screening_max_age_hv_neg = 50
screening_min_age_hv_pos = 25
screening_max_age_hv_pos = 50
yrs_between_screen_hv_pos = 3
yrs_between_screen_hv_neg = 5


hpv_cin_options = ['hpv', 'cin1', 'cin2', 'cin3']
hpv_stage_options = ['stage1', 'stage2a', 'stage2b', 'stage3', 'stage4']

def screen_subset_population(year, p, eligible_population, df, rng, sim, module):
    screening_methods = {
        'VIA': {
            'prob_key': 'prob_via_screen',
            'event_class': HSI_CervicalCancer_AceticAcidScreening,
            'selected_column': 'ce_selected_for_via_this_month'
        },
        'Xpert': {
            'prob_key': 'prob_xpert_screen',
            'event_class': HSI_CervicalCancer_XpertHPVScreening,
            'selected_column': 'ce_selected_for_xpert_this_month'
        }
    }
    selected_method = 'VIA' if year <= p['transition_screening_year'] else 'Xpert'
    method_info = screening_methods[selected_method]

    # Randomly select for screening
    df.loc[eligible_population, method_info['selected_column']] = (
        rng.random(size=len(df[eligible_population])) < p[method_info['prob_key']]
    )

    # Schedule HSI events
    for idx in df.index[df[method_info['selected_column']]]:
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=method_info['event_class'](module=module, person_id=idx),
            priority=0,
            topen=sim.date,
            tclose=None
        )
def schedule_cin_procedure(year, p, person_id, hs, module, sim):
    treatment_methods = {
        'Thermoablation': {
            'event_class': HSI_CervicalCancer_Thermoablation_CIN
        },
        'Cryotherapy': {
            'event_class': HSI_CervicalCancer_Cryotherapy_CIN
        }
    }

    selected_method = 'Thermoablation' if year >= p['transition_testing_year'] else 'Cryotherapy'
    method_info = treatment_methods[selected_method]

    # Schedule HSI event
    hs.schedule_hsi_event(
        hsi_event=method_info['event_class'](module=module, person_id=person_id),
        priority=0,
        topen=sim.date,
        tclose=None
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CervicalCancer(Module, GenericFirstAppointmentsMixin):
    """Cervical Cancer Disease Module"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.linear_models_for_progression_of_hpv_cc_status = dict()
        self.lm_onset_vaginal_bleeding = None
        self.daly_wts = dict()
        self.item_codes_cervical_can = dict()

    INIT_DEPENDENCIES = {
        'Demography', 'SimplifiedBirths', 'HealthSystem', 'Lifestyle', 'SymptomManager'
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden', 'HealthSeekingBehaviour'}

    ADDITIONAL_DEPENDENCIES = {'Tb', 'Hiv'}

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
            Types.REAL, "rate ratio for progressing through cin and cervical cancer stages if have unsuppressed hiv"
        ),
        "rr_hpv_vaccinated": Parameter(
            Types.REAL,
            "rate ratio for hpv if vaccinated - this is combined effect of probability the hpv is "
            "vaccine-preventable and vaccine efficacy against vaccine-preventable hpv ",
        ),
        "rr_hpv_age50plus": Parameter(
            Types.REAL,
            "rate ratio for hpv if age 50 plus"
        ),
        "prob_cure_stage1": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 1 cervical cancer",
        ),
        "prob_cure_stage2a": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 1 cervical cancer",
        ),
        "prob_cure_stage2b": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 1 cervical cancer",
        ),
        "prob_cure_stage3": Parameter(
            Types.REAL,
            "probability of cure if treated in stage 1 cervical cancer",
        ),
        "r_death_cervical_cancer": Parameter(
            Types.REAL,
            "probability per month of death from cervical cancer amongst people with stage 4 cervical cancer",
        ),
        "r_vaginal_bleeding_cc_stage1": Parameter(
            Types.REAL, "rate of vaginal bleeding if have stage 1 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2a": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 2a cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage2b": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 2b cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage3": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 3 cervical cancer"
        ),
        "rr_vaginal_bleeding_cc_stage4": Parameter(
            Types.REAL, "rate ratio for vaginal bleeding if have stage 4 cervical cancer"
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
            Types.REAL, "sensitivity of via for cin and cervical cancer bu stage"
        ),
        "prob_xpert_screen": Parameter(
            Types.REAL, "prob_xpert_screen"
        ),
        "prob_via_screen": Parameter(
            Types.REAL, "prob_via_screen"
        ),
        "prob_thermoabl_successful": Parameter(
            Types.REAL, "prob_thermoabl_successful"
        ),
        "prob_cryotherapy_successful": Parameter(
            Types.REAL, "prob_cryotherapy_successful"
        ),
        "transition_testing_year": Parameter(
            Types.REAL, "transition_testing_year"
        ),
        "transition_screening_year": Parameter(
            Types.REAL, "transition_screening_year"
        )
    }

    """
    note: hpv vaccination is in epi.py
    """

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
            "the date of last cin removal (pd.NaT if never diagnosed)"
        ),
        "ce_date_treatment": Property(
            Types.DATE,
            "date of first receiving attempted curative treatment (pd.NaT if never started treatment)"
        ),
        "ce_ever_screened": Property(
            Types.DATE,
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
            # currently this property has levels to match ce_hov_cc_status to enable the code as written, even
            # though can only be treated when in stage 1-3
        "ce_stage_at_which_treatment_given": Property(
            Types.CATEGORICAL,
            "the cancer stage at which treatment was given (because the treatment only has an effect during the stage"
            "at which it is given).",
            categories=["none", "hpv", "cin1", "cin2", "cin3", "stage1", "stage2a", "stage2b", "stage3", "stage4"],
        ),
        "ce_date_palliative_care": Property(
            Types.DATE,
            "date of first receiving palliative care (pd.NaT is never had palliative care)"
        ),
        "ce_ever_diagnosed": Property(
            Types.DATE,
            "ever diagnosed with cervical cancer (even if now cured)"
        ),
        "ce_date_death": Property(
            Types.DATE,
            "date of cervical cancer death"
        ),
        "ce_new_stage_this_month": Property(
            Types.BOOL,
            "new_stage_this month"
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
        "ce_current_cc_diagnosed": Property(
            Types.BOOL,
            "currently has diagnosed cervical cancer (which until now has not been cured)"
        ),
        "ce_selected_for_via_this_month": Property(
            Types.BOOL,
            "selected for via this period"
        ),
        "ce_selected_for_xpert_this_month": Property(
            Types.BOOL,
            "selected for xpert this month"
        ),
        "ce_biopsy": Property(
            Types.BOOL,
            "ce biopsy done"
        )
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module, now including disability weights"""

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_Cervical_Cancer.xlsx",
                          sheet_name="parameter_values")
        )

        # note that health seeking probability quite high even though or =1
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='vaginal_bleeding',
                    odds_ratio_health_seeking_in_adults=1.00)
        )

        # in order to implement screening for cervical cancer creating a dummy symptom - likely there is a better way
        # self.sim.modules['SymptomManager'].register_symptom(
        #     Symptom(name='chosen_via_screening_for_cin_cervical_cancer',
        #             odds_ratio_health_seeking_in_adults=100.00)
        # )
# todo: in order to implement screening for cervical cancer creating a dummy symptom - likely there is a better way

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='chosen_via_screening_for_cin_cervical_cancer',
                    odds_ratio_health_seeking_in_adults=100.00)
        )

        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='chosen_xpert_screening_for_hpv_cervical_cancer',
                    odds_ratio_health_seeking_in_adults=100.00)
        )


    def initialise_population(self, population):
        """Set property values for the initial population."""
        df = population.props  # a shortcut to the data-frame
        p = self.parameters
        rng = self.rng

        # defaults
        df.loc[df.is_alive, "ce_hpv_cc_status"] = "none"
        df.loc[df.is_alive, "ce_date_diagnosis"] = pd.NaT
        df.loc[df.is_alive, "ce_date_treatment"] = pd.NaT
        df.loc[df.is_alive, "ce_stage_at_which_treatment_given"] = "none"
        df.loc[df.is_alive, "ce_date_palliative_care"] = pd.NaT
        df.loc[df.is_alive, "ce_date_death"] = pd.NaT
        df.loc[df.is_alive, "ce_new_stage_this_month"] = False
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
        df.loc[df.is_alive, "ce_selected_for_via_this_month"] = False
        df.loc[df.is_alive, "ce_selected_for_xpert_this_month"] = False
        df.at[df.is_alive, "days_since_last_via"] = pd.NaT
        df.at[df.is_alive, "days_since_last_xpert"] = pd.NaT
        df.loc[df.is_alive, "ce_biopsy"] = False
        df.loc[df.is_alive, "ce_ever_screened"] = False
        df.loc[df.is_alive, "ce_ever_diagnosed"] = False
        df.loc[df.is_alive, "ce_cured_date_cc"] = pd.NaT
        df.loc[df.is_alive, "ce_date_last_screened"] = pd.NaT

        # -------------------- ce_hpv_cc_status -----------
        # this was not assigned here at outset because baseline value of hv_inf was not accessible - it is assigned
        # st start of main polling event below

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


        # ----- SCHEDULE MAIN POLLING EVENTS -----
        # Schedule main polling event to happen immediately
        sim.schedule_event(CervicalCancerMainPollingEvent(self), sim.date)

        # ----- SCHEDULE LOGGING EVENTS -----
        # Schedule logging event to happen immediately
        sim.schedule_event(CervicalCancerLoggingEvent(self), sim.date + DateOffset(months=1))

        # Look-up consumable item codes
        self.item_codes_cervical_can = get_consumable_item_codes_cancers(self)

        # ----- LINEAR MODELS -----
        # Define LinearModels for the progression of cancer, in each 1 month period
        # NB. The effect being produced is that treatment only has the effect in the stage at which the
        # treatment was received.

        df = sim.population.props
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
            .when('.between(50,110)', p['rr_hpv_age50plus']),
            Predictor('sex').when('M', 0.0),
            Predictor('ce_hpv_cc_status').when('none', 1.0).otherwise(0.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['cin1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin1_hpv'],
            Predictor('ce_hpv_cc_status').when('hpv', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['cin2'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin2_cin1'],
            Predictor('ce_hpv_cc_status').when('cin1', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['cin3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_cin3_cin2'],
            Predictor('ce_hpv_cc_status').when('cin2', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage1'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage1_cin3'],
            Predictor('ce_hpv_cc_status').when('cin3', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage2a'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2a_stage1'],
            Predictor('ce_hpv_cc_status').when('stage1', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage2b'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage2b_stage2a'],
            Predictor('ce_hpv_cc_status').when('stage2a', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage3'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage3_stage2b'],
            Predictor('ce_hpv_cc_status').when('stage2b', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        lm['stage4'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['r_stage4_stage3'],
            Predictor('ce_hpv_cc_status').when('stage3', 1.0).otherwise(0.0),
#           Predictor('hv_inf', conditions_are_mutually_exclusive=True)
#           .when(False, 0.0)
#           .when(True, 1.0),
            Predictor('ce_hiv_unsuppressed').when(True, p['rr_progress_cc_hiv']).otherwise(1.0),
            Predictor('ce_new_stage_this_month').when(True, 0.0).otherwise(1.0)
        )

        # Check that the dict labels are correct as these are used to set the value of ce_hpv_cc_status
        assert set(lm).union({'none'}) == set(df.ce_hpv_cc_status.cat.categories)

        # Linear Model for the onset of vaginal bleeding, in each 1 month period
        # Create variables for used to predict the onset of vaginal bleeding at
        # various stages of the disease

        stage1 = p['r_vaginal_bleeding_cc_stage1']
        stage2a = p['rr_vaginal_bleeding_cc_stage2a'] * p['r_vaginal_bleeding_cc_stage1']
        stage2b = p['rr_vaginal_bleeding_cc_stage2b'] * p['r_vaginal_bleeding_cc_stage1']
        stage3 = p['rr_vaginal_bleeding_cc_stage3'] * p['r_vaginal_bleeding_cc_stage1']
        stage4 = p['rr_vaginal_bleeding_cc_stage4'] * p['r_vaginal_bleeding_cc_stage1']

        self.lm_onset_vaginal_bleeding = LinearModel.multiplicative(
            Predictor('sex').when('M', 0.0),
            Predictor(
                'ce_hpv_cc_status',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('none', 0.00001)
            .when('cin1', 0.00001)
            .when('cin2', 0.00001)
            .when('cin3', 0.00001)
            .when('stage1', stage1)
            .when('stage2a', stage2a)
            .when('stage2b', stage2b)
            .when('stage3', stage3)
            .when('stage4', stage4)
        )

        # ----- DX TESTS -----
        # Create the diagnostic test representing the use of a biopsy
        # This properties of conditional on the test being done only to persons with the Symptom, 'vaginal_bleeding!

        # in future could add different sensitivity according to target category

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            biopsy_for_cervical_cancer=DxTest(
                property='ce_hpv_cc_status',
                sensitivity=self.parameters['sensitivity_of_biopsy_for_cervical_cancer'],
                target_categories=["stage1", "stage2a", "stage2b", "stage3", "stage4"]
            )
        )

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            screening_with_xpert_for_hpv=DxTest(
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
                sequlae_code=607
                # "Diagnosis and primary therapy phase of cervical cancer":
                #  "Cancer, diagnosis and primary therapy ","has pain, nausea, fatigue, weight loss and high anxiety."
            )

            # For those with cancer (any stage prior to stage 4) and has been treated
            self.daly_wts["stage_1_3_treated"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=608
                # "Controlled phase of cervical cancer,Generic uncomplicated disease":
                # "worry and daily medication,has a chronic disease that requires medication every day and causes some
                #   worry but minimal interference with daily activities".
            )

            # For those in stage 4: no palliative care
            self.daly_wts["stage4"] = self.sim.modules["HealthBurden"].get_daly_weight(
                sequlae_code=609
                # "Metastatic phase of cervical cancer:
                # "Cancer, metastatic","has severe pain, extreme fatigue, weight loss and high anxiety."
            )

            # For those in stage 4: with palliative care
            self.daly_wts["stage4_palliative_care"] = self.daly_wts["stage_1_3"]
            # By assumption, we say that that the weight for those in stage 4 with palliative care is the same as
            # that for those with stage 1-3 cancers.

        # ----- HSI FOR PALLIATIVE CARE -----
        on_palliative_care_at_initiation = df.index[df.is_alive & ~pd.isnull(df.ce_date_palliative_care)]
        for person_id in on_palliative_care_at_initiation:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_PalliativeCare(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(weeks=1)
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
        df.at[child_id, "ce_new_stage_this_month"] = False
        df.at[child_id, "ce_date_palliative_care"] = pd.NaT
        df.at[child_id, "ce_date_death"] = pd.NaT
        df.at[child_id, "ce_date_cin_removal"] = pd.NaT
        df.at[child_id, "ce_stage_at_diagnosis"] = 'none'
        df.at[child_id, "ce_ever_treated"] = False
        df.at[child_id, "ce_cc_ever"] = False
        df.at[child_id, "ce_xpert_hpv_ever_pos"] = False
        df.at[child_id, "ce_via_cin_ever_detected"] = False
        df.at[child_id, "ce_date_thermoabl"] = pd.NaT
        df.loc[child_id, "ce_date_cryotherapy"] = pd.NaT
        df.at[child_id, "days_since_last_via"] = pd.NaT
        df.at[child_id, "days_since_last_xpert"] = pd.NaT
        df.at[child_id, "ce_current_cc_diagnosed"] = False
        df.at[child_id, "ce_selected_for_via_this_month"] = False
        df.at[child_id, "ce_selected_for_xpert_this_month"] = False
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
            (
                (df.ce_hpv_cc_status == "stage1") |
                (df.ce_hpv_cc_status == "stage2a") |
                (df.ce_hpv_cc_status == "stage2b") |
                (df.ce_hpv_cc_status == "stage3")
            )
        ] = self.daly_wts['stage_1_3']

        # Assign daly_wt to those with cancer stages before stage4 and who have been treated and who are still in the
        # stage in which they were treated.
        disability_series_for_alive_persons.loc[
            (
                ~pd.isnull(df.ce_date_treatment) & (
                    (df.ce_hpv_cc_status == "stage1") |
                    (df.ce_hpv_cc_status == "stage2a") |
                    (df.ce_hpv_cc_status == "stage2b") |
                    (df.ce_hpv_cc_status == "stage3")
                ) & (df.ce_hpv_cc_status == df.ce_stage_at_which_treatment_given)
            )
        ] = self.daly_wts['stage_1_3_treated']

        # todo: check
        # I'm a bit surprised this works, because the masks being used are wrt to df, but the indexing
        # into a series with a difference index. Maybe it only works as long as everyone is alive!?


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


    def onset_xpert_properties(self, idx: pd.Index):
        """Represents the screened property for the person_id given in `idx`"""
        df = self.sim.population.props
        if df.loc[idx, 'ce_selected_for_xpert_this_month'].any():
            df.loc[idx, 'ce_ever_screened'] = True
        else:
            df.loc[idx, 'ce_ever_screened'] = False

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
                HSI_CervicalCancerPresentationVaginalBleeding(
                    person_id=person_id,
                    module=self
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None)

        # if 'chosen_via_screening_for_cin_cervical_cancer' in symptoms:
        #     schedule_hsi_event(
        #         HSI_CervicalCancer_AceticAcidScreening(
        #             person_id=person_id,
        #             module=self
        #         ),
        #         priority=0,
        #         topen=self.sim.date,
        #         tclose=None)
        #
        # if 'chosen_xpert_screening_for_hpv_cervical_cancer' in symptoms:
        #     schedule_hsi_event(
        #         HSI_CervicalCancer_XpertHPVScreening(
        #             person_id=person_id,
        #             module=self
        #         ),
        #         priority=0,
        #         topen=self.sim.date,
        #         tclose=None)

        # else:
        # schedule_hsi_event(
        #     HSI_CervicalCancer_Screening(
        #         person_id=person_id,
        #         module=self
        #     ),
        #     priority=0,
        #     topen=self.sim.date,
        #     tclose=None)

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

    def apply(self, population):
        df = population.props  # shortcut to dataframe
        year = self.sim.date.year
        m = self.module
        rng = m.rng
        p = self.sim.modules['CervicalCancer'].parameters

        # ------------------- SET INITIAL CE_HPV_CC_STATUS -------------------------------------------------------------------
        # this was done here and not at outset because baseline value of hv_inf was not accessible

        given_date = pd.to_datetime('2010-01-03')

        if self.sim.date < given_date:

            women_over_15_nhiv_idx = df.index[(df["age_years"] > 15) & (df["sex"] == 'F') & ~df["hv_inf"]]

            df.loc[women_over_15_nhiv_idx, 'ce_hpv_cc_status'] = rng.choice(
                ['none', 'hpv', 'cin1', 'cin2', 'cin3', 'stage1', 'stage2a', 'stage2b', 'stage3', 'stage4'],
                size=len(women_over_15_nhiv_idx), p=p['init_prev_cin_hpv_cc_stage_nhiv']
            )

            women_over_15_hiv_idx = df.index[(df["age_years"] > 15) & (df["sex"] == 'F') & df["hv_inf"]]

            df.loc[women_over_15_hiv_idx, 'ce_hpv_cc_status'] = rng.choice(
                ['none', 'hpv', 'cin1', 'cin2', 'cin3', 'stage1', 'stage2a', 'stage2b', 'stage3', 'stage4'],
                size=len(women_over_15_hiv_idx), p=p['init_prev_cin_hpv_cc_stage_hiv']
            )

        # -------------------- ACQUISITION AND PROGRESSION OF CANCER (ce_hpv_cc_status) -----------------------------------

        # todo:
        # this is being broadcast. it should be lmited to those with is_alive: ie. df.loc[df.is_alive,
        # 'cc_new_stage_this_month'] = False
        # As I expect this is going to be over-written (further down) it would be more efiicent to not
        # write it into the main sim.population.props df yet (reading/writing there is time-consuming),
        # and instead do one write to it at the end of the event, when everything is settled.

        df['ce_hiv_unsuppressed'] = ((df['hv_art'] == 'on_not_vl_suppressed') | (df['hv_art'] == 'not')) & (df['hv_inf'])

        # determine if the person had a treatment during this stage of cancer (nb. treatment only has an effect on
        #  reducing progression risk during the stage at which is received.

        for stage, lm in self.module.linear_models_for_progression_of_hpv_cc_status.items():
            gets_new_stage = lm.predict(df.loc[df.is_alive], rng)

            idx_gets_new_stage = gets_new_stage[gets_new_stage].index

#           print(stage, lm, gets_new_stage, idx_gets_new_stage)

            df.loc[idx_gets_new_stage, 'ce_hpv_cc_status'] = stage
            df['ce_new_stage_this_month'] = df.index.isin(idx_gets_new_stage)

        # Identify rows where the status is 'cin1'
        has_cin1 = (
            (df.is_alive) &
            (df.sex == 'F') &
            (df.ce_hpv_cc_status == 'cin1')
        )

        # Apply the reversion probability to change some 'cin1' to 'none'
        df.loc[has_cin1, 'ce_hpv_cc_status'] = np.where(
            self.module.rng.random(size=len(df[has_cin1])) < p['prob_revert_from_cin1'],
            'none',
            df.loc[has_cin1, 'ce_hpv_cc_status']
        )



        # todo:
        # this is also broadcasting to all dataframe (including dead peple and never alive people,
        # potentially).
        #
        # Also, it will over-write to False those people not in any of those categories. I can see
        # that this will not violate the logic, but the safest thing would be to also include in the
        # chanied union statement the current value, in order to absolute prevent reversions... i.e.
        # add in ce_cc_ever on the end of this line.

        df.loc[
            (df['is_alive']) & (~df['ce_cc_ever']),  # Apply only if is_alive is True and ce_cc_ever is not True
            'ce_cc_ever'
        ] = (
            (df['ce_hpv_cc_status'].isin(['stage1', 'stage2a', 'stage2b', 'stage3', 'stage4']))
            | df['ce_ever_treated']
        )

        # -------------------------------- SCREENING FOR CERVICAL CANCER USING XPERT HPV TESTING AND VIA---------------
        # A subset of women aged 30-50 will receive a screening test

        # in future this may be triggered by family planning visit

        # todo:
        # Instead, for the individuals that are chosen to be screened, create and schedule the HSI
        # event directly.
        #
        # e.g. for each individual to be screened... make an HSI_Event_CervicalCancer_Screening.....
        # and in that event, do whatever is required for the screening. (might be the same as happens
        # in the generic appointment, in which case point them both to the same function)


        #todo: create a date of last via screen (and same for xpert) and make it a condition of screening
        # that last screen was x years ago

        df.ce_selected_for_via_this_month = False
        df.ce_selected_for_xpert_this_month = False

        days_since_last_screen = (self.sim.date - df.ce_date_last_screened).dt.days
        days_since_last_thermoabl = (self.sim.date - df.ce_date_thermoabl).dt.days
        days_since_last_via = (self.sim.date - df.ce_date_via).dt.days
        days_since_last_xpert = (self.sim.date - df.ce_date_xpert).dt.days

        # todo: screening probability depends on date last screen and result (who guidelines)

        eligible_population = (
            (df.is_alive) &
            (df.sex == 'F') &
            (df.age_years >= screening_min_age) &
            (df.age_years < screening_max_age) &
            (~df.ce_current_cc_diagnosed) &
            (
                pd.isna(df.ce_date_last_screened) |
                ((days_since_last_via > 1825) & (days_since_last_xpert > 1825)) |
                ((days_since_last_screen > 730) & (days_since_last_thermoabl < 1095))
            )
        )

        # todo: consider fact that who recommend move towards xpert screening away from via
        # todo: start with via as screening tool and move to xpert in about 2024

        m = self.module
        rng = m.rng

        screen_subset_population(year, p, eligible_population, df, rng, self.sim, self.module)

        # xpert_select_ind_id = df.loc[df['ce_selected_for_xpert_this_month']].index
            # self.module.onset_xpert_properties(xpert_select_ind_id)


        # self.sim.modules['SymptomManager'].change_symptom(
        #     person_id=df.loc[df['ce_selected_for_via_this_month']].index,
        #     symptom_string='chosen_via_screening_for_cin_cervical_cancer',
        #     add_or_remove='+',
        #     disease_module=self.module
        # )
        #
        # self.sim.modules['SymptomManager'].change_symptom(
        #     person_id=df.loc[df['ce_selected_for_xpert_this_month']].index,
        #     symptom_string='chosen_xpert_screening_for_hpv_cervical_cancer',
        #     add_or_remove='+',
        #     disease_module=self.module
        # )


    # -------------------- UPDATING OF SYMPTOM OF vaginal bleeding OVER TIME --------------------------------
        # Each time this event is called (every month) individuals with cervical cancer may develop the symptom of
        # vaginal bleeding.  Once the symptom is developed it never resolves naturally. It may trigger
        # health-care-seeking behaviour.
        onset_vaginal_bleeding = self.module.lm_onset_vaginal_bleeding.predict(
            df.loc[
                np.bitwise_and(df.is_alive, df.ce_stage_at_diagnosis == 'none')
            ],
            rng
        )

        self.sim.modules['SymptomManager'].change_symptom(
            person_id=onset_vaginal_bleeding[onset_vaginal_bleeding].index.tolist(),
            symptom_string='vaginal_bleeding',
            add_or_remove='+',
            disease_module=self.module
        )

        # -------------------- DEATH FROM cervical CANCER ---------------------------------------
        # There is a risk of death for those in stage4 only. Death is assumed to go instantly.
        stage4_idx = df.index[df.is_alive & (df.ce_hpv_cc_status == "stage4")]
        selected_to_die = stage4_idx[
            rng.random_sample(size=len(stage4_idx)) < self.module.parameters['r_death_cervical_cancer']]

        for person_id in selected_to_die:
            self.sim.schedule_event(
                InstantaneousDeath(self.module, person_id, "CervicalCancer"), self.sim.date
            )
            days_spread = 90
            date_min = self.sim.date
            date_max = self.sim.date + pd.DateOffset(days=days_spread)
            df.loc[person_id, 'ce_date_death'] = pd.to_datetime(rng.uniform(date_min.value, date_max.value), unit='ns')

    # todo: distribute death dates across next 30 days


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_CervicalCancer_AceticAcidScreening(HSI_Event, IndividualScopeEventMixin):

    """
    This event will be scheduled by family planning HSI - for now we determine at random a screening event,
    and we determine at random whether this is AceticAcidScreening or HPVXpertScreening

    In future this might be scheduled by the contraception module

    may in future want to modify slightly to reflect this: biopsy is taken if via looks abnormal and the facility
    has the capacity to take a biopsy - otherwise thermoablation is performed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_AceticAcidScreening"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        year = self.sim.date.year
        p = self.sim.modules['CervicalCancer'].parameters
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        # Check consumables are available
        cons_avail = self.get_consumables(
            item_codes=self.module.item_codes_cervical_can['cervical_cancer_screening_via'])

        if cons_avail:
            self.add_equipment({'Infusion pump', 'Drip stand'})
            # self.add_equipment(self.healthcare_system.equipment.from_pkg_names('Major Surgery'))

            # Run a test to diagnose whether the person has condition:
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='screening_with_via_for_cin_and_cervical_cancer',
                hsi_event=self
            )
            df.at[person_id, "ce_date_last_screened"] = self.sim.date
            df.at[person_id, "ce_date_via"] = self.sim.date
            df.at[person_id, "ce_ever_screened"] = True

            if dx_result:
                df.at[person_id, 'ce_via_cin_ever_detected'] = True

                if (df.at[person_id, 'ce_hpv_cc_status'] == 'cin2'
                            or df.at[person_id, 'ce_hpv_cc_status'] == 'cin3'
                            ):
                    schedule_cin_procedure(year, p, person_id, self.sim.modules['HealthSystem'], self.module, self.sim)

                elif (df.at[person_id, 'ce_hpv_cc_status'] == 'stage1'
                            or df.at[person_id, 'ce_hpv_cc_status'] == 'stage2a'
                            or df.at[person_id, 'ce_hpv_cc_status'] == 'stage2b'
                            or df.at[person_id, 'ce_hpv_cc_status'] == 'stage3'
                            or df.at[person_id, 'ce_hpv_cc_status'] == 'stage4'):
                    hs.schedule_hsi_event(
                        hsi_event=HSI_CervicalCancer_Biopsy(
                            module=self.module,
                            person_id=person_id
                        ),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                )

        # sy_chosen_via_screening_for_cin_cervical_cancer reset to 0
        # if df.at[person_id, 'sy_chosen_via_screening_for_cin_cervical_cancer'] == 2:
        #     self.sim.modules['SymptomManager'].change_symptom(
        #         person_id=person_id,
        #         symptom_string='chosen_via_screening_for_cin_cervical_cancer',
        #         add_or_remove='-',
        #         disease_module=self.module
        #         )
        #
        # df.at[person_id, 'ce_selected_for_via_this_month'] = False


class HSI_CervicalCancer_XpertHPVScreening(HSI_Event, IndividualScopeEventMixin):

    """
     This event will be scheduled by family planning HSI - for now we determine at random a screening event, and
     we determine at random whether this is AceticAcidScreening or HPVXpertScreening

     In future this might be scheduled by the contraception module
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_XpertHPVScreening"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        p = self.sim.modules['CervicalCancer'].parameters
        year = self.sim.date.year
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        # todo: if positive on xpert then do via if hiv negative but go straight to thermoablation
        # todo: if hiv positive ?

        # Run a test to diagnose whether the person has condition:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='screening_with_xpert_for_hpv',
            hsi_event=self
        )
        df.at[person_id, "ce_date_last_screened"] = self.sim.date
        df.at[person_id, "ce_date_xpert"] = self.sim.date
        df.at[person_id, "ce_ever_screened"] = True

        if dx_result:
            df.at[person_id, 'ce_xpert_hpv_ever_pos'] = True

        hpv_cin_options = ['hpv','cin1','cin2','cin3']
        hpv_stage_options = ['stage1','stage2a','stage2b','stage3','stage4']

        # If HIV negative, do VIA
        if not person['hv_inf']:
            if dx_result and (df.at[person_id, 'ce_hpv_cc_status'] in (hpv_cin_options+hpv_stage_options)
                            ):
                    hs.schedule_hsi_event(
                        hsi_event=HSI_CervicalCancer_AceticAcidScreening(
                            module=self.module,
                            person_id=person_id
                               ),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                               )
        # IF HIV positive,
        if person['hv_inf']:
            if dx_result and (df.at[person_id, 'ce_hpv_cc_status'] in (hpv_cin_options+hpv_stage_options)
                            ):
                if year >= p['transition_testing_year']:
                    hs.schedule_hsi_event(
                            hsi_event=HSI_CervicalCancer_Thermoablation_CIN(
                                module=self.module,
                                person_id=person_id
                                   ),
                            priority=0,
                            topen=self.sim.date,
                            tclose=None
                                   )
                else:
                    hs.schedule_hsi_event(
                            hsi_event=HSI_CervicalCancer_Cryotherapy_CIN(
                                module=self.module,
                                person_id=person_id
                                   ),
                            priority=0,
                            topen=self.sim.date,
                            tclose=None
                                   )

        # sy_chosen_via_screening_for_cin_cervical_cancer reset to 0
        # if df.at[person_id, 'sy_chosen_xpert_screening_for_hpv_cervical_cancer'] == 2:
        #     self.sim.modules['SymptomManager'].change_symptom(
        #         person_id=person_id,
        #         symptom_string='chosen_xpert_screening_for_hpv_cervical_cancer',
        #         add_or_remove='-',
        #         disease_module=self.module
        #         )
        #
        # df.at[person_id, 'ce_selected_for_xpert_this_month'] = False



class HSI_CervicalCancerPresentationVaginalBleeding(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_presentation_vaginal_bleeding"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        p = self.sim.modules['CervicalCancer'].parameters
        m = self.module
        rng = m.rng
        random_value = rng.random()

        if random_value <= p['prob_referral_biopsy_given_vaginal_bleeding']:
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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

#       print(person_id, self.sim.date, 'vaginal_bleeding_hsi_called -1')

        self.TREATMENT_ID = "CervicalCancer_Biopsy"

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        year = self.sim.date.year
        p = self.sim.modules['CervicalCancer'].parameters

        # Use a biopsy to diagnose whether the person has cervical cancer
        # todo: request consumables needed for this and elsewhere

        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='biopsy_for_cervical_cancer',
            hsi_event=self
        )

        df.at[person_id, "ce_biopsy"] = True

        if dx_result and (df.at[person_id, 'ce_hpv_cc_status'] in (hpv_cin_options) ):
            schedule_cin_procedure(year, p, person_id, self.sim.modules['HealthSystem'], self.module, self.sim)

        elif dx_result and (df.at[person_id, 'ce_hpv_cc_status'] == 'stage1'
                        or df.at[person_id, 'ce_hpv_cc_status'] == 'stage2a'
                        or df.at[person_id, 'ce_hpv_cc_status'] == 'stage2b'
                        or df.at[person_id, 'ce_hpv_cc_status'] == 'stage3'
                        or df.at[person_id, 'ce_hpv_cc_status'] == 'stage4'):
            # Record date of diagnosis:
            df.at[person_id, 'ce_date_diagnosis'] = self.sim.date
            df.at[person_id, 'ce_stage_at_diagnosis'] = df.at[person_id, 'ce_hpv_cc_status']
            df.at[person_id, 'ce_current_cc_diagnosed'] = True
            df.at[person_id, 'ce_ever_diagnosed'] = True

            # Check if is in stage4:
            in_stage4 = df.at[person_id, 'ce_hpv_cc_status'] == 'stage4'
            # If the diagnosis does detect cancer, it is assumed that the classification as stage4 is made accurately.

            if not in_stage4:
                # start treatment:
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_StartTreatment(
                        module=self.module,
                        person_id=person_id
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

            if in_stage4:
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


class HSI_CervicalCancer_Thermoablation_CIN(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Thermoablation_CIN"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.sim.modules['CervicalCancer'].parameters

       # (msyamboza et al 2016)

        # Record date and stage of starting treatment
        df.at[person_id, "ce_date_thermoabl"] = self.sim.date

        random_value = self.module.rng.random()

        if df.at[person_id, "ce_hpv_cc_status"] in (hpv_cin_options):
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_Biopsy(
                    module=self.module,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )
        else:
            if random_value <= p['prob_thermoabl_successful']:
                df.at[person_id, "ce_hpv_cc_status"] = 'none'


class HSI_CervicalCancer_Cryotherapy_CIN(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_Cryotherapy_CIN"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.sim.modules['CervicalCancer'].parameters

       # (msyamboza et al 2016)

        # Record date and stage of starting treatment
        df.at[person_id, "ce_date_cryotherapy"] = self.sim.date

        random_value = self.module.rng.random()

        if df.at[person_id, "ce_hpv_cc_status"] in (hpv_cin_options):
            hs.schedule_hsi_event(
                hsi_event=HSI_CervicalCancer_Biopsy(
                    module=self.module,
                    person_id=person_id
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )
        else:
            if random_value <= p['prob_cryotherapy_successful']:
                df.at[person_id, "ce_hpv_cc_status"] = 'none'


class HSI_CervicalCancer_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_Biopsy following a diagnosis of
    cervical Cancer. It initiates the treatment of cervical Cancer.
    It is only for persons with a cancer that is not in stage4 and who have been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_StartTreatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MajorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({"general_bed": 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.sim.modules['CervicalCancer'].parameters

        # If the status is already in `stage4`, start palliative care (instead of treatment)
        if df.at[person_id, "ce_hpv_cc_status"] == 'stage4':
            logger.warning(key="warning", data="Cancer is in stage 4 - aborting HSI_CervicalCancer_StartTreatment,"
                                               "scheduling HSI_CervicalCancer_PalliativeCare")

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

        # Check that the person has been diagnosed and is not on treatment
        assert not pd.isnull(df.at[person_id, "ce_date_diagnosis"])

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

        random_value = self.module.rng.random()

        if (random_value <= p['prob_cure_stage1'] and df.at[person_id, "ce_hpv_cc_status"] == "stage1"
            and df.at[person_id, "ce_date_treatment"] == self.sim.date):
            df.at[person_id, "ce_hpv_cc_status"] = 'none'
            df.at[person_id, 'ce_current_cc_diagnosed'] = False
            df.at[person_id, 'ce_cured_date_cc'] = self.sim.date
        else:
            df.at[person_id, "ce_hpv_cc_status"] = 'stage1'

        if (random_value <= p['prob_cure_stage2a'] and df.at[person_id, "ce_hpv_cc_status"] == "stage2a"
            and df.at[person_id, "ce_date_treatment"] == self.sim.date):
            df.at[person_id, "ce_hpv_cc_status"] = 'none'
            df.at[person_id, 'ce_current_cc_diagnosed'] = False
            df.at[person_id, 'ce_cured_date_cc'] = self.sim.date
        else:
            df.at[person_id, "ce_hpv_cc_status"] = 'stage2a'

        if (random_value <= p['prob_cure_stage2b'] and df.at[person_id, "ce_hpv_cc_status"] == "stage2b"
            and df.at[person_id, "ce_date_treatment"] == self.sim.date):
            df.at[person_id, "ce_hpv_cc_status"] = 'none'
            df.at[person_id, 'ce_current_cc_diagnosed'] = False
            df.at[person_id, 'ce_cured_date_cc'] = self.sim.date
        else:
            df.at[person_id, "ce_hpv_cc_status"] = 'stage2b'

        if (random_value <= p['prob_cure_stage3'] and df.at[person_id, "ce_hpv_cc_status"] == "stage3"
            and df.at[person_id, "ce_date_treatment"] == self.sim.date):
            df.at[person_id, "ce_hpv_cc_status"] = 'none'
            df.at[person_id, 'ce_current_cc_diagnosed'] = False
            df.at[person_id, 'ce_cured_date_cc'] = self.sim.date
        else:
            df.at[person_id, "ce_hpv_cc_status"] = 'stage3'

        # Schedule a post-treatment check for 3 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )

class HSI_CervicalCancer_PostTreatmentCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_CervicalCancer_StartTreatment and itself.
    It is only for those who have undergone treatment for cervical Cancer.
    If the person has developed cancer to stage4, the patient is initiated on palliative care; otherwise a further
    appointment is scheduled for one year.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CervicalCancer_PostTreatmentCheck"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '3'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        assert not pd.isnull(df.at[person_id, "ce_date_diagnosis"])
        assert not pd.isnull(df.at[person_id, "ce_date_treatment"])

        # todo:
        # could use pd.Dateoffset(years =...) instead of the number of days for ease for
        # reading/comprehension

        if df.at[person_id, 'ce_hpv_cc_status'] == 'stage4':
            # If has progressed to stage4, then start Palliative Care immediately:
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
            if df.at[person_id, 'ce_date_treatment'] > (self.sim.date - pd.DateOffset(years=1)):
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                    ),
                    topen=self.sim.date + DateOffset(months=3),
                    tclose=None,
                    priority=0
                )
            if df.at[person_id, 'ce_date_treatment'] < (self.sim.date - pd.DateOffset(years=1)) \
                and df.at[person_id, 'ce_date_treatment'] > (self.sim.date - pd.DateOffset(years=3)):
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                    ),
                    topen=self.sim.date + DateOffset(months=6),
                    tclose=None,
                    priority=0
                )
            if df.at[person_id, 'ce_date_treatment'] < (self.sim.date - pd.DateOffset(years=3)) \
                and df.at[person_id, 'ce_date_treatment'] > (self.sim.date - pd.DateOffset(years=5)):
                hs.schedule_hsi_event(
                    hsi_event=HSI_CervicalCancer_PostTreatmentCheck(
                    module=self.module,
                    person_id=person_id
                    ),
                    topen=self.sim.date + DateOffset(months=12),
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

        self.TREATMENT_ID = "CervicalCancer_PalliativeCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 15})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        # Check that the person is in stage4
        # assert df.at[person_id, "ce_hpv_cc_status"] == 'stage4'

        # Record the start of palliative care if this is first appointment
        if pd.isnull(df.at[person_id, "ce_date_palliative_care"]):
            df.at[person_id, "ce_date_palliative_care"] = self.sim.date



        # todo:
        # for scheduling the same class of HSI_Event to multiple people, more
        # efficient to use schedule_batch_of_individual_hsi_events




        # Schedule another instance of the event for one month
        hs.schedule_hsi_event(
            hsi_event=HSI_CervicalCancer_PalliativeCare(
                module=self.module,
                person_id=person_id
            ),
            topen=self.sim.date + DateOffset(months=1),
            tclose=None,
            priority=0
        )

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class CervicalCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """The only logging event for this module"""

    # the use of groupby might be more efficient in computing the statistics below;

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

        date_lastlog = self.sim.date - pd.DateOffset(days=29)

        # Current counts, total
        out.update({
            f'total_{k}': v for k, v in df.loc[df.is_alive & (df['sex'] == 'F') &
                                               (df['age_years'] > 15)].ce_hpv_cc_status.value_counts().items()})

        # Current counts, total hiv negative
        out.update({
            f'total_hivneg_{k}': v for k, v in df.loc[df.is_alive & (df['sex'] == 'F') &
                                               (df['age_years'] > 15) & (~df['hv_inf'])].ce_hpv_cc_status.value_counts().items()})

        # Current counts, total hiv positive
        out.update({
            f'total_hivpos_{k}': v for k, v in df.loc[df.is_alive & (df['sex'] == 'F') &
                                               (df['age_years'] > 15) & (df['hv_inf'])].ce_hpv_cc_status.value_counts().items()})

        out.update({
            f'total_males': len(df[df.is_alive & (df['sex'] == 'M')])})
        out.update({
            f'total_dead': len(df[df.is_alive == False])})
        out.update({
            f'total_overall': len(df)})

        # Get the day of the year
        day_of_year = self.sim.date.timetuple().tm_yday

        # Calculate the decimal year
        decimal_year = self.sim.date.year + (day_of_year - 1) / 365.25
        rounded_decimal_year = round(decimal_year, 2)

        date_1_year_ago = self.sim.date - pd.DateOffset(days=365)
        date_30_days_ago = self.sim.date - pd.DateOffset(days=30)
        n_deaths_past_year = df.ce_date_death.between(date_1_year_ago, self.sim.date).sum()
        n_deaths_cc_hivneg_past_year = ((~df['hv_inf']) & df.ce_date_death.between(date_1_year_ago, self.sim.date)).sum()
        n_deaths_cc_hivpos_past_year = ((df['hv_inf']) & df.ce_date_death.between(date_1_year_ago, self.sim.date)).sum()
        n_deaths_cc_hiv_past_year = ((df['hv_inf']) & df.ce_date_death.between(date_1_year_ago, self.sim.date)).sum()
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


        n_screened_via_this_month = (df.is_alive & df.ce_selected_for_via_this_month ).sum()
        n_screened_xpert_this_month = (df.is_alive & df.ce_selected_for_xpert_this_month ).sum()
        n_ever_screened = (
                (df['is_alive']) & (df['ce_ever_screened']) & (df['age_years'] > screening_min_age) & (df['age_years'] < screening_max_age)).sum()


        # n_screened_via_this_month = (df.is_alive & df.ce_selected_for_via_this_month & df.ce_date_via.between(date_30_days_ago, self.sim.date)).sum()
        # n_screened_xpert_this_month = (df.is_alive & df.ce_selected_for_xpert_this_month & df.ce_date_xpert.between(date_30_days_ago, self.sim.date)).sum()
        # n_ever_screened = (
        #         (df['is_alive']) & (df['ce_ever_screened']) & (df['age_years'] > 15) & (df['age_years'] < 50)).sum()

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

        n_women_alive = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] > 15)).sum()
        n_women_alive_1549 = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] > 15)
                              & (df['age_years'] < 50)).sum()

        n_women_vaccinated = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] > 15)
                              & df['va_hpv']).sum()

        n_women_hiv_unsuppressed = ((df['is_alive']) & (df['sex'] == 'F') & (df['age_years'] > 15)
                                    & df['ce_hiv_unsuppressed']).sum()

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
        out.update({"n_deaths_cc_hiv_past_year": n_deaths_cc_hiv_past_year})
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


        pop = len(df[df.is_alive])
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
        # todo:
        # ? move to using the logger:
        # i.e. logger.info(key='cervical_cancer_stats_every_month', description='XX', data=out)

        print(self.sim.date, 'total_none:', out['total_none'], 'total_hpv:', out['total_hpv'], 'total_cin1:',out['total_cin1'],
              'total_cin2:', out['total_cin2'], 'total_cin3:', out['total_cin3'], 'total_stage1:', out['total_stage1'],
              'total_stage2a:', out['total_stage2a'], 'total_stage2b:', out['total_stage2b'],
              'total_stage3:', out['total_stage3'],'total_stage4:', out['total_stage4'],
              'total_hivneg_none:', out['total_hivneg_none'], 'total_hivneg_hpv:', out['total_hivneg_hpv'], 'total_hivneg_cin1:', out['total_hivneg_cin1'],
              'total_hivneg_cin2:', out['total_hivneg_cin2'], 'total_hivneg_cin3:', out['total_hivneg_cin3'], 'total_hivneg_stage1:', out['total_hivneg_stage1'],
              'total_hivneg_stage2a:', out['total_hivneg_stage2a'], 'total_hivneg_stage2b:', out['total_hivneg_stage2b'],
              'total_hivneg_stage3:', out['total_hivneg_stage3'], 'total_hivneg_stage4:', out['total_hivneg_stage4'],
              'year:', out['rounded_decimal_year'], 'deaths_past_year:', out['n_deaths_past_year'],out['n_via_past_year'],out['n_xpert_past_year'],
              'n_deaths_cc_hivneg_past_year:', out['n_deaths_cc_hivneg_past_year'],
              'n_deaths_cc_hivpos_past_year:', out['n_deaths_cc_hivpos_past_year'],
              'n_deaths_cc_hiv_past_year:', out['n_deaths_cc_hiv_past_year'],
              'treated past year:', out['n_treated_past_year'], 'prop cc hiv:', out['prop_cc_hiv'],
              'n_vaginal_bleeding_stage1:', out['n_vaginal_bleeding_stage1'],
              'n_vaginal_bleeding_stage2a:', out['n_vaginal_bleeding_stage2a'],
              'n_vaginal_bleeding_stage2b:', out['n_vaginal_bleeding_stage2b'],
              'n_vaginal_bleeding_stage3:', out['n_vaginal_bleeding_stage3'],
              'n_vaginal_bleeding_stage4:', out['n_vaginal_bleeding_stage4'],
              'diagnosed_past_year_stage1:', out['n_diagnosed_past_year_stage1'],
              'diagnosed_past_year_stage2a:', out['n_diagnosed_past_year_stage2a'],
              'diagnosed_past_year_stage2b:', out['n_diagnosed_past_year_stage2b'],
              'diagnosed_past_year_stage3:', out['n_diagnosed_past_year_stage3'],
              'diagnosed_past_year_stage4:', out['n_diagnosed_past_year_stage4'],
              'n_ever_diagnosed', out['n_ever_diagnosed'],
              'n_screened_xpert_this_month:', out['n_screened_xpert_this_month'],
              'n_screened_via_this_month:', out['n_screened_via_this_month'],
              'n_women_alive', out['n_women_alive'],
              'n_women_alive_1549', out['n_women_alive_1549'],
              'n_women_vaccinated', out['n_women_vaccinated'],
              'n_ever_screened', out['n_ever_screened'],
              'n_diagnosed_past_year:', out['n_diagnosed_past_year'],
              'n_cured_past_year:', out['n_cured_past_year'],
              'n_thermoabl_past_year:', out['n_thermoabl_past_year'],
              'n_cryotherapy_past_year:', out['n_cryotherapy_past_year'],
              'n_women_alive:', out['n_women_alive'],
              'rate_diagnosed_cc:', out['rate_diagnosed_cc'],
              'n_women_with_cc:', out['cc'],
              'n_women_living_with_diagnosed_cc:', out['n_women_living_with_diagnosed_cc'],
              'n_women_living_with_diagnosed_cc_age_lt_30:', out['n_women_living_with_diagnosed_cc_age_lt_30'],
              'n_women_living_with_diagnosed_cc_age_3050:', out['n_women_living_with_diagnosed_cc_age_3050'],
              'n_women_living_with_diagnosed_cc_age_gt_50:', out['n_women_living_with_diagnosed_cc_age_gt_50'],
              'n_diagnosed_1_year_ago_died:', out['n_diagnosed_1_year_ago_died'],
              'n_diagnosed_1_year_ago:', out['n_diagnosed_1_year_ago'],
              'n_women_hiv_unsuppressed:', out['n_women_hiv_unsuppressed'],
              'n_women_hivneg', out['n_women_hivneg'],
              'n_women_hivpos', out['n_women_hivpos'])

        # comment out this below when running tests

        # Specify the file path for the CSV file
        out_csv = Path("./outputs/output1_data.csv")

# comment out this code below only when running tests

        with open(out_csv, "a", newline="") as csv_file:
            # Create a CSV writer
            csv_writer = csv.DictWriter(csv_file, fieldnames=out.keys())

            # If the file is empty, write the header
            if csv_file.tell() == 0:
                csv_writer.writeheader()

            # Write the data to the CSV file
            csv_writer.writerow(out)

#       print(out)

        # Disable column truncation
        pd.set_option('display.max_columns', None)

        # Set the display width to a large value to fit all columns in one row
        pd.set_option('display.width', 1000)

        selected_columns = ["ce_hpv_cc_status",
        "ce_date_treatment",
        "ce_stage_at_which_treatment_given",
        "ce_date_diagnosis",
        "ce_new_stage_this_month",
        "ce_date_palliative_care",
        "ce_date_death",
        "ce_date_cin_removal",
        "ce_date_treatment",
        "ce_stage_at_diagnosis",
        "ce_ever_treated",
        "ce_cc_ever",
        "ce_xpert_hpv_ever_pos",
        "ce_via_cin_ever_detected",
        "ce_date_thermoabl",
        "ce_date_cryotherapy",
        "ce_current_cc_diagnosed",
        "ce_selected_for_via_this_month",
        "ce_selected_for_xpert_this_month",
        "ce_biopsy"]


        selected_columns = ['ce_hpv_cc_status', 'sy_vaginal_bleeding', 'ce_biopsy','ce_current_cc_diagnosed',
        'ce_selected_for_xpert_this_month', 'sy_chosen_xpert_screening_for_hpv_cervical_cancer',
        'ce_xpert_hpv_ever_pos', 'ce_date_thermoabl','ce_date_cryotherapy',
        'ce_date_diagnosis', 'ce_date_treatment','ce_cured_date_cc',
        'ce_date_palliative_care', 'ce_selected_for_via_this_month', 'sy_chosen_via_screening_for_cin_cervical_cancer',
        'ce_via_cin_ever_detected']

#       selected_columns = ["hv_inf", "ce_hiv_unsuppressed", "hv_art", "ce_hpv_cc_status",'ce_cured_date_cc']

        selected_columns = ["ce_selected_for_via_this_month", "ce_selected_for_xpert_this_month",
                            "ce_ever_screened", "ce_date_last_screened", "ce_date_cin_removal",
                            "ce_xpert_hpv_ever_pos", "ce_via_cin_ever_detected",  "ce_date_thermoabl","ce_date_cryotherapy",
                            "ce_biopsy"]

        # selected_columns = ["ce_hpv_cc_status"]

        selected_rows = df[(df['sex'] == 'F') & (df['age_years'] > 15) & df['is_alive'] & (df['hv_inf'])]

#       pd.set_option('display.max_rows', None)
        print(selected_rows[selected_columns])

#       selected_columns = ['sex', 'age_years', 'is_alive']
#       pd.set_option('display.max_rows', None)
#       print(df[selected_columns])








