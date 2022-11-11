"""
The joint Cardio-Metabolic Disorders model determines onset, outcome and treatment of:
* Diabetes
* Hypertension
* Chronic Kidney Disease
* Chronic Ischemic Heart Disease
* Stroke
* Heart Attack

And:
* Chronic Lower Back Pain
"""

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods import demography as de
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CardioMetabolicDisorders(Module):
    """
    CardioMetabolicDisorders module covers a subset of cardio-metabolic conditions and events. Conditions are binary
    and individuals experience a risk of acquiring or losing a condition based on annual probability and
    demographic/lifestyle risk factors.
    """
    # Save a master list of the events that are covered in this module
    conditions = ['diabetes',
                  'hypertension',
                  'chronic_kidney_disease',
                  'chronic_lower_back_pain',
                  'chronic_ischemic_hd']

    # Save a master list of the events that are covered in this module
    events = ['ever_stroke',
              'ever_heart_attack']

    INIT_DEPENDENCIES = {'Demography', 'Lifestyle', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    ADDITIONAL_DEPENDENCIES = {'Depression'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'diabetes': Cause(
            gbd_causes='Diabetes mellitus', label='Diabetes'),
        'chronic_ischemic_hd': Cause(
            gbd_causes={'Ischemic heart disease', 'Hypertensive heart disease'}, label='Heart Disease'),
        'ever_heart_attack': Cause(
            gbd_causes={'Ischemic heart disease', 'Hypertensive heart disease'}, label='Heart Disease'),
        'ever_stroke': Cause(
            gbd_causes='Stroke', label='Stroke'),
        'chronic_kidney_disease': Cause(
            gbd_causes='Chronic kidney disease', label='Kidney Disease')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'diabetes': Cause(
            gbd_causes='Diabetes mellitus', label='Diabetes'),
        'chronic_ischemic_hd': Cause(
            gbd_causes={'Ischemic heart disease', 'Hypertensive heart disease'}, label='Heart Disease'),
        'heart_attack': Cause(
            gbd_causes={'Ischemic heart disease', 'Hypertensive heart disease'}, label='Heart Disease'),
        'stroke': Cause(
            gbd_causes='Stroke', label='Stroke'),
        'chronic_kidney_disease': Cause(
            gbd_causes='Chronic kidney disease', label='Kidney Disease'),
        'lower_back_pain': Cause(
            gbd_causes={'Low back pain'}, label='Lower Back Pain'
        )
    }

    # Create separate dicts for params for conditions and events which are read in via excel documents in resources/cmd
    onset_conditions_param_dicts = {
        f"{p}_onset": Parameter(Types.DICT, f"all the parameters that specify the linear models for onset of {p}")
        for p in conditions
    }
    removal_conditions_param_dicts = {
        f"{p}_removal": Parameter(Types.DICT, f"all the parameters that specify the linear models for removal of {p}")
        for p in conditions
    }
    hsi_conditions_param_dicts = {
        f"{p}_hsi": Parameter(Types.DICT, f"all the parameters that specify diagnostic tests and treatments for {p}")
        for p in conditions
    }
    onset_events_param_dicts = {
        f"{p}_onset": Parameter(Types.DICT, f"all the parameters that specify the linear models for onset of {p}")
        for p in events
    }
    hsi_events_param_dicts = {
        f"{p}_hsi": Parameter(Types.DICT, f"all the parameters that specify diagnostic tests and treatments for {p}")
        for p in events
    }
    death_conditions_param_dicts = {
        f"{p}_death": Parameter(Types.DICT, f"all the parameters that specify the linear models for death from {p}")
        for p in conditions
    }
    death_events_param_dicts = {
        f"{p}_death": Parameter(Types.DICT, f"all the parameters that specify the linear models for death from {p}")
        for p in events
    }
    initial_prev_param_dicts = {
        f"{p}_initial_prev": Parameter(Types.DICT, 'initial prevalence of condition') for p in conditions
    }
    other_params_dict = {
        'interval_between_polls': Parameter(Types.INT, 'months between the main polling event'),
        'pr_bmi_reduction': Parameter(Types.INT, 'probability of an individual having a reduction in BMI following '
                                                 'weight loss treatment')
    }

    PARAMETERS = {**onset_conditions_param_dicts, **removal_conditions_param_dicts, **hsi_conditions_param_dicts,
                  **onset_events_param_dicts, **death_conditions_param_dicts, **death_events_param_dicts,
                  **hsi_events_param_dicts, **initial_prev_param_dicts, **other_params_dict}

    # Convert conditions and events to dicts and merge together into PROPERTIES
    condition_list = {
        f"nc_{p}": Property(Types.BOOL, f"Whether or not someone has {p}") for p in conditions
    }
    condition_diagnosis_list = {
        f"nc_{p}_ever_diagnosed": Property(Types.BOOL, f"Whether or not someone has ever been diagnosed with {p}") for p
        in conditions
    }
    condition_date_diagnosis_list = {
        f"nc_{p}_date_diagnosis": Property(Types.DATE, f"When someone has been diagnosed with {p}") for p
        in conditions
    }
    condition_date_of_last_test_list = {
        f"nc_{p}_date_last_test": Property(Types.DATE, f"When someone has last been tested for {p}") for p
        in conditions
    }
    condition_medication_list = {
        f"nc_{p}_on_medication": Property(Types.BOOL, f"Whether or not someone is on medication for {p}") for p
        in conditions
    }
    condition_medication_death_list = {
        f"nc_{p}_medication_prevents_death": Property(Types.BOOL, f"Whether or not medication (if provided) will "
                                                                  f"prevent death from {p}") for p in conditions
    }
    event_list = {
        f"nc_{p}": Property(Types.BOOL, f"Whether or not someone has had a {p}") for p in events}
    event_date_last_list = {
        f"nc_{p}_date_last_event": Property(Types.DATE, f"Date of last {p}") for p in events
    }
    event_diagnosis_list = {
        f"nc_{p}_ever_diagnosed": Property(Types.BOOL, f"Whether or not someone has ever been diagnosed with {p}") for p
        in events
    }
    event_date_diagnosis_list = {
        f"nc_{p}_date_diagnosis": Property(Types.DATE, f"When someone has last been diagnosed with {p}") for p
        in events}
    event_medication_list = {
        f"nc_{p}_on_medication": Property(Types.BOOL, f"Whether or not someone has ever been diagnosed with {p}") for p
        in events
    }
    event_scheduled_date_death_list = {
        f"nc_{p}_scheduled_date_death": Property(Types.DATE, f"Scheduled date of death from {p}") for p
        in events
    }
    event_medication_death_list = {
        f"nc_{p}_medication_prevents_death": Property(Types.BOOL, f"Whether or not medication will prevent death from "
                                                                  f"{p}") for p in events
    }

    PROPERTIES = {**condition_list, **event_list, **condition_diagnosis_list, **condition_date_diagnosis_list,
                  **condition_date_of_last_test_list, **condition_medication_list, **condition_medication_death_list,
                  **event_date_last_list, **event_diagnosis_list, **event_date_diagnosis_list, **event_medication_list,
                  **event_medication_death_list, **event_scheduled_date_death_list,
                  'nc_ever_weight_loss_treatment': Property(Types.BOOL,
                                                            'whether or not the person has ever had weight loss '
                                                            'treatment'),
                  'nc_weight_loss_worked': Property(Types.BOOL,
                                                    'whether or not weight loss treatment worked'),
                  'nc_risk_score': Property(Types.INT, 'score to represent number of risk conditions the person has')
                  }

    def __init__(self, name=None, resourcefilepath=None, do_log_df: bool = False, do_condition_combos: bool = False):

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.conditions = CardioMetabolicDisorders.conditions
        self.events = CardioMetabolicDisorders.events

        # Create list that includes the nc_ prefix for conditions in this module
        self.condition_list = ['nc_' + cond for cond in CardioMetabolicDisorders.conditions]

        # Store the symptoms that this module will use (for conditions only):
        self.symptoms = {f"{s}_symptoms" for s in self.conditions if s != "hypertension"}

        # Dict to hold the probability of onset of different types of symptom given a condition
        self.prob_symptoms = dict()
        # Retrieve age range categories from Demography module
        self.age_cats = None
        # Dict to hold the DALY weights
        self.daly_wts = dict()

        # Retrieve age range categories from Demography module
        self.age_cats = None

        # Store bools for whether or not to log the df or log combinations of co-morbidities
        self.do_log_df = do_log_df
        self.do_condition_combos = do_condition_combos

        # Dict to hold trackers for counting events
        self.trackers = None

        # Dicts of linear models
        self.lms_onset = dict()
        self.lms_removal = dict()
        self.lms_death = dict()
        self.lms_symptoms = dict()
        self.lms_testing = dict()

        # Build the LinearModel for occurrence of events
        self.lms_event_onset = dict()
        self.lms_event_death = dict()
        self.lms_event_symptoms = dict()

    def read_parameters(self, data_folder):
        """Read parameter values from files for condition onset, removal, deaths, and initial prevalence.

        ResourceFile_cmd_condition_onset.xlsx = parameters for onset of conditions
        ResourceFile_cmd_condition_removal.xlsx  = parameters for removal of conditions
        ResourceFile_cmd_condition_death.xlsx  = parameters for death rate from conditions
        ResourceFile_cmd_condition_prevalence.xlsx  = initial and target prevalence for conditions
        ResourceFile_cmd_condition_symptoms.xlsx  = symptoms for conditions
        ResourceFile_cmd_condition_hsi.xlsx  = HSI parameters for conditions
        ResourceFile_cmd_condition_testing.xlsx  = community testing parameters for conditions (currently only
        hypertension)
        ResourceFile_cmd_events.xlsx  = parameters for occurrence of events
        ResourceFile_cmd_events_death.xlsx  = parameters for death rate from events
        ResourceFile_cmd_events_symptoms.xlsx  = symptoms for events
        ResourceFile_cmd_events_hsi.xlsx  = HSI parameters for events

        """
        cmd_path = Path(self.resourcefilepath) / "cmd"
        cond_onset = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_onset.xlsx", sheet_name=None)
        cond_removal = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_removal.xlsx", sheet_name=None)
        cond_death = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_death.xlsx", sheet_name=None)
        cond_prevalence = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_prevalence.xlsx", sheet_name=None)
        cond_symptoms = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_symptoms.xlsx", sheet_name=None)
        cond_hsi = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_hsi.xlsx", sheet_name=None)
        cond_testing = pd.read_excel(cmd_path / "ResourceFile_cmd_condition_testing.xlsx", sheet_name=None)
        events_onset = pd.read_excel(cmd_path / "ResourceFile_cmd_events.xlsx", sheet_name=None)
        events_death = pd.read_excel(cmd_path / "ResourceFile_cmd_events_death.xlsx", sheet_name=None)
        events_symptoms = pd.read_excel(cmd_path / "ResourceFile_cmd_events_symptoms.xlsx", sheet_name=None)
        events_hsi = pd.read_excel(cmd_path / "ResourceFile_cmd_events_hsi.xlsx", sheet_name=None)

        def get_values(params, value):
            """replaces nans in the 'value' key with specified value"""
            params['value'] = params['value'].replace(np.nan, value)
            params['value'] = params['value'].astype(float)
            return params.set_index('parameter_name')['value']

        p = self.parameters

        for condition in self.conditions:
            p[f'{condition}_onset'] = get_values(cond_onset[condition], 1)
            p[f'{condition}_removal'] = get_values(cond_removal[condition], 1)
            p[f'{condition}_death'] = get_values(cond_death[condition], 1)
            p[f'{condition}_initial_prev'] = get_values(cond_prevalence[condition], 0)
            p[f'{condition}_symptoms'] = get_values(cond_symptoms[condition], 1)
            p[f'{condition}_hsi'] = get_values(cond_hsi[condition], 1)

        p['hypertension_testing'] = get_values(cond_testing['hypertension'], 1)

        for event in self.events:
            p[f'{event}_onset'] = get_values(events_onset[event], 1)
            p[f'{event}_death'] = get_values(events_death[event], 1)
            p[f'{event}_symptoms'] = get_values(events_symptoms[event], 1)
            p[f'{event}_hsi'] = get_values(events_hsi[event], 1)

        # Set the interval (in months) between the polls
        p['interval_between_polls'] = 3
        # Set the probability of an individual losing weight in the CardioMetabolicDisordersWeightLossEvent (this value
        # doesn't vary by condition)
        p['pr_bmi_reduction'] = 0.1

        # Check that every value has been read-in successfully
        for param_name in self.PARAMETERS:
            assert self.parameters[param_name] is not None, f'Parameter "{param_name}" has not been set.'

        # -------------------------------------------- SYMPTOMS -------------------------------------------------------
        # Retrieve symptom probabilities
        for condition in self.conditions:
            if not self.parameters[f'{condition}_symptoms'].empty:
                self.prob_symptoms[condition] = self.parameters[f'{condition}_symptoms']
            else:
                self.prob_symptoms[condition] = {}

        for event in self.events:
            if not self.parameters[f'{event}_symptoms'].empty:
                self.prob_symptoms[event] = self.parameters[f'{event}_symptoms']
            else:
                self.prob_symptoms[event] = {}

        # Register symptoms for conditions and give non-generic symptom 'average' healthcare seeking, except for
        # chronic lower back pain, which has no healthcare seeking in adults
        for symptom_name in self.symptoms:
            if symptom_name == "chronic_lower_back_pain_symptoms":
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name,
                            no_healthcareseeking_in_adults=True)
                )
            else:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name,
                            odds_ratio_health_seeking_in_adults=1.0)
                )
        # Register symptoms from events and make them emergencies
        for event in self.events:
            self.sim.modules['SymptomManager'].register_symptom(
                Symptom(
                    name=f'{event}_damage',
                    emergency_in_adults=True
                ),
            )

    def initialise_population(self, population):
        """
        Set property values for the initial population.
        """
        self.age_cats = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
        df = population.props

        men = df.is_alive & (df.sex == 'M')
        women = df.is_alive & (df.sex == 'F')

        def sample_eligible(_filter, _p, _condition):
            """uses filter to get eligible population and samples individuals for condition using p"""
            eligible = df.index[_filter]
            init_prev = self.rng.choice([True, False], size=len(eligible), p=[_p, 1 - _p])
            if sum(init_prev):
                df.loc[eligible[init_prev], f'nc_{_condition}'] = True

        def sample_eligible_diagnosis_medication(_filter, _p, _condition):
            """uses filter to get eligible population and samples individuals for prior diagnosis & medication use
             using p"""
            eligible = df.index[_filter]
            init_diagnosis = self.rng.choice([True, False], size=len(eligible), p=[_p, 1 - _p])
            if sum(init_diagnosis):
                df.loc[eligible[init_diagnosis], f'nc_{_condition}_ever_diagnosed'] = True
                df.loc[eligible[init_diagnosis], f'nc_{_condition}_date_diagnosis'] = self.sim.date
                df.loc[eligible[init_diagnosis], f'nc_{_condition}_date_last_test'] = self.sim.date
                df.loc[eligible[init_diagnosis], f'nc_{_condition}_on_medication'] = True

        def sample_eligible_treatment_success(_filter, _p, _condition):
            """uses filter to get eligible population and samples individuals for prior diagnosis & medication use
             using p"""
            eligible = df.index[_filter]
            init_treatment_works = self.rng.choice([True, False], size=len(eligible), p=[_p, 1 - _p])
            if sum(init_treatment_works):
                df.loc[eligible[init_treatment_works], f'nc_{_condition}_medication_prevents_death'] = True

        for condition in self.conditions:
            p = self.parameters[f'{condition}_initial_prev']
            # Men & women without condition
            men_wo_cond = men & ~df[f'nc_{condition}']
            women_wo_cond = women & ~df[f'nc_{condition}']
            for _age_range in self.age_cats:
                # Select all eligible individuals (men & women w/o condition and in age range)
                sample_eligible(men_wo_cond & (df.age_range == _age_range), p[f'm_{_age_range}'], condition)
                sample_eligible(women_wo_cond & (df.age_range == _age_range), p[f'f_{_age_range}'], condition)

            # ----- Set variables to false / NaT for everyone
            df.loc[df.is_alive, f'nc_{condition}_date_last_test'] = pd.NaT
            df.loc[df.is_alive, f'nc_{condition}_ever_diagnosed'] = False
            df.loc[df.is_alive, f'nc_{condition}_date_diagnosis'] = pd.NaT
            df.loc[df.is_alive, f'nc_{condition}_on_medication'] = False
            df.loc[df.is_alive, f'nc_{condition}_medication_prevents_death'] = False

            # ----- Sample among eligible population who have condition to set initial proportion diagnosed & on
            # medication

            p_initial_diagnosis = self.parameters[f'{condition}_hsi']['pr_diagnosed']
            # Population with condition
            w_cond = df[f'nc_{condition}']
            sample_eligible_diagnosis_medication(w_cond, p_initial_diagnosis, condition)

            # For those on medication, sample to set initial proportion for whom medication prevents death
            p_treatment_works = self.parameters[f'{condition}_hsi']['pr_treatment_works']
            # Population already on medication
            on_med = df[f'nc_{condition}_on_medication']
            sample_eligible_treatment_success(on_med, p_treatment_works, condition)

            # ----- Impose the symptom on random sample of those with each condition to have:
            # TODO: @britta make linear model data-specific and add in needed complexity
            for symptom in self.prob_symptoms[condition].keys():
                lm_init_symptoms = LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    self.prob_symptoms[condition].get(f'{symptom}'),
                    Predictor(
                        f'nc_{condition}',
                        conditions_are_mutually_exclusive=True,
                        conditions_are_exhaustive=True
                    )
                    .when(True, 1.0)
                    .when(False, 0.0))
                has_symptom_at_init = lm_init_symptoms.predict(df.loc[df.is_alive], self.rng)
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=has_symptom_at_init.index[has_symptom_at_init].tolist(),
                    symptom_string=f'{symptom}',
                    add_or_remove='+',
                    disease_module=self)

        # ----- Set ever_diagnosed, date of diagnosis, and on_medication to false / NaT
        # for everyone
        for event in self.events:
            df.loc[df.is_alive, f'nc_{event}'] = False
            df.loc[df.is_alive, f'nc_{event}_date_last_event'] = pd.NaT
            df.loc[df.is_alive, f'nc_{event}_ever_diagnosed'] = False
            df.loc[df.is_alive, f'nc_{event}_date_diagnosis'] = pd.NaT
            df.loc[df.is_alive, f'nc_{event}_on_medication'] = False
            df.loc[df.is_alive, f'nc_{event}_scheduled_date_death'] = pd.NaT
            df.loc[df.is_alive, f'nc_{event}_medication_prevents_death'] = False

        # ----- Generate the initial "risk score" for the population based on exercise, diet, tobacco, alcohol, BMI
        self.update_risk_score()

        # ----- Set all other parameters to False / NaT
        df.loc[df.is_alive, 'nc_ever_weight_loss_treatment'] = False
        df.loc[df.is_alive, 'nc_weight_loss_worked'] = False

    def initialise_simulation(self, sim):
        """Schedule:
        * Main Polling Event
        * Main Logging Event
        * Build the LinearModels for the onset/removal of each condition:
        """
        sim.schedule_event(CardioMetabolicDisorders_MainPollingEvent(self, self.parameters['interval_between_polls']),
                           sim.date)
        sim.schedule_event(CardioMetabolicDisorders_LoggingEvent(self), sim.date)

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_diabetes_uncomplicated'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=971)
            self.daly_wts['daly_diabetes_complicated'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=970)
            self.daly_wts['daly_chronic_kidney_disease_moderate'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=978)
            self.daly_wts['daly_chronic_ischemic_hd'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=697)
            self.daly_wts['daly_chronic_lower_back_pain'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=1180)
            self.daly_wts['daly_stroke'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=701)
            self.daly_wts['daly_heart_attack_acute_days_1_2'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=689)
            self.daly_wts['daly_heart_attack_acute_days_3_28'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=693)
            # Days 1-2 of heart attack have a more severe weight than days 3028; the average of these is what is used
            # for the DALY weight for any heart attack event
            self.daly_wts['daly_heart_attack_avg'] = (
                (2 / 28) * self.daly_wts['daly_heart_attack_acute_days_1_2']
                + (26 / 28) * self.daly_wts['daly_heart_attack_acute_days_3_28']
            )

        self.trackers = {
            'onset_condition': Tracker(conditions=self.conditions, age_groups=self.age_cats),
            'incident_event': Tracker(conditions=self.events, age_groups=self.age_cats),
            'prevalent_event': Tracker(conditions=self.events, age_groups=self.age_cats),
        }

        # Build the LinearModel for onset/removal/deaths for each condition
        # Baseline probability of condition onset, removal, and death are annual; in LinearModel, rates are adjusted to
        # be consistent with the polling interval
        for condition in self.conditions:
            self.lms_onset[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'],
                                                                lm_type='onset')
            self.lms_removal[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'],
                                                                  lm_type='removal')
            self.lms_death[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'],
                                                                lm_type='death')
            self.lms_symptoms[condition] = self.build_linear_model_symptoms(condition, self.parameters[
                'interval_between_polls'])

        # Hypertension is the only condition for which we assume some community-based testing occurs; build LM based on
        # age / sex
        self.lms_testing['hypertension'] = self.build_linear_model('hypertension', self.parameters[
                'interval_between_polls'], lm_type='testing')

        for event in self.events:
            self.lms_event_onset[event] = self.build_linear_model(event, self.parameters['interval_between_polls'],
                                                                  lm_type='onset')
            self.lms_event_death[event] = self.build_linear_model(event, self.parameters['interval_between_polls'],
                                                                  lm_type='death')
            self.lms_event_symptoms[event] = self.build_linear_model_symptoms(event, self.parameters[
                'interval_between_polls'])

        # ------------------------------------------ DEFINE THE TESTS -------------------------------------------------
        # Create the diagnostic representing the assessment for whether a person is diagnosed with diabetes
        # NB. Sensitivity/specificity is assumed to be 100%
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_diabetes=DxTest(
                property='nc_diabetes',
                item_codes=self.parameters['diabetes_hsi']['test_item_code'].astype(int)
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with hypertension:
        # blood pressure measurement
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_hypertension=DxTest(
                property='nc_hypertension'
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with back pain
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_chronic_lower_back_pain=DxTest(
                property='nc_chronic_lower_back_pain'
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with CKD
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_chronic_kidney_disease=DxTest(
                property='nc_chronic_kidney_disease',
                item_codes=self.parameters['chronic_kidney_disease_hsi']['test_item_code'].astype(int)
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with CIHD
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_chronic_ischemic_hd=DxTest(
                property='nc_chronic_ischemic_hd'
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with stroke
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_ever_stroke=DxTest(
                property='nc_ever_stroke'
            )
        )
        # Create the diagnostic representing the assessment for whether a person is diagnosed with heart attack
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_ever_heart_attack=DxTest(
                property='nc_ever_heart_attack'
            )
        )

    def build_linear_model(self, condition, interval_between_polls, lm_type):
        """
        Build a linear model for the risk of onset, removal, or death from a condition, occurrence or death from an
        event, and community-based testing for hypertension.
        :param condition: the condition or event to build the linear model for
        :param interval_between_polls: the duration (in months) between the polls
        :param lm_type: whether or not the lm is for onset, removal, death, or event in order to select the correct
        parameter set below
        :return: a linear model
        """

        # Load parameters for correct condition/event
        p = self.parameters[f'{condition}_{lm_type}']

        # For events, probability of death does not need to be standardised to poll interval since event is a discrete
        # occurrence
        if condition.startswith('ever_') and lm_type == 'death':
            baseline_annual_probability = p['baseline_annual_probability']
        else:
            baseline_annual_probability = 1 - math.exp(-interval_between_polls / 12 * p['baseline_annual_probability'])
        # LinearModel expects native python types - if it's numpy type, convert it
        baseline_annual_probability = float(baseline_annual_probability)

        linearmodel = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            baseline_annual_probability,
            Predictor('sex').when('M', p['rr_male']),
            Predictor(
                'age_years',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True
            )
            .when('.between(0, 4)', p['rr_0_4'])
            .when('.between(5, 9)', p['rr_5_9'])
            .when('.between(10, 14)', p['rr_10_14'])
            .when('.between(15, 19)', p['rr_15_19'])
            .when('.between(20, 24)', p['rr_20_24'])
            .when('.between(25, 29)', p['rr_25_29'])
            .when('.between(30, 34)', p['rr_30_34'])
            .when('.between(35, 39)', p['rr_35_39'])
            .when('.between(40, 44)', p['rr_40_44'])
            .when('.between(45, 49)', p['rr_45_49'])
            .when('.between(50, 54)', p['rr_50_54'])
            .when('.between(55, 59)', p['rr_55_59'])
            .when('.between(60, 64)', p['rr_60_64'])
            .when('.between(65, 69)', p['rr_65_69'])
            .when('.between(70, 74)', p['rr_70_74'])
            .when('.between(75, 79)', p['rr_75_79'])
            .when('.between(80, 84)', p['rr_80_84'])
            .when('.between(85, 89)', p['rr_85_89'])
            .when('.between(90, 94)', p['rr_90_94'])
            .when('.between(95, 99)', p['rr_95_99'])
            .when('>= 100', p['rr_100']),
            Predictor('li_urban').when(True, p['rr_urban']),
            Predictor(
                'li_wealth',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('1', p['rr_wealth_1'])
            .when('2', p['rr_wealth_2'])
            .when('3', p['rr_wealth_3'])
            .when('4', p['rr_wealth_4'])
            .when('5', p['rr_wealth_5']),
            Predictor(
                'li_bmi',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True
            )
            .when('1', p['rr_bmi_1'])
            .when('2', p['rr_bmi_2'])
            .when('3', p['rr_bmi_3'])
            .when('4', p['rr_bmi_4'])
            .when('5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor(
                'li_mar_stat',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True
            )
            .when('1', p['rr_marital_status_1'])
            .when('2', p['rr_marital_status_2'])
            .when('3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor(
                'li_ed_lev',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True
            )
            .when('1', p['rr_current_education_level_1'])
            .when('2', p['rr_current_education_level_2'])
            .when('3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove']),
            Predictor('nc_diabetes').when(True, p['rr_diabetes']),
            Predictor('nc_hypertension').when(True, p['rr_hypertension']),
            Predictor('de_depr').when(True, p['rr_depression']),
            Predictor('nc_chronic_kidney_disease').when(True, p['rr_chronic_kidney_disease']),
            Predictor('nc_chronic_lower_back_pain').when(True, p['rr_chronic_lower_back_pain']),
            Predictor('nc_chronic_ischemic_hd').when(True, p['rr_chronic_ischemic_heart_disease']),
            Predictor('nc_ever_stroke').when(True, p['rr_ever_stroke']),
            Predictor('nc_ever_heart_attack').when(True, p['rr_ever_heart_attack']),
            Predictor('nc_diabetes_on_medication').when(True, p['rr_diabetes_on_medication']),
            Predictor('nc_hypertension_on_medication').when(True, p['rr_hypertension_on_medication']),
            Predictor('nc_chronic_lower_back_pain_on_medication').when(True, p[
                'rr_chronic_lower_back_pain_on_medication']),
            Predictor('nc_chronic_kidney_disease_on_medication').when(True, p[
                'rr_chronic_kidney_disease_on_medication']),
            Predictor('nc_chronic_ischemic_hd_on_medication').when(True, p[
                'rr_chronic_ischemic_heart_disease_on_medication']),
            Predictor('nc_ever_stroke_on_medication').when(True, p['rr_stroke_on_medication']),
            Predictor('nc_ever_heart_attack_on_medication').when(True, p['rr_heart_attack_on_medication'])
        )

        return linearmodel

    def build_linear_model_symptoms(self, condition, interval_between_polls):
        """
        Build a linear model for the risk of symptoms from a condition.
        :param condition: the condition to build the linear model for
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """
        # Use temporary empty dict to save results
        lms_symptoms_dict = dict()
        lms_symptoms_dict[condition] = {}
        # Load parameters for correct condition
        p = self.prob_symptoms[condition]
        for symptom in p.keys():
            p_symptom_onset = 1 - math.exp(-interval_between_polls / 12 * p.get(f'{symptom}'))
            lms_symptoms_dict[condition][f'{symptom}'] = LinearModel(LinearModelType.MULTIPLICATIVE,
                                                                     p_symptom_onset, Predictor(f'nc_{condition}')
                                                                     .when(True, 1.0).otherwise(0.0))
        return lms_symptoms_dict[condition]

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        for condition in self.conditions:
            df.at[child_id, f'nc_{condition}'] = False
            df.at[child_id, f'nc_{condition}_ever_diagnosed'] = False
            df.at[child_id, f'nc_{condition}_date_diagnosis'] = pd.NaT
            df.at[child_id, f'nc_{condition}_date_last_test'] = pd.NaT
            df.at[child_id, f'nc_{condition}_on_medication'] = False
            df.at[child_id, f'nc_{condition}_medication_prevents_death'] = False
        for event in self.events:
            df.at[child_id, f'nc_{event}'] = False
            df.at[child_id, f'nc_{event}_date_last_event'] = pd.NaT
            df.at[child_id, f'nc_{event}_ever_diagnosed'] = False
            df.at[child_id, f'nc_{event}_on_medication'] = False
            df.at[child_id, f'nc_{event}_date_diagnosis'] = pd.NaT
            df.at[child_id, f'nc_{event}_scheduled_date_death'] = pd.NaT
            df.at[child_id, f'nc_{event}_medication_prevents_death'] = False
        df.at[child_id, 'nc_risk_score'] = 0

    def update_risk_score(self):
        """
        Generates or updates the risk score for individuals at initialisation of population or at each polling event
        """
        df = self.sim.population.props
        df.loc[df.is_alive, 'nc_risk_score'] = (df[[
            'li_low_ex', 'li_high_salt', 'li_high_sugar', 'li_tob', 'li_ex_alc']] > 0).sum(1)
        df.loc[df['li_bmi'] >= 3, ['nc_risk_score']] += 1

    def report_daly_values(self):
        """Report disability weight (average values for the last month) to the HealthBurden module"""

        def left_censor(obs, window_open):
            return obs.apply(lambda x: max(x, window_open) if pd.notnull(x) else pd.NaT)

        df = self.sim.population.props

        dw = pd.DataFrame(data=0.0, index=df.index[df.is_alive], columns=self.CAUSES_OF_DISABILITY.keys())

        # Diabetes: give everyone who is diagnosed or on medication uncomplicated diabetes daly weight
        dw['diabetes'].loc[df['nc_diabetes_ever_diagnosed'] | df['nc_diabetes_on_medication']] = self.daly_wts[
            'daly_diabetes_uncomplicated']
        # Diabetes: overwrite those with symptoms to have complicated diabetes daly weight
        dw['diabetes'].loc[
            self.sim.modules['SymptomManager'].who_has('diabetes_symptoms')] = self.daly_wts[
            'daly_diabetes_complicated']

        # Chronic Lower Back Pain: give those who have symptoms moderate weight
        dw['lower_back_pain'].loc[
            self.sim.modules['SymptomManager'].who_has('chronic_lower_back_pain_symptoms')] = self.daly_wts[
            'daly_chronic_lower_back_pain']

        # Chronic Kidney Disease: give those who have symptoms moderate weight
        dw['chronic_kidney_disease'].loc[
            self.sim.modules['SymptomManager'].who_has('chronic_kidney_disease_symptoms')] = self.daly_wts[
            'daly_chronic_kidney_disease_moderate']

        # Stroke: give everyone moderate long-term consequences daly weight
        dw['stroke'].loc[df.nc_ever_stroke] = self.daly_wts['daly_stroke']

        # Chronic Ischemic Heart Disease: give everyone with CIHD symptoms moderate CIHD daly weight
        dw['chronic_ischemic_hd'].loc[
            self.sim.modules['SymptomManager'].who_has('chronic_ischemic_hd_symptoms')] = self.daly_wts[
            'daly_chronic_ischemic_hd']

        # Heart Attack: first calculate proportion of month spent following heart attack, then attach weighted daly
        # Calculate fraction of the last month that was spent after having a heart attack
        days_in_last_month = (self.sim.date - (self.sim.date - DateOffset(months=1))).days
        start_heart_attack = left_censor(
            df.loc[df.is_alive, 'nc_ever_heart_attack_date_last_event'], self.sim.date - DateOffset(months=1)
        )
        dur_heart_attack_in_days = (self.sim.date - start_heart_attack).dt.days.clip(
            lower=0, upper=days_in_last_month).fillna(0.0)
        fraction_of_month_heart_attack = dur_heart_attack_in_days / days_in_last_month
        dw['heart_attack'] = fraction_of_month_heart_attack * self.daly_wts['daly_heart_attack_avg']

        return dw

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def determine_if_will_be_investigated(self, person_id):
        """
        This is called by the HSI generic first appts module whenever a person attends an appointment and determines
        if the person will be tested for a condition.
        """

        def is_next_test_due(current_date, date_of_last_test):
            return pd.isnull(date_of_last_test) or (current_date - date_of_last_test).days > 365.25 / 2

        df = self.sim.population.props
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id=person_id)

        for condition in self.conditions:
            # If the person hasn't been diagnosed and they don't have symptoms of the condition:
            if (not df.at[person_id, f'nc_{condition}_ever_diagnosed']) and (f'{condition}_symptoms' not in symptoms):
                # If the person hasn't ever been tested for the condition or not tested within last 6 months,
                # then test them with a given probability in the params for each condition
                if is_next_test_due(
                    current_date=self.sim.date, date_of_last_test=df.at[
                        person_id, f'nc_{condition}_date_last_test']
                ):
                    if self.rng.random_sample() < self.parameters[f'{condition}_hsi'].get('pr_assessed_other_symptoms'):
                        # Schedule HSI event
                        hsi_event = HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms(
                            module=self,
                            person_id=person_id,
                            condition=f'{condition}'
                        )
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            hsi_event,
                            priority=0,
                            topen=self.sim.date,
                            tclose=None
                        )
            # Else if the person hasn't been diagnosed and they do have symptoms of the condition:
            elif (not df.at[person_id, f'nc_{condition}_ever_diagnosed']) and (f'{condition}_symptoms' in symptoms):
                # Initiate HSI event
                hsi_event = HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms(
                    module=self,
                    person_id=person_id,
                    condition=f'{condition}'
                )
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event,
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

    def determine_if_will_be_investigated_events(self, person_id):
        """
        This is called by the HSI generic first appts module whenever a person attends an emergency appointment and
        determines if they will receive emergency care based on the duration of time since symptoms have appeared.
        """

        health_system = self.sim.modules["HealthSystem"]
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id=person_id)

        for ev in self.events:
            # If the person has symptoms of damage from within the last 3 days, schedule them for emergency care
            if f'{ev}_damage' in symptoms and \
                    ((self.sim.date - self.sim.population.props.at[person_id, f'nc_{ev}_date_last_event']).days <= 3):
                event = HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment(
                    module=self,
                    person_id=person_id,
                    ev=ev,
                )
                health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)


class Tracker:
    """Helper class for keeping a count with respect to some conditions and some age-groups"""

    def __init__(self, conditions: list, age_groups: list):
        self._conditions = conditions
        self._age_groups = age_groups
        self._tracker = self._make_new_tracker()

    def _make_new_tracker(self):
        return {c: {a: 0.0 for a in self._age_groups} for c in self._conditions}

    def reset(self):
        self._tracker = self._make_new_tracker()

    def add(self, condition: str, _to_add: dict):
        for _a in _to_add:
            self._tracker[condition][_a] += _to_add[_a]

    def report(self):
        return self._tracker


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   The regular event that actually changes individuals' condition or event status, occurring every 3 months
#   and synchronously for all persons.
#   Individual level events (HSI, death or cardio-metabolic events) may occur at other times.
# ---------------------------------------------------------------------------------------------------------


class CardioMetabolicDisorders_MainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """The Main Polling Event.
    * Establishes onset of each condition
    * Establishes removal of each condition
    * Schedules events that arise, according the condition.
    """

    def __init__(self, module, interval_between_polls):
        """The Main Polling Event of the CardioMetabolicDisorders Module

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=interval_between_polls))
        assert isinstance(module, CardioMetabolicDisorders)

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # Function to schedule deaths on random day throughout polling period
        def schedule_death_to_occur_before_next_poll(p_id, cond):
            self.sim.schedule_event(
                CardioMetabolicDisordersDeathEvent(self.module, p_id, cond),
                random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), m.rng)
            )

        # -------------------------------- COMMUNITY SCREENING FOR HYPERTENSION ---------------------------------------
        # A subset of individuals aged >= 50 will receive a blood pressure measurement without directly presenting to
        # the healthcare system
        eligible_population = df.is_alive & ~df['nc_hypertension_ever_diagnosed']
        will_test = self.module.lms_testing['hypertension'].predict(
            df.loc[eligible_population], rng, squeeze_single_row_output=False)
        idx_will_test = will_test[will_test].index

        # Schedule persons for community testing before next polling event
        for person_id in idx_will_test:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_CardioMetabolicDisorders_CommunityTestingForHypertension(person_id=person_id,
                                                                                       module=self.module),
                priority=1,
                topen=random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=2), m.rng),
                tclose=self.sim.date + self.frequency - pd.DateOffset(days=1)  # (to occur before next polling)
            )

        # ------------------------------ DETERMINE ONSET / REMOVAL OF CONDITIONS  -------------------------------------
        for condition in self.module.conditions:

            # Onset:
            eligible_population = df.is_alive & ~df[f'nc_{condition}']
            acquires_condition = self.module.lms_onset[condition].predict(
                df.loc[eligible_population], rng, squeeze_single_row_output=False)
            idx_acquires_condition = acquires_condition[acquires_condition].index
            df.loc[idx_acquires_condition, f'nc_{condition}'] = True

            # Add incident cases to the tracker
            self.module.trackers['onset_condition'].add(
                condition, df.loc[idx_acquires_condition].groupby('age_range').size().to_dict()
            )

            # Schedule symptom onset for both those with new onset of condition and those who already have condition,
            # among those who do not have the symptom already
            if len(self.module.lms_symptoms[condition]) > 0:
                symptom_eligible_population = df.is_alive & df[f'nc_{condition}'] & ~df.index.isin(
                    self.sim.modules['SymptomManager'].who_has(f'{condition}_symptoms'))
                symptom_onset = self.module.lms_symptoms[condition][f'{condition}_symptoms'].predict(
                    df.loc[symptom_eligible_population], rng, squeeze_single_row_output=False
                )
                idx_symptom_onset = symptom_onset[symptom_onset].index
                if idx_symptom_onset.any():
                    # Schedule symptom onset some time before next polling event
                    for symptom in self.module.prob_symptoms[condition].keys():
                        date_onset = random_date(
                            self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), m.rng)
                        self.sim.modules['SymptomManager'].change_symptom(
                            person_id=idx_symptom_onset.tolist(),
                            symptom_string=f'{symptom}',
                            add_or_remove='+',
                            date_of_onset=date_onset,
                            disease_module=self.module)

            # Removal:
            eligible_population = df.is_alive & df[f'nc_{condition}']
            if len(self.module.lms_symptoms[condition]) > 0:
                eligible_population &= ~symptom_onset
            loses_condition = self.module.lms_removal[condition].predict(
                df.loc[eligible_population], rng, squeeze_single_row_output=False)
            idx_loses_condition = loses_condition[loses_condition].index
            df.loc[idx_loses_condition, f'nc_{condition}'] = False
            df.loc[idx_loses_condition, f'nc_{condition}_on_medication'] = False
            if condition != 'hypertension':
                # Remove all symptoms of condition instantly
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=idx_loses_condition.tolist(),
                    symptom_string=f'{condition}_symptoms',
                    add_or_remove='-',
                    disease_module=self.module)

            # --------------------------- DEATH FROM CARDIO-METABOLIC CONDITION ---------------------------------------
            # There is a risk of death for those who have a cardio-metabolic condition.
            # Death is assumed to happen in the time before the next polling event.

            eligible_population = df.is_alive & df[f'nc_{condition}']
            selected_to_die = self.module.lms_death[condition].predict(df.loc[eligible_population], rng,
                                                                       squeeze_single_row_output=False)
            idx_selected_to_die = selected_to_die[selected_to_die].index

            for person_id in idx_selected_to_die:
                schedule_death_to_occur_before_next_poll(person_id, condition)

        # ---------------------------------- DETERMINE OCCURRENCE OF EVENTS  ------------------------------------------
        for event in self.module.events:

            eligible_population_for_event = df.is_alive
            has_event = self.module.lms_event_onset[event].predict(df.loc[eligible_population_for_event], rng)
            if has_event.any():  # catch in case no one has an event
                idx_has_event = has_event[has_event].index
                for person_id in idx_has_event:
                    self.sim.schedule_event(CardioMetabolicDisordersEvent(self.module, person_id, event),
                                            random_date(self.sim.date, self.sim.date + self.frequency -
                                                        pd.DateOffset(days=1), m.rng))

                    # Add cases to either incident or prevalent list of cases
                    if df.at[person_id, f'nc_{event}']:
                        self.module.trackers['prevalent_event'].add(event, {df.at[person_id, 'age_range']: 1})
                    else:
                        self.module.trackers['incident_event'].add(event, {df.at[person_id, 'age_range']: 1})


class CardioMetabolicDisordersEvent(Event, IndividualScopeEventMixin):
    """
    This is an Cardio Metabolic Disorders event (indicating an emergency occurrence of stroke or heart attack).
    It has been scheduled to occur by the CardioMetabolicDisorders_MainPollingEvent.
    :param event: the event occurring
    """

    def __init__(self, module, person_id, event):
        super().__init__(module, person_id=person_id)
        self.event = event

    def apply(self, person_id):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, f'nc_{self.event}'] = True
        df.at[person_id, f'nc_{self.event}_date_last_event'] = self.sim.date

        # Add the outward symptom to the SymptomManager. This will result in emergency care being sought for any
        # event that takes place
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            disease_module=self.module,
            add_or_remove='+',
            symptom_string=f'{self.event}_damage'
        )

        # ------------------------------------ DETERMINE OUTCOME OF THIS EVENT ----------------------------------------
        prob_death = self.module.lms_event_death[self.event].predict(df.loc[[person_id]])
        # Schedule a future death event for 7 days' time
        date_of_outcome = self.sim.date + DateOffset(days=7)
        if self.module.rng.random_sample() < prob_death[person_id]:
            df.at[person_id, f'nc_{self.event}_scheduled_date_death'] = date_of_outcome
            self.sim.schedule_event(CardioMetabolicDisordersDeathEvent(self.module, person_id,
                                                                       originating_cause=self.event), date_of_outcome)


class CardioMetabolicDisordersDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    :param originating_cause: the originating cause of the death, which can be a condition or an event
    """

    def __init__(self, module, person_id, originating_cause):
        super().__init__(module, person_id=person_id)
        self.originating_cause = originating_cause

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Check still have condition (has not resolved)
        if person[f'nc_{self.originating_cause}']:

            # Reduction in risk of death if being treated with regular medication for condition
            if person[f'nc_{self.originating_cause}_on_medication']:
                if not df.at[person_id, f'nc_{self.originating_cause}_medication_prevents_death']:
                    self.check_if_event_and_do_death(person_id)

            else:
                self.check_if_event_and_do_death(person_id)

    def check_if_event_and_do_death(self, person_id):
        """
        Helper function to perform do_death if person dies of a condition or event. If person dies of an event, this
        will only perform do_death if the scheduled date of death matches the current date (to allow for the possibility
        that treatment will intercede in an prevent death from the event).
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        # Check if it's a death event for an event (e.g. stroke) in order to execute death only if the date equals
        # scheduled date of death
        if f'{self.originating_cause}' in self.module.events:
            if self.sim.date == person[f'nc_{self.originating_cause}_scheduled_date_death']:
                self.sim.modules['Demography'].do_death(individual_id=person_id,
                                                        cause=f'{self.originating_cause}',
                                                        originating_module=self.module)
        else:
            # Conditions have no scheduled date of death, so proceed with death
            self.sim.modules['Demography'].do_death(individual_id=person_id,
                                                    cause=f'{self.originating_cause}',
                                                    originating_module=self.module)


class CardioMetabolicDisordersWeightLossEvent(Event, IndividualScopeEventMixin):
    """
    Gives an individual a probability of losing weight (via a reduction in li_bmi) and records the change in
    population.props
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return
        else:
            if self.module.rng.random_sample() < self.module.parameters['pr_bmi_reduction']:
                df.at[person_id, 'li_bmi'] -= 1
                df.at[person_id, 'nc_weight_loss_worked'] = True


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class CardioMetabolicDisorders_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date
        self.AGE_RANGE_LOOKUP = self.sim.modules['Demography'].AGE_RANGE_LOOKUP
        assert isinstance(module, CardioMetabolicDisorders)

    def apply(self, population):
        # Create shortcut to the Demography module
        demog_module = self.sim.modules['Demography']

        # Log counts in the trackers
        logger.info(key='incidence_count_by_condition',
                    data=self.module.trackers['onset_condition'].report(),
                    description=f"count of events occurring between each successive poll of logging event every "
                                f"{self.repeat} months")
        logger.info(key='incidence_count_by_incident_event',
                    data=self.module.trackers['incident_event'].report(),
                    description=f"count of incident events occurring between each successive poll of logging event "
                                f"every "f"{self.repeat} months")
        logger.info(key='incidence_count_by_prevalent_event',
                    data=self.module.trackers['prevalent_event'].report(),
                    description=f"count of prevalent events occurring between each successive poll of logging event "
                                f"every "f"{self.repeat} months")

        # Reset trackers
        for _tracker in self.module.trackers:
            self.module.trackers[_tracker].reset()

        def age_cats(ages_in_years):
            AGE_RANGE_CATEGORIES = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
            AGE_RANGE_LOOKUP = self.sim.modules['Demography'].AGE_RANGE_LOOKUP

            _age_cats = pd.Series(
                pd.Categorical(ages_in_years.map(AGE_RANGE_LOOKUP),
                               categories=AGE_RANGE_CATEGORIES, ordered=True)
            )
            return _age_cats

        # Function to prepare a groupby for logging
        def proportion_of_something_in_a_groupby_ready_for_logging(_df, something, groupbylist):
            dfx = _df.groupby(groupbylist).apply(lambda dft: pd.Series(
                {'something': dft[something].sum(), 'not_something': (~dft[something]).sum()}))
            pr = dfx['something'] / dfx.sum(axis=1)

            # create into a dict with keys as strings
            pr = pr.reset_index()
            pr['flat_index'] = ''
            for _i in range(len(pr)):
                pr.at[_i, 'flat_index'] = '__'.join([f"{_col}={pr.at[_i, _col]}" for _col in groupbylist])
            pr = pr.set_index('flat_index', drop=True)
            pr = pr.drop(columns=groupbylist)
            return pr[0].to_dict()

        # Output the person-years lived by single year of age in the past year
        df = population.props
        delta = pd.DateOffset(years=1)
        for cond in self.module.conditions:
            # mask is a Series restricting dataframe to individuals who do not have the condition, which is passed to
            # demography module to calculate person-years lived without the condition
            mask = (df.is_alive & ~df[f'nc_{cond}'])
            py = de.Demography.calc_py_lived_in_last_year(demog_module, delta, mask)
            py['age_range'] = age_cats(py.index)
            py = py.groupby('age_range').sum()
            logger.info(key=f'person_years_{cond}', data=py.to_dict())

        for event in self.module.events:
            # mask is a Series restricting dataframe to individuals who do not have the condition, which is passed to
            # demography module to calculate person-years lived without the condition
            mask = (df.is_alive & ~df[f'nc_{event}'])
            py = de.Demography.calc_py_lived_in_last_year(demog_module, delta, mask)
            py['age_range'] = age_cats(py.index)
            py = py.groupby('age_range').sum()
            logger.info(key=f'person_years_{event}', data=py.to_dict())

        # Make some summary statistics for prevalence by age/sex for each condition
        df = population.props

        # Prevalence of conditions broken down by sex and age
        for condition in self.module.conditions:
            prev_age_sex = proportion_of_something_in_a_groupby_ready_for_logging(df, f'nc_{condition}',
                                                                                  ['sex', 'age_range'])

            # Prevalence of conditions broken down by sex and age
            logger.info(
                key=f'{condition}_prevalence_by_age_and_sex',
                description='current fraction of the population classified as having condition, by sex and age',
                data={'data': prev_age_sex}
            )

            # Prevalence of conditions by adults aged 20 or older
            adult_prevalence = {
                'prevalence': len(df[df[f'nc_{condition}'] & df.is_alive & (df.age_years >= 20)]) / len(
                    df[df.is_alive & (df.age_years >= 20)])}

            logger.info(
                key=f'{condition}_prevalence',
                description='current fraction of the adult population classified as having condition',
                data=adult_prevalence
            )

            if len(df[df[f'nc_{condition}'] & df.is_alive & (df.age_years >= 20)]) > 0:
                diagnosed = {
                    f'{condition}_diagnosis_prevalence': len(df[df[f'nc_{condition}_ever_diagnosed'] & df.is_alive & (
                        df.age_years >= 20)]) / len(df[df[f'nc_{condition}'] & df.is_alive & (df.age_years >= 20)])
                }
            else:
                diagnosed = {0.0}

            logger.info(
                key=f'{condition}_diagnosis_prevalence',
                description='current fraction of the adult population diagnosed for the condition',
                data=diagnosed
            )

            if len(df[df[f'nc_{condition}'] & df.is_alive & (df.age_years >= 20)]) > 0:
                on_medication = {
                    f'{condition}_medication_prevalence': len(df[df[f'nc_{condition}_on_medication'] & df.is_alive & (
                        df.age_years >= 20)]) / len(df[df[f'nc_{condition}'] & df.is_alive & (df.age_years >= 20)])
                }
            else:
                on_medication = {0.0}

            logger.info(
                key=f'{condition}_medication_prevalence',
                description='current fraction of the adult population being on medication for the condition',
                data=on_medication
            )

        for event in self.module.events:
            prev_age_sex = proportion_of_something_in_a_groupby_ready_for_logging(df, f'nc_{event}',
                                                                                  ['sex', 'age_range'])

            # Prevalence of events broken down by sex and age
            logger.info(
                key=f'{event}_prevalence_by_age_and_sex',
                description='current fraction of the population classified as having condition, by sex and age',
                data={'data': prev_age_sex}
            )
            # Prevalence of events by adults aged 20 or older
            adult_prevalence = {
                'prevalence': len(df[df[f'nc_{event}'] & df.is_alive & (df.age_years >= 20)]) / len(
                    df[df.is_alive & (df.age_years >= 20)])}
            logger.info(
                key=f'{event}_prevalence',
                description='current fraction of the adult population classified as having event',
                data=adult_prevalence
            )

        # If param do_condition_combos = True, produce counters for number of co-morbidities by age and the combinations
        # of different conditions in the population
        if self.module.do_condition_combos:
            df.loc[df.is_alive, 'nc_n_conditions'] = df.loc[df.is_alive, self.module.condition_list].sum(axis=1)
            n_comorbidities_all = pd.DataFrame(index=self.module.age_cats,
                                               columns=list(range(0, len(self.module.condition_list) + 1)))
            df = df[['age_range', 'nc_n_conditions']]

            for num in range(0, len(self.module.condition_list) + 1):
                col = df.loc[df['nc_n_conditions'] == num].groupby(['age_range']).apply(lambda x: pd.Series(
                    {'count': x['nc_n_conditions'].count()}))
                n_comorbidities_all.loc[:, num] = col['count']

            prop_comorbidities_all = n_comorbidities_all.div(n_comorbidities_all.sum(axis=1), axis=0)

            logger.info(key='mm_prevalence_by_age_all',
                        description='annual summary of multi-morbidities by age for all',
                        data=prop_comorbidities_all.to_dict()
                        )

            # output combinations of different conditions
            df = population.props

            combos = combinations(self.module.condition_list, 2)
            condition_combos = list(combos)

            n_combos = pd.DataFrame(index=df['age_range'].value_counts().sort_index().index)

            for i in range(0, len(condition_combos)):
                df.loc[df.is_alive, 'nc_condition_combos'] = np.where(
                    df.loc[df.is_alive, f'{condition_combos[i][0]}'] &
                    df.loc[df.is_alive, f'{condition_combos[i][1]}'],
                    True, False)
                col = df.loc[df.is_alive].groupby(['age_range'])['nc_condition_combos'].count()
                n_combos.reset_index()
                n_combos.loc[:, (f'{condition_combos[i][0]}' + '_' + f'{condition_combos[i][1]}')] = col.values

            # output proportions of different combinations of conditions

            prop_combos = n_combos.div(df.groupby(['age_range'])['age_range'].count(), axis=0)
            prop_combos.index = prop_combos.index.astype(str)

            logger.info(key='prop_combos',
                        description='proportion of combinations of morbidities',
                        data=prop_combos.to_dict()
                        )

        # If param do_log_df = True, output entire dataframe for use in a logistic regression
        if self.module.do_log_df:
            df = population.props
            columns_of_interest = ['is_alive', 'sex', 'age_range', 'li_urban', 'li_wealth', 'li_bmi', 'li_low_ex',
                                   'li_high_salt', 'li_high_sugar', 'li_ex_alc', 'li_tob', 'nc_diabetes',
                                   'nc_hypertension']
            logger.info(key='df_snapshot', data=df[columns_of_interest],
                        message='dataframe of CMD variables for logistic regression')

        # Update the risk score for everyone
        self.module.update_risk_score()


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------
class HSI_CardioMetabolicDisorders_CommunityTestingForHypertension(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with any symptoms.
    This event results in a blood pressure measurement being taken that may result in diagnosis and the scheduling of
    treatment for a condition.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]

        # Ignore this event if the person is no longer alive:
        if not person.is_alive:
            return hs.get_blank_appt_footprint()

        # Run a test to diagnose whether the person has condition:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run='assess_hypertension',
            hsi_event=self
        )
        df.at[person_id, 'nc_hypertension_date_last_test'] = self.sim.date
        if dx_result:
            # Record date of diagnosis:
            df.at[person_id, 'nc_hypertension_date_diagnosis'] = self.sim.date
            df.at[person_id, 'nc_hypertension_ever_diagnosed'] = True
            # Schedule HSI_CardioMetabolicDisorders_StartWeightLossAndMedication event
            hs.schedule_hsi_event(
                hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                    module=self.module,
                    person_id=person_id,
                    condition='hypertension'
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )


class HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with any symptoms.
    This event results in a blood pressure measurement being taken that may result in diagnosis and the scheduling of
    treatment for a condition.
    :param condition: the condition being investigated
    """

    def __init__(self, module, person_id, condition):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CardioMetabolicDisorders_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        self.condition = condition

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        hs = self.sim.modules["HealthSystem"]
        m = self.module

        # Ignore this event if the person is no longer alive:
        if not person.is_alive:
            return hs.get_blank_appt_footprint()

        # Run a test to diagnose whether the person has condition:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run=f'assess_{self.condition}',
            hsi_event=self
        )
        df.at[person_id, f'nc_{self.condition}_date_last_test'] = self.sim.date
        if dx_result:
            # Record date of diagnosis:
            df.at[person_id, f'nc_{self.condition}_date_diagnosis'] = self.sim.date
            df.at[person_id, f'nc_{self.condition}_ever_diagnosed'] = True
            # Schedule HSI_CardioMetabolicDisorders_StartWeightLossAndMedication event
            hs.schedule_hsi_event(
                hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                    module=self.module,
                    person_id=person_id,
                    condition=f'{self.condition}'
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )
        # If person has at least 2 risk factors, start weight loss treatment
        elif person['nc_risk_score'] >= 2:
            if not person['nc_ever_weight_loss_treatment']:
                df.at[person_id, 'nc_ever_weight_loss_treatment'] = True
                # Schedule a post-weight loss event for 6-12 months for individual to potentially lose weight:
                self.sim.schedule_event(CardioMetabolicDisordersWeightLossEvent(self.module, person_id),
                                        random_date(self.sim.date + pd.DateOffset(months=6),
                                                    self.sim.date + pd.DateOffset(months=12),
                                                    m.rng))


class HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms(HSI_Event, IndividualScopeEventMixin):
    """
    This event is scheduled by HSI_GenericFirstApptAtFacilityLevel1 following presentation for care with the symptom
    for each condition.
    This event begins the investigation that may result in diagnosis and the scheduling of treatment.
    It is for people with the condition-relevant symptom (e.g. diabetes_symptoms).
    :param condition: the condition being investigated
    """

    def __init__(self, module, person_id, condition):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "CardioMetabolicDisorders_Investigation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        self.condition = condition

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        # Ignore this event if the person is no longer alive:
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        # If the person is already diagnosed, then take no action:
        if df.at[person_id, f'nc_{self.condition}_ever_diagnosed']:
            return hs.get_blank_appt_footprint()
        # Check that this event has been called for someone with the symptom for the condition
        if f'{self.condition}_symptoms' not in self.sim.modules['SymptomManager'].has_what(person_id):
            return hs.get_blank_appt_footprint()

        # Run a test to diagnose whether the person has condition:
        dx_result = hs.dx_manager.run_dx_test(
            dx_tests_to_run=f'assess_{self.condition}',
            hsi_event=self
        )
        df.at[person_id, f'nc_{self.condition}_date_last_test'] = self.sim.date
        if dx_result:
            # Record date of diagnosis:
            df.at[person_id, f'nc_{self.condition}_date_diagnosis'] = self.sim.date
            df.at[person_id, f'nc_{self.condition}_ever_diagnosed'] = True

            # Start weight loss treatment (except for CKD) and medication for all conditions
            hs.schedule_hsi_event(
                hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                    module=self.module,
                    person_id=person_id,
                    condition=self.condition
                ),
                priority=0,
                topen=self.sim.date,
                tclose=None
            )

        # Also run a test for hypertension according to some probability:
        if self.module.rng.rand() < self.module.parameters['hypertension_hsi']['pr_assessed_other_symptoms']:
            # Run a test to diagnose whether the person has condition:
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='assess_hypertension',
                hsi_event=self
            )
            df.at[person_id, 'nc_hypertension_date_last_test'] = self.sim.date
            if dx_result:
                # Record date of diagnosis:
                df.at[person_id, 'nc_hypertension_date_diagnosis'] = self.sim.date
                df.at[person_id, 'nc_hypertension_ever_diagnosed'] = True

                # Start weight loss treatment (except for CKD) and medication for all conditions
                hs.schedule_hsi_event(
                    hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                        module=self.module,
                        person_id=person_id,
                        condition='hypertension'
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

    def did_not_run(self):
        pass


class HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event in which a person receives a recommendation of weight loss.
    This results in an individual having a probability of reducing their BMI by one category in the next 6-12 months.
    :param condition: the condition to start medication for
    """

    def __init__(self, module, person_id, condition):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'CardioMetabolicDisorders_Prevention_WeightLoss'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        self.condition = condition

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.sim.modules['CardioMetabolicDisorders']

        # Don't advise those with CKD to lose weight, but do so for all other conditions if BMI is higher than normal
        if self.condition != 'chronic_kidney_disease' and (df.at[person_id, 'li_bmi'] > 2):
            self.sim.population.props.at[person_id, 'nc_ever_weight_loss_treatment'] = True
            # Schedule a post-weight loss event for individual to potentially lose weight in next 6-12 months:
            self.sim.schedule_event(CardioMetabolicDisordersWeightLossEvent(m, person_id),
                                    random_date(self.sim.date + pd.DateOffset(months=6),
                                                self.sim.date + pd.DateOffset(months=12), m.rng))

        # If person is already on medication, do not do anything
        if person[f'nc_{self.condition}_on_medication']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        assert person[f'nc_{self.condition}_ever_diagnosed'], "The person is not diagnosed and so should not be " \
                                                              "receiving an HSI."
        # Check availability of medication for condition
        if self.get_consumables(
            item_codes=self.module.parameters[f'{self.condition}_hsi'].get('medication_item_code').astype(int)
        ):
            # If medication is available, flag as being on medication
            df.at[person_id, f'nc_{self.condition}_on_medication'] = True
            # Determine if the medication will work to prevent death
            df.at[person_id, f'nc_{self.condition}_medication_prevents_death'] = \
                self.module.rng.rand() < self.module.parameters[f'{self.condition}_hsi'].pr_treatment_works
            # Schedule their next HSI for a refill of medication in one month
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_CardioMetabolicDisorders_Refill_Medication(person_id=person_id, module=self.module,
                                                                         condition=self.condition),
                priority=1,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(days=7)
            )
        else:
            # If person 'decides to' seek another appointment, schedule a new HSI appointment for tomorrow.
            # NB. With a probability of 1.0, this will keep occurring, and the person will never give up coming back to
            # pick up medication.
            if (m.rng.random_sample() <
                    m.parameters[f'{self.condition}_hsi'].get('pr_seeking_further_appt_if_drug_not_available')):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=self,
                    topen=self.sim.date + pd.DateOffset(days=1),
                    tclose=self.sim.date + pd.DateOffset(days=15),
                    priority=1
                )


class HSI_CardioMetabolicDisorders_Refill_Medication(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event in which a person seeks a refill prescription of medication.
    The next refill of medication is also scheduled.
    If the person is flagged as not being on medication, then the event does nothing and returns a blank footprint.
    If it does not run, then person ceases to be on medication and no further refill HSI are scheduled.
    :param condition: the condition to refill medication for
    """

    def __init__(self, module, person_id, condition):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'CardioMetabolicDisorders_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        self.condition = condition

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.sim.modules['CardioMetabolicDisorders']

        assert person[f'nc_{self.condition}_ever_diagnosed'], "The person is not diagnosed and so should not be " \
                                                              "receiving an HSI."

        # Check that the person still has the condition
        if not person[f'nc_{self.condition}']:
            # This person no longer has the condition so will not have this HSI and will drop off of medication
            # Return the blank_appt_footprint() so that this HSI does not occupy any time resources
            df.at[person_id, f'nc_{self.condition}_on_medication'] = False
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        # Check that the person is on medication
        if not person[f'nc_{self.condition}_on_medication']:
            # This person is not on medication so will not have this HSI
            # Return the blank_appt_footprint() so that this HSI does not occupy any time resources
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Check availability of medication for condition
        if self.get_consumables(
            item_codes=self.module.parameters[f'{self.condition}_hsi'].get('medication_item_code').astype(int)
        ):
            # Schedule their next HSI for a refill of medication, one month from now
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=self,
                priority=1,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(days=7)
            )
        else:
            # If medication was not available, the person ceases to be taking medication
            df.at[person_id, f'nc_{self.condition}_on_medication'] = False
            # If person 'decides to' seek another appointment, schedule a new HSI appointment for tomorrow.
            # NB. With a probability of 1.0, this will keep occurring, and the person will never give-up coming back to
            # pick-up medication.
            if (m.rng.random_sample() <
                    m.parameters[f'{self.condition}_hsi'].get('pr_seeking_further_appt_if_drug_not_available')):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=self,
                    topen=self.sim.date + pd.DateOffset(days=1),
                    tclose=self.sim.date + pd.DateOffset(days=15),
                    priority=1
                )

    def did_not_run(self):
        # If this HSI event did not run, then the persons ceases to be taking medication
        person_id = self.target
        self.sim.population.props.at[person_id, f'nc_{self.condition}_on_medication'] = False


class HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is the event when a person with the severe symptoms of chronic syndrome presents for emergency care
    and is immediately provided with treatment.
    :param ev: the event for which emergency care / treatment is sought
    """

    def __init__(self, module, person_id, ev):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CardioMetabolicDisorders)

        self.TREATMENT_ID = 'CardioMetabolicDisorders_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1, 'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'

        self.event = ev

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            key='debug',
            data=('This is HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment: '
                  f'We are now ready to treat this person {person_id}.'),
        )
        logger.debug(
            key='debug',
            data=('This is HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment: '
                  f'The squeeze-factor is {squeeze_factor}.'),
        )

        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        # Run a test to diagnose whether the person has condition:
        dx_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run=f'assess_{self.event}',
            hsi_event=self
        )
        if dx_result:
            # Record date of diagnosis
            df.at[person_id, f'nc_{self.event}_date_diagnosis'] = self.sim.date
            df.at[person_id, f'nc_{self.event}_ever_diagnosed'] = True
            if squeeze_factor < 0.5:
                # If squeeze factor is not too large:
                if self.get_consumables(
                    item_codes=self.module.parameters[f'{self.event}_hsi'].get(
                        'emergency_medication_item_code').astype(int)
                ):
                    logger.debug(key='debug', data='Treatment will be provided.')
                    df.at[person_id, f'nc_{self.event}_on_medication'] = True
                    df.at[person_id, f'nc_{self.event}_medication_prevents_death'] = \
                        self.module.rng.rand() < self.module.parameters[f'{self.event}_hsi'].pr_treatment_works
                    if df.at[person_id, f'nc_{self.event}_medication_prevents_death']:
                        # Cancel the scheduled death data
                        df.at[person_id, f'nc_{self.event}_scheduled_date_death'] = pd.NaT
                        # Remove all symptoms of event instantly
                        self.sim.modules['SymptomManager'].change_symptom(
                            person_id=person_id,
                            symptom_string=f'{self.event}_damage',
                            add_or_remove='-',
                            disease_module=self.module)
                        # Start the person on regular medication
                        self.sim.modules['HealthSystem'].schedule_hsi_event(
                            hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                                module=self.module,
                                person_id=person_id,
                                condition=self.event
                            ),
                            priority=0,
                            topen=self.sim.date,
                            tclose=None
                        )
                else:
                    # Consumables not available
                    logger.debug(key='debug', data='Treatment will not be provided due to no available consumables')

            else:
                # Squeeze factor is too large
                logger.debug(key='debug', data='Treatment will not be provided due to squeeze factor.')

        # Also run a test for hypertension according to some probability, since both events are related to hypertension:
        if self.module.rng.rand() < self.module.parameters['hypertension_hsi']['pr_assessed_other_symptoms']:
            # Run a test to diagnose whether the person has condition:
            dx_result = hs.dx_manager.run_dx_test(
                dx_tests_to_run='assess_hypertension',
                hsi_event=self
            )
            df.at[person_id, 'nc_hypertension_date_last_test'] = self.sim.date
            if dx_result:
                # Record date of diagnosis:
                df.at[person_id, 'nc_hypertension_date_diagnosis'] = self.sim.date
                df.at[person_id, 'nc_hypertension_ever_diagnosed'] = True

                # Start weight loss treatment medication for hypertension
                hs.schedule_hsi_event(
                    hsi_event=HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(
                        module=self.module,
                        person_id=person_id,
                        condition='hypertension'
                    ),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment: did not run')
