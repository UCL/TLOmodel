"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with:
"""

import pandas as pd
from tlo import Module, Parameter, Property, Types, logging, DateOffset
from tlo.methods import Metadata
from tlo.events import IndividualScopeEventMixin, RegularEvent, PopulationScopeEventMixin
from tlo.methods.diarrhoea import (
    HSI_Diarrhoea_Dysentery,
    HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea,
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea,
    HSI_Diarrhoea_Treatment_PlanA,
    HSI_Diarrhoea_Treatment_PlanB,
    HSI_Diarrhoea_Treatment_PlanC,
)
from tlo.methods.ALRI import (
    HSI_iCCM_Pneumonia_Treatment_level_0,
    HSI_iCCM_Severe_Pneumonia_Treatment_level_0,
    HSI_IMCI_No_Pneumonia_Treatment_level_1,
    HSI_IMCI_Pneumonia_Treatment_level_1,
    HSI_IMCI_Severe_Pneumonia_Treatment_level_1,
    HSI_IMCI_Pneumonia_Treatment_level_2,
    HSI_IMCI_Severe_Pneumonia_Treatment_level_2,
)
from tlo.methods.dxmanager import DxTest
from tlo.methods import ALRI
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
    """

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    PARAMETERS = {
        'sensitivity_of_assessment_of_pneumonia_level_0': Parameter
        (Types.REAL,
         'sensitivity of assessment and classification of pneumonia at facility level 0'
         ),
        'sensitivity_of_assessment_of_pneumonia_level_1': Parameter
        (Types.REAL,
         'sensitivity of assessment and classification of pneumonia at facility level 1'
         ),
        'sensitivity_of_assessment_of_pneumonia_level_2': Parameter
        (Types.REAL,
         'sensitivity of assessment and classification of pneumonia at facility level 2'
         ),
        'sensitivity_of_classification_of_pneumonia_severity_level_0': Parameter
        (Types.REAL,
         'sensitivity of classification of pneumonia severity at facility level 0'
         ),
        'sensitivity_of_classification_of_pneumonia_severity_level_1': Parameter
        (Types.REAL,
         'sensitivity of classification of pneumonia severity at facility level 1'
         ),
        'sensitivity_of_classification_of_pneumonia_severity_level_2': Parameter
        (Types.REAL,
         'sensitivity of classification of pneumonia severity at facility level 2'
         ),
        'hw_assessed_respiratory_rate': Parameter
        (Types.REAL,
         'probability_of assessing respiratory rate'
         ),

    }
    PROPERTIES = {
        'ri_iCCM_classification_as_gold':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on iCCM definitions as gold standard',
                     categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
                     ),
        'ri_health_worker_iCCM_classification':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on iCCM by health worker',
                     categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
                     ),
        'ri_IMCI_classification_as_gold':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on IMCI definitions as gold standard',
                     categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
                     ),
        'ri_health_worker_IMCI_classification':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on IMCI by health worker',
                     categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
                     ),
        'bacterial_infection_IMNCI_classification':
            Property(Types.CATEGORICAL, 'Classification of very severe disease and local bacterial infection '
                                        'based on IMNCI definitions',
                     categories=['very severe disease', 'local bacterial infection', 'unlikely']
                     ),
        'ri_imci_pneumonia_status':
            Property(Types.BOOL, 'IMCI-defined pneumonia - this includes both pneumonia and bronchiolitis'
                     ),
        'tmp_provider_type': Property(Types.CATEGORICAL,
                                      'Provider type',
                                      categories=['clinical officer', 'physician', 'nurse']),
        'adherence_to_IMCI_guidelines': Property(Types.BOOL,
                                                 'adherence to IMCI guidelines',
                                                 ),
        'health_worker_supervision': Property(Types.BOOL,
                                              'health worker supervision',
                                              ),
        'assessed_respiratory_rate': Property(Types.BOOL,
                                              'assessed respiratory rate',
                                              ),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.assessment_and_classification_pneumonia_by_facility_level = dict()
        self.pneumonia_treatment_by_facility_level = dict()
        self.child_disease_management_information = dict()

        self.severe_respiratory_symptoms = {'grunting', 'cyanosis', 'severe_respiratory_distress',
                                            'loss_of_appetite', 'hypoxia', 'chest_pain', 'danger_signs'}

    def read_parameters(self, data_folder):
        p = self.parameters

        p['sensitivity_of_classification_of_pneumonia_level_0'] = 0.4
        p['sensitivity_of_classification_of_pneumonia_level_1'] = [0.812, 0.387, 0.095]
        p['sensitivity_of_classification_of_pneumonia_level_2'] = 0.6
        p['prob_no_pneumonia_classified_as_non_severe'] = 0.1876
        p['prob_nonsev_pneumonia_classified_as_no_pneumonia'] = 0.6126
        p['prob_severe_pneumonia_classified_as_no_pneumonia'] = 0.619
        p['prob_no_pneumonia_classified_as_nonsev_pneumonia'] = 0.1876
        p['prob_no_pneumonia_classified_as_severe_pneumonia'] = 0.001
        p['prob_nonsev_pneumonia_classified_as_severe_pneumonia'] = 0.001
        p['prob_severe_pneumonia_classified_as_nonsev_pneumonia'] = 0.286
        p['sensitivity_of_pneumonia_care_plan_level_0'] = 0.45
        p['sensitivity_of_pneumonia_care_plan_level_1'] = 0.56
        p['sensitivity_of_pneumonia_care_plan_level_2'] = 0.7
        p['sensitivity_HSA_assessing_respiratory_rate_for_age'] = 0.81
        p['specificity_HSA_assessing_respiratory_rate_for_age'] = 0.81

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """
        Define the Diagnostics Tests that will be used
        """
        # ------------------ Register dx_tests for diagnosis of childhood illness ------------------
        # Register dx_tests needed for the childhood diseases HSI events. dx_tests in this module represent assessment
        # of main signs and symptoms, and the sensitivity & specificity of the assessment by the health worker at
        # each facility level, leading to the diagnosis, treatment or referral for treatment.
        p = self.parameters
        df = self.sim.population.props

        # Sensitivity of testing varies between community (level_0), health centres (level_1), and hospitals (level_2),

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            # sensitivities of the severity classification for IMCI-defined pneumonia at different facility levels
            # test the classification of pneumonia performance at the community level
            classify_iCCM_pneumonia_level_0=DxTest(
                property='ri_iCCM_classification_as_gold',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_0'],
                target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
            ),

            # test the classification of no pneumonia performance at the health centre level
            classify_IMCI_no_pneumonia_level_1=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_1'][0],
                target_categories=['common_cold']
            ),

            # test the classification of non-severe pneumonia performance at the health centre level
            classify_IMCI_pneumonia_level_1=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_1'][1],
                target_categories=['non-severe_pneumonia']
            ),

            # test the classification of pneumonia performance at the health centre level
            classify_IMCI_severe_pneumonia_level_1=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_1'][2],
                target_categories=['severe_pneumonia']
            ),

            # test the classification of pneumonia performance at the hospital level
            classify_IMCI_pneumonia_level_2=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_2'],
                target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
            ),

            # test the plan of care given for pneumonia at the community level
            pneumonia_care_given_level_0=DxTest(
                property='ri_iCCM_classification_as_gold',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_0'],
                target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
            ),

            # test the plan of care given for pneumonia at the health centres
            pneumonia_care_given_level_1=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_1'],
                target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
            ),

            # test the plan of care given for pneumonia at the hospital
            pneumonia_care_given_level_2=DxTest(
                property='ri_IMCI_classification_as_gold',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_2'],
                target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
            ),
        )

        # Test for the visual inspection of 'Danger signs' for a child who is dehydrated
        if 'Diarrhoea' in self.sim.modules:
            self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
                danger_signs_visual_inspection=DxTest(
                    property='gi_last_diarrhoea_dehydration',
                    target_categories=['severe'],
                    sensitivity=0.90,
                    specificity=0.80
                )
            )

        # self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
        #     # sensitivities of the severity classification for IMCI-defined pneumonia at different facility levels
        #     # test the classification of pneumonia performance at the community level
        #     determine_plueral_effusion_xray=DxTest(
        #         property='ri_last_pneumonia_complications',
        #         sensitivity=p['sensitivity_of_x_ray_plus_inter-observer_reliability'],
        #         target_categories=['pleural effusion', 'empyema', 'lung abscess']
        #         ),
        #
        #     # test the classification of pneumonia performance at the community level
        #     HSA_assess_respiratory_rate=DxTest(
        #         property='ri_pneumonia_iCCM_classification',
        #         sensitivity=p['sensitivity_HSA_assessing_respiratory_rate_for_age'],
        #         specificity=p['specificity_HSA_assessing_respiratory_rate_for_age'],
        #         target_categories=['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']
        #     ),
        # )

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(IMNCIManagementLoggingEvent(self), sim.date + DateOffset(years=1))

    def on_birth(self, mother_id, child_id):
        pass

    def imnci_as_gold_standard(self, person_id):
        df = self.sim.population.props
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            def imci_gold_classification_level_1(person_symptoms_list):
                severe_respiratory_symptoms_list = ['grunting', 'cyanosis', 'severe_respiratory_distress',
                                                    'loss_of_appetite', 'hypoxia', 'chest_pain', 'danger_signs']
                pneumonia_symptoms_list = ['chest_indrawing', 'fast_breathing']

                if 'cough' or 'difficult_breathing' in person_symptoms_list:
                    if any(item in severe_respiratory_symptoms_list for item in person_symptoms_list):
                        return 'severe_pneumonia'
                    if any(item in pneumonia_symptoms_list for item in person_symptoms_list):
                        return 'non-severe_pneumonia'
                    for item in person_symptoms_list:
                        if item not in (pneumonia_symptoms_list and severe_respiratory_symptoms_list):
                            return 'common_cold'

            imci_classification = imci_gold_classification_level_1(
                person_symptoms_list=self.sim.modules['ALRI'].ALRI_symptoms.get([person_id], first=True))
            if imci_classification == 'severe_pneumonia':
                df.at[person_id, 'ri_IMCI_classification_as_gold'] = 'severe_pneumonia'
            if imci_classification == 'non-severe_pneumonia':
                df.at[person_id, 'ri_IMCI_classification_as_gold'] = 'non-severe_pneumonia'
            if imci_classification == 'common_cold':
                df.at[person_id, 'ri_IMCI_classification_as_gold'] = 'common_cold'

            # # FACILITY LEVEL 0 --------------------------------------------------------------------------------
            # # check those that have the iCCM classification of non-severe_pneumonia
            # # these can be treated in the community
            # # if ('fast_breathing' in list(df.at[person_id, 'ri_current_ALRI_symptoms'])) & \
            # #     (('chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or
            # #       'danger_signs') not in list(df.at[person_id, 'ri_current_ALRI_symptoms'])):
            # #     df.at[person_id, 'ri_iCCM_classification_as_gold'] = 'non-severe_pneumonia'
            # if ('fast_breathing' in list(df.at[person_id, 'ri_current_ALRI_symptoms'])) & \
            #     (('chest_indrawing', 'grunting', 'cyanosis', 'severe_respiratory_distress', 'hypoxia',
            #       'danger_signs') not in list(df.at[person_id, 'ri_current_ALRI_symptoms'])):
            #     df.at[person_id, 'ri_iCCM_classification_as_gold'] = 'non-severe_pneumonia'
            # if (('cough' or 'difficult_breathing') in list(df.at[person_id, 'ri_current_ALRI_symptoms'])) & (
            #     ('fast_breathing' or 'chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or
            #         'hypoxia' or 'loss_of_appetite' or 'chest_pain' or
            #      'danger_signs') not in list(df.at[person_id, 'ri_current_ALRI_symptoms'])):
            #     df.at[person_id, 'ri_iCCM_classification_as_gold'] = 'common_cold'
            # if (('cough' or 'difficult_breathing' or 'fast_breathing' or 'chest_indrawing') in
            #     list(df.at[person_id, 'ri_current_ALRI_symptoms'])) & (
            #     ('grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'loss_of_appetite' or 'hypoxia' or
            #      'chest_pain' or 'danger_signs') in
            #     list(df.at[person_id, 'ri_current_ALRI_symptoms'])):
            #     df.at[person_id, 'ri_iCCM_classification_as_gold'] = 'severe_pneumonia'

        # -------------------------------------------------------------------------------------------------
        # FACILITY LEVEL 1 AND FACILITY LEVEL 2 (OUTPATIENT) ----------------------------------------------
        # -------------------------------------------------------------------------------------------------
        # FACILITY LEVEL 2 --------------------------------------------------------------------------------

    def do_when_facility_level_0(self, person_id, hsi_event):
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props
        p = self.parameters
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            self.imnci_as_gold_standard(person_id=person_id)
            pneum_management_info = dict()

            # ---------------------- FOR COUGH OR DIFFICULT BREATHING -------------------------------------
            # ---------------------------------------------------------------------------------------------
            # DxTest results - classification and treatment plan for those correctly classified -----------
            classification_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_iCCM_pneumonia_level_0', hsi_event=hsi_event)

            if classification_result:
                pneum_management_info.update({
                    'facility_level': 0,
                    'correct_pneumonia_classification': True,
                    'classification': df.at[person_id, 'ri_iCCM_classification_as_gold']})
                df.at[person_id, 'ri_health_worker_iCCM_classification'] = \
                    pneum_management_info.get('classification')
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_0', hsi_event=hsi_event)

                # schedule the events following the correct classification and treatment assignment ---------
                if care_plan_result and df.at[
                    person_id, 'ri_health_worker_iCCM_classification'] == 'non-severe_pneumonia':
                    schedule_hsi(hsi_event=HSI_iCCM_Pneumonia_Treatment_level_0(person_id=person_id,
                                                                                   module=self.sim.modules['ALRI']),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    pneum_management_info.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_iCCM_Severe_Pneumonia_Treatment_level_0})

                if care_plan_result and df.at[
                    person_id, 'ri_health_worker_iCCM_classification'] == 'severe_pneumonia':
                    schedule_hsi(hsi_event=HSI_iCCM_Severe_Pneumonia_Treatment_level_0(
                        person_id=person_id,
                        module=self.sim.modules['ALRI']),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )
                    pneum_management_info.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_iCCM_Severe_Pneumonia_Treatment_level_0})
            # ---------------------------------------------------------------------------------------------
            # If not correctly classified, determine the classification given:
            else:
                pneum_management_info.update({
                    'facility_level': 1,
                    'correct_pneumonia_classification': False})

                # Incorrect classifications for those with underlying upper respiratory infection
                if df.at[person_id, 'ri_iCCM_classification_as_gold'] == 'common_cold':
                    # IMCI no pneumonia classified as non-severe_pneumonia
                    prob_no_pneumonia_classified_as_nonsev = \
                        p['sensitivity_of_classification_of_pneumonia_level_0'][0] + \
                        p['prob_no_pneumonia_classified_as_non_severe']
                    if self.module.rng.rand() < prob_no_pneumonia_classified_as_nonsev:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'non-severe_pneumonia'
                        pneum_management_info.update({'classification': 'non-severe_pneumonia'})
                    # IMCI non-severe_pneumonia classified as severe_pneumonia
                    else:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'severe_pneumonia'
                        pneum_management_info.update({'classification': 'severe_pneumonia'})

                # Incorrect classifications for those with underlying IMCI non-severe_pneumonia
                if df.at[person_id, 'ri_iCCM_classification_as_gold'] == 'non-severe_pneumonia':
                    # IMCI non-severe_pneumonia classified as no pneumonia or common_cold
                    prob_nonsev_classified_as_no_pneumonia = \
                        p['sensitivity_of_classification_of_pneumonia_level_0'][1] + \
                        p['prob_nonsev_pneumonia_classified_as_no_pneumonia']
                    if self.module.rng.rand() < prob_nonsev_classified_as_no_pneumonia:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'common_cold'
                        pneum_management_info.update({'classification': 'common_cold'})
                    # IMCI non-severe_pneumonia classified as severe_pneumonia
                    else:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'severe_pneumonia'
                        pneum_management_info.update({'classification': 'severe_pneumonia'})

                # Incorrect classifications for those with underlying IMCI severe_pneumonia
                if df.at[person_id, 'ri_iCCM_classification_as_gold'] == 'severe_pneumonia':
                    # IMCI severe_pneumonia classified as no pneumonia or common_cold
                    prob_severe_pneumonia_classified_as_no_pneum = \
                        p['sensitivity_of_classification_of_pneumonia_level_0'][2] + \
                        p['prob_severe_pneumonia_classified_as_no_pneumonia']
                    if self.module.rng.rand() < prob_severe_pneumonia_classified_as_no_pneum:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'common_cold'
                        pneum_management_info.update({'classification': 'common_cold'})
                    # IMCI severe_pneumonia classified as non-severe
                    else:
                        df.at[person_id, 'ri_health_worker_iCCM_classification'] = 'non-severe_pneumonia'
                        pneum_management_info.update({'classification': 'non-severe_pneumonia'})

            self.child_disease_management_information.update({person_id: pneum_management_info})
            # return pneum_management_info

    def do_when_facility_level_1(self, person_id, hsi_event):
        """
        This routine is called when a sick child presents for care at a facility level 1.
        It diagnoses the condition of the child and schedules HSI Events appropriate to the condition.

        See guidelines https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf
        NB:
            * The danger signs are classified collectively and are based on the result of a DxTest representing the
                ability of the clinician to correctly determine the true value of the danger signs
        """
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props
        p = self.parameters
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if not(df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):
            return
        # ---------------------- FOR COUGH OR DIFFICULT BREATHING -----------------------------------------
        # -------------------------------------------------------------------------------------------------
        self.imnci_as_gold_standard(person_id=person_id)
        pneum_management_level1 = dict()

        # Dx_test for IMCI no pneumonia --------------------------------------------------------------------------
        if df.at[person_id, 'ri_IMCI_classification_as_gold'] == 'common_cold':
            hw_classification_for_no_pneumonia = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_no_pneumonia_level_1', hsi_event=hsi_event)

            if hw_classification_for_no_pneumonia:
                pneum_management_level1.update({
                    'facility_level': 1,
                    'correct_pneumonia_classification': True,
                    'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                    'hw_classification': df.at[person_id, 'ri_IMCI_classification_as_gold']})
                df.at[person_id, 'ri_health_worker_IMCI_classification'] = \
                    pneum_management_level1.get('classification')
                # assign treatment plan ---
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_1', hsi_event=hsi_event)
                # todo: algorithm for the probabilities of treatment plan given
                if care_plan_result:
                    schedule_hsi(hsi_event=HSI_IMCI_No_Pneumonia_Treatment_level_1(person_id=person_id,
                                                                                   module=self.sim.modules['ALRI']),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    pneum_management_level1.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_No_Pneumonia_Treatment_level_1})
            else:
                # IMCI no pneumonia classified as non-severe_pneumonia by health worker
                prob_no_pneumonia_classified_as_nonsev = p['prob_no_pneumonia_classified_as_non_severe']
                if self.rng.rand() < prob_no_pneumonia_classified_as_nonsev:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'non-severe_pneumonia'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']
                    })
                # IMCI non-severe_pneumonia classified as severe_pneumonia
                else:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'severe_pneumonia'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']
                    })
                # run a test or print

        # Dx_test for IMCI non-severe pneumonia ------------------------------------------------------------------
        if df.at[person_id, 'ri_IMCI_classification_as_gold'] == 'non-severe_pneumonia':
            hw_classification_for_pneumonia = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_pneumonia_level_1', hsi_event=hsi_event)
            if hw_classification_for_pneumonia:
                pneum_management_level1.update({
                    'facility_level': 1,
                    'correct_pneumonia_classification': True,
                    'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                    'hw_classification': df.at[person_id, 'ri_IMCI_classification_as_gold']})
                df.at[person_id, 'ri_health_worker_IMCI_classification'] = \
                    pneum_management_level1.get('hw_classification')
                # assign treatment plan ---
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_1', hsi_event=hsi_event)
                # todo: algorithm for the probabilities of treatment plan given
                if care_plan_result:
                    schedule_hsi(hsi_event=HSI_IMCI_Pneumonia_Treatment_level_1(person_id=person_id,
                                                                                module=self.sim.modules['ALRI']),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )

                    pneum_management_level1.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Pneumonia_Treatment_level_1})
            else:
                # IMCI non-severe_pneumonia classified as no pneumonia or common_cold
                prob_nonsev_classified_as_no_pneumonia = p['prob_nonsev_pneumonia_classified_as_no_pneumonia']
                if self.rng.rand() < prob_nonsev_classified_as_no_pneumonia:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'common_cold'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']})

                # IMCI non-severe_pneumonia classified as severe_pneumonia
                else:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'severe_pneumonia'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']})

        # Dx_test for IMCI non-severe pneumonia ------------------------------------------------------------------
        if df.at[person_id, 'ri_IMCI_classification_as_gold'] == 'severe_pneumonia':
            hw_classification_for_severe_pneumonia = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_pneumonia_level_1', hsi_event=hsi_event)
            if hw_classification_for_severe_pneumonia:
                pneum_management_level1.update({
                    'facility_level': 1,
                    'correct_pneumonia_classification': True,
                    'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                    'hw_classification': df.at[person_id, 'ri_IMCI_classification_as_gold']})
                df.at[person_id, 'ri_health_worker_IMCI_classification'] = \
                    pneum_management_level1.get('hw_classification')
                # assign treatment plan ---
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_1', hsi_event=hsi_event)
                # todo: algorithm for the probabilities of treatment plan given
                if care_plan_result:
                    schedule_hsi(hsi_event=HSI_IMCI_Severe_Pneumonia_Treatment_level_1(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    pneum_management_level1.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Severe_Pneumonia_Treatment_level_1})
            else:
                # IMCI severe_pneumonia classified as no pneumonia or common_cold
                prob_severe_pneumonia_classified_as_no_pneum = p['prob_severe_pneumonia_classified_as_no_pneumonia']
                if self.rng.rand() < prob_severe_pneumonia_classified_as_no_pneum:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'common_cold'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']})
                # IMCI severe_pneumonia classified as non-severe
                else:
                    df.at[person_id, 'ri_health_worker_IMCI_classification'] = 'non-severe_pneumonia'
                    pneum_management_level1.update({
                        'facility_level': 1,
                        'correct_pneumonia_classification': False,
                        'IMCI_classification': df.at[person_id, 'ri_IMCI_classification_as_gold'],
                        'hw_classification': df.at[person_id, 'ri_health_worker_IMCI_classification']})

        self.child_disease_management_information.update({person_id: pneum_management_level1})

        # todo:
        #  need to determine what happens in the HSI cascade for those not assessed or those not correctly classified.

        # TODO: probability of FOLLOW-UP CARE

        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # # # # FOR YOUNG INFANTS UNDER 2 MONTHS OF AGE # # # # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        # if df.at[person_id, 'age_exact_years'] >= 1/6:
        #     # ---------------- FOR VERY SEVERE DISEASE AND LOCAL BACTERIAL INFECTION ----------------
        #     # ---------------------------------------------------------------------------------------
        # Correct_IMNCI_diagnosis_result = self.assessment_and_classification_bacterial_infection_by_facility_level[
        #     'health_centre'].predict(df.loc[[person_id]]).values[0]
        #
        # # --------------------------------------------------------------------------------------------------------
        # # check if the illness matches the IMCI classification of pneumonia - 'gold standard'
        # if (('poor_feeding' or 'convulsions' or 'fast_breathing' or 'chest_indrawing' or 'fever' or
        #      'low_body_temperature' or 'no movement') in list(
        #     df.at[person_id, 'ri_current_ALRI_symptoms'])) & \
        #     ((
        #          'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') not in
        #      list(df.at[person_id, 'ri_current_ALRI_symptoms'])):
        #     df.at[person_id, 'bacterial_infection_IMNCI_classification'] = 'very severe disease'

    def do_when_facility_level_2(self, person_id, hsi_event):
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            self.imnci_as_gold_standard(person_id=person_id)
            pneum_management_level2 = dict()

            # ---------------------- FOR COUGH OR DIFFICULT BREATHING ---------------------------
            # -----------------------------------------------------------------------------------
            # DxTest results - classification and treatment plan -----------------------------------------------
            classification_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_pneumonia_level_2', hsi_event=hsi_event)

            if classification_result:
                pneum_management_level2.update({
                    'facility_level': 2,
                    'correct_pneumonia_classification': True,
                    'classification': df.at[person_id, 'ri_health_worker_IMCI_classification']})
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_2', hsi_event=hsi_event)

                # # # schedule the events following the correct classification and treatment assignment # # #
                if care_plan_result and df.at[
                    person_id, 'ri_health_worker_IMCI_classification'] == 'non-severe_pneumonia':
                    schedule_hsi(hsi_event=HSI_IMCI_Pneumonia_Treatment_level_2(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    pneum_management_level2.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Pneumonia_Treatment_level_2})

                if care_plan_result and df.at[
                    person_id, 'ri_health_worker_IMCI_classification'] == 'severe_pneumonia':
                    schedule_hsi(
                        hsi_event=HSI_IMCI_Severe_Pneumonia_Treatment_level_2(person_id=person_id, module=self),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )
                    pneum_management_level2.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Severe_Pneumonia_Treatment_level_2})

        # if suspected of having pneumonia, measure oxygen saturation with pulse oximetry
        # if possible obtain x-ray to identify pleural effusion, empyema, pneumothorax,
        # pneumatocoele, interstitial pneumonia or pericardial effusion

            self.child_disease_management_information.update({person_id: pneum_management_level2})

    def do_when_diarrhoea(self, person_id, hsi_event):
        """
        This routine is called when Diarrhoea is reported.
        It diagnoses the condition of the child and schedules HSI Events appropriate to the condition.
        See this report https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf
        (page 3).
        NB:
            * Provisions for cholera are not included
            * The danger signs are classified collectively and are based on the result of a DxTest representing the
              ability of the clinician to correctly determine the true value of the
              property 'gi_current_severe_dehydration'
        """
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props

        def run_dx_test(test):
            return self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=test, hsi_event=hsi_event)

        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        # Gather information that can be reported:
        # 1) Get duration of diarrhoea to date
        duration_in_days = (self.sim.date - df.at[person_id, 'gi_last_diarrhoea_date_of_onset']).days
        assert duration_in_days >= 0

        # 2) Get type of diarrhoea
        blood_in_stool = df.at[person_id, 'gi_last_diarrhoea_type'] == 'bloody'

        # 3) Get status of dehydration
        dehydration = 'dehydration' in symptoms

        # Gather information that cannot be reported:
        # 1) Assessment of danger signs
        danger_signs = run_dx_test('danger_signs_visual_inspection')

        # Apply the algorithms:
        # --------   Classify Extent of Dehydration   ---------
        if not dehydration:
            # Those who do NOT have DEHYDRATION

            # Treatment Plan A for uncomplicated diarrhoea (no dehydration and no danger signs)
            schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanA(person_id=person_id,
                                                                 module=self.sim.modules['Diarrhoea']),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )

            if duration_in_days >= 14:
                # 'Non_Severe_Persistent_Diarrhoea'
                schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(person_id=person_id,
                                                                                     module=self.sim.modules[
                                                                                         'Diarrhoea']),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )

        else:
            # Those who do have DEHYDRATION

            # Given that there is dehydration - schedule an HSI if the duration of diarrhoea has been long
            if duration_in_days >= 14:
                # 'Severe_Persistent_Diarrhoea'
                schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Persistent_Diarrhoea(person_id=person_id,
                                                                                 module=self.sim.modules['Diarrhoea']),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )

            if not danger_signs:
                # Treatment Plan B for some dehydration diarrhoea but not danger signs
                # TODO:add "...and not other severe classification from other disease modules (measles, pneumonia, etc)"
                schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanB(person_id=person_id,
                                                                     module=self.sim.modules['Diarrhoea']),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )
            else:
                # Danger sign for 'Severe_Dehydration'
                schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanC(person_id=person_id,
                                                                     module=self.sim.modules['Diarrhoea']),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )

        # --------  Classify Whether Dysentery or Not  --------
        if blood_in_stool:
            # 'Dysentery'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Dysentery(person_id=person_id, module=self.sim.modules['Diarrhoea']),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------


#  ----------------------------------------------------------------------------------------------------------
#  *********************************** HSI Events ***********************************************************
#  ----------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class IMNCIManagementLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date

    def apply(self, population):
        df = self.sim.population.props

        # log the IMCI classifications (Gold-standard)
        dict_to_output = {}
        dict_to_output.update({
            f'total_{k}': v for k, v in df.ri_health_worker_IMCI_classification.value_counts().items()
        })
        print(dict_to_output)

        logger.info(key='imci_classicications_count',
                    data=dict_to_output,
                    description='Summary of IMCI classification')

        # health worker classification -----
        health_worker_classification_count = \
            df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_health_worker_IMCI_classification').size()

        hw_df = pd.DataFrame(health_worker_classification_count)
        hw_df_transposed = hw_df.T

        logger.info(key='hw_pneumonia_classification',
                    data=hw_df_transposed,
                    description='health worker pneumonia classification')

        # IMCI pneumonia as gold standard -----
        imci_gold_classification_count = \
            df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_IMCI_classification_as_gold').size()
        imci_class_df = pd.DataFrame(imci_gold_classification_count)
        imci_class_df_transposed = imci_class_df.T

        logger.info(key='imci_gold_standard_classification',
                    data=imci_class_df_transposed,
                    description='IMCI pneumonia classification')

        # # get single row of dataframe (but not a series) ----------------
        # index_children_alri = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status]
        # individual_child_alri = df.loc[[index_children_alri[0]]]
        #
        # logger.debug(key='individual_check',
        #              data=individual_child_alri,
        #              description='following an individual child through simulation')

        # ------------- Cross tabulation ---------------
        # health worker classification for IMCI gold-standard -----
        health_worker_classification_for_imci_no_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_IMCI_classification_as_gold == 'common_cold')].groupby(
                'ri_health_worker_IMCI_classification').size()
        health_worker_classification_for_imci_nonsev_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_IMCI_classification_as_gold == 'non-severe_pneumonia'
                                                           )].groupby('ri_health_worker_IMCI_classification').size()
        health_worker_classification_for_imci_severe_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_IMCI_classification_as_gold == 'severe_pneumonia'
                                                           )].groupby('ri_health_worker_IMCI_classification').size()

        # convert into a dataframe
        hw_class_for_imci_no_pneumo_df = pd.DataFrame(health_worker_classification_for_imci_no_pneumonia)
        hw_class_for_imci_nonsev_pneumo_df = pd.DataFrame(health_worker_classification_for_imci_nonsev_pneumonia)
        hw_class_for_imci_severe_pneumo_df = pd.DataFrame(health_worker_classification_for_imci_severe_pneumonia)

        hw_class_for_imci_no_pneumo_df_transposed = hw_class_for_imci_no_pneumo_df.T
        hw_class_for_imci_nonsev_pneumo_df_tranposed = hw_class_for_imci_nonsev_pneumo_df.T
        hw_class_for_imci_severe_pneumo_df_transposed = hw_class_for_imci_severe_pneumo_df.T

        logger.info(key='hw_classification_for_common_cold_by_IMCI',
                    data=hw_class_for_imci_no_pneumo_df_transposed,
                    description='Health worker classification given for IMCI no pneumonia')

        logger.info(key='hw_classification_for_non-sev_pneumonia_by_IMCI',
                    data=hw_class_for_imci_nonsev_pneumo_df_tranposed,
                    description='Health worker classification given for IMCI pneumonia')

        logger.info(key='hw_classification_for_severe_pneumonia_by_IMCI',
                    data=hw_class_for_imci_severe_pneumo_df_transposed,
                    description='Health worker classification given for IMCI severe pneumonia')

        # -----------------------------------------------------------------------------------------------
        # underlying condition vs IMCI pneumonia classification -----------------------------------------
        imci_classification_for_underlying_viral_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'viral_pneumonia')].groupby(
                'ri_IMCI_classification_as_gold').size()
        imci_classification_for_underlying_bacterial_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'bacterial_pneumonia')
            ].groupby('ri_IMCI_classification_as_gold').size()
        imci_classification_for_underlying_fungal_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'fungal_pneumonia')
            ].groupby('ri_IMCI_classification_as_gold').size()
        imci_classification_for_underlying_bronchiolitis = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'bronchiolitis')
               ].groupby('ri_IMCI_classification_as_gold').size()

        # convert into a dataframe
        imci_classification_for_underlying_viral_pneumonia_df = \
            pd.DataFrame(imci_classification_for_underlying_viral_pneumonia)
        imci_classification_for_underlying_bacterial_pneumonia_df = \
            pd.DataFrame(imci_classification_for_underlying_bacterial_pneumonia)
        imci_classification_for_underlying_fungal_pneumonia_df = \
            pd.DataFrame(imci_classification_for_underlying_fungal_pneumonia)
        imci_classification_for_underlying_bronchiolitis_df = \
            pd.DataFrame(imci_classification_for_underlying_bronchiolitis)

        # swap rows to columns
        imci_class_for_underlying_viral_pneumonia_df_T = imci_classification_for_underlying_viral_pneumonia_df.T
        imci_class_for_underlying_bacterial_pneumonia_df_T = imci_classification_for_underlying_bacterial_pneumonia_df.T
        imci_class_for_underlying_fungal_pneumonia_df_T = imci_classification_for_underlying_fungal_pneumonia_df.T
        imci_class_for_underlying_bronchiolitis_df_T = imci_classification_for_underlying_bronchiolitis_df.T

        logger.info(key='IMCI_classification_for_underlying_viral_pneumonia',
                    data=imci_class_for_underlying_viral_pneumonia_df_T,
                    description='IMCI classification for underlying true condition - viral pneumonia')
        logger.info(key='IMCI_classification_for_underlying_bacterial_pneumonia',
                    data=imci_class_for_underlying_bacterial_pneumonia_df_T,
                    description='IMCI classification for underlying true condition - bacterial pneumonia')
        logger.info(key='IMCI_classification_for_underlying_fungal_pneumonia',
                    data=imci_class_for_underlying_fungal_pneumonia_df_T,
                    description='IMCI classification for underlying true condition - fungal pneumonia')
        logger.info(key='IMCI_classification_for_underlying_bronchiolitis',
                    data=imci_class_for_underlying_bronchiolitis_df_T,
                    description='IMCI classification for underlying true condition - bronchiolitis')

        # -----------------------------------------------------------------------------------------------
        # underlying condition vs health workers' classification -----------------------------------------
        hw_classification_for_underlying_viral_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'viral_pneumonia')].groupby(
                'ri_health_worker_IMCI_classification').size()
        hw_classification_for_underlying_bacterial_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'bacterial_pneumonia')
            ].groupby('ri_health_worker_IMCI_classification').size()
        hw_classification_for_underlying_fungal_pneumonia = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'fungal_pneumonia')
            ].groupby('ri_health_worker_IMCI_classification').size()
        hw_classification_for_underlying_bronchiolitis = \
            df[df.is_alive & df.age_years.between(0, 5) & (df.ri_ALRI_disease_type == 'bronchiolitis')
               ].groupby('ri_health_worker_IMCI_classification').size()

        # convert into a dataframe
        hw_classification_for_underlying_viral_pneumonia_df = \
            pd.DataFrame(hw_classification_for_underlying_viral_pneumonia)
        hw_classification_for_underlying_bacterial_pneumonia_df = \
            pd.DataFrame(hw_classification_for_underlying_bacterial_pneumonia)
        hw_classification_for_underlying_fungal_pneumonia_df = \
            pd.DataFrame(hw_classification_for_underlying_fungal_pneumonia)
        hw_classification_for_underlying_bronchiolitis_df = \
            pd.DataFrame(hw_classification_for_underlying_bronchiolitis)

        # swap rows to columns
        hw_class_for_underlying_viral_pneumonia_df_T = hw_classification_for_underlying_viral_pneumonia_df.T
        hw_class_for_underlying_bacterial_pneumonia_df_T = hw_classification_for_underlying_bacterial_pneumonia_df.T
        hw_class_for_underlying_fungal_pneumonia_df_T = hw_classification_for_underlying_fungal_pneumonia_df.T
        hw_class_for_underlying_bronchiolitis_df_T = hw_classification_for_underlying_bronchiolitis_df.T

        logger.info(key='hw_classification_for_underlying_viral_pneumonia',
                    data=hw_class_for_underlying_viral_pneumonia_df_T,
                    description='hw classification for underlying true condition - viral pneumonia')
        logger.info(key='hw_classification_for_underlying_bacterial_pneumonia',
                    data=hw_class_for_underlying_bacterial_pneumonia_df_T,
                    description='hw classification for underlying true condition - bacterial pneumonia')
        logger.info(key='hw_classification_for_underlying_fungal_pneumonia',
                    data=hw_class_for_underlying_fungal_pneumonia_df_T,
                    description='hw classification for underlying true condition - fungal pneumonia')
        logger.info(key='hw_classification_for_underlying_bronchiolitis',
                    data=hw_class_for_underlying_bronchiolitis_df_T,
                    description='hw classification for underlying true condition - bronchiolitis')

        # -----------------------------------------------------------------------------------------------
        # management of childhood illnesses -----------------------------------------

        # # log IMCI pneumonia management received -----------------------------------
        # management_info_flattened = \
        #     [{**{'dict_key': k}, **v} for k, v in self.module.child_disease_management_information.items()]
        # management_info_flattened_df = pd.DataFrame(management_info_flattened)
        # management_info_flattened_df.drop(columns='dict_key', inplace=True)
        #
        # # make a df with children with alri status as the columns -----
        # index_alri_status_true = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status]
        #
        # df_alri_management_info = pd.DataFrame(data=management_info_flattened_df,
        #                                        index=index_alri_status_true)
        # columns=list(management_info_flattened_df.keys())
