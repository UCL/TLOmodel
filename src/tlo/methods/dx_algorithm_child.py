"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with:
"""

import pandas as pd
from tlo import Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.diarrhoea import HSI_Diarrhoea_Treatment_PlanA, HSI_Diarrhoea_Treatment_PlanB, \
    HSI_Diarrhoea_Treatment_PlanC, \
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Dysentery
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
    """

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
        'ri_pneumonia_IMCI_classification':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on IMCI definitions',
                     categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
                     ),
        'ri_pneumonia_iCCM_classification':
            Property(Types.CATEGORICAL, 'Classification of pneumonia based on IMCI definitions',
                     categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
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

    def read_parameters(self, data_folder):
        p = self.parameters
        p['sensitivity_of_pneumonia_care_plan_level_0'] = 0.45
        p['sensitivity_of_pneumonia_care_plan_level_1'] = 0.5
        p['sensitivity_of_pneumonia_care_plan_level_2'] = 0.7
        p['sensitivity_of_classification_of_pneumonia_level_0'] = 0.4
        p['sensitivity_of_classification_of_pneumonia_level_1'] = 0.3
        p['sensitivity_of_classification_of_pneumonia_level_2'] = 0.6
        p['baseline_odds_classification_of_IMCI_pneumonia_level_1'] = 0.43  # prob / (1-prob) prob = 30%
        p['baseline_odds_classification_of_iCCM_pneumonia_level_0'] = 0.5
        p['baseline_odds_classification_of_IMCI_pneumonia_level_2'] = 0.5
        p['or_correct_classification_infants'] = 1.2
        p['or_correct_classification_guidelines_adherence'] = 1.4
        p['or_correct_classification_supervision'] = 1.1
        p['or_correct_classification_counted_breathes_per_minute'] = 1.2
        p['or_correct_classification_guidelines_adherence'] = 1.3
        p['or_correct_classification_supervision'] = 1.2
        p['or_correct_classification_nurse'] = 0.7
        p['baseline_odds_correct_iCCM_pneumonia_care_level_0'] = 0.5
        p['baseline_odds_correct_IMCI_pneumonia_care_level_1'] = 0.5
        p['baseline_odds_correct_IMCI_pneumonia_care_level_2'] = 0.7
        p['or_correct_pneumonia_care_nurse'] = 0.8
        p['or_correct_pneumonia_care_guidelines_adherence'] = 1.3
        p['or_correct_pneumonia_care_supervision'] = 1.2
        p['sensitivity_HSA_assessing_respiratory_rate_for_age'] = 0.81
        p['specificity_HSA_assessing_respiratory_rate_for_age'] = 0.81

        # self.assessment_and_classification_pneumonia_by_facility_level.update({
        #     'community': LinearModel(
        #         LinearModelType.LOGISTIC,
        #         p['baseline_odds_classification_of_iCCM_pneumonia_level_0'],
        #         Predictor('age_years')
        #             .when('.between(0,0)', p['or_correct_classification_infants']),
        #         Predictor('assessed_respiratory_rate' and 'respiratory_assessment_result', external=True)
        #             .when(True, p['or_correct_classification_counted_breathes_per_minute']),
        #         Predictor('adherence_to_IMCI_guidelines')
        #             .when(True, p['or_correct_classification_guidelines_adherence']),
        #         Predictor('health_worker_supervision')
        #             .when(True, p['or_correct_classification_supervision']),
        #     ),
        #     'health_centre': LinearModel(
        #         LinearModelType.LOGISTIC, p['baseline_odds_classification_of_IMCI_pneumonia_level_1'],
        #         Predictor('age_years').when('.between(0,0)', p['or_correct_classification_infants']),
        #         Predictor('tmp_provider_type').when('nurse', p['or_correct_classification_nurse']),
        #         Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_classification_guidelines_adherence']),
        #         Predictor('health_worker_supervision').when(True, p['or_correct_classification_supervision']),
        #     ),
        #     'hospital': LinearModel(
        #         LinearModelType.LOGISTIC, p['baseline_odds_classification_of_IMCI_pneumonia_level_2'],
        #         Predictor('age_years').when('.between(0,0)', p['or_correct_classification_infants']),
        #         Predictor('assessed_respiratory_rate').when(True,
        #                                                     p['or_correct_classification_counted_breathes_per_minute']),
        #         Predictor('tmp_provider_type').when('nurse', p['or_correct_classification_nurse']),
        #         Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_classification_guidelines_adherence']),
        #         Predictor('health_worker_supervision').when(True, p['or_correct_classification_supervision']),
        #     ),
        # })
        #
        # self.pneumonia_treatment_by_facility_level.update({
        #     'community': LinearModel(
        #         LinearModelType.LOGISTIC, p['baseline_odds_correct_iCCM_pneumonia_care_level_0'],
        #         Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
        #         Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
        #         Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
        #     ),
        #     'health_centre': LinearModel(
        #         LinearModelType.LOGISTIC, p['baseline_odds_correct_IMCI_pneumonia_care_level_1'],
        #         Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
        #         Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
        #         Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
        #     ),
        #     'hospital': LinearModel(
        #         LinearModelType.LOGISTIC, p['baseline_odds_correct_IMCI_pneumonia_care_level_2'],
        #         Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
        #         Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
        #         Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
        #     ),
        # })

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
                property='ri_pneumonia_iCCM_classification',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_0'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
                ),

            # test the classification of pneumonia performance at the health centre level
            classify_IMCI_pneumonia_level_1=DxTest(
                property='ri_pneumonia_IMCI_classification',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_1'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
                ),

            # test the classification of pneumonia performance at the hospital level
            classify_IMCI_pneumonia_level_2=DxTest(
                property='ri_pneumonia_IMCI_classification',
                sensitivity=p['sensitivity_of_classification_of_pneumonia_level_2'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
                ),

            # test the plan of care given for pneumonia at the community level
            pneumonia_care_given_level_0=DxTest(
                property='ri_pneumonia_iCCM_classification',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_0'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
            ),

            # test the plan of care given for pneumonia at the community level
            pneumonia_care_given_level_1=DxTest(
                property='ri_pneumonia_IMCI_classification',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_1'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
            ),

            pneumonia_care_given_level_2=DxTest(
                property='ri_pneumonia_IMCI_classification',
                sensitivity=p['sensitivity_of_pneumonia_care_plan_level_2'],
                target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
            ),
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
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #     ),
        # )

    def on_birth(self, mother_id, child_id):
        pass

    def imnci_as_gold_standard(self, person_id):
        df = self.sim.population.props
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            # FACILITY LEVEL 0 --------------------------------------------------------------------------------
            # check those that have the iCCM classification of non-severe pneumonia
            # these can be treated in the community
            if ('fast_breathing' in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])) & \
                (('chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or
                  'danger_signs') not in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
                df.at[person_id, 'ri_pneumonia_iCCM_classification'] = 'non-severe pneumonia'

            # -------------------------------------------------------------------------------------------------
            # FACILITY LEVEL 1 AND FACILITY LEVEL 2 (OUTPATIENT) ----------------------------------------------
            # check if the illness matches the IMCI classification of pneumonia
            if (('fast_breathing' or 'chest_indrawing') in list(
                df.at[person_id, 'ri_last_pneumonia_symptoms'])) & \
                (('grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') not in
                 list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
                df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'non-severe pneumonia'

            if (('cough' or 'difficult_breathing' or 'fast_breathing' or 'chest_indrawing') in
                list(df.at[person_id, 'ri_last_pneumonia_symptoms'])) & (
                ('grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') in
                list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
                df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'severe pneumonia'

            if (('cough' or 'difficult_breathing') in
                list(df.at[person_id, 'ri_last_pneumonia_symptoms'])) & (
                ('fast_breathing' or 'chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or
                 'hypoxia' or 'danger_signs') not in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
                df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'common cold'

            # -------------------------------------------------------------------------------------------------
            # FACILITY LEVEL 2 --------------------------------------------------------------------------------

    def do_when_facility_level_0(self, person_id, hsi_event):
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props

        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            self.imnci_as_gold_standard(person_id=person_id)

            # ---------------------- FOR COUGH OR DIFFICULT BREATHING -------------------------------------
            # ---------------------------------------------------------------------------------------------
            # DxTest results - classification and treatment plan -----------------------------------------------
            classification_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_iCCM_pneumonia_level_0', hsi_event=hsi_event)

            if classification_result:
                self.child_disease_management_information.update({
                    'facility_level': 0,
                    'correct_pneumonia_classification': True,
                    'classification': df.at[person_id, 'ri_pneumonia_iCCM_classification']})
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_0', hsi_event=hsi_event)

            # schedule the events following the correct classification and treatment assignment ---------
                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_iCCM_classification'] == 'non-severe pneumonia':
                    schedule_hsi(hsi_event=HSI_iCCM_Pneumonia_Treatment_level_0(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_iCCM_Pneumonia_Treatment_level_0})

                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_iCCM_classification'] == 'severe pneumonia':
                    schedule_hsi(hsi_event=HSI_iCCM_Severe_Pneumonia_Treatment_level_0(
                        person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_iCCM_Severe_Pneumonia_Treatment_level_0})

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
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1/6) & (df.at[person_id, 'age_exact_years'] < 5):

            self.imnci_as_gold_standard(person_id=person_id)

            # ---------------------- FOR COUGH OR DIFFICULT BREATHING -------------------------------------
            # ---------------------------------------------------------------------------------------------
            # DxTest results - classification and treatment plan -----------------------------------------------
            classification_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_pneumonia_level_1', hsi_event=hsi_event)

            if classification_result:
                self.child_disease_management_information.update({
                    'facility_level': 1,
                    'correct_pneumonia_classification': True,
                    'classification': df.at[person_id, 'ri_pneumonia_IMCI_classification']})
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_1', hsi_event=hsi_event)

            # # # schedule the events following the correct classification and treatment assignment # # #
                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_IMCI_classification'] == 'non-severe pneumonia':
                    schedule_hsi(hsi_event=HSI_IMCI_Pneumonia_Treatment_level_1(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Pneumonia_Treatment_level_1})

                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_IMCI_classification'] == 'severe pneumonia':
                    schedule_hsi(hsi_event=HSI_IMCI_Severe_Pneumonia_Treatment_level_1(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Severe_Pneumonia_Treatment_level_1})

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
        #     df.at[person_id, 'ri_last_pneumonia_symptoms'])) & \
        #     ((
        #          'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') not in
        #      list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
        #     df.at[person_id, 'bacterial_infection_IMNCI_classification'] = 'very severe disease'

    def do_when_facility_level_2(self, person_id, hsi_event):
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props
        # -------------------------------------------------------------------------------------------------
        # # # # # # # # # # # # FOR CHILDREN OVER 2 MONTHS AND UNDER 5 YEARS OF AGE # # # # # # # # # # # #
        # -------------------------------------------------------------------------------------------------
        if (df.at[person_id, 'age_exact_years'] >= 1 / 6) & (df.at[person_id, 'age_exact_years'] < 5):

            self.imnci_as_gold_standard(person_id=person_id)

            # ---------------------- FOR COUGH OR DIFFICULT BREATHING ---------------------------
            # -----------------------------------------------------------------------------------
            # DxTest results - classification and treatment plan -----------------------------------------------
            classification_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='classify_IMCI_pneumonia_level_2', hsi_event=hsi_event)

            if classification_result:
                self.child_disease_management_information.update({
                    'facility_level': 2,
                    'correct_pneumonia_classification': True,
                    'classification': df.at[person_id, 'ri_pneumonia_IMCI_classification']})
                care_plan_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run='pneumonia_care_given_level_2', hsi_event=hsi_event)

                # # # schedule the events following the correct classification and treatment assignment # # #
                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_IMCI_classification'] == 'non-severe pneumonia':
                    schedule_hsi(hsi_event=HSI_IMCI_Pneumonia_Treatment_level_2(person_id=person_id, module=self),
                                 priority=0,
                                 topen=self.sim.date,
                                 tclose=None
                                 )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Pneumonia_Treatment_level_2})

                if care_plan_result and df.at[
                    person_id, 'ri_pneumonia_IMCI_classification'] == 'severe pneumonia':
                    schedule_hsi(
                        hsi_event=HSI_IMCI_Severe_Pneumonia_Treatment_level_2(person_id=person_id, module=self),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                        )
                    self.child_disease_management_information.update(
                        {'correct_pneumonia_care': True, 'care_plan': HSI_IMCI_Severe_Pneumonia_Treatment_level_2})

        # if suspected of having pneumonia, measure oxygen saturation with pulse oximetry
        # if possible obtain x-ray to identify pleural effusion, empyema, pneumothorax,
        # pneumatocoele, interstitial pneumonia or pericardial effusion

    def do_when_diarrhoea(self, person_id, hsi_event):
        """
        This routine is called when Diarrhoea is reported.

        It diagnoses the condition of the child and schedules HSI Events appropriate to the condition.

        See this report https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf
        (page 3).
        NB:
            * Provisions for cholera are not included
            * The danger signs are classified collectively and are based on the result of a DxTest representing the
                ability of the clinician to correctly determine the true value of the property 'gi_current_severe_dehydration'
        """
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props
        run_dx_test = lambda test: self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run=test,
            hsi_event=hsi_event
        )
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        # Gather information that can be reported:
        # 1) Get duration of diarrhoea to date
        duration_in_days = (self.sim.date - df.at[person_id, 'gi_last_diarrhoea_date_of_onset']).days

        # 2) Get type of diarrhoea
        blood_in_stool = df.at[person_id, 'gi_last_diarrhoea_type'] == 'bloody'

        # 3) Get status of dehydration
        dehydration = 'dehydration' in symptoms

        # Gather information that cannot be reported:
        # 1) Assessment of danger signs
        danger_signs = run_dx_test('danger_signs_visual_inspection')

        # Apply the algorithms:
        # --------   Classify Extent of Dehydration   ---------
        if dehydration and danger_signs:
            # 'Severe_Dehydration'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanC(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif not dehydration and not danger_signs:
            # Treatment Plan A for uncomplicated diarrhoea
            schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanA(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif dehydration and not danger_signs: #TODO: add - and not other severe classsification
            # Treatment Plan B for some dehydration diarrhoea
            schedule_hsi(hsi_event=HSI_Diarrhoea_Treatment_PlanB(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )

        # ----------------------------------------------------

        # --------   Classify Type of Diarrhoea   -----------
        if (duration_in_days >= 14) and dehydration:
            # 'Severe_Persistent_Diarrhoea'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif (duration_in_days >= 14) and not dehydration:
            # 'Non_Severe_Persistent_Diarrhoea'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

        # --------  Classify Whether Dysentery or Not  --------
        if blood_in_stool:
            # 'Dysentery'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Dysentery(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

#  ----------------------------------------------------------------------------------------------------------
#  *********************************** HSI Events ***********************************************************
#  ----------------------------------------------------------------------------------------------------------


class HSI_iCCM_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'HSI_iCCM_Pneumonia_Treatment_level_0'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_pneumonia = {'Intervention_Package_Code': {pkg_code_pneumonia: 1}, 'Item_Code': {}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_pneumonia)

        # Currently we do not stop the event from running if consumables are unavailble
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            if df.at[person_id, 'ri_last_pneumonia_status']:
                df.at[person_id, 'ri_pneumonia_treatment'] = True
                df.at[person_id, 'ri_pneumonia_tx_start_date'] = self.sim.date
            if df.at[person_id, 'ri_last_bronchiolitis_status']:
                df.at[person_id, 'ri_bronchiolitis_treatment'] = True
                df.at[person_id, 'ri_bronchiolitis_tx_start_date'] = self.sim.date

        else:
            logger.debug('The required consumables are not available')
            # todo: prbability of referral if no drug available
            self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)


class HSI_iCCM_Severe_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'iCCM_Severe_Pneumonia_Treatment_level_0'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # first dose of antibiotic is given - give first dose of oral antibiotic
        # (amoxicillin tablet - 250mg)
        # Age 2 months up to 12 months - 1 tablet
        # Age 12 months up to 5 years - 2 tablets

        # give first dose of an appropriate antibiotic
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        item_code_amoxycilin = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_amoxycilin: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        if outcome_of_request_for_consumables['Intervention_Package_Code'][item_code_amoxycilin]:
            self.module.child_disease_management_information.update({
                'first_dose_before_referral': True
            })

        # then refer to facility level 1 or 2
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)
        # todo: which facility level is closest to be refered to?
        # todo: what about those wo are lost to follow up? - incorporate in the code


class HSI_IMCI_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'IMCI_Pneumonia_Treatment_level_1'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_pneumonia = {'Intervention_Package_Code': {pkg_code_pneumonia: 1}, 'Item_Code': {}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_pneumonia)

        # Currently we do not stop the event from running if consumables are unavailble
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            if df.at[person_id, 'ri_last_pneumonia_status']:
                df.at[person_id, 'ri_pneumonia_treatment'] = True
                df.at[person_id, 'ri_pneumonia_tx_start_date'] = self.sim.date
            # if df.at[person_id, 'ri_last_bronchiolitis_status']:
            #     df.at[person_id, 'ri_bronchiolitis_treatment'] = True
            #     df.at[person_id, 'ri_bronchiolitis_tx_start_date'] = self.sim.date

        else:
            logger.debug('The required consumables are not available')
            # todo: probability of referral if no drug available
            self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)

        # todo: If coughing for more than 2 weeks or if having recurrent wheezing, assess for TB or asthma

        # todo: follow-up in 2 days


class HSI_IMCI_Severe_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'IMCI_Severe_Pneumonia_Treatment_level_1'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # give first dose of an appropriate antibiotic
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        item_code_benzylpenicillin = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzylpenicillin 3g (5MU), PFR_each_CMST', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_benzylpenicillin: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        if outcome_of_request_for_consumables:
            self.module.child_disease_management_information.update({
                'first_dose_before_referral': True
            })

        # then refer to hospital
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_level_2(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'IMCI_Pneumonia_Treatment_level_2'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_pneumonia = {'Intervention_Package_Code': {pkg_code_pneumonia: 1}, 'Item_Code': {}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_pneumonia)

        # Currently we do not stop the event from running if consumables are unavailble
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            if df.at[person_id, 'ri_last_pneumonia_status']:
                df.at[person_id, 'ri_pneumonia_treatment'] = True
                df.at[person_id, 'ri_pneumonia_tx_start_date'] = self.sim.date
            if df.at[person_id, 'ri_last_bronchiolitis_status']:
                df.at[person_id, 'ri_bronchiolitis_treatment'] = True
                df.at[person_id, 'ri_bronchiolitis_tx_start_date'] = self.sim.date

        #todo: follow up after 3 days


class HSI_IMCI_Severe_Pneumonia_Treatment_level_2(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'IMCI_Severe_Pneumonia_Treatment_level_2'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # TODO: admit to hospital. HOW IS IT WITH NUMBER OF BEDS USED?
        # Check for oxygen saturation with pulse oxymetry if available
        # manage airway as appropriately
        # give recommended antibiotic
        # treat hgh fever if present
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

#         if oxygen_saturation == '< 90%':
#             item_code_oxygen = pd.unique(
#                 consumables.loc[
#                     consumables['Intervention_Pkg'] == 'Oxygen, 1000 liters, primarily with oxygen concentrators',
#                     'Intervention_Pkg_Code'])[0]
#             item_code_nasal_tube = pd.unique(
#                 consumables.loc[
#                     consumables['Intervention_Pkg'] == 'Tube, nasogastric CH 8_each_CMST',
#                     'Intervention_Pkg_Code'])[0]
#             consumables_needed = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_oxygen: 1,
#                                                                                  item_code_nasal_tube: 1}}
#             # check availability of consumables
#             outcome_of_request_for_oxygen = self.sim.modules['HealthSystem'].request_consumables(
#                 hsi_event=self, cons_req_as_footprint=consumables_needed)
#
#         # Antibiotic therapy
#         item_code_gentamicin = pd.unique(
#             consumables.loc[
#                 consumables['Intervention_Pkg'] == 'Gentamicin Sulphate 40mg/ml, 2ml_each_CMST',
#                 'Intervention_Pkg_Code'])[0]
#         item_code_seringe = pd.unique(
#             consumables.loc[
#                 consumables['Intervention_Pkg'] == 'Syringe, needle + swab',
#                 'Intervention_Pkg_Code'])[0]
#         item_code_cannula = pd.unique(
#             consumables.loc[
#                 consumables['Intervention_Pkg'] == 'Cannula iv (winged with injection pot) 16_each_CMST',
#                 'Intervention_Pkg_Code'])[0]
#         consumables_needed = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_gentamicin: 1,
#                                                                              item_code_seringe: 1,
#                                                                              item_code_cannula: 1}}
#
#         outcome_of_request_for_oxygen = self.sim.modules['HealthSystem'].request_consumables(
#             hsi_event=self, cons_req_as_footprint=consumables_needed)
#
# # todo: OR just have as a package deal for consumables?
