"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with:
"""

from tlo import Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.diarrhoea import HSI_Diarrhoea_Treatment_PlanA, HSI_Diarrhoea_Treatment_PlanB, \
    HSI_Diarrhoea_Treatment_PlanC, \
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Dysentery
from tlo.methods.healthsystem import HSI_Event


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

    def read_parameters(self, data_folder):
        p = self.parameters
        p['sensitivity_of_assessment_of_pneumonia_level_0'] = 0.3
        p['sensitivity_of_assessment_of_pneumonia_level_1'] = 0.5
        p['sensitivity_of_assessment_of_pneumonia_level_2'] = 0.7
        p['sensitivity_of_classification_of_pneumonia_severity_level_0'] = 0.3
        p['sensitivity_of_classification_of_pneumonia_severity_level_1'] = 0.4
        p['sensitivity_of_classification_of_pneumonia_severity_level_2'] = 0.5
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

        self.assessment_and_classification_pneumonia_by_facility_level.update({
            'community': LinearModel(
                LinearModelType.LOGISTIC,
                p['baseline_odds_classification_of_iCCM_pneumonia_level_0'],
                Predictor('age_years')
                    .when('.between(0,0)', p['or_correct_classification_infants']),
                Predictor('assessed_respiratory_rate')
                    .when(True, p['or_correct_classification_counted_breathes_per_minute']),
                Predictor('adherence_to_IMCI_guidelines')
                    .when(True, p['or_correct_classification_guidelines_adherence']),
                Predictor('health_worker_supervision')
                    .when(True, p['or_correct_classification_supervision']),
            ),
            'health_centre': LinearModel(
                LinearModelType.LOGISTIC, p['baseline_odds_classification_of_IMCI_pneumonia_level_1'],
                Predictor('age_years').when('.between(0,0)', p['or_correct_classification_infants']),
                Predictor('tmp_provider_type').when('nurse', p['or_correct_classification_nurse']),
                Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_classification_guidelines_adherence']),
                Predictor('health_worker_supervision').when(True, p['or_correct_classification_supervision']),
            ),
            'hospital': LinearModel(
                LinearModelType.LOGISTIC, p['baseline_odds_classification_of_IMCI_pneumonia_level_2'],
                Predictor('age_years').when('.between(0,0)', p['or_correct_classification_infants']),
                Predictor('assessed_respiratory_rate').when(True,
                                                            p['or_correct_classification_counted_breathes_per_minute']),
                Predictor('tmp_provider_type').when('nurse', p['or_correct_classification_nurse']),
                Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_classification_guidelines_adherence']),
                Predictor('health_worker_supervision').when(True, p['or_correct_classification_supervision']),
            ),
        })

        self.pneumonia_treatment_by_facility_level.update({
            'community': LinearModel(
                LinearModelType.LOGISTIC, p['baseline_odds_correct_iCCM_pneumonia_care_level_0'],
                Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
                Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
                Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
            ),
            'health_centre': LinearModel(
                LinearModelType.LOGISTIC, p['baseline_odds_correct_IMCI_pneumonia_care_level_1'],
                Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
                Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
                Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
            ),
            'hospital': LinearModel(
                LinearModelType.LOGISTIC, p['baseline_odds_correct_IMCI_pneumonia_care_level_2'],
                Predictor('tmp_provider_type').when('nurse', p['or_correct_pneumonia_care_nurse']),
                Predictor('adherence_to_IMCI_guidelines').when(True, p['or_correct_pneumonia_care_guidelines_adherence']),
                Predictor('health_worker_supervision').when(True, p['or_correct_pneumonia_care_supervision']),
            ),
        })

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

        # self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
        #     # sensitivities of the severity classification for IMCI-defined pneumonia at different facility levels
        #     # test the classification of pneumonia performance at the community level
        #     classify_IMCI_pneumonia_level_0=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.assessment_and_classification_pneumonia_by_facility_level['community'].predict(df),
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #         ),
        #
        #     # test the classification of pneumonia performance at the health centre level
        #     classify_IMCI_pneumonia_level_1=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.assessment_and_classification_pneumonia_by_facility_level['health_centre'],
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #         ),
        #
        #     # test the classification of pneumonia performance at the hospital level
        #     classify_IMCI_pneumonia_level_2=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.assessment_and_classification_pneumonia_by_facility_level['hospital'],
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #         ),
        #
        #     # test the plan of care given for pneumonia at the community level
        #     pneumonia_care_given_level_0=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.pneumonia_treatment_by_facility_level['community'],
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #     ),
        #
        #     # test the plan of care given for pneumonia at the community level
        #     pneumonia_care_given_level_1=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.pneumonia_treatment_by_facility_level['community'],
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #     ),
        #
        #     pneumonia_care_given_level_2=DxTest(
        #         property='ri_pneumonia_IMCI_classification',
        #         sensitivity=self.pneumonia_treatment_by_facility_level['community'],
        #         target_categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']
        #     ),
        # )
        pass

    def on_birth(self, mother_id, child_id):
        pass

    def do_when_facility_level_0(self, person_id, hsi_event):
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props

        # ---------------------- FOR COUGH OR DIFFICULT BREATHING ------------------------------------------------
        # check those that have the iCCM classification of non-severe pneumonia - 'gold standard'
        # these can be treated in the community
        if ('fast_breathing' in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])) &\
            (('chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or
              'danger_signs') not in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])):
            df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'non-severe pneumonia'

        # schedule the events following the correct classification and treatment assignment ----------------------
        correct_iCCM_pneumonia_diagnosis = self.assessment_and_classification_pneumonia_by_facility_level[
            'community'].predict(df.loc[[person_id]]).values[0]
        if correct_iCCM_pneumonia_diagnosis:
            correct_iCCM_pneumonia_care = self.pneumonia_treatment_by_facility_level['community'] \
                .predict(df.loc[[person_id]]).values[0]
            if correct_iCCM_pneumonia_care and df.at[
                person_id, 'ri_pneumonia_IMCI_classification'] == 'non-severe pneumonia':
                schedule_hsi(hsi_event=HSI_iCCM_Pneumonia_Treatment_level_0(person_id=person_id, module=self),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )
            if correct_iCCM_pneumonia_care and df.at[
                person_id, 'ri_pneumonia_IMCI_classification'] == 'severe pneumonia':
                schedule_hsi(hsi_event=HSI_iCCM_Severe_Pneumonia_Treatment_level_0(person_id=person_id, module=self),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )

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

        # FIRST check for general danger signs -------------------------------------------------------------------
        # if 'inability_to_drink_or_breastfeed' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'vomiting_everything' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'convulsions' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'lethargic' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True

        # ---------------------- FOR COUGH OR DIFFICULT BREATHING ------------------------------------------------
        # check those that have the IMCI classification of pneumonia - 'gold standard'
        if (('fast_breathing' or 'chest_indrawing') in list(df.at[person_id, 'ri_last_pneumonia_symptoms'])) &\
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

        # schedule the events following the correct classification and treatment assignment ----------------------
        correct_IMCI_diagnosis_result = self.assessment_and_classification_pneumonia_by_facility_level[
            'health_centre'].predict(df.loc[[person_id]]).values[0]
        if correct_IMCI_diagnosis_result:
            correct_IMCI_pneumonia_care = self.pneumonia_treatment_by_facility_level['health_centre']\
                .predict(df.loc[[person_id]]).values[0]
            if correct_IMCI_pneumonia_care and df.at[
                person_id, 'ri_pneumonia_IMCI_classification'] == 'non-severe pneumonia':
                schedule_hsi(hsi_event=HSI_IMCI_Pneumonia_Treatment_level_1(person_id=person_id, module=self),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )
            if correct_IMCI_pneumonia_care and df.at[
                person_id, 'ri_pneumonia_IMCI_classification'] == 'severe pneumonia':
                schedule_hsi(hsi_event=HSI_IMCI_Severe_Pneumonia_Treatment_level_1(person_id=person_id, module=self),
                             priority=0,
                             topen=self.sim.date,
                             tclose=None
                             )

        # # if correctly assessed, determine with DxTest those correctly classified with disease severity
        # dx_diagnosis_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
        #     dx_tests_to_run=f'classify_IMCI_pneumonia_{facility_level}', hsi_event=hsi_event)
        # # schedule HSI event
        # if dx_diagnosis_result:
        #     dx_care_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
        #         dx_tests_to_run=f'pneumonia_care_given_{facility_level}', hsi_event=hsi_event)
        #     if dx_care_result:
        #         self.sim.schedule_event(HSI_IMCI_Pneumonia_Treatment(self.module, person_id), self.sim.date)

        # TODO: those assessed for cough or difficult breathing, will be put down into the following categories next:
        #  common cold, non-severe pneumonia, and severe pneumonia. In this first step of assessment,
        #  children who were not assessed for cough/difficulty breathing will not proceed to the classification part
        #  Then, after assessement of symptoms the classification of the disease severity will be assigned:
        #  common cold (no pneumonia), non-severe pneumonia, and severe pneumonia

        # todo:
        #  need to determine what happens in the HSI cascade for those not assessed or those not correctly classified.

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
#  HSI events
#  ----------------------------------------------------------------------------------------------------------


class HSI_iCCM_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_treatment'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return


class HSI_iCCM_Severe_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_treatment'
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

        # then refer to facility level 1 or 2
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_treatment'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return


class HSI_IMCI_Severe_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_treatment'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 1

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

