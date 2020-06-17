"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with:
"""

from tlo import Module, Property, Types
from tlo.methods.diarrhoea import HSI_Diarrhoea_Treatment_PlanA, HSI_Diarrhoea_Treatment_PlanB, HSI_Diarrhoea_Treatment_PlanC,\
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Dysentery


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
    """

    PARAMETERS = {}
    PROPERTIES = {
        'ri_pneumonia_IMCI_classification': Property(Types.CATEGORICAL,
                                               'Classification of pneumonia based on IMCI definitions',
                                               categories=['common cold', 'non-severe pneumonia', 'severe pneumonia']),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """
        Define the Diagnostics Tests that will be used
        """

        pass

        # ------------------ Register dx_tests for diagnosis of childhood illness ------------------
        # Register dx_tests needed for the childhood diseases HSI events. dx_tests in this module represent assessment
        # of main signs and symptoms, and the sensitivity & specificity of the assessment by the health worker at
        # each facility level, leading to the diagnosis, treatment or referral for treatment.

        # Sensitivity of testing varies between community (level_0), health centres (level_1), and hospitals (level_2),
        # p = self.parameters
        #
        # self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
        #     # test for he visual inspection of 'Danger signs'
        #     assess_general_danger_signs=DxTest(
        #         property='ri_pneumonia_status',
        #         sensitivity=p['sensitivity_of_assessment_of_pneumonia']),
        #     specificity = 0.8,
        #
        #     assess_obstructed_labour_hp=DxTest(
        #         property='la_obstructed_labour',
        #         sensitivity=p['sensitivity_of_assessment_of_obstructed_labour_hp']),
        #
        #     # Sepsis diagnosis intrapartum...
        #     # dx_tests for intrapartum and postpartum sepsis only differ in the 'property' variable
        #     assess_sepsis_hc_ip=DxTest(
        #         property='la_sepsis',
        #         sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),
        #
        #     assess_sepsis_hp_ip=DxTest(
        #         property='la_sepsis',
        #         sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),
        #
        #     # Sepsis diagnosis postpartum
        #     assess_sepsis_hc_pp=DxTest(
        #         property='la_sepsis_postpartum',
        #         sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),
        #
        #     assess_sepsis_hp_pp=DxTest(
        #         property='la_sepsis_postpartum',
        #         sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),
        #
        #     # Hypertension diagnosis
        #     assess_hypertension_hc=DxTest(
        #         property='ps_currently_hypertensive',
        #         sensitivity=p['sensitivity_of_assessment_of_hypertension_hc']),
        #
        #     assess_hypertension_hp=DxTest(
        #         property='ps_currently_hypertensive',
        #         sensitivity=p['sensitivity_of_assessment_of_hypertension_hp']),

    def on_birth(self, mother_id, child_id):
        pass

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
        run_dx_test = lambda test: self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run=test,
            hsi_event=hsi_event
        )
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        # ---------------------- FOR COUGH OR DIFFICULT BREATHING ------------------------------

        if (('fast_breathing' or 'chest_indrawing') in self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms']) & ((
            'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') not in
            self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms'](person_id)):
            df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'non-severe pneumonia'

        if (('cough' or 'difficult_breathing' or 'fast_breathing' or 'chest_indrawing') in
            self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms'](person_id)) & ((
            'grunting' or 'cyanosis' or 'severe_respiratory_distress' or 'hypoxia' or 'danger_signs') in
            self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms'](person_id)):
            df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'severe pneumonia'

        if (('cough' or 'difficult_breathing') in
            self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms'](person_id)) & (
            ('fast_breathing' or 'chest_indrawing' or 'grunting' or 'cyanosis' or 'severe_respiratory_distress' or
                'hypoxia' or 'danger_signs') not in
            self.sim.modules['Pneumonia']['ri_last_pneumonia_symptoms'](person_id)):
            df.at[person_id, 'ri_pneumonia_IMCI_classification'] = 'common cold'

        # FIRST check for general danger signs -------------------------------------------------------------------
        # if 'inability_to_drink_or_breastfeed' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'vomiting_everything' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'convulsions' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
        # if 'lethargic' in symptoms:
        #     df.at[person_id, 'at_least_one_danger_sign'] = True
