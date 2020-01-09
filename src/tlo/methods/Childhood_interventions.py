import logging

import pandas as pd
from tlo import Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChildhoodDiseaseInterventions(Module):
    PARAMETER = {
        # Parameters for the iCCM algorithm performed by the HSA
        'prob_correct_id_danger_sign':
            Parameter(Types.REAL, 'probability of HSA correctly identified a general danger sign'
                      ),
        'prob_correct_id_fast_breathing_and_cough':
            Parameter(Types.REAL, 'probability of HSA correctly identified fast breathing for age and cough'
                      ),
        'prob_correct_id_diarrhoea_dehydration':
            Parameter(Types.REAL, 'probability of HSA correctly identified diarrhoea and dehydrayion'
                      ),
        'prob_correct_classified_diarrhoea_danger_sign':
            Parameter(Types.REAL, 'probability of HSA correctly identified diarrhoea with a danger sign'
                      ),
        'prob_correct_identified_persist_or_bloody_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA correctly identified persistent diarrhoea or dysentery'
                      ),
        'prob_correct_classified_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA correctly classified diarrhoea'
                      ),
        'prob_correct_referral_decision':
            Parameter(Types.REAL, 'probability of HSA correctly referred the case'
                      ),
        'prob_correct_treatment_advice_given':
            Parameter(Types.REAL, 'probability of HSA correctly treated and advised caretaker'
                      ),

    }
    PROPERTIES = {
        # iCCM - Integrated community case management classification, referral and treatment algorithm
        'iccm_danger_sign': Property
        (Types.BOOL, 'child has at least one iccm danger signs : '
                     'convulsions, very sleepy or unconscious, chest indrawing, vomiting everything, '
                     'not able to drink or breastfeed, red on MUAC strap, swelling of both feet, '
                     'fever for last 7 days or more, blood in stool, diarrhoea for 14 days or more, '
                     'and cough for at least 21 days'
         ),
        'ccm_correctly_identified_general_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one of the IMCI 4 general danger signs - '
                     'convulsions, lethargic or unconscious, vomiting everything, not able to drink or breastfeed'
         ),
        'ccm_correctly_identified_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one danger sign, including '
                     'convulsions, very sleepy or unconscious, chest indrawing, vomiting everything, '
                     'not able to drink or breastfeed, red on MUAC strap, swelling of both feet, '
                     'fever for last 7 days or more, blood in stool, diarrhoea for 14 days or more, '
                     'and cough for at least 21 days'
         ),
        'ccm_correctly_assessed_fast_breathing_and_cough': Property
        (Types.BOOL, 'HSA correctly assessed for fast breathing for age and cough'
         ),
        'ccm_correctly_assessed_diarrhoea_and_dehydration': Property
        (Types.BOOL, 'HSA correctly assessed diarrhoea and dehydration'
         ),
        'ccm_correctly_assessed_fever_performed_RDT': Property
        (Types.BOOL, 'HSA correctly assessed for fever and performed RDT test'
         ),
        'ccm_correctly_classified_severe_pneumonia': Property
        (Types.BOOL, 'HSA correctly classified as fast breathing with danger sign and RDT negative (severe pneumonia)'
         ),
        'ccm_correctly_classified_pneumonia': Property
        (Types.BOOL, 'HSA correctly classified as fast breathing for age, no danger sign and RDT negative (pneumonia)'
         ),
        'ccm_correctly_classified_cough_over_21days': Property
        (Types.BOOL, 'HSA correctly noted cough for more than 3 weeks'
         ),
        'ccm_correctly_classified_common_cold_or_cough': Property
        (Types.BOOL, 'HSA correctly classified as common cold or cough with no fast breathing'
         ),
        'ccm_correctly_classified_diarrhoea_with_danger_sign': Property
        (Types.BOOL, 'HSA correctly classified as diarrhoea with danger sign or with signs of severe dehydration'
         ),
        'ccm_correctly_classified_persistent_or_bloody_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified persistent diarrhoea or dysentery'
         ),
        'ccm_correctly_classified_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified diarrhoea without blood and less than 14 days'
         ),
        'ccm_correctly_classified_severe_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever with 1 or more danger sign, positive RDT (severe malaria)'
         ),
        'ccm_correctly_classified_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever for 7 days or more, no danger sign, positive RDT (malaria)'
         ),
        'ccm_correctly_classified_uncomplicated_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever for less than 7 days, no danger sign, positive RDT'
                     ' (uncomplicated malaria)'
         ),
        'ccm_correctly_classified_as_other_illness': Property
        (Types.BOOL,
         'HSA correctly classified fever, no danger sign, negative RDT (other illness)'
         ),
        'ccm_referral_options': Property
        (Types.CATEGORICAL,
         'Referral decisions', categories=['refer immediately', 'refer to health facility', 'do not refer']),
        'ccm_correct_referral_decision': Property
        (Types.BOOL,
         'HSA made the correct referral decision based on the assessment and classification process'
         ),
        'ccm_correct_treatment_and_advice_given': Property
        (Types.BOOL,
         'HSA has given the correct treatment for the classified condition'
         ),

         # IMCNI - Integrated Management of Neonatal and Childhood Illnesses algorithm
         'imci_any_general_danger_sign': Property
        (Types.BOOL,
         'any of the 4 general danger signs defined by the IMNCI guidelines'
         ),
        'imci_assessment_of_main_symptoms': Property
        (Types.CATEGORICAL,
         'main symptoms assessments', categories=['correctly assessed', 'not assessed']
         ),
        'imci_classification_of_illness': Property
        (Types.CATEGORICAL,
         'disease classification', categories=['correctly classified', 'incorrectly classified']
         ),
        'imci_treatment': Property
        (Types.CATEGORICAL,
         'treatment given', categories=['correctly treated', 'incorrectly treated', 'not treated']
         ),
        'classification_for_cough_or_difficult_breathing': Property
        (Types.CATEGORICAL,
         'classification for cough or difficult breathing',
         categories=['classified as severe pneumonia or very severe disease',
                     'classified as pneumonia', 'classified as no pneumonia']
         ),
        'correct_classification_for_cough_or_difficult_breathing': Property
        (Types.BOOL,
         'classification for cough or difficult breathing is correct'
         ),
    }

    def read_parameters(self, data_folder):
        p = self.parameters

        p['prob_correct_id_diarrhoea_dehydration'] = 0.8
        p['prob_correct_id_convulsions'] = 0.8
        p['prob_correct_id_vomits_everything'] = 0.8
        p['prob_correct_id_not_able_to_drink_or_breastfeed'] = 0.8
        p['prob_correct_id_unusually_sleepy_unconscious'] = 0.8
        p['prob_correct_id_red_MUAC'] = 0.8
        p['prob_correct_id_swelling_both_feet'] = 0.8
        p['prob_correct_id_diarrhoea'] = 0.9
        p['prob_correct_id_bloody_stools'] = 0.8
        p['prob_correct_id_persistent_diarrhoea'] = 0.8
        p['prob_correct_id_danger_sign'] = 0.7
        p['prob_correct_id_fast_breathing'] = 0.8
        p['prob_correct_id_fast_breathing_and_cough'] = 0.8
        p['prob_correct_id_chest_indrawing'] = 0.8
        p['prob_correct_id_cough_more_than_21days'] = 0.8
        p['prob_correct_id_fever_more_than_7days'] = 0.8
        p['prob_correct_id_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea_danger_sign'] = 0.8
        p['prob_correctly_classified_persist_or_bloody_diarrhoea'] = 0.8
        p['prob_correctly_classified_diarrhoea'] = 0.8
        p['prob_correct_referral_decision'] = 0.8
        p['prob_correct_treatment_advice_given'] = 0.8

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass

    def diagnose(self, person_id, hsi_event):
        """
        This will diagnose the condition of the person. It is being called from inside an HSI Event.

        :param person_id: The person is to be diagnosed
        :param hsi_event: The calling hsi_event.
        :return: a string representing the diagnosis
        """

        # get the symptoms of the person:
        symptoms = self.sim.population.props.loc[person_id, self.sim.population.props.columns.str.startswith('sy_')]

        # Make a request for consumables (making reference to the hsi_event from which this is called)
        # TODO: Finish this demonstration **

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_code_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Proteinuria test (dipstick)', 'Item_Code']
        )[0]
        consumables_needed = {
            'Intervention_Package_Code': [],
            'Item_Code': [{item_code_test: 1}],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Item_Code'][item_code_test]:
            # The neccessary diagnosis was available...

            # Example of a diangostic algorithm
            if symptoms.sum() > 2:
                diagnosis_str = 'measles'
            else:
                diagnosis_str = 'just_a_common_cold'

        else:
            # Without the diagnostic test, there cannot be a determinant diagnsosi
            diagnosis_str = 'indeterminate'

        # return the diagnosis as a string
        return diagnosis_str



class HSI_ICCM(Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event in the community for all childhood diseases modules.
    A sick child presenting symptoms is taken to the HSA for assessment, referral and treatment. """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = ['NewPneumonia']

    def apply(self, person_id):

        logger.debug('This is HSI_ICCM, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        symptoms = self.sim.population.props.loc[person_id, self.sim.population.props.columns.str.startswith('sy_')]

        # CHECKING FOR COUGH ---------------------------------------------------------------------------------
        HSA_asked_for_cough = self.module.rng.rand() < p['prob_checked_for_cough']
        if HSA_asked_for_cough:
            df.at[person_id, 'ccm_assessed_cough'] = True
        else:
            df.at[person_id, 'ccm_assessed_cough'] = False
        if HSA_asked_for_cough & df.at[person_id, 'sy_cough']:
            HSA_asked_for_duration_cough = self.module.rng.rand() < p['prob_ask_duration_of_cough']
            if HSA_asked_for_duration_cough & df.at[person_id, 'ds_cough_more_than_21days']:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            HSA_counted_breaths_per_minute = \
                self.module.rng.rand() < p['prob_correct_id_fast_breathing']
            if HSA_counted_breaths_per_minute & df.at[person_id, 'has_fast_breathing']:
                df.at[person_id, 'ccm_correctly_assessed_fast_breathing'] = True

        # CHECKING FOR DIARRHOEA -----------------------------------------------------------------------------
        HSA_asked_for_diarrhoea = self.module.rng.rand() < p['prob_checked_for_diarrhoea']
        if HSA_asked_for_diarrhoea:
            df.at[person_id, 'ccm_assessed_diarrhoea'] = True
        else:
            df.at[person_id, 'ccm_assessed_diarrhoea'] = False
        if HSA_asked_for_diarrhoea & df.at[person_id, 'sy_diarrhoea']:
        # Blood in stools
            HSA_asked_blood_in_stools = self.module.rng.rand() < p['prob_check_bloody_stools']
            if HSA_asked_blood_in_stools & df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery':
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False
                df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
        # Diarrhoea over 14 days
            HSA_asked_duration_diarrhoea = self.module.rng.rand() < p['prob_check_persistent_diarrhoea']
            if HSA_asked_duration_diarrhoea & df.at[person_id, df.gi_persistent_diarrhoea]:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False
                df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
        # Diarrhoea + danger sign
            if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign']):
                df.at[person_id, 'ccm_correctly_classified_diarrhoea_with_danger_sign'] = True
            if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                df.at[person_id, 'iccm_danger_sign' == False]):
                df.at[person_id, 'ccm_correctly_classified_diarrhoea_with_danger_sign'] = False

                # Just Diarrhoea
                if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                    df.at[person_id, 'iccm_danger_sign'] == False):
                    df.at[person_id, 'ccm_correctly_classified_diarrhoea'] = True

        # danger sign - cough for more than 21 days ----------------------------------------------------------
        if :
            cough_more_than_21days_identified_by_HSA = \
                self.module.rng.rand() < p['prob_correct_id_cough_more_than_21days']
            if cough_more_than_21days_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

        # first checking for imci general danger signs
        # danger sign - convulsions ---------------------------------------------------------------------------
        if df.at[person_id, 'ds_convulsions']:
            convulsions_identified_by_HSA = self.module.rng.rand() < p['prob_correct_id_convulsions']
            if convulsions_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False
                # df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

        # danger sign - not able to drink or breastfeed -------------------------------------------------------
        if df.at[person_id, 'ds_not_able_to_drink_or_breastfeed']:
            inability_to_drink_breastfeed_identified_by_HSA = \
                self.module.rng.rand() < p['prob_correct_id_not_able_to_drink_or_breastfeed']
            if inability_to_drink_breastfeed_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False

        # danger sign - vomits everything ---------------------------------------------------------------------
        if df.at[person_id, 'ds_vomits_everything']:
            vomiting_everything_identified_by_HSA = self.module.rng.rand() < p['prob_correct_id_vomits_everything']
            if vomiting_everything_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False

        # danger sign - unusually sleepy or unconscious -------------------------------------------------------
        if df.at[person_id, 'ds_unusually_sleepy_unconscious']:
            unusually_sleepy_unconscious_identified_by_HSA = self.module.rng.rand() < p['prob_correct_id_unusually_sleepy_unconscious']
            if unusually_sleepy_unconscious_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False

        # Then, check for other danger signs of the iccm # # # # # # # # # # # #
        # TODO: complete this section with malnutrition code
        # danger sign - for child aged 6-59 months, red on MUAC strap ----------------------------------------
        if df.at[person_id, 'ds_red_MUAC_strap']:
            red_MUAC_strap_identified_by_HSA = self.module.rng.rand() < p['prob_correct_id_red_MUAC']
            if red_MUAC_strap_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

        # danger sign - swelling of both feet ----------------------------------------------------------------
        if df.at[person_id, 'ds_swelling_both_feet']:
            swelling_both_feet_identified_by_HSA =self.module.rng.rand() < ['prob_correct_id_swelling_both_feet']
            if swelling_both_feet_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False


        # danger sign - chest indrawing ----------------------------------------------------------------------
        if df.at[person_id, 'ds_chest_indrawing']:
            chest_indrawing_identified_by_HSA = self.module.rng.rand() < p['prob_correct_id_chest_indrawing']
            if chest_indrawing_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

        # danger sign - fever for last 7 days or more --------------------------------------------------------
        # TODO: Add in malaria stuff
        if df.at[person_id, 'ds_fever_more_than_7days']:
            fever_more_than_7days_identified_by_HSA =self.module.rng.rand() < ['prob_correct_id_fever_more_than_7days']
            if fever_more_than_7days_identified_by_HSA:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
            else:
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

        # diarrhoea danger signs ----------------------------------------------------------------------------
        HSA_identified_diarrhoea = self.module.rng.rand() < p['prob_correct_id_diarrhoea']
        if HSA_identified_diarrhoea:
            # Blood in stools
            if df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery':
                HSA_identified_bloody_diarrhoea = self.module.rng.rand() < p['prob_correct_id_bloody_stools']
                if HSA_identified_bloody_diarrhoea:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                    df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
                else:
                        df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False
                        df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
            # Diarrhoea over 14 days
            if df.at[person_id, df.gi_persistent_diarrhoea]:
                HSA_identified_persistent_diarrhoea = \
                        self.module.rng.rand() < p['prob_correct_id_persistent_diarrhoea']
                if HSA_identified_persistent_diarrhoea:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                    df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
                else:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False
                    df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
            # Diarrhoea + danger sign
            if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign']):
                df.at[person_id, 'ccm_correctly_classified_diarrhoea_with_danger_sign'] = True
            if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                df.at[person_id, 'iccm_danger_sign' == False]):
                df.at[person_id, 'ccm_correctly_classified_diarrhoea_with_danger_sign'] = False

                # Just Diarrhoea
                if df.at[person_id, df.gi_diarrhoea_acute_type] == 'acute watery diarrhoea' & (
                    df.at[person_id, 'iccm_danger_sign'] == False):
                    df.at[person_id, 'ccm_correctly_classified_diarrhoea'] = True

            # Do NOT refer, treat at home
            # Checking for fast breathing and cough ----------------------------------------------------------------
            HSA_identified_fast_breathing = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_fast_breathing'],
                                                              (1 - p['prob_correct_id_fast_breathing'])])
            if HSA_identified_fast_breathing[True] & (
                df.at[person_id, 'iccm_danger_sign'] == False):
                df.at[person_id, 'ccm_correctly_classified_pneumonia'] = True
                # do not refer, treat at home

            if HSA_identified_fast_breathing[True] & df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign']:
                df.at[person_id, 'ccm_correctly_classified_severe_pneumonia'] = True
                # give first dose of treatment before assisting referral

        # give referral decision
        HSA_referral_decision = self.module.rng.rand() < p['prob_correct_referral_decision']

        HSA_referral_decision = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                          (1 - p['prob_correct_referral_decision'])])
        HSA_correct_treatment_given = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_treatment_advice_given'],
                                                          (1 - p['prob_correct_treatment_advice_given'])])
        if HSA_referral_decision[True] & HSA_correct_treatment_given[True] & \
            df.at[person_id, 'ccm_correctly_classified_severe_pneumonia']:
            'ccm_correct_treatment_and_advice_given'


class HSI_IMNCI (Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event at the first level health facilities for all
    childhood diseases modules. A sick child taken to the health centre for assessment, classification and treatment.
    Also, sick children referred by the HSA """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_IMNCI, a first appointment for person %d in the health centre',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        child_has_imci_severe_pneumonia = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                                   ((df.ri_pneumonia_severity == 'very severe pneumonia') |
                                                    (df.ri_pneumonia_severity == 'severe pneumonia'))]

        if child_has_imci_severe_pneumonia:
            assess_and_classify_severe_pneum = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correctly_classified_severe_pneumonia'],
                                                              (1 - p['prob_correctly_classified_severe_pneumonia'])])
            # this probability will be influenced by the signs and symptoms identified
            # by the health worker, and the type of health provider

            if assess_and_classify_severe_pneum[True]:
                identify_treatment_severe_pneum = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correctly_identified_treatment'],
                                                                  (1 - p['prob_correctly_identified_treatment'])])
                if identify_treatment_severe_pneum[True]:
                    # get the consumables and schedule referral
                    severe_pneumonia_start_treatment = IMNCI_Severe_Pneumonia_Treatment(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(severe_pneumonia_start_treatment,
                                                                        priority=1,
                                                                        topen=self.sim.date,
                                                                        tclose=None
                                                                        )
            if assess_and_classify_severe_pneum[False]:
                df.at[person_id, 'imci_misclassified'] = True
                misclassified_categories = ['as no pneumonia', 'as non-severe pneumonia']
                probabilities = [0.77, 0.23]
                random_choice = self.sim.rng.choice(misclassified_categories,
                                                    size=len(assess_and_classify_severe_pneum[False]), p=probabilities)
                df['imci_misclassified_pneumonia'].values[:] = random_choice

        child_has_imci_pneumonia = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                            (df.ri_pneumonia_severity == 'pneumonia')]
        if child_has_imci_pneumonia:
            assess_and_classify_pneumonia = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correctly_classified_pneumonia'],
                                                              (1 - p['prob_correctly_classified_pneumonia'])])
            if assess_and_classify_pneumonia[True]:
                identify_treatment_pneumonia = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correctly_identified_treatment'],
                                                                  (1 - p['prob_correctly_identified_treatment'])])
                if identify_treatment_pneumonia[True]:
                    # get the consumables for outpatient pneumonia
                    pneumonia_start_treatment = IMNCI_Pneumonia_Treatment(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(pneumonia_start_treatment,
                                                                        priority=1,
                                                                        topen=self.sim.date,
                                                                        tclose=None
                                                                        )
            if assess_and_classify_pneumonia[False]:
                df.at[person_id, 'imci_misclassified'] = True
                misclassified_categories = ['as no pneumonia', 'as severe pneumonia']
                probabilities = [0.98, 0.2]
                random_choice = self.sim.rng.choice(misclassified_categories,
                                                    size=len(assess_and_classify_pneumonia[False]), p=probabilities)
                df['imci_misclassified_pneumonia'].values[:] = random_choice

        child_has_imci_pneumonia = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                            (df.ri_pneumonia_severity == 'pneumonia')]


class HSI_IMNCI2 (Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event at the first level health facilities for all
    childhood diseases modules. A sick child taken to the health centre for assessment, classification and treatment.
    Also, sick children referred by the HSA """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug('This is HSI_IMNCI, a first appointment for person %d in the health centre',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        # THIS EVENT CHECKS FOR ALL SICK CHILDREN PRESENTING FOR CARE AT AN OUTPATIENT HEALTH FACILITY

        cough_or_difficult_breathing = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                                df.ALL the symptoms related to cough/difficult breathing]
        diarrhoea_present = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                                df.gi_diarrhoea_status]
        presenting_any_general_danger_signs = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                                                df.ds_convulsion | df.ds_not_able_to_drink_or_breastfeed |
                                                       df.ds_vomits_everything | df.ds_unusually_sleepy_unconscious]
        children_with_severe_pneumonia_or_very_sev_disease = \
            df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
                     (df.ri_pneumonia_severity == 'severe pneumonia') | (
                         df.ri_pneumonia_severity == 'very severe pneumonia')]

        # first for *ALL SICK CHILDREN*, health worker should check for general danger signs + the 5 main symptoms:
        # cough/difficult breathing, diarrhoea, fever, ear problem, malnutrition and anaemia (even if not reported by mother)

        # ASSESSMENT OF GENERAL DANGER SIGNS - check if each sign were a component of the consultation
        convulsions_checked_by_health_worker = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_convulsions'],
                                                            (1 - p['prob_checked_convulsions'])])
        inability_to_drink_breastfeed_checked_by_health_worker = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_not_able_to_drink_or_breastfeed'],
                                                            (1 - p['prob_checked_not_able_to_drink_or_breastfeed'])])
        vomiting_everything_checked_by_health_worker = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_vomits_everything'],
                                                            (1 - p['prob_checked_vomits_everything'])])
        # unusually_sleepy_unconscious_checked_by_health_worker = \
            # self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_unusually_sleepy_unconscious'],
                                                            #(1 - p['prob_checked_unusually_sleepy_unconscious'])])
        # Let's assume for now that checked danger sign = correct identification of the danger sign
        # health worker has identified at least one general danger sign
        imci_at_least_one_danger_sign_identified = \
            presenting_any_general_danger_signs & \
            (convulsions_checked_by_health_worker[True] | inability_to_drink_breastfeed_checked_by_health_worker[True] |
            vomiting_everything_checked_by_health_worker[True] |
            unusually_sleepy_unconscious_checked_by_health_worker[True])

        # # # # # HEALTH WORKER CHECKS FOR 5 MAIN SYMPTOMS IN IMCI # # # # #

        # ASSESSMENT OF MAIN SYMPTOMS (3) - checked for the presence of each main symptom at the consultation:
        # cough or difficult breathing, diarrhoea, and fever
        health_worker_asked_about_cough_or_difficult_breathing = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_cough_difficult_breathing'], # prob=74%
                                                          (1 - p['prob_asked_cough_difficult_breathing'])])
        health_worker_asked_about_diarrhoea = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_diarrhoea'], # prob=39%
                                                          (1 - p['prob_asked_diarrhoea'])])
        health_worker_asked_about_fever = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_fever'], # prob=77%
                                                          (1 - p['prob_asked_fever'])])
        # Assessment of all 3 main symptoms
        health_worker_assessed_all_3_main_symptoms = \
            health_worker_asked_about_cough_or_difficult_breathing & health_worker_asked_about_diarrhoea & \
            health_worker_asked_about_fever # this probability is 24% according to SPA 2013-14

        # Assessment of ear problems - ear pain or discharge
        health_worker_asked_about_ear_problem = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_ear_problem'], # 5%
                                                          (1 - p['prob_asked_ear_problem'])])
        if health_worker_asked_about_ear_problem[True]:
            HCW_looked_for_pus_draining_from_ear = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_looked_for pus_draining_from_ear'],  # 4%
                                                              (1 - p['prob_looked_for pus_draining_from_ear'])])
            HCW_felt_behind_ear_for_tenderness = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_felt_behind_ear_for_tenderness'],  # 4%
                                                              (1 - p['prob_felt_behind_ear_for_tenderness'])])

        # health care worker looked for signs of malnutrition and anaemia
        HCW_looked_for_visible_severe_wasting = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_visible_severe_wasting'],
                                                          (1 - p['prob_checked_for_visible_severe_wasting'])])
        HCW_looked_for_palmar_pallor = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_palmar_pallor'], # 24%
                                                          (1 - p['prob_checked_for_palmar_pallor'])])
        HCW_looked_for_oedema_of_both_feet = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_oedema_of_both_feet'], # 8%
                                                          (1 - p['prob_checked_for_oedema_of_both_feet'])])
        HCW_determined_weight_for_age = \
            self.sim.rng.choice([True, False], size=1, p=[p['prob_determined_weight_for_age'],
                                                          (1 - p['prob_determined_weight_for_age'])])

        # ASSESSMENT PROCESS OF COUGH AND/OR DIFFICULT BREATHING
        # there are 4 elements to complete in the assessment of cough or difficult breathing: the duration of illness,
        # the count of breaths per minute, to look for chest indrawing, and to listen for stridor in a calm child
        # in the algorithm below, the probabilities refer to whether the health worker has completed and correctly
        # assessed for each element
        if cough_or_difficult_breathing & health_worker_asked_about_cough_or_difficult_breathing[True]:
            health_worker_asked_duration_of_illness = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_duration_cough_difficult_breathing'],
                                                              (1 - p['prob_asked_duration_cough_difficult_breathing'])])
            # if health_worker_asked_duration_of_illness[True] & df.at[person_id, has asthma or tb or whooping cough] or longer than 30 days:
            # health_worker_referral_or_hospitalization
            health_worker_counted_breaths_per_minute = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_assessed_fast_breathing'], # 16%
                                                              (1 - p['prob_assessed_fast_breathing'])])
            health_worker_checked_chest_indrawing = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_chest_indrawing'],
                                                              (1 - p['prob_assessed_chest_indrawing'])])
            health_worker_checked_stridor = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_stridor'],
                                                              (1 - p['prob_assessed_stridor'])])
            if health_worker_asked_duration_of_illness[True] & \
                health_worker_counted_breaths_per_minute[True] \
                & health_worker_checked_chest_indrawing[True] & \
                health_worker_checked_stridor[True]:
                df.at[person_id, df.imci_all_elements_of_cough_or_difficult_breathing_complete] = True

            # next, need to consider the sensitivity and specificity of assessing these signs
            # for those checked for cough/difficult breathing assessment components, multiply by the sensitivity of the assssement
            hw_correctly_assessed_fast_breathing = health_worker_counted_breaths_per_minute[True] * sensitivity and specificity

            hw_correctly_assessed_chest_indrawing = health_worker_checked_chest_indrawing[True] * sensitivity and specificity

            hw_correctly_assessed_stridor = health_worker_checked_stridor[True] * sensitivity and specificity

            # CLASSIFICATION PROCESS
            if children_with_severe_pneumonia_or_very_sev_disease & \
                (hw_correctly_assessed_chest_indrawing[True] | hw_correctly_assessed_stridor[True] |
                 imci_at_least_one_danger_sign_identified):
                hw_correctly_classified_severe_pneumonia_or_very_sev_disease = \
                    self.sim.rng.choice([True, False], size=1,
                                        p=[p['prob_correctly_classified_severe_pneumonia'],  # high probability
                                           (1 - p['prob_correctly_classified_severe_pneumonia'])])


                if hw_correctly_classified_severe_pneumonia_or_very_sev_disease[True]:
                    df.at[person_id, df.correct_classification_for_cough_or_difficult_breathing] = True
                    df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as severe pneumonia or very severe disease'

                if hw_correctly_classified_severe_pneumonia_or_very_sev_disease[False]: # severe pneumonia/disease not correctly classified
                    df.at[person_id, df.correct_classification_for_cough_or_difficult_breathing] = False
                    df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as pneumonia' # 16.5%
                    df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as no pneumonia'  # 54.4%

            if hw_correctly_assessed_fast_breathing[True] &


            if df.at[cases, df.imci_any_general_danger_sign | df.chest_indrawing | df.stridor ]:

                if health_worker_counted_breath_per_minute[True]:

        # ASSESSMENT OF DIARRHOEA
        if diarrhoea_present & health_worker_asked_about_cough_or_difficult_breathing[True]:
            HCW_asked_duration_of_illness = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_duration_diarrhoea'],
                                                              (1 - p['prob_asked_duration_diarrhoea'])])
            HCW_asked_blood_in_stool = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_presence_of_blood_in_stool'],
                                                              (1 - p['prob_asked_presence_of_blood_in_stool'])])
            HCW_checked_lethargic_or_restless = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_lethargic_or_restless'], # this was a danger sign to be checked
                                                              (1 - p['prob_checked_lethargic_or_restless'])])
            HCW_looked_for_sunken_eyes = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_looked_for_sunken_eyes'],
                                                              (1 - p['prob_looked_for_sunken_eyes'])])
            HCW_checked_ability_to_drink = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_ability_to_drink'],
                                                              (1 - p['prob_checked_ability_to_drink'])])
            HCW_pinched_abdomen = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_pinched_abdomen'], # 10%
                                                              (1 - p['prob_pinched_abdomen'])])


class IMNCI_Severe_Pneumonia_Treatment(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                              'Treatment of severe pneumonia', 'Intervention_Pkg_Code'])[138]
        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }
        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint # self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug(
            '....IMNCI_VerySevere_Pneumonia_Treatment: giving treatment for %d with very severe pneumonia',
            person_id)

        # schedule referral event
        imci_severe_pneumonia_referral = Referral_Severe_Pneumonia_Treatment(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(imci_severe_pneumonia_referral,
                                                            priority=1,
                                                            topen=self.sim.date,
                                                            tclose=None
                                                            )


class Referral_Severe_Pneumonia_Treatment(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                              'Treatment of severe pneumonia', 'Intervention_Pkg_Code'])[138]
        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }
        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint # self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []


class IMNCI_Pneumonia_Treatment(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                              'Treatment of severe pneumonia', 'Intervention_Pkg_Code'])[138]
        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }
        # Define the necessary information for an HSI
        # self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint  # self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []


''' 
     # # # # # ASSESS GENERAL DANGER SIGNS # # # # #
        if df.at[person_id, 'imci_any_general_danger_sign' == True]:
            if df.at[person_id, 'ds_convulsions']:
                convulsions_identified_imci = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_convulsions'],
                                                                  (1 - p['prob_correct_id_convulsions'])])
                if convulsions_identified_imci[True]:
                    df.at[person_id, 'imci_correctly_identified_general_danger_signs'] = True
            if df.at[person_id, 'ds_not_able_to_drink_or_breastfeed']:
                inability_to_drink_breastfeed_identified_imci = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_not_able_to_drink_or_breastfeed'],
                                                                  (1 - p['prob_correct_id_not_able_to_drink_or_breastfeed'])])
                if inability_to_drink_breastfeed_identified_imci[True]:
                    df.at[person_id, 'imci_correctly_identified_general_danger_signs'] = True
            if df.at[person_id, 'ds_vomits_everything']:
                vomits_everything_identified_imci = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_vomits_everything'],
                                                                  (1 - p['prob_correct_id_vomits_everything'])])
                if vomits_everything_identified_imci[True]:
                    df.at[person_id, 'imci_correctly_identified_general_danger_signs'] = True
            if df.at[person_id, 'ds_unusually_sleepy_unconscious']:
                unusually_sleepy_unconscious_identified_imci = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_unusually_sleepy_unconscious'],
                                                                  (1 - p['prob_correct_id_unusually_sleepy_unconscious'])])
                if unusually_sleepy_unconscious_identified_imci[True]:
                    df.at[person_id, 'imci_correctly_identified_general_danger_signs'] = True
                if convulsions_identified_imci[False] & inability_to_drink_breastfeed_identified_imci[False] &\
                    vomits_everything_identified_imci[False] & unusually_sleepy_unconscious_identified_imci[False]:
                    df.at[person_id, 'imci_correctly_identified_general_danger_signs'] = False           
        # # # # # ASSESS COUGH OR DIFFICULT BREATHING # # # # #
        if df.at[person_id, 'cough' | 'difficult_breathing' | 'fast_breathing']: # any respiratory symptoms
            if df.at[person_id, 'pn_chest_indrawing']:
                chest_indrawing_identified_imci =\
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_chest_indrawing'],
                                                                  (1 - p['prob_correct_id_chest_indrawing'])])
                if chest_indrawing_identified_imci[True]:
                    correctly_classified_severe_pneumonia = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_classification_severe_pneum'],
                                                                      (1 - p['prob_correct_classification_severe_pneum'])])
                    if correctly_classified_severe_pneumonia[True]:
                        df.at[person_id, 'imci_correctly_classified_severe_pneumonia'] = True

            if df.at[person_id, 'pn_fast_breathing']:
                fast_breathing_identified_imci = self.sim.rng.choice([True, False], size=1,
                                                                     p=[p['prob_correct_id_fast_breathing'],
                                                                        (1 - p['prob_correct_id_fast_breathing'])])
                if fast_breathing_identified_imci[True] & df.at['imci_any_general_danger_sign' == False]:
                    correctly_classified_pneumonia = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_classification_pneumonia'],
                                                                      (1 - p['prob_correct_classification_pneumonia'])])
                    if correctly_classified_pneumonia[True]:
                        df.at[person_id, 'imci_correctly_classified_pneumonia'] = True
            if df.at[person_id, 'pn_stridor_in_calm_child']:
                stridor_identified_imci = self.sim.rng.choice([True, False], size=1,
                                                                    p=[p['prob_correct_id_stridor'],
                                                                       (1 - p['prob_correct_id_stridor'])])
                if stridor_identified_imci[True]:
                    correctly_classified_severe_pneumonia = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_classification_severe_pneum'],
                                                                      (1 - p[
                                                                          'prob_correct_classification_severe_pneum'])])
                    if correctly_classified_severe_pneumonia[True]:
                        df.at[person_id, 'imci_correctly_classified_severe_pneumonia'] = True

        # # # # # ASSESS Diarrhoea # # # # #
        if df.at[person_id, 'gi_diarrhoea_status']:
            diarrhoea_identified_imci = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_diarrhoea'],
                                                              (1 - p['prob_correct_id_diarrhoea'])])
            if diarrhoea_identified_imci[True]:
                # check for dehydration
                if df.at[person_id, 'gi_dehydration_status'] == 'severe dehydration':
                    lethargic_or_unconscious_identified_imci =\
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_lethargic_or_unconscious'],
                                                                      (1 - p['prob_correct_id_lethargic_or_unconscious'])])
                    sunken_eyes_identified_imci = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_sunken_eyes'],
                                                                      (1 - p['prob_correct_id_sunken_eyes'])])
                    not_drinking_or_poorly_identified_imci = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_not_drinking_or_poorly'],
                                                                      (1 - p['prob_correct_id_not_drinking_or_poorly'])])
                    skin_pinch_goes_back_very_slowly = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_skin_pinch_goes_back_very_slowly'],
                                                                      (1 - p['prob_correct_id_skin_pinch_goes_back_very_slowly'])])
                    if lethargic_or_unconscious_identified_imci[True] or sunken_eyes_identified_imci[True] \
                        or not_drinking_or_poorly_identified_imci[True] or skin_pinch_goes_back_very_slowly[True]:
                        df.at[person_id, 'imci_correctly_classified_severe_dehydration'] = True

                if df.at[person_id, 'gi_dehydration_status'] == 'severe dehydration':
                    restless_irritable_identified_imci = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_restless_irritable'],
                                                                      (1 - p[
                                                                          'prob_correct_id_restless_irritable'])])
                    sunken_eyes_identified_imci = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_sunken_eyes'],
                                                                      (1 - p['prob_correct_id_sunken_eyes'])])
                    drinks_eagerly_thirsty_identified_imci = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_drinks_eagerly_thirsty'],
                                                                      (1 - p[
                                                                          'prob_correct_id_drinks_eagerly_thirsty'])])
                    skin_pinch_goes_back_slowly = \
                        self.sim.rng.choice([True, False], size=1,
                                            p=[p['prob_correct_id_skin_pinch_goes_back_slowly'],
                                               (1 - p['prob_correct_id_skin_pinch_goes_back_slowly'])])
                    if restless_irritable_identified_imci[True] or sunken_eyes_identified_imci[True] \
                        or drinks_eagerly_thirsty_identified_imci[True] or skin_pinch_goes_back_slowly[True]:
                        df.at[person_id, 'imci_correctly_classified_severe_dehydration'] = True

                if df.at[person_id, 'gi_dehydration_status'] == 'no dehydration':
                    df.at[person_id, 'imci_correctly_classified_no_dehydration'] = True

                # persistent diarrhoea
                if df.at[person_id, df.gi_persistent_diarrhoea]:
                    persistent_diarrhoea_identified_imci= \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_persistent_diarrhoea'],
                                                                      (1 - p['prob_correct_id_persistent_diarrhoea'])])
                    if persistent_diarrhoea_identified_imci[True] & \
                        df.at[person_id, 'imci_correctly_classified_no_dehydration']:
                        df.at[person_id, 'imci_correctly_classified_persistent_diarrhoea'] = True

                # acute bloody diarrhoea
                if (df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery') &\
                    df.at[person_id, df.gi_persistent_diarrhoea] == False:
                    dysentery_identified_imci= \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_bloody_stools'],
                                                                      (1 - p['prob_correct_id_bloody_stools'])])
                    if dysentery_identified_imci[True]:
                        df.at[person_id, 'imci_correctly_classified_dysentery'] = True

                # Diarrhoea over 14 days
                if df.at[person_id, df.gi_persistent_diarrhoea]:
                    HSA_identified_persistent_diarrhoea = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_persistent_diarrhoea'],
                                                                      (1 - p['prob_correct_id_persistent_diarrhoea'])])

            will_CHW_ask_about_fever = self.module.rng.rand() < 0.5
            will_CHW_ask_about_cough = self.module.rng.rand() < 0.5

            # in the apply() part of the event:
            # here is where we have the CHW going through the algorithm

            # will_CHW_ask_about_fever = rand()<0.5
            # will_CHW_ask_about_cough = rand()<0.5

            # fever_is_detected = (df[person_id,'fever'] is True) and will_ask_CHW_ask_about_fever
            # cough_is_detected = (df[person_id,'cough'] is True) and will_CHW_ask_about_cough

            # if fever_is_detected:
            #   if cough_is_detected:
            #       -- child has bouth fever and cough
            # make a event for the treatment for this condition
            # HSI_Treatment_For_Fever_And_Cough

        # follow-up care ---------------------------

        # class follow_up_visit

'''

"""  
        # stepone : work out if the child has 'malaria'
        has_malaria = df.atperson_id,'Malaria'[]
        sens_dx_malaria= 0.9
        spec_dx_malaria = 0.8
        snes_dx_pneumo = 0.7
        spec_dx_pneuo = 0.5

        _bad_CHW = send_dx_malria_good_CHW * 0.5*bad_CHW
        -bad_CHW = 0.2

        good_CHW = self.rng.rand < prob_good_CHW

        - will the child be diagonosed by the algogorith
        correctly_diagnoed_malaria = has_malria and self.rng.rand<sens_dx_malaria
        missed_diagnosed_malaria = has malria and not correctly_diagnoed
        false_positive_diagnosed_malria = (not has_malaria and self.rng.rand<(1-spec_dx_malaria)

     

