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
         'any of the 4 general danger signs defined by the IMNCI guidelines'),
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

        sick_child = df.index[df.is_alive & (df.gi_diarrhoea_status | df.ri_pneumonia_status) & (df.age_years < 5)]
        # TODO: add other childhood diseases to the index: measles, malaria, malnutrition and other

        for person_id in sick_child:
            # first checking for imci general danger signs
            # danger sign - convulsions ---------------------------------------------------------------------------
            if df.at[person_id, 'ds_convulsions']:
                convulsions_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_convulsions'],
                                                                  (1 - p['prob_correct_id_convulsions'])])
                if convulsions_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if convulsions_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - not able to drink or breastfeed -------------------------------------------------------
            if df.at[person_id, 'ds_not_able_to_drink_or_breastfeed']:
                inability_to_drink_breastfeed_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_not_able_to_drink_or_breastfeed'],
                                                                  (1 - p[
                                                                      'prob_correct_id_not_able_to_drink_or_breastfeed'])])
                if inability_to_drink_breastfeed_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if inability_to_drink_breastfeed_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - vomits everything ---------------------------------------------------------------------
            if df.at[person_id, 'ds_vomits_everything']:
                vomiting_everything_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_vomits_everything'],
                                                                  (1 - p['prob_correct_id_vomits_everything'])])
                if vomiting_everything_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if vomiting_everything_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - unusually sleepy or unconscious -------------------------------------------------------
            if df.at[person_id, 'ds_unusually_sleepy_unconscious']:
                unusually_sleepy_unconscious_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_unusually_sleepy_unconscious'],
                                                                  (1 - p[
                                                                      'prob_correct_id_unusually_sleepy_unconscious'])])
                if unusually_sleepy_unconscious_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = True
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if unusually_sleepy_unconscious_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_general_danger_signs'] = False
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # Then, check for other danger signs of the iccm # # # # # # # # # # # #
            # TODO: complete this section with malnutrition code
            # danger sign - for child aged 6-59 months, red on MUAC strap ----------------------------------------
            if df.at[person_id, 'ds_red_MUAC_strap']:
                red_MUAC_strap_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_red_MUAC'],
                                                                  (1 - p['prob_correct_id_red_MUAC'])])
                if red_MUAC_strap_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if red_MUAC_strap_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - swelling of both feet ----------------------------------------------------------------
            if df.at[person_id, 'ds_swelling_both_feet']:
                swelling_both_feet_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_swelling_both_feet'],
                                                                  (1 - p['prob_correct_id_swelling_both_feet'])])
                if swelling_both_feet_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if swelling_both_feet_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - cough for more than 21 days ----------------------------------------------------------
            if df.at[person_id, 'ds_cough_more_than_21days']:
                cough_more_than_21days_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1,
                                        p=[p['prob_correct_id_cough_more_than_21days'],
                                            (1 - p['prob_correct_cough_more_than_21days'])])
                if cough_more_than_21days_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if cough_more_than_21days_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - chest indrawing ----------------------------------------------------------------------
            if df.at[person_id, 'ds_chest_indrawing']:
                chest_indrawing_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_chest_indrawing'],
                                                                  (1 - p['prob_correct_id_chest_indrawing'])])
                if chest_indrawing_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if chest_indrawing_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # danger sign - fever for last 7 days or more --------------------------------------------------------
            # TODO: Add in malaria stuff
            if df.at[person_id, 'ds_fever_more_than_7days']:
                fever_more_than_7days_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_fever_more_than_7days'],
                                                                  (1 - p['prob_correct_id_fever_more_than_7days'])])
                if fever_more_than_7days_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                if fever_more_than_7days_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False

            # diarrhoea danger signs ----------------------------------------------------------------------------
            HSA_identified_diarrhoea = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_diarrhoea'],
                                                              (1 - p['prob_correct_id_diarrhoea'])])
            if HSA_identified_diarrhoea[True]:
                # Blood in stools
                if df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery':
                    HSA_identified_bloody_diarrhoea = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_bloody_stools'],
                                                                      (1 - p['prob_correct_id_bloody_stools'])])
                    if HSA_identified_bloody_diarrhoea[True]:
                        df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                        df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
                    if HSA_identified_bloody_diarrhoea[False]:
                        df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = False
                        df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
                # Diarrhoea over 14 days
                if df.at[person_id, df.gi_persistent_diarrhoea]:
                    HSA_identified_persistent_diarrhoea = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_persistent_diarrhoea'],
                                                                      (1 - p['prob_correct_id_persistent_diarrhoea'])])
                    if HSA_identified_persistent_diarrhoea[True]:
                        df.at[person_id, 'ccm_correctly_identified_iccm_danger_sign'] = True
                        df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
                    if HSA_identified_persistent_diarrhoea[False]:
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

        iCCM
        if correctly identified_fast breathing and cough:
                correctly_diagnosed_pneumonia
        correctly_identified_danger_signs
        correctly_medication_given

        if correctly identified_diarrhoea and dehydration:
        if correctly identified danger sign
        correctly classified as severe pneumonia
        correctly refered
        correctly gave treatment and advice


        if correctly identified_fever

        IMCI
        correctly_diagnosed_pneumon
        correclty_dianoged _severe_pneu_OR_very_severe_diasease

        hospital management
        correctly_diagnosed_pneumon
        correclty_dianoged _severe_pneu
        correctly_diagnosed_very_severe_diasease


        # all_seeking_care_from_HSA = df.index[all those seeking care in pneumonia, diarrhoea and malaria modules
        # and maybe also other modules??]
        # for child in all_seeking_care_from_HSA:
        if df.at[person_id, 'pn_any_general_danger_sign' == True]:
            will_CHW_ask_about_fever = self.module.rng.rand() < 0.5
            will_CHW_ask_about_cough = self.module.rng.rand() < 0.5

            hsa_will_give_first-dose_of_antibiotic = if there is drugs available
        hsa_will refer_immediatly
        child_goes_to health_facility


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

    PROPERTIES = {
        'ccm_cough_14days_or_more': Property(Types.BOOL, 'danger sign - cough for 14 days or more'),
        'ccm_diarrhoea_14days_or_more': Property(Types.BOOL, 'danger sign - diarrhoea for 14 days or more'),
        'ccm_blood_in_stool': Property(Types.BOOL, 'danger sign - blood in the stool'),
        'ccm_fever_7days_or_more': Property(Types.BOOL, 'danger sign - fever for the last 7 days or more'),
        'ccm_convulsions': Property(Types.BOOL, 'danger sign - convulsions'),
        'ccm_not_able_drink_or_eat': Property(Types.BOOL, 'danger sign - not able to drink or eat anything'),
        'ccm_vomits_everything': Property(Types.REAL, 'danger sign - vomits everything'),
        'ccm_chest_indrawing': Property(Types.BOOL, 'danger sign - chest indrawing'),
        'ccm_unusually_sleepy_unconscious': Property(Types.BOOL, 'danger sign - unusually sleepy or unconscious'),
        'ccm_red_MUAC_strap': Property(Types.BOOL, 'danger sign - red on MUAC strap'),
        'ccm_swelling_both_feet': Property(Types.BOOL, 'danger sign - swelling of both feet'),
        'ccm_diarrhoea_lt14days': Property(Types.BOOL, 'treat - diarrhoea less than 14 days and no blood in stool'),
        'ccm_fever_lt7days': Property(Types.BOOL, 'treat - fever less than 7 days in malaria area'),
        'ccm_fast_breathing': Property(Types.BOOL, 'treat - fast brething'),
        'ccm_yellow_MUAC_strap': Property(Types.BOOL, 'treat - yellow on MUAC strap')
    }
    'asked_about_cough'
        'asked_about_diarrhoea'
        'if_diarrhoea_asked_about_blood_in_stool'
        'asked_about_fever'
        'if_fever_asked_how_long'
        'asked_about_convulsions'
        'asked_about_difficult_drinking_or_feeding'
        'if_difficult_drink/feed_asked_not_able_to_drink/feed_anything'
        'asked_vomiting'
        'if_vomiting_asked_vomits_everything'
        'asked_about_HIV'
        'looked_for_chest_indrawing'
        'looked_for_fast_breathing'
        'looked_for_unusually_sleepy_unconscious'
        'looked_for_signs_severe_malnutrition'

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self

        # DEFAULTS
        df['ccm_cough_14days_or_more'] = False
        df['ccm_diarrhoea_14days_or_more'] = False
        df['ccm_blood_in_stool'] = False
        df['ccm_fever_7days_or_more'] = False
        df['ccm_convulsions'] = False
        df['ccm_not_able_drink_or_eat'] = False
        df['ccm_vomits_everything'] = False
        df['ccm_chest_indrawing'] = False
        df['ccm_unusually_sleepy_unconscious'] = False
        df['ccm_red_MUAC_strap'] = False
        df['ccm_swelling_both_feet'] = False
        df['ccm_diarrhoea_lt14days'] = False
        df['ccm_fever_lt7days'] = False
        df['ccm_fast_breathing'] = False
        df['ccm_yellow_MUAC_strap'] = False


    def diarrhoea_diagnosis(self, person_id):
        now = self.sim.date
        df = self.sim.population.props

        # work out if child has diarrhoea, is correctly diagnosed by the algorithm and treatment
        has_diarrhoea = df.at[person_id, 'gi_diarrhoea_status']
        has_danger_signs = df.at[person_id, 'any_danger_signs']
        has_some_dehydration = df.at[person_id, 'gi_dehydration_status'] = 'some dehydration'
        has_severe_dehydration = df.at[person_id, 'gi_dehydration_status'] = 'severe dehydration'

    def pneumonia_diagnosis(self, person_id):
        now = self.sim.date
        df = population.props

        # work out if child has diarrhoea, is correctly diagnosed by the algorithm and treatment
        has_fast_breathing = df.at[person_id, 'gi_diarrhoea_status']
        has_danger_signs = df.at[person_id, 'any_danger_signs']

"""

