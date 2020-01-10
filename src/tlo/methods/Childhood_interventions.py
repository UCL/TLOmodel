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
        # iCCM - Integrated community case management properties used
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

        # iCCM symptoms
        'ds_cough_for_more_than_21days': Property
        (Types.BOOL, 'iCCM danger sign - cough for 21 days or more'
         ),








        # HSA assessement of symptoms outcome
        'ccm_assessed_cough': Property
        (Types.BOOL, 'HSA asked if the child has cough, or mother\'s report'
         ),
        'ccm_assessed_diarrhoea': Property
        (Types.BOOL, 'HSA asked if the child has diarrhoea, or mother\'s report'
         ),
        'ccm_assessed_fever': Property
        (Types.BOOL, 'HSA asked if the child has fever, or mother\'s report'
         ),
        'ccm_id_fast_breathing': Property
        (Types.BOOL, 'HSA identified fast breathing in child'
         ),
        'ccm_id_ds_blood_in_stools': Property
        (Types.BOOL, 'HSA identified bloody stool in child'
         ),
        'ccm_id_ds_diarrhoea_for_14days_or_more': Property
        (Types.BOOL, 'HSA identified diarrhoea for 14 days or more in child'
         ),
        'ccm_id_ds_fever_for_last_7days': Property
        (Types.BOOL, 'HSA identified fever lasting 7 days or more'
         ),
        'ccm_assessed_red_eyes': Property
        (Types.BOOL, 'HSA asked if the child has red eyes, or mother\'s report'
         ),
        'ccm_id_ds_red_eye_for_4days_or_more': Property
        (Types.BOOL, 'HSA identified red eye for 4 days or more in child'
         ),
        'ccm_id_ds_red_eye_with_visual_problem': Property
        (Types.BOOL, 'HSA identified red eye with visual problem in child'
         ),
        'ccm_id_ds_convulsions': Property
        (Types.BOOL, 'HSA identified convulsions in child'
         ),
        'ccm_id_ds_not_able_to_drink_or_feed': Property
        (Types.BOOL, 'HSA identified inability to drink or breastfeed/feed in child'
         ),
        'ccm_id_ds_vomits_everything': Property
        (Types.BOOL, 'HSA identified vomiting everything in child'
         ),
        'ccm_id_ds_very_sleepy_or_unconscious': Property
        (Types.BOOL,
         'HSA identified child to be very sleepy or unconscious'
         ),
        'ccm_id_ds_chest_indrawing': Property
        (Types.BOOL,
         'HSA identified chest indrawing in child'
         ),
        'ccm_id_ds_red_on_MUAC': Property
        (Types.BOOL,
         'HSA measured red on MUAC tape'
         ),
        'ccm_id_ds_swelling_of_both_feet': Property
        (Types.BOOL,
         'HSA identified swelling of both feet in child'
         ),
        'ccm_id_ds_palmar_pallor': Property
        (Types.BOOL,
         'HSA identified palmar pallor in child'
         ),
        'at_least_one_ccm_danger_sign_identified': Property
        (Types.BOOL,
         'HSA identified at least one iCCM danger sign'
         ),
        # iCCM treatment action
        'ccm_referral_decision': Property
        (Types.CATEGORICAL,
         'HSA decided to refer or to treat at home', categories=['referred to health facility', 'home treatment']
         ),



         # IMCNI - Integrated Management of Neonatal and Childhood Illnesses algorithm

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

        # # # # # # # # # # # # FIRST IS THE ASSESSMENT OF SYMPTOMS # # # # # # # # # # # #
        # CHECKING FOR COUGH ---------------------------------------------------------------------------------
        HSA_asked_for_cough = self.module.rng.rand() < p['prob_checked_for_cough']
        if HSA_asked_for_cough:
            df.at[person_id, 'ccm_assessed_cough'] = True
        else:
            df.at[person_id, 'ccm_assessed_cough'] = False
        if HSA_asked_for_cough & df.at[person_id, 'sy_cough']:
            HSA_asked_for_duration_cough = self.module.rng.rand() < p['prob_ask_duration_of_cough']
            if HSA_asked_for_duration_cough & df.at[person_id, 'ds_cough_more_than_21days']:
                df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] = True # CAN I HAVE THIS AS A DICTIONARY OF DANGER SIGNS FOR EACH PERSON_ID
            else:
                df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] = True
            HSA_counted_breaths_per_minute = \
                self.module.rng.rand() < p['prob_assess_fast_breathing']
            if HSA_counted_breaths_per_minute & df.at[person_id, 'has_fast_breathing']:
                df.at[person_id, 'ccm_id_fast_breathing'] = True
            else:
                df.at[person_id, 'ccm_id_fast_breathing'] = False

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
                df.at[person_id, 'ccm_id_ds_blood_in_stools'] = True
                # df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
            else:
                df.at[person_id, 'ccm_id_ds_blood_in_stools'] = False
                # df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
        # Diarrhoea over 14 days
            HSA_asked_duration_diarrhoea = self.module.rng.rand() < p['prob_check_persistent_diarrhoea']
            if HSA_asked_duration_diarrhoea & df.at[person_id, df.gi_diarrhoea_type] == 'persistent':
                df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] = True
            else: # does this else checks for those persistent but were not asked the duration?
                df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] = False

        # CHECKING FOR FEVER ----------------------------------------------------------------------------------
        HSA_asked_for_fever = self.module.rng.rand() < p['prob_checked_for_fever']
        if HSA_asked_for_fever:
            df.at[person_id, 'ccm_assessed_fever'] = True
        else:
            df.at[person_id, 'ccm_assessed_fever'] = False
        if HSA_asked_for_fever & df.at[person_id, 'sy_fever']:
            HSA_asked_duration_fever = self.module.rng.rand() < p['prob_check_duration_fever']
            if HSA_asked_duration_fever & df.at[person_id, 'sy_fever_over_7days']:
                df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] = True
            else:
                df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] = False

        # CHECKING FOR RED EYE --------------------------------------------------------------------------------
        HSA_checked_for_red_eyes = self.module.rng.rand() < p['prob_checked_for_red_eyes']
        if HSA_checked_for_red_eyes:
            df.at[person_id, 'ccm_assessed_red_eyes'] = True
        else:
            df.at[person_id, 'ccm_assessed_red_eyes'] = False
        if HSA_checked_for_red_eyes & df.at[person_id, 'sy_red_eyes']:
            HSA_asked_duration_red_eyes = self.module.rng.rand() < p['prob_asked_duration_red_eyes']
            HSA_asked_visual_difficulty = self.module.rng.rand() < p['prob_asked_visual_difficulty']
            if HSA_asked_duration_red_eyes & df.at[person_id, 'sy_red_eyes_over_4days']:
                df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] = True
            else:
                df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] = False
            if HSA_asked_visual_difficulty & df.at[person_id, 'sy_red_eye_with_visual_problem']:
                df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] = True
            else:
                df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] = False

        # CHECKING FOR GENERAL DANGER SIGNS -------------------------------------------------------------------
        # danger sign - convulsions ---------------------------------------------------------------------------
        HSA_asked_for_convulsions = self.module.rng.rand() < p['prob_check_convulsions']
        if HSA_asked_for_convulsions & df.at[person_id, 'ds_convulsions']:
            df.at[person_id, 'ccm_id_ds_convulsions'] = True
        else:
            df.at[person_id, 'ccm_id_ds_convulsions'] = False

        # danger sign - not able to drink or breastfeed -------------------------------------------------------
        HSA_asked_problem_feeding_drinking = self.module.rng.rand() < p['prob_check_feeding_drinking']
        if HSA_asked_problem_feeding_drinking & df.at[person_id, 'ds_not_able_to_drink_or_breastfeed']:
            df.at[person_id, 'ccm_id_ds_not_able_to_drink_or_feed'] = True
        else:
            df.at[person_id, 'ccm_id_ds_not_able_to_drink_or_feed'] = False

        # danger sign - vomits everything ---------------------------------------------------------------------
        HSA_asked_vomiting = self.module.rng.rand() < p['prob_check_vomiting_everything']
        if HSA_asked_vomiting & df.at[person_id, 'ds_vomiting_everything']:
            df.at[person_id, 'ccm_id_ds_vomits_everything'] = True
        else:
            df.at[person_id, 'ccm_id_ds_vomits_everything'] = False

        # danger sign - unusually sleepy or unconscious -------------------------------------------------------
        HSA_looked_for_sleepy_unconscious = self.module.rng.rand() < p['prob_correct_id_unusually_sleepy_unconscious']
        if HSA_looked_for_sleepy_unconscious & df.at[person_id, 'ds_unusually_sleepy_unconscious']:
            df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] = True
        else:
            df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] = False

        # CHECKING FOR ICCM DANGER SIGNS ---------------------------------------------------------------------
        # danger sign - chest indrawing ----------------------------------------------------------------------
        HSA_looked_for_chest_indrawing = self.module.rng.rand() < p['prob_check_chest_indrawing']
        if HSA_looked_for_chest_indrawing & df.at[person_id, 'ds_chest_indrawing']:
            df.at[person_id, 'ccm_id_ds_chest_indrawing'] = True
        else:
            df.at[person_id, 'ccm_id_ds_chest_indrawing'] = False

        # danger sign - for child aged 6-59 months, red on MUAC strap ----------------------------------------
        HSA_used_MUAC_tape = self.module.rng.rand() < p['prob_using_MUAC_tape']
        if HSA_used_MUAC_tape & df.at[person_id, 'ds_red_MUAC_strap']:
            df.at[person_id, 'ccm_id_ds_red_on_MUAC'] = True
        else:
            df.at[person_id, 'ccm_id_ds_red_on_MUAC'] = False

        # danger sign - swelling of both feet ----------------------------------------------------------------
        HSA_looked_for_swelling_feet = self.module.rng.rand() < p['prob_check_swelling_both_feet']
        if HSA_looked_for_swelling_feet & df.at[person_id, 'ds_swelling_both_feet']:
            df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] = True
        else:
            df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] = False

        # danger sign - swelling of both feet ----------------------------------------------------------------
        HSA_looked_for_palmar_pallor = self.module.rng.rand() < p['prob_check_palmar_pallor'] # TODO: need to add sensitivity and specificity of HSA's assessment of symptoms
        if HSA_looked_for_palmar_pallor & df.at[person_id, 'ds_palmar_pallor']:
            df.at[person_id, 'ccm_id_ds_palmar_pallor'] = True
        else:
            df.at[person_id, 'ccm_id_ds_palmar_pallor'] = False

        # AT LEAST ONE ICCM DANGER SIGN
        if (df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] | df.at[person_id, 'ccm_id_ds_blood_in_stools'] |
            df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] | df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] |
            df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] | df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] |
            df.at[person_id, 'ccm_id_ds_convulsions'] | df.at[person_id, 'ccm_ds_not_able_to_drink_or_feed'] |
            df.at[person_id, 'ccm_id_ds_vomits_everything'] | df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] |
            df.at[person_id, 'ccm_id_ds_chest_indrawing'] | df.at[person_id, 'ccm_id_ds_red_on_MUAC'] |
            df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] | df.at[person_id, 'ccm_id_ds_palmar_pallor']):
            df.at[person_id, 'at_least_one_ccm_danger_sign_identified'] = True

        # CHECKING FOR OTHER PROBLEMS -------------------------------------------------------------------------
            HSA_checked_for_other_problems = self.module.rng.rand() < p[
                'prob_checked_for_other_problems']  # HSA check or mother's report
            if HSA_checked_for_other_problems:
                HSA_referred_other_problems = self.module.rng.rand() < p['prob_refer_other_problems']
                if HSA_referred_other_problems:
                    df.at[
                        person_id, 'ccm_referral_decision'] = 'referred to health facility'  # TODO: put at the bottom in referral, and complete
                else:

        # # # # # # # # # # # # SECOND, IS THE DECISION TO REFER OR TREAT # # # # # # # # # # # #
        # give referral decision
        if df.at[person_id, 'at_least_one_ccm_danger_sign_identified']:
            HSA_referral_decision = self.module.rng.rand() < p['prob_correct_referral_decision_for_any_danger_signs']
            if HSA_referral_decision:
                df.at[person_id, 'ccm_referral_decision'] = 'referred to health facility'
            else:
                df.at[person_id, 'ccm_referral_decision'] = 'home treatment'
        else:
            HSA_home_treatment_decision = self.module.rng.rand() < p['prob_correct_no_referral_decision_for_uncomplicated_cases']
            if HSA_home_treatment_decision:
                df.at[person_id, 'ccm_referral_decision'] = 'home treatment'
            else:
                df.at[person_id, 'ccm_referral_decision'] = 'referred to health facility'

        # # # # # # # # # # # # THIRD, IS THE DECISION TO REFER OR TREAT # # # # # # # # # # # #
        # danger signs identified and referred to health facility --------------------------------------------------
        # diarrhoea + danger sign
        if (df.at[person_id, 'sy_diarrhoea'] & df.at[person_id, 'ccm_assessed_diarrhoea'] &
            df.at[person_id, 'at_least_one_danger_sign_identified'] &
            df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORS
        # fever + danger_sign
        if (df.at[person_id, 'sy_fever'] & df.at[person_id, 'ccm_assessed_fever'] &
            df.at[person_id, 'at_least_one_danger_sign_identified'] &
            df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE FIRST DOSE OF LA (age dependent-dose)
        # chest indrawing or fast breathing + danger_sign
        if ((df.at[person_id, 'ccm_id_ds_chest_indrawing'] | df.at[person_id, 'ccm_id_fast_breathing']) &
            df.at[person_id, 'at_least_one_danger_sign_identified'] &
            df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE FIRST DOSE OF ORAL ANTIBIOTIC (age dependent-dose)
        # red eye for 4 days or more
        if (df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] &
            df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- APPLY ANTIBIOTIC EYE OINTMENT

        # no danger signs identified and referred to health facility -------------------------------------











        # if no danger signs identified and home management -------------------------------------------------
        # diarrhoea
        if (df.at[person_id, 'sy_diarrhoea'] & df.at[person_id, 'ccm_assessed_diarrhoea'] &
            df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
            HSA_given_right_treatment = self.module.rng.rand() < p['prob_right_treatment_plan_diarrhoea']
            HSA_given_complete_treatment_plan = self.module.rng.rand() < p['prob_complete_treatment_diarrhoea']
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORS, + 2 ORS for mother, GIVE ZINC SUPPLEMENT (age dependent)
        # fever
        if (df.at[person_id, 'sy_fever'] & df.at[person_id, 'ccm_assessed_fever'] &
            df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE LA (age dependent-dose), GIVE PARACETAMOL (age dependent)
        # fast breathing
        if (df.at[person_id, 'ccm_id_fast_breathing'] &
            df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORAL ANTIBIOTIC (age dependent-dose)
        # red eye
        if (df.at[person_id, 'sy_red_eyes'] & df.at[person_id, 'ccm_assessed_red_eyes'] &
            df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
            # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ANTIBIOTIC EYE OINTMENT

        # ----------------------------------------------------------------------------------------------------
        # GET ALL THE CORRECT ICCM ACTION PLAN
        # ----------------------------------------------------------------------------------------------------
        # get all the danger signs
        if (df.at[person_id, 'ds_cough_more_than_21days'] | (
            df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery') |
            (df.at[person_id, df.gi_diarrhoea_type] == 'persistent') | df.at[person_id, 'ds_cough_more_than_21days'] |
            df.at[person_id, 'sy_fever_over_7days'] | df.at[person_id, 'sy_red_eyes_over_4days'] | df.at[
                person_id, 'ds_convulsions'] |
            df.at[person_id, 'ds_not_able_to_drink_or_breastfeed'] | df.at[person_id, 'ds_vomiting_everything'] |
            df.at[person_id, 'ds_unusually_sleepy_unconscious'] | df.at[person_id, 'ds_chest_indrawing'] |
            df.at[person_id, 'ds_palmar_pallor'] |
            df.at[person_id, 'ds_red_MUAC_strap'] | df.at[person_id, 'ds_swelling_both_feet']):
            df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] = True

        # any danger sign to be referred
        if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
            (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
            (df.at[person_id, 'sy_fever'] == False) & (df.at[person_id, 'sy_diarrhoea'] == False) &
            (df.at[person_id, 'sy_chest_indrawing'] == False) & (df.at[person_id, 'sy_fast_breathing'] == False)):
            df.at[person_id, 'ccm_correct_action_plan'] = True
        # for fever + danger sign
        if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
            (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') & df.at[person_id, 'sy_fever'] &
            df.at[person_id, 'pre-referral treatment given']):
            df.at[person_id, 'ccm_correct_action_plan'] = True
        # for diarrhoea + danger sign
        if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
            (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') & df.at[person_id, 'sy_diarrhoea'] &
            df.at[person_id, 'pre-referral treatment given']):
            df.at[person_id, 'ccm_correct_action_plan'] = True
        # for fast breathing + danger sign
        if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
            (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
            (df.at[person_id, 'sy_fast_breathing'] | df.at[person_id, 'sy_chest_indrawing']) &
            df.at[person_id, 'pre-referral treatment given']):
            df.at[person_id, 'ccm_correct_action_plan'] = True
        # for red eye for 4 days or more
        if ((df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
            df.at[person_id, 'sy_red_eye_over_4_days'] & df.at[person_id, 'pre-referral treatment given']):
            df.at[person_id, 'ccm_correct_action_plan'] = True

        # # # # # # # # # # # # FOURTH, CHECK VACCINES RECEIVED # # # # # # # # # # # #
        # TODO: complete later with Tara's vaccine code
        HSA_checked_vaccines_received = self.module.rng.rand() < p['prob_check_vaccines_status']
        if HSA_checked_vaccines_received:
            if ((df.at[person_id, 'vacc_DHH1'] == False | df.at[person_id, 'vacc_OPV1']) &
                df.at[person_id, df.age_exact_years == 56/487]):
                HSA_advise_on_vaccine_schedule = self.module.rng.rand() < p['prob_advise_vaccination']
            if df.at[person_id, 'vacc_DHH2'] == False & df.at[person_id, df.age_exact_years == 280/1461]:

            if df.at[person_id, 'vacc_DHH3'] == False & df.at[person_id, df.age_exact_years == 392/1461]:

        # # # # # # # # # # # # FIFTH, FOLLOW UP # # # # # # # # # # # #



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

     

