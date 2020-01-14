import logging

import pandas as pd
from tlo import Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
This module determines how the diagnosis of children at their initial presentation at a health facility

"""

class ChildhoodManagement(Module):

    PARAMETER = {
        # Parameters for the iCCM algorithm performed by the HSA
        'prob_checked_for_cough':
            Parameter(Types.REAL, 'probability of HSA checking for presence of cough in the child, or mother\'s report'
                      ),
        'prob_asked_for_duration_of_cough':
            Parameter(Types.REAL, 'probability of HSA asking for the duration of cough'
                      ),
        'prob_assess_fast_breathing':
            Parameter(Types.REAL, 'probability of HSA counting breaths per minute'
                      ),
        'prob_check_for_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA asking for the presence of diarrhoea, or mother\'s report'
                      ),
        'prob_check_bloody_stools':
            Parameter(Types.REAL, 'probability of HSA asking for the presence of blood in stools'
                      ),
        'prob_check_persistent_diarrhoea':
            Parameter(Types.REAL, 'probability of HSA asking for the duration of diarrhoea episode'
                      ),
        'prob_checked_for_fever':
            Parameter(Types.REAL, 'probability of HSA asked for fever, or mother\'s report'
                      ),
        'prob_check_duration_fever':
            Parameter(Types.REAL, 'probability of HSA asked the duration of fever'
                      ),
        'prob_checked_for_red_eyes':
            Parameter(Types.REAL, 'probability of HSA checking the presence of red eyes in child'
                      ),
        'prob_asked_duration_red_eyes':
            Parameter(Types.REAL, 'probability of HSA asking the duration of child having red eyes'
                      ),
        'prob_asked_visual_difficulty':
            Parameter(Types.REAL, 'probability of HSA asked for visual difficulty in child'
                      ),
        'prob_check_convulsions':
            Parameter(Types.REAL, 'probability of HSA asked for convulsions'
                      ),
        'prob_check_feeding_drinking':
            Parameter(Types.REAL, 'probability of HSA asking for problems in feeding or drinking'
                      ),
        'prob_check_vomiting_everything':
            Parameter(Types.REAL, 'probability of HSA asking for vomiting, specifically vomiting everything'
                      ),
        'prob_id_unusually_sleepy_unconscious':
            Parameter(Types.REAL, 'probability of HSA identifying child as very sleepy or unconscious'
                      ),
        'pprob_check_chest_indrawing':
            Parameter(Types.REAL, 'probability of HSA checking for chest indrawing in child'
                      ),
        'prob_using_MUAC_tape':
            Parameter(Types.REAL, 'probability of HSA using MUAC tape for measurement of arm circumference'
                      ),
        'prob_check_swelling_both_feet':
            Parameter(Types.REAL, 'probability of HSA checking for swelling of booth feet'
                      ),
        'prob_check_palmar_pallor':
            Parameter(Types.REAL, 'probability of HSA checking for palmar pallor in child'
                      ),
        'prob_check_for_other_problems':
            Parameter(Types.REAL, 'probability of HSA checking fother problems not included in the iCCM'
                      ),
        }

    PROPERTIES = {}   # TODO: I think we want this module to not have any properties of its own.

    # PROPERTIES = {
    #     # iCCM - Integrated community case management properties used
    #     'iccm_danger_sign': Property
    #     (Types.BOOL, 'child has at least one iccm danger signs : '
    #                  'convulsions, very sleepy or unconscious, chest indrawing, vomiting everything, '
    #                  'not able to drink or breastfeed, red on MUAC strap, swelling of both feet, '
    #                  'fever for last 7 days or more, blood in stool, diarrhoea for 14 days or more, '
    #                  'and cough for at least 21 days'
    #      ),
    #     'ccm_correctly_identified_general_danger_signs': Property
    #     (Types.BOOL, 'HSA correctly identified at least one of the IMCI 4 general danger signs - '
    #                  'convulsions, lethargic or unconscious, vomiting everything, not able to drink or breastfeed'
    #      ),
    #     'ccm_correctly_identified_danger_signs': Property
    #     (Types.BOOL, 'HSA correctly identified at least one danger sign, including '
    #                  'convulsions, very sleepy or unconscious, chest indrawing, vomiting everything, '
    #                  'not able to drink or breastfeed, red on MUAC strap, swelling of both feet, '
    #                  'fever for last 7 days or more, blood in stool, diarrhoea for 14 days or more, '
    #                  'and cough for at least 21 days'
    #      ),
    #
    #     # iCCM symptoms
    #     'sy_cough': Property
    #     (Types.BOOL, 'symptom - cough'
    #      ),
    #     'ds_cough_for_more_than_21days': Property
    #     (Types.BOOL, 'iCCM danger sign - cough for 21 days or more'
    #      ),
    #     'sy_fast_breathing': Property
    #     (Types.BOOL, 'symptom - fast breathing'
    #      ),
    #     'sy_diarrhoea': Property
    #     (Types.BOOL, 'symptom - diarrhoea'
    #      ),
    #     'sy_fever': Property
    #     (Types.BOOL, 'symptom - fever'
    #      ),
    #     'ds_fever_over_7days': Property
    #     (Types.BOOL, 'iCCM danger sign - fever for last 7 days'
    #      ),
    #     'sy_red_eyes': Property
    #     (Types.BOOL, 'symptom - red eye'
    #      ),
    #     'ds_red_eyes_over_4days': Property
    #     (Types.BOOL, 'iCCM danger sign - red eye for 4 days or more'
    #      ),
    #     'ds_red_eye_with_visual_problem': Property
    #     (Types.BOOL, 'iCCM danger sign - red eye with visual problem'
    #      ),
    #     'ds_convulsions': Property
    #     (Types.BOOL, 'iCCM danger sign - convulsions'
    #      ),
    #     'ds_not_able_to_drink_or_breastfeed': Property
    #     (Types.BOOL, 'iCCM danger sign - unable to drink or breastfeed'
    #      ),
    #     'ds_vomiting_everything': Property
    #     (Types.BOOL, 'iCCM danger sign - vomiting everything'
    #      ),
    #     'ds_unusually_sleepy_unconscious': Property
    #     (Types.BOOL, 'iCCM danger sign - unusually sleepy or unconscious'
    #      ),
    #     'ds_chest_indrawing': Property
    #     (Types.BOOL, 'iCCM danger sign - chest indrawing'
    #      ),
    #     'ds_red_MUAC_strap': Property
    #     (Types.BOOL, 'iCCM danger sign - red on MUAC tape'
    #      ),
    #     'ds_sweeling_both_feet': Property
    #     (Types.BOOL, 'iCCM danger sign - swelling of both feet'
    #      ),
    #     'ds_palmar_pallor': Property
    #     (Types.BOOL, 'iCCM danger sign - palmar pallor'
    #      ),
    #     # HSA assessement of symptoms outcome
    #     'ccm_assessed_cough': Property
    #     (Types.BOOL, 'HSA asked if the child has cough, or mother\'s report'
    #      ),
    #     'ccm_assessed_diarrhoea': Property
    #     (Types.BOOL, 'HSA asked if the child has diarrhoea, or mother\'s report'
    #      ),
    #     'ccm_assessed_fever': Property
    #     (Types.BOOL, 'HSA asked if the child has fever, or mother\'s report'
    #      ),
    #     'ccm_id_fast_breathing': Property
    #     (Types.BOOL, 'HSA identified fast breathing in child'
    #      ),
    #     'ccm_id_ds_blood_in_stools': Property
    #     (Types.BOOL, 'HSA identified bloody stool in child'
    #      ),
    #     'ccm_id_ds_diarrhoea_for_14days_or_more': Property
    #     (Types.BOOL, 'HSA identified diarrhoea for 14 days or more in child'
    #      ),
    #     'ccm_id_ds_fever_for_last_7days': Property
    #     (Types.BOOL, 'HSA identified fever lasting 7 days or more'
    #      ),
    #     'ccm_assessed_red_eyes': Property
    #     (Types.BOOL, 'HSA asked if the child has red eyes, or mother\'s report'
    #      ),
    #     'ccm_id_ds_red_eye_for_4days_or_more': Property
    #     (Types.BOOL, 'HSA identified red eye for 4 days or more in child'
    #      ),
    #     'ccm_id_ds_red_eye_with_visual_problem': Property
    #     (Types.BOOL, 'HSA identified red eye with visual problem in child'
    #      ),
    #     'ccm_id_ds_convulsions': Property
    #     (Types.BOOL, 'HSA identified convulsions in child'
    #      ),
    #     'ccm_id_ds_not_able_to_drink_or_feed': Property
    #     (Types.BOOL, 'HSA identified inability to drink or breastfeed/feed in child'
    #      ),
    #     'ccm_id_ds_vomits_everything': Property
    #     (Types.BOOL, 'HSA identified vomiting everything in child'
    #      ),
    #     'ccm_id_ds_very_sleepy_or_unconscious': Property
    #     (Types.BOOL,
    #      'HSA identified child to be very sleepy or unconscious'
    #      ),
    #     'ccm_id_ds_chest_indrawing': Property
    #     (Types.BOOL,
    #      'HSA identified chest indrawing in child'
    #      ),
    #     'ccm_id_ds_red_on_MUAC': Property
    #     (Types.BOOL,
    #      'HSA measured red on MUAC tape'
    #      ),
    #     'ccm_id_ds_swelling_of_both_feet': Property
    #     (Types.BOOL,
    #      'HSA identified swelling of both feet in child'
    #      ),
    #     'ccm_id_ds_palmar_pallor': Property
    #     (Types.BOOL,
    #      'HSA identified palmar pallor in child'
    #      ),
    #     'at_least_one_ccm_danger_sign_identified': Property
    #     (Types.BOOL,
    #      'HSA identified at least one iCCM danger sign'
    #      ),
    #     # iCCM treatment action
    #     'ccm_referral_decision': Property
    #     (Types.CATEGORICAL,
    #      'HSA decided to refer or to treat at home', categories=['referred to health facility', 'home treatment']
    #      ),
    #
    #      # IMCNI - Integrated Management of Neonatal and Childhood Illnesses algorithm
    #     'imci_assessment_of_main_symptoms': Property
    #     (Types.CATEGORICAL,
    #      'main symptoms assessments', categories=['correctly assessed', 'not assessed']
    #      ),
    #     'imci_classification_of_illness': Property
    #     (Types.CATEGORICAL,
    #      'disease classification', categories=['correctly classified', 'incorrectly classified']
    #      ),
    #     'imci_treatment': Property
    #     (Types.CATEGORICAL,
    #      'treatment given', categories=['correctly treated', 'incorrectly treated', 'not treated']
    #      ),
    #     'classification_for_cough_or_difficult_breathing': Property
    #     (Types.CATEGORICAL,
    #      'classification for cough or difficult breathing',
    #      categories=['classified as severe pneumonia or very severe disease',
    #                  'classified as pneumonia', 'classified as no pneumonia']
    #      ),
    #     'correct_classification_for_cough_or_difficult_breathing': Property
    #     (Types.BOOL,
    #      'classification for cough or difficult breathing is correct'
    #      ),
    # }

    def read_parameters(self, data_folder):
        p = self.parameters

        p['prob_checked_for_cough'] = 0.8
        p['prob_ask_duration_of_cough'] = 0.8
        p['prob_assess_fast_breathing'] = 0.8
        p['prob_check_for_diarrhoea'] = 0.8
        p['prob_check_bloody_stools'] = 0.8
        p['prob_check_persistent_diarrhoea'] = 0.8
        p['prob_checked_for_fever'] = 0.7
        p['prob_asked_duration_fever'] = 0.8
        p['prob_checked_for_red_eyes'] = 0.8
        p['prob_asked_duration_red_eyes'] = 0.8
        p['prob_asked_visual_difficulty'] = 0.8
        p['prob_check_convulsions'] = 0.8
        p['prob_check_feeding_drinking'] = 0.8
        p['prob_check_vomiting_everything'] = 0.8
        p['prob_id_unusually_sleepy_unconscious'] = 0.8
        p['prob_check_chest_indrawing'] = 0.8
        p['prob_using_MUAC_tape'] = 0.8
        p['prob_check_swelling_both_feet'] = 0.8
        p['prob_check_palmar_pallor'] = 0.9
        p['prob_check_for_other_problems'] = 0.8

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

    def do_management_of_child_at_facility_level_0(self, person_id, hsi_event):

        logger.debug('This is do_management_of_child_at_facility_level_0, for person %d in the community', person_id)
        df = self.sim.population.props
        p = self.module.parameters
        print("I will now do anything that needs to be done in an initial appointment")



    #     # # # # # # # # # # # # FIRST IS THE ASSESSMENT OF SYMPTOMS # # # # # # # # # # # #
    #     # CHECKING FOR COUGH ---------------------------------------------------------------------------------
    #     HSA_asked_for_cough = self.module.rng.rand() < p['prob_checked_for_cough']
    #     if HSA_asked_for_cough:
    #         df.at[person_id, 'ccm_assessed_cough'] = True
    #     else:
    #         df.at[person_id, 'ccm_assessed_cough'] = False
    #     if HSA_asked_for_cough & df.at[person_id, 'sy_cough']:
    #         HSA_asked_for_duration_cough = self.module.rng.rand() < p['prob_ask_duration_of_cough']
    #         if HSA_asked_for_duration_cough & df.at[person_id, 'ds_cough_more_than_21days']:
    #             df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] = True # CAN I HAVE THIS AS A DICTIONARY OF DANGER SIGNS FOR EACH PERSON_ID
    #         else:
    #             df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] = True
    #         HSA_counted_breaths_per_minute = \
    #             self.module.rng.rand() < p['prob_assess_fast_breathing']
    #         if HSA_counted_breaths_per_minute & df.at[person_id, 'has_fast_breathing']:
    #             df.at[person_id, 'ccm_id_fast_breathing'] = True
    #         else:
    #             df.at[person_id, 'ccm_id_fast_breathing'] = False
    #
    #     # CHECKING FOR DIARRHOEA -----------------------------------------------------------------------------
    #     HSA_asked_for_diarrhoea = self.module.rng.rand() < p['prob_checked_for_diarrhoea']
    #     if HSA_asked_for_diarrhoea:
    #         df.at[person_id, 'ccm_assessed_diarrhoea'] = True
    #     else:
    #         df.at[person_id, 'ccm_assessed_diarrhoea'] = False
    #     if HSA_asked_for_diarrhoea & df.at[person_id, 'sy_diarrhoea']:
    #     # Blood in stools
    #         HSA_asked_blood_in_stools = self.module.rng.rand() < p['prob_check_bloody_stools']
    #         if HSA_asked_blood_in_stools & df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery':
    #             df.at[person_id, 'ccm_id_ds_blood_in_stools'] = True
    #             # df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
    #         else:
    #             df.at[person_id, 'ccm_id_ds_blood_in_stools'] = False
    #             # df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = False
    #     # Diarrhoea over 14 days
    #         HSA_asked_duration_diarrhoea = self.module.rng.rand() < p['prob_check_persistent_diarrhoea']
    #         if HSA_asked_duration_diarrhoea & df.at[person_id, df.gi_diarrhoea_type] == 'persistent':
    #             df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] = True
    #         else: # does this else checks for those persistent but were not asked the duration?
    #             df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] = False
    #
    #     # CHECKING FOR FEVER ----------------------------------------------------------------------------------
    #     HSA_asked_for_fever = self.module.rng.rand() < p['prob_checked_for_fever']
    #     if HSA_asked_for_fever:
    #         df.at[person_id, 'ccm_assessed_fever'] = True
    #     else:
    #         df.at[person_id, 'ccm_assessed_fever'] = False
    #     if HSA_asked_for_fever & df.at[person_id, 'sy_fever']:
    #         HSA_asked_duration_fever = self.module.rng.rand() < p['prob_check_duration_fever']
    #         if HSA_asked_duration_fever & df.at[person_id, 'ds_fever_over_7days']:
    #             df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] = True
    #         else:
    #             df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] = False
    #
    #     # CHECKING FOR RED EYE --------------------------------------------------------------------------------
    #     HSA_checked_for_red_eyes = self.module.rng.rand() < p['prob_checked_for_red_eyes']
    #     if HSA_checked_for_red_eyes:
    #         df.at[person_id, 'ccm_assessed_red_eyes'] = True
    #     else:
    #         df.at[person_id, 'ccm_assessed_red_eyes'] = False
    #     if HSA_checked_for_red_eyes & df.at[person_id, 'sy_red_eyes']:
    #         HSA_asked_duration_red_eyes = self.module.rng.rand() < p['prob_asked_duration_red_eyes']
    #         HSA_asked_visual_difficulty = self.module.rng.rand() < p['prob_asked_visual_difficulty']
    #         if HSA_asked_duration_red_eyes & df.at[person_id, 'ds_red_eyes_over_4days']:
    #             df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] = True
    #         else:
    #             df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] = False
    #         if HSA_asked_visual_difficulty & df.at[person_id, 'ds_red_eye_with_visual_problem']:
    #             df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] = True
    #         else:
    #             df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] = False
    #
    #     # CHECKING FOR GENERAL DANGER SIGNS -------------------------------------------------------------------
    #     # danger sign - convulsions ---------------------------------------------------------------------------
    #     HSA_asked_for_convulsions = self.module.rng.rand() < p['prob_check_convulsions']
    #     if HSA_asked_for_convulsions & df.at[person_id, 'ds_convulsions']:
    #         df.at[person_id, 'ccm_id_ds_convulsions'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_convulsions'] = False
    #
    #     # danger sign - not able to drink or breastfeed -------------------------------------------------------
    #     HSA_asked_problem_feeding_drinking = self.module.rng.rand() < p['prob_check_feeding_drinking']
    #     if HSA_asked_problem_feeding_drinking & df.at[person_id, 'ds_not_able_to_drink_or_breastfeed']:
    #         df.at[person_id, 'ccm_id_ds_not_able_to_drink_or_feed'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_not_able_to_drink_or_feed'] = False
    #
    #     # danger sign - vomits everything ---------------------------------------------------------------------
    #     HSA_asked_vomiting = self.module.rng.rand() < p['prob_check_vomiting_everything']
    #     if HSA_asked_vomiting & df.at[person_id, 'ds_vomiting_everything']:
    #         df.at[person_id, 'ccm_id_ds_vomits_everything'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_vomits_everything'] = False
    #
    #     # danger sign - unusually sleepy or unconscious -------------------------------------------------------
    #     HSA_looked_for_sleepy_unconscious = self.module.rng.rand() < p['prob_id_unusually_sleepy_unconscious']
    #     if HSA_looked_for_sleepy_unconscious & df.at[person_id, 'ds_unusually_sleepy_unconscious']:
    #         df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] = False
    #
    #     # CHECKING FOR ICCM DANGER SIGNS ---------------------------------------------------------------------
    #     # danger sign - chest indrawing ----------------------------------------------------------------------
    #     HSA_looked_for_chest_indrawing = self.module.rng.rand() < p['prob_check_chest_indrawing']
    #     if HSA_looked_for_chest_indrawing & df.at[person_id, 'ds_chest_indrawing']:
    #         df.at[person_id, 'ccm_id_ds_chest_indrawing'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_chest_indrawing'] = False
    #
    #     # danger sign - for child aged 6-59 months, red on MUAC strap ----------------------------------------
    #     HSA_used_MUAC_tape = self.module.rng.rand() < p['prob_using_MUAC_tape']
    #     if HSA_used_MUAC_tape & df.at[person_id, 'ds_red_MUAC_strap']:
    #         df.at[person_id, 'ccm_id_ds_red_on_MUAC'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_red_on_MUAC'] = False
    #
    #     # danger sign - swelling of both feet ----------------------------------------------------------------
    #     HSA_looked_for_swelling_feet = self.module.rng.rand() < p['prob_check_swelling_both_feet']
    #     if HSA_looked_for_swelling_feet & df.at[person_id, 'ds_swelling_both_feet']:
    #         df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] = False
    #
    #     # danger sign - swelling of both feet ----------------------------------------------------------------
    #     HSA_looked_for_palmar_pallor = self.module.rng.rand() < p['prob_check_palmar_pallor'] # TODO: need to add sensitivity and specificity of HSA's assessment of symptoms
    #     if HSA_looked_for_palmar_pallor & df.at[person_id, 'ds_palmar_pallor']:
    #         df.at[person_id, 'ccm_id_ds_palmar_pallor'] = True
    #     else:
    #         df.at[person_id, 'ccm_id_ds_palmar_pallor'] = False
    #
    #     # AT LEAST ONE ICCM DANGER SIGN
    #     if (df.at[person_id, 'ccm_id_ds_cough_for_21days_or_more'] | df.at[person_id, 'ccm_id_ds_blood_in_stools'] |
    #         df.at[person_id, 'ccm_id_ds_diarrhoea_for_14days_or_more'] | df.at[person_id, 'ccm_id_ds_fever_for_last_7days'] |
    #         df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] | df.at[person_id, 'ccm_id_ds_red_eye_with_visual_problem'] |
    #         df.at[person_id, 'ccm_id_ds_convulsions'] | df.at[person_id, 'ccm_ds_not_able_to_drink_or_feed'] |
    #         df.at[person_id, 'ccm_id_ds_vomits_everything'] | df.at[person_id, 'ccm_id_ds_very_sleepy_or_unconscious'] |
    #         df.at[person_id, 'ccm_id_ds_chest_indrawing'] | df.at[person_id, 'ccm_id_ds_red_on_MUAC'] |
    #         df.at[person_id, 'ccm_id_ds_swelling_of_both_feet'] | df.at[person_id, 'ccm_id_ds_palmar_pallor']):
    #         df.at[person_id, 'at_least_one_ccm_danger_sign_identified'] = True
    #
    #     # CHECKING FOR OTHER PROBLEMS -------------------------------------------------------------------------
    #         HSA_checked_for_other_problems = self.module.rng.rand() < p[
    #             'prob_checked_for_other_problems']  # HSA check or mother's report
    #         if HSA_checked_for_other_problems:
    #             HSA_referred_other_problems = self.module.rng.rand() < p['prob_refer_other_problems']
    #             if HSA_referred_other_problems:
    #                 df.at[
    #                     person_id, 'ccm_referral_decision'] = 'referred to health facility'  # TODO: put at the bottom in referral, and complete
    #             else:
    #
    #     # # # # # # # # # # # # SECOND, IS THE DECISION TO REFER OR TREAT # # # # # # # # # # # #
    #     # give referral decision
    #     if df.at[person_id, 'at_least_one_ccm_danger_sign_identified']:
    #         HSA_referral_decision = self.module.rng.rand() < p['prob_correct_referral_decision_for_any_danger_signs']
    #         if HSA_referral_decision:
    #             df.at[person_id, 'ccm_referral_decision'] = 'referred to health facility'
    #         else:
    #             df.at[person_id, 'ccm_referral_decision'] = 'home treatment'
    #     else:
    #         HSA_home_treatment_decision = self.module.rng.rand() < p['prob_correct_no_referral_decision_for_uncomplicated_cases']
    #         if HSA_home_treatment_decision:
    #             df.at[person_id, 'ccm_referral_decision'] = 'home treatment'
    #         else:
    #             df.at[person_id, 'ccm_referral_decision'] = 'referred to health facility'
    #
    #     # # # # # # # # # # # # THIRD, IS THE DECISION TO REFER OR TREAT # # # # # # # # # # # #
    #     # danger signs identified and referred to health facility --------------------------------------------------
    #     # diarrhoea + danger sign
    #     if (df.at[person_id, 'sy_diarrhoea'] & df.at[person_id, 'ccm_assessed_diarrhoea'] &
    #         df.at[person_id, 'at_least_one_danger_sign_identified'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORS
    #     # fever + danger_sign
    #     if (df.at[person_id, 'sy_fever'] & df.at[person_id, 'ccm_assessed_fever'] &
    #         df.at[person_id, 'at_least_one_danger_sign_identified'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE FIRST DOSE OF LA (age dependent-dose)
    #     # chest indrawing or fast breathing + danger_sign
    #     if ((df.at[person_id, 'ccm_id_ds_chest_indrawing'] | df.at[person_id, 'ccm_id_fast_breathing']) &
    #         df.at[person_id, 'at_least_one_danger_sign_identified'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE FIRST DOSE OF ORAL ANTIBIOTIC (age dependent-dose)
    #     # red eye for 4 days or more
    #     if (df.at[person_id, 'ccm_id_ds_red_eye_for_4days_or_more'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- APPLY ANTIBIOTIC EYE OINTMENT
    #
    #     # no danger signs identified and referred to health facility -------------------------------------
    #
    #
    #
    #
    #
    #     # if no danger signs identified and home management -------------------------------------------------
    #     # diarrhoea
    #     if (df.at[person_id, 'sy_diarrhoea'] & df.at[person_id, 'ccm_assessed_diarrhoea'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
    #         HSA_given_right_treatment = self.module.rng.rand() < p['prob_right_treatment_plan_diarrhoea']
    #         HSA_given_complete_treatment_plan = self.module.rng.rand() < p['prob_complete_treatment_diarrhoea']
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORS, + 2 ORS for mother, GIVE ZINC SUPPLEMENT (age dependent)
    #     # fever
    #     if (df.at[person_id, 'sy_fever'] & df.at[person_id, 'ccm_assessed_fever'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE LA (age dependent-dose), GIVE PARACETAMOL (age dependent)
    #     # fast breathing
    #     if (df.at[person_id, 'ccm_id_fast_breathing'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ORAL ANTIBIOTIC (age dependent-dose)
    #     # red eye
    #     if (df.at[person_id, 'sy_red_eyes'] & df.at[person_id, 'ccm_assessed_red_eyes'] &
    #         df.at[person_id, 'ccm_referral_decision'] == 'home treatment'):
    #         # TODO: INTERACTION WITH HEALTH SYSTEM CODE HERE ---- GIVE ANTIBIOTIC EYE OINTMENT
    #
    #     # ----------------------------------------------------------------------------------------------------
    #     # GET ALL THE CORRECT ICCM ACTION PLAN
    #     # ----------------------------------------------------------------------------------------------------
    #     # get all the danger signs
    #     if (df.at[person_id, 'ds_cough_more_than_21days'] | (
    #         df.at[person_id, df.gi_diarrhoea_acute_type] == 'dysentery') |
    #         (df.at[person_id, df.gi_diarrhoea_type] == 'persistent') | df.at[person_id, 'ds_cough_more_than_21days'] |
    #         df.at[person_id, 'sy_fever_over_7days'] | df.at[person_id, 'sy_red_eyes_over_4days'] | df.at[
    #             person_id, 'ds_convulsions'] |
    #         df.at[person_id, 'ds_not_able_to_drink_or_breastfeed'] | df.at[person_id, 'ds_vomiting_everything'] |
    #         df.at[person_id, 'ds_unusually_sleepy_unconscious'] | df.at[person_id, 'ds_chest_indrawing'] |
    #         df.at[person_id, 'ds_palmar_pallor'] |
    #         df.at[person_id, 'ds_red_MUAC_strap'] | df.at[person_id, 'ds_swelling_both_feet']):
    #         df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] = True
    #
    #     # any danger sign to be referred
    #     if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
    #         (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
    #         (df.at[person_id, 'sy_fever'] == False) & (df.at[person_id, 'sy_diarrhoea'] == False) &
    #         (df.at[person_id, 'sy_chest_indrawing'] == False) & (df.at[person_id, 'sy_fast_breathing'] == False)):
    #         df.at[person_id, 'ccm_correct_action_plan'] = True
    #     # for fever + danger sign
    #     if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
    #         (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') & df.at[person_id, 'sy_fever'] &
    #         df.at[person_id, 'pre-referral treatment given']):
    #         df.at[person_id, 'ccm_correct_action_plan'] = True
    #     # for diarrhoea + danger sign
    #     if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
    #         (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') & df.at[person_id, 'sy_diarrhoea'] &
    #         df.at[person_id, 'pre-referral treatment given']):
    #         df.at[person_id, 'ccm_correct_action_plan'] = True
    #     # for fast breathing + danger sign
    #     if (df.at[person_id, 'presenting_at_least_one_ccm_danger_sign_symptom'] &
    #         (df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
    #         (df.at[person_id, 'sy_fast_breathing'] | df.at[person_id, 'sy_chest_indrawing']) &
    #         df.at[person_id, 'pre-referral treatment given']):
    #         df.at[person_id, 'ccm_correct_action_plan'] = True
    #     # for red eye for 4 days or more
    #     if ((df.at[person_id, 'ccm_referral_decision'] == 'referred to health facility') &
    #         df.at[person_id, 'sy_red_eye_over_4_days'] & df.at[person_id, 'pre-referral treatment given']):
    #         df.at[person_id, 'ccm_correct_action_plan'] = True
    #
    #     # # # # # # # # # # # # FOURTH, CHECK VACCINES RECEIVED # # # # # # # # # # # #
    #     # TODO: complete later with Tara's vaccine code
    #     HSA_checked_vaccines_received = self.module.rng.rand() < p['prob_check_vaccines_status']
    #     if HSA_checked_vaccines_received:
    #         if ((df.at[person_id, 'vacc_DHH1'] == False | df.at[person_id, 'vacc_OPV1']) &
    #             df.at[person_id, df.age_exact_years == 56/487]):
    #             HSA_advise_on_vaccine_schedule = self.module.rng.rand() < p['prob_advise_vaccination']
    #         if df.at[person_id, 'vacc_DHH2'] == False & df.at[person_id, df.age_exact_years == 280/1461]:
    #
    #         if df.at[person_id, 'vacc_DHH3'] == False & df.at[person_id, df.age_exact_years == 392/1461]:
    #
    #     # # # # # # # # # # # # FIFTH, FOLLOW UP # # # # # # # # # # # #
    #
    # def do_management_of_child_at_facility_level_1(self, person_id, hsi_event):
    #     logger.debug('This is HSI_IMNCI, a first appointment for person %d in the health centre',person_id)
    #
    #     df = self.sim.population.props
    #     p = self.module.parameters
    #
    #     # THIS EVENT CHECKS FOR ALL SICK CHILDREN PRESENTING FOR CARE AT AN OUTPATIENT HEALTH FACILITY
    #
    #     # cough_or_difficult_breathing = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
    #     #                                         df.ALL the symptoms related to cough/difficult breathing]
    #     diarrhoea_present = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
    #                                             df.gi_diarrhoea_status]
    #     presenting_any_general_danger_signs = df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
    #                                             df.ds_convulsion | df.ds_not_able_to_drink_or_breastfeed |
    #                                                    df.ds_vomits_everything | df.ds_unusually_sleepy_unconscious]
    #     children_with_severe_pneumonia_or_very_sev_disease = \
    #         df.index[df.is_alive & (df.age_exact_years > 1.667 & df.age_exact_years < 5) &
    #                  (df.ri_pneumonia_severity == 'severe pneumonia') | (
    #                      df.ri_pneumonia_severity == 'very severe pneumonia')]
    #
    #     # first for *ALL SICK CHILDREN*, health worker should check for general danger signs + the 5 main symptoms:
    #     # cough/difficult breathing, diarrhoea, fever, ear problem, malnutrition and anaemia (even if not reported by mother)
    #
    #     # ASSESSMENT OF GENERAL DANGER SIGNS - check if each sign were a component of the consultation
    #     convulsions_checked_by_health_worker = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_convulsions'],
    #                                                         (1 - p['prob_checked_convulsions'])])
    #     inability_to_drink_breastfeed_checked_by_health_worker = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_not_able_to_drink_or_breastfeed'],
    #                                                         (1 - p['prob_checked_not_able_to_drink_or_breastfeed'])])
    #     vomiting_everything_checked_by_health_worker = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_vomits_everything'],
    #                                                         (1 - p['prob_checked_vomits_everything'])])
    #     # unusually_sleepy_unconscious_checked_by_health_worker = \
    #         # self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_unusually_sleepy_unconscious'],
    #                                                         #(1 - p['prob_checked_unusually_sleepy_unconscious'])])
    #     # Let's assume for now that checked danger sign = correct identification of the danger sign
    #     # health worker has identified at least one general danger sign
    #     imci_at_least_one_danger_sign_identified = \
    #         presenting_any_general_danger_signs & \
    #         (convulsions_checked_by_health_worker[True] | inability_to_drink_breastfeed_checked_by_health_worker[True] |
    #         vomiting_everything_checked_by_health_worker[True] \
    #          | unusually_sleepy_unconscious_checked_by_health_worker[True])
    #
    #     # # # # # HEALTH WORKER CHECKS FOR 5 MAIN SYMPTOMS IN IMCI # # # # #
    #
    #     # ASSESSMENT OF MAIN SYMPTOMS (3) - checked for the presence of each main symptom at the consultation:
    #     # cough or difficult breathing, diarrhoea, and fever
    #     health_worker_asked_about_cough_or_difficult_breathing = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_cough_difficult_breathing'], # prob=74%
    #                                                       (1 - p['prob_asked_cough_difficult_breathing'])])
    #     health_worker_asked_about_diarrhoea = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_diarrhoea'], # prob=39%
    #                                                       (1 - p['prob_asked_diarrhoea'])])
    #     health_worker_asked_about_fever = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_fever'], # prob=77%
    #                                                       (1 - p['prob_asked_fever'])])
    #     # Assessment of all 3 main symptoms
    #     health_worker_assessed_all_3_main_symptoms = \
    #         health_worker_asked_about_cough_or_difficult_breathing & health_worker_asked_about_diarrhoea & \
    #         health_worker_asked_about_fever # this probability is 24% according to SPA 2013-14
    #
    #     # Assessment of ear problems - ear pain or discharge
    #     health_worker_asked_about_ear_problem = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_ear_problem'], # 5%
    #                                                       (1 - p['prob_asked_ear_problem'])])
    #     if health_worker_asked_about_ear_problem[True]:
    #         HCW_looked_for_pus_draining_from_ear = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_looked_for pus_draining_from_ear'],  # 4%
    #                                                           (1 - p['prob_looked_for pus_draining_from_ear'])])
    #         HCW_felt_behind_ear_for_tenderness = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_felt_behind_ear_for_tenderness'],  # 4%
    #                                                           (1 - p['prob_felt_behind_ear_for_tenderness'])])
    #
    #     # health care worker looked for signs of malnutrition and anaemia
    #     HCW_looked_for_visible_severe_wasting = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_visible_severe_wasting'],
    #                                                       (1 - p['prob_checked_for_visible_severe_wasting'])])
    #     HCW_looked_for_palmar_pallor = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_palmar_pallor'], # 24%
    #                                                       (1 - p['prob_checked_for_palmar_pallor'])])
    #     HCW_looked_for_oedema_of_both_feet = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_for_oedema_of_both_feet'], # 8%
    #                                                       (1 - p['prob_checked_for_oedema_of_both_feet'])])
    #     HCW_determined_weight_for_age = \
    #         self.sim.rng.choice([True, False], size=1, p=[p['prob_determined_weight_for_age'],
    #                                                       (1 - p['prob_determined_weight_for_age'])])
    #
    #     # ASSESSMENT PROCESS OF COUGH AND/OR DIFFICULT BREATHING
    #     # there are 4 elements to complete in the assessment of cough or difficult breathing: the duration of illness,
    #     # the count of breaths per minute, to look for chest indrawing, and to listen for stridor in a calm child
    #     # in the algorithm below, the probabilities refer to whether the health worker has completed and correctly
    #     # assessed for each element
    #     if cough_or_difficult_breathing & health_worker_asked_about_cough_or_difficult_breathing[True]:
    #         health_worker_asked_duration_of_illness = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_duration_cough_difficult_breathing'],
    #                                                           (1 - p['prob_asked_duration_cough_difficult_breathing'])])
    #         # if health_worker_asked_duration_of_illness[True] & df.at[person_id, has asthma or tb or whooping cough] or longer than 30 days:
    #         # health_worker_referral_or_hospitalization
    #         health_worker_counted_breaths_per_minute = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_assessed_fast_breathing'], # 16%
    #                                                           (1 - p['prob_assessed_fast_breathing'])])
    #         health_worker_checked_chest_indrawing = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_chest_indrawing'],
    #                                                           (1 - p['prob_assessed_chest_indrawing'])])
    #         health_worker_checked_stridor = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_stridor'],
    #                                                           (1 - p['prob_assessed_stridor'])])
    #         if health_worker_asked_duration_of_illness[True] & \
    #             health_worker_counted_breaths_per_minute[True] \
    #             & health_worker_checked_chest_indrawing[True] & \
    #             health_worker_checked_stridor[True]:
    #             df.at[person_id, df.imci_all_elements_of_cough_or_difficult_breathing_complete] = True
    #
    #         # next, need to consider the sensitivity and specificity of assessing these signs
    #         # for those checked for cough/difficult breathing assessment components, multiply by the sensitivity of the assssement
    #         hw_correctly_assessed_fast_breathing = health_worker_counted_breaths_per_minute[True] * sensitivity and specificity
    #
    #         hw_correctly_assessed_chest_indrawing = health_worker_checked_chest_indrawing[True] * sensitivity and specificity
    #
    #         hw_correctly_assessed_stridor = health_worker_checked_stridor[True] * sensitivity and specificity
    #
    #         # CLASSIFICATION PROCESS
    #         if children_with_severe_pneumonia_or_very_sev_disease & \
    #             (hw_correctly_assessed_chest_indrawing[True] | hw_correctly_assessed_stridor[True] |
    #              imci_at_least_one_danger_sign_identified):
    #             hw_correctly_classified_severe_pneumonia_or_very_sev_disease = \
    #                 self.sim.rng.choice([True, False], size=1,
    #                                     p=[p['prob_correctly_classified_severe_pneumonia'],  # high probability
    #                                        (1 - p['prob_correctly_classified_severe_pneumonia'])])
    #
    #
    #             if hw_correctly_classified_severe_pneumonia_or_very_sev_disease[True]:
    #                 df.at[person_id, df.correct_classification_for_cough_or_difficult_breathing] = True
    #                 df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as severe pneumonia or very severe disease'
    #
    #             if hw_correctly_classified_severe_pneumonia_or_very_sev_disease[False]: # severe pneumonia/disease not correctly classified
    #                 df.at[person_id, df.correct_classification_for_cough_or_difficult_breathing] = False
    #                 df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as pneumonia' # 16.5%
    #                 df.at[person_id, df.classification_for_cough_or_difficult_breathing] = 'classified as no pneumonia'  # 54.4%
    #
    #         if hw_correctly_assessed_fast_breathing[True] &
    #
    #
    #         if df.at[cases, df.imci_any_general_danger_sign | df.chest_indrawing | df.stridor ]:
    #
    #             if health_worker_counted_breath_per_minute[True]:
    #
    #     # ASSESSMENT OF DIARRHOEA
    #     if diarrhoea_present & health_worker_asked_about_cough_or_difficult_breathing[True]:
    #         HCW_asked_duration_of_illness = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_duration_diarrhoea'],
    #                                                           (1 - p['prob_asked_duration_diarrhoea'])])
    #         HCW_asked_blood_in_stool = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_asked_presence_of_blood_in_stool'],
    #                                                           (1 - p['prob_asked_presence_of_blood_in_stool'])])
    #         HCW_checked_lethargic_or_restless = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_lethargic_or_restless'], # this was a danger sign to be checked
    #                                                           (1 - p['prob_checked_lethargic_or_restless'])])
    #         HCW_looked_for_sunken_eyes = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_looked_for_sunken_eyes'],
    #                                                           (1 - p['prob_looked_for_sunken_eyes'])])
    #         HCW_checked_ability_to_drink = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_checked_ability_to_drink'],
    #                                                           (1 - p['prob_checked_ability_to_drink'])])
    #         HCW_pinched_abdomen = \
    #             self.sim.rng.choice([True, False], size=1, p=[p['prob_pinched_abdomen'], # 10%
    #                                                           (1 - p['prob_pinched_abdomen'])])


# --------------------------------------------------------------------------------------------------
#
# --- HSI for specific courses of treatment ---
#
# --------------------------------------------------------------------------------------------------

class HSI_Diarrhoea_Treatment(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        self.TREATMENT_ID = 'Diarrhoea_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id, squeeze_factor):
        logger.debug('Do the action of the HSI')
        pass




