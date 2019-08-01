import logging

from tlo import Property, Parameter, Types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class iCCM(Property, Parameter):
    PROPERTIES = {
        'correctly_identified_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one danger sign'
         ),
        'correctly_identified_fast_breathing_and_cough':Property
        (Types.BOOL, 'HSA correctly identified fast breathing for age and cough'
         ),
        'correctly_identified_diarrhoea_and_dehydration': Property
        (Types.BOOL, 'HSA correctly identified diarrhoea and dehydration'
         ),
        'correctly_identified_fever_performed_RDT': Property
        (Types.BOOL, 'HSA correctly identified fever and performed RDT test'
         ),
        'correctly_classified_severe_pneumonia': Property
        (Types.BOOL, 'HSA correctly classified as fast breathing with danger sign and RDT negative (severe pneumonia)'
         ),
        'correctly_classified_pneumonia': Property
        (Types.BOOL, 'HSA correctly classified as fast breathing for age, no danger sign and RDT negative (pneumonia)'
         ),
        'correctly_classified_cough_over_21days': Property
        (Types.BOOL, 'HSA correctly noted cough for more than 3 weeks'
         ),
        'correctly_classified_common_cold_or_cough': Property
        (Types.BOOL, 'HSA correctly classified as common cold or cough with no fast breathing'
         ),
        'correctly_classified_diarrhoea_with_danger_sign': Property
        (Types.BOOL, 'HSA correctly classified as diarrhoea with danger sign or with signs of severe dehydration'
         ),
        'correctly_classified_persistent_or_bloody_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified persistent diarrhoea or dysentery'
         ),
        'correctly_classified_diarrhoea': Property
        (Types.BOOL, 'HSA correctly classified diarrhoea without blood and less than 14 days'
         ),
        'correctly_classified_severe_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever with 1 or more danger sign, positive RDT (severe malaria)'
         ),
        'correctly_classified_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever for 7 days or more, no danger sign, positive RDT (malaria)'
         ),
        'correctly_classified_uncomplicated_malaria': Property
        (Types.BOOL, 'HSA correctly classified fever for less than 7 days, no danger sign, positive RDT'
                     ' (uncomplicated malaria)'
         ),
        'correctly_classified_as_other_illness': Property
        (Types.BOOL,
         'HSA correctly classified fever, no danger sign, negative RDT (other illness)'
         ),
        'referral_options': Property
        (Types.CATEGORICAL,
         'Referral decisions', categories=['refer immediately', 'refer to health facility', 'do not refer']),
        'correct_referral_decision': Property
        (Types.BOOL,
         'HSA made the correct referral decision based on the assessment and classification process'
         ),
        'correct_treatment_and_advice_given': Property
        (Types.BOOL,
         'HSA has given the correct treatment for the classified condition'
         ),
    }


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

    def read_parameters(self, data_folder):

        p = self.parameters

        p['base_prev_dysentery'] = 0.3

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

"""
