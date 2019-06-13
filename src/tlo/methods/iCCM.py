"""
Integrated Community Case Management of Childhood Illness (iCCM) module
Documentation: 04 - Methods Repository/Method_Child_iCCM.xlsx
"""
import logging

from tlo import Module, Property, Types
from tlo.events import Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ICCM(Module):

    PROPERTIES = {
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
    }

# -------------------------- HEALTH SYSTEM INTERACTION EVENTS IN THE COMMUNITY ---- ICCM ----------------------------


class HSI_Sick_Child_Seeks_Care_From_HSA(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Sick child presents for care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = ['childhood_pneumonia', 'childhood_diarrhoea']

    def apply(self, person_id):
        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        now = self.sim.date

        # stepone : work out if the child has 'malaria'
        has_malaria = df.atperson_id,'Malaria'[]
        sens_dx_malaria= 0.9
        spec_dx_malaria = 0.8
        snes_dx_pneumo = 0.7
        spec_dx_pneuo = 0.5

        _ bad_CHW = send_dx_malria_good_CHW * 0.5*bad_CHW
        -bdad_CHW = 0.2

        good_CHW = self.rng.rand < prob_good_CHW

        - will the child be diagonosed by the algogorith
        correctly_diagnoed_malaria = has_malria and self.rng.rand<sens_dx_malaria
        missed_diagnosed_malaria = has malria and not correctly_diagnoed
        false_positive_diagnosed_malria = (not has_malaria and self.rng.rand<(1-spec_dx_malaria)

        correxctly_dianogsed_pneo ...
        correclty_fianoged _pneu




        # all_seeking_care_from_HSA = df.index[all those seeking care in pneumonia, diarrhoea and malaria modules
        # and maybe also other modules??]
        # for child in all_seeking_care_from_HSA:
        if df.at[person_id, 'pn_any_general_danger_sign' == True]:
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

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
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

