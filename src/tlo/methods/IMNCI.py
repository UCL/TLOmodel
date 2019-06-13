"""
Integrated Management of Neonatal and Childhood Illness (IMNCI) module
Documentation: 04 - Methods Repository/Method_Child_IMNCI.xlsx
"""
import logging

from tlo import Module
from tlo.events import Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IMNCI(Module):

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

# ------------------------ HEALTH SYSTEM INTERACTION EVENTS AT FIRST LEVEL HEALTH FACILITIES --------------------------


class HSI_Sick_Child_Seeks_Care_From_First_Level(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Sick child presents for care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = ['childhood_pneumonia', 'childhood_diarrhoea']

    def apply(self, person_id):
        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_First_Level, a first appointment for person %d '
                     'at the first level health facility', person_id)

        df = self.sim.population.props
        now = self.sim.date

        # all_seeking_care_from_first level health facility_plus refered ones? =
        # df.index[all those seeking care in pneumonia, diarrhoea and malaria modules
        # and maybe also other modules??]

        danger_sign_is_detected = df(person_id, 'pn_any_general_danger_sign' == True) and 'looked_for_unusually_sleepy_unconscious' == True
        # for child in all_seeking_care_from_1st_level_facility:
        if df.at[person_id, 'pn_any_general_danger_sign' == True]:
            if danger_sign_is_detected:

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
