"""
Integrated Management of Neonatal and Childhood Illness (IMNCI) module
Documentation: 04 - Methods Repository/Method_Child_IMNCI.xlsx
"""
import logging

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

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass


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

        # # # # # ASSESS GENERAL DANGER SIGNS # # # # #
