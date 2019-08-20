"""
Integrated Community Case Management of Childhood Illness (iCCM) module
Documentation: 04 - Methods Repository/Method_Child_iCCM.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------- HEALTH SYSTEM INTERACTION EVENTS IN THE COMMUNITY ---- ICCM ----------------------------


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
        'ccm_correctly_identified_danger_signs': Property
        (Types.BOOL, 'HSA correctly identified at least one danger sign, including '
                     'convulsions, very sleepy or unconscious, chest in-drawing, vomiting everything, '
                     'not able to drink or breastfeed'
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

         # IMCNI - Integrated Management of Neonatal and Childhood Illnesses algorithm
         ),
    }

    def read_parameters(self, data_folder):
        p = self.parameters

        p['prob_correct_id_diarrhoea_dehydration'] = 0.8
        p['prob_correct_id_danger_sign'] = 0.7
        p['prob_correct_id_fast_breathing_and_cough'] = 0.8
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


class HSI_Pneumonia_Seeks_Care_From_HSA(Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event for pneumonia module.
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

        logger.debug('This is HSI_Pneumonia_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        # ---------------- Work out if child with pneumonia is correctly diagnosed by the HSA ----------------

        current_pneumonia = df.index[df.is_alive & df.ri_pneumonia_status & (df.age_years < 5)]

        for person_id in current_pneumonia:
            fast_breathing_assessed_by_HSA = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_fast_breathing_and_cough'],
                                                              (1 - p['prob_correct_id_fast_breathing_and_cough'])])
            if fast_breathing_assessed_by_HSA[True]:
                df.at[person_id, 'ccm_correctly_assessed_fast_breathing_and_cough'] = True
            if df.at[person_id, 'pn_any_general_danger_sign']: # TODO: have to include stridor
                danger_sign_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_danger_sign'],
                                                                  (1 - p['prob_correct_id_danger_sign'])])
                if df.at[person_id, 'ccm_correctly_assessed_fast_breathing_and_cough'] & danger_sign_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_classified_severe_pneumonia'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'ccm_correct_referral_decision'] = True
                if df.at[person_id, 'ccm_correctly_assessed_fast_breathing_and_cough'] & danger_sign_identified_by_HSA[False]:
                    df.at[person_id, 'ccm_correctly_classified_pneumonia'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'ccm_correct_referral_decision'] = True
            if df.at[person_id, 'pn_fast_breathing'] == False & df.at[person_id, 'pn_cough']:
                df.at[person_id, 'ccm_correctly_classified_common_cold_or_cough'] = True
                HSA_referral_decision = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                  (1 - p['prob_correct_referral_decision'])])
                if HSA_referral_decision[True]:
                    df.at[person_id, 'ccm_correct_referral_decision'] = True
            if df.at[person_id, 'ccm_correct_referral_decision'] & \
                df.at[person_id, 'correctly_classified_diarrhoea_with_danger_sign']:
                df.at[person_id, 'referral_options'] = 'refer immediately'


class HSI_Diarrhoea_Seeks_Care_From_HSA(Event, IndividualScopeEventMixin):
    """ This is the first Health Systems Interaction event for diarrhoea module.
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

        logger.debug('This is HSI_Diarrhoea_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        p = self.module.parameters

        # ---------------- Work out if child with diarrhoea is correctly diagnosed by the HSA ----------------

        current_diarrhoea = df.index[df.is_alive & df.gi_diarrhoea_status & (df.age_years < 5)]

        for person_id in current_diarrhoea:
            diarrhoea_assessed_by_HSA = \
                self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_diarrhoea_dehydration'],
                                                              (1 - p['prob_correct_id_diarrhoea_dehydration'])])
            if diarrhoea_assessed_by_HSA[True]:
                df.at[person_id, 'ccm_correctly_assessed_diarrhoea_and_dehydration'] = True
            if df.at[person_id, 'di_any_general_danger_sign']:
                danger_sign_identified_by_HSA = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_id_danger_sign'],
                                                                  (1 - p['prob_correct_id_danger_sign'])])
                if diarrhoea_assessed_by_HSA[True] & danger_sign_identified_by_HSA[True]:
                    df.at[person_id, 'correctly_classified_diarrhoea_with_danger_sign'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'ccm_correct_referral_decision'] = True
            if (df.at[person_id, 'gi_diarrhoea_acute_type'] == 'dysentery') | (
                df.at[person_id, 'gi_persistent_diarrhoea']):
                persistent_or_bloody_diarr_identified_by_HSA = self.sim.rng.choice([True, False], size=1, p=[
                    p['prob_correct_id_persist_or_bloody_diarrhoea'],
                    (1 - p['prob_correct_id_persist_or_bloody_diarrhoea'])])
                if diarrhoea_assessed_by_HSA[True] & persistent_or_bloody_diarr_identified_by_HSA[True]:
                    df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea'] = True
                    HSA_referral_decision = \
                        self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                      (1 - p['prob_correct_referral_decision'])])
                    if HSA_referral_decision[True]:
                        df.at[person_id, 'ccm_correct_referral_decision'] = True
            if not (df.at[person_id, 'gi_persistent_diarrhoea'] & (
                (df.at[person_id, 'gi_diarrhoea_acute_type']) == 'dysentery') & (
                    df.at[person_id, 'di_any_general_danger_sign'])) & diarrhoea_assessed_by_HSA[True]:
                df.at[person_id, 'ccm_correctly_classified_diarrhoea'] = True
                HSA_referral_decision = \
                    self.sim.rng.choice([True, False], size=1, p=[p['prob_correct_referral_decision'],
                                                                  (1 - p['prob_correct_referral_decision'])])
                if HSA_referral_decision[True]:
                    df.at[person_id, 'ccm_correct_referral_decision'] = True
            if df.at[person_id, 'ccm_correct_referral_decision'] & \
                df.at[person_id, 'ccm_correctly_classified_diarrhoea_with_danger_sign']:
                df.at[person_id, 'ccm_referral_options'] = 'refer immediately'
                # Get the consumables required - give pre-referral treatment with ORS solution
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                event_immediate_referral = HSI_HSA_Diarrhoea_Referred_Immediately(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_immediate_referral,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None)

            if df.at[person_id, 'ccm_correct_referral_decision'] & \
                df.at[person_id, 'ccm_correctly_classified_persistent_or_bloody_diarrhoea']:
                df.at[person_id, 'ccm_referral_options'] = 'refer to health facility'
                # Get the consumables required - give pre-referral treatment with ORS solution
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                the_cons_footprint = {
                    'Intervention_Package_Code': [pkg_code1],
                    'Item_Code': []
                }

                event_referral = HSI_HSA_Diarrhoea_Referred_HealthFacility(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_referral,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

            if df.at[person_id, 'ccm_correct_referral_decision'] & \
                df.at[person_id, 'ccm_correctly_classified_diarrhoea']:
                df.at[person_id, 'ccm_referral_options'] = 'do not refer'
                # Get the consumables required - not referred, treated at home with ORS and zinc
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                pkg_code1 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'ORS',
                                                      'Intervention_Pkg_Code'])[37]
                pkg_code2 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'Zinc for Children 0-6 months',
                                                      'Intervention_Pkg_Code'])[38]
                pkg_code3 = pd.unique(consumables.loc[consumables[
                                                          'Intervention_Pkg'] ==
                                                      'Zinc for Children 6-59 months',
                                                      'Intervention_Pkg_Code'])[39]

            if df.at[person_id, 'ccm_correct_referral_decision']:
                logger.debug(
                    '...This is HSI_Sick_Child_Seeks_Care_From_HSA: \
                    there should now be treatment for person %d',
                    person_id)
                event_treatment_decision = HSI_HSA_Diarrhoea_StartTreatment(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_event(event_treatment_decision)
            else:
                logger.debug(
                    '...This is HSI_Sick_Child_Seeks_Care_From_HSA: there will not be treatment for person %d',
                    person_id)

            target_date_for_followup_appt = self.sim.date + DateOffset(days=5)

            logger.debug(
                '....This is HSI_Sick_Child_Seeks_Care_From_HSA: '
                'scheduling a follow-up appointment for person %d on date %s',
                person_id, target_date_for_followup_appt)

            followup_appt = HSI_HSA_followup_care(self.module, person_id=person_id)

            # Request the heathsystem to have this follow-up appointment
            self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                                priority=2,
                                                                topen=target_date_for_followup_appt,
                                                                tclose=target_date_for_followup_appt + DateOffset(
                                                                    weeks=2))


class HSI_HSA_Diarrhoea_Referred_Immediately(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'HSI_HSA_Diarrhoea_Referred_Immediately'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]  # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, population):
        df = population.props
        m = self.module
        rng = m.rng


class HSI_HSA_Diarrhoea_Referred_HealthFacility(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'HSI_HSA_Diarrhoea_Referred_HealthFacility'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]  # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, population):
        df = population.props
        m = self.module
        rng = m.rng


class HSI_HSA_followup_care(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]  # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, population):
        df = population.props
        m = self.module
        rng = m.rng

class HSI_Sick_Child_Seeks_Care_From_HSA(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Sick_child_presents_for_care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        rng = self.module.rng

        if df.at[person_id, 'gi_diarrhoea_status']:
            logger.debug(
                '...This is HSI_Sick_Child_Seeks_Care_From_HSA: \
                there should now be treatment for person %d',
                person_id)
            event = HSI_HSA_Diarrhoea_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None)

        else:
            logger.debug(
                '...This is HSI_Sick_Child_Seeks_Care_From_HSA: there will not be treatment for person %d',
                person_id)


class HSI_HSA_Diarrhoea_StartTreatment(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation

        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'ORS',
                                              'Intervention_Pkg_Code'])[37]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'Zinc for Children 0-6 months',
                                              'Intervention_Pkg_Code'])[38]
        pkg_code3 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'Zinc for Children 6-59 months',
                                              'Intervention_Pkg_Code'])[39]

        item_code1 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

    def apply(self, person_id):
        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props
        now = self.sim.date

        target_date_for_followup_appt = self.sim.date + DateOffset(days=5)

        logger.debug(
            '....This is HHSI_Sick_Child_Seeks_Care_From_HSA: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_HSA_followup_care(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2))

