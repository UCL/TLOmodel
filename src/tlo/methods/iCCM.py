"""
Integrated Community Case Management of Childhood Illness (iCCM) module
Documentation: 04 - Methods Repository/Method_Child_iCCM.xlsx
"""
import logging

import pandas as pd
from tlo.events import Event, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# -------------------------- HEALTH SYSTEM INTERACTION EVENTS IN THE COMMUNITY ---- ICCM ----------------------------


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

    def apply(self, person_id):

        logger.debug('This is HSI_Sick_Child_Seeks_Care_From_HSA, a first appointment for person %d in the community',
                     person_id)

        df = self.sim.population.props

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

