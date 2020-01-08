"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.
"""

from tlo.events import IndividualScopeEventMixin
from tlo.methods.healthsystem import HSI_Event
from tlo.methods import malaria
from tlo.methods.malaria import HSI_Malaria_tx_compl_child, HSI_Malaria_tx_compl_adult
from tlo.population import logger
from tlo import DateOffset

# ---------------------------------------------------------------------------------------------------------
#
#    ** NON-EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericFirstApptAtFacilityLevel1
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 1

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

        # todo: expected appt footprint

    def apply(self, person_id, squeeze_factor):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel1 for person %d', person_id)

        df = self.sim.population.props

        # NOTES: this section is repeated from the malaria.HSI_Malaria_rdt
        # requests for comsumables occur inside the HSI_treatment events
        # perhaps requests also need to occur here in case alternative treatments need to be scheduled

        # make sure query consumables has the generic hsi as the module requesting

        # ----------------------------------- CHILD <5 -----------------------------------

        # diagnostic algorithm for child <5 yrs
        if df.at[person_id, 'age_years'] < 5.0:
            # It's a child:
            logger.debug('Run the ICMI algorithm for this child [dx_algorithm_child]')

            # Get the diagnosis from the algorithm
            diagnosis = self.sim.modules['DxAlgorithmChild'].diagnose(person_id=person_id, hsi_event=self)

            # Treat / refer based on diagnosis
            if diagnosis == 'severe_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_compl_child(self.sim.modules['Malaria'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            elif diagnosis == 'clinical_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_0_5 for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_0_5(self.sim.modules['Malaria'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            else:
                logger.debug(
                    'HSI_GenericFirstApptAtFacilityLevel1: negative / no malaria test for person %d so doing nothing',
                    person_id)

        # ----------------------------------- CHILD 5-15 -----------------------------------

        # diagnostic algorithm for child 5-15 yrs
        if (df.at[person_id, 'age_years'] >= 5) & (df.at[person_id, 'age_years'] < 15):
            # It's a child:
            logger.debug('Run the ICMI algorithm for this child [dx_algorithm_child]')

            # Get the diagnosis from the algorithm
            diagnosis = self.sim.modules['DxAlgorithmChild'].diagnose(person_id=person_id,
                                                                      hsi_event=self)

            # Treat / refer based on diagnosis
            if diagnosis == 'severe_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_compl_child(self.sim.modules['Malaria'],
                                                           person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            elif diagnosis == 'clinical_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_5_15 for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_5_15(self.sim.modules['Malaria'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            else:
                logger.debug(
                    'HSI_GenericFirstApptAtFacilityLevel1: negative / no malaria test for person %d so doing nothing',
                    person_id)

        # ----------------------------------- ADULT -----------------------------------

        # diagnostic algorithm for adult
        if df.at[person_id, 'age_years'] >= 15:
            # It's an adult:
            logger.debug('Run the diagnostic algorithm for this adult [dx_algorithm_adult]')

            # Get the diagnosis from the algorithm
            diagnosis = self.sim.modules['DxAlgorithmAdult'].diagnose(person_id=person_id,
                                                                      hsi_event=self)

            # Treat / refer based on diagnosis
            if diagnosis == 'severe_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_compl_adult for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_compl_adult(self.sim.modules['Malaria'],
                                                           person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            elif diagnosis == 'clinical_malaria':

                logger.debug(
                    "HSI_GenericFirstApptAtFacilityLevel1: scheduling HSI_Malaria_tx_5_15 for person %d on date %s",
                    person_id, (self.sim.date + DateOffset(days=1)))

                treat = malaria.HSI_Malaria_tx_adult(self.sim.modules['Malaria'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                    priority=1,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            else:
                logger.debug(
                    'HSI_GenericFirstApptAtFacilityLevel1: negative / no malaria test for person %d so doing nothing',
                    person_id)

    # todo return actual footprint including LabPOC

    def did_not_run(self):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel1: did not run')


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericFirstApptAtFacilityLevel0
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericFirstApptAtFacilityLevel0(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 0

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ConWithDCSA'] = 1.0  # Consultantion with DCSA

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel0'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel0 for person %d', person_id)

    def did_not_run(self):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel0: did not run')


# ---------------------------------------------------------------------------------------------------------
#
#    ** EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericEmergencyFirstApptAtFacilityLevel1
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericEmergencyFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 1

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericEmergencyFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericEmergencyFirstApptAtFacilityLevel1 for person %d', person_id)

        # Quick diagnosis algorithm - just perfectly recognises the symptoms of severe malaria
        if 'em_coma' in self.sim.modules['SymptomManager'].has_what(person_id):
            if 'Malaria' in self.sim.population.props.at[person_id, 'sy_em_coma']:
                # Launch the HSI for treatment for Malaria - choosing the right one for adults/children
                if self.sim.population.props.at[person_id,'age_years']<5.0:
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Malaria_tx_compl_child(self.sim.modules['malaria'], person_id=person_id),
                        priority=0,
                        topen=self.sim.date
                    )
                else:
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Malaria_tx_compl_adult(self.sim.modules['malaria'], person_id=person_id),
                        priority=0,
                        topen=self.sim.date
                    )
            print('its working')

    def did_not_run(self):
        logger.debug('HSI_GenericEmergencyFirstApptAtFacilityLevel1: did not run')
