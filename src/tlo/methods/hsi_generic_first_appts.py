"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.
"""

from tlo.events import IndividualScopeEventMixin
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms
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

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel1 for person %d', person_id)

        diagnosis = self.module.sim.modules['DxAlgorithmChild'].diagnose(person_id=person_id, hsi_event=self)

        # Work out what to do with this person....
        if self.sim.population.props.at[person_id, 'age_years'] < 5.0:
            # It's a child:
            logger.debug('Run the ICMI algorithm for this child')

            # Get the diagnosis from the algorithm
            diagnosis = self.module.sim.modules['DxAlgorithmChild'].diagnose(person_id=person_id, hsi_event=self)

            # Do something based on this diagnosis...
            if diagnosis == 'measles':
                logger.info('Start treatment for measles')
            else:
                logger.info('No treatment. HSI ends.')

        else:
            # It's an adult
            logger.debug('To fill in ... what to with an adult')

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

        # simple diagnosis to work out which HSI event to trigger
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        if 'em_craving_sandwiches' in symptoms:
            event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(self.module, person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                event, priority=1, topen=self.sim.date, tclose=None
            )

        elif 'em_extreme_pain_in_the_nose' in symptoms:
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
        else:
            logger.debug('Another symptom must have caused this emergency appointment')

        pass

    def did_not_run(self):
        logger.debug('HSI_GenericEmergencyFirstApptAtFacilityLevel1: did not run')
        pass
