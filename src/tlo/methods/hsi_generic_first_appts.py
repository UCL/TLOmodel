"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel0, which describes the first interaction with
the health system following the onset of acute generic symptoms.
"""

from tlo.events import IndividualScopeEventMixin
from tlo.methods.healthsystem import HSI_Event
from tlo.population import logger


class HSI_GenericFirstApptAtFacilityLevel0(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

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
            the_appt_footprint['Over5OPD'] = 1.0  # Child out-patient appointment
            #TODO: this should be the_appt_footprint['Under5OPD'] = 1  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel0'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel0 for person %d', person_id)
        pass

    def did_not_run(self):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel0: did not run')

        diagnosis = self.sim.modules['DxAlgortihmChild'].diagnose(person_id=self.person_id, hsi_event=self)

        # Work out what to do with this person....

        if self.sim.population.props.at[self.person_id,'age_years'] < 5.0:
            # It's a child:
            logger.debug('Run the ICMI algorithm for this child')

            # Get the diagnosis from the algorithm
            diagnosis = self.sim.modules['DxAlgortihmChild'].diagnose(person_id=self.person_id,hsi_event=self)

        else:
            # It's an adult
            logger.debug('To fill in ... what to with an adult')
