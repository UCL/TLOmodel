"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.
"""
from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.labour import (
    HSI_Labour_PresentsForSkilledBirthAttendanceInLabour,
    HSI_Labour_ReceivesCareForPostpartumPeriod,
)
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms
from tlo.methods.oesophagealcancer import HSI_OesophagealCancer_Investigation_Following_Dysphagia

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        # Confirm that this appointment has been created by the HealthSeekingBehaviour module
        assert module is self.sim.modules['HealthSeekingBehaviour']

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        if is_child:
            the_appt_footprint = self.make_appt_footprint({'Under5OPD': 1})  # Child out-patient appointment
        else:
            the_appt_footprint = self.make_appt_footprint({'Over5OPD': 1})   # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel1 for person %d', person_id)

        # Work out what to do with this person....
        if self.sim.population.props.at[person_id, 'age_years'] < 5.0:
            # It's a child and we are in FacilityLevel1, so run the the child management routine:
            symptoms = self.sim.modules['SymptomManager'].has_what(person_id=person_id)

            # If one of the symptoms is diarrhoea, then run the diarrhoea for a child routine:
            if 'diarrhoea' in symptoms:
                self.sim.modules['DxAlgorithmChild'].do_when_diarrhoea(person_id=person_id, hsi_event=self)

        else:
            # It's an adult
            logger.debug('To fill in ... what to with an adult')

            symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

            # If the symptoms include dysphagia, then begin investigation for Oesophageal Cancer:
            if 'dysphagia' in symptoms:
                hsi_event = HSI_OesophagealCancer_Investigation_Following_Dysphagia(
                    module=self.sim.modules['OesophagealCancer'],
                    person_id=person_id,
                )
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event,
                    priority=0,
                    topen=self.sim.date,
                    tclose=None
                )

                # If the symptoms include blood_urine, then begin investigation for Bladder Cancer:
                if 'blood_urine' in symptoms:
                    hsi_event = HSI_BladderCancer_Investigation_Following_Blood_Urine(
                        module=self.sim.modules['BladderCancer'],
                        person_id=person_id,
                    )
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event,
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )

                # If the symptoms include pelvic_pain, then begin investigation for Bladder Cancer:
                if 'pelvic_pain' in symptoms:
                    hsi_event = HSI_BladderCancer_Investigation_Following_pelvic_pain(
                        module=self.sim.modules['BladderCancer'],
                        person_id=person_id,
                    )
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event,
                        priority=0,
                        topen=self.sim.date,
                        tclose=None
                    )



            # ---- ROUTINE ASSESSEMENT FOR DEPRESSION ----
            if 'Depression' in self.sim.modules:
                depr = self.sim.modules['Depression']
                if (squeeze_factor == 0.0) and (self.module.rng.rand() <
                                                depr.parameters['pr_assessed_for_depression_in_generic_appt_level1']):
                    depr.do_when_suspected_depression(person_id=person_id, hsi_event=self)
            # -------------------------------

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

        # Confirm that this appointment has been created by the HealthSeekingBehaviour module
        assert module is self.sim.modules['HealthSeekingBehaviour']

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel0'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})
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

        # Confirm that this appointment has been created by the HealthSeekingBehaviour module or Labour module
        assert module.name in ['HealthSeekingBehaviour', 'Labour']

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        if is_child:
            the_appt_footprint = self.make_appt_footprint({'Under5OPD': 1})  # Child out-patient appointment
        else:
            the_appt_footprint = self.make_appt_footprint({'Over5OPD': 1})   # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericEmergencyFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericEmergencyFirstApptAtFacilityLevel1 for person %d', person_id)
        df = self.sim.population.props
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        if 'Labour' in self.sim.modules:
            mni = self.sim.modules['Labour'].mother_and_newborn_info
            labour_list = self.sim.modules['Labour'].women_in_labour

            # -----  COMPLICATION DURING BIRTH  -----
            if person_id in labour_list:
                if df.at[person_id, 'la_currently_in_labour'] & (mni[person_id]['sought_care_for_complication']) \
                        & (mni[person_id]['sought_care_labour_phase'] == 'intrapartum'):
                    event = HSI_Labour_PresentsForSkilledBirthAttendanceInLabour(
                        module=self.sim.modules['Labour'], person_id=person_id,
                        facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=1, topen=self.sim.date)

            # -----  COMPLICATION AFTER BIRTH  -----
                if df.at[person_id, 'la_currently_in_labour'] & (mni[person_id]['sought_care_for_complication']) \
                        & (mni[person_id]['sought_care_labour_phase'] == 'postpartum'):
                    event = HSI_Labour_ReceivesCareForPostpartumPeriod(
                        module=self.sim.modules['Labour'], person_id=person_id,
                        facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=1, topen=self.sim.date)

        # -----  SUSPECTED DEPRESSION  -----
        if 'Injuries_From_Self_Harm' in symptoms:
            self.sim.modules['Depression'].do_when_suspected_depression(person_id=person_id, hsi_event=self)
            # TODO: Trigger surgical care for injuries.

        # -----  EXAMPLES FOR MOCKITIS AND CHRONIC SYNDROME  -----
        if 'craving_sandwiches' in symptoms:
            event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
                module=self.sim.modules['ChronicSyndrome'],
                person_id=person_id
            )
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,
                                                                topen=self.sim.date
                                                                )

        if 'extreme_pain_in_the_nose' in symptoms:
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(
                module=self.sim.modules['Mockitis'],
                person_id=person_id
            )
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,
                                                                topen=self.sim.date
                                                                )

    def did_not_run(self):
        logger.debug('HSI_GenericEmergencyFirstApptAtFacilityLevel1: did not run')
        return False  # Labour debugging
        # pass

    def not_available(self):
        pass
