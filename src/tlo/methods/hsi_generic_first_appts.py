"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.

This file contains the HSI events that represent the first contact with the Health System, which are triggered by
the onset of symptoms. Non-emergency symptoms lead to `HSI_GenericFirstApptAtFacilityLevel0` and emergency symptoms
lead to `HSI_GenericEmergencyFirstApptAtFacilityLevel1`.
"""
from collections import namedtuple
from typing import OrderedDict, TYPE_CHECKING

import pandas as pd

from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.care_of_women_during_pregnancy import (
    HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement,
    HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy,
)
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.epilepsy import HSI_Epilepsy_Start_Anti_Epileptic
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.labour import HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms

if TYPE_CHECKING:
    from tlo import Module

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HSI_GenericNonEmergencyFirstAppt(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the first interaction with the health system following
     the onset of non-emergency symptom(s). This is the HSI that is generated by the HealthSeekingBehaviour module. By
     default, it occurs at level '0' but it could occur also at other levels."""

    def __init__(self, module, person_id, facility_level='0'):
        super().__init__(module, person_id=person_id, )

        assert module is self.sim.modules['HealthSeekingBehaviour']

        self.TREATMENT_ID = 'FirstAttendance_NonEmergency'

        self.ACCEPTED_FACILITY_LEVEL = facility_level
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})  # <-- No footprint, as this HSI (mostly just)
        #                                                                  determines which further HSI will be needed
        #                                                                  for this person. In some cases, small bits
        #                                                                  of care are provided (e.g. a diagnosis, or
        #                                                                  the provision of inhaler.).

    def apply(self, person_id, squeeze_factor):
        """Run the actions required during the HSI."""
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        do_at_generic_first_appt_non_emergency(hsi_event=self, squeeze_factor=squeeze_factor)


class HSI_GenericEmergencyFirstAppt(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of emergency symptom(s)."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert module.name in ['HealthSeekingBehaviour', 'Labour', 'PregnancySupervisor', 'RTI']

        self.TREATMENT_ID = 'FirstAttendance_Emergency'
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})  # <-- No footprint, as this HSI (mostly just)
        #                                                                  determines which further HSI will be needed
        #                                                                  for this person. In some cases, small bits
        #                                                                  of care are provided (e.g. a diagnosis, or
        #                                                                  the provision of inhaler.).

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        do_at_generic_first_appt_emergency(hsi_event=self, squeeze_factor=squeeze_factor)


class HSI_EmergencyCare_SpuriousSymptom(HSI_Event, IndividualScopeEventMixin):
    """This is an HSI event that provides Accident & Emergency Care for a person that has spurious emergency symptom."""

    def __init__(self, module, person_id, accepted_facility_level='1a'):
        super().__init__(module, person_id=person_id)
        assert module is self.sim.modules['HealthSeekingBehaviour']

        self.TREATMENT_ID = "FirstAttendance_SpuriousEmergencyCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1})
        self.ACCEPTED_FACILITY_LEVEL = accepted_facility_level  # '1a' in default or '1b' as an alternative

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        else:
            sm = self.sim.modules['SymptomManager']
            sm.change_symptom(person_id, "spurious_emergency_symptom", '-', sm)


def do_at_generic_first_appt_non_emergency(hsi_event: HSI_Event, squeeze_factor):
    """The actions are taken during the non-emergency generic HSI, HSI_GenericFirstApptAtFacilityLevel0."""

    # Make top-level reads of information, to avoid repeat accesses.
    person_id = hsi_event.target
    modules: OrderedDict[str, "Module"] = hsi_event.sim.modules
    schedule_hsi = hsi_event.healthcare_system.schedule_hsi_event
    symptoms = hsi_event.sim.modules["SymptomManager"].has_what(person_id)
    # Create the diagnosis test runner function
    def diagnosis_fn(tests, use_dict: bool = False, report_tried: bool = False):
        return hsi_event.healthcare_system.dx_manager.run_dx_test(
            tests,
            hsi_event=hsi_event,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    # Dynamically create immutable container with the target's details stored.
    # This will avoid repeat DataFrame reads when we call the module-level functions.
    df = hsi_event.sim.population.props
    patient_details = namedtuple("PatientDetails", df.columns)(*df.loc[person_id])

    proposed_df_updates = {}

    for module in modules.values():
        event_info, df_updates = module.do_at_generic_first_appt(
            patient_id=person_id,
            patient_details=patient_details,
            symptoms=symptoms,
            diagnosis_fn=diagnosis_fn,
        )
        # Schedule any requested updates
        for info in event_info:
            event = info[0]
            options = info[1]
            schedule_hsi(event, **options)
        # Record any requested DataFrame updates, but do not implement yet
        # NOTE: |= syntax is only available in Python >=3.9
        proposed_df_updates = {**proposed_df_updates, **df_updates}
    
    # Perform any DataFrame updates that were requested, all in one go.
    df.loc[person_id, proposed_df_updates.keys()] = proposed_df_updates.values()

    # ----------------------------------- ALL AGES -----------------------------------

    if "injury" in symptoms:
        if "RTI" in modules:
            modules["RTI"].do_rti_diagnosis_and_treatment(person_id)

    if "Schisto" in modules:
        modules["Schisto"].do_on_presentation_with_symptoms(
            person_id=person_id, symptoms=symptoms
        )

    if "Malaria" in modules:
        malaria_associated_symptoms = {
            "fever",
            "headache",
            "stomachache",
            "diarrhoea",
            "vomiting",
        }
        if bool(set(symptoms) & malaria_associated_symptoms):
            modules["Malaria"].do_for_suspected_malaria_case(
                person_id=person_id, hsi_event=hsi_event
            )

    if patient_details.age_years <= 5:
        # ----------------------------------- CHILD < 5 -----------------------------------
        if "Diarrhoea" in modules:
            if "diarrhoea" in symptoms:
                modules["Diarrhoea"].do_when_presentation_with_diarrhoea(
                    person_id=person_id, hsi_event=hsi_event
                )

        if "Alri" in modules:
            if ("cough" in symptoms) or ("difficult_breathing" in symptoms):
                modules["Alri"].on_presentation(
                    person_id=person_id, hsi_event=hsi_event
                )

        # Routine assessments
        if "Stunting" in modules:
            modules["Stunting"].do_routine_assessment_for_chronic_undernutrition(
                person_id=person_id
            )

    else:
        # ----------------------------------- ADULT -----------------------------------

        if "Depression" in modules:
            modules["Depression"].do_on_presentation_to_care(
                person_id=person_id, hsi_event=hsi_event
            )

        if "CardioMetabolicDisorders" in modules:
            modules["CardioMetabolicDisorders"].determine_if_will_be_investigated(
                person_id=person_id
            )

        if "Copd" in modules:
            if ("breathless_moderate" in symptoms) or ("breathless_severe" in symptoms):
                modules["Copd"].do_when_present_with_breathless(
                    person_id=person_id, hsi_event=hsi_event
                )


def do_at_generic_first_appt_emergency(hsi_event: HSI_Event, squeeze_factor):
    """
    The actions are taken during the non-emergency generic HSI,
    HSI_GenericEmergencyFirstApptAtFacilityLevel1.
    """

    sim = hsi_event.sim
    rng = hsi_event.module.rng
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
    schedule_hsi = hsi_event.sim.modules["HealthSystem"].schedule_hsi_event
    age = df.at[person_id, 'age_years']

    if 'PregnancySupervisor' in sim.modules:

        # -----  ECTOPIC PREGNANCY  -----
        if df.at[person_id, 'ps_ectopic_pregnancy'] != 'none':
            event = HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(
                module=sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
            schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

        # -----  COMPLICATIONS OF ABORTION  -----
        abortion_complications = sim.modules['PregnancySupervisor'].abortion_complications
        if abortion_complications.has_any([person_id], 'sepsis', 'injury', 'haemorrhage', first=True):
            event = HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(
                module=sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
            schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

    if 'Labour' in sim.modules:
        mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
        labour_list = sim.modules['Labour'].women_in_labour

        if person_id in labour_list:
            la_currently_in_labour = df.at[person_id, 'la_currently_in_labour']
            if (
                la_currently_in_labour &
                mni[person_id]['sought_care_for_complication'] &
                (mni[person_id]['sought_care_labour_phase'] == 'intrapartum')
            ):
                event = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    module=sim.modules['Labour'], person_id=person_id,
                    facility_level_of_this_hsi=rng.choice(['1a', '1b']))
                schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

    if "Depression" in sim.modules:
        sim.modules['Depression'].do_on_presentation_to_care(person_id=person_id,
                                                             hsi_event=hsi_event)

    if "Malaria" in sim.modules:
        if 'severe_malaria' in symptoms:
            sim.modules['Malaria'].do_on_emergency_presentation_with_severe_malaria(person_id=person_id,
                                                                                    hsi_event=hsi_event)

    # ------ CARDIO-METABOLIC DISORDERS ------
    if 'CardioMetabolicDisorders' in sim.modules:
        sim.modules['CardioMetabolicDisorders'].determine_if_will_be_investigated_events(person_id=person_id)

    if "Epilepsy" in sim.modules:
        if 'seizures' in symptoms:
            schedule_hsi(HSI_Epilepsy_Start_Anti_Epileptic(person_id=person_id,
                                                           module=sim.modules['Epilepsy']),
                         priority=0,
                         topen=sim.date,
                         tclose=None)

    if 'severe_trauma' in symptoms:
        if 'RTI' in sim.modules:
            sim.modules['RTI'].do_rti_diagnosis_and_treatment(person_id=person_id)

    if 'Alri' in sim.modules:
        if (age <= 5) and (('cough' in symptoms) or ('difficult_breathing' in symptoms)):
            sim.modules['Alri'].on_presentation(person_id=person_id, hsi_event=hsi_event)

    # ----- spurious emergency symptom -----
    if 'spurious_emergency_symptom' in symptoms:
        event = HSI_EmergencyCare_SpuriousSymptom(
            module=sim.modules['HealthSeekingBehaviour'],
            person_id=person_id
        )
        schedule_hsi(event, priority=0, topen=sim.date)

    if 'Copd' in sim.modules:
        if ('breathless_moderate' in symptoms) or ('breathless_severe' in symptoms):
            sim.modules['Copd'].do_when_present_with_breathless(person_id=person_id, hsi_event=hsi_event)

    # -----  EXAMPLES FOR MOCKITIS AND CHRONIC SYNDROME  -----
    if 'craving_sandwiches' in symptoms:
        event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
            module=sim.modules['ChronicSyndrome'],
            person_id=person_id
        )
        schedule_hsi(event, priority=1, topen=sim.date)

    if 'extreme_pain_in_the_nose' in symptoms:
        event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(
            module=sim.modules['Mockitis'],
            person_id=person_id
        )
        schedule_hsi(event, priority=1, topen=sim.date)
