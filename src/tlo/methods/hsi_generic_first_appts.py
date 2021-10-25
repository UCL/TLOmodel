"""
This file contains the HSI events that are the first contact with the Health System, triggered by the onset of symptoms.
"""

from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.bladder_cancer import (
    HSI_BladderCancer_Investigation_Following_Blood_Urine,
    HSI_BladderCancer_Investigation_Following_pelvic_pain,
)
from tlo.methods.breast_cancer import (
    HSI_BreastCancer_Investigation_Following_breast_lump_discernible,
)
from tlo.methods.care_of_women_during_pregnancy import (
    HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement,
    HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy,
)
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.labour import (
    HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour,
    HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour,
)
from tlo.methods.malaria import (
    HSI_Malaria_complicated_treatment_adult,
    HSI_Malaria_complicated_treatment_child,
    HSI_Malaria_non_complicated_treatment_adult,
    HSI_Malaria_non_complicated_treatment_age0_5,
    HSI_Malaria_non_complicated_treatment_age5_15,
)
from tlo.methods.measles import HSI_Measles_Treatment
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms
from tlo.methods.oesophagealcancer import HSI_OesophagealCancer_Investigation_Following_Dysphagia
from tlo.methods.other_adult_cancers import (
    HSI_OtherAdultCancer_Investigation_Following_early_other_adult_ca_symptom,
)
from tlo.methods.prostate_cancer import (
    HSI_ProstateCancer_Investigation_Following_Pelvic_Pain,
    HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#
#    ** NON-EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericFirstAppt_BaseClass(HSI_Event, IndividualScopeEventMixin):
    """Base Class for the Generic First Appt HSI that can occur at either level 0 or level 1."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by the HealthSeekingBehaviour module
        assert module is self.sim.modules['HealthSeekingBehaviour']

        # Define the necessary information for an HSI as null values
        self.TREATMENT_ID = ''
        self.EXPECTED_APPT_FOOTPRINT = None
        self.ACCEPTED_FACILITY_LEVEL = None
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Run the actions required during the HSI."""
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        do_at_non_emergency_first_hsi(hsi_event=self, squeeze_factor=squeeze_factor)


def do_at_non_emergency_first_hsi(hsi_event, squeeze_factor):
    """The actions taken during the non-emergency generic HSI which is called by the HSI event.
    NB. Currently the same actions are taken irrespective of whether the HSI occurs at level 0 or 1, but
    differentiation could be introduced here in reference to `hsi_event.ACCEPTED_FACILITY_LEVEL`."""

    # Gather useful shortcuts
    sim = hsi_event.sim
    rng = hsi_event.module.rng
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    person = df.loc[person_id]
    symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
    age = person.age_years
    schedule_hsi = hsi_event.sim.modules["HealthSystem"].schedule_hsi_event

    # ----------------------------------- ALL AGES -----------------------------------
    # Consider Measles if rash.
    if 'Measles' in sim.modules:
        if "rash" in symptoms:
            schedule_hsi(
                HSI_Measles_Treatment(
                    person_id=person_id,
                    module=hsi_event.sim.modules['Measles']),
                priority=0,
                topen=hsi_event.sim.date,
                tclose=None)

    # 'Automatic' testing for HIV for everyone attending care:
    #  - suppress the footprint (as it done as part of another appointment)
    #  - do not do referrals if the person is HIV negative (assumed not time for counselling etc).
    if 'Hiv' in sim.modules:
        schedule_hsi(
            HSI_Hiv_TestAndRefer(
                person_id=person_id,
                module=hsi_event.sim.modules['Hiv'],
                suppress_footprint=True,
                do_not_refer_if_neg=True),
            topen=hsi_event.sim.date,
            tclose=None,
            priority=0)

    if age <= 5:
        # ----------------------------------- CHILD <=5 -----------------------------------
        # It's a child: Run the ICMI algorithm for this child:
        # NB. This includes children who are aged 5
        # todo - bring this into the main loop to occur for all persons < 15
        # If one of the symptoms is diarrhoea, then run the algorithm for when a child presents with diarrhoea:
        if 'diarrhoea' in symptoms:
            if 'Diarrhoea' in sim.modules:
                sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
                    person_id=person_id, hsi_event=hsi_event)

    if age < 5:
        # ----------------------------------- CHILD < 5 -----------------------------------
        if "Malaria" in sim.modules:
            if 'fever' in symptoms:
                malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                    person_id=person_id, hsi_event=hsi_event)

                # Treat / refer based on diagnosis
                if malaria_test_result == "severe_malaria":
                    schedule_hsi(
                        HSI_Malaria_complicated_treatment_child(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

                elif malaria_test_result == "clinical_malaria":
                    schedule_hsi(
                        HSI_Malaria_non_complicated_treatment_age0_5(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

    elif age < 15:
        # ----------------------------------- CHILD 5-14 -----------------------------------
        if 'fever' in symptoms and "Malaria" in sim.modules:
            malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                person_id=person_id, hsi_event=hsi_event)

            # Treat / refer based on diagnosis
            if malaria_test_result == "severe_malaria":
                schedule_hsi(
                    HSI_Malaria_complicated_treatment_child(
                        person_id=person_id,
                        module=sim.modules["Malaria"]),
                    priority=1,
                    topen=sim.date,
                    tclose=None)

            elif malaria_test_result == "clinical_malaria":
                schedule_hsi(
                    HSI_Malaria_non_complicated_treatment_age5_15(
                        person_id=person_id,
                        module=sim.modules["Malaria"]),
                    priority=1,
                    topen=sim.date,
                    tclose=None)

    else:
        # ----------------------------------- ADULT -----------------------------------
        if 'OesophagealCancer' in sim.modules:
            # If the symptoms include dysphagia, then begin investigation for Oesophageal Cancer:
            if 'dysphagia' in symptoms:
                schedule_hsi(
                    HSI_OesophagealCancer_Investigation_Following_Dysphagia(
                        person_id=person_id,
                        module=sim.modules['OesophagealCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None
                )

        if 'BladderCancer' in sim.modules:
            # If the symptoms include blood_urine, then begin investigation for Bladder Cancer:
            if 'blood_urine' in symptoms:
                schedule_hsi(
                    HSI_BladderCancer_Investigation_Following_Blood_Urine(
                        person_id=person_id,
                        module=sim.modules['BladderCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None
                )

            # If the symptoms include pelvic_pain, then begin investigation for Bladder Cancer:
            if 'pelvic_pain' in symptoms:
                schedule_hsi(
                    HSI_BladderCancer_Investigation_Following_pelvic_pain(
                        person_id=person_id,
                        module=sim.modules['BladderCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'ProstateCancer' in sim.modules:
            # If the symptoms include urinary, then begin investigation for prostate cancer:
            if 'urinary' in symptoms:
                schedule_hsi(
                    HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms(
                        person_id=person_id,
                        module=sim.modules['ProstateCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

            if 'pelvic_pain' in symptoms:
                schedule_hsi(
                    HSI_ProstateCancer_Investigation_Following_Pelvic_Pain(
                        person_id=person_id,
                        module=sim.modules['ProstateCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'OtherAdultCancer' in sim.modules:
            if 'early_other_adult_ca_symptom' in symptoms:
                schedule_hsi(
                    HSI_OtherAdultCancer_Investigation_Following_early_other_adult_ca_symptom(
                        person_id=person_id,
                        module=sim.modules['OtherAdultCancer']
                    ),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'BreastCancer' in sim.modules:
            # If the symptoms include breast lump discernible:
            if 'breast_lump_discernible' in symptoms:
                schedule_hsi(
                    HSI_BreastCancer_Investigation_Following_breast_lump_discernible(
                        person_id=person_id,
                        module=sim.modules['BreastCancer'],
                    ),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'Depression' in sim.modules:
            depr = sim.modules['Depression']
            if (squeeze_factor == 0.0) and (rng.rand() <
                                            depr.parameters['pr_assessed_for_depression_in_generic_appt_'
                                                            'level1']):
                depr.do_when_suspected_depression(person_id=person_id, hsi_event=hsi_event)

        if "Malaria" in sim.modules:
            if 'fever' in symptoms:
                malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                    person_id=person_id, hsi_event=hsi_event)

                if malaria_test_result == "severe_malaria":
                    schedule_hsi(
                        HSI_Malaria_complicated_treatment_adult(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

                elif malaria_test_result == "clinical_malaria":
                    schedule_hsi(
                        HSI_Malaria_non_complicated_treatment_adult(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)


class HSI_GenericFirstApptAtFacilityLevel0(HSI_GenericFirstAppt_BaseClass):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of non-emergency symptom(s). It occurs at level 0."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel0'
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})


class HSI_GenericFirstApptAtFacilityLevel1(HSI_GenericFirstAppt_BaseClass):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of non-emergency symptom(s). It occurs at level 1."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel1'
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            'Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD': 1})


# ---------------------------------------------------------------------------------------------------------
#
#    ** EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericEmergencyFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of emergency symptom(s). It occurs at level 1."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert module.name in ['HealthSeekingBehaviour', 'Labour', 'PregnancySupervisor']

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericEmergencyFirstApptAtFacilityLevel1'
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            'Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD': 1})

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        health_system = self.sim.modules["HealthSystem"]

        if 'PregnancySupervisor' in self.sim.modules:

            # -----  ECTOPIC PREGNANCY  -----
            if df.ps_ectopic_pregnancy != 'none':
                event = HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(
                    module=self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)

            # -----  COMPLICATIONS OF ABORTION  -----
            abortion_complications = self.sim.modules['PregnancySupervisor'].abortion_complications
            if abortion_complications.has_any([person_id], 'sepsis', 'injury', 'haemorrhage', first=True):
                event = HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(
                    module=self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)

        if 'Labour' in self.sim.modules:
            mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
            labour_list = self.sim.modules['Labour'].women_in_labour

            # -----  COMPLICATION DURING BIRTH  -----
            if person_id in labour_list:
                if (
                    df.la_currently_in_labour &
                    mni[person_id]['sought_care_for_complication'] &
                    (mni[person_id]['sought_care_labour_phase'] == 'intrapartum')
                ):
                    event = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                        module=self.sim.modules['Labour'], person_id=person_id,
                        facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                    health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)

                # -----  COMPLICATION AFTER BIRTH  -----
                if (
                    person.la_currently_in_labour &
                    mni[person_id]['sought_care_for_complication'] &
                    (mni[person_id]['sought_care_labour_phase'] == 'postpartum')
                ):
                    event = HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(
                        module=self.sim.modules['Labour'], person_id=person_id,
                        facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                    health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)

        if "Depression" in self.sim.modules:
            if 'Injuries_From_Self_Harm' in symptoms:
                self.sim.modules['Depression'].do_when_suspected_depression(person_id=person_id, hsi_event=self)
                # TODO: Trigger surgical care for injuries.

        if 'Hiv' in self.sim.modules:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

        if "Malaria" in self.sim.modules:
            # Quick diagnosis algorithm - just perfectly recognises the symptoms of severe malaria
            sev_set = {"acidosis",
                       "coma_convulsions",
                       "renal_failure",
                       "shock",
                       "jaundice",
                       "anaemia"}

            # if person's symptoms are on severe malaria list then consider treatment for malaria
            any_symptoms_indicative_of_severe_malaria = len(sev_set.intersection(symptoms)) > 0

            if any_symptoms_indicative_of_severe_malaria:
                # Check if malaria parasitaemia:
                malaria_test_result = self.sim.modules["Malaria"].check_if_fever_is_caused_by_malaria(
                    person_id=person_id, hsi_event=self)

                # if any symptoms indicative of malaria and they have parasitaemia (would return a positive rdt)
                if malaria_test_result in ("severe_malaria", "clinical_malaria"):
                    # Launch the HSI for treatment for Malaria - choosing the right one for adults/children
                    if person.age_years < 5.0:
                        health_system.schedule_hsi_event(
                            hsi_event=HSI_Malaria_complicated_treatment_child(
                                self.sim.modules["Malaria"], person_id=person_id
                            ),
                            priority=0,
                            topen=self.sim.date,
                        )
                    else:
                        health_system.schedule_hsi_event(
                            hsi_event=HSI_Malaria_complicated_treatment_adult(
                                self.sim.modules["Malaria"], person_id=person_id
                            ),
                            priority=0,
                            topen=self.sim.date,
                        )

        # -----  EXAMPLES FOR MOCKITIS AND CHRONIC SYNDROME  -----
        if 'craving_sandwiches' in symptoms:
            event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
                module=self.sim.modules['ChronicSyndrome'],
                person_id=person_id
            )
            health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)

        if 'extreme_pain_in_the_nose' in symptoms:
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(
                module=self.sim.modules['Mockitis'],
                person_id=person_id
            )
            health_system.schedule_hsi_event(event, priority=1, topen=self.sim.date)
