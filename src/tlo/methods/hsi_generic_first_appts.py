"""Events which describes the first interaction with the health system.

This module contains the HSI events that represent the first contact with the health
system, which are triggered by the onset of symptoms. Non-emergency symptoms lead to
:py:class:`HSI_GenericNonEmergencyFirstAppt` and emergency symptoms lead to
:py:class:`HSI_GenericEmergencyFirstAppt`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Set, Union

import numpy as np
import pandas as pd

from tlo import Date, Module, logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.hsi_event import HSI_Event

if TYPE_CHECKING:
    from typing import Optional, TypeAlias

    from tlo.methods.dxmanager import DiagnosisTestReturnType
    from tlo.population import IndividualProperties
from tlo.methods.bladder_cancer import (
    HSI_BladderCancer_Investigation_Following_Blood_Urine,
    HSI_BladderCancer_Investigation_Following_pelvic_pain,
)
from tlo.methods.breast_cancer import (
    HSI_BreastCancer_Investigation_Following_breast_lump_discernible,
)
from tlo.methods.cervical_cancer import (
    HSI_CervicalCancerPresentationVaginalBleeding, HSI_CervicalCancer_Screening,
    HSI_CervicalCancer_AceticAcidScreening, HSI_CervicalCancer_XpertHPVScreening
)
from tlo.methods.care_of_women_during_pregnancy import (
    HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement,
    HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy,
)
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.epilepsy import HSI_Epilepsy_Start_Anti_Epileptic
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.labour import HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour
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


DiagnosisFunction: TypeAlias = Callable[[str, bool, bool], Any]
ConsumablesChecker: TypeAlias = Callable[
    [
        Union[None, np.integer, int, List, Set, Dict],
        Union[None, np.integer, int, List, Set, Dict],
    ],
    Union[bool, Dict],
]


class HSIEventScheduler(Protocol):

    def __call__(
        self,
        hsi_event: HSI_Event,
        priority: int,
        topen: Date,
        tclose: Optional[Date] = None,
    ) -> None: ...


class GenericFirstAppointmentsMixin:
    """Mix-in for modules with actions to perform on generic first appointments."""

    def do_at_generic_first_appt(
        self,
        *,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        diagnosis_function: DiagnosisFunction,
        consumables_checker: ConsumablesChecker,
        facility_level: str,
        treatment_id: str,
    ) -> None:
        """
        Actions to take during a non-emergency generic health system interaction (HSI).

        Derived classes should overwrite this method so that they are compatible with
        the :py:class:`~.HealthSystem` module, and can schedule HSI events when a
        individual presents symptoms indicative of the corresponding illness or
        condition.

        When overwriting, arguments that are not required can be left out of the
        definition. If done so, the method **must** take a ``**kwargs`` input to avoid
        errors when looping over all disease modules and running their generic HSI
        methods.

        HSI events should be scheduled by the :py:class:`Module` subclass implementing
        this method using the ``schedule_hsi_event`` argument.

        Implementations of this method should **not** make any updates to the population
        dataframe directly - if the target individuals properties need to be updated
        this should be performed by updating the ``individual_properties`` argument.

        :param person_id: Row index (ID) of the individual target of the HSI event in
            the population dataframe.
        :param individual_properties: Properties of individual target as provided in the
            population dataframe. Updates to individual properties may be written to
            this object.
        :param symptoms: List of symptoms the patient is experiencing.
        :param schedule_hsi_event: A function that can schedule subsequent HSI events.
        :param diagnosis_function: A function that can run diagnosis tests based on the
            patient's symptoms.
        :param consumables_checker: A function that can query the health system to check
            for available consumables.
        :param facility_level: The level of the facility that the patient presented at.
        :param treatment_id: The treatment id of the HSI event triggering the generic
            appointment.
        """

    def do_at_generic_first_appt_emergency(
        self,
        *,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        diagnosis_function: DiagnosisFunction,
        consumables_checker: ConsumablesChecker,
        facility_level: str,
        treatment_id: str,
    ) -> None:
        """
        Actions to take during an emergency generic health system interaction (HSI).

        Call signature is identical to the
        :py:meth:`~GenericFirstAppointmentsMixin.do_at_generic_first_appt` method.

        Derived classes should overwrite this method so that they are compatible with
        the :py:class`~.HealthSystem` module, and can schedule HSI events when a
        individual presents symptoms indicative of the corresponding illness or
        condition.
        """


class _BaseHSIGenericFirstAppt(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id) -> None:
        super().__init__(module, person_id=person_id)
        # No footprint, as this HSI (mostly just) determines which further HSI will be
        # needed for this person. In some cases, small bits of care are provided (e.g. a
        # diagnosis, or the provision of inhaler).
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})

    def _diagnosis_function(
        self, tests, use_dict: bool = False, report_tried: bool = False
    ) -> DiagnosisTestReturnType:
        """
        Passed to modules when determining HSI events to be scheduled based on
        this generic appointment. Intended as the ``diagnosis_function`` argument to
        :py:meth:`GenericFirstAppointmentsMixin.do_at_generic_first_appt` or
        :py:meth:`GenericFirstAppointmentsMixin.do_at_generic_first_appt_emergency`.

        Class-level definition avoids the need to redefine this method each time
        the :py:meth:`apply` method is called.

        :param tests: The name of the test(s) to run via the diagnosis manager.
        :param use_dict_for_single: If ``True``, the return type will be a dictionary
            even if only one test was requested.
        :param report_dxtest_tried: Report if a test was attempted but could not
            be carried out due to for example lack of consumables, etc.
        :returns: Test results as dictionary key/value pairs.
        """
        return self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
            tests,
            hsi_event=self,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    @staticmethod
    def _do_at_generic_first_appt_for_module(module: Module) -> Callable:
        """Retrieves relevant do_at_generic_first_appt* method for a module.

        Must be implemented by concrete classes derived from this base class.
        """
        raise NotImplementedError

    def apply(self, person_id: int, squeeze_factor: float = 0.0) -> None:
        """
        Run the actions required during the health system interaction (HSI).

        TODO: person_id is not needed any more - but would have to go through the
        whole codebase to manually identify instances of this class to change call
        syntax, and leave other HSI_Event-derived classes alone.
        """
        # Create a memoized view of target individuals' properties as a context manager
        # that will automatically synchronize any updates back to the population
        # dataframe on exit
        with self.sim.population.individual_properties(
            self.target, read_only=False
        ) as individual_properties:
            if not individual_properties["is_alive"]:
                return
            # Pre-evaluate symptoms for individual to avoid repeat accesses
            # TODO: Use individual_properties to populate symptoms
            symptoms = self.sim.modules["SymptomManager"].has_what(self.target)
            schedule_hsi_event = self.sim.modules["HealthSystem"].schedule_hsi_event
            for module in self.sim.modules.values():
                if isinstance(module, GenericFirstAppointmentsMixin):
                    self._do_at_generic_first_appt_for_module(module)(
                        person_id=self.target,
                        individual_properties=individual_properties,
                        symptoms=symptoms,
                        schedule_hsi_event=schedule_hsi_event,
                        diagnosis_function=self._diagnosis_function,
                        consumables_checker=self.get_consumables,
                        facility_level=self.ACCEPTED_FACILITY_LEVEL,
                        treatment_id=self.TREATMENT_ID,
                    )


class HSI_GenericNonEmergencyFirstAppt(_BaseHSIGenericFirstAppt):
    """
    This is a health system interaction event that represents the first interaction with
    the health system following the onset of non-emergency symptom(s).

    It is generated by the :py:class:`~HealthSeekingBehaviour` module.

    By default, it occurs at level '0' but it could occur also at other levels.

    It uses the non-emergency generic first appointment methods of the disease modules
    to determine any follow-up events that need to be scheduled.
    """

    def __init__(self, module, person_id, facility_level="0"):
        super().__init__(
            module,
            person_id=person_id,
        )

        assert module is self.sim.modules["HealthSeekingBehaviour"]

        self.TREATMENT_ID = "FirstAttendance_NonEmergency"
        self.ACCEPTED_FACILITY_LEVEL = facility_level

    @staticmethod
    def _do_at_generic_first_appt_for_module(
        module: GenericFirstAppointmentsMixin,
    ) -> Callable:
        return module.do_at_generic_first_appt


class HSI_GenericEmergencyFirstAppt(_BaseHSIGenericFirstAppt):
    """
    This is a health system interaction event that represents the generic appointment
    which is the first interaction with the health system following the onset of
    emergency symptom(s).

    It uses the emergency generic first appointment methods of the disease modules to
    determine any follow-up events that need to be scheduled.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert module.name in [
            "HealthSeekingBehaviour",
            "Labour",
            "PregnancySupervisor",
            "RTI",
        ]

        self.TREATMENT_ID = "FirstAttendance_Emergency"
        self.ACCEPTED_FACILITY_LEVEL = "1b"

    @staticmethod
    def _do_at_generic_first_appt_for_module(
        module: GenericFirstAppointmentsMixin,
    ) -> Callable:
        return module.do_at_generic_first_appt_emergency


class HSI_EmergencyCare_SpuriousSymptom(HSI_Event, IndividualScopeEventMixin):
    """HSI event providing accident & emergency care on spurious emergency symptoms."""

    def __init__(self, module, person_id, accepted_facility_level="1a"):
        super().__init__(module, person_id=person_id)
        assert module is self.sim.modules["HealthSeekingBehaviour"]

        self.TREATMENT_ID = "FirstAttendance_SpuriousEmergencyCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint(
            {"AccidentsandEmerg": 1}
        )
        self.ACCEPTED_FACILITY_LEVEL = (
            accepted_facility_level  # '1a' in default or '1b' as an alternative
        )

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, "is_alive"]:
            return self.make_appt_footprint({})
        else:
            sm = self.sim.modules['SymptomManager']
            sm.change_symptom(person_id, "spurious_emergency_symptom", '-', sm)


def do_at_generic_first_appt_non_emergency(hsi_event, squeeze_factor):
    """The actions are taken during the non-emergency generic HSI, HSI_GenericFirstApptAtFacilityLevel0."""

    # Gather useful shortcuts
    sim = hsi_event.sim
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
    age = df.at[person_id, 'age_years']
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

    # 'Automatic' testing for HIV for everyone attending care with AIDS symptoms:
    #  - suppress the footprint (as it done as part of another appointment)
    #  - do not do referrals if the person is HIV negative (assumed not time for counselling etc).
    if 'Hiv' in sim.modules:
        if 'aids_symptoms' in symptoms:
            schedule_hsi(
                HSI_Hiv_TestAndRefer(
                    person_id=person_id,
                    module=hsi_event.sim.modules['Hiv'],
                    referred_from="hsi_generic_first_appt",
                    suppress_footprint=True,
                    do_not_refer_if_neg=True),
                topen=hsi_event.sim.date,
                tclose=None,
                priority=0)

    if 'injury' in symptoms:
        if 'RTI' in sim.modules:
            sim.modules['RTI'].do_rti_diagnosis_and_treatment(person_id)

    if 'Schisto' in sim.modules:
        sim.modules['Schisto'].do_on_presentation_with_symptoms(person_id=person_id, symptoms=symptoms)

    if "Malaria" in sim.modules:
        malaria_associated_symptoms = {'fever', 'headache', 'stomachache', 'diarrhoea', 'vomiting'}
        if bool(set(symptoms) & malaria_associated_symptoms):
            sim.modules['Malaria'].do_for_suspected_malaria_case(person_id=person_id, hsi_event=hsi_event)

    if age <= 5:
        # ----------------------------------- CHILD < 5 -----------------------------------
        if 'Diarrhoea' in sim.modules:
            if 'diarrhoea' in symptoms:
                sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
                    person_id=person_id, hsi_event=hsi_event)

        if 'Alri' in sim.modules:
            if ('cough' in symptoms) or ('difficult_breathing' in symptoms):
                sim.modules['Alri'].on_presentation(person_id=person_id, hsi_event=hsi_event)

        # Routine assessments
        if 'Stunting' in sim.modules:
            sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)

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

        if 'CervicalCancer' in sim.modules:
            # If the symptoms include vaginal bleeding:
            if 'vaginal_bleeding' in symptoms:
                schedule_hsi(
                    HSI_CervicalCancerPresentationVaginalBleeding(
                        person_id=person_id,
                        module=sim.modules['CervicalCancer']
                    ),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

            # else:
            schedule_hsi(
                HSI_CervicalCancer_Screening(
                    person_id=person_id,
                    module=sim.modules['CervicalCancer']
                ),
                priority=0,
                topen=sim.date,
                tclose=None)
            # if 'chosen_via_screening_for_cin_cervical_cancer' in symptoms:
            #     schedule_hsi(
            #         HSI_CervicalCancer_AceticAcidScreening(
            #             person_id=person_id,
            #             module=sim.modules['CervicalCancer']
            #         ),
            #         priority=0,
            #         topen=sim.date,
            #         tclose=None)
            #
            #
            # if 'chosen_xpert_screening_for_hpv_cervical_cancer' in symptoms:
            #     schedule_hsi(
            #         HSI_CervicalCancer_XpertHPVScreening(
            #             person_id=person_id,
            #             module=sim.modules['CervicalCancer']
            #         ),
            #         priority=0,
            #         topen=sim.date,
            #         tclose=None)

        if 'Depression' in sim.modules:
            sim.modules['Depression'].do_on_presentation_to_care(person_id=person_id,
                                                                 hsi_event=hsi_event)

        if 'CardioMetabolicDisorders' in sim.modules:
            sim.modules['CardioMetabolicDisorders'].determine_if_will_be_investigated(person_id=person_id)

        if 'Copd' in sim.modules:
            if ('breathless_moderate' in symptoms) or ('breathless_severe' in symptoms):
                sim.modules['Copd'].do_when_present_with_breathless(person_id=person_id, hsi_event=hsi_event)


def do_at_generic_first_appt_emergency(hsi_event, squeeze_factor):
    """The actions are taken during the non-emergency generic HSI, HSI_GenericEmergencyFirstApptAtFacilityLevel1."""

    # Gather useful shortcuts
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
        sm = sim.modules["SymptomManager"]
        sm.change_symptom(person_id, "spurious_emergency_symptom", "-", sm)
