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

from tlo import Date, Module, logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.hsi_event import HSI_Event

if TYPE_CHECKING:
    from typing import Optional, TypeAlias

    from tlo.methods.dxmanager import DiagnosisTestReturnType
    from tlo.population import IndividualProperties

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


class GenericFirstApptModule(Module):
    """Base class for modules with actions to perform on generic first appointments."""

    def do_at_generic_first_appt(
        self,
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
        Actions to be take during a non-emergency generic HSI.

        Derived classes should overwrite this method so that they are compatible with
        the :py:class:`HealthSystem` module, and can schedule HSI events when a
        individual presents symptoms indicative of the corresponding illness or
        condition.

        When overwriting, arguments that are not required can be left out of the
        definition. If done so, the method **must** take a ``**kwargs`` input to avoid
        errors when looping over all disease modules and running their generic HSI
        methods.

        HSI events should be scheduled by the :py:class:`Module` subclass implementing
        this method using the ``schedule_hsi_event`` argument..

        Implementations of this method should **not** make any update to the population
        dataframe directly - if the target individuals properties need to be updated
        this should be performed by updating the ``individual_properties`` argument.

        :param person_id: Row index (ID) of the individual target of the HSI event in
            the population dataframe.
        :param individual_properties: Properties of individual target as provided in the
            population dataframe. Updates to individual properties may be written to
            this object.
        :param symptoms: List of symptoms the patient is experiencing.
        :param schedule_hsi_event: A function that can schedule sussequent HSI events.
        :param diagnosis_function: A function that can run diagnosis tests based on the
            patient's symptoms.
        :param consumables_checker: A function that can query the HealthSystem to check
            for available consumables.
        :param facility_level: The level of the facility that the patient presented at.
        :param treatment_id: The treatment id of the HSI event triggering the generic
            appointment.
        """

    def do_at_generic_first_appt_emergency(
        self,
        person_id: int,
        individual_properties: IndividualProperties,
        symptoms: str,
        schedule_hsi_event: HSIEventScheduler,
        diagnosis_function: DiagnosisFunction,
        consumables_checker: ConsumablesChecker,
        facility_level: str,
        treatment_id: str,
    ) -> None:
        """
        Actions to be take during an emergency generic HSI.

        Call signature is identical to the :py:meth:`~Module.do_at_generic_first_appt`
        method.

        Derived classes should overwrite this method so that they are compatible with
        the :py:class`HealthSystem` module, and can schedule HSI events when a
        individual presents symptoms indicative of the corresponding illness or
        condition.
        """


class _BaseHSIGenericFirstAppt(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id) -> None:
        super().__init__(module, person_id=person_id)
        # No footprint, as this HSI (mostly just) determines which
        # further HSI will be needed for this person. In some cases,
        # small bits of care are provided (e.g. a diagnosis, or the
        # provision of inhaler).
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})

    def _diagnosis_function(
        self, tests, use_dict: bool = False, report_tried: bool = False
    ) -> DiagnosisTestReturnType:
        """
        Passed to modules when determining HSI_Events to be scheduled based on
        this generic appointment. Intended as the diagnosis_function argument to the
        Module.do_at_generic_{non_}_emergency.

        Class-level definition avoids the need to redefine this method each time
        the .apply() method is called.

        :param tests: The name of the test(s) to run via the diagnosis manager.
        :param use_dict_for_single: If True, the return type will be a dictionary
        even if only one test was requested.
        :param report_dxtest_tried: Report if a test was attempted but could not
        be carried out due to EG lack of consumables, etc.
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
        """Retrieves relevant do_at_generic_first_appt method for a module.

        Must be implemented by concrete classes derived from this base class.
        """
        raise NotImplementedError

    def apply(self, person_id: int, squeeze_factor: float = 0.0) -> None:
        """
        Run the actions required during the HSI.

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
                if isinstance(module, GenericFirstApptModule):
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
    This is a Health System Interaction Event that represents the
    first interaction with the health system following the onset
    of non-emergency symptom(s).

    It is generated by the HealthSeekingBehaviour module.

    By default, it occurs at level '0' but it could occur also at
    other levels.

    It uses the non-emergency generic first appointment methods of
    the disease modules to determine any follow-up events that need
    to be scheduled.
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
        module: GenericFirstApptModule,
    ) -> Callable:
        return module.do_at_generic_first_appt


class HSI_GenericEmergencyFirstAppt(_BaseHSIGenericFirstAppt):
    """
    This is a Health System Interaction Event that represents
    the generic appointment which is the first interaction with
    the health system following the onset of emergency symptom(s).

    It uses the emergency generic first appointment methods of
    the disease modules to determine any follow-up events that need
    to be scheduled.
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
        module: GenericFirstApptModule,
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
            sm = self.sim.modules["SymptomManager"]
            sm.change_symptom(person_id, "spurious_emergency_symptom", "-", sm)
