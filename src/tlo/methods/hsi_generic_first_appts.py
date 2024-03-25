"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.

This file contains the HSI events that represent the first contact with the Health System, which are triggered by
the onset of symptoms. Non-emergency symptoms lead to `HSI_GenericFirstApptAtFacilityLevel0` and emergency symptoms
lead to `HSI_GenericEmergencyFirstApptAtFacilityLevel1`.
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, OrderedDict

from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.hsi_event import HSI_Event

if TYPE_CHECKING:
    from tlo import Module
    from tlo.methods.dxmanager import DiagnosisTestReturnType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FIRST_APPT_NON_EMERGENCY_MODULE_ORDER = [
    "Measles",
    "Hiv",
    "RTI",
    "Schisto",
    "Malaria",
    "Diarrhoea",
    "Alri",
    "Stunting",
    "OesophagealCancer",
    "BladderCancer",
    "ProstateCancer",
    "OtherAdultCancer",
    "BreastCancer",
    "Depression",
    "CardioMetabolicDisorders",
    "Copd",
]

FIRST_APPT_EMERGENCY_MODULE_ORDER = [
    "PregnancySupervisor",
    "Labour",
    "Depression",
    "Malaria",
    "CardioMetabolicDisorders",
    "Epilepsy",
    "RTI",
    "Alri",
    "Copd",
]

def sort_preserving_order(to_sort: Iterable[Any], relative_to: Iterable[Any]) -> List:
    """
    Sort the items in a given list using the relative ordering of another list.

    Items in the list that do not appear in the relative ordering are assumed
    to be of the lowest priority (moved to the back of the sorted list).

    :param to_sort: List of values to sort. Sorting returns a copy.
    :param relative_to: List of values that appear in to_sort,
    defining relative order.
    """

    def sort_key(item):
        try:
            return relative_to.index(item)
        except ValueError:
            return len(relative_to)

    return sorted(to_sort, key=sort_key)

class HSI_BaseGenericFirstAppt(HSI_Event, IndividualScopeEventMixin):
    """
    """
    MODULE_METHOD_ON_APPLY: Literal[
        "do_at_generic_first_appt", "do_at_generic_first_appt_emergency"
    ]

    @property
    def MODULE_ORDER_ON_APPLY(self) -> List[str]:
        if "emergency" in self.MODULE_METHOD_ON_APPLY:
            return FIRST_APPT_EMERGENCY_MODULE_ORDER
        else:
            return FIRST_APPT_NON_EMERGENCY_MODULE_ORDER

    def __init__(self, module, person_id) -> None:
        super().__init__(module, person_id=person_id)
        # No footprint, as this HSI (mostly just) determines which
        # further HSI will be needed for this person. In some cases,
        # small bits of care are provided (e.g. a diagnosis, or the
        # provision of inhaler).
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint(
            {}
        )

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
        return self.healthcare_system.dx_manager.run_dx_test(
            tests,
            hsi_event=self,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    def _do_on_generic_first_appt(self, squeeze_factor: float = 0.) -> None:
        """
        """
        # Make top-level reads of information, to avoid repeat accesses.
        modules: OrderedDict[str, "Module"] = self.sim.modules
        symptoms = modules["SymptomManager"].has_what(self.target)

        # Dynamically create immutable container with the target's details stored.
        # This will avoid repeat DataFrame reads when we call the module-level functions.
        df = self.sim.population.props
        patient_details = namedtuple("PatientDetails", df.columns)(*df.loc[self.target])

        proposed_df_updates = {}

        module_order = self._module_order(modules.keys())
        for name in module_order:
            module = modules[name]
            df_updates = getattr(module, self.MODULE_METHOD_ON_APPLY)(
                patient_id=self.target,
                patient_details=patient_details,
                symptoms=symptoms,
                diagnosis_function=self._diagnosis_function,
                consumables_checker=self.get_consumables,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                treatment_id=self.TREATMENT_ID,
            )
            # Record any requested DataFrame updates, but do not implement yet
            # NOTE: |= syntax is only available in Python >=3.9
            proposed_df_updates = {**proposed_df_updates, **df_updates}

        # Perform any DataFrame updates that were requested, all in one go.
        df.loc[self.target, proposed_df_updates.keys()] = proposed_df_updates.values()

    def _module_order(self, module_list: Iterable[str]) -> List[str]:
        """"""
        return sort_preserving_order(module_list, self.MODULE_ORDER_ON_APPLY)

    def apply(self, person_id, squeeze_factor=0.) -> None:
        """
        Run the actions required during the HSI.

        TODO: person_id is not needed any more - but would have to go through the
        whole codebase to manually identify instances of this class to change call
        syntax, and leave other HSI_Event-derived classes alone.
        """
        if self.target_is_alive:
            self._do_on_generic_first_appt(squeeze_factor=squeeze_factor)


class HSI_GenericNonEmergencyFirstAppt(HSI_BaseGenericFirstAppt):
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
    MODULE_METHOD_ON_APPLY = "do_at_generic_first_appt"

    def __init__(self, module, person_id, facility_level='0'):
        super().__init__(module, person_id=person_id, )

        assert module is self.sim.modules['HealthSeekingBehaviour']

        self.TREATMENT_ID = 'FirstAttendance_NonEmergency'
        self.ACCEPTED_FACILITY_LEVEL = facility_level


class HSI_GenericEmergencyFirstAppt(HSI_BaseGenericFirstAppt):
    """
    This is a Health System Interaction Event that represents
    the generic appointment which is the first interaction with
    the health system following the onset of emergency symptom(s).

    It uses the emergency generic first appointment methods of
    the disease modules to determine any follow-up events that need
    to be scheduled.
    """
    MODULE_METHOD_ON_APPLY = "do_at_generic_first_appt_emergency"

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

class HSI_EmergencyCare_SpuriousSymptom(HSI_Event, IndividualScopeEventMixin):
    """This is an HSI event that provides Accident & Emergency Care for a person that has spurious emergency symptom."""

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
