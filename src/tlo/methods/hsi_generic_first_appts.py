"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.

This file contains the HSI events that represent the first contact with the Health System, which are triggered by
the onset of symptoms. Non-emergency symptoms lead to `HSI_GenericFirstApptAtFacilityLevel0` and emergency symptoms
lead to `HSI_GenericEmergencyFirstApptAtFacilityLevel1`.
"""
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Iterable, List, OrderedDict

import pandas as pd

from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.care_of_women_during_pregnancy import (
    HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement,
    HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy,
)
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.labour import HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms

if TYPE_CHECKING:
    from tlo import Module

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

    def _do_at_generic_first_appt_non_emergency(self, squeeze_factor: float):
        """
        The actions are taken during the non-emergency generic HSI,
        HSI_GenericFirstApptAtFacilityLevel0.
        """

        # Make top-level reads of information, to avoid repeat accesses.
        modules: OrderedDict[str, "Module"] = self.sim.modules
        schedule_hsi = self.healthcare_system.schedule_hsi_event
        symptoms = modules["SymptomManager"].has_what(self.target)
        facility_level = self.ACCEPTED_FACILITY_LEVEL
        treatment_id = self.TREATMENT_ID

        # Create the diagnosis test runner function
        def diagnosis_fn(tests, use_dict: bool = False, report_tried: bool = False):
            return self.healthcare_system.dx_manager.run_dx_test(
                tests,
                hsi_event=self,
                use_dict_for_single=use_dict,
                report_dxtest_tried=report_tried,
            )

        # Create the consumables checker function
        def consumables_fn(item_codes, opt_item_codes=None):
            return self.get_consumables(
                item_codes=item_codes, optional_item_codes=opt_item_codes
            )

        # Dynamically create immutable container with the target's details stored.
        # This will avoid repeat DataFrame reads when we call the module-level functions.
        df = self.sim.population.props
        patient_details = namedtuple("PatientDetails", df.columns)(*df.loc[self.target])

        proposed_df_updates = {}

        module_order = sort_preserving_order(modules.keys(), FIRST_APPT_NON_EMERGENCY_MODULE_ORDER)
        for name in module_order:
            module = modules[name]
            event_info, df_updates = module.do_at_generic_first_appt(
                patient_id=self.target,
                patient_details=patient_details,
                symptoms=symptoms,
                diagnosis_fn=diagnosis_fn,
                consumables_checker=consumables_fn,
                facility_level=facility_level,
                treatment_id=treatment_id,
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
        df.loc[self.target, proposed_df_updates.keys()] = proposed_df_updates.values()

    def apply(self, person_id, squeeze_factor):
        """Run the actions required during the HSI."""
        # person_id is not needed anymore... but would have to go through the whole codebase
        # to manually identify instances of this class to change call syntax... and leave
        # other HSI_Event-derived classes alone.
        if self.target_is_alive:
            self._do_at_generic_first_appt_non_emergency(squeeze_factor=squeeze_factor)


class HSI_GenericEmergencyFirstAppt(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of emergency symptom(s)."""

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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint(
            {}
        )  # <-- No footprint, as this HSI (mostly just)
        #                                                                  determines which further HSI will be needed
        #                                                                  for this person. In some cases, small bits
        #                                                                  of care are provided (e.g. a diagnosis, or
        #                                                                  the provision of inhaler.).

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            return

        do_at_generic_first_appt_emergency(
            hsi_event=self, squeeze_factor=squeeze_factor
        )


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


def do_at_generic_first_appt_emergency(hsi_event: HSI_Event, squeeze_factor):
    """
    The actions are taken during the non-emergency generic HSI,
    HSI_GenericEmergencyFirstApptAtFacilityLevel1.
    """
    # OLD shortcuts, to prune later
    sim = hsi_event.sim
    rng = hsi_event.module.rng
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    symptoms = hsi_event.sim.modules["SymptomManager"].has_what(person_id=person_id)
    schedule_hsi = hsi_event.sim.modules["HealthSystem"].schedule_hsi_event
    treatment_id = hsi_event.TREATMENT_ID

    # Make top-level reads of information, to avoid repeat accesses.
    person_id = hsi_event.target
    modules: OrderedDict[str, "Module"] = hsi_event.sim.modules
    schedule_hsi = hsi_event.healthcare_system.schedule_hsi_event
    symptoms = hsi_event.sim.modules["SymptomManager"].has_what(person_id)
    facility_level = hsi_event.ACCEPTED_FACILITY_LEVEL
    treatment_id = hsi_event.TREATMENT_ID

    # Create the diagnosis test runner function
    def diagnosis_fn(tests, use_dict: bool = False, report_tried: bool = False):
        return hsi_event.healthcare_system.dx_manager.run_dx_test(
            tests,
            hsi_event=hsi_event,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    # Create the consumables checker function
    def consumables_fn(item_codes, opt_item_codes=None):
        return hsi_event.get_consumables(
            item_codes=item_codes, optional_item_codes=opt_item_codes
        )

    # Dynamically create immutable container with the target's details stored.
    # This will avoid repeat DataFrame reads when we call the module-level functions.
    df = hsi_event.sim.population.props
    patient_details = namedtuple("PatientDetails", df.columns)(*df.loc[person_id])

    proposed_df_updates = {}

    module_order = sort_preserving_order(modules.keys(), FIRST_APPT_EMERGENCY_MODULE_ORDER)
    for name in module_order:
        module = modules[name]
        event_info, df_updates = module.do_at_generic_first_appt_emergency(
            patient_id=person_id,
            patient_details=patient_details,
            symptoms=symptoms,
            diagnosis_fn=diagnosis_fn,
            consumables_checker=consumables_fn,
            facility_level=facility_level,
            treatment_id=treatment_id,
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

    # OLD FN CONTINUES

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
