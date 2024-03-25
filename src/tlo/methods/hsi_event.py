from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict, Literal, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from tlo import Date, logging
from tlo.events import Event
from tlo.population import Population

if TYPE_CHECKING:
    from tlo import Module, Simulation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_summary = logging.getLogger(f"{__name__}.summary")
logger_summary.setLevel(logging.INFO)

# Declare the level which will be used to represent the merging of levels '1b' and '2'
LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2 = "2"


class FacilityInfo(NamedTuple):
    """Information about a specific health facility."""

    id: int
    name: str
    level: str
    region: str


class HSIEventDetails(NamedTuple):
    """Non-target specific details of a health system interaction event."""

    event_name: str
    module_name: str
    treatment_id: str
    facility_level: Optional[str]
    appt_footprint: Tuple[Tuple[str, int]]
    beddays_footprint: Tuple[Tuple[str, int]]
    equipment: set


class HSIEventQueueItem(NamedTuple):
    """Properties of event added to health system queue.

    The order of the attributes in the tuple is important as the queue sorting is done
    by the order of the items in the tuple, i.e. first by `priority`, then `topen` and
    so on.

    Ensure priority is above topen in order for held-over events with low priority not
    to jump ahead higher priority ones which were opened later.
    """

    priority: int
    topen: Date
    rand_queue_counter: (
        int  # Ensure order of events with same topen & priority is not model-dependent
    )
    queue_counter: (
        int  # Include safety tie-breaker in unlikely event rand_queue_counter is equal
    )
    tclose: Date
    # Define HSI_Event type as string to avoid NameError exception as HSI_Event defined
    # later in module (see https://stackoverflow.com/a/36286947/4798943)
    hsi_event: "HSI_Event"


class HSI_Event:
    """Base HSI event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined in `src/tlo/events.py`, and implement at least an `apply` and
    `did_not_run` method.
    """

    module: Module
    target: int  # Will be overwritten by the mixin on derived classes

    TREATMENT_ID: str
    ACCEPTED_FACILITY_LEVEL: str
    # These values need to be set at runtime as they depend on the modules
    # which have been loaded.
    BEDDAYS_FOOTPRINT: Dict[str, Union[float, int]]

    _received_info_about_bed_days: Dict[str, Union[float, int]] = None
    expected_time_requests: Counter = {}
    facility_info: FacilityInfo = None

    def __init__(self, module, *args, **kwargs):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        """
        self.module = module
        super().__init__(*args, **kwargs)

        # Information that will later be received about this HSI
        self._received_info_about_bed_days = None
        self.expected_time_requests = {}
        self.facility_info = None
        self.ESSENTIAL_EQUIPMENT = None
        self.EQUIPMENT = set()

        self.TREATMENT_ID = ""
        self.ACCEPTED_FACILITY_LEVEL = None
        # Set "dynamic" default value
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({})

    @property
    def bed_days_allocated_to_this_event(self):
        if self._received_info_about_bed_days is None:
            # default to the footprint if no information about bed-days is received
            return self.BEDDAYS_FOOTPRINT

        return self._received_info_about_bed_days

    @property
    def target_is_alive(self) -> bool:
        """Return True if the target of this HSI event is alive,
        otherwise False.
        """
        return self.sim.population.props.at[self.target, "is_alive"]

    @property
    def sim(self) -> Simulation:
        return self.module.sim

    @property
    def healthcare_system(self) -> Module:
        """The healthcare module being used by the Simulation."""
        return self.sim.modules["HealthSystem"]

    def _adjust_facility_level_to_merge_1b_and_2(self) -> str:
        """Adjust the facility level of the HSI_Event,
        so that HSI_Events scheduled at level '1b' and '2' are both directed to level '2'
        """
        self.ACCEPTED_FACILITY_LEVEL = (
            self.ACCEPTED_FACILITY_LEVEL
            if self.ACCEPTED_FACILITY_LEVEL not in ("1b", "2")
            else LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2
        )

    def apply(self, squeeze_factor=0.0, *args, **kwargs):
        """Apply this event to the population.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def did_not_run(self, *args, **kwargs) -> Literal[True]:
        """Called when this event is due but it is not run. Return False to prevent the event being rescheduled, or True
        to allow the rescheduling. This is called each time that the event is tried to be run but it cannot be.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: did not run.")
        return True

    def never_ran(self) -> None:
        """Called when this event is was entered to the HSI Event Queue, but was never run."""
        logger.debug(key="message", data=f"{self.__class__.__name__}: was never run.")

    def post_apply_hook(self) -> None:
        """Impose the bed-days footprint (if target of the HSI is a person_id)"""
        if isinstance(self.target, int):
            self.healthcare_system.bed_days.impose_beddays_footprint(
                person_id=self.target, footprint=self.bed_days_allocated_to_this_event
            )

    def run(self, squeeze_factor):
        """Make the event happen."""
        updated_appt_footprint = self.apply(self.target, squeeze_factor)
        self.post_apply_hook()
        return updated_appt_footprint

    def get_consumables(
        self,
        item_codes: Union[None, np.integer, int, list, set, dict] = None,
        optional_item_codes: Union[None, np.integer, int, list, set, dict] = None,
        to_log: Optional[bool] = True,
        return_individual_results: Optional[bool] = False,
    ) -> Union[bool, dict]:
        """Function to allow for getting and checking of entire set of consumables. All requests for consumables should
        use this function.
        :param item_codes: The item code(s) (and quantities) of the consumables that are requested and which determine
        the summary result for availability/non-availability. This can be an `int` (the item_code needed [assume
        quantity=1]), a `list` or `set` (the collection  of item_codes [for each assuming quantity=1]), or a `dict`
        (with key:value pairs `<item_code>:<quantity>`).
        :param optional_item_codes: The item code(s) (and quantities) of the consumables that are requested and which do
         not determine the summary result for availability/non-availability. (Same format as `item_codes`). This is
         useful when a large set of items may be used, but the viability of a subsequent operation depends only on a
         subset.
        :param return_individual_results: If True returns a `dict` giving the availability of each item_code requested
        (otherwise gives a `bool` indicating if all the item_codes requested are available).
        :param to_log: If True, logs the request.
        :returns A `bool` indicating whether every item is available, or a `dict` indicating the availability of each
         item.
        Note that disease module can use the `get_item_codes_from_package_name` and `get_item_code_from_item_name`
         methods in the `HealthSystem` module to find item_codes.
        """
        _item_codes = self._return_item_codes_in_dict(item_codes)
        _optional_item_codes = self._return_item_codes_in_dict(optional_item_codes)

        # Determine if the request should be logged (over-ride argument provided if HealthSystem is disabled).
        _to_log = to_log if not self.healthcare_system.disable else False

        # Checking the availability and logging:
        rtn = self.healthcare_system.consumables._request_consumables(
            item_codes={**_item_codes, **_optional_item_codes},
            to_log=_to_log,
            facility_info=self.facility_info,
            treatment_id=self.TREATMENT_ID,
        )

        # Return result in expected format:
        if not return_individual_results:
            # Determine if all results for all the `item_codes` are True (discarding results from optional_item_codes).
            return all(v for k, v in rtn.items() if k in _item_codes)
        else:
            return rtn

    def make_beddays_footprint(self, dict_of_beddays) -> Dict[str, Union[float, int]]:
        """Helper function to make a correctly-formed 'bed-days footprint'"""

        # get blank footprint
        footprint = self.healthcare_system.bed_days.get_blank_beddays_footprint()

        # do checks on the dict_of_beddays provided.
        assert isinstance(dict_of_beddays, dict)
        assert all((k in footprint.keys()) for k in dict_of_beddays.keys())
        assert all(isinstance(v, (float, int)) for v in dict_of_beddays.values())

        # make footprint (defaulting to zero where a type of bed-days is not specified)
        for k, v in dict_of_beddays.items():
            footprint[k] = v

        return footprint

    def is_all_beddays_allocated(self) -> bool:
        """Check if the entire footprint requested is allocated"""
        return all(
            self.bed_days_allocated_to_this_event[k] == self.BEDDAYS_FOOTPRINT[k]
            for k in self.BEDDAYS_FOOTPRINT
        )

    def make_appt_footprint(self, dict_of_appts) -> Counter:
        """Helper function to make appointment footprint in format expected downstream.

        Should be passed a dictionary keyed by appointment type codes with non-negative
        values.
        """
        if self.healthcare_system.appt_footprint_is_valid(dict_of_appts):
            return Counter(dict_of_appts)

        raise ValueError(
            "Argument to make_appt_footprint should be a dictionary keyed by "
            "appointment type code strings in Appt_Types_Table with non-negative "
            "values"
        )

    def get_equip_item_code_from_item_name(self, equip_item_name: str) -> int:
        """Helper function to provide the equip_item_code (an int) when provided with the equip_item_name of the item"""
        lookup_df = self.sim.modules['HealthSystem'].parameters['Equipment']
        return int(pd.unique(lookup_df.loc[lookup_df["Equip_Item"] == equip_item_name, "Equip_Code"])[0])

    def get_equip_item_codes_from_pkg_name(self, equip_pkg_name: str) -> Set[int]:
        """Helper function to provide the equip_item_codes (a set of ints) when provided with the equip_pkg_name of the
        equipment package"""
        lookup_df = self.sim.modules['HealthSystem'].parameters['Equipment']
        return set(lookup_df.loc[lookup_df["Equip_Pkg"] == equip_pkg_name, "Equip_Code"])

    def ignore_unknown_equip_names(self, set_of_names: Set[str], type_in_set: str) -> Set[str]:
        """Helper function to check if the equipment item or pkg names (depending on type_in_set: 'item' or 'pkg') from
        the provided set are in the RF_Equipment. If they are not, they are added to a set to be warned about at the end
        of the simulation.

        Only known (item or pkg) names are returned."""
        if set_of_names in [set(), None, {''}]:
            return set()

        def add_unknown_names_to_dict(unknown_names_to_add: Set[str], dict_to_be_added_to: Dict) -> Dict:
            if self.__class__.__name__ not in dict_to_be_added_to.keys():
                dict_to_be_added_to.update(
                    {self.__class__.__name__: unknown_names_to_add}
                )
            else:
                dict_to_be_added_to[self.__class__.__name__].update(
                    unknown_names_to_add
                )
            return dict_to_be_added_to

        lookup_df = self.sim.modules['HealthSystem'].parameters['Equipment']
        if type_in_set == "item":
            unknown_names = set_of_names.difference(set(lookup_df["Equip_Item"]))
            if unknown_names:
                self.sim.modules['HealthSystem']._equip_items_missing_in_RF = \
                    add_unknown_names_to_dict(
                        unknown_names, self.sim.modules['HealthSystem']._equip_items_missing_in_RF
                    )

        elif type_in_set == "pkg":
            unknown_names = set_of_names.difference(set(lookup_df["Equip_Pkg"]))
            if unknown_names:
                self.sim.modules['HealthSystem']._equip_pkgs_missing_in_RF = \
                    add_unknown_names_to_dict(
                        unknown_names, self.sim.modules['HealthSystem']._equip_pkgs_missing_in_RF
                    )
        # TODO: What happens if all equip in set_of_names has unknown name?
        return set_of_names.difference(unknown_names)

    def set_equipment_essential_to_run_event(self, set_of_equip: Set[str]) -> None:
        """Helper function to set essential equipment.

        Should be passed a set of equipment items names (strings) or an empty set.
        """
        # Set EQUIPMENT if the given set_of_equip in correct format, ie a set of strings or an empty set
        if not isinstance(set_of_equip, set) or any(not isinstance(item, str) for item in set_of_equip):
            raise ValueError(
                "Argument to set_equipment_essential_to_run_event should be an empty set or a set of strings of "
                "equipment item names from ResourceFile_Equipment.csv."
            )

        set_of_equip = self.ignore_unknown_equip_names(set_of_equip, "item")
        if set_of_equip:
            equip_codes = set(self.get_equip_item_code_from_item_name(item_name) for item_name in set_of_equip)
            self.ESSENTIAL_EQUIPMENT = equip_codes
        else:
            self.ESSENTIAL_EQUIPMENT = set()

    # todo add function to set essential equipment

    def add_equipment(self, set_of_equip: Set[str]) -> None:
        """Helper function to update equipment.

        Should be passed a set of equipment item names (strings).
        """
        # Update EQUIPMENT if the given set_of_equip in correct format, ie a non-empty set of strings
        if (
            (not isinstance(set_of_equip, set))
            or
            any(not isinstance(item, str) for item in set_of_equip)
            or
            (set_of_equip in [set(), None, {''}])
        ):
            raise ValueError(
                "Argument to add_equipment should be a non-empty set of strings of "
                "equipment item names from ResourceFile_Equipment.csv."
            )
        # from the set of equip item names create a set of equip item codes, ignore unknown equip names
        # (ie not included in RF_Equipment)
        set_of_equip = self.ignore_unknown_equip_names(set_of_equip, "item")
        if set_of_equip:
            equip_codes = set(self.get_equip_item_code_from_item_name(item_name) for item_name in set_of_equip)
            self.EQUIPMENT.update(equip_codes)

    def add_equipment_from_pkg(self, set_of_pkgs: Set[str]) -> None:
        """Helper function to update equipment with equipment from pkg(s).

        Should be passed a set of equipment pkgs names (strings).
        """
        # Update EQUIPMENT if the given set_of_pkgs in correct format, ie a non-empty set of strings
        if not isinstance(set_of_pkgs, set) or any(not isinstance(item, str) for item in set_of_pkgs) or \
           (set_of_pkgs in [set(), None, {''}]):
            raise ValueError(
                "Argument to add_equipment_from_pkg should be a non-empty set of strings of "
                "equipment pkg names from ResourceFile_Equipment.csv."
            )
        # update EQUIPMENT with eqip item codes from equip pkgs with provided names, ignore unknown equip names
        # (ie not included in RF_Equipment)
        set_of_pkgs = self.ignore_unknown_equip_names(set_of_pkgs, "pkg")
        if set_of_pkgs:
            for pkg_name in set_of_pkgs:
                self.EQUIPMENT.update(self.get_equip_item_codes_from_pkg_name(pkg_name))

    def get_essential_equip_availability(self, set_of_pkgs: Set[str]) -> bool:
        # TODO: Or, should it be called set_essential_equip_and_get_availability to be more transparent about what the
        #  fnc does?
        self.set_equipment_essential_to_run_event(set_of_pkgs)
        return self.sim.modules['HealthSystem'].get_essential_equip_availability(self.ESSENTIAL_EQUIPMENT)

    def initialise(self) -> None:
        """Initialise the HSI:
        * Set the facility_info
        * Compute appt-footprint time requirements
        """
        health_system = self.healthcare_system

        # Over-write ACCEPTED_FACILITY_LEVEL to to redirect all '1b' appointments to '2'
        self._adjust_facility_level_to_merge_1b_and_2()

        if not isinstance(self.target, Population):
            self.facility_info = health_system.get_facility_info(self)

            # If there are bed-days specified, add (if needed) the in-patient admission and in-patient day Appointment
            # Types.
            # (HSI that require a bed for one or more days always need such appointments, but this may have been
            # missed in the declaration of the `EXPECTED_APPT_FOOTPRINT` in the HSI.)
            # NB. The in-patient day Appointment time is automatically applied on subsequent days.
            if sum(self.BEDDAYS_FOOTPRINT.values()):
                self.EXPECTED_APPT_FOOTPRINT = (
                    health_system.bed_days.add_first_day_inpatient_appts_to_footprint(
                        self.EXPECTED_APPT_FOOTPRINT
                    )
                )

            # Write the time requirements for staff of the appointments to the HSI:
            self.expected_time_requests = (
                health_system.get_appt_footprint_as_time_request(
                    facility_info=self.facility_info,
                    appt_footprint=self.EXPECTED_APPT_FOOTPRINT,
                )
            )

        # Do checks
        self._check_if_appt_footprint_can_run()

        # Set essential equip to empty set if not exists and warn about missing settings
        if self.ESSENTIAL_EQUIPMENT is None:
            self.set_equipment_essential_to_run_event({''})
            self.sim.modules['HealthSystem']._hsi_event_names_missing_ess_equip.update({self.__class__.__name__})

    def _check_if_appt_footprint_can_run(self) -> bool:
        """Check that event (if individual level) is able to run with this configuration of officers (i.e. check that
        this does not demand officers that are _never_ available), and issue warning if not.
        """
        if not isinstance(self.target, Population):
            if self.healthcare_system._officers_with_availability.issuperset(
                self.expected_time_requests.keys()
            ):
                return True
            else:
                logger.warning(
                    key="message",
                    data=(
                        f"The expected footprint of {self.TREATMENT_ID} is not possible with the configuration of "
                        f"officers."
                    ),
                )
                return False

    @staticmethod
    def _return_item_codes_in_dict(
        item_codes: Union[None, np.integer, int, list, set, dict]
    ) -> dict:
        """Convert an argument for 'item_codes` (provided as int, list, set or dict) into the format
        dict(<item_code>:quantity)."""

        if item_codes is None:
            return {}

        if isinstance(item_codes, (int, np.integer)):
            return {int(item_codes): 1}

        elif isinstance(item_codes, list):
            if not all([isinstance(i, (int, np.integer)) for i in item_codes]):
                raise ValueError("item_codes must be integers")
            return {int(i): 1 for i in item_codes}

        elif isinstance(item_codes, dict):
            if not all(
                [
                    (
                        isinstance(code, (int, np.integer))
                        and isinstance(quantity, (float, np.floating, int, np.integer))
                    )
                    for code, quantity in item_codes.items()
                ]
            ):
                raise ValueError(
                    "item_codes must be integers and quantities must be integers or floats."
                )
            return {int(i): float(q) for i, q in item_codes.items()}

        else:
            raise ValueError("The item_codes are given in an unrecognised format")

    def as_namedtuple(
        self, actual_appt_footprint: Optional[dict] = None
    ) -> HSIEventDetails:
        appt_footprint = (
            getattr(self, "EXPECTED_APPT_FOOTPRINT", {})
            if actual_appt_footprint is None
            else actual_appt_footprint
        )
        return HSIEventDetails(
            event_name=type(self).__name__,
            module_name=type(self.module).__name__,
            treatment_id=self.TREATMENT_ID,
            facility_level=getattr(self, "ACCEPTED_FACILITY_LEVEL", None),
            appt_footprint=tuple(sorted(appt_footprint.items())),
            beddays_footprint=tuple(
                sorted((k, v) for k, v in self.BEDDAYS_FOOTPRINT.items() if v > 0)
            ),
            equipment=(tuple(sorted(self.EQUIPMENT))),
        )


class HSIEventWrapper(Event):
    """This is wrapper that contains an HSI event.

    It is used:
     1) When the healthsystem is in mode 'disabled=True' such that HSI events sent to the health system scheduler are
     passed to the main simulation scheduler for running on the date of `topen`. (Note, it is run with
     squeeze_factor=0.0.)
     2) When the healthsystem is in mode `disable_and_reject_all=True` such that HSI are not run but the `never_ran`
     method is run on the date of `tclose`.
     3) When an HSI has been submitted to `schedule_hsi_event` but the service is not available.
    """

    def __init__(self, hsi_event, run_hsi=True, *args, **kwargs):
        super().__init__(hsi_event.module, *args, **kwargs)
        self.hsi_event = hsi_event
        self.target = hsi_event.target
        self.run_hsi = run_hsi  # True to call the HSI's `run` method; False to call the HSI's `never_ran` method

    def run(self):
        """Do the appropriate action on the HSI event"""

        # Check that the person is still alive (this check normally happens in the HealthSystemScheduler and silently
        # do not run the HSI event)

        if isinstance(self.hsi_event.target, Population) or (
            self.hsi_event.module.sim.population.props.at[
                self.hsi_event.target, "is_alive"
            ]
        ):

            if self.run_hsi:
                # Run the event (with 0 squeeze_factor) and ignore the output
                _ = self.hsi_event.run(squeeze_factor=0.0)
            else:
                self.hsi_event.module.sim.modules[
                    "HealthSystem"
                ].call_and_record_never_ran_hsi_event(
                    hsi_event=self.hsi_event, priority=-1
                )
