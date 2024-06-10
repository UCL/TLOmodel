from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict, Iterable, Literal, NamedTuple, Optional, Set, Tuple, Union

import numpy as np

from tlo import Date, logging
from tlo.events import Event
from tlo.population import Population

if TYPE_CHECKING:
    from tlo import Module, Simulation
    from tlo.methods.bed_days import BedDaysFootprint
    from tlo.methods.healthsystem import HealthSystem

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
    equipment: Tuple[str]


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
    target: int # Will be overwritten by the mixin on derived classes

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

        # Information that will later be received/computed about this HSI
        self._received_info_about_bed_days = None
        self.expected_time_requests = {}
        self.facility_info = None
        self._is_all_declared_equipment_available = None

        self.TREATMENT_ID = ""
        self.ACCEPTED_FACILITY_LEVEL = None
        # Set "dynamic" default value
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({})
        self._EQUIPMENT: Set[int] = set()  # The set of equipment that is used in the HSI. If any items in this set are
        #                                     not available at the point when the HSI will be run, then the HSI is not
        #                                     run, and the `never_ran` method is called instead. This is a declaration
        #                                     of resource needs, but is private because users are expected to use
        #                                     `add_equipment` to declare equipment needs.

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
    def healthcare_system(self) -> HealthSystem:
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
        """Do any required processing after apply() completes."""

    def _run_after_hsi_event(self) -> None:
        """
        Do things following the event's `apply` and `post_apply_hook` functions running.
         * Impose the bed-days footprint (if target of the HSI is a person_id)
         * Record the equipment that has been added before and during the course of the HSI Event.
        """
        if isinstance(self.target, int):
            are_new_inpatient = (
                self.healthcare_system.bed_days.impose_beddays_footprint(
                    footprint=self.bed_days_allocated_to_this_event,
                    facility=self.healthcare_system.get_facility_id_for_beds(
                        self.target
                    ),
                    first_day=self.sim.date,
                    patient_id=self.target,
                )
            )
            if are_new_inpatient:
                self.sim.population.props.at[self.target, "hs_is_inpatient"] = True

        if self.facility_info is not None:
            # If there is a facility_info (e.g., healthsystem not running in disabled mode), then record equipment used
            self.healthcare_system.equipment.record_use_of_equipment(
                item_codes=self._EQUIPMENT,
                facility_id=self.facility_info.id
            )

    def run(self, squeeze_factor):
        """Make the event happen."""
        updated_appt_footprint = self.apply(self.target, squeeze_factor)
        self.post_apply_hook()
        self._run_after_hsi_event()
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

    def make_beddays_footprint(
        self, dict_of_beddays: Dict[str, int | float] = {}
    ) -> BedDaysFootprint:
        """
        Helper function to make a correctly-formed 'bed-days footprint',
        may be overwritten by subclasses.
        """
        return self.healthcare_system.bed_days.get_blank_beddays_footprint(
            **dict_of_beddays
        )

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

    def add_equipment(self, item_codes: Union[int, str, Iterable[int], Iterable[str]]) -> None:
        """Declare that piece(s) of equipment are used in this HSI_Event. Equipment items can be identified by their
        item_codes (int) or descriptors (str); a singular item or an iterable of items (either codes or descriptors but
        not a mix of both) can be defined at once. Checks are done on the validity of the item_codes/item
        descriptions and a warning issued if any are not recognised."""
        self._EQUIPMENT.update(self.healthcare_system.equipment.parse_items(item_codes))

    @property
    def is_all_declared_equipment_available(self) -> bool:
        """Returns ``True`` if all the (currently) declared items of equipment are available. This is called by the
        ``HealthSystem`` module before the HSI is run and so is looking only at those items that are declared when this
        instance was created. The evaluation of whether equipment is available is only done *once* for this instance of
        the event: i.e., if the equipment is not available for the instance of this ``HSI_Event``, then it will remain not
        available if the same event is re-scheduled/re-entered into the HealthSystem queue. This is representing that
        if the facility that a particular person attends for the ``HSI_Event`` does not have the equipment available, then
        it will also not be available on another day."""

        if self._is_all_declared_equipment_available is None:
            # Availability has not already been evaluated: determine availability
            self._is_all_declared_equipment_available = self.healthcare_system.equipment.is_all_items_available(
                item_codes=self._EQUIPMENT,
                facility_id=self.facility_info.id,
            )
        return self._is_all_declared_equipment_available

    def probability_all_equipment_available(self, item_codes: Union[int, str, Iterable[int], Iterable[str]]) -> float:
        """Returns the probability that all the equipment item_codes are available. This does not imply that the
        equipment is being used and no logging happens. It is provided as a convenience to disease module authors in
        case the logic during an ``HSI_Event`` depends on the availability of a piece of equipment. This function
        accepts the item codes/descriptions in a variety of formats, so the argument needs to be parsed."""
        return self.healthcare_system.equipment.probability_all_equipment_available(
            item_codes=self.healthcare_system.equipment.parse_items(item_codes),
            facility_id=self.facility_info.id,
        )

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
            equipment=tuple(sorted(self._EQUIPMENT)),
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
