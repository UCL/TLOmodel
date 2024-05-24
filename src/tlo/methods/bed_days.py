from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from tlo import Date

if TYPE_CHECKING:
    from tlo.logging.core import Logger


@dataclass
class BedOccupancy:
    """
    Logs an allocation of bed-days resources, and when it will be freed up.
    """

    bed_type: str
    facility: int
    freed_date: Date
    patient_id: int
    start_date: Date

    def __post_init__(self) -> None:
        """
        Post-init checks for BedOccupancies:
        - start_date is not later than freed_date
        """
        assert self.start_date <= self.freed_date, (
            f"BedOccupancy created which starts ({self.start_date.strftime('%Y-%m-%d')}) "
            f"later than it finishes ({self.freed_date.strftime('%Y-%m-%d')})!"
        )


class BedDaysFootprint(dict):
    """
    Represents an allocation of bed days, otherwise acts as a dictionary.

    Is initialised with attributes corresponding to the different types of
    available beds in the simulation. After instantiation, no additional
    keys may be added to the object, but the number of days the beds are
    requested for may be updated, if necessary.

    It is necessary for footprints to track 0-day bed requests so that when
    it comes to allocating bed days, higher-priority bed requests that
    cannot be fulfilled can be cascaded down to lower-priority bed requests.
    """

    def __bool__(self) -> bool:
        """
        Implicit boolean-ness equates to whether this footprint imposes any
        bed days or not.
        """
        return bool(self.total_days())

    def __init__(
        self, permitted_bed_types: Iterable[str], **initial_values: int | float
    ) -> None:
        """ """
        super().__init__({b: 0 for b in permitted_bed_types})

        for bed_type, n_days in initial_values.items():
            self[bed_type] = n_days

    def __setitem__(self, bed_type: str, n_days: int | float) -> None:
        """ """
        assert (
            bed_type in self.keys()
        ), f"{bed_type} is not a valid bed type for a bed occupancy footprint."
        assert (
            n_days >= 0
        ), f"Cannot assign negative amount of days ({n_days}) to bed of type {bed_type}."
        super().__setitem__(bed_type, n_days)

    def as_occupancies(
        self,
        first_day: Date,
        facility: int,
        patient_id: int,
    ) -> List[BedOccupancy]:
        """
        Covert a bed days footprint to a list of bed occupancies, with the first
        bed-day occurring on the date provided.

        :param footprint: Bed days footprint to apply.
        :param first_day: Bed occupation of this footprint will start from the date
        provided.
        :param facility: The facility that will host the patient.
        :param patient_id: DataFrame index of the patient who will occupy the bed.
        """
        start_date_of_occupancy = first_day
        new_occupancies = []
        for bed_type, occupancy_length in self.without_0s().items():
            new_occupancy = BedOccupancy(
                bed_type=bed_type,
                facility=facility,
                patient_id=patient_id,
                start_date=start_date_of_occupancy,
                freed_date=start_date_of_occupancy
                + pd.DateOffset(
                    days=occupancy_length - 1
                ),  # 1 day occupancy starts and ends on same day!
            )
            # Next occupancy will start the day after this one ended
            start_date_of_occupancy += pd.DateOffset(occupancy_length)
            # Record the occupancy and move on to the next
            new_occupancies.append(new_occupancy)
        return new_occupancies

    def total_days(self) -> int | float:
        """
        Total number of bed days that this footprint imposes.

        Specifically, the sum of the values.
        """
        return sum(self.values())

    def without_0s(self) -> Dict[str, int | float]:
        """
        Return a dictionary representation of the object without the
        bed_type: n_days pairs where the number of days is 0.
        """
        return {bed: n_days for bed, n_days in self.items() if n_days > 0}


# Define the appointment types that should be associated with the use of bed-days (of any type), for a given number of
# patients.
IN_PATIENT_ADMISSION = {"IPAdmission": 2}
# One of these appointments is for the admission and the other is for the discharge (even patients who die whilst an
# in-patient require discharging). The limitation is that the discharge appointment occurs on the same day as the
# admission. See: https://github.com/UCL/TLOmodel/issues/530

IN_PATIENT_DAY_FIRST_DAY = {"InpatientDays": 0}
# There is no in-patient appointment day needed on the first day, as the care is covered under the admission.

IN_PATIENT_DAY_SUBSEQUENT_DAYS = {"InpatientDays": 1}
# Care required on days after the day of admission (including the day of discharge).


class BedDays:
    """
    Tracks bed days resources, in a better way than using dataframe columns...
    TODO: Docstring

    bed_capacities is intended to replace self.hs_module.parameters["BedCapacity"]
    availability has been superseded by capacity_scaling_factor

    :param capacity_scaling_factor: Capacities read from resource files are scaled
    by this factor. "all" will map to 1.0 (use provided occupancies), whilst "none"
    will map to 0.0 (no beds available anywhere). Default is 1.
    """

    # Class-wide variable: the name given to the "bed_type" that actually
    # indicates there are no beds available at a particular facility.
    __NO_BEDS_BED_TYPE: str = "non_bed_space"

    # All available bed types, including the "non_bed_space".
    # Tuple order dictates bed priority (earlier beds are higher priority).
    _bed_types: Tuple[str]
    # Index: FacilityID, Cols: max capacity of given bed type
    _max_capacities: pd.DataFrame
    # Logger to write info, warnings etc to
    _logger: Logger
    # Counter class for the summary statistics
    _summary_counter: BedDaysSummaryCounter

    # List of current bed occupancies, scheduled and waiting
    occupancies: List[BedOccupancy]

    @property
    def bed_facility_level(self) -> str:
        """Facility level at which all beds are taken to be pooled."""
        return "3"

    @property
    def bed_types(self) -> Tuple[str]:
        """
        All available bed types, including the bed type that corresponds to no
        space being available in a facility.

        Tuple order dictates bed priority: beds types that appear first in the
        tuple are higher priority than those which follow, with the no space
        available bed being the lowest priority.
        """
        return self._bed_types

    @bed_types.setter
    def bed_types(self, bed_names: Iterable[str]) -> None:
        assert (
            self.__NO_BEDS_BED_TYPE in bed_names
            and self.__NO_BEDS_BED_TYPE == bed_names[-1]
        ), (
            "Lowest priority bed type (corresponding to no available bed space)"
            f" must be {self.__NO_BEDS_BED_TYPE}, but it is {bed_names[-1]}"
        )
        self._bed_types = tuple(x for x in bed_names)

    @staticmethod
    def date_ranges_overlap(
        start_1: Date, start_2: Date, end_1: Date, end_2: Date
    ) -> bool:
        """
        Return True if there is any overlap between the timeboxed intervals [start_1, end_1]
        and [start_2, end_2].
        Endpoints are included in the date ranges.
        """
        latest_start = max(start_1, start_2)
        earliest_end = min(end_1, end_2)
        delta = (earliest_end - latest_start).days + 1
        return max(0, delta) > 0

    @staticmethod
    def add_first_day_inpatient_appts_to_footprint(
        appt_footprint: Dict[Any, int | float]
    ) -> Dict[Any, int | float]:
        """Return an APPT_FOOTPRINT with the addition (if not already present)
        of the in-patient admission appointment and the in-patient day
        appointment type (for the first day of the in-patient stay).

        TODO: Not sure why this is a method in BedDays as it doesn't concern BedDays...
        maybe move elsewhere.
        """
        return {**appt_footprint, **IN_PATIENT_ADMISSION, **IN_PATIENT_DAY_FIRST_DAY}

    @staticmethod
    def multiply_footprint(_footprint: Dict[Any, int | float], _num: int):
        """
        Multiply the number of appointments of each type in a footprint by a number.

        Specifically, multiples all values in a dictionary by a scalar,
        returning the result:
        {key: value * _num for key, value in _footprint.items()}

        TODO: Not sure why this is a method in BedDays as it doesn't concern BedDays...
        maybe move elsewhere.
        """
        return {
            appt_type: num_needed * _num for appt_type, num_needed in _footprint.items()
        }

    def __init__(
        self,
        bed_capacities: pd.DataFrame,
        capacity_scaling_factor: float | Literal["all", "none"] = 1.0,
        logger: Optional[Logger] = None,
        summary_logger: Optional[Logger] = None,
    ) -> None:
        """
        TODO
        NB two loggers because
                logger = getLogger("tlo.methods.healthsystem")
        logger_summary = getLogger("tlo.methods.healthsystem.summary")

        one of which is only used by the summary tracker. Move this to main class docstring when you come to write the docstrings Will.
        """
        self._logger = logger
        self.bed_types = tuple(x for x in bed_capacities.columns if x != "Facility_ID")

        # Determine the (scaled) bed capacity of each facility, and store the information
        if isinstance(capacity_scaling_factor, str):
            if capacity_scaling_factor == "all":
                capacity_scaling_factor = 1.0
            elif capacity_scaling_factor == "none":
                capacity_scaling_factor = 0.0
            else:
                raise ValueError(
                    f"Cannot interpret capacity scaling factor: {capacity_scaling_factor}."
                    " Provide a numeric value, 'all' (1.0), or 'none' (0.0)."
                )
        self._max_capacities = (
            (
                bed_capacities.set_index("Facility_ID", inplace=False)
                * capacity_scaling_factor
            )
            .apply(np.ceil)
            .astype(int)
        )

        # No occupancies on instantiation
        self.occupancies = []
        # Summary counter starts at 0
        self._summary_counter = BedDaysSummaryCounter(logging_target=summary_logger)

    def is_inpatient(self, patient_id: int) -> List[BedOccupancy]:
        """
        Return a list of bed occupancies this person is scheduled for.
        Note that the implicit boolean cast of this method's return
        value will be True if and only if the patient is an inpatient.

        A person is an inpatient if they are currently occupying a bed,
        or are scheduled to occupy a bed in the future.

        Programmatically, they are an inpatient if their patient_id
        appears in any of the occupancies that are scheduled.

        :param patient_id: Index of the patient in the population DataFrame.
        """
        return self.find_occupancies(patient_id=patient_id)

    def bed_type_to_priority(self, bed_type: str) -> int:
        """
        Return the priority ranking of the given bed type,
        which is determined by the order of occurrence in the
        self.bed_types tuple.

        Earlier occurrences are higher priority.
        """
        return self.bed_types.index(bed_type)

    def end_occupancies(self, *occupancies: BedOccupancy) -> None:
        """
        Action taken when a bed occupancy in the list of occupancies
        is to end, for any reason.

        :param occupancy: The BedOccupancy(s) that have ended.
        """
        for o in occupancies:
            self.occupancies.remove(o)

    def start_of_day(self, todays_date: Date) -> None:
        """
        Actions to take at the start of a new day.
        Currently:
        - End any bed occupancies that are set to be freed today.

        :param todays_date: The day that is starting.
        """
        # End bed occupancies that expired yesterday
        for o in self.find_occupancies(
            end_on_or_before=todays_date - pd.DateOffset(days=1)
        ):
            self.end_occupancies(o)

    def schedule_occupancies(self, *occupancies: BedOccupancy) -> None:
        """
        Bulk schedule the provided bed occupancies.
        NOTE: Occupancies are assumed to be valid.
        """
        self.occupancies.extend(occupancies)

    def find_occupancies(
        self,
        facility: Optional[List[int] | int] = None,
        patient_id: Optional[List[int] | int] = None,
        logical_or: bool = False,
        end_on_or_before: Optional[Date] = None,
        on_date: Optional[Date] = None,
        start_on_or_after: Optional[Date] = None,
        occurs_between_dates: Optional[Tuple[Date]] = None,
    ) -> List[BedOccupancy]:
        """
        Find all occupancies in the current list that match the criteria given.
        Unspecified criteria are ignored.
        
        Multiple criteria will be combined using logical AND. This behaviour can
        be toggled with the logical_or argument.

        :param facility: Facility the occupancy takes place at.
        :param patient_id: ID in the population DataFrame for the inpatient.
        :param logical_or: Combine criteria via logical OR rather than AND.
        :param end_on_or_before: Occupancy must end before or on this date.
        :param occurs_between_dates: At least part of the occupancy must occur
        between these two dates. Provided as a Tuple of Dates, the first element
        being the earlier date (inclusive) and the second element the later
        date (inclusive).
        :param start_on_or_after: Occupancy must start on or after this date.
        """
        # Cast single-values to lists to make parsing easier
        if isinstance(patient_id, int):
            patient_id = [patient_id]
        if isinstance(facility, int):
            facility = [facility]
        # Correct logical operator to use
        logic_operation = any if logical_or else all

        matches = [
            o
            for o in self.occupancies
            if logic_operation(
                [
                    patient_id is None or o.patient_id in patient_id,
                    facility is None or o.facility in facility,
                    start_on_or_after is None or o.start_date >= start_on_or_after,
                    end_on_or_before is None or o.freed_date <= end_on_or_before,
                    on_date is None or o.start_date <= on_date <= o.freed_date,
                    occurs_between_dates is None
                    or self.date_ranges_overlap(
                        o.start_date,
                        occurs_between_dates[0],
                        o.freed_date,
                        occurs_between_dates[1],
                    ),
                ]
            )
        ]
        return matches

    def forecast_availability(
        self,
        start_date: Date,
        n_days: int,
        facility_id: int,
        as_bool: bool = False,
        int_indexing: bool = False,
    ) -> pd.DataFrame:
        """
        Return a n_days forecast of the number of beds available at the given facility.

        Forecast is provided as a `pd.DataFrame` whose rows are indexed from the start
        date to the end date, and whose columns are indexed by the bed type.

        Values in the DataFrame are the number of beds of that type available
        on the given day. This behaviour can be changed with the as_bool and
        int_indexing arguments.

        :param start_date: The date on which the forecast will start.
        :param n_days: Number of days to forecast. Final day in forecast will INCLUDE
        the day n_days after the start.
        :param facility_id: Facility to forecast bed allocations of.
        :param as_bool: If True, DataFrame values are booleans indicating whether at least
        one bed of the given type is free on a particular day (True) or not (False).
        :param int_indexing: If True, DataFrame row indices will run from 0 through to
        n_days, rather than being indexed by the dates themselves.
        """
        final_forecast_day = start_date + pd.DateOffset(days=n_days)
        relevant_occupancies = self.find_occupancies(
            occurs_between_dates=[start_date, final_forecast_day],
            facility=facility_id,
        )
        facility_max_capacities = self._max_capacities.loc[facility_id, :]
        # Rows = index by date, days into the future (index 0 = start_date)
        # Cols = bed_types for this facility
        forecast = pd.DataFrame(
            data=[facility_max_capacities] * (n_days + 1),
            index=pd.date_range(start=start_date, end=final_forecast_day, freq="D"),
            columns=facility_max_capacities.index,
            dtype=float,
        )
        # Forecast has been initialised with max capacities for each bed
        # now go through the relevant occupancies to determine the actual capacities for these days!
        for o in relevant_occupancies:
            forecast.loc[o.start_date : o.freed_date, o.bed_type] -= 1

        if as_bool:
            # Convert to true/false values based on whether there is at least one bed available
            forecast = pd.DataFrame(forecast > 0, dtype=bool)
        if int_indexing:
            # Convert from datetime indexes to 0-based int indexing
            forecast.rename(
                index={date: i for i, date in enumerate(forecast.index)}, inplace=True
            )

        return forecast

    def resolve_overlapping_occupancies(
        self,
        incoming_occupancies: List[BedOccupancy],
        current_occupancies: List[BedOccupancy],
    ) -> List[BedOccupancy]:
        """
        Resolve conflicting lists of bed days occupancies, returning the simplest
        consistent allocation.

        The simplest consistent allocation minimises the total number of occupancies
        that are required, and on any given day allocates the patient to the highest
        priority bed that the two sets of occupancies wish to provide.

        Occupancy conflicts are resolved in the following manner:
        - Determine the date interval spanned by all events to consider.
        - Assemble an array containing the current bed type allocated for each day that
        the date interval spans.
        - Assemble an array containing the incoming bed type to be allocated for each day
        that the date interval spans.
        - Take the element-wise "priority maximum" of the two arrays - this represents higher
        priority beds taking precedence over lower-priority allocations.
        - Convert the resulting array into a list of bed occupancies, and return.

        It is assumed that the patient_id and facility of all occupancies provided
        are identical (as otherwise, this process does not make sense). Furthermore,
        patients are only ever able to attend one facility in their district / region
        for bed care anyway, so this should never arise as an issue.

        It is assumed that the patient is assumed to be scheduled to occupy at least
        one bed type for each day between the start of the earliest occupancy and end
        of the last occupancy. If there are any days in this range where the patient
        is not going to occupy a bed, an AssertionError is raised. In these
        circumstances, the event which is not in conflict with the others should be
        removed from one of the lists prior to calling this method.
        """
        all_occupancies = incoming_occupancies + current_occupancies
        # Assume all events have same facility and patient_id
        facility = all_occupancies[0].facility
        patient_id = all_occupancies[0].patient_id

        earliest_start = min([o.start_date for o in all_occupancies])
        latest_end = max([o.start_date for o in all_occupancies])

        # Create an array/DF spanning the time interval of the two sets of
        # occupancies. Initially, fill it with values corresponding to the
        # lowest priority bed.
        lowest_priority = len(self.bed_types) - 1
        bed_on_each_day = pd.Series(
            index=pd.date_range(earliest_start, latest_end, freq="D"),
            data=lowest_priority,
        )
        # Then, for each occupancy, overwrite the assigned bed type for the
        # duration of the occupancy if and only if it wants to assign a
        # higher priority bed than the one already assigned for that day.
        # Recall that "higher priority" are "earlier bed type indices"
        for o in all_occupancies:
            priority = self.bed_type_to_priority(o.bed_type)
            bed_on_each_day[o.start_date : o.freed_date].loc[
                bed_on_each_day[o.start_date : o.freed_date] > priority
            ] = priority
        # At this point, we should have no "lowest priority" beds assigned
        # since these are always "non_bed_space", IE not beds.
        # In such cases, the occupancies should not have been scheduled in
        # anyway, so throw an error.
        assert (bed_on_each_day < len(self.bed_types) - 1).all(), (
            "Patient is scheduled to have at least one day in a non-bed-space "
            "when resolving conflicting bed day occupancies. "
            "This implies that at least one occupancy in the two sets of "
            "occupancies does not conflict with the others. "
            "Such non-conflicting occupancies should be removed before "
            "calling this method."
        )

        # We now know the bed allocation for this person.
        # Convert it back to a list of bed occupancies.
        dates_bed_occupancies_change = bed_on_each_day.diff()[
            bed_on_each_day.diff() != 0
        ].index.values
        dates_bed_occupancies_change = np.append(
            dates_bed_occupancies_change, bed_on_each_day.index[-1]
        )

        reconciled_occupancies = []
        for occ_starts, occ_ends in zip(
            dates_bed_occupancies_change[:-1],
            dates_bed_occupancies_change[1:] - pd.DateOffset(days=1),
        ):
            bed_type = self.bed_types[bed_on_each_day[occ_starts]]
            assert self.bed_type_to_priority(bed_type) != lowest_priority, (
                "Patient is scheduled to have at least one day in a non-bed-space "
                "when resolving conflicting bed day occupancies. "
                "This implies that at least one occupancy in the two sets of "
                "occupancies does not conflict with the others. "
                "Such non-conflicting occupancies should be removed before "
                "calling this method."
            )
            reconciled_occupancies.append(
                BedOccupancy(
                    bed_type=bed_type,
                    facility=facility,
                    patient_id=patient_id,
                    start_date=occ_starts,
                    freed_date=occ_ends,
                )
            )
        return reconciled_occupancies

    def remove_patient_footprint(self, patient_id: int) -> None:
        """
        Remove all occupancies scheduled by the patient.
        Typically used when a patient dies or is otherwise removed
        from the simulation.

        :param patient_id: Index in the population DataFrame of the
        individual whose occupancies are to be cancelled.
        """
        self.end_occupancies(*self.find_occupancies(patient_id=patient_id))

    def assert_valid_footprint(self, footprint: BedDaysFootprint) -> None:
        """
        Assert that the footprint provided does not allocate any
        time to the no-beds-available bed type (and is thus invalid).
        """
        assert footprint[self.__NO_BEDS_BED_TYPE] == 0, (
            "Invalid footprint: "
            "requests non-zero allocation of days to the no-beds-available type."
        )

    def get_blank_beddays_footprint(
        self, **initial_values: int | float
    ) -> BedDaysFootprint:
        """
        Return a BedDaysFootprint with the given days allocated to each bed type,
        or zero days if not specified.
        """
        return BedDaysFootprint(self.bed_types, **initial_values)

    def get_inpatient_appts(self, date: Date) -> Dict[str, Dict[str, int | float]]:
        """
        Return a dictionary of the form {<facility_id>: APPT_FOOTPRINT},
        giving the total APPT_FOOTPRINT required for the servicing of the
        in-patients (in beds of any types) for each Facility_ID.
        """
        # For each facility, compute the total number of beds that are occupied
        # total_inpatients has n_facilities elements that are the number of beds occupied
        active_occupancies_today = self.find_occupancies(on_date=date)
        total_inpatients: Dict[int, int] = defaultdict(int)
        for o in active_occupancies_today:
            total_inpatients[o.facility] += 1

        # Construct the appointment footprint for all facilities with inpatients
        inpatient_appointments = {
            fac_id: self.multiply_footprint(
                IN_PATIENT_DAY_SUBSEQUENT_DAYS, num_inpatients
            )
            for fac_id, num_inpatients in total_inpatients.items()
        }
        return inpatient_appointments

    def impose_beddays_footprint(
        self,
        footprint: BedDaysFootprint,
        facility: int,
        first_day: Date,
        patient_id: int,
    ) -> None:
        """
        Impose the footprint provided on the availability of beds.
        """
        # Exit if the footprint is empty
        if not footprint:
            return

        new_footprint_end_date = first_day + pd.DateOffset(
            days=footprint.total_days() - 1
        )
        new_occupancies = footprint.as_occupancies(first_day, facility, patient_id)

        scheduled_occupancies = self.is_inpatient(patient_id=patient_id)
        conflicting_occupancies = [
            o
            for o in scheduled_occupancies
            if self.date_ranges_overlap(
                o.start_date, first_day, o.freed_date, new_footprint_end_date
            )
        ]

        if conflicting_occupancies:
            # This person is already an inpatient.
            # For those occupancies that conflict, we will need to overwrite the lower
            # priority bed occupancies with the higher priority bed occupancies.
            new_occupancies = self.resolve_overlapping_occupancies(
                new_occupancies, conflicting_occupancies
            )

            # Remove all conflicting dependencies that are currently scheduled,
            # before we add the resolved conflicts
            for o in conflicting_occupancies:
                self.end_occupancies(o)

        # Schedule the new occupancies, which are now conflict-free
        # (if they weren't already)
        self.schedule_occupancies(*new_occupancies)

    def issue_bed_days_according_to_availability(
        self, start_date: Date, facility_id: int, requested_footprint: BedDaysFootprint
    ) -> BedDaysFootprint:
        """
        Return the 'best possible' footprint can be provided to an HSI,
        given the current bed allocations.

        The rules for determining the 'best possible' footprint, given a requested
        footprint current state of bed occupancy is:

        - For each type of bed specified in the footprint, in order from highest tier
        to lowest tier, check if there are sufficient bed-days available of that type:
          - Provide as many consecutive days in that bed-type as possible to this HSI.
          - Re-allocate any remaining days to the next bed-type.

        The lowest-priority bed ranking is 'non_bed_space'. If the number of days to be
        allocated to this bed type is non-zero, then the footprint cannot be supported.
        That is, the requested footprint cannot be met given the current availability of
        beds.

        :param start_date: The date (inclusive) from which bed occupancy is to start.
        :param facility_id: The facility at which beds will be used.
        :param requested_footprint: The bed days footprint to attempt to allocate.
        """
        available_footprint = self.get_blank_beddays_footprint()
        # If footprint is empty, then the returned footprint is empty too.
        if not requested_footprint:
            return available_footprint
        footprint_length = requested_footprint.total_days()

        forecast_availability = self.forecast_availability(
            start_date=start_date,
            n_days=footprint_length - 1,
            facility_id=facility_id,
            as_bool=True,
            int_indexing=True,
        )

        # This tracks how many days of the previous bed type we could not provide,
        # and thus must attempt to provide with the following bed type.
        day_deficit = 0
        # The total number of days for which we have allocated beds (so far)
        days_allocated = 0
        # Move through the beds that have been requested and attempt to allocate days
        for bed_type in self.bed_types:
            # We want to allocate the number of days to this bed type as the footprint
            # requests, PLUS any that were held over from unavailability of other beds.
            days_to_allocate = requested_footprint[bed_type] + day_deficit
            # Determine if we can provide a bed for the number of days required,
            # from the date in the future we currently are at.
            can_be_allocated = forecast_availability.loc[
                days_allocated : days_allocated + days_to_allocate - 1,
                bed_type,
            ]
            # If we can allocate all the requested bed_days of this type,
            # update the available footprint and move on to the next bed type
            if can_be_allocated.all():
                available_footprint[bed_type] = days_to_allocate
                # If we could allocated all desired beds, we have made up
                # any deficit carried over from other bed types too.
                day_deficit = 0
                days_allocated += days_to_allocate
            else:
                # Bed is not available for all the days it was requested.
                # Assume it is unavailable after the first day, and assign
                # the bed for the days up to this one.
                # Then hold over the remaining days for the next bed type in the
                # footprint.
                n_days_that_can_be_provided = can_be_allocated.argmin()
                # Provide as many days as possible for this bed type
                available_footprint[bed_type] = n_days_that_can_be_provided
                # Record that there is now a deficit in the number of days that we've
                # allocated thus far
                day_deficit = days_to_allocate - n_days_that_can_be_provided
                days_allocated += n_days_that_can_be_provided

        return available_footprint

    def on_end_of_day(self, day_that_is_ending: Date) -> None:
        """
        Actions that are to be taken at the end of a day.

        Current actions are:
        - Log bed occupancies, per bed type, for today.
        - Update year-long summary statistics
        - Remove expired bed occupancies. This must be done after the logging step,
        since occupancies are still in effect on their freed_date.

        :param day_that_is_ending: The simulation day that is coming to an end.
        """
        occupancies_today = self.find_occupancies(on_date=day_that_is_ending)
        bed_type_by_facility: Dict[str, Dict[str, int]] = defaultdict(defaultdict(int))
        for o in occupancies_today:
            bed_type_by_facility[o.bed_type][o.facility] += 1

        # Dump today's status of bed-day tracker to the debugging log.
        # Logger is expected to report {fac: available beds}, for each bed type.
        if self._logger is not None:
            for bed_type, occupancy_info in bed_type_by_facility.items():
                self._logger.info(
                    key=f"bed_tracker_{bed_type}",
                    data=occupancy_info,
                    description=f"Use of bed_type {bed_type}, by day and facility",
                )
        # Summary counter is expected to log total bed usage (and remaining capacity) by
        # bed type.
        bed_usage = {
            bed_type: sum(by_facility.values())
            for bed_type, by_facility in bed_type_by_facility.items()
        }
        # Record the total usage of each bed type today (across all facilities)
        self._summary_counter.record_usage_of_beds(
            bed_usage, self._max_capacities
        )

        # Remove any occupancies that expire today,
        # after having logged their occurrence!
        expired_occupancies = self.find_occupancies(end_on_or_before=day_that_is_ending)
        for expired in expired_occupancies:
            self.end_occupancies(expired)

    def on_end_of_year(self, **kwargs) -> None:
        # SOME LOGGING NEEDS TO BE DONE
        raise ValueError("Will you need to impliment this method")

    def on_simulation_end(self, *args, **kwargs) -> None:
        pass

    # THESE MAY NOT BE NEEDED BUT ARE THERE TO KEEP THE REWORK HAPPY
    # Will change all these to be raising errors once we know what needs replacing

    @property
    def availability(self) -> None:
        # don't think this attribute needs to be tracked anymore,
        # but there's a random event that thinks it should change it.
        # But then the codebase doesn't actually use the value
        # of this attribute AFTER that point in the simulation anyway.
        # Have opened https://github.com/UCL/TLOmodel/issues/1346
        raise ValueError("Tried to access BedDays.availability")


class BedDaysSummaryCounter:
    """
    Helper class to keep running counts of bed-days used.

    Bed usage is stored in dictionaries, whose structure is:
        {<bed_type>: <number_of_beddays>}.
    Both the number of bed days used, and the number of bed
    days available, are recorded by this class.
    """

    _bed_days_used: Dict[str, int]
    _bed_days_available: Dict[str, int]

    def __init__(self, logging_target: Optional[Logger] = None):
        self.logger = logging_target
        self._reset_internal_stores()

    def _reset_internal_stores(self) -> None:
        self._bed_days_used = defaultdict(int)
        self._bed_days_available = defaultdict(int)

    def record_usage_of_beds(self, bed_days_used: Dict[str, int], max_capacities: Dict[str, int]) -> None:
        """Add record of usage of beds. `bed_days_used` is a dict of the form
        {<bed_type>: <total_beds_available_across_facilities>}.
        """
        for _bed_type, days_used in bed_days_used.items():
            max_cap = max_capacities[_bed_type]
            self._bed_days_used[_bed_type] += days_used
            self._bed_days_available[_bed_type] += max_cap - days_used

    def write_to_log_and_reset_counters(self):
        """Log summary statistics and reset the data structures."""

        if self.logger is not None:
            self.logger.info(
                key="BedDays",
                description="Counts of the bed-days that have been used (by type).",
                data=self._bed_days_used,
            )

            self.logger.info(
                key="FractionOfBedDaysUsed",
                description="Fraction of the bed-days available in the last year that were used (by type).",
                data={
                    _bed_type: self._bed_days_used[_bed_type] / _total
                    for _bed_type, _total in self._bed_days_available.items()
                },
            )

        self._reset_internal_stores()
