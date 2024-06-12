from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from tlo import Date

if TYPE_CHECKING:
    from tlo.logging.core import Logger

import os
class Debugger:

    output_dir: str = ".wills-stuff/beddays/"

    first_write: bool = True

    @property
    def output_file(self) -> str:
        return self.output_dir + "exp-branch-log.log"

    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def write(self, message: str) -> None:
        if self.first_write:
            self.first_write = False
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
        if not message.endswith("\n"):
            message += "\n"
        with open(self.output_file, "a") as f:
            f.write(message)

DEBUGGER = Debugger()

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

    @property
    def length(self) -> int:
        """
        Number of days this occupancy runs for,
        = (freed_date - start_date).days + 1
        """
        return (self.freed_date - self.start_date).days + 1

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

    NOTE: We encounter
    https://github.com/pandas-dev/pandas/issues/8757#issuecomment-1479522330
    in BedDays if we don't explicitly cast the n_days value to a native
    Python type. For whatever reason, pd.DateOffset doesn't like numpy integers.
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
        if type(n_days).__module__ == "numpy":
            n_days = n_days.item()
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
    Tracks the allocation of the BedDays resource.

    Each facility in the simulation provides a given number of bed spaces for
    each "bed type", with certain beds being reserved for high-priority cases.
    As the simulation progresses, patients will be admitted to a facility based
    on whether the facility has the capacity to provide a bed for them, and
    once they have recovered their allocation will be freed.

    The class tracks the available number of beds by recording "occupancies".
    Each time a person needs to be admitted to a facility, they provide a
    footprint which details how many days (of each bed) that they will need
    before they recover. This footprint is converted into a set of `BedOccupancies`
    which this class tracks.

    From this information, this class is able to:
    - Provide a forecast any number of days into the future on the number of
    available beds, broken down by type and facility if necessary.
    - Determine whether or not there is capacity for a person to receive a particular
    treatment.
    - Determine which members of the public are currently inpatients.
    - Log the number of used and free beds.

    :param bed_capacities: DataFrame whose rows consist of the facility IDs of the
    facilities in the simulation, and the columns their maximum bed capacities.
    :param capacity_scaling_factor: Capacities read from resource files are scaled
    by this factor. "all" will map to 1.0 (use provided occupancies), whilst "none"
    will map to 0.0 (no beds available anywhere). Default is 1.
    :param logger: Logger object to write outputs and debug information to.
    :param summary_logger: Logger object to write summary statistics to.
    """

    # Class-wide variable: the name given to the "bed_type" that actually
    # indicates there are no beds available at a particular facility.
    __NO_BEDS_BED_TYPE: str = "non_bed_space"

    # All available bed types, including the "non_bed_space".
    # Tuple order dictates bed priority (earlier beds are higher priority).
    _bed_types: Tuple[str]
    # Index: FacilityID, Cols: max capacity of given bed type
    # This is the table passed in from the HealthSystem parameters,
    # which we need for when the availability switch event fires.
    _raw_max_capacities: pd.DataFrame
    # Logger to write info, warnings etc to
    _logger: Logger
    # Counter class for the summary statistics
    _summary_counter: BedDaysSummaryCounter

    # Facility level at which all beds are taken to be pooled.
    # This is a class-wide variable.
    bed_facility_level: str = "3"
    # Table of maximum bed capacities, subject to availability
    # constraints
    max_capacities: pd.DataFrame
    # List of current bed occupancies, scheduled and waiting
    occupancies: List[BedOccupancy]

    @property
    def bed_types(self) -> Tuple[str]:
        """
        All available bed types, including the bed type that corresponds to no
        space being available in a facility.

        Tuple order dictates bed priority: bed types that appear first in the
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

    @property
    def all_inpatients(self) -> List[int]:
        """
        Return a list of the (unique) person_ids of those
        people who are inpatients.
        """
        return list({o.patient_id for o in self.occupancies})

    @staticmethod
    def date_ranges_overlap(
        start_1: Date, start_2: Date, end_1: Date, end_2: Date
    ) -> bool:
        """
        Return True if there is any overlap between the time-boxed intervals
        [start_1, end_1] and [start_2, end_2].
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
        """
        return {**appt_footprint, **IN_PATIENT_ADMISSION, **IN_PATIENT_DAY_FIRST_DAY}

    @staticmethod
    def multiply_footprint(_footprint: Dict[Any, int | float], _num: int):
        """
        Multiply the number of appointments of each type in a footprint by a number.

        Specifically, multiples all values in a dictionary by a scalar,
        returning the result:
        {key: value * _num for key, value in _footprint.items()}
        """
        return {
            appt_type: num_needed * _num for appt_type, num_needed in _footprint.items()
        }

    def __init__(
        self,
        bed_capacities: pd.DataFrame,
        capacity_scaling_factor: Optional[float | Literal["all", "none"]] = None,
        logger: Optional[Logger] = None,
        summary_logger: Optional[Logger] = None,
    ) -> None:
        self._logger = logger
        self._raw_max_capacities = bed_capacities.set_index("Facility_ID", inplace=False)
        self.bed_types = tuple(x for x in bed_capacities.columns if x != "Facility_ID")

        # Note that the simulation may not have setup the initial population
        # when the BedDays class is initialised, so we need to account for the
        # possibility that the scaled bed capacity will be provided later.
        if capacity_scaling_factor is not None:
            self.set_max_capacities(capacity_scaling_factor)

        # No occupancies on instantiation
        self.occupancies = []
        # Summary counter starts at 0
        self._summary_counter = BedDaysSummaryCounter(logging_target=summary_logger)

    def set_max_capacities(
        self,
        capacity_scaling_factor: float | Literal["all", "none"],
        fallback_value: Optional[float] = None,
    ) -> None:
        """
        Set the new maximum capacities either for the first time, or update them based on
        a change in availability event during the simulation.

        :param capacity_scaling_factor: The new scaling factor for the raw maximum capacities.
        :param fallback_value: If the capacity_scaling_factor is given as a string, but is not
        "all" or "none", use this value as the scaling factor.
        """
        # Determine the (scaled) bed capacity of each facility, and store the information
        if capacity_scaling_factor == "all":
            capacity_scaling_factor = 1.0
        elif capacity_scaling_factor == "none":
            capacity_scaling_factor = 0.0
        elif isinstance(capacity_scaling_factor, str) and fallback_value is not None:
            capacity_scaling_factor = fallback_value
        elif not isinstance(capacity_scaling_factor, (float | int)):
            raise ValueError(
                f"Cannot interpret capacity scaling factor: {capacity_scaling_factor}."
                " Provide a numeric value, 'all' (1.0), or 'none' (0.0)."
            )
        # Update new effective bed capacities
        self.max_capacities = (self._raw_max_capacities * capacity_scaling_factor).apply(np.ceil).astype(int)
        DEBUGGER.write(f"Setting max capacities:\n{self.max_capacities}")

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

        :param occupancies: The BedOccupancy(s) that have ended.
        """
        for o in occupancies:
            self.occupancies.remove(o)

    def schedule_occupancies(self, *occupancies: BedOccupancy) -> None:
        """
        Bulk schedule the provided bed occupancies.

        This method does not check that there is capacity for the
        occupancies provided to be scheduled. Use the `assert_valid_footprint`
        and `impose_beddays_footprint` to ensure generated BedOccupancies
        can be provided.

        :param occupancies: The BedOccupancy(s) to be scheduled.
        """
        self.occupancies.extend(occupancies)

    def find_occupancies(
        self,
        facility: Optional[List[int] | int] = None,
        patient_id: Optional[List[int] | int] = None,
        logical_or: bool = False,
        end_on_or_before: Optional[Date] = None,
        start_on_or_after: Optional[Date] = None,
        on_date: Optional[Date] = None,
        occurs_between_dates: Optional[Tuple[Date]] = None,
    ) -> List[BedOccupancy]:
        """
        Find all occupancies in the current list of occupancies that match the
        criteria given. Unspecified criteria are ignored.
        
        Multiple criteria will be combined using logical AND. This behaviour can
        be toggled with the logical_or argument.

        :param facility: Facility the occupancy takes place at.
        :param patient_id: ID in the population DataFrame for the inpatient.
        :param logical_or: Combine criteria via logical OR rather than AND.
        :param end_on_or_before: Occupancy must end before or on this date.
        :param start_on_or_after: Occupancy must start on or after this date.
        :param on_date: Occupancy must start on or after this date, and end on or before it.
        :param occurs_between_dates: At least part of the occupancy must occur
        between these two dates. Provided as a Tuple of Dates, the first element
        being the earlier date (inclusive) and the second element the later
        date (inclusive).
        """
        # Cast single-values to lists to make parsing easier
        if isinstance(patient_id, int):
            patient_id = [patient_id]
        if isinstance(facility, int):
            facility = [facility]
        # Correct logical operator to use
        if logical_or:
            matches = [
                o
                for o in self.occupancies
                if any(
                    [
                        patient_id is not None and o.patient_id in patient_id,
                        facility is not None and o.facility in facility,
                        start_on_or_after is not None and o.start_date >= start_on_or_after,
                        end_on_or_before is not None and o.freed_date <= end_on_or_before,
                        on_date is not None and o.start_date <= on_date <= o.freed_date,
                        occurs_between_dates is not None
                        and self.date_ranges_overlap(
                            o.start_date,
                            occurs_between_dates[0],
                            o.freed_date,
                            occurs_between_dates[1],
                        ),
                    ]
                )
            ]
        else:
            matches = [
                o
                for o in self.occupancies
                if all(
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
        facility_max_capacities = self.max_capacities.loc[facility_id, :]
        # Rows = index by date, days into the future (index 0 = start_date)
        # Cols = bed_types for this facility
        forecast = pd.DataFrame(
            data=[facility_max_capacities] * (n_days + 1),
            index=pd.date_range(start=start_date, end=final_forecast_day, freq="D"),
            columns=facility_max_capacities.index,
            dtype=int,
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
        Resolve conflicting lists of bed days occupancies, a consistent allocation.

        The consistent allocation minimises the total number of occupancies
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

        :param incoming_occupancies: A list of occupancies that are to be scheduled, but
        conflict with existing occupancies.
        :param current_occupancies: The occupancies currently scheduled that will conflict
        with the incoming occupancies.
        """
        all_occupancies = incoming_occupancies + current_occupancies
        # Assume all events have same facility and patient_id
        facility = all_occupancies[0].facility
        patient_id = all_occupancies[0].patient_id

        earliest_start = min([o.start_date for o in all_occupancies])
        latest_end = max([o.freed_date for o in all_occupancies])

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
        if not (bed_on_each_day < len(self.bed_types) - 1).all():
            self._logger.warning(
                key="message",
                data=(
                    f"Patient {patient_id} is scheduled to have at least one day "
                    "in a non-bed-space when resolving conflicting bed day occupancies."
                ),
            )

        # We now know the bed allocation for this person.
        # Convert it back to a list of bed occupancies.
        dates_bed_occupancies_change = bed_on_each_day.diff()[
            bed_on_each_day.diff() != 0
        ].index.values
        dates_bed_occupancies_end = np.append(
            dates_bed_occupancies_change[1:] - np.timedelta64(1, "D"),
            bed_on_each_day.index.values[-1],
        )

        reconciled_occupancies = []
        for occ_starts, occ_ends in zip(
            dates_bed_occupancies_change,
            dates_bed_occupancies_end,
        ):
            bed_type = self.bed_types[bed_on_each_day[occ_starts]]
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

    def combine_overlapping_occupancies(
        self,
        incoming_occupancies: List[BedOccupancy],
        current_occupancies: List[BedOccupancy],
    ) -> List[BedOccupancy]:
        """
        Resolve conflicting lists of bed days occupancies, returning a consistent
        allocation.

        The allocation returned requests (for each bed type) a number of bed days
        equal to the maximum number of days the current and incoming occupancies
        request.

        Occupancy conflicts are resolved in the following manner:
        - Convert both the current occupancies and incoming occupancies to footprints.
        - Take the key-wise maximum of the resulting footprints to form the resolved
        footprint.
        - Cast the resolved footprint back to a list of occupancies.

        It is assumed that the patient_id and facility of all occupancies provided
        are identical (as otherwise, this process does not make sense). Furthermore,
        patients are only ever able to attend one facility in their district / region
        for bed care anyway, so this should never arise as an issue.

        :param incoming_occupancies: A list of occupancies that are to be scheduled, but
        conflict with existing occupancies.
        :param current_occupancies: The occupancies currently scheduled that will conflict
        with the incoming occupancies.
        """
        # Plan: convert to a footprint that can then be imposed from
        # the start_date
        all_occupancies = incoming_occupancies + current_occupancies
        earliest_start = min([o.start_date for o in all_occupancies])

        current_time_in_beds = self.get_blank_beddays_footprint()
        for o in current_occupancies:
            current_time_in_beds[o.bed_type] += o.length
        incoming_time_in_beds = self.get_blank_beddays_footprint()
        for o in incoming_occupancies:
            incoming_time_in_beds[o.bed_type] += o.length

        combined_footprint = self.get_blank_beddays_footprint()
        for bed_type, current_n_days in current_time_in_beds.items():
            incoming_n_days = incoming_time_in_beds[bed_type]
            combined_footprint[bed_type] = max(current_n_days, incoming_n_days)

        # Having created the "combined footprint", turn it into a list
        # of occupancies
        facility = all_occupancies[0].facility
        patient_id = all_occupancies[0].patient_id
        return combined_footprint.as_occupancies(earliest_start, facility, patient_id)

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

        :param date: Day to extract number of inpatient appointments on.
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
        overlay_instead_of_combine: bool = False,
    ) -> bool:
        """
        Impose the footprint provided on the availability of beds.
        Return True/False indicating whether or not the person is a new inpatient.

        In the event that the person is already an inpatient, it is necessary to
        reconcile their existing occupancies with the new ones they will receive
        from being admitted.
        - The default resolution option is add the time (in each bed type) that their
        incoming occupancy imposes to their current allocation. However, this means
        we encounter https://github.com/UCL/TLOmodel/issues/1399.
        - Alternatively, footprints can be overlaid by determining the highest priority
        bed on each day that the person is due to occupy. This is the method suggested
        at the end of https://github.com/UCL/TLOmodel/issues/1399, and provides the
        guarantee that beds will never be over-allocated by mistake. However, it also
        reduces the amount of bed-time allocated overall and is distinct from the
        default behaviour, so can (and will) lead to different population evolution.

        :param footprint: Footprint to impose.
        :param facility: Which facility will host this footprint.
        :param first_day: Day on which this footprint will start.
        :param patient_id: The index in the population DataFrame of the person
        occupying this bed.
        :param overlay_instead_of_combine: If True, footprints are overlaid rather than combined.
        """
        # Exit if the footprint is empty
        if not footprint:
            return False
        conflict_resolver = (
            self.resolve_overlapping_occupancies
            if overlay_instead_of_combine
            else self.combine_overlapping_occupancies
        )

        new_footprint_end_date = first_day + pd.DateOffset(
            days=footprint.total_days() - 1
        )
        new_occupancies = footprint.as_occupancies(first_day, facility, patient_id)

        # Identify any occupancies this person currently has during the period
        # this footprint is expected to run over, to resolve potential conflicts.
        conflicting_occupancies = self.find_occupancies(
            patient_id=patient_id,
            occurs_between_dates=(first_day, new_footprint_end_date),
        )

        is_new_inpatient = True
        if conflicting_occupancies:
            # This person is already an inpatient.
            # For those occupancies that conflict, we will need to overwrite the lower
            # priority bed occupancies with the higher priority bed occupancies.
            new_occupancies = conflict_resolver(
                new_occupancies, conflicting_occupancies
            )

            # Remove all conflicting dependencies that are currently scheduled,
            # before we add the resolved conflicts
            self.end_occupancies(*conflicting_occupancies)

            # This person is not a new inpatient
            is_new_inpatient = False

        # Schedule the new occupancies, which are now conflict-free
        # (if they weren't already)
        self.schedule_occupancies(*new_occupancies)

        DEBUGGER.write(f"{first_day} scheduling for patient {patient_id} |")
        for o in new_occupancies:
            DEBUGGER.write(
                f"\t{o.bed_type} freed on {o.freed_date}, at facility {o.facility}\n"
            )
        DEBUGGER.write("\tHaving received footprint:")
        for bed_type, days in footprint.items():
            DEBUGGER.write(f"\t\t{bed_type} : {days}")

        return is_new_inpatient

    def issue_bed_days_according_to_availability(
        self, start_date: Date, facility_id: int, requested_footprint: BedDaysFootprint
    ) -> BedDaysFootprint:
        """
        Return the 'best possible' footprint can be provided to an HSI,
        given the current bed allocations.

        The rules for determining the 'best possible' footprint, given a requested
        footprint and the current state of bed occupancies is:

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

        DEBUGGER.write(
            f"{start_date} issuing for facility {facility_id} |"
        )
        for bed_type, n_days in available_footprint.items():
            if bed_type in requested_footprint.keys():
                wanted_days = requested_footprint[bed_type]
            else:
                wanted_days = 0
            DEBUGGER.write(f"\t{bed_type} : {n_days} (wanted {wanted_days})")

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
        bed_type_by_facility: Dict[str, Dict[str, int]] = {
            bed_type: defaultdict(int) for bed_type in self.bed_types
        }
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
            bed_usage, self.max_capacities.sum(axis=0).to_dict()
        )

        # Remove any occupancies that expire today,
        # after having logged their occurrence!
        expired_occupancies = self.find_occupancies(end_on_or_before=day_that_is_ending)
        for expired in expired_occupancies:
            self.end_occupancies(expired)

    def on_end_of_year(self) -> None:
        """
        Actions to be taken at the end of a simulation year.
        """
        self._summary_counter.write_to_log_and_reset_counters()

class BedDaysSummaryCounter:
    """
    Helper class to keep running counts of bed-days used.

    Bed usage is stored in dictionaries, whose structure is:
        {<bed_type>: <number_of_beddays>}.
    Both the number of bed days used, and the number of bed
    days available, are recorded by this class.

    :param logging_target: The Logger instance to write outputs to.
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
        """
        Record the use of beds, provided as a dictionary.

        :param bed_days_used: Dictionary of the form
        {<bed_type>: total beds of this type used across all facilities since last record}.

        :param max_capacities: Dictionary of the form
        {<bed_type>: sum of the maximum capacities of beds of this type across all facilities}.
        """
        for _bed_type, days_used in bed_days_used.items():
            max_cap = max_capacities[_bed_type]
            self._bed_days_used[_bed_type] += days_used
            self._bed_days_available[_bed_type] += max_cap - days_used

    def write_to_log_and_reset_counters(self):
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
