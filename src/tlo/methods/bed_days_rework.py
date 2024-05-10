from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd

from tlo import Date

BedDaysFootprint: TypeAlias = Dict[str, float | int]

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
        assert (
            self.start_date <= self.freed_date
        ), f"BedOccupancy created which starts ({self.start_date}) later than it finishes ({self.freed_date})!"

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

class BedDaysRework:
    """
    Tracks bed days resources, in a better way than using dataframe columns
    """
    # Index: FacilityID, Cols: max capacity of given bed type
    _max_capacities: pd.DataFrame
    # List of bed occupancies, on removal from the queue, the resources an occupancy takes up
    # are returned to the DF
    occupancies: List[BedOccupancy]

    bed_types: Tuple[str]

    @property
    def bed_facility_level(self) -> str:
        """Facility level at which all beds are taken to be pooled."""
        return "3"

    @staticmethod
    def date_ranges_overlap(start_1: Date, start_2: Date, end_1: Date, end_2: Date) -> bool:
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
        return {appt_type: num_needed * _num for appt_type, num_needed in _footprint.items()}

    @staticmethod
    def total_footprint_days(_footprint: Dict[Any, int | float]) -> int:
        """
        The total number of days that this beddays footprint requires.

        Specifically, the sum of the values of the keys.
        """
        return sum(_footprint.values())

    def __init__(
        self,
        bed_capacities: pd.DataFrame,
        capacity_scaling_factor: float | Literal["all", "none"] = 1.0,
    ) -> None:
        """
        bed_capacities is intended to replace self.hs_module.parameters["BedCapacity"]
        availability has been superseded by capacity_scaling_factor

        :param capacity_scaling_factor: Capacities read from resource files are scaled by this factor. "all" will map to 1.0, whilst "none" will map to 0.0 (no beds available anywhere). Default is 1.
        """
        # List of bed-types
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
        # Create the dataframe that will track the current bed occupancy of each facility
        self.occupied_beds = pd.DataFrame(0, index=self._max_capacities.index.copy(), columns=self._max_capacities.columns.copy())

        # Set the initial list of bed occupancies as an empty list
        self.occupancies = []

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

        The occupancy is removed from the list of events.

        :param occupancy: The BedOccupancy that has ended.
        """
        for o in occupancies:
            self.occupancies.remove(o)

    def start_of_day(self, todays_date: Date) -> None:
        """
        Actions to take at the start of a new day.

        - End any bed occupancies that are set to be freed today.
        """
        # End bed occupancies that expired yesterday
        for o in self.find_occupancies(
            end_on_or_before=todays_date - pd.DateOffset(days=1)
        ):
            self.end_occupancies(o)

    def schedule_occupancies(self, *occupancies: BedOccupancy) -> None:
        """
        Bulk schedule the provided bed occupancies.
        Occupancies are assumed to be valid.
        """
        self.occupancies.extend(occupancies)

    def find_occupancies(
        self,
        end_on_or_before: Optional[Date] = None,
        facility: Optional[List[int] | int] = None,
        on_date: Optional[Date] = None,
        patient_id: Optional[List[int] | int] = None,
        start_on_or_after: Optional[Date] = None,
    ) -> List[BedOccupancy]:
        """
        Find all occupancies in the current list that match the criteria given.

        :param end_on_or_before: Only match occupancies that are scheduled to end on or before the date provided.
        :param facility: Only match occupancies that take place at the given facility (or facilities if list).
        :param on_date: Only match occupancies that occur on the given date. end_on_or_before and start_on_or_before will be ignored if provided.
        :param patient_id: Only match occupancies scheduled for the given person (or persons if list).
        :param start_on_or_after: Only match occupancies that are scheduled to start on or after the date provided.
        """
        # Cast single-values to lists to make parsing easier
        if isinstance(patient_id, int):
            patient_id = [patient_id]
        if isinstance(facility, int):
            facility = [facility]
        if on_date is not None:
            # Overwrite any other date inputs if a specific day was requested.
            end_on_or_before = on_date
            start_on_or_after = on_date

        matches = [
            o
            for o in self.occupancies
            if (patient_id is None or o.patient_id in patient_id)
            and (facility is None or o.facility in facility)
            and (start_on_or_after is None or o.start_date < start_on_or_after)
            and (end_on_or_before is None or end_on_or_before < o.freed_date)
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
            end_on_or_before=final_forecast_day,
            start_on_or_after=start_date,
            facility=facility_id,
        )
        facility_max_capacities = self._max_capacities.loc[facility_id, :]
        # Rows = index by date, days into the future (index 0 = start_date)
        # Cols = bed_types for this facility
        forecast = pd.DataFrame(
            data=[facility_max_capacities] * (n_days + 1),
            index=pd.date_range(start=start_date, end=final_forecast_day, freq='D'),
            columns=facility_max_capacities.index,
            dtype=float,
        )
        # Forecast has been initialised with max capacities for each bed
        # now go through the relevant occupancies to determine the actual capacities for these days!
        for o in relevant_occupancies:
            forecast.loc[o.start_date:o.freed_date, o.bed_type] -= 1

        if as_bool:
            # Convert to true/false values based on whether there is at least one bed available
            forecast = pd.DataFrame(forecast > 0, dtype=bool)
        if int_indexing:
            # Convert from datetime indexes to 0-based int indexing
            forecast.rename(index={date: i for i, date in enumerate(forecast.index)}, inplace=True)

        return forecast

    def resolve_overlapping_occupancies(self, incoming_occupancies: List[BedOccupancy], current_occupancies: List[BedOccupancy]) -> List[BedOccupancy]:
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
        - Take the element-wise "maximum" of the two arrays - this represents higher
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

        # We now know the bed allocation for this person
        # Convert it back to a list of bed occupancies
        dates_bed_occupancies_change = bed_on_each_day.diff()[bed_on_each_day.diff() != 0].index.values
        dates_bed_occupancies_change = np.append(dates_bed_occupancies_change, bed_on_each_day.index[-1])

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

    # THESE MAY NOT BE NEEDED BUT ARE THERE TO KEEP THE REWORK HAPPY
    # Will change all these to be raising errors once we know what needs replacing

    @property
    def availability(self) -> None:
        # don't think this attribute needs to be tracked anymore,
        # but there's a random event that thinks it should change it.
        # But then the codebase doesn't actually use the value
        # of this attribute AFTER that point in the simulation anyway.
        raise ValueError("Tried to access BedDays.availability")

    def remove_beddays_footprint(self, *args, **kwargs) -> None:
        raise ValueError("Will you need to impliment this method")

    def check_beddays_footprint_format(self, *args, **kwargs) -> None:
        """
        Checking for a valid beddays footprint format is done like 20+ times in the entire codebase and it's getting both old and redundant. We don't need to check this if we're adhering to a strict format!"""
        pass

    def pre_initialise_population(self, *args, **kwargs) -> None:
        """We don't need to do anything, I think...
        TODO: Validate then remove this method and it's call in the new HealthSystem"""
        pass

    def get_blank_beddays_footprint(self, *args, **kwargs) -> Dict[str, float | int]:
        """
        Provides a dictionary whose keys are the bed types and initial values are 0.

        It is necessary for footprints to track 0-day requests so that when it comes
        to allocating bed days, higher-priority bed requests that cannot be fulfilled
        can be cascaded down to lower-priority bed requests.
        """
        return {bed_type: 0 for bed_type in self.bed_types}

    def get_inpatient_appts(self, date: Date, *args, **kwargs) -> None:
        """
        Return a dictionary of the form {<facility_id>: APPT_FOOTPRINT},
        giving the total APPT_FOOTPRINT required for the servicing of the
        in-patients (in beds of any types) for each Facility_ID.
        """
        # For each facility, compute the total number of beds that are occupied
        # total_inpatients has n_facilities elements that are the number of beds occupied
        active_occupancies_today = self.find_occupancies(on_date=date)
        total_inpatients: Dict[int, int] = {}
        for o in active_occupancies_today:
            try:
                total_inpatients[o.facility] += 1
            except KeyError:
                # This facility has not been added as a key yet,
                # so start the counter at 1
                total_inpatients[o.facility] = 1

        # Construct the appointment footprint for all facilities with inpatients
        inpatient_appointments = {
            fac_id: self.multiply_footprint(IN_PATIENT_DAY_SUBSEQUENT_DAYS, num_inpatients)
            for fac_id, num_inpatients in total_inpatients.items()
        }
        return inpatient_appointments

    def footprint_to_occupancies(
        self,
        footprint: BedDaysFootprint,
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
        # Don't pollute the list of occupancies with 0-length objects.
        footprint_without_0s = {
            bed_type: length for bed_type, length in footprint.items() if length > 0
        }

        start_date_of_occupancy = first_day
        new_occupancies = []
        for bed_type, occupancy_length in footprint_without_0s.items():
            new_occupancy = BedOccupancy(
                bed_type=bed_type,
                facility=facility,
                patient_id=patient_id,
                start_date=start_date_of_occupancy,
                freed_date=start_date_of_occupancy
                + pd.DateOffset(days=occupancy_length - 1), # 1 day occupancy starts and ends on same day!
            )
            # Next occupancy will start the day after this one ended
            start_date_of_occupancy += pd.DateOffset(occupancy_length)
            # Record the occupancy and move on to the next
            new_occupancies.append(new_occupancy)
        return new_occupancies

    def impose_beddays_footprint(
        self, footprint: BedDaysFootprint, facility: int, first_day: Date, patient_id: int,
    ) -> None:
        """
        Impose the footprint provided on the availability of beds.
        """
        # Exit if the footprint is empty
        if self.total_footprint_days(footprint) == 0:
            return

        new_footprint_end_date = first_day + pd.DateOffset(
            days=self.total_footprint_days(footprint) - 1
        )
        new_occupancies = self.footprint_to_occupancies(
            footprint, first_day, facility, patient_id
        )

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

    def initialise_population(self, *args, **kwargs) -> None:
        """We don't need to do anything, I think...
        TODO: Validate then remove this method and it's call in the new HealthSystem"""
        pass

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
        # Compute footprint that can be provided
        # First, check the forecast bed availability for this facility
        footprint_length = self.total_footprint_days(requested_footprint)
        # If footprint is empty, then the returned footprint is empty too.
        if footprint_length == 0:
            return self.get_blank_beddays_footprint()

        forecast_availability = self.forecast_availability(
            start_date=start_date,
            n_days=footprint_length - 1,
            facility_id=facility_id,
            as_bool=True,
            int_indexing=True,
        )

        available_footprint = {}
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

    def on_birth(self, *args, **kwargs) -> None:
        raise ValueError("Will you need to impliment this method")

    def on_start_of_day(self, *args, **kwargs) -> None:
        """
        Don't think we need to do anything here. But at the end of the day we will need to remove events that expired today.
        """
        pass

    def on_end_of_day(self, day_that_is_ending: Date) -> None:
        """
        The old method does some logging here and no other activities.
        But we need to clean up the occupancy queue and remove the ones that expired.

        :param day_that_is_ending: The simulation day that is coming to an end.
        """
        expired_occupancies = self.find_occupancies(end_on_or_before=day_that_is_ending)

        for expired in expired_occupancies:
            self.end_occupancies(expired)

        print(f"Removed {len(expired_occupancies)} expired bed occupancies.")

    def on_end_of_year(self, **kwargs) -> None:
        raise ValueError("Will you need to impliment this method")

    def on_simulation_end(self, *args, **kwargs) -> None:
        pass # legit is the content in the non-rework too
