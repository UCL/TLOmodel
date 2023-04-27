"""
This is the Bed days class.

It maintains a current record of the availability and usage of beds in the healthcare system.

"""
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tlo import Property, Types, logging

# ---------------------------------------------------------------------------------------------------------
#   CLASS DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger('tlo.methods.healthsystem')
logger_summary = logging.getLogger('tlo.methods.healthsystem.summary')

# Define the appointment types that should be associated with the use of bed-days (of any type), for a given number of
# patients.
IN_PATIENT_ADMISSION = {'IPAdmission': 2}
# One of these appointments is for the admission and the other is for the discharge (even patients who die whilst an
# in-patient require discharging). The limitation is that the discharge appointment occurs on the same day as the
# admission. See: https://github.com/UCL/TLOmodel/issues/530

IN_PATIENT_DAY = {'InpatientDays': 1}


class BedDays:
    """
    The BedDays class. This is expected to be registered in the HealthSystem module.
    """

    def __init__(self, hs_module, availability: str = 'default'):
        self.hs_module = hs_module

        # Number of days to the last day of bed_tracker
        self.days_until_last_day_of_bed_tracker = 150

        # A dictionary to create a footprint according to facility bed days capacity
        self.available_footprint = {}

        # A dictionary to track inpatient bed days
        self.bed_tracker = dict()
        self.list_of_cols_with_internal_dates = dict()

        # List of bed-types
        self.bed_types = list()

        self.availability = availability

        # Internal store of the number of beds by facility and bed_type that is scaled to the model population size (if
        #  the HealthSystem is not running in mode `disable=True`).
        self._scaled_capacity = None

        # Create pointer to the `BedDaysSummaryCounter` helper class
        self._summary_counter = BedDaysSummaryCounter()

    def get_bed_types(self):
        """Helper function to get the bed_types from the resource file imported to parameter 'BedCapacity' of the
        health system module"""
        return [x for x in self.hs_module.parameters['BedCapacity'].columns if x != 'Facility_ID']

    def pre_initialise_population(self):
        """Define the properties that will then be added to the sim.population.props dataframe by the healthsystem
        module """

        self.bed_types = self.get_bed_types()
        for bed_type in self.bed_types:
            self.hs_module.PROPERTIES[f"hs_next_first_day_in_bed_{bed_type}"] = Property(
                Types.DATE, f"Date when person will next enter bed_type {bed_type}. (pd.NaT) is nothing scheduled")
            self.hs_module.PROPERTIES[f"hs_next_last_day_in_bed_{bed_type}"] = Property(
                Types.DATE,
                f"Date of the last day in the next stay in bed_type {bed_type}. (pd.NaT) is nothing scheduled")

        # Create store for columns names
        self.list_of_cols_with_internal_dates['entries'] = [
            f"hs_next_first_day_in_bed_{bed_type}" for bed_type in self.bed_types]
        self.list_of_cols_with_internal_dates['exits'] = [
            f"hs_next_last_day_in_bed_{bed_type}" for bed_type in self.bed_types]
        self.list_of_cols_with_internal_dates['all'] = \
            self.list_of_cols_with_internal_dates['entries'] + self.list_of_cols_with_internal_dates['exits']

    def initialise_population(self, df):
        df.loc[df.is_alive, 'hs_is_inpatient'] = False

        # Put pd.NaT for all the properties concerned with entry/exit of different types of bed.
        df.loc[df.is_alive, self.list_of_cols_with_internal_dates['all']] = pd.NaT

    def on_birth(self, df, mother_id, child_id):
        df.at[child_id, 'hs_is_inpatient'] = False
        df.loc[child_id, self.list_of_cols_with_internal_dates['all']] = pd.NaT

    def on_simulation_end(self):
        pass

    def set_scaled_capacity(self, model_to_data_popsize_ratio):
        """Set the internal `_scaled_capacity` variable to represent the number of beds available of each type in each
         facility, after scaling according to the model population relative to the real population size. """

        if self.availability == 'all':
            _scaling_factor = 1.0
        elif self.availability == 'none':
            _scaling_factor = 0.0
        else:
            _scaling_factor = model_to_data_popsize_ratio

        self._scaled_capacity = (
            self.hs_module.parameters['BedCapacity'].set_index('Facility_ID') * _scaling_factor
        ).apply(np.ceil).astype(int)

    def initialise_beddays_tracker(self, model_to_data_popsize_ratio=1.0):
        """Initialise the bed days tracker:
        Create a dataframe for each type of beds that give the total number of beds currently available in each facility
         (rows) by the date during the simulation (columns).

        The current implementation assumes that bed capacity is held constant throughout the simulation; but it could be
         changed through modifications here.

         :param: `capabilities_coefficient` is the scaler needed to reduce the number of beds available according to the
         size of the model population relative to the real population size.
        """

        # Set the internal `_scaled_capacity` variable to reflect the model population size.
        self.set_scaled_capacity(model_to_data_popsize_ratio)

        max_number_of_bed_days = self.days_until_last_day_of_bed_tracker
        end_date = self.hs_module.sim.start_date + pd.DateOffset(days=max_number_of_bed_days)
        date_range = pd.date_range(self.hs_module.sim.start_date, end_date, freq='D')

        for bed_type in self.bed_types:
            df = pd.DataFrame(
                index=date_range,  # <- Days in the simulation
                columns=self._scaled_capacity.index,  # <- Facility_ID
                data=1
            )
            df = df.mul(self._scaled_capacity[bed_type], axis=1)
            assert not df.isna().any().any()
            self.bed_tracker[bed_type] = df

    def on_start_of_day(self):
        """Things to do at the start of each new day:
        * Refresh inpatient status
        * Log yesterday's usage of beds
        * Move the tracker by one day
        * Schedule an HSI for today that represents the care of in-patients
        """
        # Refresh the hs_in_patient status
        self.refresh_in_patient_status()

        # Move tracker by one day
        # NB. This is skipped on the first day of the simulation as there is nothing to log from yesterday and the
        # tracker is already set.
        if self.hs_module.sim.date != self.hs_module.sim.start_date:
            self.move_each_tracker_by_one_day()

    def move_each_tracker_by_one_day(self):

        for bed_type, tracker in self.bed_tracker.items():
            start_date = min(tracker.index)

            # reset all the columns for the start_date with the values of `bed_capacity` - this row is going to become
            # the new day (at the end of the tracker)
            tracker.loc[start_date] = self._scaled_capacity[bed_type]

            # make new index
            end_date = max(tracker.index)  # get the latest day in the dataframe
            new_day = end_date + pd.DateOffset(days=1)  # the new day is the next day after the last in the tracker
            new_index = list(tracker.index)
            new_index[0] = new_day  # the earliest day is replaced with the next day
            new_index = pd.DatetimeIndex(new_index)

            # update the index and sort the index (will put the 'new_day' at the end of the index).
            tracker = tracker.set_index(new_index).sort_index()

            # save the updated tracker
            self.bed_tracker[bed_type] = tracker

    def on_end_of_day(self):
        """Do the actions required at the end of each day"""
        self.log_todays_info_from_all_bed_trackers()

    def log_todays_info_from_all_bed_trackers(self):
        """Log the occupancy of beds for today."""
        today = self.hs_module.sim.date

        # 1) Dump today's status of bed-day tracker to the debugging log
        for bed_type, tracker in self.bed_tracker.items():
            occupancy_info = tracker.loc[today].to_dict()

            logger.info(
                key=f'bed_tracker_{bed_type}',
                data=occupancy_info,
                description=f'Use of bed_type {bed_type}, by day and facility'
            )

        # 2) Record the total usage of each bed type today (across all facilities)
        self._summary_counter.record_usage_of_beds(
            {
                bed_type:
                    (self._scaled_capacity[bed_type].sum(), tracker.iloc[0].sum())  # (Total, Number-available-now)
                for bed_type, tracker in self.bed_tracker.items()
             }
        )

    def get_blank_beddays_footprint(self):
        """
        Generate a blank footprint for the bed-days
        :return: a footprint of the correct format, specifying no bed days
        """
        assert 0 < len(self.bed_types), "No bed types have been defined"
        return {b: 0 for b in self.bed_types}

    def check_beddays_footprint_format(self, beddays_footprint):
        """Check that the format of the beddays footprint is correct"""
        assert type(beddays_footprint) is dict
        assert len(self.bed_types) == len(beddays_footprint)
        assert all([(bed_type in beddays_footprint) for bed_type in self.bed_types])
        assert all([((v >= 0) and (type(v) is int)) for v in beddays_footprint.values()])
        if 'non_bed_space' in self.bed_types:
            assert beddays_footprint['non_bed_space'] == 0, "A request cannot be made for a non-bed space"

    def issue_bed_days_according_to_availability(self, facility_id: int, footprint: dict) -> dict:
        """Return the 'best possible' footprint can be provided to an HSI, given the current status of the trackers.
        The rules for determining the 'best possible' footprint, given a requested footprint and the current state of
         the trackers are as follows:
        * For each type of bed specified in the footprint (in order from highest tier to lowest tier), check if there
         are sufficient bed-days available of that type:
           * Provide as many consecutive days in that bed-type as possible to this HSI.
           * Re-allocate any remaining days to the next bed-type.
        """

        # If footprint is empty, then the returned footprint is empty too
        if footprint == self.get_blank_beddays_footprint():
            return footprint

        # Convert the footprint into a format that will make it easy to compare with the trackers
        dates_for_bed_use = self.compute_dates_of_bed_use(footprint)
        dates_for_bed_use_not_null = [_date for _date in dates_for_bed_use.values() if pd.notnull(_date)]
        dates_for_bed_use_date_range = pd.date_range(min(dates_for_bed_use_not_null), max(dates_for_bed_use_not_null))
        footprint_as_date_ranges = dict()
        for _bed_type in self.bed_types:
            if pd.notnull(dates_for_bed_use[f'hs_next_first_day_in_bed_{_bed_type}']):
                footprint_as_date_ranges[_bed_type] = list(pd.date_range(
                    dates_for_bed_use[f'hs_next_first_day_in_bed_{_bed_type}'],
                    dates_for_bed_use[f'hs_next_last_day_in_bed_{_bed_type}'])
                )
            else:
                footprint_as_date_ranges[_bed_type] = list()

        # Compute footprint that can be provided
        available_footprint = dict()
        hold_over_dates_for_next_bed_type = None
        for _bed_type in self.bed_types:
            # Add in any days needed for this bed-type held over from higher bed-types
            if hold_over_dates_for_next_bed_type:
                footprint_as_date_ranges[_bed_type].extend(hold_over_dates_for_next_bed_type)
                footprint_as_date_ranges[_bed_type].sort()

            # Check if beds are available on each day
            tracker = self.bed_tracker[_bed_type][facility_id].loc[dates_for_bed_use_date_range]
            available = tracker[footprint_as_date_ranges[_bed_type]] > 0

            if not available.all():
                # If the bed is not available on all days, assume it cannot be used after the first day
                # that it is not available.
                available.loc[available[~available].index[0]:] = False

                # Add any days for which a bed of this type is not available to the footprint for next bed-type:
                hold_over_dates_for_next_bed_type = list(available.loc[~available].index)

            else:
                hold_over_dates_for_next_bed_type = None

            # Record the days that are allocated:
            available_footprint[_bed_type] = int(available.sum())

        return available_footprint

    def impose_beddays_footprint(self, person_id, footprint):
        """This is called to reflect that a new occupancy of bed-days should be recorded:
        * Cause to be reflected in the bed_tracker that an hsi_event is being run that will cause bed to be
         occupied.
        * Update the property ```hs_is_inpatient``` to show that this person is now an in-patient

         NB. If multiple bed types are required, then it is assumed that these run in the sequence given in
         ```bed_types```.
         """
        # Exit if the footprint is empty
        if footprint == self.get_blank_beddays_footprint():
            return

        df = self.hs_module.sim.population.props

        if not df.at[person_id, 'hs_is_inpatient']:
            # apply the new footprint if the person is not already an in-patient
            self.apply_footprint(person_id, footprint)
            # label person as an in-patient
            df.at[person_id, 'hs_is_inpatient'] = True

        else:
            # if person is already an in-patient:
            # calculate how much time left in each bed remains for the person who is already an in-patient
            remaining_footprint = self.get_remaining_footprint(person_id)

            # combine the remaining footprint with the new footprint, with days in each bed-type running concurrently:
            combo_footprint = {bed_type: max(footprint[bed_type], remaining_footprint[bed_type])
                               for bed_type in self.bed_types
                               }

            # remove the old footprint and apply the combined footprint
            self.remove_beddays_footprint(person_id)
            self.apply_footprint(person_id, combo_footprint)

    def apply_footprint(self, person_id, footprint):
        """Edit the internal properties in the dataframe to reflect this in-patient stay"""

        # check that the number of inpatient days does not exceed the maximum of 150 days
        if self.days_until_last_day_of_bed_tracker < sum(footprint.values()):
            logger.warning(
                key='warning',
                data=f'the requested bed days in footprint is greater than the tracking period, {footprint}'
            )

        df = self.hs_module.sim.population.props
        # reset all internal properties about dates of transition between bed use states:
        df.loc[person_id, self.list_of_cols_with_internal_dates['all']] = pd.NaT

        # compute the entry/exit dates of bed use states:
        dates_of_bed_use = self.compute_dates_of_bed_use(footprint)

        # record these dates in the internal properties:
        df.loc[person_id, dates_of_bed_use.keys()] = dates_of_bed_use.values()

        # enter these to tracker
        self.edit_bed_tracker(
            dates_of_bed_use=dates_of_bed_use,
            person_id=person_id,
            add_footprint=True
        )

    def edit_bed_tracker(self, dates_of_bed_use, person_id, add_footprint=True):
        """Helper function to record the usage (or freeing-up) of beds in the bed-tracker
        dates_of_bed_use: the dates of entry/exit from beds of each type
        person_id: used to find the facility to use
        add_footprint: whether the footprint should be added (i.e. consume a bed), or the reversed (i.e. free a bed).
            The latter is used to when a footprint is removed when a person dies or before a new footprint is added.
        """
        # Exit silently if bed_tracker has not been initialised
        if not hasattr(self, 'bed_tracker'):
            return

        the_facility_id = self.get_facility_id_for_beds(person_id)
        operation = -1 if add_footprint else 1

        for bed_type in self.bed_types:
            date_start_this_bed = dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"]
            date_end_this_bed = dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"]

            if pd.isna(date_start_this_bed or date_end_this_bed):  # filter empty bed days
                pass
            else:
                self.bed_tracker[bed_type].loc[date_start_this_bed: date_end_this_bed, the_facility_id] += \
                    operation

    def compute_dates_of_bed_use(self, footprint):
        """Helper function to compute the dates of entry/exit from beds of each type according to a bed-days footprint
         (which provides information in terms of number of whole days).
        NB. It is always assumed that the footprint begins with today's date. """
        now = self.hs_module.sim.date
        start_allbeds = now
        end_allbeds = now + pd.DateOffset(days=sum(footprint.values()) - 1)

        dates_of_bed_use = dict()

        start_this_bed = start_allbeds
        for bed_type in self.bed_types:
            if footprint[bed_type] > 0:
                end_this_bed = start_this_bed + pd.DateOffset(days=footprint[bed_type] - 1)

                # record these dates:
                dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"] = start_this_bed
                dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"] = end_this_bed

                # get ready for next bed type:
                start_this_bed = end_this_bed + pd.DateOffset(days=1)
            else:
                dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"] = pd.NaT
                dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"] = pd.NaT

        # check that dates run as expected
        assert (start_this_bed - pd.DateOffset(days=1)) == end_allbeds

        return dates_of_bed_use

    def get_remaining_footprint(self, person_id):
        """Helper function to work out how many days remaining in each bed-type of current stay for a given person."""
        df = self.hs_module.sim.population.props
        now = self.hs_module.sim.date

        d = df.loc[person_id, self.list_of_cols_with_internal_dates['all']]
        remaining_footprint = self.get_blank_beddays_footprint()

        for bed_type in self.bed_types:
            if d[f'hs_next_last_day_in_bed_{bed_type}'] >= now:
                remaining_footprint[bed_type] = 1 + \
                                                (
                                                    d[f'hs_next_last_day_in_bed_{bed_type}']
                                                    - max(now, d[f'hs_next_first_day_in_bed_{bed_type}'])
                                                ).days
                # NB. The '+1' accounts for the fact that 'today' is included
        return remaining_footprint

    def get_facility_id_for_beds(self, persons_id):
        """Helper function to find the facility at which an HSI event will take place.
        We say that all the beds are pooled at the level 3."""

        the_district = self.hs_module.sim.population.props.at[persons_id, 'district_of_residence']
        facility_level = '3'

        # Return an id of the (one) health_facility available to this person (based on their district)
        return self.hs_module._facilities_for_each_district[facility_level][the_district][0]

    def remove_beddays_footprint(self, person_id):
        """Helper function that will remove from the bed-days tracker the days of bed-days remaining for a person.
        This is called when the person dies or when a new footprint is imposed"""

        remaining_footprint = self.get_remaining_footprint(person_id)
        dates_of_bed_use = self.compute_dates_of_bed_use(remaining_footprint)
        self.edit_bed_tracker(
            dates_of_bed_use=dates_of_bed_use,
            person_id=person_id,
            add_footprint=False
        )

    def refresh_in_patient_status(self):
        df = self.hs_module.sim.population.props
        exit_cols = self.list_of_cols_with_internal_dates['exits']

        # if any "date of last day in bed" in not null and in the future, then the person is an in-patient:
        df.loc[df.is_alive, "hs_is_inpatient"] = \
            (df.loc[df.is_alive, exit_cols].notnull() & (
                df.loc[df.is_alive, exit_cols] >= self.hs_module.sim.date)).any(axis=1)

    @staticmethod
    def add_first_day_inpatient_appts_to_footprint(appt_footprint):
        """Return an APPT_FOOTPRINT with the addition (if not already present) of the in-patient admission appointment
        and the in-patient day appointment type (for the first day of the in-patient stay)."""
        return {**appt_footprint, **IN_PATIENT_ADMISSION, **IN_PATIENT_DAY}

    def get_inpatient_appts(self) -> dict:
        """Return a dict of the form {<facility_id>: APPT_FOOTPRINT} giving the total APPT_FOOTPRINT required for the
        servicing of the in-patients (in beds of any types) for each Facility_ID."""

        total_inpatients = pd.DataFrame([
            (self._scaled_capacity[_bed_type] - self.bed_tracker[_bed_type].loc[self.hs_module.sim.date]).to_dict()
            for _bed_type in self.bed_types
        ]).sum()

        def multiply_footprint(_footprint, _num):
            """Multiply the number of appointments of each type in a footprint by a number"""
            return {appt_type: num_needed * _num for appt_type, num_needed in _footprint.items()}

        return {
            fac_id: multiply_footprint(IN_PATIENT_DAY, num_inpatients)
            for fac_id, num_inpatients in total_inpatients[total_inpatients > 0].to_dict().items()
        }
        # NB. As we haven't got a record of which person is in which bed, we cannot associate a person with a particular
        # set of appointments associated for in-patient bed-days (after the first). This could be accomplished by
        # changing the way BedDays stores internally the information about in-patients and creating HSI for the
        # in-patients.

    def on_end_of_year(self):
        self._summary_counter.write_to_log_and_reset_counters()


class BedDaysSummaryCounter:
    """Helper class to keep running counts of bed-days used."""

    def __init__(self):
        self._reset_internal_stores()
        self.dates = []

    def _reset_internal_stores(self) -> None:
        """Create empty versions of the data structures used to store a running records. The structure is
        {<bed_type>: <number_of_beddays>}."""

        self._bed_days_used = defaultdict(int)
        self._bed_days_available = defaultdict(int)

    def record_usage_of_beds(self, bed_days_used: Dict[str, Tuple[int, int]]) -> None:
        """Add record of usage of beds. `bed_days_used` is a dict of the form
        {<bed_type>: tuple(total_numbers_of_bed_available, total_number_available_now)}."""

        for _bed_type, (_total, _num_available) in bed_days_used.items():
            self._bed_days_used[_bed_type] += (_total - _num_available)
            self._bed_days_available[_bed_type] += _total

    def write_to_log_and_reset_counters(self):
        """Log summary statistics and reset the data structures."""

        logger_summary.info(
            key="BedDays",
            description="Counts of the bed-days that have been used (by type).",
            data=self._bed_days_used,
        )

        logger_summary.info(
            key="FractionOfBedDaysUsed",
            description="Fraction of the bed-days available in the last year that were used (by type).",
            data={
                _bed_type: self._bed_days_used[_bed_type] / _total
                for _bed_type, _total in self._bed_days_available.items()
            }
        )

        self._reset_internal_stores()
