"""
This is the Bed days class.

It maintains a current record of the availability and usage of beds in the healthcare system.

"""

import pandas as pd

from tlo import Property, Types, logging

# ---------------------------------------------------------------------------------------------------------
#   CLASS DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger('tlo.methods.healthsystem')


class BedDays:
    """
    The BedDays class. This is expected to be registered in the HealthSystem module.
    """

    def __init__(self, hs_module):
        self.hs_module = hs_module
        # Number of days to the last day of bed_tracker
        self.days_until_last_day_of_bed_tracker = 150

        # a dictionary to create a footprint according to facility bed days capacity
        self.available_footprint = {}

        # a dictionary to track inpatient bed days
        self.bed_tracker = dict()
        self.list_of_cols_with_internal_dates = dict()

        self.bed_types = list()  # List of bed-types

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
        """Put out to the log the information from the tracker of the last day of the simulation"""
        self.log_yesterday_info_from_all_bed_trackers()

    def initialise_beddays_tracker(self):
        """Initialise the bed days tracker:
        Create a dataframe for each type of beds that give the total number of beds currently available in each facility
         (rows) by the date during the simulation (columns).

        The current implementation assumes that bed capacity is held constant throughout the simulation; but it could be
         changed through modifications here.
        """

        capacity = self.hs_module.parameters['BedCapacity'].set_index('Facility_ID')
        max_number_of_bed_days = self.days_until_last_day_of_bed_tracker

        end_date = self.hs_module.sim.start_date + pd.DateOffset(days=max_number_of_bed_days)

        date_range = pd.date_range(self.hs_module.sim.start_date, end_date, freq='D')

        for bed_type in self.bed_types:
            df = pd.DataFrame(
                index=date_range,  # <- Days in the simulation
                columns=capacity.index,  # <- Facility_ID
                data=1
            )
            df = df.mul(capacity[bed_type], axis=1)
            assert not df.isna().any().any()
            self.bed_tracker[bed_type] = df

    def processing_at_start_of_new_day(self):
        """Things to do at the start of each new day:
        * Refresh inpatient status
        * Log yesterday's usage of beds
        * Move the tracker by one day
        """
        # Refresh the hs_in_patient status
        self.refresh_in_patient_status()

        # NB. This is skipped on the first day of the simulation as there is nothing to log from yesterday and the
        # tracker is already set.

        if self.hs_module.sim.date != self.hs_module.sim.start_date:
            self.log_yesterday_info_from_all_bed_trackers()
            self.move_each_tracker_by_one_day()

    def move_each_tracker_by_one_day(self):
        bed_capacity = self.hs_module.parameters['BedCapacity']

        for bed_type, tracker in self.bed_tracker.items():
            start_date = min(tracker.index)

            # reset all the columns for the earliest entry - it's going to become the new day
            tracker.loc[start_date] = bed_capacity.loc[bed_capacity.index[0], bed_type]

            # make new index
            end_date = max(tracker.index)  # get the latest day in the dataframe
            new_day = end_date + pd.DateOffset(days=1)  # the new day is the next day
            new_index = list(tracker.index)
            new_index[0] = new_day  # the earliest day is replaced with the next day
            new_index = pd.DatetimeIndex(new_index)

            # update the index
            tracker = tracker.set_index(new_index).sort_index()

            # save the updated tracker
            self.bed_tracker[bed_type] = tracker

    def log_yesterday_info_from_all_bed_trackers(self):
        """Dump yesterday's status of bed-day tracker to the log"""

        for bed_type, tracker in self.bed_tracker.items():
            occupancy_info = tracker.iloc[0].to_dict()
            occupancy_info.update({'date_of_bed_occupancy': tracker.index[0]})

            logger.info(
                key=f'bed_tracker_{bed_type}',
                data=occupancy_info,
                description=f'Use of bed_type {bed_type}, by day and facility'
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
                available.loc[available.idxmin(~available):] = False

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
