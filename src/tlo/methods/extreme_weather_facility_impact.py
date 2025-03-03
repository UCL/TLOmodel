## Need to determine the effect of the EWE









# def switch_service_availability(
#     self,
#     new_availability: Literal["all", "none", "default", "projected"], # projected is disruptions from CC
#     effective_on_and_from: Date,
#     effective_to_and_until: Date,
#     services_affected: Literal["ANC"],
#     model_to_data_popsize_ratio: float = 1.0,
# ) -> None:
#     """
#     Action to be taken if the service availability changes in the middle
#     of the simulation due to extreme weather events.
#
#     If service capacities are reduced below the currently scheduled occupancy.
#     Inpatients are not evicted from beds and are allowed to remain in the
#     bed until they are scheduled to leave. Obviously, no new patients will
#     be admitted if there is no room in the new capacities.
#
#     :param new_availability: The new service availability. See __init__ for details.
#     :param effective_on_and_from: First day from which the new service availabilities will be imposed.
#     :param effective_to_and_until: Last day on which the new service availabilities will be imposed.
#     :param services_affected: Which service types are affected by the disruptions.
#     :param model_to_data_popsize_ratio: As in initialise_population.
#     """
#     # Store new service availability
#     self.availability = new_availability
#     # Before we update the service capacity, we need to store its old values
#     # This is because we will need to update the trackers to reflect the new#
#     # maximum capacities for each bed type.
#     old_max_capacities: pd.DataFrame = self._scaled_capacity.copy()
#     # Set the new capacity for beds
#     self.set_scaled_capacity(model_to_data_popsize_ratio)
#     # Compute the difference between the new max capacities and the old max capacities
#     difference_in_max = self._scaled_capacity - old_max_capacities
#     # For each tracker, after the effective date, impose the difference on the max
#     # number of beds
#     for bed_type, tracker in self.bed_tracker.items():
#         tracker.loc[effective_on_and_from:] += difference_in_max[bed_type]

class Extreme_Weather_Events:
    def __init__(self,
                 service_availability_data: pd.DataFrame = None,
                 rng: np.random = None,
                 availability: str = 'default'
                 ) -> None:

        self._options_for_availability = {
            'none',
            'default',
            'all',
            'ANC', # can add more with further research
        }

    # Create internal items:
        self._rng = rng
        self._availability = None  # Internal storage of availability assumption (only accessed through getter/setter)
        self._prob_service_codes_available = None  # Data on the probability of each service_code being available
        self._is_available = None  # Dict of sets giving the set of services available, by facility_id

    def availability(self):
        """Returns the internally stored value for the assumption of availability of services."""
        return self._availability

    @availability.setter
    def availability(self, value: str):
        """Changes the effective availability of consumables and updates the internally stored value for that
        assumption.
        Note that this overrides any changes effected by `override_availability()`.
        """
        assert value in self._options_for_availability, f"Argument `cons_availability` is not recognised: {value}."
        self._availability = value
        self._update_prob_item_codes_available(self._availability)
    def _refresh_availability_of_services(self, date: datetime.datetime):
        """Update the availability of all services based on the data for the probability of availability, given the current
        date and the projected climate."""
        # Work out which items are available in which facilities for this date.
        month = date.month
        availability_this_month = self._prob_item_codes_available.loc[(month, slice(None), slice(None))]
        items_available_this_month = availability_this_month.index[
            availability_this_month.values > self._rng.random_sample(len(availability_this_month))
            ]

        # Convert to dict-of-sets to enable checking of item_code availability.
        self._is_available = defaultdict(set)
        for _fac_id, _item in items_available_this_month.to_list():
            self._is_available[_fac_id].add(_item)

        # Update the default return value (based on the average probability of availability of items at the facility)
        average_availability_of_items_by_facility_id = availability_this_month.groupby(level=0).mean()
        self._is_unknown_item_available = (average_availability_of_items_by_facility_id >
                                           self._rng.random_sample(len(average_availability_of_items_by_facility_id))
                                           ).to_dict()
