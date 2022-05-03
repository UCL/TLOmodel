import datetime
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger('tlo.methods.healthsystem')
logger_summary = logging.getLogger('tlo.methods.healthsystem.summary')


class Consumables:
    """This is the Consumables Class. It maintains a current record of the availability of consumables in the
     HealthSystem. It is expected that this is instantiated by the `HealthSystem` module.

    :param: `data`: Specifies the probability with which each consumable (identified by an `item_code`) is available
     in a particular month at a particular Facility_ID.

    :param: `rng`: The Random Number Generator object to use for random numbers.

    :param: `availability`: Determines the availability of consumables. If 'default' then use the availability
     specified in the ResourceFile; if 'none', then let no consumable be ever be available; if 'all', then all
     consumables are always available. When using 'all' or 'none', requests for consumables are not logged.

    If an item_code is requested that is not recognised (not included in `data`), a `UserWarning` is issued, and the
     result returned is on the basis of the average availability of other consumables in that facility in that month.
    """

    def __init__(self, data: pd.DataFrame = None, rng: np.random = None, availability: str = 'default') -> None:

        assert availability in ['none', 'default', 'all'], "Argument `availability` is not recognised."
        self.cons_availability = availability  # Governs availability  - none/default/all
        self.item_codes = set()  # All item_codes that are recognised.

        self._rng = rng
        self._prob_item_codes_available = None  # Data on the probability of each item_code being available
        self._cons_available_today = None  # Index for the item_codes available
        self._is_available = None  # Dict of sets giving the set of item_codes available, by facility_id
        self._is_unknown_item_available = None  # Whether an unknown item is available, by facility_id
        self._not_recognised_item_codes = set()  # The item codes requested but which are not recognised.

        self._process_consumables_df(data)

        # Create pointer to the `HealthSystemSummaryCounter` helper class
        self._summary_counter = ConsumablesSummaryCounter()

    def processing_at_start_of_new_day(self, date: datetime.datetime) -> None:
        """Do the jobs at the start of each new day.
        * Update the availability of the consumables
        """
        self._refresh_availability_of_consumables(date)

    def _process_consumables_df(self, df: pd.DataFrame) -> None:
        """Helper function for processing the consumables data, passed in here as pd.DataFrame that has been read-in by
        the HealthSystem.
        * Saves the data as `self._prob_item_codes_available`
        * Saves the set of all recognised item_codes to `self.item_codes`
        """
        self.item_codes = set(df.item_code)  # Record all consumables identified
        self._prob_item_codes_available = df.set_index(['month', 'Facility_ID', 'item_code'])['available_prop']

    def _refresh_availability_of_consumables(self, date: datetime.datetime):
        """Update the availability of all items based on the data for the probability of availability, givem the current
        date."""
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

    def override_availability(self, item_codes: dict = None) -> None:
        """
        Over-ride the availability (for all months and all facilities) of certain item_codes.
        Note this should not be called directly: Disease modules should call `override_availability_of_consumables` in
         `HealthSystem`.
        :param item_codes: Dictionary of the form {<item_code>: probability_that_item_is_available}
        :return: None
        """

        def check_item_codes_argument_is_valid(_item_codes):
            assert set(_item_codes.keys()).issubset(self.item_codes), 'Some item_codes not recognised.'
            assert all([0.0 <= x <= 1.0 for x in list(_item_codes.values())]), 'Probability of availability must be ' \
                                                                               'between 0.0 and 1.0'

        check_item_codes_argument_is_valid(item_codes)

        # Update the internally-held data on availability for these item_codes (for all months and at all facilities)
        for item, prob in item_codes.items():
            self._prob_item_codes_available.loc[(slice(None), slice(None), item)] = prob

    @staticmethod
    def _determine_default_return_value(cons_availability, default_return_value):
        if cons_availability == 'all':
            return True
        elif cons_availability == 'none':
            return False
        else:
            return default_return_value

    def _request_consumables(self,
                             facility_info: 'FacilityInfo',  # noqa: F821
                             item_codes: dict,
                             to_log: bool = True,
                             treatment_id: Optional[str] = None
                             ) -> dict:
        """This is a private function called by 'get_consumables` in the `HSI_Event` base class. It queries whether
        item_codes are currently available at a particular Facility_ID and logs the request.

        :param facility_info: The facility_info from which the request for consumables originates
        :param item_codes: dict of the form {<item_code>: <quantity>} for the items requested
        :param to_log: whether the request is logged.
        :param treatment_id: the TREATMENT_ID of the HSI (which is entered to the log, if provided).
        :return: dict of the form {<item_code>: <bool>} indicating the availability of each item requested.
        """

        # Issue warning if any item_code is not recognised.
        if not self.item_codes.issuperset(item_codes.keys()):
            self._not_recognised_item_codes.add((treatment_id, tuple(set(item_codes.keys()) - self.item_codes)))

        # Determine availability of consumables:
        if self.cons_availability == 'all':
            # All item_codes available available if all consumables should be considered available by default.
            available = {k: True for k in item_codes}
        elif self.cons_availability == 'none':
            # All item_codes not available if consumables should be considered not available by default.
            available = {k: False for k in item_codes.keys()}
        else:
            available = self._lookup_availability_of_consumables(item_codes=item_codes,
                                                                 facility_info=facility_info)

        # Log the request and the outcome:
        if to_log:
            items_available = {k: v for k, v in item_codes.items() if available[k]}
            items_not_available = {k: v for k, v in item_codes.items() if not available[k]}
            logger.info(key='Consumables',
                        data={
                            'TREATMENT_ID': (treatment_id if treatment_id is not None else ""),
                            'Item_Available': str(items_available),
                            'Item_NotAvailable': str(items_not_available),
                        },
                        # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
                        description="Record of each consumable item that is requested."
                        )

            self._summary_counter.record_availability(items_available=items_available,
                                                      items_not_available=items_not_available)

        # Return the result of the check on availability
        return available

    def _lookup_availability_of_consumables(self,
                                            facility_info: 'FacilityInfo',   # noqa: F821
                                            item_codes: dict
                                            ) -> dict:
        """Lookup whether a particular item_code is in the set of available items for that facility (in
        `self._is_available`). If any code is not recognised, use the `_is_unknown_item_available`."""
        avail = dict()

        if facility_info is None:
            # If `facility_info` is None, it implies that the HSI has not been initialised because the HealthSystem
            #  is running with `disable=True`.
            return {_i: True for _i in item_codes}

        for _i in item_codes.keys():
            if _i in self.item_codes:
                avail.update({_i: _i in self._is_available[facility_info.id]})
            else:
                avail.update({_i: self._is_unknown_item_available[facility_info.id]})
        return avail

    def on_simulation_end(self):
        """Do tasks at the end of the simulation: Raise warnings and enter to log about item_codes not recognised."""
        if self._not_recognised_item_codes:
            warnings.warn(UserWarning(f"Item_Codes were not recognised./n"
                                      f"{self._not_recognised_item_codes}"))
            for _treatment_id, _item_codes in self._not_recognised_item_codes:
                logger.info(
                    key="item_codes_not_recognised",
                    data={_treatment_id if _treatment_id is not None else "": list(_item_codes)}
                )

    def write_to_log_and_reset_counters(self):
        return self._summary_counter.write_to_log_and_reset_counters()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:
    """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
     a given package name."""
    ser = lookup_df.loc[
        lookup_df['Intervention_Pkg'] == package, ['Item_Code', 'Expected_Units_Per_Case']].set_index(
        'Item_Code')['Expected_Units_Per_Case'].apply(np.ceil).astype(int)
    return ser.groupby(ser.index).sum().to_dict()  # de-duplicate index before converting to dict


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])


def create_dummy_data_for_cons_availability(intrinsic_availability: Optional[Dict[int, float]] = None,
                                            months: Optional[List[int]] = None,
                                            facility_ids: Optional[List[int]] = None,
                                            ) -> pd.DataFrame:
    """Returns a pd.DataFrame that is a dummy for the imported `ResourceFile_Consumables.csv`.
    By default, it describes the availability of two items, one of which is always available, and one of which is
    never available."""

    if intrinsic_availability is None:
        intrinsic_availability = {0: False, 1: True}

    if months is None:
        months = [1]

    if facility_ids is None:
        facility_ids = [0]

    list_of_items = []
    for _item, _avail in intrinsic_availability.items():
        for _month in months:
            for _fac_id in facility_ids:
                list_of_items.append({
                    'item_code': _item,
                    'month': _month,
                    'Facility_ID': _fac_id,
                    'available_prop': _avail
                })
    return pd.DataFrame(data=list_of_items)


def check_format_of_consumables_file(df, fac_ids):
    """Check that we have a complete set of estimates, for every region & facility_type, as defined in the model."""
    months = set(range(1, 13))
    item_codes = set(df.item_code.unique())

    assert set(df.columns) == {'Facility_ID', 'month', 'item_code', 'available_prop'}

    # Check that all permutations of Facility_ID, month and item_code are present
    pd.testing.assert_index_equal(
        df.set_index(['Facility_ID', 'month', 'item_code']).index,
        pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code']),
        check_order=False
    )

    # Check that every entry for a probability is a float on [0,1]
    assert (df.available_prop <= 1.0).all() and (df.available_prop >= 0.0).all()
    assert not pd.isnull(df.available_prop).any()


class ConsumablesSummaryCounter:
    """Helper class to keep running counts of consumable."""

    def __init__(self):
        self._reset_internal_stores()

    def _reset_internal_stores(self) -> None:
        """Create empty versions of the data structures used to store a running records."""

        self._items = {
            'Available': defaultdict(int),
            'NotAvailable': defaultdict(int)
        }

    def record_availability(self, items_available: dict, items_not_available: dict) -> None:
        """Add information about the availability of requested items to the running summaries."""

        # Record items that were available
        for _item, _num in items_available.items():
            self._items['Available'][_item] += _num

        # Record items that were not available
        for _item, _num in items_not_available.items():
            self._items['NotAvailable'][_item] += _num

    def write_to_log_and_reset_counters(self):
        """Log summary statistics reset the data structures."""

        logger_summary.info(
            key="Consumables",
            description="Counts of the items that were requested in this calendar year, which were available and"
                        "not available.",
            data={
                "Item_Available": self._items['Available'],
                "Item_NotAvailable": self._items['NotAvailable'],
            },
        )

        self._reset_internal_stores()
