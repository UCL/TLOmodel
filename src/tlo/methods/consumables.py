import datetime
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger('tlo.methods.healthsystem')

# todo -- handling of unrecognised codes
# todo - issue warning if a consumable is requested from an HSI at level 0
# todo - what do we do about level 0

class Consumables:
    """This is the Consumables Class. It maintains a current record of the availability and usage of consumables in the
     HealthSystem. It is expected that this is instantiated by the `HealthSystem` module.

    :param: `cons_availability: Determines the availability of consumbales. If 'default' then use the availability
     specified in the ResourceFile; if 'none', then let no consumable be ever be available; if 'all', then all
     consumables are always available. When using 'all' or 'none', requests for consumables are not logged.

    If an item_code is requested that is not recognised, a response is returned that is based on the average
    availability of other consumables in that facility at that time, and a `UserWarning` is issued.
    """

    def __init__(self, data: pd.DataFrame = None, rng: np.random = None, cons_availabilty: str = 'default') -> None:

        assert cons_availabilty in ['none', 'default', 'all'], "Argument `cons_availability` is not recognised."  # todo spelling error
        self.cons_availability = cons_availabilty  # Governs availability  - none/default/all
        self.rng = rng

        self.item_codes = set()  # All item_codes that are recognised.
        self.prob_item_codes_available = None  # Data on the probability of each item_code being available
        self.cons_available_today = None  # Index for the item_codes available
        self._is_unknown_item_available = None  # Whether an unknown item is available, by facility_id

        if data is not None:
            self._process_consumables_df(data)

    def processing_at_start_of_new_day(self, date: datetime.datetime) -> None:
        """Do the jobs at the start of each new day.
        * Update the availability of the consumables
        """
        self._refresh_availability_of_consumables(date)

    def _process_consumables_df(self, df: pd.DataFrame) -> None:
        """Helper function for processing the consumables data, passed in here as pd.DataFrame that has been read-in by
        the HealthSystem.
        * Saves the data as `self.prob_item_codes_available`
        * Saves the set of all recognised item_codes to `self.item_codes`
        """
        self.item_codes = set(df.item_code)  # Record all consumables identified
        self.prob_item_codes_available = df.set_index(['month', 'facility_id', 'item_code'])['available_prop']

    def _refresh_availability_of_consumables(self, date: datetime.datetime):
        """Update the availability of all items based on the data for the probability of availability, givem the current
        date."""
        # Work out which items are available in which facilities for this date.
        month = date.month
        availability_this_month = self.prob_item_codes_available.loc[(month, slice(None), slice(None))]
        items_available_this_month = availability_this_month.index[
            availability_this_month.values > self.rng.rand(len(availability_this_month))
            ]

        # Convert to dict-of-sets to enable checking of item_code availability.
        self.is_available = defaultdict(set)
        for _fac_id, _item in items_available_this_month.to_list():
            self.is_available[_fac_id].add(_item)

        # Update the default return value (based on the average probability of availability of items at the facility)
        average_availability_of_items_by_facility_id = availability_this_month.groupby(level=0).mean()
        self._is_unknown_item_available = (average_availability_of_items_by_facility_id >
                                           self.rng.random_sample(len(average_availability_of_items_by_facility_id))
                                           ).to_dict()

    @ staticmethod
    def _determine_default_return_value(cons_availability, default_return_value):
        if cons_availability == 'all':
            return True
        elif cons_availability == 'none':
            return False
        else:
            return default_return_value

    def _request_consumables(self, facility_id: int, item_codes: dict, to_log: bool = True,
                             treatment_id: Optional[str] = None) -> dict:
        """
        This is a private function called by the 'get_consumables` in the `HSI_Event` base class. It queries whether
        item_codes are currently available for a particular `hsi_event` and logs the request.

        :param facility_id: The facility_id from which the request for consumables originates
        :param item_codes: dict of the form {<item_code>: <quantity>} for the items requested
        :param optional_item_codes: di
        :param to_log: whether the request is logged.
        :return:
        """
        # Issue warning if any item_code is not recognised.
        if not self.item_codes.issuperset(item_codes.keys()):
            for _i in item_codes.keys():
                if _i not in self.item_codes:
                    warnings.warn(UserWarning(f"Item_Code {_i} is not recognised."))

        # Determine availability of consumables:
        if self.cons_availability == 'all':
            # All item_codes available available if all consumables should be considered available by default.
            available = {k: True for k in item_codes}
        elif self.cons_availability == 'none':
            # All item_codes not available if consumables should be considered not available by default.
            available = {k: False for k in item_codes.keys()}
        else:
            available = self._lookup_availability_of_consumables(item_codes=item_codes, facility_id=facility_id)

        # Log the request and the outcome:
        if to_log:
            logger.info(key='Consumables',
                        data={
                            'TREATMENT_ID': (treatment_id if treatment_id is not None else ""),
                            'Item_Available': str({k: v for k, v in item_codes.items() if available[k]}),
                            'Item_NotAvailable': str({k: v for k, v in item_codes.items() if not available[k]}),
                        },
                        # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
                        description="Record of each consumable item that is requested."
                        )

        # Return the result of the check on availability
        return available

    def _lookup_availability_of_consumables(self, facility_id: int, item_codes: dict) -> dict:
        """Lookup whether a particular item_code is in the set of available items for that facility_id (in
        `self.is_available`). If any code is not recognised, use the `_is_unknown_item_available`."""
        avail = dict()
        for _i in item_codes.keys():
            if _i in self.item_codes:
                avail.update({_i: _i in self.is_available[facility_id]})
            else:
                avail.update({_i: self._is_unknown_item_available[facility_id]})
        return avail

    @staticmethod
    def _get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:
        """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
         a given package name."""
        ser = lookup_df.loc[
            lookup_df['Intervention_Pkg'] == package, ['Item_Code', 'Expected_Units_Per_Case']].set_index(
            'Item_Code')['Expected_Units_Per_Case'].apply(np.ceil).astype(int)
        return ser.groupby(ser.index).sum().to_dict()  # de-duplicate index before converting to dict

    @staticmethod
    def _get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
        """Helper function to provide the item_code (an int) when provided with the name of the item"""
        return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])



def create_dummy_data_for_cons_availability(intrinsic_availability: Dict[int, bool] = {0: False, 1: True},
                                            months: List[int] = [1],
                                            facility_ids: List[int] = [0]
                                            ) -> pd.DataFrame:
    """Returns a pd.DataFrame that is a dummy for the imported `ResourceFile_Consumables.csv`.
    By default, it describes the availability of two items, one of which is always available, and one of which is
    never available."""
    list_of_items = []
    for _item, _avail in intrinsic_availability.items():
            for _month in months:
                for _fac_id in facility_ids:
                    list_of_items.append({
                        'item_code': _item,
                        'month': _month,
                        'facility_id': _fac_id,
                        'available_prop': _avail
                    })
    return pd.DataFrame(data=list_of_items)
