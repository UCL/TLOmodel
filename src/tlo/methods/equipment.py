import warnings
from collections import defaultdict
from typing import Dict, Set, Iterable, Union, Counter, Optional, Any

import numpy as np
import pandas as pd

from tlo import logging

logger_summary = logging.getLogger("tlo.methods.healthsystem.summary")


class Equipment:
    """This is the Equipment Class. It maintains a current record of the availability of equipment in the
     HealthSystem. It is expected that this is instantiated by the `HealthSystem` module.

    :param: 'catalogue': The database of all recognised item_codes.

    :param: `data_availability`: Specifies the probability with which each equipment (identified by an `item_code`) is
     available at a facility level. Note that information is not necessarily provided for every item in the `catalogue`.

    :param: `rng`: The Random Number Generator object to use for random numbers.

    :param: `availability`: Determines the mode availability of the equipment. If 'default' then use the availability
     specified in the ResourceFile; if 'none', then let no equipment be ever be available; if 'all', then all
     equipment is always available.

    If an item_code is requested that is not recognised (not included in `data`), a `UserWarning` is issued, and the
     result returned is on the basis of the average availability of other consumables in that facility in that month.
    """

    def __init__(
        self,
        catalogue: pd.DataFrame,
        data_availability: pd.DataFrame,
        rng: np.random,
        master_facilities_list: pd.DataFrame,
        availability: Optional[str] = "default",
    ) -> None:
        # Store arguments
        self.catalogue = catalogue
        self.rng = rng
        self.data_availability = data_availability
        self.master_facilities_list = master_facilities_list

        # Create internal storage structures
        self._items_available: Dict = dict()  # Will be the internal store of which items are available at each
        #                                       facility_id. This is of the form {facility_id: {items_available}}.

        self._record_of_equipment_used = defaultdict(Counter)  # Will be the internal store of which items have been
        #                                                        used at each facility_id. This is of the form
        #                                                        {facility_id: {item_code: count}}.

        # Set up the internal stores of equipment items that are available, ready for calls.
        self._set_equipment_items_available(availability=availability)

        # Set up internal lookup for item_descriptor -> item_code
        self.item_code_lookup = self.catalogue.set_index('Description')['Item_Code'].to_dict()

    def on_simulation_end(self):
        """Things to do when the simulation end:
         * Log (to the summary logger) the equipment that has been used.
        """
        self.write_to_log()

    def update_availability(self, availability) -> None:
        """Update the availability of equipment. This is expected to be called midway through the simulation if
        the assumption of the equipment availability needs to change."""
        self._set_equipment_items_available(availability=availability)

    def _set_equipment_items_available(self, availability: str):
        """Update internal store of which items of equipment are available. This is called at the beginning of the
        simulation and whenever an update in `availability` is needed."""

        # For any facility_id in the data
        all_fac_ids = self.master_facilities_list['Facility_ID'].unique()

        # All equipment items in the catalogue
        all_eq_items = self.catalogue["Item_Code"].unique()

        # Create full dataset, where we force that there is probability of availability for every item_code at every
        # observed facility
        df = pd.Series(
            index=pd.MultiIndex.from_product(
                [all_fac_ids, all_eq_items], names=["Facility_ID", "Item_Code"]
            ),
            data=float("nan"),
        ).combine_first(
            self.data_availability.set_index(["Facility_ID", "Item_Code"])[
                "Pr_Available"
            ]
        )

        # Merge in original dataset and use the mean in that facility_id to impute availability of missing item_code
        df = df.groupby("Facility_ID").transform(lambda x: x.fillna(x.mean()))
        # ... and also impute availability for any facility_ids for which no data, based on all other facilities
        df = df.groupby("Item_Code").transform(lambda x: x.fillna(x.mean()))

        # Check no missing values
        assert not df.isnull().any()

        # Over-write these data if `availability` argument specifies that `none` or `all` items should be available
        if availability == "default":
            pass
        elif availability == "all":
            df = (df + 1).clip(upper=1.0)
        elif availability == "none":
            df = df.mul(0.0)
        else:
            raise KeyError(f"Unknown equipment availability specified: {availability}")

        # Sample these probability to find which items are actually available
        is_available = df > self.rng.random(size=len(df))

        # Organise into dict of set, of the form: {facility_id: {items_available}} for known facility_ids
        # (N.B. Has to be done this way around in order to guarantee that we have each known facility_id in the keys
        #  even if there are no item available.)
        self._items_available: Dict = is_available.groupby("Facility_ID").agg(
            lambda x: set(x[x].index.get_level_values("Item_Code"))
        ).to_dict()

    def _parse_items(self, items: Union[int, str, Iterable[int | str]]) -> Set[int]:
        """Parse equipment items specified as an item_code (integer), an item descriptor (string), or an iterable of
         either, and return as a set of item_code (integers)."""

        def first_element_in_iterable(it: Iterable[Any]) -> Any | None:
            for el in it:
                return el

        if isinstance(items, str):
            # Single descriptor
            return set([self.item_code_lookup(items)])
        elif isinstance(items, int):
            # Single item_code provided
            return set([items])
        elif isinstance(items, Iterable) and len(items) == 0:
            # Iterable of length 0
            return set()
        elif isinstance(items, Iterable) and isinstance(first_element_in_iterable(items), str):
            # Iterable of descriptors
            return set(map(self.item_code_lookup, items))
        elif isinstance(items, Iterable) and isinstance(first_element_in_iterable(items), int):
            # Iterable of item_cods
            return set(items)

        else:
            raise ValueError(f'Item_Code format not recognised: {items=}')

        if isinstance(items, int):
            item_codes = set([items])  # If single int provided, place it into a list
        elif isinstance(items, list):
            item_codes = set(items)


    def is_all_items_available(
        self, item_codes: Union[int, str, Iterable[int | str]], facility_id: int
    ) -> bool:
        """Determine if all equipments are available at the given facility_id (or from the default if the faciluty_id
        is not recognised). Returns True only if all items are available at the facility_id, otherwise returns False."""
        return self._parse_items(item_codes).issubset(self._items_available[facility_id])


    def record_use_of_equipment(
        self, item_codes: Union[int, str, Iterable[int | str]], facility_id: int
    ) -> None:
        """Update internal record of the usage of items at equipment at the specified facility_id."""
        self._record_of_equipment_used[facility_id].update(self._parse_items(item_codes))

    def write_to_log(self) -> None:
        """Write to the log:
         * Summary of the equipment that was _ever_ used at each facility_level
         * For each facility_id, a set of the equipment items ever used.
        """

        mfl = self.master_facilities_list

        def set_of_keys_or_empty_set(x: Union[set, dict]):
            if isinstance(x, set):
                return x
            elif isinstance(x, dict):
                return set(x.keys())
            else:
                return None

        set_of_equipment_ever_used_at_each_facility_id = pd.Series({
            fac_id: set_of_keys_or_empty_set(self._record_of_equipment_used.get(fac_id, set()))
            for fac_id in mfl['Facility_ID']
        }, name='EquipmentEverUsed').astype(str)

        output = mfl.merge(
            set_of_equipment_ever_used_at_each_facility_id,
            left_on='Facility_ID',
            right_index=True,
            how='left',
        ).drop(columns=['Facility_ID', 'Facility_Name'])

        # Log multi-row data-frame
        for _, row in output.iterrows():
            logger_summary.info(
                key='EquipmentEverUsed_ByFacilityID',
                description='For each facility_id (the set of facilities of the same level in a district), the set'
                            'equipment items that are ever used.',
                data=row.to_dict(),
            )
