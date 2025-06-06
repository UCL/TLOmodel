import warnings
from collections import defaultdict
from typing import Counter, Dict, Iterable, Literal, Set, Union

import numpy as np
import pandas as pd

from tlo import logging
from tlo.logging.helpers import get_dataframe_row_as_dict_for_logging

logger_summary = logging.getLogger("tlo.methods.healthsystem.summary")


class Equipment:
    """
    This is the equipment class. It maintains a current record of the availability of equipment in the health system. It
    is expected that this is instantiated by the :py:class:`~.HealthSystem` module.

    The basic paradigm is that an :py:class:`~.HSI_Event` can declare equipment that is required for delivering the
    healthcare service that the ``HSI_Event`` represents. The ``HSI_Event`` uses :py:meth:`HSI_event.add_equipment` to
    make these declarations, with reference to the items of equipment that are defined in
    ``ResourceFile_EquipmentCatalogue.csv``. (These declaration can be in the form of the descriptor or the equipment
    item code). These declarations can be used when the ``HSI_Event`` is created but before it is run (in ``__init__``),
    or during execution of the ``HSI_Event`` (in :py:meth:`.HSI_Event.apply`).

    As the ``HSI_Event`` can declare equipment that is required before it is run, the HealthSystem *can* use this to
    prevent an ``HSI_Event`` running if the equipment declared is not available. Note that for equipment that is
    declared whilst the ``HSI_Event`` is running, there are no checks on availability, and the ``HSI_Event`` is allowed
    to continue running even if equipment is declared is not available. For this reason, the ``HSI_Event`` should
    declare equipment that is *essential* for the healthcare service in its ``__init__`` method. If the logic inside the
    ``apply`` method of the ``HSI_Event`` depends on the availability of equipment, then it can find the probability
    with which item(s) will be available using :py:meth:`.HSI_Event.probability_equipment_available`.

    The data on the availability of equipment data refers to the proportion of facilities in a district of a particular
    level (i.e., the ``Facility_ID``) that do have that piece of equipment. In the model, we do not know which actual
    facility the person is attending (there are many actual facilities grouped together into one ``Facility_ID`` in the
    model). Therefore, the determination of whether equipment is available is made probabilistically for the
    ``HSI_Event`` (i.e., the probability that the actual facility being attended by the person has the equipment is
    represented by the proportion of such facilities that do have that equipment). It is assumed that the probabilities
    of each item being available are independent of one other (so that the probability of all items being available is
    the product of the probabilities for each item). This probabilistic determination of availability is only done
    _once_ for the ``HSI_Event``: i.e., if the equipment is determined to not be available for the instance of the
    ``HSI_Event``, then it will remain not available if the same event is re-scheduled / re-entered into the
    ``HealthSystem`` queue. This represents that if the facility that a particular person attends for the ``HSI_Event``
    does not have the equipment available, then it will still not be available on another day.

    Where data on availability is not provided for an item, the probability of availability is inferred from the average
    availability of other items in that facility ID. Likewise, the probability of an item being available at a facility
    ID is inferred from the average availability of that item at other facilities. If an item code is referred in
    ``add_equipment`` that is not recognised (not included in :py:attr:`catalogue`), a :py:exc:`UserWarning` is issued,
    but that item is then silently ignored. If a facility ID is ever referred that is not recognised (not included in
    :py:attr:`master_facilities_list`), an :py:exc:`AssertionError` is raised.

    :param catalogue: The database of all recognised item_codes.
    :param data_availability: Specifies the probability with which each equipment (identified by an ``item_code``) is
        available at a facility level. Note that information must be provided for every item in the :py:attr`catalogue`
        and every facility ID in the :py:attr`master_facilities_list`.
    :param: rng: The random number generator object to use for random numbers.
    :param availability: Determines the mode availability of the equipment. If 'default' then use the availability
        specified in :py:attr:`data_availability`; if 'none', then let no equipment be ever be available; if 'all', then
        all equipment is always available.
    :param master_facilities_list: The :py:class:`~pandas.DataFrame` with the line-list of all the facilities in the
        health system.
    """

    def __init__(
        self,
        catalogue: pd.DataFrame,
        data_availability: pd.DataFrame,
        rng: np.random.RandomState,
        master_facilities_list: pd.DataFrame,
        availability: Literal["all", "default", "none"] = "default",
    ) -> None:
        # - Store arguments
        self.catalogue = catalogue
        self.rng = rng
        self.data_availability = data_availability
        self.availability = availability
        self.master_facilities_list = master_facilities_list

        # - Data structures for quick look-ups for items and descriptors
        self._item_code_lookup = self.catalogue.set_index('Item_Description')['Item_Code'].to_dict()
        self._pkg_lookup = self._create_pkg_lookup()
        self._all_item_descriptors = set(self._item_code_lookup.keys())
        self._all_item_codes = set(self._item_code_lookup.values())
        self._all_fac_ids = self.master_facilities_list['Facility_ID'].unique()

        # - Probabilities of items being available at each facility_id
        self._probabilities_of_items_available = self._get_equipment_availability_probabilities()

        # - Internal store of which items have been used at each facility_id This is of the form
        # {facility_id: {item_code: count}}.
        self._record_of_equipment_used_by_facility_id = defaultdict(Counter)

    def on_simulation_end(self):
        """Things to do when the simulation ends:
         * Log (to the summary logger) the equipment that has been used.
        """
        self.write_to_log()

    @property
    def availability(self):
        return self._availability

    @availability.setter
    def availability(self, value: Literal["all", "default", "none"]):
        assert value in {"all", "none", "default"}, f"New availability value {value} not recognised."
        self._availability = value

    def _get_equipment_availability_probabilities(self) -> pd.Series:
        """
        Extract the probabilities that each equipment item is available (at a given
        facility), for use when the equipment availability is set to "default".

        The probabilities extracted in this method are constant throughout the simulation,
        however they will not be used when the equipment availability is "all" or "none".
        Extracting them once and storing the result allows us to avoid repeating this
        calculation if the equipment availability change event occurs during the simulation.
        """
        dat = self.data_availability.set_index(
            [
                self.data_availability["Facility_ID"].astype(np.int64),
                self.data_availability["Item_Code"].astype(np.int64),
            ]
        )["Pr_Available"]

        # Confirm that there is an estimate for every item_code at every facility_id
        full_index = pd.MultiIndex.from_product(
                [self._all_fac_ids, self._all_item_codes], names=["Facility_ID", "Item_Code"]
        )
        pd.testing.assert_index_equal(full_index, dat.index, check_order=False)
        assert not dat.isnull().any()

        return dat

    def parse_items(self, items: Union[int, str, Iterable[int], Iterable[str]]) -> Set[int]:
        """Parse equipment items specified as an item_code (integer), an item descriptor (string), or an iterable of
         item_codes or descriptors (but not a mix of the two), and return as a set of item_code (integers). For any
         item_code/descriptor not recognised, a ``UserWarning`` is issued."""

        def check_item_codes_recognised(item_codes: set[int]):
            if not item_codes.issubset(self._all_item_codes):
                warnings.warn(f'At least one item code was unrecognised: "{item_codes}".')

        def check_item_descriptors_recognised(item_descriptors: set[str]):
            if not item_descriptors.issubset(self._all_item_descriptors):
                warnings.warn(f'At least one item descriptor was unrecognised "{item_descriptors}".')

        # Make into a set if it is not one already
        if isinstance(items, (str, int)):
            items = set([items])
        else:
            items = set(items)

        items_are_ints = all(isinstance(element, int) for element in items)

        if items_are_ints:
            check_item_codes_recognised(items)
            # In the return, any unrecognised item_codes are silently ignored.
            return items.intersection(self._all_item_codes)
        else:
            check_item_descriptors_recognised(items)  # Warn for any unrecognised descriptors
            # In the return, any unrecognised descriptors are silently ignored.
            return set(self._item_code_lookup[i] for i in items if i in self._item_code_lookup)

    def probability_all_equipment_available(
        self, facility_id: int, item_codes: Set[int]
    ) -> float:
        """
        Returns the probability that all the equipment item_codes are available
        at the given facility.

        It does so by looking at the probabilities of each equipment item being
        available and multiplying these together to find the probability that *all*
        are available.

        NOTE: This will error if the facility ID or any of the item codes is not recognised.

        :param facility_id: Facility at which to check for the equipment.
        :param item_codes: Integer item codes corresponding to the equipment to check.
        """

        assert facility_id in self._all_fac_ids, f"Unrecognised facility ID: {facility_id=}"
        assert item_codes.issubset(self._all_item_codes), f"At least one item code was unrecognised: {item_codes=}"

        if self.availability == "all":
            return 1.0
        elif self.availability == "none":
            return 0.0
        return self._probabilities_of_items_available.loc[
            (facility_id, list(item_codes))
        ].prod()

    def is_all_items_available(
        self, item_codes: Set[int], facility_id: int
    ) -> bool:
        """
        Determine if all equipment items are available at the given facility_id.
        Returns True only if all items are available at the facility_id,
        otherwise returns False.
        """
        if item_codes:
            return self.rng.random_sample() < self.probability_all_equipment_available(
                facility_id=facility_id,
                item_codes=item_codes,
            )
        else:
            # In the case of an empty set, default to True without doing anything else ('no equipment' is always
            # "available"). This is the most common case, so optimising for speed.
            return True

    def record_use_of_equipment(
        self, item_codes: Set[int], facility_id: int
    ) -> None:
        """Update internal record of the usage of items at equipment at the specified facility_id."""
        self._record_of_equipment_used_by_facility_id[facility_id].update(item_codes)

    def write_to_log(self) -> None:
        """Write to the log:
         * Summary of the equipment that was _ever_ used at each district/facility level.
         Note that the info-level health system logger (key: `hsi_event_counts`) contains logging of the equipment used
         in each HSI event (if finer splits are needed). Alternatively, different aggregations could be created here for
         the summary logger, using the same pattern as used here.
        """

        mfl = self.master_facilities_list

        def sorted_keys_or_empty_list(x: Union[dict, None]) -> list:
            if isinstance(x, dict):
                return sorted(x.keys())
            else:
                return []

        set_of_equipment_ever_used_at_each_facility_id = pd.Series({
            fac_id: sorted_keys_or_empty_list(
                self._record_of_equipment_used_by_facility_id.get(fac_id)
            )
            for fac_id in mfl['Facility_ID']
        }, name='EquipmentEverUsed').astype(str)

        output = mfl.merge(
            set_of_equipment_ever_used_at_each_facility_id,
            left_on='Facility_ID',
            right_index=True,
            how='left',
        ).drop(columns=['Facility_ID', 'Facility_Name'])
        # Log multi-row data-frame
        for row_index in output.index:
            logger_summary.info(
                key='EquipmentEverUsed_ByFacilityID',
                description='For each facility_id (the set of facilities of the same level in a district), the set of'
                            'equipment items that are ever used.',
                data=get_dataframe_row_as_dict_for_logging(output, row_index)
            )

    def from_pkg_names(self, pkg_names: Union[str, Iterable[str]]) -> Set[int]:
        """Convenience function to find the set of item_codes that are grouped under requested package name(s) in the
        catalogue."""
        # Make into a set if it is not one already
        if isinstance(pkg_names, (str, int)):
            pkg_names = set([pkg_names])
        else:
            pkg_names = set(pkg_names)

        item_codes = set()
        for pkg_name in pkg_names:
            if pkg_name in self._pkg_lookup.keys():
                item_codes.update(self._pkg_lookup[pkg_name])
            else:
                raise ValueError(f'That Pkg_Name is not in the catalogue: {pkg_name=}')

        return item_codes

    def _create_pkg_lookup(self) -> Dict[str, Set[int]]:
        """Create a lookup from a Package Name to a set of Item_Codes that are contained with that package.
        N.B. In the Catalogue, there is one row for each Item, and the Packages to which each Item belongs (if any)
        is given in a column 'Pkg_Name': if an item belongs to multiple packages, these names are separated by commas,
        and if it doesn't belong to any package, then there is a NULL value."""
        df = self.catalogue

        # Make dataframe with columns for each package, and bools showing whether each item_code is included
        pkgs = df['Pkg_Name'].replace({float('nan'): None}) \
                             .str.get_dummies(sep=',') \
                             .set_index(df.Item_Code) \
                             .astype(bool)

        # Make dict of the form: {'Pkg_Code': <Set of item_codes>}
        pkg_lookup_dict = {
            pkg_name.strip(): set(pkgs[pkg_name].loc[pkgs[pkg_name]].index.to_list())
            for pkg_name in pkgs.columns
        }

        return pkg_lookup_dict
