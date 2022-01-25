import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger('tlo.methods.healthsystem')

# todo -- handling of unrecognised codes


class Consumables:
    """This is the Consumables Class. It maintains a current record of the availability and usage of consumables in the
     HealthSystem. It is expected that this is instantiated by the `HealthSystem` module.

    :param: `cons_availability: Determines the availability of consumbales. If 'default' then use the availability
     specified in the ResourceFile; if 'none', then let no consumable be ever be available; if 'all', then all
     consumables are always available. When using 'all' or 'none', requests for consumables are not logged.

    :param: `if_unrecognised`: Determines behaviour if an item_code is requested that is not recognised (contained in
    the data held on availability). If 'available` then such items are deemed available; if 'not_available`, then such
    items are deemed not available; if 'average' then such items are deemed available with a probability equal to the
    average availability of other items (given the same other factor: district, facility_level etc); if 'error' than
    the simulation is terminated with an AssertionError if a request for such an item is made.
    todo - maybe Simplify just to average and error; pipe this through to healthsystem
    """

    def __init__(self, hs_module, cons_availabilty: str, if_unrecognised: str = None) -> None:
        self.hs_module = hs_module

        assert cons_availabilty in ['none', 'default', 'all'], "Argument `cons_availability` not recognised."
        self.cons_availability = cons_availabilty

        assert if_unrecognised in [None, 'average', 'error'], "Argument `if_unrecognised` not recognised."
        self.if_unrecognised = if_unrecognised if if_unrecognised is not None else 'average'

        self.item_codes = set()  # All item_codes that are recognised.
        self.prob_item_codes_available = None  # Data on the probability of each item_code being available
        self.cons_available_today = None  # Index for the item_codes available

        self.default_return_value = None

    def process_consumables_df(self, df: pd.DataFrame) -> None:
        """Helper function for processing the consumables data, passed in here as pd.DataFrame that has been read-in by
        the HealthSystem.
        * Saves the data as `self.prob_item_codes_available`
        * Saves the set of all recognised item_codes to `self.item_codes`
        """
        _df = df
        _df['Facility_Level'] = _df['fac_type_tlo'].astype(
            pd.CategoricalDtype(
                categories=self.hs_module.sim.modules['HealthSystem']._facility_levels))
        _df['District'] = _df['district'].astype(
            pd.CategoricalDtype(
                categories=self.hs_module.sim.modules['Demography'].PROPERTIES['district_of_residence'].categories))
        self.prob_item_codes_available = _df.set_index(
            ['item_code', 'District', 'Facility_Level', 'month'])['available_prop']
        self.item_codes = set(df.item_code)
        # index on facility_id rather than district and facility_level

    def processing_at_start_of_new_day(self) -> None:
        """Do the jobs at the start of each new day.
        * Update the availability of the consumables
        """
        self._refresh_availability_of_consumables()

    def _refresh_availability_of_consumables(self):
        """Update the availability of all items based on the data for the probability of availability."""
        # Random draws for the availability of each consumable (assuming independence betweeen district,
        # facility_levels, and item_codes).
        random_draws = self.hs_module.rng.rand(len(self.prob_item_codes_available))

        # Save the index for the item_codes that are available currently.
        self.cons_available_today = self.prob_item_codes_available.index[self.prob_item_codes_available > random_draws]
        # todo don't use month here - just use the current month when determining availability.
        # todo don't make this a long pd.Index: have a set that is specific to a facility_id

        # todo Compute the availability of a consumable requested that is not recognised.
        self.default_return_value = True

    def _request_consumables(self, hsi_event, item_codes: dict, to_log: bool) -> dict:
        """
        This is a private function called by the 'get_consumables` in the `HSI_Event` base class. It queries whether
        item_codes are currently available for a particular `hsi_event` and logs the request.

        :param hsi_event: The hsi_event from which the request for consumables originates
        :param item_codes: dict of the form {<item_code>: <quantity>} for the items requested
        :param to_log: whether the request is logged.
        :return:
        """
        # Issue warning if any item_code is not recognised.
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
            available = self._lookup_availability_of_consumables(
                item_codes=item_codes,
                district=self.hs_module.sim.population.props.at[hsi_event.target, 'district_of_residence'],
                facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL,
                month=int(self.hs_module.sim.date.month)
            )

        # Log the request and the outcome:
        if to_log:
            logger.info(key='Consumables',
                        data={
                            'TREATMENT_ID': hsi_event.TREATMENT_ID,
                            'Item_Available': str({k: v for k, v in item_codes.items() if available[k]}),
                            'Item_NotAvailable': str({k: v for k, v in item_codes.items() if not available[k]}),
                        },
                        # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
                        description="Record of each consumable item that is requested."
                        )

        # Return the result of the check on availability
        return available

    def _lookup_availability_of_consumables(self, item_codes: dict, district: str, facility_level: str, month: int
                                            ) -> dict:
        """Lookup whether a particular item_code, at a particular facility_level, district and month, is in the index
        `self.cons_available_today`, which represents what is currently available."""
        # todo don't use month here - just use the current month when determining availability.
        # todo don't make this a long pd.Index: have a set that is specific to a facility_id

        avail = dict()
        for _i in item_codes.keys():
            if _i in self.item_codes:
                avail.update({_i: (_i, district, facility_level, month) in self.cons_available_today})
            else:
                avail.update({_i: self.default_return_value})
        return avail

    def _get_item_codes_from_package_name(self, package: str) -> dict:
        """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
         a given package name."""
        lookups = self.hs_module.parameters['item_and_package_code_lookups']
        ser = lookups.loc[
            lookups['Intervention_Pkg'] == package, ['Item_Code', 'Expected_Units_Per_Case']].set_index(
            'Item_Code')['Expected_Units_Per_Case'].apply(np.ceil).astype(int)
        return ser.groupby(ser.index).sum().to_dict()  # de-duplicate index before converting to dict

    def _get_item_code_from_item_name(self, item: str) -> int:
        """Helper function to provide the item_code (an int) when provided with the name of the item"""
        lookups = self.hs_module.parameters['item_and_package_code_lookups']
        return int(pd.unique(lookups.loc[lookups["Items"] == item, "Item_Code"])[0])


def create_dummy_data_for_cons_availability(intrinsic_availability: Dict[int, bool] = {0: False, 1: True},
                                            districts: List[str] = None,
                                            months: List[int] = [1],
                                            facility_levels: List[int] = ['1a']
                                            ) -> pd.DataFrame:
    """Returns a pd.DataFrame that is a dummy for the imported `ResourceFile_Consumables.csv`.
    By default, it describes the availability of two items, one of which is always available, and one of which is
    never available."""
    list_of_items = []
    for _item, _avail in intrinsic_availability.items():
        for _district in districts:
            for _month in months:
                for _fac_level in facility_levels:
                    list_of_items.append({
                        'item_code': _item,
                        'district': _district,
                        'month': _month,
                        'fac_type_tlo': _fac_level,
                        'available_prop': _avail
                    })
    return pd.DataFrame(data=list_of_items)
