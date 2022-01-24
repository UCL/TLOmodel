import warnings

import pandas as pd
import numpy as np

from tlo import logging

logger = logging.getLogger('tlo.methods.healthsystem')

class Consumables:
    """This is the Consumables Class. It maintains a current record of the availability and usage of consumables in the
     HealthSystem. It is expected that this is instantiated by the `HealthSystem` module.

    :param: `cons_availability: Determines the availability of consumbales. If 'default' then use the availability
     specified in the ResourceFile; if 'none', then let no consumable be ever be available; if 'all', then all
     consumables are always available. When using 'all' or 'none', requests for consumables are not logged."""

    def __init__(self, hs_module, cons_availabilty: str) -> None:
        self.hs_module = hs_module

        assert cons_availabilty in ['none', 'default', 'all'], "Argument `cons_availability` not recognised."
        self.cons_availability = cons_availabilty

    @property
    def item_codes(self) -> set:
        return {0, 1, 2}  # todo - let this be a property of all the item_codes available

    def process_consumables_df(self, df: pd.DataFrame) -> None:
        """Helper function for processing the consumables data, passed in here as pd.DataFrame that has been read-in by
        the HealthSystem.

        # todo - tidy this!?!
        * Creates ```parameters['Consumables']```
        * Creates ```df_mapping_pkg_code_to_intv_code```
        * Creates ```prob_item_code_available```
        """

        # ------------------------------------------------------------------------------------------------
        # Create a pd.DataFrame that maps pkg code (as index) to item code:
        # This is used to quickly look-up which items are required in each package
        _df = df[['Intervention_Pkg_Code', 'Item_Code', 'Expected_Units_Per_Case']]
        _df = _df.set_index('Intervention_Pkg_Code')
        self.df_mapping_pkg_code_to_intv_code = _df

        # -------------------------------------------------------------------------------------------------
        # Make ```prob_item_codes_available```
        # This is a data-frame that organise the probabilities of individual consumables items being available
        # (by the item codes)
        unique_item_codes = pd.DataFrame(data={'Item_Code': pd.unique(df['Item_Code'])})

        # merge in probabilities of being available
        filter_col = [col for col in df if col.startswith('Available_Facility_Level_')]
        filter_col.append('Item_Code')
        prob_item_codes_available = unique_item_codes.merge(
            df.drop_duplicates(['Item_Code'])[filter_col], on='Item_Code', how='inner'
        )
        assert len(prob_item_codes_available) == len(unique_item_codes)

        # set the index as the Item_Code and save
        self.prob_item_codes_available = prob_item_codes_available.set_index('Item_Code', drop=True)

    def processing_at_start_of_new_day(self) -> None:
        """Do the jobs at the start of each new day.
        * Update the availability of the consumables
        """
        self._refresh_availability_of_consumables()

    def _refresh_availability_of_consumables(self):
        """Update the availability of all items and packages"""
        rng = self.hs_module.rng

        # Determine the availability of the consumables *items* today

        # Random draws: assume that availability of the same item is independent between different facility levels
        random_draws = rng.rand(
            len(self.prob_item_codes_available), len(self.prob_item_codes_available.columns)
        )
        items = self.prob_item_codes_available > random_draws

        # Determine the availability of packages today
        # (packages are made-up of the individual items: if one item is not available, the package is not available)
        pkgs = self.df_mapping_pkg_code_to_intv_code.merge(items, left_on='Item_Code', right_index=True)
        pkgs = pkgs.groupby(level=0)[pkgs.columns[pkgs.columns.str.startswith('Available_Facility_Level')]].all()

        self.cons_available_today = {
            "Item_Code": items,
            "Intervention_Package_Code": pkgs
        }

    def _request_consumables(self, hsi_event, item_codes: dict, to_log: bool) -> dict:
        """
        This is a private function called by the 'get_consumables` in the `HSI_Event` base class. It queries whether
        item_codes are currently available for a particular `hsi_event` and logs the request.

        :param hsi_event: The hsi_event from which the request for consumables originates
        :param item_codes: dict of the form {<item_code>: <quantity>} for the items requested
        :param to_log: whether the request is logged.
        :return:
        """

        for _i in item_codes.keys():
            if _i not in self.item_codes:
                warnings.warn(f"Item_Code {_i} is not recognised.", UserWarning)

        # Determine availability of consumables:
        if self.cons_availability == 'all':
            # All item_codes available available if all consumables should be considered available by default.
            available = {k: True for k in item_codes}
        elif self.cons_availability == 'none':
            # All item_codes not available if consumables should be considered not available by default.
            available = {k: False for k in item_codes.keys()}
        else:
            # Determine availability of each item and package:
            select_col = f'Available_Facility_Level_{hsi_event.ACCEPTED_FACILITY_LEVEL}'
            available = self.cons_available_today['Item_Code'].loc[item_codes.keys(), select_col].to_dict()

        # Log the request and the outcome:
        if to_log:
            logger.info(key='Consumables',
                        data={
                            'TREATMENT_ID': hsi_event.TREATMENT_ID,
                            'Item_Available': str({k: v for k, v in item_codes.items() if v}),
                            'Item_NotAvailable': str({k: v for k, v in item_codes.items() if not v}),
                        },
                        # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
                        description="Record of each consumable item that is requested."
                        )

        # Return the result of the check on availability
        return available

    def _get_item_codes_from_package_name(self, package: str) -> dict:
        """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
         a given package name."""
        consumables = self.parameters['Consumables_OneHealth']
        return consumables.loc[
            consumables['Intervention_Pkg'] == package, ['Item_Code', 'Expected_Units_Per_Case']].set_index(
            'Item_Code')['Expected_Units_Per_Case'].apply(np.ceil).astype(int).to_dict()

    def _get_item_code_from_item_name(self, item: str) -> int:
        """Helper function to provide the item_code (an int) when provided with the name of the item"""
        consumables = self.parameters['Consumables_OneHealth']
        return pd.unique(consumables.loc[consumables["Items"] == item, "Item_Code"])[0]
