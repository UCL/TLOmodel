import datetime
import heapq as hp
import itertools
import math
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

import tlo
from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import (  # get_filtered_treatment_ids,
    flatten_multi_index_series_into_dict_for_logging,
)
from tlo.events import Event, PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import Metadata
from tlo.methods.bed_days import BedDays
from tlo.methods.consumables import (
    Consumables,
    get_item_code_from_item_name,
    get_item_codes_from_package_name,
)
from tlo.methods.dxmanager import DxManager
from tlo.methods.equipment import Equipment
from tlo.methods.hsi_event import (
    LABEL_FOR_MERGED_FACILITY_LEVELS_1B_AND_2,
    FacilityInfo,
    HSI_Event,
    HSIEventDetails,
    HSIEventQueueItem,
    HSIEventWrapper,
)
from tlo.util import read_csv_files

class Extreme_Weather_Events:
    def __init__(self,
                 service_availability_data: pd.DataFrame = None,
                 rng: np.random = None,
                 availability: str = 'default'
                 ) -> None:

        self._options_for_ewe_consideration = {
            'none',
            'all',
            'ANC', # can add services with further research
        }

    # Create internal items:
        self._rng = rng
        self.affected_services = None
        self._prob_services_disrupted = None  # Data on the probability of each service_code being available

    @property
    def affected_services(self):
        """Returns the internally stored value for the assumption of affected of services."""
        return self._affected_services

    @affected_services.setter
    def affected_services(self, value: str):
        """Changes the effective availability of consumables and updates the internally stored value for that
        assumption.
        Note that this overrides any changes effected by `override_availability()`.
        """
        assert value in self._options_for_affected_services, f"Argument `cons_availability` is not recognised: {value}."
        self._affected_services = value
        self._update_prob_affected_services(self._affected_services)

    def on_start_of_month(self, date: datetime.datetime) -> None:
        """Do the jobs at the start of each new month.
        * Update the probability of disruptions
        """
        self._refresh_affected_services(date)

    def _refresh_affected_services(self, date: datetime.datetime):
        """Update the availability of all services based on the data for the probability of availability, given the current
        date and the projected climate."""
        # Work out which items are available in which facilities for this date.
        month = date.month
        year = date.year
        disruptions_this_month = self._prop_services_disrupted.loc[(year, month, slice(None), slice(None))]
        # Convert to dict to enable checking of disruptions by facility.
        self._is_disruptions = defaultdict(set)
        for _fac_id, _item in disruptions_this_month.to_list():
            self._is_affected[_fac_id].add(_item)

        # Update the default return value (based on the average probability of disruptions at the facility)
        average_affected_services_by_facility_id = disruptions_this_month.groupby(level=0).mean()
        self._is_unknown_disruption = average_affected_services_by_facility_id.to_dict()

