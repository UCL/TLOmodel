import datetime
import warnings
from collections import defaultdict
from itertools import repeat
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tlo import logging

logger = logging.getLogger('tlo.methods.healthsystem')
logger_summary = logging.getLogger('tlo.methods.healthsystem.summary')


class Climate_Disruptions:
    """This is the Climate Disruptions Class. It determines whether a particular HSI event is delayed or cancelled due to
    weather.

    :param climate_ssp: Which future shared socioeconomic pathway (determines degree of warming) is under consideration.
                Options are ssp126, ssp245, and ssp585, in terms of increasing severity.

    :param climate_model_ensemble_model: Which model from the model ensemble for each climate ssp is under consideratin.
                Options are 'lowest', 'mean', and 'highest', based on total precipitation between 2025 and 2070.

    :param services_affected_precip: Which modelled services can be affected by weather. Options are 'all', 'none'

    :param response_to_disruption: How an appointment that is determined to be affected by weather will be handled. Options are 'delay', 'cancel'

    :param delay_in_seeking_care_weather: The number of weeks' delay in reseeking healthcare after an appointmnet has been delayed by weather. Unit is week.
    """

    def __init__(self,
                 climate_ssp: str = None,
                 climate_model_ensemble_model: pd.DataFrame = None,
                 services_affected_precip: str = 'default',
                 delay_in_seeking_care_weather: int = 4
                 ) -> None:

        self._climate_ssp = {
            'none',
            'default',
            'all',
            'all_diagnostics_available',
            'all_medicines_available',
            'all_medicines_and_other_available',
            'all_vital_available',
            'all_drug_or_vaccine_available',
        }

        # Create internal items:
        self._rng = rng
        self._availability = None  # Internal storage of availability assumption (only accessed through getter/setter)
        self._prob_item_codes_available = None  # Data on the probability of each item_code being available
        self._is_available = None  # Dict of sets giving the set of item_codes available, by facility_id
        self._is_unknown_item_available = None  # Whether an unknown item is available, by facility_id
        self._not_recognised_item_codes = defaultdict(set)  # The item codes requested but which are not recognised.

        # Save designations
        self._item_code_designations = item_code_designations

        # Save all item_codes that are defined and pd.Series with probs of availability from ResourceFile
        self.item_codes, self._processed_consumables_data = \
            self._process_consumables_data(availability_data=availability_data)

        # Set the availability based on the argument provided (this can be updated later after the class is initialised)
        self.availability = availability

        # Create (and save pointer to) the `ConsumablesSummaryCounter` helper class
        self._summary_counter = ConsumablesSummaryCounter()
