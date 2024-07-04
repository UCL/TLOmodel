import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    summarize,
)

from tlo.analysis.life_expectancy import (
    _map_age_to_age_group,
    _extract_person_years,
    _num_deaths_by_age_group,
    _aggregate_person_years_by_age,
    _estimate_life_expectancy,
    get_life_expectancy_estimates)

from tlo.analysis.life_expectancy import get_life_expectancy_estimates

get_life_expectancy_estimates('/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-2024-05-31T160939Z', median=True,
                                         target_period=(Date(2019, 1, 1), Date(2020, 1, 1)))
