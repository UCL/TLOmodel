import datetime
import os
from pathlib import Path

import pandas as pd

from tlo.analysis.life_expectancy import get_life_expectancy_estimates


results_folder_dummy_results = Path('/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-2024-05-31T160939Z')

    # Non-summary measure: Estimate should be for each run/draw
rtn_full = get_life_expectancy_estimates(
        results_folder=results_folder_dummy_results,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2020, 12, 31)),
        summary=False,
    )

# now run the test
print(rtn_full)
#rtn_full.to_csv(results_folder_dummy_results"/rtn_full.csv")
