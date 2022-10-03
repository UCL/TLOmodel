# tlo batch-download scenario4_SI-2022-08-25T115554Z


import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results = get_scenario_outputs("scenario4_SI.py", outputspath)[-1]
baseline_results = get_scenario_outputs("scenario0.py", outputspath)[-1]

# %%: get difference in numbers of deaths by intervention

# extract numbers of deaths (median, 95% UI) by scenario (9 scenarios)

# aggregate over 2023-2035

# for plot - difference from baseline (scenario 0)
# (scenario - scenario0) / scenario0





# %%: get difference in DALYs by intervention
