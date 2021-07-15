from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'multi_run_calibration.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
# create_pickles_locally(results_folder)  # if not created via batch
