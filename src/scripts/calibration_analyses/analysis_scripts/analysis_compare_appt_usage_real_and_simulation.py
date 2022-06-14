"""
Compare appointment usage from model output with real appointment usage.

The real appointment usage is collected from DHIS2 system and HIV Dept.

N.B. This script uses the package `squarify`: so run, `pip install squarify` first.
"""

from pathlib import Path

from tlo.analysis.utils import get_scenario_outputs

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import calendar

sns.set_theme(style="darkgrid")


# path of resource files
rfp = Path('./resources')


# real usage data path
def real_usage_data_path(scenario='raw'):
    p_str = './resources/real_appt_usage_data/' + scenario
    p = Path(p_str)

    return p


actual_usage_path_raw = real_usage_data_path('raw')
actual_usage_path_s2 = real_usage_data_path('s1-s2')
actual_usage_path_s3 = real_usage_data_path('s1-s3')
actual_usage_path_s4 = real_usage_data_path('s1-s4')

# TLO simulation usage path
# the name of the file that specified the scenarios used in this run.
scenario_filename = 'long_run_all_diseases.py'
# path of model output
model_output_path = Path('./outputs/bshe@ic.ac.uk')
# the results folder for the most recent run generated using that scenario_filename
results_folder = get_scenario_outputs(scenario_filename, model_output_path)[-1]

# the simulation data
simulation_usage = pd.read_csv(results_folder/'Simulated appt usage between 2015 and 2019.csv')

# Output path
output_path = Path(results_folder)
