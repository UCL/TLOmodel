from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

import os
from scipy.stats import t

import pandas as pd
from tableone import TableOne

# from scripts.comparison_of_horizontal_and_vertical_programs.economic_analysis_for_manuscript.roi_analysis_horizontal_vs_vertical import \
#     icers_summarized
from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, parse_log_file
from src.scripts.costing.cost_estimation import (do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios, summarize_cost_data
)

# Get results file
outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

scenario = 'testing_scenario_747943'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
sim_start_year = 2025

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')

# Get scenario details
info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

modelled_pop = 40_000
p_scaling_factor = 750_000 / modelled_pop # TODO - find source for predicted pregnancies in 2026

#  ======================================= DEFINE HELPER FUNCTIONS  =================================================
