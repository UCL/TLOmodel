from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

import matplotlib.colors as colors
import seaborn as sns

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

outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'testing_scenario_747943'
