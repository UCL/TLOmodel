"""
An analysis file for the wasting module to compare outcomes of one intervention under multiple assumptions.
"""
# %% Import statements
import analysis_utility_functions_wast
import glob
import gzip
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PyPDF2 import PdfReader, PdfWriter
from scipy import stats

from tlo.analysis.utils import compare_number_of_deaths, get_scenario_outputs, parse_log_file

# start time of the whole analysis
total_time_start = time.time()

# ####### TO SET #######################################################################################################
# Create dicts for the intervention scenarios. 'Interv_abbrev': {'Intervention scenario title/abbreviation': draw_nmb}
scenarios_dict = {'GM': {'Status Quo': 0, 'GM_all': 1, 'GM_1-2': 2, 'GM_FullAttend': 3}}
# Set the intervention to be analysed, and for which years they were simulated
intervs_of_interest = ['GM']
intervention_years = list(range(2026, 2031))
# Which years to plot (from post burn-in period)
plot_years = list(range(2015, 2031))
# Plot settings
legend_fontsize = 12
title_fontsize = 16

# Where to find the modelled intervention scenarios
interv_scenarios_folder_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios")
# Files names prefix
scenario_filename_prefix = 'wasting_analysis__minimal_model'
# Where to save the outcomes
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/outcomes")
########################################################################################################################

def run_interventions_analysis_wasting(outputspath:Path, plotyears:list, interventionyears:list,
                                       intervs_ofinterest:list) -> None:
    """
    This function saves outcomes from analyses conducted for the Janoušková et al. (2025) paper on acute malnutrition.

    The analyses examine the impact of improved screening or treatment coverage. Outcomes from a single intervention,
    evaluated under multiple assumptions, are compared with each other and with outcomes from a status quo scenario.

    :param outputspath: path to the directory to save output plots/tables
    :param plotyears: the years to be included in the plots/tables
    :param interventionyears: the years during which an intervention is implemented (if any)
    :param intervs_ofinterest: list of interventions being analysed;
            (GM = growth monitoring, CS = care-seeking, FS = food supplements)
    """

    # Find the most recent folders containing results for each intervention of interest
    iterv_folders_dict = {
        interv: get_scenario_outputs(
            scenario_filename_prefix, Path(interv_scenarios_folder_path / interv)
        )[-1] for interv in intervs_of_interest
    }
    # Define folders for each scenario
    scenario_folders = {
        interv: {
            scen_name: Path(iterv_folders_dict[interv] / str(scen_draw_nmb))
            for scen_name, scen_draw_nmb in scenarios_dict[interv].items()
        }
        for interv in intervs_of_interest
    }

    # ---------------------------------------- NEONATAL AND UNDER-5 MORTALITY ---------------------------------------- #
    # Extract neonatal and under-5 death data from each of the scenario files
    print()
    empty_df = pd.DataFrame() # TODO need to be replaced by births_dict[scen_folder]['births_data_frame'])
    death_data = {scen_name: analysis_utility_functions_wast.extract_death_data_frames_and_outcomes(scen_folder, empty_df)
                  for interv, scenarios in scenario_folders.items() for scen_name, scen_folder in scenarios.items()}



# ---------------- #
# RUN THE ANALYSIS #
# ---------------- #
run_interventions_analysis_wasting(outputs_path, plot_years, intervention_years, intervs_of_interest)

total_time_end = time.time()
print(f"total running time (s): {(total_time_end - total_time_start)}")
