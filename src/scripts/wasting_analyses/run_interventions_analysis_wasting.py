"""
An analysis file for the wasting module to compare outcomes of one intervention under multiple assumptions.
"""

# %% Import statements
import time
from pathlib import Path

import analysis_utility_functions_wast
import pandas as pd

from tlo.analysis.utils import get_scenario_outputs

# start time of the whole analysis
total_time_start = time.time()

# ####### TO SET #######################################################################################################
# Create dicts for the intervention scenarios. 'Interv_abbrev': {'Intervention scenario title/abbreviation': draw_nmb}
scenarios_dict = {'SQ': {'Status Quo': 0},
                  'GM': {'GM_all': 0, 'GM_1-2': 1, 'GM_FullAttend': 2},
                  'CS': {'CS_10': 0, 'CS_30': 1, 'CS_50': 2}}
# Set the intervention to be analysed, and for which years they were simulated
intervs_of_interest = ['GM', 'CS']
intervention_years = list(range(2026, 2031))
# Which years to plot (from post burn-in period)
plot_years = list(range(2015, 2031))
# Plot settings
legend_fontsize = 12
title_fontsize = 16

# Where to find the modelled intervention scenarios
interv_scenarios_folder_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios")
# Files names prefix
scenario_filename_prefix = 'wasting_analysis__full_model'
# Where to save the outcomes
outputs_path = Path("./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/_outcomes")
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

    # Find the most recent folders containing results for each intervention
    interventions = intervs_ofinterest + ['SQ']
    iterv_folders_dict = {
        interv: get_scenario_outputs(
            scenario_filename_prefix, Path(interv_scenarios_folder_path / interv)
        )[-1] for interv in interventions
    }
    interv_timestamps_dict = {
        interv: get_scenario_outputs(
            scenario_filename_prefix, Path(interv_scenarios_folder_path / interv)
        )[-1].name.split(f"{scenario_filename_prefix}_{interv}-")[-1]
        for interv in interventions
    }
    print(f"{interv_timestamps_dict=}")
    # Define folders for each scenario
    scenario_folders = {
        interv: {
            scen_name: Path(iterv_folders_dict[interv] / str(scen_draw_nmb))
            for scen_name, scen_draw_nmb in scenarios_dict[interv].items()
        }
        for interv in interventions
    }

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_colwidth', None)  # Show full content of each row
    # ---------------------------------------- NEONATAL AND UNDER-5 MORTALITY ---------------------------------------- #
    # Extract birth outcomes for each intervention to calculate mortality as number of deaths per 1,000 live births
    print("\nbirth outcomes calculation ...")
    birth_outcomes_dict = {
        interv: analysis_utility_functions_wast.extract_birth_data_frames_and_outcomes(iterv_folders_dict[interv],
                                                                                       plotyears, interventionyears)
        for interv in scenario_folders
    }
    # TODO: rm
    # print("\nBIRTH OUTCOMES")
    # for interv in birth_outcomes_dict.keys():
    #     print(f"### {interv=}")
    #     for outcome in birth_outcomes_dict[interv]:
    #         print(f"{outcome}:\n{birth_outcomes_dict[interv][outcome]}")
    #
    # Extract neonatal and under-5 death data for each intervention
    print("\ndeath outcomes calculation ...")
    death_outcomes_dict = {
        interv: analysis_utility_functions_wast.extract_death_data_frames_and_outcomes(
            iterv_folders_dict[interv], birth_outcomes_dict[interv]['births_df'], plotyears, interventionyears
        ) for interv in scenario_folders
    }
    # TODO: rm
    print("\nDEATH OUTCOMES")
    for interv in death_outcomes_dict.keys():
        print(f"### {interv=}")
        for outcome in death_outcomes_dict[interv]:
            print(f"{outcome}:\n{death_outcomes_dict[interv][outcome]}")
    #

    for cohort in ['Neonatal', 'Under-5']:
        analysis_utility_functions_wast.plot_mortality__by_interv_multiple_settings(
            cohort, interv_timestamps_dict, scenarios_dict, intervs_ofinterest, plotyears, death_outcomes_dict,
            outputspath
        )

# ---------------- #
# RUN THE ANALYSIS #
# ---------------- #
run_interventions_analysis_wasting(outputs_path, plot_years, intervention_years, intervs_of_interest)

total_time_end = time.time()
print(f"total running time (s): {(total_time_end - total_time_start)}")
