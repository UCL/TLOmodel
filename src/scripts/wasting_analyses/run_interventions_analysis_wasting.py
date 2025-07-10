"""
An analysis file for the wasting module to compare outcomes of one intervention under multiple assumptions.
"""

# %% Import statements
import pickle
import time
from pathlib import Path

import analysis_utility_functions_wast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tlo.analysis.utils import get_scenario_outputs

# start time of the whole analysis
total_time_start = time.time()

# ####### TO SET #######################################################################################################
# Create dicts for the intervention scenarios. 'Interv_abbrev': {'Intervention scenario title/abbreviation': draw_nmb}
# scenarios_dict = {
#     'SQ': {'Status Quo': 0},
#     'GM': {'GM_all': 0, 'GM_1-2': 1, 'GM_FullAttend': 2},
#     'CS': {'CS_10': 0, 'CS_30': 1, 'CS_50': 2, 'CS_100': 3},
#     'FS': {'FS_70':0, 'FS_Full': 1}
# }
scenarios_dict = {
    'SQ': {'Status Quo': 0},
    'GM': {'GM_FullAttend': 0},  # 'GM_all': 0, 'GM_1-2': 1, 'GM_FullAttend': 2},
    'CS': {'CS_100': 0},  # 'CS_10': 0 ,'CS_30': 1, 'CS_50': 2, 'CS_100': 3},
    'FS': {'FS_Full': 0}  # 'FS_70':0, 'FS_Full': 1}
}
# Set the intervention to be analysed, and for which years they were simulated
intervs_all = ['SQ', 'GM', 'CS', 'FS']
intervs_of_interest = ['GM', 'CS', 'FS']
intervention_years = list(range(2026, 2031))
scenarios_to_compare = ['GM_FullAttend', 'CS_100', 'FS_Full']
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
cohorts_to_plot = ['Under-5'] # ['Neonatal', 'Under-5'] #
########################################################################################################################
assert all(interv in intervs_all for interv in intervs_of_interest), ("Some interventions in intervs_of_interest are not"
                                                                      "in intervs_all")
# Ensure Status Quo is always included within the both intervs_of_interest and scenarios_to_compare
if 'SQ' not in intervs_of_interest:
    intervs_of_interest = intervs_of_interest + ['SQ']
if 'Status Quo' not in scenarios_to_compare:
    scenarios_to_compare = scenarios_to_compare + ['Status Quo']
def run_interventions_analysis_wasting(outputspath:Path, plotyears:list, interventionyears:list,
                                       intervs_ofinterest:list, scenarios_tocompare, intervsall) -> None:
    """
    This function saves outcomes from analyses conducted for the Janoušková et al. (2025) paper on acute malnutrition.

    The analyses examine the impact of improved screening or treatment coverage.
    * Outcome 1:
        line plots for each intervs_ofinterest to compare mortality rate over time under multiple settings of the
        intervention and the status quo scenarios
    * Outcome 2:
        line plots to compare mean deaths over time for scenarios_tocompare to each other
    * Outcome 3:
        bars to compare sum of deaths over intervention period for scenarios_tocompare to each other

    :param outputspath: Path to the directory to save output plots/tables
    :param plotyears: The years to be included in the plots/tables
    :param interventionyears: The years during which an intervention is implemented (if any)
    :param intervs_ofinterest: List of interventions to plot scenarios with multiple settings of those interventions;
            (SQ = status quo, GM = growth monitoring, CS = care-seeking, FS = food supplements)
    :param scenarios_tocompare: List of scenarios to be plotted together for comparison
    :param intervsall: List of all interventions
    """

    # Find the most recent folders containing results for each intervention
    iterv_folders_dict = {
        interv: get_scenario_outputs(
            scenario_filename_prefix, Path(interv_scenarios_folder_path / interv)
        )[-1] for interv in intervs_ofinterest
    }
    interv_timestamps_dict = {
        interv: get_scenario_outputs(
            scenario_filename_prefix, Path(interv_scenarios_folder_path / interv)
        )[-1].name.split(f"{scenario_filename_prefix}_{interv}-")[-1]
        for interv in intervs_ofinterest
    }
    print(f"\n{interv_timestamps_dict=}\n")
    # Define folders for each scenario
    scenario_folders = {
        interv: {
            scen_name: Path(iterv_folders_dict[interv] / str(scen_draw_nmb))
            for scen_name, scen_draw_nmb in scenarios_dict[interv].items()
        }
        for interv in intervs_ofinterest
    }

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_colwidth', None)  # Show full content of each row

    # --------------------------------- NEONATAL AND UNDER-5 BIRTH AND DEATH OUTCOMES -------------------------------- #
    # Define paths for saving/loading outcomes
    birth_outcomes_path = outputspath / f"outcomes_data/birth_outcomes_{'_'.join(interv_timestamps_dict.values())}.pkl"
    death_outcomes_path = outputspath / f"outcomes_data/death_outcomes_{'_'.join(interv_timestamps_dict.values())}.pkl"
    dalys_outcomes_path = outputspath / f"outcomes_data/dalys_outcomes_{'_'.join(interv_timestamps_dict.values())}.pkl"

    # Extract or load birth outcomes
    if birth_outcomes_path.exists():
        print("loading birth outcomes from file ...")
        with birth_outcomes_path.open("rb") as f:
            birth_outcomes_dict = pickle.load(f)
    else:
        print("birth outcomes calculation ...")
        birth_outcomes_dict = {
            interv: analysis_utility_functions_wast.extract_birth_data_frames_and_outcomes(
                iterv_folders_dict[interv], plotyears, interventionyears, interv
            )
            for interv in scenario_folders
        }
        print("saving birth outcomes to file ...")
        with birth_outcomes_path.open("wb") as f:
            pickle.dump(birth_outcomes_dict, f)
    # TODO: rm
    # print("\nBIRTH OUTCOMES")
    # for interv in birth_outcomes_dict.keys():
    #     print(f"### {interv=}")
    #     for outcome in birth_outcomes_dict[interv]:
    #         print(f"{outcome}:\n{birth_outcomes_dict[interv][outcome]}")
    #

    # Extract or load death outcomes
    if death_outcomes_path.exists():
        print("loading death outcomes from file ...")
        with death_outcomes_path.open("rb") as f:
            death_outcomes_dict = pickle.load(f)
    else:
        print("death outcomes calculation ...")
        death_outcomes_dict = {
            interv: analysis_utility_functions_wast.extract_death_data_frames_and_outcomes(
                iterv_folders_dict[interv], birth_outcomes_dict[interv]['births_df'], plotyears, interventionyears,
                interv
            ) for interv in scenario_folders
        }
        print("saving death outcomes to file ...")
        with death_outcomes_path.open("wb") as f:
            pickle.dump(death_outcomes_dict, f)
    # # TODO: rm
    # print("\nDEATH OUTCOMES")
    # for interv in death_outcomes_dict.keys():
    #     print(f"### {interv=}")
    #     for outcome in death_outcomes_dict[interv]:
    #         print(f"{outcome}:\n{death_outcomes_dict[interv][outcome]}")
    # #

    # Extract or load dalys outcomes
    dalys_outcomes_path = outputspath / f"outcomes_data/dalys_outcomes_{'_'.join(interv_timestamps_dict.values())}.pkl"
    if dalys_outcomes_path.exists():
        print("loading dalys outcomes from file ...")
        with dalys_outcomes_path.open("rb") as f:
            dalys_outcomes_dict = pickle.load(f)
    else:
        print("dalys outcomes calculation ...")
        dalys_outcomes_dict = {
            interv: analysis_utility_functions_wast.extract_daly_data_frames_and_outcomes(
                iterv_folders_dict[interv], plotyears, interventionyears, interv
            ) for interv in scenario_folders
        }
        print("saving dalys outcomes to file ...")
        with dalys_outcomes_path.open("wb") as f:
            pickle.dump(dalys_outcomes_dict, f)

    # ---------------------------------------------------- Plots  ---------------------------------------------------- #
    print("\n--------------")
    # Prepare scenarios_tocompare_prefix
    if 'Status Quo' in scenarios_tocompare:
        scenarios_tocompare_sq_shorten = [
            'SQ' if scenario == 'Status Quo' else scenario for scenario in scenarios_tocompare
        ]
    else:
        scenarios_tocompare_sq_shorten = scenarios_tocompare
    scenarios_tocompare_prefix = "_".join(scenarios_tocompare_sq_shorten)
    # Prepare timestamps_scenarios_comparison_suffix
    timestamps_scenarios_comparison_suffix = ''
    for interv in intervsall:
        if any(scenario.startswith(interv) for scenario in scenarios_tocompare):
            if timestamps_scenarios_comparison_suffix == '':
                timestamps_scenarios_comparison_suffix = f"{interv_timestamps_dict[interv]}"
            else:
                timestamps_scenarios_comparison_suffix = timestamps_scenarios_comparison_suffix + f"_{interv_timestamps_dict[interv]}"
    if 'Status Quo' in scenarios_tocompare:
        if timestamps_scenarios_comparison_suffix == '':
            timestamps_scenarios_comparison_suffix = f"{interv_timestamps_dict['SQ']}"
        else:
            timestamps_scenarios_comparison_suffix = timestamps_scenarios_comparison_suffix + f"_{interv_timestamps_dict['SQ']}"

    for cohort in cohorts_to_plot:
        print(f"plotting {cohort} outcomes ...")
        print("    plotting mortality rates ...")
        analysis_utility_functions_wast.plot_mortality_rate__by_interv_multiple_settings(
            cohort, interv_timestamps_dict, scenarios_dict, intervs_ofinterest, plotyears, death_outcomes_dict,
            outputspath
        )
        print("    plotting mean deaths ...")
        analysis_utility_functions_wast.plot_mean_outcome_and_CIs__scenarios_comparison(
            cohort, scenarios_dict, scenarios_tocompare, plotyears, "deaths", death_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        print("    plotting sum of deaths ...")
        analysis_utility_functions_wast.plot_sum_deaths_and_CIs__intervention_period(
            cohort, scenarios_dict, scenarios_tocompare, death_outcomes_dict, outputspath,
            scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        print("    plotting mean DALYs ...")
        analysis_utility_functions_wast.plot_mean_outcome_and_CIs__scenarios_comparison(
            cohort, scenarios_dict, scenarios_tocompare, plotyears, "DALYs", dalys_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )

    # --------------------- Create a PDF to save all figures and save each page also as PNG file --------------------- #
    # Create cohort prefix
    cohort_prefix = "_".join(
        ["Neo" if cohort == "Neonatal" else "Under5" if cohort == "Under-5" else cohort for cohort in cohorts_to_plot]
    )
    # Create interventions prefix and timestamps_intervs_plotted suffix
    intervs_ofinterest_prefix = "_".join(intervs_ofinterest) # mortality rates - multiple settings of Interventions
    intervs_plotted = [
        interv for interv in intervsall
        if interv in intervs_ofinterest or \
           any(scenario.startswith(interv) for scenario in scenarios_tocompare_sq_shorten)
    ]
    timestamps_intervs_plotted = "_".join(interv_timestamps_dict[interv] for interv in intervs_plotted)

    pdf_path = outputs_path / (
        f"{cohort_prefix}_{intervs_ofinterest_prefix}_interventions__{scenarios_tocompare_prefix}_scenarios_"
        f"{timestamps_intervs_plotted}.pdf"
    )
    with PdfPages(pdf_path) as pdf:
        # Outcome 1: figures with mortality rates for each interv of interest, comparing different settings
        for page_start in range(0, len(intervs_ofinterest), 2):
            fig1, axes1 = plt.subplots(
                min(2, len(intervs_ofinterest) - page_start),
                len(cohorts_to_plot),
                figsize=(12, 12)
            )

            # Ensure `axes1` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes1 = np.expand_dims(axes1, axis=-1)

            for i, interv in enumerate(intervs_ofinterest[page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot):
                    if interv == 'SQ':
                        mort_rate_png_file_path = outputs_path / (
                            f"{cohort}_mort_rate_{interv}_UNICEF_WPP__"
                            f"{interv_timestamps_dict[interv]}.png"
                        )
                    else:
                        mort_rate_png_file_path = outputs_path / (
                            f"{cohort}_mort_rate_{interv}_multiple_settings__"
                            f"{interv_timestamps_dict[interv]}__{interv_timestamps_dict['SQ']}.png"
                        )
                    interv_title = (
                        'Growth Monitoring attendance (GM)' if interv == 'GM'
                        else 'Food Supplements availability (FS)' if interv == 'FS'
                        else 'Care-Seeking in MAM cases (CS)' if interv == 'CS'
                        else 'SQ' if interv == 'SQ'
                        else 'n/a'
                    )
                    if mort_rate_png_file_path.exists():
                        img = plt.imread(mort_rate_png_file_path)
                        axes1[i, j].imshow(img)
                        axes1[i, j].axis('off')
                        axes1[i, j].set_title(f"{cohort} - {interv_title}", fontsize=10)
            pdf.savefig(fig1)  # Save the current page to the PDF
            fig1_png_file_path = outputs_path / (
                f"{cohort_prefix}_mortality_rates_{'_'.join(intervs_ofinterest[page_start:page_start + 2])}__"
                f"{'_'.join(interv_timestamps_dict[interv] for interv in intervs_ofinterest[page_start:page_start + 2])}.png"
            )
            fig1.savefig(fig1_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG

        # Outcome 2: figures with mean deaths and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            fig2, axes2 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes1` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes2 = np.expand_dims(axes2, axis=-1)

            for i, cause_of_death in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot):
                    mean_deaths_png_file_path = outputs_path / (
                        f"{cohort}_mean_{cause_of_death}_deaths_CI_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if mean_deaths_png_file_path.exists():
                        img = plt.imread(mean_deaths_png_file_path)
                        axes2[i, j].imshow(img)
                        axes2[i, j].axis('off')
                        axes2[i, j].set_title(f"{cohort} - {cause_of_death}", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig2)  # Save the current page to the PDF
            fig2_png_file_path = outputs_path / (
                f"{cohort_prefix}_mean_deaths_comparison_{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig2.savefig(fig2_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
            plt.close(fig2)

        # Outcome 3: figures with sum of deaths and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            fig3, axes3 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes3` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes3 = np.expand_dims(axes3, axis=-1)

            for i, cause_of_death in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot):
                    sum_deaths_png_file_path = outputs_path / (
                        f"{cohort}_sum_{cause_of_death}_deaths_CI_intervention_period_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if sum_deaths_png_file_path.exists():
                        img = plt.imread(sum_deaths_png_file_path)
                        axes3[i, j].imshow(img)
                        axes3[i, j].axis('off')
                        axes3[i, j].set_title(f"{cohort} - {cause_of_death}", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig3)  # Save the current page to the PDF
            fig3_png_file_path = outputs_path / (
                f"{cohort_prefix}_sum_deaths_comparison_{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig3.savefig(fig3_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
            plt.close(fig3)

        # Outcome 4: figures with mean DALYs and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            fig4, axes4 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes4` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes4 = np.expand_dims(axes4, axis=-1)

            for i, cause_of_daly in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot):
                    mean_dalys_png_file_path = outputs_path / (
                        f"{cohort}_mean_{cause_of_daly}_DALYs_CI_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if mean_dalys_png_file_path.exists():
                        img = plt.imread(mean_dalys_png_file_path)
                        axes4[i, j].imshow(img)
                        axes4[i, j].axis('off')
                        axes4[i, j].set_title(f"{cohort} - {cause_of_daly}", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig4)  # Save the current page to the PDF
            fig4_png_file_path = outputs_path / (
                f"{cohort_prefix}_mean_DALYs_comparison_{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig4.savefig(fig4_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
            plt.close(fig4)

# ---------------- #
# RUN THE ANALYSIS #
# ---------------- #
run_interventions_analysis_wasting(outputs_path, plot_years, intervention_years, intervs_of_interest,
                                   scenarios_to_compare, intervs_all)

total_time_end = time.time()
print(f"\ntotal running time (s): {(total_time_end - total_time_start)}")
