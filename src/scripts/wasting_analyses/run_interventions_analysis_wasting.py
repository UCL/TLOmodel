"""
An analysis file for the wasting module to compare outcomes of one intervention under multiple assumptions.
"""

# %% Import statements
import pickle
import time
from pathlib import Path

import analysis_utility_functions_wast as util_fncs
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
    'GM': {'GM': 0},  # 'GM_all': 0, 'GM_1-2': 1, 'GM_FullAttend': 2},
    'CS': {'CS': 0},  # 'CS_10': 0 ,'CS_30': 1, 'CS_50': 2, 'CS_100': 3},
    'FS': {'FS': 0},  # 'FS_70':0, 'FS_Full': 1}
    'GM_FS': {'GM_FS': 0},
    'CS_FS': {'CS_FS': 0},
    'GM_CS_FS': {'GM_CS_FS': 0},
    'GM_CS': {'GM_CS': 0},
}
# Set the intervention to be analysed, and for which years they were simulated
intervs_all = ['SQ', 'GM', 'CS', 'FS', 'GM_CS', 'GM_FS', 'CS_FS', 'GM_CS_FS']
intervs_of_interest = ['GM', 'CS', 'FS', 'GM_CS', 'GM_FS', 'CS_FS', 'GM_CS_FS']
intervention_years = list(range(2026, 2031))
scenarios_to_compare = ['GM', 'CS', 'FS', 'GM_CS', 'GM_FS', 'CS_FS', 'GM_CS_FS']
# Which years to plot (from post burn-in period)
plot_years = list(range(2015, 2032))
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
# force_calculation of [births_data, deaths_data, dalys_data, tx_data, medical_cost_data, all_cost_data],
#   if True, enables to force recalculation of the corresponding data
force_calculation = [False, False, False, False, False, False]
# force_calculation = [False, False, False, False, False, True]
######################################################################################################################
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

    :param outputspath: Path to the directory to save output plots/tables;
            Data calculated during analysis will be saved in outputspath/outcomes_data folder for later use
    :param plotyears: The years to be included in the plots/tables
    :param interventionyears: The years during which an intervention is implemented (if any)
    :param intervs_ofinterest: List of interventions to plot scenarios with multiple settings of those interventions;
            (SQ = status quo, GM = growth monitoring, CS = care-seeking, FS = food supplements)
    :param scenarios_tocompare: List of scenarios to be plotted together for comparison
    :param intervsall: List of all interventions
    """

    # deaths and dalys data are extracted for the whole year, which means when plotted in discrete times, at the point
    # of year xxxx, which is beginning of the year data from xxxx-1 year needs to be plotted
    datayears = [year-1 for year in plotyears]
    # when plotting means for intervention years, it needs to be plotted from the first year of interventions being
    # implemented until the beginning of year after last year of interventions
    interv_plotyears = interventionyears + [interventionyears[-1] + 1]
    # to plot the mean for year xxxx, since it shows as in first day of the year, the data from the end of previous
    # years need to be used
    interv_datayears = [year-1 for year in interv_plotyears]

    print("\n----------------------------")
    print("   --- MAIN ANALYSES ---")
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
    print(f"\n{interv_timestamps_dict=}")
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
    if birth_outcomes_path.exists() and not force_calculation[0]:
        print("\nloading birth outcomes from file ...")
        with birth_outcomes_path.open("rb") as f:
            birth_outcomes_dict = pickle.load(f)
    else:
        print("\nbirth outcomes calculation ...")
        birth_outcomes_dict = {
            interv: util_fncs.extract_birth_data_frames_and_outcomes(
                iterv_folders_dict[interv], datayears, interv_datayears, interv
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
    if death_outcomes_path.exists() and not force_calculation[1]:
        print("\nloading death outcomes from file ...")
        with death_outcomes_path.open("rb") as f:
            death_outcomes_dict = pickle.load(f)
    else:
        print("\ndeath outcomes calculation ...")
        death_outcomes_dict = {
            interv: util_fncs.extract_death_data_frames_and_outcomes(
                iterv_folders_dict[interv], birth_outcomes_dict[interv]['births_df'], datayears, interventionyears,
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
    if dalys_outcomes_path.exists() and not force_calculation[2]:
        print("\nloading dalys outcomes from file ...")
        with dalys_outcomes_path.open("rb") as f:
            dalys_outcomes_dict = pickle.load(f)
            SQ_dalys = dalys_outcomes_dict['SQ']
    else:
        print("\ndalys outcomes for intervention period calculation ...")
        sq_dalys = util_fncs.extract_daly_data_frames_and_outcomes(
            iterv_folders_dict['SQ'], datayears, interventionyears, 'SQ'
        )
        dalys_outcomes_dict = {
            interv: util_fncs.extract_daly_data_frames_and_outcomes(
                iterv_folders_dict[interv], datayears, interventionyears, interv, sq_dalys
            ) for interv in scenario_folders
        }
        print("saving dalys outcomes for intervention period to file ...")
        with dalys_outcomes_path.open("wb") as f:
            pickle.dump(dalys_outcomes_dict, f)
    # # TODO: rm
    # print("\nDALY OUTCOMES")
    # for interv in scenario_folders:
    #     print(f"### {interv=}")
    #     for outcome in dalys_outcomes_dict[interv]:
    #         print(f"{outcome}:\n{dalys_outcomes_dict[interv][outcome]}")
    #

    # --------------------------------------------- Main Analyses Plots  --------------------------------------------- #
    # Prepare scenarios_tocompare_prefix
    if 'Status Quo' in scenarios_tocompare:
        scenarios_tocompare_sq_shorten = [
            'SQ' if scenario == 'Status Quo' else scenario for scenario in scenarios_tocompare
        ]
    else:
        scenarios_tocompare_sq_shorten = scenarios_tocompare
    if len(scenarios_tocompare_sq_shorten) > 4:
        scenarios_tocompare_prefix = f"_{len(scenarios_tocompare_sq_shorten)}scenarios_inclSQ"
    else:
        scenarios_tocompare_prefix = "_".join(scenarios_tocompare_sq_shorten)
    # Prepare timestamps_scenarios_comparison_suffix
    timestamps_scenarios_comparison_suffix = ''
    for interv in intervsall:
        if len(intervsall) > 4:
            timestamps_scenarios_comparison_suffix = f"{interv_timestamps_dict['SQ']}"
        else:
            if any(scenario.startswith(interv) for scenario in scenarios_tocompare):
                if timestamps_scenarios_comparison_suffix == '':
                    timestamps_scenarios_comparison_suffix = f"{interv_timestamps_dict[interv]}"
                else:
                    timestamps_scenarios_comparison_suffix = \
                        timestamps_scenarios_comparison_suffix + f"_{interv_timestamps_dict[interv]}"
    if 'Status Quo' in scenarios_tocompare:
        if timestamps_scenarios_comparison_suffix == '':
            timestamps_scenarios_comparison_suffix = f"{interv_timestamps_dict['SQ']}"
        else:
            timestamps_scenarios_comparison_suffix = \
                timestamps_scenarios_comparison_suffix + f"_{interv_timestamps_dict['SQ']}"

    for cohort in cohorts_to_plot:
        print(f"\nplotting {cohort} outcomes ...")
        print("    plotting mortality rates ...")
        util_fncs.plot_mortality_rate__by_interv_multiple_settings(
            cohort, interv_timestamps_dict, scenarios_dict, intervs_ofinterest, plotyears, death_outcomes_dict,
            outputspath
        )
        print("    plotting mean deaths ...")
        util_fncs.plot_mean_outcome_and_CIs__scenarios_comparison(
            cohort, scenarios_dict, scenarios_tocompare, plotyears, "deaths", death_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        util_fncs.plot_mean_outcome_and_CIs__scenarios_comparison(
            cohort, scenarios_dict, scenarios_tocompare, plotyears, "deaths_with_SAM", death_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        print("    plotting sum of deaths ...")
        util_fncs.plot_sum_outcome_and_CIs_intervention_period(
            cohort, scenarios_dict, scenarios_tocompare, "deaths", death_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        util_fncs.plot_sum_outcome_and_CIs_intervention_period(
            cohort, scenarios_dict, scenarios_tocompare, "deaths_with_SAM", death_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        print("    plotting mean DALYs ...")
        util_fncs.plot_mean_outcome_and_CIs__scenarios_comparison(
            cohort, scenarios_dict, scenarios_tocompare, plotyears, "DALYs", dalys_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix
        )
        print("    plotting sum of DALYs ...")
        util_fncs.plot_sum_outcome_and_CIs_intervention_period(
            cohort, scenarios_dict, scenarios_tocompare, "DALYs", dalys_outcomes_dict,
            outputspath, scenarios_tocompare_prefix, timestamps_scenarios_comparison_suffix, interv_timestamps_dict,
            birth_outcomes_dict, force_calculation
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
            nrows = min(2, len(intervs_ofinterest) - page_start)
            ncols = len(cohorts_to_plot)
            fig1, axes1 = plt.subplots(nrows, ncols, figsize=(12, 12))

            # Normalize axes1 to a 2D numpy array so indexing axes1[i, j] always works
            if nrows == 1 and ncols == 1:
                axes1 = np.array([[axes1]])
            elif nrows == 1:
                # axes1 is 1D array of length ncols -> make it shape (1, ncols)
                axes1 = np.expand_dims(axes1, axis=0)
            elif ncols == 1:
                # axes1 is 1D array of length nrows -> make it shape (nrows, 1)
                axes1 = np.expand_dims(axes1, axis=1)
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
                        axes1[i, j].set_title(f"{interv_title}", fontsize=10)
            pdf.savefig(fig1)  # Save the current page to the PDF
            fig1_png_file_path = outputs_path / (
                f"{cohort_prefix}_mortality_rates_{'_'.join(intervs_ofinterest[page_start:page_start + 2])}__"
                f"{'_'.join(interv_timestamps_dict[interv] for interv in intervs_ofinterest[page_start:page_start + 2])}"
                ".png"
            )
            fig1.savefig(fig1_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        # Outcome 2: figures with mean deaths and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            fig2, axes2 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes2 = np.expand_dims(axes2, axis=-1)

            # ### Mean deaths by cause
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
            plt.tight_layout()
            pdf.savefig(fig2)  # Save the current page to the PDF
            fig2_png_file_path = outputs_path / (
                f"{cohort_prefix}_mean_deaths_comparison_"
                f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig2.savefig(fig2_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        for page_start in range(0, len(['ALRI', 'Diarrhoea']), 2):
            fig2_sam, axes2_sam = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes2_sam = np.expand_dims(axes2_sam, axis=-1)

            # ### Mean deaths with SAM by cause
            for i, cause_of_death in enumerate(['ALRI', 'Diarrhoea']):
                fig2_sam, axes2_sam = plt.subplots(1, len(cohorts_to_plot), figsize=(12, 6))
                if len(cohorts_to_plot) == 1:
                    axes2_sam = np.expand_dims(axes2_sam, axis=-1)
                for j, cohort in enumerate(cohorts_to_plot):
                    mean_deaths_with_SAM_png_file_path = outputs_path / (
                        f"{cohort}_mean_{cause_of_death}_deaths_with_SAM_CI_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if mean_deaths_with_SAM_png_file_path.exists():
                        img = plt.imread(mean_deaths_with_SAM_png_file_path)
                        axes2_sam[j].imshow(img)
                        axes2_sam[j].axis('off')
                plt.tight_layout()
                pdf.savefig(fig2_sam)
                fig2_sam_png_file_path = outputs_path / (
                    f"{cohort_prefix}_mean_deaths_with_SAM_comparison_{cause_of_death}__"
                    f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                )
                fig2_sam.savefig(fig2_sam_png_file_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        # Outcome 3: figures with sum of deaths and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            fig3, axes3 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes3` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes3 = np.expand_dims(axes3, axis=-1)

            # ### Sum of deaths over intervention period by cause
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
            plt.tight_layout()
            pdf.savefig(fig3)  # Save the current page to the PDF
            fig3_png_file_path = outputs_path / (
                f"{cohort_prefix}_sum_deaths_comparison_"
                f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig3.savefig(fig3_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        for page_start in range(0, len(['ALRI', 'Diarrhoea']), 2):
            fig3_sam, axes3_sam = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))

            # Ensure `axes3_sam` is always a 2D array for consistent indexing
            if len(cohorts_to_plot) == 1:
                axes3_sam = np.expand_dims(axes3_sam, axis=-1)

            # ### Sum of deaths with SAM over intervention period by cause
            for i, cause_of_death in enumerate(['ALRI', 'Diarrhoea']):
                fig3_sam, axes3_sam = plt.subplots(1, len(cohorts_to_plot), figsize=(12, 6))
                if len(cohorts_to_plot) == 1:
                    axes3_sam = np.expand_dims(axes3_sam, axis=-1)
                for j, cohort in enumerate(cohorts_to_plot):
                    sum_deaths_with_SAM_png_file_path = outputs_path / (
                        f"{cohort}_sum_{cause_of_death}_deaths_with_SAM_CI_intervention_period_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if sum_deaths_with_SAM_png_file_path.exists():
                        img = plt.imread(sum_deaths_with_SAM_png_file_path)
                        axes3_sam[j].imshow(img)
                        axes3_sam[j].axis('off')
                plt.tight_layout()
                pdf.savefig(fig3_sam)
                fig3_sam_png_file_path = outputs_path / (
                    f"{cohort_prefix}_sum_deaths_with_SAM_comparison_{cause_of_death}__"
                    f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                )
                fig3_sam.savefig(fig3_sam_png_file_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        # Outcome 4: figures with mean DALYs and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            cohorts_to_plot_fig4 = [c for c in cohorts_to_plot if c != "Neonatal"]
            fig4, axes4 = plt.subplots(2, len(cohorts_to_plot_fig4), figsize=(12, 12))

            # Ensure `axes4` is always a 2D array for consistent indexing
            if len(cohorts_to_plot_fig4) == 1:
                axes4 = np.expand_dims(axes4, axis=-1)

            for i, cause_of_daly in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot_fig4):
                    mean_dalys_png_file_path = outputs_path / (
                        f"{cohort}_mean_{cause_of_daly}_DALYs_CI_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if mean_dalys_png_file_path.exists():
                        img = plt.imread(mean_dalys_png_file_path)
                        axes4[i, j].imshow(img)
                        axes4[i, j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig4)  # Save the current page to the PDF
            fig4_png_file_path = outputs_path / (
                f"{cohort_prefix}_mean_DALYs_comparison_"
                f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig4.savefig(fig4_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        # Outcome 5: figures with sum of DALYs and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            cohorts_to_plot_fig5 = [c for c in cohorts_to_plot if c != "Neonatal"]
            fig5, axes5 = plt.subplots(2, len(cohorts_to_plot_fig5), figsize=(12, 12))

            # Ensure `axes5` is always a 2D array for consistent indexing
            if len(cohorts_to_plot_fig5) == 1:
                axes5 = np.expand_dims(axes5, axis=-1)

            for i, cause_of_daly in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot_fig5):
                    sum_dalys_png_file_path = outputs_path / (
                        f"{cohort}_sum_{cause_of_daly}_DALYs_CI_intervention_period_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if sum_dalys_png_file_path.exists():
                        img = plt.imread(sum_dalys_png_file_path)
                        axes5[i, j].imshow(img)
                        axes5[i, j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig5)  # Save the current page to the PDF
            fig5_png_file_path = outputs_path / (
                f"{cohort_prefix}_sum_DALYs_comparison_"
                f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig5.savefig(fig5_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        # # Outcome 6: figures with averted sum of deaths and CI, scenarios comparison to SQ
        # for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
        #     fig6, axes6 = plt.subplots(2, len(cohorts_to_plot), figsize=(12, 12))
        #
        #     # Ensure `axes` is always a 2D array for consistent indexing
        #     if len(cohorts_to_plot) == 1:
        #         axes6 = np.expand_dims(axes6, axis=-1)
        #
        #     # ### Sum of averted deaths over intervention period by cause
        #     for i, cause_of_death in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
        #         for j, cohort in enumerate(cohorts_to_plot):
        #             sum_deaths_png_file_path = outputs_path / (
        #                 f"{cohort}_sum_averted_{cause_of_death}_deaths_CI_intervention_period_scenarios_comparison__"
        #                 f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
        #             )
        #             if sum_deaths_png_file_path.exists():
        #                 img = plt.imread(sum_deaths_png_file_path)
        #                 axes6[i, j].imshow(img)
        #                 axes6[i, j].axis('off')
        #     plt.tight_layout()
        #     pdf.savefig(fig6)  # Save the current page to the PDF
        #     fig6_png_file_path = outputs_path / (
        #         f"{cohort_prefix}_averted_sum_deaths_comparison_"
        #         f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
        #         f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
        #     )
        #     fig6.savefig(fig6_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        # plt.close('all')

        # Outcome 7: figures with averted sum of DALYs and CI, scenarios comparison
        for page_start in range(0, len(['any cause', 'SAM', 'ALRI', 'Diarrhoea']), 2):
            cohorts_to_plot_fig7 = [c for c in cohorts_to_plot if c != "Neonatal"]
            fig7, axes7 = plt.subplots(2, len(cohorts_to_plot_fig7), figsize=(12, 12))

            # Ensure `axes` is always a 2D array for consistent indexing
            if len(cohorts_to_plot_fig7) == 1:
                axes7 = np.expand_dims(axes7, axis=-1)

            for i, cause_of_daly in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2]):
                for j, cohort in enumerate(cohorts_to_plot_fig7):
                    sum_dalys_png_file_path = outputs_path / (
                        f"{cohort}_sum_averted_{cause_of_daly}_DALYs_CI_intervention_period_scenarios_comparison__"
                        f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
                    )
                    if sum_dalys_png_file_path.exists():
                        img = plt.imread(sum_dalys_png_file_path)
                        axes7[i, j].imshow(img)
                        axes7[i, j].axis('off')
            plt.tight_layout()
            pdf.savefig(fig7)  # Save the current page to the PDF
            fig7_png_file_path = outputs_path / (
                f"{cohort_prefix}_sum_DALYs_comparison_"
                f"{'_'.join(['any cause', 'SAM', 'ALRI', 'Diarrhoea'][page_start:page_start + 2])}__"
                f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
            )
            fig7.savefig(fig7_png_file_path, dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close('all')

        # Outcome 8: cost-effectiveness sensitivity plot
        cost_effectiveness_png_path = outputs_path / (
            f"{cohort_prefix}_cost_effectiveness_sensitivity_grid__"
            f"{scenarios_tocompare_prefix}__{timestamps_scenarios_comparison_suffix}.png"
        )
        if cost_effectiveness_png_path.exists():
            # Read image and set figure size to match pixel dimensions so embedding keeps original quality
            img = plt.imread(cost_effectiveness_png_path)
            h, w = img.shape[0], img.shape[1]
            target_dpi = 300
            figsize = (w / target_dpi, h / target_dpi)
            fig_ce = plt.figure(figsize=figsize, dpi=target_dpi)
            ax_ce = fig_ce.add_axes([0, 0, 1, 1])
            ax_ce.imshow(img, interpolation='nearest', aspect='auto')
            ax_ce.axis('off')
            pdf.savefig(fig_ce, dpi=target_dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig_ce)
        plt.close('all')

# --------------------------------------- Behind the scene Analyses Plots  --------------------------------------- #
def run_behind_the_scene_analysis_wasting(
    outputspath: Path,
    plotyears: list,
    interventionyears: list,
    intervs_ofinterest: list,
    scenariosdict
) -> None:
    """
    Loads or extracts treatment outcomes for behind-the-scenes analysis.
    """

    datayears = [year-1 for year in plotyears]

    print("\n----------------------------")
    print("--- BEHIND-THE-SCENE ANALYSES ---")
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
    print(f"\n{interv_timestamps_dict=}")
    # Define folders for each scenario
    scenario_folders = {
        interv: {
            scen_name: Path(iterv_folders_dict[interv] / str(scen_draw_nmb))
            for scen_name, scen_draw_nmb in scenariosdict[interv].items()
        }
        for interv in intervs_ofinterest
    }

    info_pickles_file_path = outputspath / "outcomes_data/pickles_regenerated.pkl"
    regenerate_pickles_bool = False
    if info_pickles_file_path.exists():
        print("\nloading pickles_regenerated_df from file ...")
        with info_pickles_file_path.open("rb") as f:
            pickles_regenerated_df = pickle.load(f)
    else:
        pickles_regenerated_df = pd.DataFrame(columns=["interv", "timestamp"])
    # check all are already regenerated, if any is not regenerate them all and add the timestamps to the df
    for interv, timestamp in interv_timestamps_dict.items():
        if not (
            (pickles_regenerated_df["interv"] == interv) & (pickles_regenerated_df["timestamp"] == timestamp)
        ).any():
            regenerate_pickles_bool = True
            pickles_regenerated_df = pd.concat([
                pickles_regenerated_df,
                pd.DataFrame({"interv": [interv], "timestamp": [timestamp]})
            ], ignore_index=True)

    if regenerate_pickles_bool:
        print("saving pickles_regenerated_df to file ...")
        with info_pickles_file_path.open("wb") as f:
            pickle.dump(pickles_regenerated_df, f)
        print("\nRegenerating pickles with debug logs ...")
        util_fncs.regenerate_pickles_with_debug_logs(iterv_folders_dict)

    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_colwidth', None)  # Show full content of each row

    tx_outcomes_path = \
        (outputspath /
         f"outcomes_data/tx_outcomes_{'_'.join(iterv_folders_dict[interv].name for interv in scenario_folders)}.pkl")

    # Extract or load treatment outcomes
    if tx_outcomes_path.exists() and not force_calculation[3]:
        print("\nloading tx outcomes from file ...")
        with tx_outcomes_path.open("rb") as f:
            tx_outcomes_dict = pickle.load(f)
    else:
        print("\ntx outcomes calculation ...")
        tx_outcomes_dict = {
            interv: util_fncs.extract_tx_data_frames(
                iterv_folders_dict[interv], datayears, interventionyears, interv
            ) for interv in scenario_folders
		}
        print("saving tx outcomes to file ...")
        with tx_outcomes_path.open("wb") as f:
            pickle.dump(tx_outcomes_dict, f)

    # Further analysis and plotting will be added here
    # TODO: rm
    # print("\nTX OUTCOMES")
    # for interv in tx_outcomes_dict.keys():
    #     print(f"### {interv=}")
    #     for outcome in tx_outcomes_dict[interv]:
    #         print(f"{outcome}:\n{tx_outcomes_dict[interv][outcome]}")

    # print("    plotting mean nmbs of tx...")
    # util_fncs.plot_mean_tx_and_CIs__scenarios_comparison()

# ---------------- #
# RUN THE ANALYSIS #
# ---------------- #
run_interventions_analysis_wasting(outputs_path, plot_years, intervention_years, intervs_of_interest,
                                   scenarios_to_compare, intervs_all)
# run_behind_the_scene_analysis_wasting(outputs_path, plot_years, intervention_years, intervs_of_interest,
#                                       scenarios_dict)

total_time_end = time.time()
print(f"\ntotal running time (s): {(total_time_end - total_time_start)}")
