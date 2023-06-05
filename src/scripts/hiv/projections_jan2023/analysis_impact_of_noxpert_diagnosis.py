"""Analyse the results of scenario to test impact of Noxpert diagnosis."""

# python src\scripts\hiv\projections_jan2023\analysis_impact_of_noXpert_diagnosis.py --scenario-outputs-folder outputs\nic503@york.ac.uk
import argparse
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tlo.methods.demography
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
resourcefilepath = Path("./resources")
#datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputspath = Path("./outputs/nic503@york.ac.uk")
#results_folder = get_scenario_outputs("scenario_impact_noXpert_diagnosis.py", outputspath)[-1]
def extract_total_deaths(results_folder):
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})
    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=False
    )
def extract_total_dalys(results_folder):
    def extract_dalys_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})
    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys",
        custom_generate_series=extract_dalys_total,
        do_scaling=False
    )
def make_plot(summarized_total_deaths, param_strings):
    fig, ax = plt.subplots()
    number_of_draws = len(param_strings)
    statistic_values = {
        s: np.array(
            [summarized_total_deaths[(d, s)].values[0] for d in range(number_of_draws)]
        )
        for s in ["mean", "lower", "upper"]
    }
    ax.bar(
        param_strings,
        statistic_values["mean"],
        yerr=[
            statistic_values["mean"] - statistic_values["lower"],
            statistic_values["upper"] - statistic_values["mean"]
        ]
    )
    ax.set_ylabel("Total number of deaths")
    fig.tight_layout()
    return fig, ax
#
# def compute_difference_in_deaths_across_runs(total_deaths, scenario_info):
#     deaths_difference_by_run = [
#         total_deaths[0][run_number]["total_deaths"] - total_deaths[1][run_number]["total_deaths"]
#         for run_number in range(scenario_info["runs_per_draw"])
#     ]
#     return np.mean(deaths_difference_by_run)
# def compute_difference_in_dalys_across_runs(total_dalys, scenario_info):
#     dalys_difference_by_run = [
#         total_dalys[0][run_number]["total_dalys"] - total_dalys[1][run_number]["total_dalys"]
#         for run_number in range(scenario_info["runs_per_draw"])
#     ]
#     return np.mean(dalys_difference_by_run)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Analyse scenario results for noxpert diagnosis pathway"
    )
    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show generated Matplotlib figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save generated Matplotlib figures to results folder",
    )
    args = parser.parse_args()

    # Find results_folder associated with a given batch_file and get most recent
    results_folder = get_scenario_outputs("scenario_impact_noXpert_diagnosis.py", outputspath)[-1]
    print(f"this is the results folder {results_folder}")
    # Load log (useful for checking what can be extracted)
    log = load_pickled_dataframes(results_folder)
    #print(f" the log is {log['tlo.methods.demography'].keys()}")
    log = load_pickled_dataframes(results_folder)
   # print(f" the log is {log['tlo.methods.population']['scaling_factor'].keys()}")

    # # output serialises mortality patterns
    print(f"expected deaths {log['tlo.methods.demography']['death']}")
    #sample_deaths = log['tlo.methods.demography']['death'].groupby(['date', 'cause', 'sex']).size()
    summary_deaths = log['tlo.methods.demography']['death'].drop(columns=[])
    summary_deaths.to_excel(outputspath / "Expected_mortality_NoXpert.xlsx")

    print(f"expected dalys{log['tlo.methods.healthburden']['dalys_stacked']}")
    # sample_dalys= output['tlo.methods.healthburden']['dalys_stacked'].groupby(['cause', 'sex']).size()
    expected_dalys = log['tlo.methods.healthburden']['dalys_stacked'].drop(columns=[])
    expected_dalys.to_excel(outputspath / "Expected_dalys_NoXpert.xlsx")

    # Get basic information about the results
    scenario_info = get_scenario_info(results_folder)

    # Get the parameters that have varied over the set of simulations
    params = extract_params(results_folder)

    # Create a list of strings summarizing the parameter values in the different draws
    param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]

    # Extracts and prints health outcomes to excel-DALYs and mortality
    total_deaths = extract_total_deaths(results_folder)
    print(f"these are expected deaths {total_deaths}")
    total_deaths.to_excel(outputspath / "deaths_NoXpert.xlsx")
    total_dalys = extract_total_dalys(results_folder)
    print(f"these are expected dalys draw {total_dalys}")
    total_dalys.to_excel(outputspath / "dalys_NoXpert.xlsx")
    # Compute and print the difference between the deaths across the scenario draws
    # mean_deaths_difference_by_run = compute_difference_in_deaths_across_runs(total_deaths, scenario_info)
    # print(f"Mean difference in total deaths = {mean_deaths_difference_by_run:.3g}")

    # mean_dalys_difference_by_run = compute_difference_in_dalys_across_runs (total_dalys, scenario_info)
    # print(f"Mean difference in total dalys = {mean_dalys_difference_by_run:.3g}")

    # Plot the total deaths across the two scenario draws as a bar plot with error bars
    fig_1, ax_1 = make_plot(summarize(total_deaths), param_strings)
    fig_2, ax_1 = make_plot(summarize(total_dalys), param_strings)

    # Show Matplotlib figure windows
    if args.show_figures:
        plt.show()

    if args.save_figures:
        fig_1.savefig(results_folder / "total_deaths_across_noxpert.pdf")
        fig_2.savefig(results_folder / "total_dalys_across_noxpert.pdf")









