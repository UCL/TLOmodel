"""This file uses the results of the results of running `nurse_analyses/nurses_scenario_analyses.py` to make some summary
 graphs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)


def extract_total_deaths(results_folder):
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=True
    )


# def plot_summarized_total_deaths(summarized_total_deaths, param_strings):
#     fig, ax = plt.subplots()
#     # number_of_draws = len(param_strings)
#     number_of_draws = len(
#         summarized_total_deaths.columns.get_level_values(0).unique()
#     )
#
#     statistic_values = {
#         s: np.array(
#             [summarized_total_deaths[(d, s)].values[0] for d in range(number_of_draws)]
#         )
#         for s in ["mean", "lower", "upper"]
#     }
#     ax.bar(
#         param_strings,
#         statistic_values["mean"],
#         yerr=[
#             statistic_values["mean"] - statistic_values["lower"],
#             statistic_values["upper"] - statistic_values["mean"]
#         ]
#     )
#     ax.set_ylabel("Total number of deaths")
#     fig.tight_layout()
#     return fig, ax

def plot_summarized_total_deaths(summarized_total_deaths):
    fig, ax = plt.subplots()

    # Get actual draw IDs from the dataframe
    draw_ids = summarized_total_deaths.columns.get_level_values(0).unique()

    means = np.array([
        summarized_total_deaths[(d, "mean")].values[0] for d in draw_ids
    ])
    lowers = np.array([
        summarized_total_deaths[(d, "lower")].values[0] for d in draw_ids
    ])
    uppers = np.array([
        summarized_total_deaths[(d, "upper")].values[0] for d in draw_ids
    ])

    ax.bar(
        draw_ids,
        means,
        yerr=[means - lowers, uppers - means]
    )

    ax.set_ylabel("Total number of deaths")
    ax.set_xlabel("Scenario draw")
    fig.tight_layout()
    return fig, ax


def compute_difference_in_deaths_across_runs(total_deaths, scenario_info):
    deaths_difference_by_run = [
        total_deaths[0][run_number]["Total"] - total_deaths[1][run_number]["Total"]
        for run_number in range(scenario_info["runs_per_draw"])
    ]
    return np.mean(deaths_difference_by_run)


def extract_deaths_by_age(results_folder):
    def extract_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        _, age_group_lookup = make_age_grp_lookup()
        df["Age_Grp"] = df["age"].map(age_group_lookup).astype(make_age_grp_types())
        df = df.rename(columns={"sex": "Sex"})
        return df.groupby(["Age_Grp"])["person_id"].count()

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_by_age_group,
        do_scaling=True
    )


def plot_summarized_deaths_by_age(deaths_summarized_by_age, param_strings):
    fig, ax = plt.subplots()
    for i, param in enumerate(param_strings):
        central_values = deaths_summarized_by_age[(i, "mean")].values
        lower_values = deaths_summarized_by_age[(i, "lower")].values
        upper_values = deaths_summarized_by_age[(i, "upper")].values
        ax.plot(
            deaths_summarized_by_age.index, central_values,
            color=f"C{i}",
            label=param
        )
        ax.fill_between(
            deaths_summarized_by_age.index, lower_values, upper_values,
            alpha=0.5,
            color=f"C{i}",
            label="_"
        )
    ax.set(xlabel="Age-Group", ylabel="Total deaths")
    ax.set_xticks(deaths_summarized_by_age.index)
    ax.set_xticklabels(labels=deaths_summarized_by_age.index, rotation=90)
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        "Analyse scenario results for testing nurses scenario"
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
    # results_folder = get_scenario_outputs(
    #     "scenario_impact_of_consumables_availability.py", args.scenario_outputs_folder
    # )[-1]

    results_folder = Path(
        './outputs/wamulwafu@kuhes.ac.mw/nurses_scenario_outputs-2026-02-09T110530Z'
    )

    # Load log (useful for checking what can be extracted)
    log = load_pickled_dataframes(results_folder)

    # Get basic information about the results
    scenario_info = get_scenario_info(results_folder)

    # # Get the parameters that have varied over the set of simulations
    # params = extract_params(results_folder)
    #
    # # Create a list of strings summarizing the parameter values in the different draws
    # param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]

    number_of_draws = scenario_info["number_of_draws"]
    param_strings = [f"Draw {i}" for i in range(number_of_draws)]

    # We first look at total deaths in the scenario runs
    total_deaths = extract_total_deaths(results_folder)

    # Compute and print the difference between the deaths across the scenario draws
    # mean_deaths_difference_by_run = compute_difference_in_deaths_across_runs(
    #     total_deaths, scenario_info
    # )
    # print(f"Mean difference in total deaths = {mean_deaths_difference_by_run:.3g}")

    # Plot the total deaths across the six scenario draws
    # fig_1, ax_1 = plot_summarized_total_deaths(summarize(total_deaths), param_strings)
    fig_1, ax_1 = plot_summarized_total_deaths(summarize(total_deaths))

    # Age breakdown
    deaths_by_age = extract_deaths_by_age(results_folder)

    # Plot the deaths by age across the six scenario draws as a line plot
    fig_2, ax_2 = plot_summarized_deaths_by_age(summarize(deaths_by_age), param_strings)

    if args.show_figures:
        plt.show()

    if args.save_figures:
        fig_1.savefig(results_folder / "total_deaths_across_scenario_draws.pdf")
        fig_2.savefig(results_folder / "death_by_age_across_scenario_draws.pdf")
