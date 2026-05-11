"""This file uses the results of the results of running `nurse_analyses/nurses_scenario_analyses.py` to make some summary
 graphs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)


# Rename draw numbers to scenario names
def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


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


def plot_summarized_total_deaths(summarized_total_deaths):
    fig, ax = plt.subplots()

    scenario_names = summarized_total_deaths.columns.get_level_values(0).unique()

    means = np.array([
        summarized_total_deaths[(s, "mean")].values[0]
        for s in scenario_names
    ])
    lowers = np.array([
        summarized_total_deaths[(s, "lower")].values[0]
        for s in scenario_names
    ])
    uppers = np.array([
        summarized_total_deaths[(s, "upper")].values[0]
        for s in scenario_names
    ])

    ax.bar(
        scenario_names,
        means,
        yerr=[means - lowers, uppers - means],
        capsize=5
    )

    ax.set_ylabel("Total number of deaths")
    ax.set_xticklabels(scenario_names, rotation=45, ha="right")
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


def plot_summarized_deaths_by_age(deaths_summarized_by_age):
    fig, ax = plt.subplots()

    scenario_names = deaths_summarized_by_age.columns.get_level_values(0).unique()

    for i, scenario in enumerate(scenario_names):
        central_values = deaths_summarized_by_age[(scenario, "mean")].values
        lower_values = deaths_summarized_by_age[(scenario, "lower")].values
        upper_values = deaths_summarized_by_age[(scenario, "upper")].values

        ax.plot(
            deaths_summarized_by_age.index,
            central_values,
            label=scenario
        )

        ax.fill_between(
            deaths_summarized_by_age.index,
            lower_values,
            upper_values,
            alpha=0.3
        )

    ax.set(xlabel="Age-Group", ylabel="Total deaths")
    ax.set_xticks(deaths_summarized_by_age.index)
    ax.set_xticklabels(deaths_summarized_by_age.index, rotation=90)
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Analyse scenario results for nurses scenario"
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
        help="Whether to interactively show figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save figures to results folder",
    )
    args = parser.parse_args()

    # results_folder = args.scenario_outputs_folder

    results_folder = Path(
        './outputs/wamulwafu@kuhes.ac.mw/nurses_scenario_outputs-2026-02-09T110530Z'
    )

    # Load log (optional, but useful)
    log = load_pickled_dataframes(results_folder)

    scenario_info = get_scenario_info(results_folder)

    # Get scenario names directly from Scenario class

    param_names = tuple(StaffingScenario()._scenarios.keys())

    # Total deaths
    total_deaths = extract_total_deaths(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names
    )

    summarized_total_deaths = summarize(total_deaths)

    fig_1, ax_1 = plot_summarized_total_deaths(summarized_total_deaths)

    # Deaths by age
    deaths_by_age = extract_deaths_by_age(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names
    )

    summarized_deaths_by_age = summarize(deaths_by_age)

    fig_2, ax_2 = plot_summarized_deaths_by_age(summarized_deaths_by_age)

    if args.show_figures:
        plt.show()

    if args.save_figures:
        fig_1.savefig(results_folder / "total_deaths_across_scenarios.pdf")
        fig_2.savefig(results_folder / "deaths_by_age_across_scenarios.pdf")
