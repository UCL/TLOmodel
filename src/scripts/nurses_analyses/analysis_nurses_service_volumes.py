"""
This script produces service volume figures for the Nurse Shortages analysis:
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import extract_results, load_pickled_dataframes, summarize


def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    print("\nInside find_difference_relative_to_comparison_series")
    print(type(_ser))
    print(_ser.index)
    print(_ser.head())
    print(_ser.index.names)

    return (
        _ser
        .unstack(level=0)
        .apply(
            lambda x: (
                (x - x[comparison]) /
                (x[comparison] if scaled else 1.0)
            ),
            axis=1,
        )
        .drop(
            columns=([comparison] if drop_comparison else [])
        )
        .stack()
    )


def find_difference_relative_to_comparison_series_dataframe(
    _df: pd.DataFrame,
    **kwargs,
):
    return pd.concat(
        {
            idx: find_difference_relative_to_comparison_series(
                row,
                **kwargs,
            )
            for idx, row in _df.iterrows()
        },
        axis=1,
    ).T


def extract_annual_service_volumes(results_folder):
    def get_num_service_volumes_yearly(df: pd.DataFrame):
        df = df.copy()

        if "year" not in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year

        yearly = (
            df.assign(
                total_hsis=df["TREATMENT_ID"].apply(lambda d: sum(d.values()))
            )
            .groupby("year")["total_hsis"]
            .sum()
        )

        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=get_num_service_volumes_yearly,
        do_scaling=True,
    )


def plot_annual_service_volumes(
    summarized_annual_service_volumes,
    scenarios,
    title,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function":
            "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function":
            "More CNP by district",
    }

    for scenario in scenarios:
        mean = summarized_annual_service_volumes[scenario]["mean"]
        lower = summarized_annual_service_volumes[scenario]["lower"]
        upper = summarized_annual_service_volumes[scenario]["upper"]
        ax.plot(mean.index, mean.values, linewidth=2, label=label_map.get(scenario, scenario))
        ax.fill_between(mean.index, lower.values, upper.values, alpha=0.2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Service Volume")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# def calculate_percent_service_volume_change(
#     summarized_annual_service_volumes,
#     scenarios,
#     baseline_scenario,
# ):
#     results = {}
#
#     baseline = summarized_annual_service_volumes[baseline_scenario]
#
#     for scenario in scenarios:
#
#         if scenario == baseline_scenario:
#             continue
#
#         percent_change = find_difference_relative_to_comparison_series(
#             summarized_annual_service_volumes[scenario]["mean"],
#             baseline["mean"],
#         )
#
#         results[scenario] = percent_change
#
#     return pd.DataFrame(results)

def calculate_percent_service_volume_change(
    annual_service_volumes,
    baseline_scenario,
):
    pct_change = find_difference_relative_to_comparison_series_dataframe(
        annual_service_volumes,
        comparison=baseline_scenario,
        scaled=True,
    )

    return summarize(pct_change)


# def plot_percent_service_volume_change(
#     percent_change_df,
#     title,
# ):
#     fig, ax = plt.subplots(figsize=(9, 6))
#
#     label_map = {
#         "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
#         "More Nurses / Default Healthsystem Function": "More nurses",
#         "More CNP staff / Default Healthsystem Function": "More CNP",
#         "More Nurses by District / Default Healthsystem Function":
#             "More nurses by district",
#         "More CNP staff by District / Default Healthsystem Function":
#             "More CNP by district",
#
#         "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
#         "More Nurses / Improved Healthsystem Function": "More nurses",
#         "More CNP staff / Improved Healthsystem Function": "More CNP",
#         "More Nurses by District / Improved Healthsystem Function":
#             "More nurses by district",
#         "More CNP staff by District / Improved Healthsystem Function":
#             "More CNP by district",
#     }
#
#     for scenario in percent_change_df.columns:
#         ax.plot(
#             percent_change_df.index,
#             percent_change_df[scenario],
#             linewidth=2,
#             label=label_map.get(scenario, scenario),
#         )
#
#     ax.axhline(0, color="black", linestyle="--", linewidth=1)
#
#     ax.set_xlabel("Year")
#     ax.set_ylabel("% change in service volume")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     return fig

def plot_percent_service_volume_change(
    summarized_percent_change,
    title,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    label_map = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function":
            "More CNP by district",

        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function":
            "More CNP by district",
    }

    # Get the scenario names (top level of the MultiIndex)
    scenarios = (
        summarized_percent_change.columns
        .get_level_values(0)
        .unique()
    )

    years = summarized_percent_change.index

    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios

    x = np.arange(len(years))

    offsets = (
                  np.arange(n_scenarios) - (n_scenarios - 1) / 2
              ) * width

    for i, scenario in enumerate(scenarios):
        mean = summarized_percent_change[scenario]["mean"] * 100
        lower = summarized_percent_change[scenario]["lower"] * 100
        upper = summarized_percent_change[scenario]["upper"] * 100

        xpos = x + offsets[i]

        ax.bar(xpos, mean.values, width=width, label=label_map.get(scenario, scenario))

        ax.errorbar(xpos, mean.values,
                    yerr=[
                        mean.values - lower.values,
                        upper.values - mean.values,
                    ],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    lw=1,
                    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("% change in service volume")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Analyse service volume across nurse staffing scenarios"
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
    # Use command-line folder
    results_folder = args.scenario_outputs_folder
    # Optional: load logs
    log = load_pickled_dataframes(results_folder)

    # Getting scenario names from scenario class
    param_names = tuple(StaffingScenario()._scenarios.keys())

    default_hs_scenarios = [
        "Baseline Nurses / Default Healthsystem Function",
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    baseline_scenario = "Baseline Nurses / Default Healthsystem Function"

    improved_hs_scenarios = [
        "Baseline Nurses / Improved Healthsystem Function",
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    baseline_improved_scenario = ("Baseline Nurses / Improved Healthsystem Function")

    annual_service_volumes = extract_annual_service_volumes(results_folder)

    annual_service_volumes = set_param_names_as_column_index_level_0(
        annual_service_volumes,
        param_names,
    )

    summarized_annual_service_volumes = summarize(
        annual_service_volumes
    )

    # percent_change_default = calculate_percent_service_volume_change(
    #     summarized_annual_service_volumes,
    #     default_hs_scenarios,
    #     baseline_scenario,
    # )
    #
    # percent_change_improved = calculate_percent_service_volume_change(
    #     summarized_annual_service_volumes,
    #     improved_hs_scenarios,
    #     baseline_improved_scenario,
    # )

    percent_change_default = calculate_percent_service_volume_change(
        annual_service_volumes[default_hs_scenarios],
        baseline_scenario,
    )

    percent_change_improved = calculate_percent_service_volume_change(
        annual_service_volumes[improved_hs_scenarios],
        baseline_improved_scenario,
    )

    fig_default = plot_annual_service_volumes(
        summarized_annual_service_volumes,
        default_hs_scenarios,
        "Annual Service Volumes (Default Healthsystem Function)",
    )

    fig_improved = plot_annual_service_volumes(
        summarized_annual_service_volumes,
        improved_hs_scenarios,
        "Annual Service Volumes (Improved Healthsystem Function)",
    )


    fig_default_percent = plot_percent_service_volume_change(
        percent_change_default,
        "% Change in Service Volume vs Baseline\n(Default Healthsystem Function)",
    )

    fig_improved_percent = plot_percent_service_volume_change(
        percent_change_improved,
        "% Change in Service Volume vs Baseline\n(Improved Healthsystem Function)",
    )

    if args.save_figures:
        output_folder = results_folder / "service_volume_plots"

        output_folder.mkdir(exist_ok=True)

        fig_default.savefig(
            output_folder / "annual_service_volumes_default.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_improved.savefig(
            output_folder / "annual_service_volumes_improved.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_default_percent.savefig(
            output_folder / "percent_change_service_volumes_default.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig_improved_percent.savefig(
            output_folder / "percent_change_service_volumes_improved.png",
            dpi=300,
            bbox_inches="tight",
        )
