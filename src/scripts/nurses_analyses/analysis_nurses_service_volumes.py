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


def extract_annual_treatment_volumes(results_folder):
    """
    Extract annual treatment volumes by TREATMENT_ID.

    Returns a DataFrame where:
        index   = year
        columns = TREATMENT_ID
        values  = number of treatments delivered
    """

    def get_treatment_volumes_yearly(df: pd.DataFrame):
        df = df.copy()

        if "year" not in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year

        rows = []

        for _, row in df.iterrows():
            treatment_dict = row["TREATMENT_ID"]

            if not isinstance(treatment_dict, dict):
                continue

            year = row["year"]

            for treatment_id, count in treatment_dict.items():
                rows.append({"year": year, "treatment_id": treatment_id, "count": count, })

        treatment_df = pd.DataFrame(rows)

        if treatment_df.empty:
            return pd.Series(dtype=float)

        yearly = (
            treatment_df
            .groupby(["year", "treatment_id"])["count"]
            .sum()
        )

        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=get_treatment_volumes_yearly,
        do_scaling=True,
    )


def aggregate_treatment_volumes_by_service_area(
    annual_treatment_volumes,
    comparison_years=range(2027, 2035),
):
    """
    Aggregate treatment volumes into broad service areas.
    The service area is defined as everything before the first
    underscore in the treatment_id.
    """

    # Select the comparison years
    years = annual_treatment_volumes.index.get_level_values("year").astype(int)

    year_mask = np.isin(
        years,
        list(comparison_years),
    )

    selected = annual_treatment_volumes.loc[year_mask].copy()

    # Get treatment IDs
    treatment_ids = selected.index.get_level_values(
        "treatment_id"
    )

    # Create service-area names
    service_areas = treatment_ids.to_series().str.split(
        "_",
        n=1,
        expand=True,
    )[0].values

    # Add service area as another index level
    selected.index = pd.MultiIndex.from_arrays(
        [
            selected.index.get_level_values("year"),
            service_areas,
        ],
        names=["year", "service_area"],
    )

    # Sum across years and treatment types within each service area
    service_area_volumes = selected.groupby(
        level="service_area"
    ).sum()

    return service_area_volumes


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

    color_map = {
        "Baseline Nurses / Default Healthsystem Function": "black",
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",

        "Baseline Nurses / Improved Healthsystem Function": "black",
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    for scenario in scenarios:
        mean = summarized_annual_service_volumes[scenario]["mean"]
        lower = summarized_annual_service_volumes[scenario]["lower"]
        upper = summarized_annual_service_volumes[scenario]["upper"]
        color = color_map.get(scenario, "gray")

        ax.plot(mean.index, mean.values, linewidth=2, color=color, label=label_map.get(scenario, scenario))
        ax.fill_between(mean.index, lower.values, upper.values, color=color, alpha=0.2)

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
    comparison_years=range(2027, 2035),
):
    """
    Calculate percentage change in total service volumes over the
    comparison period relative to baseline.

    Service volumes are first summed across 2027-2034 for each
    scenario and simulation run. The percentage change is then
    calculated run-to-run relative to the baseline scenario.
    """
    # Selecting comparison years
    years = annual_service_volumes.index.astype(int)
    year_mask = np.isin(years, list(comparison_years),)
    selected = annual_service_volumes.loc[year_mask]

    # Sum service volumes over 2027-2034 for each scenario/run
    total_service_volumes = selected.sum(axis=0).to_frame().T

    # Run-to-run comparison with baseline
    pct_diff = find_difference_relative_to_comparison_series_dataframe(
        total_service_volumes,
        comparison=baseline_scenario,
        scaled=True,
    )

    # Summarize across simulation runs
    summarized = summarize(pct_diff)
    return summarized


def make_treatment_volume_table(
    annual_treatment_volumes,
    comparison_years=range(2027, 2035),
):
    # Creates a table of total treatment volumes by treatment ID
    # for the specified comparison years.
    years = annual_treatment_volumes.index.get_level_values(0).astype(int)
    year_mask = np.isin(years, list(comparison_years))
    selected = annual_treatment_volumes.loc[year_mask]
    table = (selected.groupby(level=1).sum())
    return table


def calculate_treatment_volume_percent_change(
    treatment_volumes,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate percentage change in treatment volumes relative to baseline
    using run-to-run comparisons.
    """

    years = treatment_volumes.index.get_level_values("year").astype(int)

    year_mask = np.isin(
        years,
        list(comparison_years),
    )

    treatment_volumes = treatment_volumes.loc[year_mask]

    # Sum treatment volumes over 2027-2034
    treatment_volumes_agg = treatment_volumes.groupby(
        level="treatment_id"
    ).sum()

    # Run-to-run comparison

    pct_diff = (
        100.0
        * find_difference_relative_to_comparison_series_dataframe(
            treatment_volumes_agg,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    # Summarize across runs
    summarized = summarize(pct_diff)

    return summarized


def calculate_service_area_volume_percent_change(
    service_area_volumes,
    baseline_scenario,
):
    """
    Calculate percentage change in service volume by service area
    relative to baseline using run-to-run comparisons.
    """

    pct_diff = (
        100.0
        * find_difference_relative_to_comparison_series_dataframe(
            service_area_volumes,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    return summarized


# def add_service_area_to_treatment_results(treatment_results):
#     # Adds service area based on the text before the first underscore in the treatment ID.
#     results = treatment_results.copy()
#
#     results["service_area"] = (
#         results.index.to_series()
#         .str.split("_")
#         .str[0]
#     )
#
#     return results
#
#
# def plot_treatment_volumes(
#     treatment_volume_table,
#     scenario,
#     title,
#     top_n=15,
# ):
#     """
#     Plot total treatment volumes for a scenario.
#     Only the top_n treatments by volume are shown.
#     """
#
#     data = treatment_volume_table[scenario].sort_values(
#         ascending=False
#     ).head(top_n)
#     fig, ax = plt.subplots(figsize=(12, 7))
#     data.sort_values().plot(kind="barh", ax=ax)
#     ax.set_xlabel("Number of treatments")
#     ax.set_ylabel("Treatment")
#     ax.set_title(title)
#     fig.tight_layout()
#     return fig
#
#
# def plot_annual_treatment_volumes_by_service_area(
#     annual_treatment_volumes,
#     scenarios,
#     service_area,
#     title,
# ):
#     """
#     Plot annual treatment volumes for one disease/service area
#     across nurse staffing scenarios.
#     """
#
#     # Select the requested service area
#     data = annual_treatment_volumes[
#         annual_treatment_volumes.index.get_level_values(
#             "service_area"
#         ) == service_area
#     ]
#
#     # Remove service_area from index so that year is the index
#     data = data.droplevel("service_area")
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     label_map = {
#         "Baseline Nurses / Default Healthsystem Function":
#             "Baseline",
#         "Fewer Nurses / Default Healthsystem Function":
#             "Fewer nurses",
#         "More Nurses / Default Healthsystem Function":
#             "More nurses",
#         "More CNP staff / Default Healthsystem Function":
#             "More CNP",
#         "More Nurses by District / Default Healthsystem Function":
#             "More nurses by district",
#         "More CNP staff by District / Default Healthsystem Function":
#             "More CNP by district",
#
#         "Baseline Nurses / Improved Healthsystem Function":
#             "Baseline",
#         "Fewer Nurses / Improved Healthsystem Function":
#             "Fewer nurses",
#         "More Nurses / Improved Healthsystem Function":
#             "More nurses",
#         "More CNP staff / Improved Healthsystem Function":
#             "More CNP",
#         "More Nurses by District / Improved Healthsystem Function":
#             "More nurses by district",
#         "More CNP staff by District / Improved Healthsystem Function":
#             "More CNP by district",
#     }
#
#     color_map = {
#         "Baseline Nurses / Default Healthsystem Function": "black",
#         "Fewer Nurses / Default Healthsystem Function": "indianred",
#         "More Nurses / Default Healthsystem Function": "steelblue",
#         "More CNP staff / Default Healthsystem Function": "darkgreen",
#         "More Nurses by District / Default Healthsystem Function":
#             "mediumpurple",
#         "More CNP staff by District / Default Healthsystem Function":
#             "orange",
#
#         "Baseline Nurses / Improved Healthsystem Function": "black",
#         "Fewer Nurses / Improved Healthsystem Function": "indianred",
#         "More Nurses / Improved Healthsystem Function": "steelblue",
#         "More CNP staff / Improved Healthsystem Function": "darkgreen",
#         "More Nurses by District / Improved Healthsystem Function":
#             "mediumpurple",
#         "More CNP staff by District / Improved Healthsystem Function":
#             "orange",
#     }
#
#     for scenario in scenarios:
#         mean = data[scenario]["mean"]
#         color = color_map.get(scenario, "gray")
#         ax.plot(
#             mean.index,
#             mean.values,
#             linewidth=2,
#             color=color,
#             label=label_map.get(
#                 scenario,
#                 scenario,
#             ),
#         )
#
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Number of treatments")
#     ax.set_title(title)
#     ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     return fig


def plot_percent_service_volume_change(
    summarized_percent_change,
    title,
):
    fig, ax = plt.subplots(figsize=(10, 6))

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

    color_map = {
        "Baseline Nurses / Default Healthsystem Function": "black",
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",

        "Baseline Nurses / Improved Healthsystem Function": "black",
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    # Get scenario names
    scenarios = (
        summarized_percent_change.columns
        .get_level_values(0)
        .unique()
    )

    x = np.arange(len(scenarios))

    for i, scenario in enumerate(scenarios):

        # Extract the single summary value for this scenario.
        # .iloc[0] ensures that mean, lower and upper are scalars.
        mean = float(
            summarized_percent_change[scenario]["mean"].iloc[0]
        ) * 100

        lower = float(
            summarized_percent_change[scenario]["lower"].iloc[0]
        ) * 100

        upper = float(
            summarized_percent_change[scenario]["upper"].iloc[0]
        ) * 100

        ax.bar(
            x[i],
            mean,
            color=color_map.get(
                scenario,
                "gray",
            ),
            label=label_map.get(
                scenario,
                scenario,
            ),
        )

        # Calculate asymmetric error bars as scalars.
        lower_error = mean - lower
        upper_error = upper - mean

        ax.errorbar(
            x[i],
            mean,
            yerr=[[lower_error], [upper_error]],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1,)
    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            label_map.get(scenario, scenario)
            for scenario in scenarios
        ],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Nurse staffing scenario")
    ax.set_ylabel("% change in total service volume")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3,)
    fig.tight_layout()
    return fig


def plot_percent_service_area_volume_change(
    summarized_percent_change,
    scenarios,
    title,
):
    """
    Plot percentage change in service volume by service area
    relative to baseline for 2027-2034.

    Values are calculated using run-to-run comparisons and
    summarized across simulation draws.
    """

    label_map = {
        "Baseline Nurses / Default Healthsystem Function":
            "Baseline",
        "Fewer Nurses / Default Healthsystem Function":
            "Fewer nurses",
        "More Nurses / Default Healthsystem Function":
            "More nurses",
        "More CNP staff / Default Healthsystem Function":
            "More CNP",
        "More Nurses by District / Default Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function":
            "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function":
            "Baseline",
        "Fewer Nurses / Improved Healthsystem Function":
            "Fewer nurses",
        "More Nurses / Improved Healthsystem Function":
            "More nurses",
        "More CNP staff / Improved Healthsystem Function":
            "More CNP",
        "More Nurses by District / Improved Healthsystem Function":
            "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function":
            "More CNP by district",
    }

    color_map = {
        "Baseline Nurses / Default Healthsystem Function": "black",
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function":
            "orange",

        "Baseline Nurses / Improved Healthsystem Function": "black",
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function":
            "orange",
    }

    # Get service areas
    service_areas = summarized_percent_change.index

    # Get scenario names
    scenarios = [
        scenario
        for scenario in scenarios
        if scenario in summarized_percent_change.columns
        .get_level_values(0)
        .unique()
    ]

    x = np.arange(len(service_areas))

    n_scenarios = len(scenarios)

    width = 0.8 / n_scenarios

    offsets = (
        np.arange(n_scenarios)
        - (n_scenarios - 1) / 2
    ) * width

    fig, ax = plt.subplots(
        figsize=(18, 8)
    )

    for i, scenario in enumerate(scenarios):
        mean = (summarized_percent_change[scenario]["mean"])
        lower = (summarized_percent_change[scenario]["lower"])
        upper = (summarized_percent_change[scenario]["upper"])
        xpos = x + offsets[i]

        ax.bar(
            xpos,
            mean.values,
            width=width,
            color=color_map.get(
                scenario,
                "gray",
            ),
            label=label_map.get(
                scenario,
                scenario,
            ),
        )

        ax.errorbar(
            xpos,
            mean.values,
            yerr=[
                mean.values - lower.values,
                upper.values - mean.values,
            ],
            fmt="none",
            ecolor="black",
            capsize=2,
            lw=1,
        )

    ax.axhline(0, color="black", linestyle="--",linewidth=1,)
    ax.set_xlabel("Service area")
    ax.set_ylabel("% change in service volume\n(2027–2034)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(service_areas, rotation=45, ha="right",)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,)
    ax.grid(axis="y", alpha=0.3,)
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

    # TREATMENT-SPECIFIC SERVICE VOLUMES

    annual_treatment_volumes = extract_annual_treatment_volumes(
        results_folder
    )

    annual_treatment_volumes = set_param_names_as_column_index_level_0(
        annual_treatment_volumes,
        param_names,
    )

    # Service-area treatment volumes 2027-2034
    service_area_treatment_volumes = (
        aggregate_treatment_volumes_by_service_area(
            annual_treatment_volumes,
            comparison_years=range(2027, 2035),
        )
    )

    summarized_service_area_treatment_volumes = summarize(
        service_area_treatment_volumes
    )

    # Service-area % change vs baseline
    percent_service_area_volume_change_default = (
        calculate_service_area_volume_percent_change(
            service_area_treatment_volumes[
                default_hs_scenarios
            ],
            baseline_scenario=baseline_scenario,
        )
    )

    percent_service_area_volume_change_improved = (
        calculate_service_area_volume_percent_change(
            service_area_treatment_volumes[
                improved_hs_scenarios
            ],
            baseline_scenario=baseline_improved_scenario,
        )
    )

    # Aggregate treatment volumes into broader disease/service areas
    annual_treatment_volumes_by_service_area = (
        aggregate_treatment_volumes_by_service_area(
            annual_treatment_volumes
        )
    )

    treatment_volume_table = make_treatment_volume_table(
        annual_treatment_volumes,
        comparison_years=range(2027, 2035),
    )

    print("\nTreatment volume table 2027-2034")
    print(
        treatment_volume_table.head(30)
    )

    # Treatment-specific % change VS baseline
    percent_treatment_volume_change_default = (
        calculate_treatment_volume_percent_change(
            annual_treatment_volumes[
                default_hs_scenarios
            ],
            baseline_scenario=baseline_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    percent_treatment_volume_change_improved = (
        calculate_treatment_volume_percent_change(
            annual_treatment_volumes[
                improved_hs_scenarios
            ],
            baseline_scenario=baseline_improved_scenario,
            comparison_years=range(2027, 2035),
        )
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

    # percent_change_default = calculate_percent_service_volume_change(
    #     annual_service_volumes[default_hs_scenarios],
    #     baseline_scenario,
    # )

    # percent_change_default = calculate_percent_service_volume_change(
    #     annual_service_volumes[default_hs_scenarios],
    #     baseline_scenario=baseline_scenario,
    #     comparison_years=range(2027, 2035),
    # )

    # percent_change_improved = calculate_percent_service_volume_change(
    #     annual_service_volumes[improved_hs_scenarios],
    #     baseline_improved_scenario,
    # )

    # percent_change_improved = calculate_percent_service_volume_change(
    #     annual_service_volumes[improved_hs_scenarios],
    #     baseline_scenario=baseline_improved_scenario,
    #     comparison_years=range(2027, 2035),
    # )

    percent_change_default = calculate_percent_service_volume_change(
        annual_service_volumes[default_hs_scenarios],
        baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    percent_change_improved = calculate_percent_service_volume_change(
        annual_service_volumes[improved_hs_scenarios],
        baseline_improved_scenario,
        comparison_years=range(2027, 2035),
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
        "% Change in Total Service Volume vs Baseline\n(2027–2034, Default Healthsystem Function)",
    )

    fig_improved_percent = plot_percent_service_volume_change(
        percent_change_improved,
        "% Change in Total Service Volume vs Baseline\n(2027–2034, Improved Healthsystem Function)",
    )

    fig_service_area_default = (
        plot_percent_service_area_volume_change(
            percent_service_area_volume_change_default,
            default_hs_scenarios[1:],
            "% Change in Service Volume by Area vs Baseline\n"
            "(2027–2034, Default Healthsystem Function)",
        )
    )

    fig_service_area_improved = (
        plot_percent_service_area_volume_change(
            percent_service_area_volume_change_improved,
            improved_hs_scenarios[1:],
            "% Change in Service Volume by Area vs Baseline\n"
            "(2027–2034, Improved Healthsystem Function)",
        )
    )

    if args.save_figures:
        output_folder = results_folder / "service_volume_plots"

        output_folder.mkdir(exist_ok=True)

        fig_default.savefig(
            output_folder / "annual_service_volumes_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_improved.savefig(
            output_folder / "annual_service_volumes_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_default_percent.savefig(
            output_folder / "percent_change_service_volumes_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_improved_percent.savefig(
            output_folder / "percent_change_service_volumes_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_service_area_default.savefig(
            output_folder / "percent_change_service_volume_by_area_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_service_area_improved.savefig(
            output_folder / "percent_change_service_volume_by_area_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )
