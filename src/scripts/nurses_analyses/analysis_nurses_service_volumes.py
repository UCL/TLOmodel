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


def extract_annual_nurse_time_use(results_folder):
    """
    Extract annual nurse time used to deliver appointments.
    Nurse time is calculated as:
        number of appointments delivered
        * nursing time requirement per appointment

    Appointment counts are taken from: Number_By_Appt_Type_Code_And_Level

    Calculation is performed separately by facility level and
    appointment type so that the correct nursing time requirement
    can be applied.
    """
    repo_root = Path(__file__).resolve().parents[3]

    appt_time_file = (
        repo_root
        / "resources"
        / "healthsystem"
        / "human_resources"
        / "definitions"
        / "ResourceFile_Appt_Time_Table.csv"
    )

    appt_time_table = pd.read_csv(appt_time_file)

    # Keep only time requirements for Nursing and Midwifery.
    nurse_time_table = appt_time_table[
        appt_time_table["Officer_Category"]
        == "Nursing_and_Midwifery"
    ].copy()

    # Create a lookup: (appointment type, facility level) -> nurse minutes
    nurse_time_lookup = dict(
        zip(
            zip(
                nurse_time_table["Appt_Type_Code"],
                nurse_time_table["Facility_Level"].astype(str),
            ),
            nurse_time_table["Time_Taken_Mins"],
        )
    )

    def get_nurse_time_yearly(df: pd.DataFrame):
        df = df.copy()
        if "year" not in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year

        yearly_nurse_time = {}
        for _, row in df.iterrows():
            year = row["year"]
            appointment_counts_by_level = (row["Number_By_Appt_Type_Code_And_Level"])
            total_nurse_minutes = 0.0

            if not isinstance(appointment_counts_by_level, dict,):
                continue

            for facility_level, appointment_counts in (
                appointment_counts_by_level.items()
            ):
                if not isinstance(appointment_counts, dict):
                    continue

                facility_level = str(facility_level)
                for appt_type, number_of_appointments in (
                    appointment_counts.items()
                ):
                    nurse_minutes_per_appointment = (
                        nurse_time_lookup.get(
                            (
                                appt_type,
                                facility_level,
                            ),
                            0.0,
                        )
                    )

                    total_nurse_minutes += (
                        number_of_appointments
                        * nurse_minutes_per_appointment
                    )

            yearly_nurse_time[year] = (
                yearly_nurse_time.get(year, 0.0)
                + total_nurse_minutes
            )

        return pd.Series(
            yearly_nurse_time,
            dtype=float,
        )

    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=get_nurse_time_yearly,
        do_scaling=True,
    )


def aggregate_treatment_volumes_by_service_area(
    annual_treatment_volumes,
    comparison_years=range(2027, 2035),
):
    """
    Aggregate treatment volumes into the general health areas
    used in the nurse-shortages DALYs cause plots.
    """

    years = annual_treatment_volumes.index.get_level_values("year").astype(int)

    year_mask = np.isin(years, list(comparison_years))
    selected = annual_treatment_volumes.loc[year_mask].copy()

    treatment_ids = (
        selected.index.get_level_values("treatment_id")
        .astype(str)
    )

    def map_to_health_area(treatment_id):
        treatment = treatment_id.lower()

        # HIV/AIDS
        if treatment.startswith("hiv"):
            return "HIV/AIDS"

        # TB
        elif treatment.startswith("tb"):
            return "TB"

        # Malaria
        elif treatment.startswith("malaria"):
            return "Malaria"

        # RMNCH
        elif treatment.startswith((
            "alri",
            "antenatalcare",
            "contraception",
            "deliverycare",
            "diarrhoea",
            "epi",
            "measles",
            "postnatalcare",
            "schisto",
            "undernutrition",
        )):
            return "RMNCH"

        # NCDs
        elif treatment.startswith((
            "bladdercancer",
            "breastcancer",
            "cardiometabolicdisorders",
            "copd",
            "depression",
            "epilepsy",
            "oesophagealcancer",
            "otheradultcancer",
            "prostatecancer",
        )):
            return "NCDs"

        # Transport injuries
        elif treatment.startswith((
            "transport",
            "roadtraffic",
        )):
            return "Transport Injuries"

        return "Other"

    health_areas = [
        map_to_health_area(treatment_id)
        for treatment_id in treatment_ids
    ]

    selected.index = pd.MultiIndex.from_arrays(
        [
            selected.index.get_level_values("year"),
            health_areas,
        ],
        names=["year", "service_area"],
    )

    health_area_volumes = selected.groupby(level="service_area").sum()
    return health_area_volumes


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


def calculate_absolute_service_volume_change(
    annual_service_volumes,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate the absolute change in total service volumes
    relative to baseline over 2027-2034.
    """
    years = annual_service_volumes.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))
    selected = annual_service_volumes.loc[year_mask]

    total_service_volumes = selected.sum(axis=0).to_frame().T

    absolute_diff = find_difference_relative_to_comparison_series_dataframe(
        total_service_volumes,
        comparison=baseline_scenario,
        scaled=False,
    )
    return summarize(absolute_diff)


def calculate_absolute_service_area_volume_change(
    service_area_volumes,
    baseline_scenario,
):
    """
    Calculate absolute changes in service volumes by health area
    relative to baseline.
    """
    absolute_diff = find_difference_relative_to_comparison_series_dataframe(
        service_area_volumes,
        comparison=baseline_scenario,
        scaled=False,
    )
    return summarize(absolute_diff)


def calculate_absolute_nurse_time_change(
    annual_nurse_time,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate absolute change in nurse time use relative to baseline
    over 2027-2034.
    """
    years = annual_nurse_time.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))
    selected = annual_nurse_time.loc[year_mask]

    total_nurse_time = selected.sum(axis=0).to_frame().T

    absolute_diff = find_difference_relative_to_comparison_series_dataframe(
        total_nurse_time,
        comparison=baseline_scenario,
        scaled=False,
    )
    return summarize(absolute_diff)


def calculate_nurse_time_percent_change(
    annual_nurse_time,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate percentage change in nurse time use relative to
    the baseline scenario.
    Nurse time is first summed over 2027-2034 for each
    scenario and simulation run.
    Percentage change is then calculated run-to-run:
        ((alternative / baseline) - 1) * 100
    """

    # Select comparison years.
    years = annual_nurse_time.index.astype(int)

    year_mask = np.isin(years, list(comparison_years),)
    selected = annual_nurse_time.loc[year_mask]

    # Total nurse time over 2027-2034 for each scenario/run.
    total_nurse_time = selected.sum(axis=0).to_frame().T

    pct_diff = (
        100.0
        * find_difference_relative_to_comparison_series_dataframe(
            total_nurse_time,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    return summarized


def plot_percent_service_volume_change(
    summarized_percent_change,
    scenarios,
    title,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    label_map = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",

        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
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

    scenarios = [
        scenario
        for scenario in scenarios
        if scenario in summarized_percent_change.columns
        .get_level_values(0)
        .unique()
    ]

    x = np.arange(len(scenarios))

    for i, scenario in enumerate(scenarios):

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
        rotation=0,
        ha="center",
    )

    # ax.set_xlabel("Nurse staffing scenario")
    ax.set_ylabel("% Service volume change")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def annotate_averted_bars(
    ax,
    bars,
    percent_df,
    upper_values,
    lower_values,
    horizontal=False,
    fontsize=7,
):
    """
    Annotate absolute-change bars with percentage change and its 95% CI.
    Same format as in the DALYs/deaths averted plots.
    percent_df: rows in the same order as the bars, with mean/lower/upper
    already in percent.
    """

    for i, (bar, (_, row)) in enumerate(
        zip(bars, percent_df.iterrows())
    ):

        pct_mean = row["mean"]
        pct_lower = row["lower"]
        pct_upper = row["upper"]

        if horizontal:
            annotation = (
                f"{pct_mean:+.1f}% "
                f"(95% CI: {pct_lower:+.1f} to {pct_upper:+.1f})"
            )
        else:
            annotation = (
                f"{pct_mean:+.1f}%\n"
                f"95% CI: ({pct_lower:+.1f}%, {pct_upper:+.1f}%)"
            )

        # Place the label just beyond the end of the error bar, on the side
        # the bar points to: upper limit for positive bars, lower limit for
        # negative bars. The bar's own length gives its sign.
        bar_value = bar.get_width() if horizontal else bar.get_height()
        positive = bar_value >= 0
        limit = upper_values[i] if positive else lower_values[i]
        offset = 6 if positive else -6

        if horizontal:
            y_position = bar.get_y() + bar.get_height() / 2

            ax.annotate(
                annotation,
                xy=(limit, y_position),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left" if positive else "right",
                va="center",
                fontsize=fontsize,
                annotation_clip=False,
            )

        else:
            x_position = bar.get_x() + bar.get_width() / 2

            ax.annotate(
                annotation,
                xy=(x_position, limit),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if positive else "top",
                fontsize=fontsize,
                annotation_clip=False,
            )


def plot_absolute_change_with_percent_annotations(
    summarized_absolute_change,
    summarized_percent_change,
    scenarios,
    title,
    ylabel,
    percent_multiplier=100,
):
    """
    Plot absolute change relative to baseline, with percentage
    change and 95% CI shown as annotations.
    Same figure size and format as the DALYs/deaths averted plots.

    percent_multiplier: 100 when summarized_percent_change holds
    proportions (total service volume), 1 when it is already in
    percent (nurse time).
    """

    label_map = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function":
            "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function":
            "More CNP\nby district",

        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function":
            "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function":
            "More CNP\nby district",
    }

    color_map = {
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",

        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    available_scenarios = (
        summarized_absolute_change.columns
        .get_level_values(0)
        .unique()
    )

    scenarios = [
        scenario for scenario in scenarios
        if scenario in available_scenarios
    ]

    labels = [label_map.get(scenario, scenario) for scenario in scenarios]
    colors = [color_map.get(scenario, "gray") for scenario in scenarios]

    # Absolute change: bar heights and error bars
    means = np.array([float(summarized_absolute_change[s]["mean"].iloc[0]) for s in scenarios])
    lowers = np.array([float(summarized_absolute_change[s]["lower"].iloc[0]) for s in scenarios])
    uppers = np.array([float(summarized_absolute_change[s]["upper"].iloc[0]) for s in scenarios])

    # Percentage change: annotations
    percent_df = pd.DataFrame(
        {
            stat: [
                float(summarized_percent_change[s][stat].iloc[0]) * percent_multiplier
                for s in scenarios
            ]
            for stat in ["mean", "lower", "upper"]
        },
        index=scenarios,
    )

    yerr = np.vstack([means - lowers, uppers - means])

    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(labels, means, yerr=yerr, capsize=6, color=colors, width=0.55)

    annotate_averted_bars(
        ax,
        bars,
        percent_df,
        upper_values=uppers,
        lower_values=lowers,
    )

    ax.axhline(0, color="black", linewidth=1)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_absolute_service_area_volume_change(
    summarized_absolute_change,
    summarized_percent_change,
    scenarios,
    title,
):
    """
    Plot absolute change in service volume by general health area,
    with percentage change and 95% CI as annotations.
    """

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
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function":
            "orange",

        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function":
            "orange",
    }

    health_areas = sorted(summarized_absolute_change.index)

    available_scenarios = (
        summarized_absolute_change.columns
        .get_level_values(0)
        .unique()
    )

    scenarios = [
        scenario for scenario in scenarios
        if scenario in available_scenarios
    ]

    y = np.arange(len(health_areas))

    n_scenarios = len(scenarios)
    height = 0.8 / n_scenarios

    offsets = (
        np.arange(n_scenarios) - (n_scenarios - 1) / 2
    ) * height

    fig, ax = plt.subplots(figsize=(16, 9))

    for i, scenario in enumerate(scenarios):
        mean = summarized_absolute_change[scenario]["mean"].reindex(health_areas)
        lower = summarized_absolute_change[scenario]["lower"].reindex(health_areas)
        upper = summarized_absolute_change[scenario]["upper"].reindex(health_areas)

        # Percentage change is already in percent.
        percent_df = pd.DataFrame({
            stat: summarized_percent_change[scenario][stat].reindex(health_areas)
            for stat in ["mean", "lower", "upper"]
        })

        ypos = y + offsets[i]

        bars = ax.barh(
            ypos,
            mean.values,
            height=height,
            color=color_map.get(scenario, "gray"),
            label=label_map.get(scenario, scenario),
        )

        ax.errorbar(
            mean.values,
            ypos,
            xerr=[
                mean.values - lower.values,
                upper.values - mean.values,
            ],
            fmt="none",
            ecolor="black",
            capsize=2,
            lw=1,
        )

        # Annotate bars with percentage change and 95% CI
        annotate_averted_bars(
            ax,
            bars,
            percent_df,
            upper_values=upper.values,
            lower_values=lower.values,
            horizontal=True,
            fontsize=8,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(health_areas, color="black")
    ax.set_xlabel("Absolute change in service volumes")
    ax.set_ylabel("Health area")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis="x", alpha=0.3)
    ax.margins(x=0.40)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_nurse_time_use_percent_change(
    summarized_percent_change,
    scenarios,
    title,
):
    """
    Plot percentage change in nurse time use relative to baseline.
    """

    label_map = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",

        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
    }

    color_map = {
        "Fewer Nurses / Default Healthsystem Function":
            "indianred",
        "More Nurses / Default Healthsystem Function":
            "steelblue",
        "More CNP staff / Default Healthsystem Function":
            "darkgreen",
        "More Nurses by District / Default Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function":
            "orange",
        "Fewer Nurses / Improved Healthsystem Function":
            "indianred",
        "More Nurses / Improved Healthsystem Function":
            "steelblue",
        "More CNP staff / Improved Healthsystem Function":
            "darkgreen",
        "More Nurses by District / Improved Healthsystem Function":
            "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function":
            "orange",
    }

    # Keep only scenarios requested for this plot.
    scenarios = [
        scenario
        for scenario in scenarios
        if scenario
        in summarized_percent_change.columns
        .get_level_values(0)
        .unique()
    ]

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
        mean = float(summarized_percent_change[scenario]["mean"].iloc[0])
        lower = float(summarized_percent_change[scenario]["lower"].iloc[0])
        upper = float(summarized_percent_change[scenario]["upper"].iloc[0])

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

        lower_error = mean - lower
        upper_error = upper - mean

        ax.errorbar(
            x[i],
            mean,
            yerr=[
                [lower_error],
                [upper_error],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
            lw=1,
        )

    # Baseline = 0% change.
    ax.axhline(0, color="black", linestyle="--", linewidth=1,)
    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            label_map.get(
                scenario,
                scenario,
            )
            for scenario in scenarios
        ],
        rotation=0,
        ha="center",
    )

    # ax.set_xlabel("Nurse staffing scenario")
    ax.set_ylabel("% Nurse time use change")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3,)
    fig.tight_layout()
    return fig



def prepare_service_volume_plot_data_for_excel(
    summarized_data,
    value_multiplier=1.0,
    category_name="Measure",
    single_row_label="Total change (2027–2034)",
):
    """
    Prepare the numerical values used in a service-volume plot
    for export to Excel.

    Includes the mean, lower and upper bounds, and the sizes
    of the lower and upper error bars.

    value_multiplier=100 converts proportions to percentages.
    """
    rows = []

    # The columns are grouped by scenario, with statistics
    # such as mean, lower, and upper under each scenario.
    scenarios = summarized_data.columns.get_level_values(0).unique()

    for scenario in scenarios:
        scenario_data = summarized_data[scenario]

        for index_value, values in scenario_data.iterrows():
            mean = float(values["mean"]) * value_multiplier
            lower = float(values["lower"]) * value_multiplier
            upper = float(values["upper"]) * value_multiplier

            # For plots showing a single overall value, use a
            # descriptive label instead of an index such as 0.
            if len(summarized_data.index) == 1:
                category = single_row_label
            else:
                category = str(index_value)

            rows.append(
                {
                    "Scenario": scenario,
                    category_name: category,
                    "Mean (%)": mean,
                    "Lower bound (%)": lower,
                    "Upper bound (%)": upper,
                    "Lower error (percentage points)": mean - lower,
                    "Upper error (percentage points)": upper - mean,
                }
            )

    return pd.DataFrame(rows)


def combine_percent_and_absolute(
    percent_data,
    absolute_data,
    value_multiplier,
    category_name,
    single_row_label,
):
    """
    One table with the % columns (unchanged names) followed by the
    absolute columns, matched on scenario and measure/service area.
    """
    percent_df = prepare_service_volume_plot_data_for_excel(
        summarized_data=percent_data,
        value_multiplier=value_multiplier,
        category_name=category_name,
        single_row_label=single_row_label,
    )

    absolute_df = prepare_service_volume_plot_data_for_excel(
        summarized_data=absolute_data,
        value_multiplier=1,
        category_name=category_name,
        single_row_label=single_row_label,
    ).rename(
        columns={
            "Mean (%)": "Absolute mean",
            "Lower bound (%)": "Absolute lower bound",
            "Upper bound (%)": "Absolute upper bound",
            "Lower error (percentage points)": "Absolute lower error",
            "Upper error (percentage points)": "Absolute upper error",
        }
    )

    return percent_df.merge(
        absolute_df,
        on=["Scenario", category_name],
        how="left",
        validate="one_to_one",
    )


def export_service_volume_plot_data_to_excel(
    results_folder,
    plot_data,
):
    """
    Export the numerical data underlying the six selected plots
    to separate worksheets in one Excel workbook.

    plot_data: list of (sheet_name, percent_data, absolute_data,
    value_multiplier, category_name, single_row_label).
    value_multiplier applies to the percent data only.
    """
    output_file = results_folder / "nurses_service_volume_plot_data.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for (
            sheet_name,
            percent_data,
            absolute_data,
            multiplier,
            category_name,
            row_label,
        ) in plot_data:
            excel_df = combine_percent_and_absolute(
                percent_data,
                absolute_data,
                value_multiplier=multiplier,
                category_name=category_name,
                single_row_label=row_label,
            )

            excel_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            # Improve readability in Excel.
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(
                    max_length + 2, 45
                )

    print(f"Excel plot data saved to: {output_file}")



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

    # NURSE TIME USE
    annual_nurse_time = extract_annual_nurse_time_use(
        results_folder
    )

    annual_nurse_time = set_param_names_as_column_index_level_0(
        annual_nurse_time,
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

    # Absolute service-volume change by general health area
    absolute_service_area_volume_change_default = (
        calculate_absolute_service_area_volume_change(
            service_area_treatment_volumes[default_hs_scenarios],
            baseline_scenario=baseline_scenario,
        )
    )

    absolute_service_area_volume_change_improved = (
        calculate_absolute_service_area_volume_change(
            service_area_treatment_volumes[improved_hs_scenarios],
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

    # Absolute service-volume change relative to baseline
    absolute_service_volume_change_default = (
        calculate_absolute_service_volume_change(
            annual_service_volumes[default_hs_scenarios],
            baseline_scenario=baseline_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    absolute_service_volume_change_improved = (
        calculate_absolute_service_volume_change(
            annual_service_volumes[improved_hs_scenarios],
            baseline_scenario=baseline_improved_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    # Nurse time use relative to baseline
    nurse_time_change_default = (
        calculate_nurse_time_percent_change(
            annual_nurse_time[
                default_hs_scenarios
            ],
            baseline_scenario=baseline_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    nurse_time_change_improved = (
        calculate_nurse_time_percent_change(
            annual_nurse_time[
                improved_hs_scenarios
            ],
            baseline_scenario=baseline_improved_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    # Absolute nurse time-use change relative to baseline
    absolute_nurse_time_change_default = (
        calculate_absolute_nurse_time_change(
            annual_nurse_time[default_hs_scenarios],
            baseline_scenario=baseline_scenario,
            comparison_years=range(2027, 2035),
        )
    )

    absolute_nurse_time_change_improved = (
        calculate_absolute_nurse_time_change(
            annual_nurse_time[improved_hs_scenarios],
            baseline_scenario=baseline_improved_scenario,
            comparison_years=range(2027, 2035),
        )
    )


    # Export the numerical data underlying the six selected plots.
    # value_multiplier=100 converts the first two outputs from
    # proportions to percentages, matching their plotting functions.
    # The other four outputs are already expressed as percentages.

    plot_data_for_excel = [
        # 1–2: Overall service-volume percentage change
        (
            "ServiceVol_Default",
            percent_change_default,
            absolute_service_volume_change_default,
            100,
            "Measure",
            "Total service-volume change (2027–2034)",
        ),
        (
            "ServiceVol_Improved",
            percent_change_improved,
            absolute_service_volume_change_improved,
            100,
            "Measure",
            "Total service-volume change (2027–2034)",
        ),

        # 3–4: Percentage change by service area
        (
            "Area_Default",
            percent_service_area_volume_change_default,
            absolute_service_area_volume_change_default,
            1,
            "Service area",
            "Total change (2027–2034)",
        ),
        (
            "Area_Improved",
            percent_service_area_volume_change_improved,
            absolute_service_area_volume_change_improved,
            1,
            "Service area",
            "Total change (2027–2034)",
        ),

        # 5–6: Nurse time use relative to baseline
        (
            "NurseTime_Default",
            nurse_time_change_default,
            absolute_nurse_time_change_default,
            1,
            "Measure",
            "Total nurse-time change (2027–2034)",
        ),
        (
            "NurseTime_Improved",
            nurse_time_change_improved,
            absolute_nurse_time_change_improved,
            1,
            "Measure",
            "Total nurse-time change (2027–2034)",
        ),
    ]

    export_service_volume_plot_data_to_excel(
        results_folder=results_folder,
        plot_data=plot_data_for_excel,
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

    # Overall absolute service-volume changes
    fig_default_percent = plot_absolute_change_with_percent_annotations(
        absolute_service_volume_change_default,
        percent_change_default,
        default_hs_scenarios[1:],
        "",
        "Absolute change in service volumes",
    )

    fig_improved_percent = plot_absolute_change_with_percent_annotations(
        absolute_service_volume_change_improved,
        percent_change_improved,
        improved_hs_scenarios[1:],
        "",
        "Absolute change in service volumes",
    )

    # Absolute service-volume changes by general health area
    fig_service_area_default = plot_absolute_service_area_volume_change(
        absolute_service_area_volume_change_default,
        percent_service_area_volume_change_default,
        default_hs_scenarios[1:],
        "",
    )

    fig_service_area_improved = plot_absolute_service_area_volume_change(
        absolute_service_area_volume_change_improved,
        percent_service_area_volume_change_improved,
        improved_hs_scenarios[1:],
        "",
    )

    # Absolute nurse time-use changes
    fig_nurse_time_default = plot_absolute_change_with_percent_annotations(
        absolute_nurse_time_change_default,
        nurse_time_change_default,
        default_hs_scenarios[1:],
        "",
        "Absolute change in nurse time used (minutes)",
        percent_multiplier=1,  # already in percent
    )

    fig_nurse_time_improved = plot_absolute_change_with_percent_annotations(
        absolute_nurse_time_change_improved,
        nurse_time_change_improved,
        improved_hs_scenarios[1:],
        "",
        "Absolute change in nurse time used (minutes)",
        percent_multiplier=1,  # already in percent
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
            output_folder / "abs_and_pct_change_service_volumes_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_improved_percent.savefig(
            output_folder / "abs_and_pct_change_service_volumes_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_service_area_default.savefig(
            output_folder / "abs_and_pct_change_service_volume_by_area_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_service_area_improved.savefig(
            output_folder / "abs_and_pct_change_service_volume_by_area_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_nurse_time_default.savefig(
            output_folder
            / "nurse_time_use_relative_to_baseline_default.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        fig_nurse_time_improved.savefig(
            output_folder
            / "nurse_time_use_relative_to_baseline_improved.pdf",
            dpi=300,
            bbox_inches="tight",
        )
