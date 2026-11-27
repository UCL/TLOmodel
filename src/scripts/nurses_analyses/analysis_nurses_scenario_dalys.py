"""Plot DALYs and Deaths across nurse staffing scenarios.

This script produces two figures for the Nurse Shortages analysis:

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import compute_summary_statistics, extract_results, load_pickled_dataframes


def summarize(results):
    """Mean and 95% CI of the mean (mean ± 1.96 × SE), same layout as TLO's summarize."""
    return compute_summary_statistics(
        results, central_measure="mean", use_standard_error=True
    ).rename(columns={"central": "mean"}, level=1)


DALY_METADATA_COLUMNS = {"date", "year", "sex", "age_range", "li_wealth", "district_of_residence"}


# Group individual causes into broader health areas.
CAUSE_GROUP = {
    "AIDS": "HIV/AIDS",
    "TB (non-AIDS)": "TB (non-AIDS)",
    "Malaria": "Malaria",
    "Childhood Diarrhoea": "RMNCH",
    "Childhood Undernutrition": "RMNCH",
    "Congenital birth defects": "RMNCH",
    "Lower respiratory infections": "RMNCH",
    "Maternal Disorders": "RMNCH",
    "Measles": "RMNCH",
    "Neonatal Disorders": "RMNCH",
    "Schistosomiasis": "RMNCH",
    "COPD": "NCDs",
    "Cancer (Bladder)": "NCDs",
    "Cancer (Breast)": "NCDs",
    "Cancer (Cervix)": "RMNCH",
    "Cancer (Oesophagus)": "NCDs",
    "Cancer (Other)": "NCDs",
    "Cancer (Prostate)": "NCDs",
    "Depression / Self-harm": "NCDs",
    "Diabetes": "NCDs",
    "Epilepsy": "NCDs",
    "Heart Disease": "NCDs",
    "Kidney Disease": "NCDs",
    "Lower Back Pain": "NCDs",
    "Stroke": "NCDs",
    "Transport Injuries": "Transport Injuries",
    "Other": "Other",
}

CAUSE_GROUP_ORDER = [
    "HIV/AIDS",
    "TB (non-AIDS)",
    "Malaria",
    "RMNCH",
    "NCDs",
    "Transport Injuries",
    "Other",
]

CAUSE_GROUP_COLOR = {
    "HIV/AIDS": "deepskyblue",
    "TB (non-AIDS)": "mediumslateblue",
    "Malaria": "khaki",
    "RMNCH": "mediumaquamarine",
    "NCDs": "violet",
    "Transport Injuries": "lightsalmon",
    "Other": "dimgrey",
}


def prepare_plot_data_for_excel(data, category_name="Category"):
    """
    Prepare the mean and error-bar values used in a plot
    for export to Excel.
    For baseline-comparison plots, data is a DataFrame indexed
    by scenario.
    For cause and age-group plots, data is a dictionary of
    scenario names and DataFrames indexed by cause or age group.
    """
    if isinstance(data, pd.DataFrame):
        # Baseline-comparison plots
        result = data.copy()
        result.index.name = "Scenario"
        result = result.reset_index()

    else:
        # Cause and age-group plots
        frames = []

        for scenario, scenario_df in data.items():
            temp = scenario_df.copy()
            temp.index.name = category_name
            temp = temp.reset_index()
            temp.insert(0, "Scenario", scenario)
            frames.append(temp)

        result = pd.concat(frames, ignore_index=True)

    # Include the numerical size of each error bar.
    if {"mean", "lower", "upper"}.issubset(result.columns):
        result["Lower error"] = result["mean"] - result["lower"]
        result["Upper error"] = result["upper"] - result["mean"]

        # Arrange the columns in a useful order.
        result = result[
            [
                col for col in [
                    "Scenario",
                    category_name,
                    "mean",
                    "lower",
                    "upper",
                    "Lower error",
                    "Upper error",
                ]
                if col in result.columns
            ]
        ]

        result = result.rename(
            columns={
                "mean": "Mean",
                "lower": "Lower bound",
                "upper": "Upper bound",
            }
        )

    return result


def combine_percent_and_absolute(percent_data, absolute_data, category_name="Category"):
    """
    One table with the % columns (unchanged names) followed by the
    absolute columns, matched on scenario (and cause/age group).
    """
    percent_df = prepare_plot_data_for_excel(percent_data, category_name=category_name)
    absolute_df = prepare_plot_data_for_excel(absolute_data, category_name=category_name)

    keys = [col for col in ["Scenario", category_name] if col in percent_df.columns]
    absolute_df = absolute_df.rename(
        columns={
            col: f"Absolute {col[0].lower()}{col[1:]}"
            for col in absolute_df.columns if col not in keys
        }
    )

    return percent_df.merge(absolute_df, on=keys, how="left", validate="one_to_one")


def export_nurse_plot_data_to_excel(results_folder, plot_data):
    """
    Export the data underlying the selected nurse-scenario
    plots into separate worksheets in one Excel workbook.
    plot_data: list of (sheet_name, percent_data, absolute_data, category_name).
    """
    output_file = results_folder / "nurses_scenario_plot_data.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, percent_data, absolute_data, category_name in plot_data:
            excel_df = combine_percent_and_absolute(
                percent_data,
                absolute_data,
                category_name=category_name,
            )

            excel_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            # Make columns easier to read in Excel.
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


def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def extract_annual_dalys(results_folder):

    def get_num_dalys_yearly(df: pd.DataFrame) -> pd.Series:
        """Return total DALYs for each year."""

        # Add year if it isn't already present
        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)

        cause_cols = [
            c
            for c in df.columns
            if c not in DALY_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        yearly = (
            df.groupby("year")[cause_cols]
              .sum()
              .sum(axis=1)
        )

        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_num_dalys_yearly,
        do_scaling=True,
    )


# Extract annual Deaths
def extract_annual_deaths(results_folder):
    def get_num_deaths_yearly(df: pd.DataFrame) -> pd.Series:
        """Return total deaths for each year."""
        yearly = (
            df.assign(year=df["date"].dt.year)
            .groupby("year")["person_id"]
            .count()
        )
        return yearly

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_num_deaths_yearly,
        do_scaling=True,
    )


# Plot: Annual DALYs over time
def plot_annual_dalys(summarized_annual_dalys, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = summarized_annual_dalys.columns.get_level_values(0).unique()

    # Short labels for legend
    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP by district",
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

    for scenario in scenario_names:
        years = summarized_annual_dalys.index.astype(int)
        means = summarized_annual_dalys[(scenario, "mean")].values
        lowers = summarized_annual_dalys[(scenario, "lower")].values
        uppers = summarized_annual_dalys[(scenario, "upper")].values

        color = color_map.get(scenario, "gray")
        ax.plot(years, means, linewidth=2, color=color, label=label_map.get(scenario, scenario))
        ax.fill_between(years, lowers, uppers, color=color, alpha=0.2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual DALYs")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    ax.set_ylim(bottom=8e6)
    # ax.set_ylim(bottom=0.8)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    # fig.suptitle(title, fontsize=14)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, ax


# Plot: Annual Deaths over time
def plot_annual_deaths(summarized_annual_deaths, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = (
        summarized_annual_deaths.columns
        .get_level_values(0)
        .unique()
    )

    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",
        "More CNP staff / Default Healthsystem Function": "More CNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP by district",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
        "More CNP staff / Improved Healthsystem Function": "More CNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses by district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP by district",
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

    for scenario in scenario_names:
        years = summarized_annual_deaths.index.astype(int)
        means = summarized_annual_deaths[(scenario, "mean")].values
        lowers = summarized_annual_deaths[(scenario, "lower")].values

        uppers = summarized_annual_deaths[(scenario, "upper")].values

        color = color_map.get(scenario, "gray")
        ax.plot(years, means, linewidth=2, color=color, label=label_map.get(scenario, scenario))
        ax.fill_between(years, lowers, uppers, color=color, alpha=0.2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual deaths")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    # fig.suptitle(title, fontsize=14)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax


# Extract deaths by cause
def extract_deaths_by_cause(results_folder):
    def get_deaths_by_cause(df: pd.DataFrame) -> pd.Series:
        """
        Return deaths by cause aggregated across 2027–2034.
        """
        # Add year
        df = df.assign(year=df["date"].dt.year)
        # Restrict years
        df = df[df["year"].between(2027, 2034)]
        # Changed to "label" in order to capture group causes
        # cause_col = "cause"
        cause_col = "label"
        deaths_by_cause = (df.groupby(cause_col)["person_id"].count())
        return deaths_by_cause

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_cause,
        do_scaling=True,
    )


# Extract deaths by age group
def extract_deaths_by_age_group(results_folder):

    def get_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return deaths by age group aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        # Create age groups
        age_bins = [
            0, 5, 10, 15, 20, 25, 30, 35,
            40, 45, 50, 55, 60, 65, 70,
            75, 80, np.inf
        ]

        age_labels = [
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+",
        ]

        df["age_group"] = pd.cut(
            df["age"],
            bins=age_bins,
            labels=age_labels,
            right=False,
        )
        # Aggregate deaths by age group
        deaths_by_age = (df.groupby("age_group")["person_id"].count())
        return deaths_by_age

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_age_group,
        do_scaling=True,
    )


# Extract DALYs by cause
def extract_dalys_by_cause(results_folder):
    def get_dalys_by_cause(df: pd.DataFrame) -> pd.Series:
        """
        Return DALYs by cause aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]
        # Removing metadata columns
        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
               and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Sum DALYs for each cause
        return df[cause_cols].sum()

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_cause,
        do_scaling=True,
    )


def aggregate_causes_by_health_area(data):
    """
    Aggregate individual causes into broader health areas.
    """

    data = data.copy()

    # Identify causes not explicitly included in the mapping
    unmapped_causes = [
        cause for cause in data.index
        if cause not in CAUSE_GROUP
    ]

    if unmapped_causes:
        print(
            "WARNING: The following causes were not explicitly "
            "mapped and will be assigned to Other:",
            unmapped_causes,
        )

    # Mapping individual causes to broader health areas
    data.index = data.index.map(
        lambda cause: CAUSE_GROUP.get(cause, "Other")
    )

    # Summing individual causes belonging to the same health area
    data = data.groupby(level=0).sum()

    # Keep only health areas that actually exist in the data,
    # while preserving the desired order
    available_groups = [
        group
        for group in CAUSE_GROUP_ORDER
        if group in data.index
    ]

    data = data.reindex(available_groups)
    return data



# Extract DALYs by age group
def extract_dalys_by_age_group(results_folder):

    def get_dalys_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return DALYs by age group aggregated across 2027–2034.
        """
        df = df.assign(year=df["date"].dt.year)
        df = df[df["year"].between(2027, 2034)]

        cause_cols = [
            c for c in df.columns
            if c not in DALY_METADATA_COLUMNS
               and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Sum DALYs across causes first
        df["total_dalys"] = df[cause_cols].sum(axis=1)
        # Aggregating by age group
        dalys_by_age = (
            df.groupby("age_range")["total_dalys"]
            .sum()
        )
        return dalys_by_age

    return extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_dalys_by_age_group,
        do_scaling=True,
    )


# Plot: Percent DALYs averted relative to baseline (2027–2034)
def calculate_percent_dalys_averted(
    annual_dalys,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate % DALYs averted using run-to-run differences.
    """
    years = annual_dalys.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_dalys = annual_dalys.loc[year_mask]
    annual_dalys_agg = annual_dalys.sum(axis=0)

    pct_diff = pd.DataFrame(
        -100.0
        * find_difference_relative_to_comparison_series(
            annual_dalys_agg,
            comparison=baseline_scenario,
            scaled=True,
        )
    ).T

    summarized = summarize(pct_diff)
    results = {}

    scenario_names = (
        summarized.columns
        .get_level_values(0)
        .unique()
    )

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T


def calculate_percent_deaths_averted(
    annual_deaths,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate % deaths averted using run-to-run differences.
    """
    years = annual_deaths.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_deaths = annual_deaths.loc[year_mask]
    annual_deaths_agg = annual_deaths.sum(axis=0)

    pct_diff = pd.DataFrame(
        -100.0
        * find_difference_relative_to_comparison_series(
            annual_deaths_agg,
            comparison=baseline_scenario,
            scaled=True,
        )
    ).T

    summarized = summarize(pct_diff)
    results = {}

    scenario_names = (
        summarized.columns
        .get_level_values(0)
        .unique()
    )

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T



def calculate_dalys_averted(
    annual_dalys,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate absolute DALYs averted using run-to-run differences.
    """
    years = annual_dalys.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_dalys = annual_dalys.loc[year_mask]
    annual_dalys_agg = annual_dalys.sum(axis=0)

    dalys_diff = pd.DataFrame(
        -1.0
        * find_difference_relative_to_comparison_series(
            annual_dalys_agg,
            comparison=baseline_scenario,
            scaled=False,
        )
    ).T

    summarized = summarize(dalys_diff)

    results = {}
    scenario_names = summarized.columns.get_level_values(0).unique()

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T


def calculate_deaths_averted(
    annual_deaths,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate absolute deaths averted using run-to-run differences.
    """
    years = annual_deaths.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    annual_deaths = annual_deaths.loc[year_mask]
    annual_deaths_agg = annual_deaths.sum(axis=0)

    deaths_diff = pd.DataFrame(
        -1.0
        * find_difference_relative_to_comparison_series(
            annual_deaths_agg,
            comparison=baseline_scenario,
            scaled=False,
        )
    ).T

    summarized = summarize(deaths_diff)

    results = {}
    scenario_names = summarized.columns.get_level_values(0).unique()

    for scenario in scenario_names:
        results[scenario] = {
            "mean": summarized[(scenario, "mean")].iloc[0],
            "lower": summarized[(scenario, "lower")].iloc[0],
            "upper": summarized[(scenario, "upper")].iloc[0],
        }

    return pd.DataFrame(results).T


# Calculate % deaths averted by cause
def calculate_percent_deaths_averted_by_cause(
    deaths_by_cause,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            deaths_by_cause,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


def calculate_percent_dalys_averted_by_cause(
    dalys_by_cause,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            dalys_by_cause,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


# Calculate % DALYs averted by age group
def calculate_percent_dalys_averted_by_age_group(
    dalys_by_age_group,
    baseline_scenario,
):
    """
    Run-level comparison first,
    then summarize.
    """

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            dalys_by_age_group,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


# Calculate % deaths averted by age group
def calculate_percent_deaths_averted_by_age_group(
    deaths_by_age_group,
    baseline_scenario,
):

    pct_diff = (
        -100.0
        * find_difference_relative_to_comparison_series_dataframe(
            deaths_by_age_group,
            comparison=baseline_scenario,
            scaled=True,
        )
    )

    summarized = summarize(pct_diff)
    results = {}
    scenario_names = (summarized.columns.get_level_values(0).unique())

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


def calculate_absolute_averted_by_category(
    data,
    baseline_scenario,
):
    """
    Calculate absolute DALYs/deaths averted by cause or age group.
    Uses run-to-run differences before summarising.
    """

    absolute_diff = (
        -1.0
        * find_difference_relative_to_comparison_series_dataframe(
            data,
            comparison=baseline_scenario,
            scaled=False,
        )
    )

    summarized = summarize(absolute_diff)

    results = {}
    scenario_names = summarized.columns.get_level_values(0).unique()

    for scenario in scenario_names:
        results[scenario] = pd.DataFrame({
            "mean": summarized[(scenario, "mean")],
            "lower": summarized[(scenario, "lower")],
            "upper": summarized[(scenario, "upper")],
        })

    return results


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
    Annotate absolute DALYs/deaths averted bars with
    percentage change and its 95% CI.
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

        # Position at the upper/lower absolute confidence limit.
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


def plot_percent_dalys_averted_comparison(default_absolute, default_percent, improved_absolute, improved_percent):
    # Default Healthsystem
    ordered_scenarios_default = [
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    label_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",
    }

    color_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",
    }

    labels_default = [label_map_default[s] for s in ordered_scenarios_default]
    means_default = default_absolute.loc[ordered_scenarios_default, "mean"].values
    lowers_default = default_absolute.loc[ordered_scenarios_default, "lower"].values
    uppers_default = default_absolute.loc[ordered_scenarios_default, "upper"].values

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default,])

    colors_default = [color_map_default[s] for s in ordered_scenarios_default]

    fig_default, ax_default = plt.subplots(figsize=(7, 6))

    ax_default.bar(labels_default, means_default, yerr=yerr_default, capsize=6, color=colors_default, width=0.55)

    annotate_averted_bars(
        ax_default,
        ax_default.patches,
        default_percent.loc[ordered_scenarios_default],
        upper_values=uppers_default,
        lower_values=lowers_default,
    )

    ax_default.axhline(0, color="black", linewidth=1)
    # ax_default.set_title("Default Healthsystem")
    ax_default.set_ylabel("DALYs averted")
    ax_default.margins(y=0.25)
    ax_default.grid(axis="y", alpha=0.3)

    # fig_default.suptitle(
    #     "% DALYs averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_default.tight_layout()


    # Improved Healthsystem
    ordered_scenarios_improved = [
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    label_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
    }

    color_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    labels_improved = [label_map_improved[s] for s in ordered_scenarios_improved]
    means_improved = improved_absolute.loc[ordered_scenarios_improved, "mean"].values
    lowers_improved = improved_absolute.loc[ordered_scenarios_improved, "lower"].values
    uppers_improved = improved_absolute.loc[ordered_scenarios_improved, "upper"].values

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved,])
    colors_improved = [color_map_improved[s] for s in ordered_scenarios_improved]

    fig_improved, ax_improved = plt.subplots(figsize=(7, 6))

    ax_improved.bar(
        labels_improved,
        means_improved,
        yerr=yerr_improved,
        capsize=6,
        color=colors_improved,
        width=0.55,
    )

    annotate_averted_bars(
        ax_improved,
        ax_improved.patches,
        improved_percent.loc[ordered_scenarios_improved],
        upper_values=uppers_improved,
        lower_values=lowers_improved,
    )

    ax_improved.axhline(0, color="black", linewidth=1)
    # ax_improved.set_title("Improved Healthsystem")
    ax_improved.set_ylabel("DALYs averted")
    ax_improved.margins(y=0.25)
    ax_improved.grid(axis="y", alpha=0.3)
    # fig_improved.suptitle(
    #     "% DALYs averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_improved.tight_layout()
    return fig_default, ax_default, fig_improved, ax_improved


def plot_percent_deaths_averted_comparison(default_absolute, default_percent, improved_absolute, improved_percent):
    # Default Healthsystem
    ordered_scenarios_default = [
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
        "More CNP staff / Default Healthsystem Function",
        "More Nurses by District / Default Healthsystem Function",
        "More CNP staff by District / Default Healthsystem Function",
    ]

    label_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Default Healthsystem Function": "More\nnurses",
        "More CNP staff / Default Healthsystem Function": "More\nCNP",
        "More Nurses by District / Default Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Default Healthsystem Function": "More CNP\nby district",
    }

    color_map_default = {
        "Fewer Nurses / Default Healthsystem Function": "indianred",
        "More Nurses / Default Healthsystem Function": "steelblue",
        "More CNP staff / Default Healthsystem Function": "darkgreen",
        "More Nurses by District / Default Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Default Healthsystem Function": "orange",
    }

    labels_default = [label_map_default[s] for s in ordered_scenarios_default]
    means_default = default_absolute.loc[ordered_scenarios_default, "mean"].values
    lowers_default = default_absolute.loc[ordered_scenarios_default, "lower"].values
    uppers_default = default_absolute.loc[ordered_scenarios_default, "upper"].values

    yerr_default = np.vstack([means_default - lowers_default, uppers_default - means_default,])
    colors_default = [color_map_default[s] for s in ordered_scenarios_default]

    fig_default, ax_default = plt.subplots(figsize=(7, 6))

    ax_default.bar(labels_default, means_default, yerr=yerr_default, capsize=6, color=colors_default, width=0.55)
    annotate_averted_bars(
        ax_default,
        ax_default.patches,
        default_percent.loc[ordered_scenarios_default],
        upper_values=uppers_default,
        lower_values=lowers_default,
    )

    ax_default.axhline(0, color="black", linewidth=1)
    # ax_default.set_title("Default Healthsystem")
    ax_default.set_ylabel("Deaths averted")
    ax_default.margins(y=0.25)
    ax_default.grid(axis="y", alpha=0.3)

    # fig_default.suptitle(
    #     "% deaths averted relative to baseline (2027–2034)",
    #     fontsize=14,
    # )
    fig_default.tight_layout()


    # Improved Healthsystem
    ordered_scenarios_improved = [
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
        "More CNP staff / Improved Healthsystem Function",
        "More Nurses by District / Improved Healthsystem Function",
        "More CNP staff by District / Improved Healthsystem Function",
    ]

    label_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "Fewer\nnurses",
        "More Nurses / Improved Healthsystem Function": "More\nnurses",
        "More CNP staff / Improved Healthsystem Function": "More\nCNP",
        "More Nurses by District / Improved Healthsystem Function": "More nurses\nby district",
        "More CNP staff by District / Improved Healthsystem Function": "More CNP\nby district",
    }

    color_map_improved = {
        "Fewer Nurses / Improved Healthsystem Function": "indianred",
        "More Nurses / Improved Healthsystem Function": "steelblue",
        "More CNP staff / Improved Healthsystem Function": "darkgreen",
        "More Nurses by District / Improved Healthsystem Function": "mediumpurple",
        "More CNP staff by District / Improved Healthsystem Function": "orange",
    }

    labels_improved = [label_map_improved[s] for s in ordered_scenarios_improved]
    means_improved = improved_absolute.loc[ordered_scenarios_improved, "mean"].values
    lowers_improved = improved_absolute.loc[ordered_scenarios_improved, "lower"].values
    uppers_improved = improved_absolute.loc[ordered_scenarios_improved, "upper"].values

    yerr_improved = np.vstack([means_improved - lowers_improved, uppers_improved - means_improved,])
    colors_improved = [color_map_improved[s] for s in ordered_scenarios_improved]

    fig_improved, ax_improved = plt.subplots(figsize=(7, 6))

    ax_improved.bar(labels_improved, means_improved, yerr=yerr_improved, capsize=6, color=colors_improved, width=0.55)
    annotate_averted_bars(
        ax_improved,
        ax_improved.patches,
        improved_percent.loc[ordered_scenarios_improved],
        upper_values=uppers_improved,
        lower_values=lowers_improved,
    )

    ax_improved.axhline(0, color="black", linewidth=1)
    ax_improved.set_title("Improved Healthsystem")
    ax_improved.set_ylabel("Deaths averted")
    ax_improved.margins(y=0.25)
    ax_improved.grid(axis="y", alpha=0.3)

    fig_improved.suptitle(
        "% deaths averted relative to baseline (2027–2034)",
        fontsize=14,
    )
    fig_improved.tight_layout()
    return fig_default, ax_default, fig_improved, ax_improved


# Shared plotting function for percentage DALYs/deaths averted by health area
def _plot_percent_averted_by_cause(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
    outcome_label,
):
    """
    Plot percentage DALYs or deaths averted by health area
    """

    # Health areas available in the results.
    # health_areas = [
    #     group
    #     for group in CAUSE_GROUP_ORDER
    #     if group in default_df[
    #         "Fewer Nurses / Default Healthsystem Function"
    #     ].index
    #     and group in improved_df[
    #         "Fewer Nurses / Improved Healthsystem Function"
    #     ].index
    # ]

    health_areas = sorted(
        [
            area
            for area in CAUSE_GROUP_ORDER
            if area != "Other"
               and area in default_absolute[
                   "Fewer Nurses / Default Healthsystem Function"
               ].index
               and area in improved_absolute[
                   "Fewer Nurses / Improved Healthsystem Function"
               ].index
        ],
        key=str.lower,
    )

    # Scenario order and colours.
    scenarios = [
        ("Fewer nurses", "Fewer Nurses", "indianred"),
        ("More nurses", "More Nurses", "steelblue"),
        ("More CNP", "More CNP staff", "darkgreen"),
        ("More nurses by district", "More Nurses by District", "mediumpurple"),
        ("More CNP by district", "More CNP staff by District", "orange"),
    ]

    # Font sizes, set once so labels, ticks, annotations and legend
    # stay consistent with each other.
    axis_label_fontsize = 16    # x- and y-axis titles
    area_tick_fontsize = 15     # health-area names
    value_tick_fontsize = 13    # numbers on the x-axis
    annotation_fontsize = 10    # % change and 95% CI next to bars
    legend_fontsize = 13

    # Figure dimensions.
    figure_width = 15
    figure_height = max(12, len(health_areas) * 2.2)

    # Spacing between health areas.
    group_spacing = 1.8

    y = np.arange(len(health_areas)) * group_spacing

    # Spacing between the five scenario bars.
    bar_height = 0.18

    offsets = (
        np.arange(len(scenarios))
        - (len(scenarios) - 1) / 2
    ) * bar_height

    def plot_healthsystem(absolute_data, percent_data, healthsystem):

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        for i, (label, scenario_prefix, color) in enumerate(scenarios):
            scenario_name = (
                f"{scenario_prefix} / "
                f"{healthsystem} Healthsystem Function"
            )

            # Absolute values determine bar lengths and error bars.
            absolute_df = absolute_data[scenario_name].reindex(
                health_areas
            )

            # Percentage values determine annotations.
            percent_df = percent_data[scenario_name].reindex(
                health_areas
            )

            means = absolute_df["mean"].to_numpy()
            lower = absolute_df["lower"].to_numpy()
            upper = absolute_df["upper"].to_numpy()

            ypos = y + offsets[i]

            # Draw absolute horizontal bars.
            bars = ax.barh(
                ypos,
                means,
                height=bar_height * 0.9,
                color=color,
                label=label,
                xerr=np.vstack([
                    means - lower,
                    upper - means,
                ]),
                capsize=3,
                error_kw={
                    "elinewidth": 0.9,
                    "capthick": 0.9,
                },
            )

            # Annotate bars with percentage change and 95% CI
            annotate_averted_bars(ax, bars, percent_df,
                                  upper_values=upper, lower_values=lower,
                                  horizontal=True, fontsize=annotation_fontsize)

        # Reference line at zero.
        ax.axvline(0, color="black", linewidth=1)

        # Health-area labels.
        ax.set_yticks(y)
        ax.set_yticklabels(health_areas)

        # Separate health areas.
        for i in range(len(health_areas) - 1):
            separator_position = (
                                     y[i] + y[i + 1]
                                 ) / 2

            ax.axhline(
                separator_position,
                color="grey",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        ax.invert_yaxis()

        ax.set_xlabel(
            f"{outcome_label} averted",
            fontsize=axis_label_fontsize,
        )
        ax.set_ylabel(
            "Health area",
            fontsize=axis_label_fontsize,
        )

        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        ax.tick_params(
            axis="y",
            labelsize=area_tick_fontsize,
            length=0,
            pad=12,
        )

        ax.tick_params(axis="x", labelsize=value_tick_fontsize)

        # Allow room for percentage annotations.
        ax.margins(x=0.30)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=5,
            frameon=True,
            fontsize=legend_fontsize,
        )

        fig.subplots_adjust(
            left=0.20,
            right=0.98,
            top=0.97,
            bottom=0.15,
        )

        return fig, ax

    fig_default, ax_default = plot_healthsystem(
        default_absolute,
        default_percent,
        "Default",
    )

    fig_improved, ax_improved = plot_healthsystem(
        improved_absolute,
        improved_percent,
        "Improved",
    )

    return (
        fig_default,
        ax_default,
        fig_improved,
        ax_improved,
    )


# Plot % DALYs averted by health area
def plot_percent_dalys_averted_by_cause(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
):
    return _plot_percent_averted_by_cause(
        default_absolute=default_absolute,
        default_percent=default_percent,
        improved_absolute=improved_absolute,
        improved_percent=improved_percent,
        outcome_label="DALYs",
    )


# Plot % deaths averted by health area
def plot_percent_deaths_averted_by_cause(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
):
    return _plot_percent_averted_by_cause(
        default_absolute=default_absolute,
        default_percent=default_percent,
        improved_absolute=improved_absolute,
        improved_percent=improved_percent,
        outcome_label="Deaths",
    )


def _plot_averted_by_age_group(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
    outcome_label,
):
    """
    Plot absolute and percentage DALYs/deaths averted
    by age group, using two subplots per healthsystem.
    """

    age_order = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ]

    scenarios = [
        ("Fewer nurses", "Fewer Nurses", "indianred"),
        ("More nurses", "More Nurses", "steelblue"),
        ("More CNP", "More CNP staff", "darkgreen"),
        ("More nurses by district", "More Nurses by District", "mediumpurple"),
        ("More CNP by district", "More CNP staff by District", "orange"),
    ]

    # Font sizes, set once and matching the by-cause plots.
    axis_label_fontsize = 16    # x- and y-axis titles
    panel_title_fontsize = 16   # "Absolute ..." / "Percentage ..." titles
    category_tick_fontsize = 15 # age-group names
    value_tick_fontsize = 13    # numbers on the x-axes
    legend_fontsize = 13

    # Five bars per age group.
    bar_height = 0.15
    offsets = (
        np.arange(len(scenarios))
        - (len(scenarios) - 1) / 2
    ) * bar_height

    def plot_healthsystem(
        absolute_data,
        percent_data,
        healthsystem,
    ):

        fig, axes = plt.subplots(
            ncols=2,
            figsize=(19, 11),
            sharey=True,
        )

        ax_absolute, ax_percent = axes

        y = np.arange(len(age_order))

        # Plot absolute and percentage values separately.
        plot_panels = [
            (
                ax_absolute,
                absolute_data,
                "mean",
                "lower",
                "upper",
                f"{outcome_label} averted",
            ),
            (
                ax_percent,
                percent_data,
                "mean",
                "lower",
                "upper",
                f"% {outcome_label} averted",
            ),
        ]

        for panel_index, (
            ax,
            panel_data,
            mean_col,
            lower_col,
            upper_col,
            xlabel,
        ) in enumerate(plot_panels):

            for offset, (label, scenario_prefix, color) in zip(
                offsets,
                scenarios,
            ):

                scenario_name = (
                    f"{scenario_prefix} / "
                    f"{healthsystem} Healthsystem Function"
                )

                df = panel_data[scenario_name].reindex(
                    age_order
                ).iloc[::-1]

                means = df[mean_col].to_numpy()
                lowers = df[lower_col].to_numpy()
                uppers = df[upper_col].to_numpy()

                ypos = y + offset

                ax.barh(
                    ypos,
                    means,
                    height=bar_height * 0.9,
                    color=color,
                    label=label if panel_index == 0 else None,
                )

                ax.errorbar(
                    means,
                    ypos,
                    xerr=np.vstack([
                        means - lowers,
                        uppers - means,
                    ]),
                    fmt="none",
                    elinewidth=1,
                    capsize=1.5,
                    color="black",
                    alpha=0.4,
                )

            ax.axvline(0, color="black", linewidth=1)

            ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)

            ax.set_yticks(y)
            ax.set_yticklabels(age_order[::-1])
            ax.tick_params(axis="y", labelsize=category_tick_fontsize)
            ax.tick_params(axis="x", labelsize=value_tick_fontsize)
            # Size the scale factor (e.g. "1e6") with the tick numbers.
            ax.xaxis.get_offset_text().set_fontsize(value_tick_fontsize)

        # Titles for the two panels.
        ax_absolute.set_title(
            f"Absolute {outcome_label} averted",
            fontsize=panel_title_fontsize,
        )

        ax_percent.set_title(
            f"Percentage {outcome_label} averted",
            fontsize=panel_title_fontsize,
        )

        ax_absolute.set_ylabel("Age group", fontsize=axis_label_fontsize)

        # One shared legend.
        handles, labels = ax_absolute.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=5,
            frameon=True,
            fontsize=legend_fontsize,
        )

        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.93,
            bottom=0.12,
            wspace=0.12,
        )

        return fig, axes

    # Default Healthsystem.
    fig_default, axes_default = plot_healthsystem(
        default_absolute,
        default_percent,
        "Default",
    )

    # Improved Healthsystem.
    fig_improved, axes_improved = plot_healthsystem(
        improved_absolute,
        improved_percent,
        "Improved",
    )

    return (
        fig_default,
        axes_default,
        fig_improved,
        axes_improved,
    )

def plot_percent_dalys_averted_by_age_group(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
):
    return _plot_averted_by_age_group(
        default_absolute,
        default_percent,
        improved_absolute,
        improved_percent,
        outcome_label="DALYs",
    )


def plot_percent_deaths_averted_by_age_group(
    default_absolute,
    default_percent,
    improved_absolute,
    improved_percent,
):
    return _plot_averted_by_age_group(
        default_absolute,
        default_percent,
        improved_absolute,
        improved_percent,
        outcome_label="Deaths",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Analyse DALYs/Deaths across nurse staffing scenarios"
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

    # Scnarios to keep (Default Healthsystem Function only)
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

    # Extract annual DALYs
    annual_dalys = extract_annual_dalys(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # Summarize across runs
    # Filter to Default Healthsystem Function scenarios only
    summarized_annual_dalys = summarize(annual_dalys)

    # Filter to Default Healthsystem Function scenarios only
    summarized_annual_dalys_default = summarized_annual_dalys.loc[
                                      :,
                                      summarized_annual_dalys.columns.get_level_values(0).isin(
                                          default_hs_scenarios
                                      ),
                                      ]

    # Filter to Improved Healthsystem Function scenarios only
    summarized_annual_dalys_improved = summarized_annual_dalys.loc[
                                       :,
                                       summarized_annual_dalys.columns.get_level_values(0).isin(
                                           improved_hs_scenarios
                                       ),
                                       ]

    # Plot 1: Annual DALYs over time
    fig_1, ax_1 = plot_annual_dalys(
        summarized_annual_dalys_default,
        title="DALYs at national level: Default Healthsystem",
    )

    # Plot 2: Percent DALYs averted relative to baseline (2027–2034)
    percent_dalys_averted = calculate_percent_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(default_hs_scenarios)
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    percent_dalys_averted_improved = calculate_percent_dalys_averted(
        annual_dalys.loc[
            :,
            annual_dalys.columns.get_level_values(0).isin(improved_hs_scenarios)
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    # Calculate absolute DALYs averted.
    dalys_averted = calculate_dalys_averted(
        annual_dalys.loc[
        :,
        annual_dalys.columns.get_level_values(0).isin(
            default_hs_scenarios
        )
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    dalys_averted_improved = calculate_dalys_averted(
        annual_dalys.loc[
        :,
        annual_dalys.columns.get_level_values(0).isin(
            improved_hs_scenarios
        )
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_2_default, ax_2_default, fig_2_improved, ax_2_improved = (
        plot_percent_dalys_averted_comparison(
            dalys_averted,
            percent_dalys_averted,
            dalys_averted_improved,
            percent_dalys_averted_improved,
        )
    )

    # Sensitivity analysis: DALYs under Improved Healthsystem Function
    fig_5, ax_5 = plot_annual_dalys(
        summarized_annual_dalys_improved,
        title="DALYs at national level: Improved Healthsystem",
    )

    # Extract annual deaths
    annual_deaths = extract_annual_deaths(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_annual_deaths = summarize(annual_deaths)

    # Default Healthsystem Function deaths
    summarized_annual_deaths_default = summarized_annual_deaths.loc[
                                       :,
                                       summarized_annual_deaths.columns.get_level_values(0).isin(
                                           default_hs_scenarios
                                       ),
                                       ]

    # Improved Healthsystem Function deaths
    summarized_annual_deaths_improved = summarized_annual_deaths.loc[
                                        :,
                                        summarized_annual_deaths.columns.get_level_values(0).isin(
                                            improved_hs_scenarios
                                        ),
                                        ]

    # Plot annual deaths
    fig_3, ax_3 = plot_annual_deaths(
        summarized_annual_deaths_default,
        title="Deaths at national level: Default Healthsystem",
    )

    # Plot % deaths averted
    percent_deaths_averted = calculate_percent_deaths_averted(
        annual_deaths.loc[
            :,
            annual_deaths.columns.get_level_values(0).isin(default_hs_scenarios)
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    percent_deaths_averted_improved = calculate_percent_deaths_averted(
        annual_deaths.loc[
            :,
            annual_deaths.columns.get_level_values(0).isin(improved_hs_scenarios)
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    # Calculate absolute deaths averted.
    deaths_averted = calculate_deaths_averted(
        annual_deaths.loc[
        :,
        annual_deaths.columns.get_level_values(0).isin(
            default_hs_scenarios
        )
        ],
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    deaths_averted_improved = calculate_deaths_averted(
        annual_deaths.loc[
        :,
        annual_deaths.columns.get_level_values(0).isin(
            improved_hs_scenarios
        )
        ],
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_4_default, ax_4_default, fig_4_improved, ax_4_improved = (
        plot_percent_deaths_averted_comparison(
            deaths_averted,
            percent_deaths_averted,
            deaths_averted_improved,
            percent_deaths_averted_improved,
        )
    )

    # Sensitivity analysis: deaths under Improved Healthsystem Function
    fig_7, ax_7 = plot_annual_deaths(
        summarized_annual_deaths_improved,
        title="Deaths at national level: Improved Healthsystem",
    )

    # Extract deaths by cause
    deaths_by_cause = extract_deaths_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total deaths equal to sum of deaths by cause
    total_deaths = annual_deaths.loc[
        (annual_deaths.index >= 2027) & (annual_deaths.index <= 2034)
        ].sum(axis=0)
    total_deaths_cause = deaths_by_cause.sum(axis=0)
    assert (total_deaths.index == total_deaths_cause.index).all()
    assert (abs(total_deaths.values - total_deaths_cause.values) < 1e-7).all()

    # Aggregate individual causes into broader health areas.
    deaths_by_cause = aggregate_causes_by_health_area(deaths_by_cause)
    # Alphabetical order of causes
    # death_order = sorted(deaths_by_cause.index.tolist(), key=str.lower, reverse=True)

    deaths_by_cause_default = (
        deaths_by_cause.loc[
            :,
            deaths_by_cause.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_default = (
        calculate_percent_deaths_averted_by_cause(
            deaths_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_by_cause_improved = (
        deaths_by_cause.loc[
            :,
            deaths_by_cause.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_improved = (
        calculate_percent_deaths_averted_by_cause(
            deaths_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    # Absolute deaths averted by cause.
    deaths_averted_by_cause_default = (
        calculate_absolute_averted_by_category(
            deaths_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_averted_by_cause_improved = (
        calculate_absolute_averted_by_category(
            deaths_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_10_default, ax_10_default, fig_10_improved, ax_10_improved = (
        plot_percent_deaths_averted_by_cause(
            deaths_averted_by_cause_default,
            percent_deaths_by_cause_default,
            deaths_averted_by_cause_improved,
            percent_deaths_by_cause_improved,
        )
    )

    # Extract deaths by age group
    deaths_by_age_group = extract_deaths_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total deaths equal to sum of deaths by age group
    total_deaths_age = deaths_by_age_group.sum(axis=0)
    assert (total_deaths.index == total_deaths_age.index).all()
    assert (abs(total_deaths.values - total_deaths_age.values) < 1e-7).all()

    deaths_by_age_group_default = (
        deaths_by_age_group.loc[
            :,
            deaths_by_age_group.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_age_default = (
        calculate_percent_deaths_averted_by_age_group(
            deaths_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_by_age_group_improved = (
        deaths_by_age_group.loc[
            :,
            deaths_by_age_group.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_age_improved = (
        calculate_percent_deaths_averted_by_age_group(
            deaths_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    # Absolute deaths averted by age group.
    deaths_averted_by_age_default = (
        calculate_absolute_averted_by_category(
            deaths_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    deaths_averted_by_age_improved = (
        calculate_absolute_averted_by_category(
            deaths_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_12_default, ax_12_default, fig_12_improved, ax_12_improved = (
        plot_percent_deaths_averted_by_age_group(
            deaths_averted_by_age_default,
            percent_deaths_by_age_default,
            deaths_averted_by_age_improved,
            percent_deaths_by_age_improved,
        )
    )

    # Extract DALYs by cause
    dalys_by_cause = extract_dalys_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total dalys equal to sum of dalys by cause
    total_dalys = annual_dalys.loc[
        (annual_dalys.index >= 2027) & (annual_dalys.index <= 2034)
        ].sum(axis=0)
    total_dalys_cause = dalys_by_cause.sum(axis=0)
    assert (total_dalys.index == total_dalys_cause.index).all()
    assert (abs(total_dalys.values - total_dalys_cause.values) < 1e-7).all()

    # Aggregate individual causes into broader health areas.
    dalys_by_cause = aggregate_causes_by_health_area(
        dalys_by_cause
    )
    # Alphabetical order of causes
    # cause_order = sorted(dalys_by_cause.index.tolist(), key=str.lower, reverse=True)

    # Default Healthsystem
    dalys_by_cause_default = (
        dalys_by_cause.loc[
            :,
            dalys_by_cause.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_by_cause_default = (
        calculate_percent_dalys_averted_by_cause(
            dalys_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    # Improved Healthsystem
    dalys_by_cause_improved = (
        dalys_by_cause.loc[
            :,
            dalys_by_cause.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_by_cause_improved = (
        calculate_percent_dalys_averted_by_cause(
            dalys_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    # Absolute DALYs averted by cause
    dalys_averted_by_cause_default = (
        calculate_absolute_averted_by_category(
            dalys_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    dalys_averted_by_cause_improved = (
        calculate_absolute_averted_by_category(
            dalys_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_9_default, ax_9_default, fig_9_improved, ax_9_improved = (
        plot_percent_dalys_averted_by_cause(
            dalys_averted_by_cause_default,
            percent_by_cause_default,
            dalys_averted_by_cause_improved,
            percent_by_cause_improved,
        )
    )

    # Extract DALYs by age group
    dalys_by_age_group = extract_dalys_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # check that total dalys equal to sum of dalys by age groups
    total_dalys_age = dalys_by_age_group.sum(axis=0)
    assert (total_dalys.index == total_dalys_age.index).all()
    assert (abs(total_dalys.values - total_dalys_age.values) < 1e-7).all()

    dalys_by_age_group_default = (
        dalys_by_age_group.loc[
            :,
            dalys_by_age_group.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_dalys_by_age_default = (
        calculate_percent_dalys_averted_by_age_group(
            dalys_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    dalys_by_age_group_improved = (
        dalys_by_age_group.loc[
            :,
            dalys_by_age_group.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_dalys_by_age_improved = (
        calculate_percent_dalys_averted_by_age_group(
            dalys_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    # Absolute DALYs averted by age group.
    dalys_averted_by_age_default = (
        calculate_absolute_averted_by_category(
            dalys_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    dalys_averted_by_age_improved = (
        calculate_absolute_averted_by_category(
            dalys_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_11_default, ax_11_default, fig_11_improved, ax_11_improved = (
        plot_percent_dalys_averted_by_age_group(
            dalys_averted_by_age_default,
            percent_dalys_by_age_default,
            dalys_averted_by_age_improved,
            percent_dalys_by_age_improved,
        )
    )

    # Export numerical data underlying 12 selected plots.
    plot_data_for_excel = [
        # 1–4: Baseline-comparison plots
        (
            "DALYs_Base_Default",
            percent_dalys_averted,
            dalys_averted,
            "Category",
        ),
        (
            "DALYs_Base_Improved",
            percent_dalys_averted_improved,
            dalys_averted_improved,
            "Category",
        ),
        (
            "Deaths_Base_Default",
            percent_deaths_averted,
            deaths_averted,
            "Category",
        ),
        (
            "Deaths_Base_Improved",
            percent_deaths_averted_improved,
            deaths_averted_improved,
            "Category",
        ),

        # 5–8: Cause-level plots
        (
            "DALYs_Cause_Default",
            percent_by_cause_default,
            dalys_averted_by_cause_default,
            "Cause",
        ),
        (
            "DALYs_Cause_Improved",
            percent_by_cause_improved,
            dalys_averted_by_cause_improved,
            "Cause",
        ),
        (
            "Deaths_Cause_Default",
            percent_deaths_by_cause_default,
            deaths_averted_by_cause_default,
            "Cause",
        ),
        (
            "Deaths_Cause_Improved",
            percent_deaths_by_cause_improved,
            deaths_averted_by_cause_improved,
            "Cause",
        ),

        # 9–12: Age-group plots
        (
            "DALYs_Age_Default",
            percent_dalys_by_age_default,
            dalys_averted_by_age_default,
            "Age group",
        ),
        (
            "DALYs_Age_Improved",
            percent_dalys_by_age_improved,
            dalys_averted_by_age_improved,
            "Age group",
        ),
        (
            "Deaths_Age_Default",
            percent_deaths_by_age_default,
            deaths_averted_by_age_default,
            "Age group",
        ),
        (
            "Deaths_Age_Improved",
            percent_deaths_by_age_improved,
            deaths_averted_by_age_improved,
            "Age group",
        ),
    ]

    export_nurse_plot_data_to_excel(
        results_folder=results_folder,
        plot_data=plot_data_for_excel,
    )

    # Showing figures
    if args.show_figures:
        plt.show()

    # Saving figures
    if args.save_figures:
        fig_1.savefig(
            results_folder / "annual_dalys_across_scenarios_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_2_default.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_vs_baseline_2027_2034_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_2_improved.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_3.savefig(
            results_folder / "annual_deaths_across_scenarios_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_4_default.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_vs_baseline_2027_2034_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_4_improved.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # Sensitivity-analysis DALY figures
        fig_5.savefig(
            results_folder /
            "annual_dalys_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # fig_6.savefig(
        #     results_folder /
        #     "percent_dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
        #     bbox_inches="tight",
        # )

        # Sensitivity-analysis death figures
        fig_7.savefig(
            results_folder /
            "annual_deaths_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # fig_8.savefig(
        #     results_folder /
        #     "percent_deaths_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
        #     bbox_inches="tight",
        # )

        fig_9_default.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_by_cause_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_9_improved.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_by_cause_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_10_default.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_by_cause_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_10_improved.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_by_cause_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_11_default.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_by_age_group_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_11_improved.savefig(
            results_folder /
            "abs_and_pct_dalys_averted_by_age_group_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_12_default.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_by_age_group_national_level_default_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_12_improved.savefig(
            results_folder /
            "abs_and_pct_deaths_averted_by_age_group_national_level_improved_healthsystem.pdf",
            bbox_inches="tight",
        )
