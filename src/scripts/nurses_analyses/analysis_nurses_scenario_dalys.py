"""Plot DALYs across nurse staffing scenarios.

This script produces two figures for the Default Healthsystem Function scenarios only:

1. Annual DALYs by year (three lines):
   - Baseline Nurses / Default Healthsystem Function
   - Fewer Nurses / Default Healthsystem Function
   - More Nurses / Default Healthsystem Function

2. Percent of DALYs averted compared to Baseline
   (total between 2027 and 2034):
   - More Nurses
   - Fewer Nurses
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import (
    extract_results,
    load_pickled_dataframes,
    summarize,
)


# -----------------------------------------------------------------------------
# Helper function: rename draw numbers to scenario names
# -----------------------------------------------------------------------------
def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


# -----------------------------------------------------------------------------
# Extract annual DALYs
# -----------------------------------------------------------------------------
def extract_annual_dalys(results_folder):
    def get_num_dalys_yearly(df: pd.DataFrame) -> pd.Series:
        """Return total DALYs for each year."""
        # Sum all cause columns after removing metadata columns
        yearly = (
            df.drop(columns=["date", "sex", "age_range"], errors="ignore")
            .groupby("year")
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


# -----------------------------------------------------------------------------
# Plot 1: Annual DALYs over time
# -----------------------------------------------------------------------------
def plot_annual_dalys(summarized_annual_dalys):
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_names = summarized_annual_dalys.columns.get_level_values(0).unique()

    # Short labels for legend
    label_map = {
        "Baseline Nurses / Default Healthsystem Function": "Baseline",
        "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
        "More Nurses / Default Healthsystem Function": "More nurses",

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
    }

    for scenario in scenario_names:
        years = summarized_annual_dalys.index.astype(int)
        means = summarized_annual_dalys[(scenario, "mean")].values
        lowers = summarized_annual_dalys[(scenario, "lower")].values
        uppers = summarized_annual_dalys[(scenario, "upper")].values

        print(means.min(), means.max())

        ax.plot(
            years,
            means,
            linewidth=2,
            label=label_map.get(scenario, scenario),
        )

        ax.fill_between(
            years,
            lowers,
            uppers,
            alpha=0.2,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual DALYs")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(2025, 2034)
    ax.set_ylim(bottom=8e6)
    # ax.set_ylim(bottom=0.8)
    fig.tight_layout()

    return fig, ax


# Plot: Annual Deaths over time
def plot_annual_deaths(summarized_annual_deaths):
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

        "Baseline Nurses / Improved Healthsystem Function": "Baseline",
        "Fewer Nurses / Improved Healthsystem Function": "Fewer nurses",
        "More Nurses / Improved Healthsystem Function": "More nurses",
    }

    for scenario in scenario_names:
        years = summarized_annual_deaths.index.astype(int)

        means = summarized_annual_deaths[
            (scenario, "mean")
        ].values

        lowers = summarized_annual_deaths[
            (scenario, "lower")
        ].values

        uppers = summarized_annual_deaths[
            (scenario, "upper")
        ].values

        ax.plot(
            years,
            means,
            linewidth=2,
            label=label_map.get(scenario, scenario),
        )

        ax.fill_between(
            years,
            lowers,
            uppers,
            alpha=0.2,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual deaths")

    ax.legend()

    ax.grid(alpha=0.3)

    ax.set_xlim(2025, 2034)

    fig.tight_layout()

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

        # CHANGE THIS if your column name differs
        cause_col = "cause"

        deaths_by_cause = (
            df.groupby(cause_col)["person_id"]
            .count()
        )

        return deaths_by_cause

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_deaths_by_cause,
        do_scaling=True,
    )


# Extract deaths by age group
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Extract deaths by age group
# -----------------------------------------------------------------------------
def extract_deaths_by_age_group(results_folder):

    def get_deaths_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return deaths by age group aggregated across 2027–2034.
        """

        # ---------------------------------------------------------
        # Add year
        # ---------------------------------------------------------
        df = df.assign(year=df["date"].dt.year)

        # Restrict years
        df = df[df["year"].between(2027, 2034)]

        # ---------------------------------------------------------
        # Create age groups
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Aggregate deaths by age group
        # ---------------------------------------------------------
        deaths_by_age = (
            df.groupby("age_group")["person_id"]
            .count()
        )

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

        # Add year
        df = df.assign(year=df["date"].dt.year)

        # Restrict years
        df = df[df["year"].between(2027, 2034)]

        # Remove metadata columns
        metadata_cols = [
            "date",
            "sex",
            "age_range",
            "year",
        ]

        cause_cols = [
            c for c in df.columns
            if c not in metadata_cols
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


# -----------------------------------------------------------------------------
# Extract DALYs by age group
# -----------------------------------------------------------------------------
def extract_dalys_by_age_group(results_folder):

    def get_dalys_by_age_group(df: pd.DataFrame) -> pd.Series:
        """
        Return DALYs by age group aggregated across 2027–2034.
        """

        # Add year
        df = df.assign(year=df["date"].dt.year)

        # Restrict years
        df = df[df["year"].between(2027, 2034)]

        # Metadata columns to exclude
        metadata_cols = [
            "date",
            "sex",
            "age_range",
            "year",
        ]

        # DALY cause columns
        cause_cols = [
            c for c in df.columns
            if c not in metadata_cols
        ]

        # Sum DALYs across causes first
        df["total_dalys"] = df[cause_cols].sum(axis=1)

        # Aggregate by age group
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


# -----------------------------------------------------------------------------
# Plot 2: Percent DALYs averted relative to baseline (2027–2034)
# -----------------------------------------------------------------------------
def calculate_percent_dalys_averted(
    summarized_annual_dalys,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    """
    Calculate % DALYs averted relative to baseline.

    Returns DataFrame with:
        mean
        lower
        upper
    """

    years = summarized_annual_dalys.index.astype(int)
    year_mask = np.isin(years, list(comparison_years))

    scenario_names = summarized_annual_dalys.columns.get_level_values(0).unique()

    results = {}

    # Baseline totals
    baseline_mean = (
        summarized_annual_dalys[(baseline_scenario, "mean")]
        .values[year_mask]
        .sum()
    )

    baseline_lower = (
        summarized_annual_dalys[(baseline_scenario, "lower")]
        .values[year_mask]
        .sum()
    )

    baseline_upper = (
        summarized_annual_dalys[(baseline_scenario, "upper")]
        .values[year_mask]
        .sum()
    )

    for scenario in scenario_names:

        if scenario == baseline_scenario:
            continue

        scenario_mean = (
            summarized_annual_dalys[(scenario, "mean")]
            .values[year_mask]
            .sum()
        )

        scenario_lower = (
            summarized_annual_dalys[(scenario, "lower")]
            .values[year_mask]
            .sum()
        )

        scenario_upper = (
            summarized_annual_dalys[(scenario, "upper")]
            .values[year_mask]
            .sum()
        )

        mean_averted = (
            (baseline_mean - scenario_mean)
            / baseline_mean
            * 100.0
        )

        lower_averted = (
            (baseline_lower - scenario_upper)
            / baseline_lower
            * 100.0
        )

        upper_averted = (
            (baseline_upper - scenario_lower)
            / baseline_upper
            * 100.0
        )

        results[scenario] = {
            "mean": mean_averted,
            "lower": lower_averted,
            "upper": upper_averted,
        }

    return pd.DataFrame(results).T


def calculate_percent_deaths_averted(
    summarized_annual_deaths,
    baseline_scenario,
    comparison_years=range(2027, 2035),
):
    years = summarized_annual_deaths.index.astype(int)

    year_mask = np.isin(years, list(comparison_years))

    scenario_names = (
        summarized_annual_deaths.columns
        .get_level_values(0)
        .unique()
    )

    results = {}

    baseline_mean = (
        summarized_annual_deaths[
            (baseline_scenario, "mean")
        ]
        .values[year_mask]
        .sum()
    )

    baseline_lower = (
        summarized_annual_deaths[
            (baseline_scenario, "lower")
        ]
        .values[year_mask]
        .sum()
    )

    baseline_upper = (
        summarized_annual_deaths[
            (baseline_scenario, "upper")
        ]
        .values[year_mask]
        .sum()
    )

    for scenario in scenario_names:

        if scenario == baseline_scenario:
            continue

        scenario_mean = (
            summarized_annual_deaths[
                (scenario, "mean")
            ]
            .values[year_mask]
            .sum()
        )

        scenario_lower = (
            summarized_annual_deaths[
                (scenario, "lower")
            ]
            .values[year_mask]
            .sum()
        )

        scenario_upper = (
            summarized_annual_deaths[
                (scenario, "upper")
            ]
            .values[year_mask]
            .sum()
        )

        mean_averted = (
            (baseline_mean - scenario_mean)
            / baseline_mean
            * 100.0
        )

        lower_averted = (
            (baseline_lower - scenario_upper)
            / baseline_lower
            * 100.0
        )

        upper_averted = (
            (baseline_upper - scenario_lower)
            / baseline_upper
            * 100.0
        )

        results[scenario] = {
            "mean": mean_averted,
            "lower": lower_averted,
            "upper": upper_averted,
        }

    return pd.DataFrame(results).T


# Calculate % DALYs averted by cause
def calculate_percent_dalys_averted_by_cause(
    summarized_dalys_by_cause,
    baseline_scenario,
):
    """
    Returns DataFrame indexed by cause with columns:
        more_nurses
        fewer_nurses
    """

    scenario_names = (
        summarized_dalys_by_cause.columns
        .get_level_values(0)
        .unique()
    )

    baseline = summarized_dalys_by_cause[
        (baseline_scenario, "mean")
    ]

    results = pd.DataFrame(index=baseline.index)

    for scenario in scenario_names:

        if scenario == baseline_scenario:
            continue

        scenario_values = summarized_dalys_by_cause[
            (scenario, "mean")
        ]

        percent_averted = (
            (baseline - scenario_values)
            / baseline
            * 100.0
        )

        if "More Nurses" in scenario:
            results["More nurses"] = percent_averted

        elif "Fewer Nurses" in scenario:
            # Make negative for mirrored plotting
            results["Fewer nurses"] = -percent_averted

    return results


# -----------------------------------------------------------------------------
# Calculate % deaths averted by cause
# -----------------------------------------------------------------------------
def calculate_percent_deaths_averted_by_cause(
    summarized_deaths_by_cause,
    baseline_scenario,
):
    """
    Returns DataFrame indexed by cause with columns:
        More nurses
        Fewer nurses
    """

    scenario_names = (
        summarized_deaths_by_cause.columns
        .get_level_values(0)
        .unique()
    )

    baseline = summarized_deaths_by_cause[
        (baseline_scenario, "mean")
    ]

    results = pd.DataFrame(index=baseline.index)

    for scenario in scenario_names:
        if scenario == baseline_scenario:
            continue

        scenario_values = summarized_deaths_by_cause[
            (scenario, "mean")
        ]

        percent_averted = (
            (baseline - scenario_values)
            / baseline
            * 100.0
        )
        if "More Nurses" in scenario:
            results["More nurses"] = percent_averted

        elif "Fewer Nurses" in scenario:
            results["Fewer nurses"] = -percent_averted

    return results


# -----------------------------------------------------------------------------
# Calculate % DALYs averted by age group
# -----------------------------------------------------------------------------
def calculate_percent_dalys_averted_by_age_group(
    summarized_dalys_by_age,
    baseline_scenario,
):

    scenario_names = (
        summarized_dalys_by_age.columns
        .get_level_values(0)
        .unique()
    )

    results = {}

    baseline_mean = summarized_dalys_by_age[
        (baseline_scenario, "mean")
    ]

    baseline_lower = summarized_dalys_by_age[
        (baseline_scenario, "lower")
    ]

    baseline_upper = summarized_dalys_by_age[
        (baseline_scenario, "upper")
    ]

    for scenario in scenario_names:

        if scenario == baseline_scenario:
            continue

        scenario_mean = summarized_dalys_by_age[
            (scenario, "mean")
        ]

        scenario_lower = summarized_dalys_by_age[
            (scenario, "lower")
        ]

        scenario_upper = summarized_dalys_by_age[
            (scenario, "upper")
        ]

        mean_averted = (
            (baseline_mean - scenario_mean)
            / baseline_mean
            * 100.0
        )

        lower_averted = (
            (baseline_lower - scenario_upper)
            / baseline_lower
            * 100.0
        )

        upper_averted = (
            (baseline_upper - scenario_lower)
            / baseline_upper
            * 100.0
        )

        print(mean_averted.describe())

        print("\n", scenario)
        print("mean:")
        print(mean_averted.head())

        if "Fewer Nurses" in scenario:
            positive_values = mean_averted[mean_averted > 0]

            print("\nPOSITIVE VALUES IN FEWER NURSES:")
            print(positive_values)

            print("\nNUMBER OF POSITIVE AGE GROUPS:")
            print(len(positive_values))

        results[scenario] = pd.DataFrame({
            "mean": mean_averted,
            "lower": lower_averted,
            "upper": upper_averted,
        })

    return results


# Calculate % deaths averted by age group
def calculate_percent_deaths_averted_by_age_group(
    summarized_deaths_by_age,
    baseline_scenario,
):
    """
    Returns DataFrame indexed by age group with columns:
        More nurses
        Fewer nurses
    """

    scenario_names = (
        summarized_deaths_by_age.columns
        .get_level_values(0)
        .unique()
    )

    baseline = summarized_deaths_by_age[
        (baseline_scenario, "mean")
    ]

    results = pd.DataFrame(index=baseline.index)

    for scenario in scenario_names:

        if scenario == baseline_scenario:
            continue

        scenario_values = summarized_deaths_by_age[
            (scenario, "mean")
        ]

        percent_averted = np.where(
            baseline > 0,
            (baseline - scenario_values)
            / baseline
            * 100.0,
            np.nan,
        )

        if "More Nurses" in scenario:
            results["More nurses"] = percent_averted

        elif "Fewer Nurses" in scenario:
            results["Fewer nurses"] = -percent_averted

    return results

# -----------------------------------------------------------------------------
# District-level plot: % DALYs averted compared to baseline (2027–2034)
# -----------------------------------------------------------------------------

# def extract_annual_dalys_by_district(results_folder):
#     """
#     Extract annual DALYs by district.
#
#     This uses the same facility-to-district mapping approach
#     that worked for staff counts.
#     """
#
#     def get_dalys_by_district(df: pd.DataFrame) -> pd.Series:
#         """Return total DALYs for each year and district."""
#
#         # Check if we have the right data structure
#         if 'date' not in df.columns:
#             return pd.Series(dtype=float)
#
#         # Extract year
#         years = df['date'].dt.year.rename("year")
#
#         # Identify district column - for DALYs, district might not be directly available
#         # Instead, we need to aggregate from facility-level data if available
#
#         # For now, if no district column, return national-level with "National" as district
#         if 'district' not in df.columns and 'District' not in df.columns:
#             # Sum all DALY causes
#             daly_cols = [c for c in df.columns if c not in ['date', 'year', 'sex', 'age_range', 'li_wealth']]
#             yearly_total = df.groupby(years)[daly_cols].sum().sum(axis=1)
#
#             # Create Series with (year, "National") index
#             result = pd.Series(
#                 yearly_total.values,
#                 index=pd.MultiIndex.from_arrays([yearly_total.index, ["National"] * len(yearly_total)],
#                                                 names=["year", "District"])
#             )
#             return result
#
#         # If district column exists, use it
#         district_col = 'district' if 'district' in df.columns else 'District'
#         daly_cols = [c for c in df.columns if c not in ['date', 'year', 'sex', 'age_range', 'li_wealth', district_col]]
#
#         # Group by year and district
#         grouped = df.groupby([years, district_col])[daly_cols].sum().sum(axis=1)
#         grouped.index = grouped.index.set_names(["year", "District"])
#
#         return grouped.astype(float)
#
#     return extract_results(
#         results_folder,
#         module="tlo.methods.healthburden",
#         key="dalys_stacked",  # Try this key instead
#         custom_generate_series=get_dalys_by_district,
#         do_scaling=True,
#     )


def find_facility_level_data(results_folder):
    """Inspect HealthBurden outputs properly."""

    from tlo.analysis.utils import load_pickled_dataframes

    log = load_pickled_dataframes(results_folder)

    print("\n" + "=" * 60)
    print("Inspecting HealthBurden outputs...")
    print("=" * 60)

    healthburden_data = log.get("tlo.methods.healthburden", {})

    for key_name, obj in healthburden_data.items():

        print(f"\nKEY: {key_name}")
        print("-" * 50)

        try:
            print(f"TYPE: {type(obj)}")

            # If DataFrame directly
            if isinstance(obj, pd.DataFrame):
                print("DataFrame detected")
                print("Columns:")
                print(obj.columns.tolist())
                print("\nHEAD:")
                print(obj.head())
                continue

            # If dict-like
            if isinstance(obj, dict):

                print(f"DICT KEYS: {list(obj.keys())[:5]}")

                first_key = list(obj.keys())[0]

                first_obj = obj[first_key]

                print(f"FIRST OBJECT TYPE: {type(first_obj)}")

                if isinstance(first_obj, pd.DataFrame):
                    print("Columns:")
                    print(first_obj.columns.tolist())

                    print("\nHEAD:")
                    print(first_obj.head())

                else:
                    print(first_obj)

                continue

            print(obj)

        except Exception as e:
            print(f"ERROR: {e}")


def check_all_dalys_columns(results_folder):
    """Check every DALY-related key for any facility/district columns"""
    from tlo.analysis.utils import load_pickled_dataframes

    log = load_pickled_dataframes(results_folder)
    healthburden = log['tlo.methods.healthburden']

    daly_keys = ['dalys', 'dalys_stacked', 'dalys_stacked_by_age_and_time',
                 'dalys_by_wealth_stacked_by_age_and_time']

    facility_keywords = ['facility', 'district', 'Facility', 'District',
                         'facility_id', 'Facility_ID', 'clinic', 'Clinic']

    for key in daly_keys:
        if key not in healthburden:
            continue

        print(f"\n{'=' * 50}")
        print(f"Checking: {key}")
        print('=' * 50)

        sample = healthburden[key][0]
        all_columns = sample.columns.tolist()

        print(f"Total columns: {len(all_columns)}")
        print(f"Sample columns: {all_columns[:15]}...")

        # Check for facility/district columns
        found = []
        for col in all_columns:
            for kw in facility_keywords:
                if kw.lower() in col.lower():
                    found.append(col)

        if found:
            print(f"\n✓ FOUND facility/district columns: {found}")
        else:
            print("\n❌ No facility or district columns found")


def check_death_columns(results_folder):
    """Inspect death log columns for district/location information."""

    from tlo.analysis.utils import load_pickled_dataframes

    log = load_pickled_dataframes(results_folder)

    death_log = log["tlo.methods.demography"]["death"]

    print("\n" + "=" * 60)
    print("CHECKING DEATH LOG COLUMNS")
    print("=" * 60)

    # Handle dict structure
    if isinstance(death_log, dict):
        first_key = list(death_log.keys())[0]
        sample = death_log[first_key]
    else:
        sample = death_log

    print(sample.columns.tolist())

    # Search for district-like columns
    keywords = [
        "district",
        "District",
        "facility",
        "Facility",
        "region",
        "location",
    ]

    found = []

    for col in sample.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                found.append(col)

    print("\nPossible district/location columns:")
    print(found)
    print("\nPossible death columns:")
    print(sample.columns.tolist())


def inspect_population_log(results_folder):
    from tlo.analysis.utils import load_pickled_dataframes

    log = load_pickled_dataframes(results_folder)

    demography = log["tlo.methods.demography"]

    print("\n" + "=" * 60)
    print("INSPECTING POPULATION LOG")
    print("=" * 60)

    population_obj = demography["population"]

    print(f"\nTYPE: {type(population_obj)}")

    print("\nDICT KEYS:")
    print(population_obj.keys())

    # Take first run
    first_run_key = list(population_obj.keys())[0]

    print(f"\nFIRST RUN KEY: {first_run_key}")

    pop_df = population_obj[first_run_key]

    print(f"\nOBJECT TYPE: {type(pop_df)}")

    if isinstance(pop_df, pd.DataFrame):
        print("\nCOLUMNS:")
        print(pop_df.columns.tolist())

        print("\nHEAD:")
        print(pop_df.head())

        print("\nPOSSIBLE LOCATION COLUMNS:")

        location_cols = [
            c for c in pop_df.columns
            if any(
                kw in c.lower()
                for kw in [
                    "district",
                    "region",
                    "facility",
                    "location",
                    "residence"
                ]
            )
        ]

        print(location_cols)


# def calculate_percent_dalys_averted_by_district(
#     summarized_annual_dalys_by_district,
#     baseline_scenario,
#     comparison_years=range(2027, 2035),
# ):
#     """
#     Calculate % DALYs averted by district relative to baseline.
#
#     Returns a DataFrame:
#         index   = District
#         columns = scenarios (excluding baseline)
#
#     Positive values = DALYs averted
#     Negative values = additional DALYs.
#
#     This function is robust to whether the summarized dataframe index is:
#         1. A MultiIndex: (year, district)
#         2. A single Index of tuples: [(year, district), ...]
#     """
#
#     # ---------------------------------------------------------------------
#     # Reconstruct a proper MultiIndex if summarize() collapsed it into
#     # a single-level Index containing tuples like (year, district)
#     # ---------------------------------------------------------------------
#     if not isinstance(
#         summarized_annual_dalys_by_district.index,
#         pd.MultiIndex
#     ):
#         first_value = summarized_annual_dalys_by_district.index[0]
#
#         # If index entries are tuples of length 2, rebuild MultiIndex
#         if isinstance(first_value, tuple) and len(first_value) == 2:
#             summarized_annual_dalys_by_district = (
#                 summarized_annual_dalys_by_district.copy()
#             )
#
#             summarized_annual_dalys_by_district.index = pd.MultiIndex.from_tuples(
#                 summarized_annual_dalys_by_district.index,
#                 names=["year", "District"],
#             )
#         else:
#             raise ValueError(
#                 "District-level DALY data does not have a "
#                 "(year, district) index structure."
#             )
#
#     # ---------------------------------------------------------------------
#     # At this point we are guaranteed to have a MultiIndex:
#     # level 0 = year
#     # level 1 = district
#     # ---------------------------------------------------------------------
#     districts = (
#         summarized_annual_dalys_by_district.index
#         .get_level_values(1)
#         .unique()
#     )
#
#     scenario_names = (
#         summarized_annual_dalys_by_district.columns
#         .get_level_values(0)
#         .unique()
#     )
#
#     comparison_scenarios = [
#         s for s in scenario_names
#         if s != baseline_scenario
#     ]
#
#     # Results DataFrame
#     results = pd.DataFrame(
#         index=districts,
#         columns=comparison_scenarios,
#         dtype=float,
#     )
#
#     # ---------------------------------------------------------------------
#     # Compute % DALYs averted for each district
#     # ---------------------------------------------------------------------
#     for district in districts:
#
#         # Select all years for this district
#         district_df = summarized_annual_dalys_by_district.xs(
#             district,
#             level=1
#         )
#
#         # Keep only years in comparison period
#         district_df = district_df.loc[
#             district_df.index.isin(comparison_years)
#         ]
#
#         # Skip if no data
#         if district_df.empty:
#             continue
#
#         # Baseline DALYs total
#         baseline_total = district_df[
#             (baseline_scenario, "mean")
#         ].sum()
#
#         # Avoid divide by zero
#         if baseline_total == 0:
#             continue
#
#         # Comparison scenarios
#         for scenario in comparison_scenarios:
#             scenario_total = district_df[
#                 (scenario, "mean")
#             ].sum()
#
#             percent_averted = (
#                 (baseline_total - scenario_total)
#                 / baseline_total
#                 * 100.0
#             )
#
#             results.loc[district, scenario] = percent_averted
#
#     # Remove districts with all missing values
#     results = results.dropna(how="all")
#
#     # Sort alphabetically
#     results = results.sort_index()
#
#     return results


# def plot_percent_dalys_averted_by_district(percent_averted_by_district):
#     """
#     Create horizontal district-level bar chart.
#     Bars to the right  = DALYs averted (positive)
#     Bars to the left   = Additional DALYs (negative)
#     """
#
#     # Desired scenario order and labels
#     scenario_order = [
#         "Fewer Nurses / Default Healthsystem Function",
#         "More Nurses / Default Healthsystem Function",
#     ]
#
#     scenario_order = [
#         s for s in scenario_order
#         if s in percent_averted_by_district.columns
#     ]
#
#     label_map = {
#         "Fewer Nurses / Default Healthsystem Function": "Fewer nurses",
#         "More Nurses / Default Healthsystem Function": "More nurses",
#     }
#
#     districts = percent_averted_by_district.index.tolist()
#     y = np.arange(len(districts))
#
#     fig_height = max(6, len(districts) * 0.35)
#     fig, ax = plt.subplots(figsize=(8, fig_height))
#
#     bar_height = 0.35
#     offsets = np.linspace(
#         -bar_height / 2,
#         bar_height / 2,
#         len(scenario_order)
#     )
#
#     for offset, scenario in zip(offsets, scenario_order):
#         values = (
#             percent_averted_by_district[scenario]
#             .fillna(0)
#             .values
#         )
#
#         ax.barh(
#             y + offset,
#             values,
#             height=bar_height,
#             label=label_map.get(scenario, scenario),
#             alpha=0.8,
#         )
#
#     # Zero reference line
#     ax.axvline(0, color="black", linewidth=1)
#
#     # Y-axis
#     ax.set_yticks(y)
#     ax.set_yticklabels(districts)
#
#     # Labels
#     ax.set_ylabel("District")
#     ax.set_xlabel(
#         "% DALYs averted\n"
#         "(total 2027–2034)\n"
#         "compared to Baseline"
#     )
#
#     # Match sketch style: first district at top
#     ax.invert_yaxis()
#     # Legend
#     ax.legend()
#     # Light grid
#     ax.grid(axis="x", alpha=0.3)
#     fig.tight_layout()
#
#     return fig, ax


def plot_percent_dalys_averted(percent_averted):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep desired ordering dynamically
    ordered_scenarios = [
                            s for s in percent_averted.index
                            if "More Nurses" in s
                        ] + [
                            s for s in percent_averted.index
                            if "Fewer Nurses" in s
                        ]

    labels = [
        "More nurses" if "More Nurses" in s else "Fewer nurses"
        for s in ordered_scenarios
    ]

    means = percent_averted.loc[
        ordered_scenarios, "mean"
    ].values

    lowers = percent_averted.loc[
        ordered_scenarios, "lower"
    ].values

    uppers = percent_averted.loc[
        ordered_scenarios, "upper"
    ].values

    yerr = np.vstack([
        means - lowers,
        uppers - means,
    ])

    ax.bar(
        labels,
        means,
        width=0.45,
        yerr=yerr,
        capsize=6,
    )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel(
        "% DALYs averted compared to Baseline\n"
        "(total between 2027 and 2034)"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_percent_deaths_averted(percent_averted):
    fig, ax = plt.subplots(figsize=(7, 6))

    ordered_scenarios = [
                            s for s in percent_averted.index
                            if "More Nurses" in s
                        ] + [
                            s for s in percent_averted.index
                            if "Fewer Nurses" in s
                        ]

    labels = [
        "More nurses" if "More Nurses" in s else "Fewer nurses"
        for s in ordered_scenarios
    ]

    means = percent_averted.loc[
        ordered_scenarios,
        "mean"
    ].values

    lowers = percent_averted.loc[
        ordered_scenarios,
        "lower"
    ].values

    uppers = percent_averted.loc[
        ordered_scenarios,
        "upper"
    ].values

    yerr = np.vstack([
        means - lowers,
        uppers - means,
    ])

    ax.bar(
        labels,
        means,
        width=0.45,
        yerr=yerr,
        capsize=6,
    )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel(
        "% deaths averted compared to Baseline\n"
        "(total between 2027 and 2034)"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig, ax


# Plot % DALYs averted by cause
def plot_percent_dalys_averted_by_cause(
    default_df,
    improved_df,
    top_n=10,
):
    # ---------------------------------------------------------
    # Select top causes based on absolute impact
    # ---------------------------------------------------------
    ranking = (
        default_df["More nurses"].abs()
        .sort_values(ascending=False)
    )

    top_causes = ranking.head(top_n).index.tolist()

    default_df = default_df.loc[top_causes]
    improved_df = improved_df.loc[top_causes]

    # Reverse so largest appears at top
    default_df = default_df.iloc[::-1]
    improved_df = improved_df.iloc[::-1]

    # ---------------------------------------------------------
    # Create subplots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 8),
        sharey=True,
    )

    panel_data = [
        (axes[0], default_df, "Default Healthsystem"),
        (axes[1], improved_df, "Improved Healthsystem"),
    ]

    for ax, df, title in panel_data:
        y = np.arange(len(df))

        # More nurses (positive)
        ax.barh(
            y,
            df["More nurses"],
            color="lightsteelblue",
            label="More nurses",
        )

        # Fewer nurses (negative)
        ax.barh(
            y,
            df["Fewer nurses"],
            color="lightsteelblue",
            label="Fewer nurses",
        )

        # Zero line
        ax.axvline(0, color="black", linewidth=1)
        # Cause labels
        ax.set_yticks(y)
        ax.set_yticklabels(df.index)
        ax.set_xlabel("% DALYs averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "% DALYs averted by causes on national level\n(2027–2034)"
    )
    fig.tight_layout()
    return fig, axes


# Plot % deaths averted by cause
def plot_percent_deaths_averted_by_cause(
    default_df,
    improved_df,
    top_n=10,
):
    # ---------------------------------------------------------
    # Select top causes
    # ---------------------------------------------------------
    ranking = (
        default_df["More nurses"].abs()
        .sort_values(ascending=False)
    )

    top_causes = ranking.head(top_n).index.tolist()

    default_df = default_df.loc[top_causes]
    improved_df = improved_df.loc[top_causes]

    # Reverse so largest appears at top
    default_df = default_df.iloc[::-1]
    improved_df = improved_df.iloc[::-1]

    # ---------------------------------------------------------
    # Create subplots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 8),
        sharey=True,
    )

    panel_data = [
        (axes[0], default_df, "Default Healthsystem"),
        (axes[1], improved_df, "Improved Healthsystem"),
    ]

    for ax, df, title in panel_data:
        y = np.arange(len(df))

        ax.barh(
            y,
            df["More nurses"],
            color="lightcoral",
            label="More nurses",
        )

        ax.barh(
            y,
            df["Fewer nurses"],
            color="lightcoral",
            label="Fewer nurses",
        )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(df.index)
        ax.set_xlabel("% deaths averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "% deaths averted by causes on national level\n(2027–2034)"
    )

    fig.tight_layout()

    return fig, axes



# Plot % DALYs averted by age group
# -----------------------------------------------------------------------------
def plot_percent_dalys_averted_by_age_group(
    default_df,
    improved_df,
):
    # ---------------------------------------------------------
    # Extract scenario DataFrames from dictionaries
    # ---------------------------------------------------------
    default_more = default_df[
        "More Nurses / Default Healthsystem Function"
    ]

    default_fewer = default_df[
        "Fewer Nurses / Default Healthsystem Function"
    ]

    improved_more = improved_df[
        "More Nurses / Improved Healthsystem Function"
    ]

    improved_fewer = improved_df[
        "Fewer Nurses / Improved Healthsystem Function"
    ]

    # ---------------------------------------------------------
    # Order age groups
    # ---------------------------------------------------------
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

    for df in [
        default_more,
        default_fewer,
        improved_more,
        improved_fewer,
    ]:
        df = df.reindex(age_order)

    default_more = default_more.reindex(age_order)
    default_fewer = default_fewer.reindex(age_order)

    improved_more = improved_more.reindex(age_order)
    improved_fewer = improved_fewer.reindex(age_order)

    # Reverse so oldest ages appear at top
    default_more = default_more.iloc[::-1]
    default_fewer = default_fewer.iloc[::-1]

    improved_more = improved_more.iloc[::-1]
    improved_fewer = improved_fewer.iloc[::-1]

    # ---------------------------------------------------------
    # Create subplots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 8),
        sharey=True,
    )

    panel_data = [
        (
            axes[0],
            default_more,
            default_fewer,
            "Default Healthsystem",
        ),
        (
            axes[1],
            improved_more,
            improved_fewer,
            "Improved Healthsystem",
        ),
    ]

    for ax, more, fewer, title in panel_data:
        y = np.arange(len(more))

        # More nurses
        ax.barh(
            y - 0.2,
            more["mean"],
            height=0.35,
            color="steelblue",
            label="More Nurses",
        )

        # Fewer nurses
        ax.barh(
            y + 0.2,
            fewer["mean"],
            height=0.35,
            color="indianred",
            label="Fewer Nurses",
        )

        # CI for More Nurses
        ax.errorbar(
            more["mean"],
            y - 0.2,
            xerr=[
                more["mean"] - more["lower"],
                more["upper"] - more["mean"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )

        # CI for Fewer Nurses
        ax.errorbar(
            fewer["mean"],
            y + 0.2,
            xerr=[
                fewer["mean"] - fewer["lower"],
                fewer["upper"] - fewer["mean"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )

        ax.axvline(0, color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(more.index)

        ax.set_xlabel("% DALYs averted")
        ax.set_title(title)

        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "% DALYs averted by age group on national level\n(2027–2034)"
    )

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    return fig, axes


# -----------------------------------------------------------------------------
# Plot % deaths averted by age group
# -----------------------------------------------------------------------------
def plot_percent_deaths_averted_by_age_group(
    default_df,
    improved_df,
):

    # ---------------------------------------------------------
    # Order age groups properly
    # ---------------------------------------------------------
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

    default_df = (
        default_df
        .reindex(age_order)
        .dropna(how="all")
    )

    improved_df = (
        improved_df
        .reindex(age_order)
        .dropna(how="all")
    )

    # Reverse for plotting
    default_df = default_df.iloc[::-1]
    improved_df = improved_df.iloc[::-1]

    # ---------------------------------------------------------
    # Create subplots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, 8),
        sharey=False,
    )

    panel_data = [
        (axes[0], default_df, "Default Healthsystem"),
        (axes[1], improved_df, "Improved Healthsystem"),
    ]

    for ax, df, title in panel_data:

        y = np.arange(len(df))

        # More nurses
        ax.barh(
            y,
            df["More nurses"],
            color="lightcoral",
            label="More nurses",
        )

        # Fewer nurses
        ax.barh(
            y,
            df["Fewer nurses"],
            color="lightcoral",
            label="Fewer nurses",
        )

        # Zero line
        ax.axvline(0, color="black", linewidth=1)
        # Labels
        ax.set_yticks(y)
        ax.set_yticklabels(df.index)
        ax.set_xlabel("% deaths averted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "% deaths averted by age group on national level\n(2027–2034)"
    )

    fig.tight_layout()

    return fig, axes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Analyse DALYs across nurse staffing scenarios"
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

    # ADD THIS LINE - Debug to find facility data
    # facility_key = find_facility_level_data(results_folder)
    # print(f"\n✓ Found facility-level data in key: {facility_key}")
    # print("=" * 60 + "\n")

    # daly_cols = check_all_dalys_columns(results_folder)
    # print(f"\n✓ Found DALY columns: {daly_cols}")
    # check_death_columns(results_folder)

    # inspect_population_log(results_folder)

    # Get scenario names from scenario class
    param_names = tuple(StaffingScenario()._scenarios.keys())

    print("\nPARAM NAMES:")
    print(param_names)

    # Scenarios to keep (Default Healthsystem Function only)
    default_hs_scenarios = [
        "Baseline Nurses / Default Healthsystem Function",
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
    ]

    baseline_scenario = "Baseline Nurses / Default Healthsystem Function"

    improved_hs_scenarios = [
        "Baseline Nurses / Improved Healthsystem Function",
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
    ]

    baseline_improved_scenario = (
        "Baseline Nurses / Improved Healthsystem Function"
    )

    # -------------------------------------------------------------------------
    # Extract annual DALYs
    # -------------------------------------------------------------------------
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

    print("\nALL DALY SCENARIOS:")
    print(
        summarized_annual_dalys.columns.get_level_values(0).unique().tolist()
    )

    print("\nFILTERED IMPROVED DALY SCENARIOS:")
    print(
        summarized_annual_dalys_improved.columns.get_level_values(0).unique().tolist()
    )

    # -------------------------------------------------------------------------
    # Plot 1: Annual DALYs over time
    # -------------------------------------------------------------------------
    fig_1, ax_1 = plot_annual_dalys(summarized_annual_dalys_default)

    # -------------------------------------------------------------------------
    # Plot 2: Percent DALYs averted relative to baseline (2027–2034)
    # -------------------------------------------------------------------------
    percent_averted = calculate_percent_dalys_averted(
        summarized_annual_dalys_default,
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),  # 2027 to 2034 inclusive
    )

    fig_2, ax_2 = plot_percent_dalys_averted(percent_averted)

    # Sensitivity analysis: DALYs under Improved Healthsystem Function
    fig_5, ax_5 = plot_annual_dalys(
        summarized_annual_dalys_improved
    )

    percent_averted_improved = calculate_percent_dalys_averted(
        summarized_annual_dalys_improved,
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    print("\nPERCENT DALYS AVERTED IMPROVED:")
    print(percent_averted_improved)

    fig_6, ax_6 = plot_percent_dalys_averted(
        percent_averted_improved
    )

    # -------------------------------------------------------------------------
    # Plot 3: Percent DALYs averted by district (2027–2034)
    # -------------------------------------------------------------------------
    # annual_dalys_by_district = extract_annual_dalys_by_district(
    #     results_folder
    # ).pipe(
    #     set_param_names_as_column_index_level_0,
    #     param_names=param_names,
    # )

    # Summarize across runs
    # summarized_annual_dalys_by_district = summarize(
    #     annual_dalys_by_district
    # )
    #
    # # Filter to Default Healthsystem Function scenarios only
    # summarized_annual_dalys_by_district = (
    #     summarized_annual_dalys_by_district.loc[
    #         :,
    #         summarized_annual_dalys_by_district.columns
    #         .get_level_values(0)
    #         .isin(default_hs_scenarios)
    #     ]
    # )

    # Calculate district-level % DALYs averted
    # percent_averted_by_district = (
    #     calculate_percent_dalys_averted_by_district(
    #         summarized_annual_dalys_by_district,
    #         baseline_scenario=baseline_scenario,
    #         comparison_years=range(2027, 2035),  # 2027 to 2034 inclusive
    #     )
    # )
    #
    # # Create district-level plot
    # fig_3, ax_3 = plot_percent_dalys_averted_by_district(
    #     percent_averted_by_district
    # )

    # -------------------------------------------------------------------------
    # Extract annual deaths
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    fig_3, ax_3 = plot_annual_deaths(
        summarized_annual_deaths_default
    )

    # -------------------------------------------------------------------------
    # Plot % deaths averted
    # -------------------------------------------------------------------------
    percent_deaths_averted = calculate_percent_deaths_averted(
        summarized_annual_deaths_default,
        baseline_scenario=baseline_scenario,
        comparison_years=range(2027, 2035),
    )

    fig_4, ax_4 = plot_percent_deaths_averted(
        percent_deaths_averted
    )

    # Sensitivity analysis: deaths under Improved Healthsystem Function
    fig_7, ax_7 = plot_annual_deaths(
        summarized_annual_deaths_improved
    )

    percent_deaths_averted_improved = calculate_percent_deaths_averted(
        summarized_annual_deaths_improved,
        baseline_scenario=baseline_improved_scenario,
        comparison_years=range(2027, 2035),
    )

    print("\nPERCENT DEATHS AVERTED IMPROVED:")
    print(percent_deaths_averted_improved)

    fig_8, ax_8 = plot_percent_deaths_averted(
        percent_deaths_averted_improved
    )

    # Extract deaths by cause
    # -------------------------------------------------------------------------
    deaths_by_cause = extract_deaths_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_deaths_by_cause = summarize(deaths_by_cause)

    summarized_deaths_by_cause_default = (
        summarized_deaths_by_cause.loc[
                :,
                summarized_deaths_by_cause.columns
                .get_level_values(0)
                .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_default = (
        calculate_percent_deaths_averted_by_cause(
            summarized_deaths_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    summarized_deaths_by_cause_improved = (
        summarized_deaths_by_cause.loc[
                :,
                summarized_deaths_by_cause.columns
                .get_level_values(0)
                .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_cause_improved = (
        calculate_percent_deaths_averted_by_cause(
            summarized_deaths_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_10, ax_10 = plot_percent_deaths_averted_by_cause(
        percent_deaths_by_cause_default,
        percent_deaths_by_cause_improved,
        top_n=10,
    )

    # -------------------------------------------------------------------------
    # Extract deaths by age group
    # -------------------------------------------------------------------------
    deaths_by_age_group = extract_deaths_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_deaths_by_age_group = summarize(
        deaths_by_age_group
    )

    # Deaths by cause Default
    summarized_deaths_by_age_group_default = (
        summarized_deaths_by_age_group.loc[
            :,
            summarized_deaths_by_age_group.columns
            .get_level_values(0)
            .isin(default_hs_scenarios)
        ]
    )

    percent_deaths_by_age_default = (
        calculate_percent_deaths_averted_by_age_group(
            summarized_deaths_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    # Deaths by cause Improved
    summarized_deaths_by_age_group_improved = (
        summarized_deaths_by_age_group.loc[
            :,
            summarized_deaths_by_age_group.columns
            .get_level_values(0)
            .isin(improved_hs_scenarios)
        ]
    )

    percent_deaths_by_age_improved = (
        calculate_percent_deaths_averted_by_age_group(
            summarized_deaths_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_12, ax_12 = plot_percent_deaths_averted_by_age_group(
        percent_deaths_by_age_default,
        percent_deaths_by_age_improved,
    )

    # Extract DALYs by cause
    # -------------------------------------------------------------------------
    dalys_by_cause = extract_dalys_by_cause(results_folder).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_dalys_by_cause = summarize(dalys_by_cause)

    # DALYs by cause Default
    summarized_dalys_by_cause_default = (
        summarized_dalys_by_cause.loc[
        :,
        summarized_dalys_by_cause.columns
        .get_level_values(0)
        .isin(default_hs_scenarios)
        ]
    )

    percent_by_cause_default = (
        calculate_percent_dalys_averted_by_cause(
            summarized_dalys_by_cause_default,
            baseline_scenario=baseline_scenario,
        )
    )

    # DALYs by cause Improved
    summarized_dalys_by_cause_improved = (
        summarized_dalys_by_cause.loc[
        :,
        summarized_dalys_by_cause.columns
        .get_level_values(0)
        .isin(improved_hs_scenarios)
        ]
    )

    percent_by_cause_improved = (
        calculate_percent_dalys_averted_by_cause(
            summarized_dalys_by_cause_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    fig_9, ax_9 = plot_percent_dalys_averted_by_cause(
        percent_by_cause_default,
        percent_by_cause_improved,
        top_n=10,
    )

    # Extract DALYs by age group
    # -------------------------------------------------------------------------
    dalys_by_age_group = extract_dalys_by_age_group(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    summarized_dalys_by_age_group = summarize(
        dalys_by_age_group
    )

    # DALYs by age group Default
    summarized_dalys_by_age_group_default = (
        summarized_dalys_by_age_group.loc[
        :,
        summarized_dalys_by_age_group.columns
        .get_level_values(0)
        .isin(default_hs_scenarios)
        ]
    )

    percent_dalys_by_age_default = (
        calculate_percent_dalys_averted_by_age_group(
            summarized_dalys_by_age_group_default,
            baseline_scenario=baseline_scenario,
        )
    )

    # DALYs by age group Improved
    summarized_dalys_by_age_group_improved = (
        summarized_dalys_by_age_group.loc[
        :,
        summarized_dalys_by_age_group.columns
        .get_level_values(0)
        .isin(improved_hs_scenarios)
        ]
    )

    percent_dalys_by_age_improved = (
        calculate_percent_dalys_averted_by_age_group(
            summarized_dalys_by_age_group_improved,
            baseline_scenario=baseline_improved_scenario,
        )
    )

    print("\nDEFAULT AGE GROUP OBJECT:")
    print(type(percent_dalys_by_age_default))
    print(percent_dalys_by_age_default.keys())

    print("\nIMPROVED AGE GROUP OBJECT:")
    print(type(percent_dalys_by_age_improved))
    print(percent_dalys_by_age_improved.keys())

    fig_11, ax_11 = plot_percent_dalys_averted_by_age_group(
        percent_dalys_by_age_default,
        percent_dalys_by_age_improved,
    )



    # -------------------------------------------------------------------------
    # Show figures
    # -------------------------------------------------------------------------
    if args.show_figures:
        plt.show()

    # -------------------------------------------------------------------------
    # Save figures
    # -------------------------------------------------------------------------
    if args.save_figures:
        fig_1.savefig(
            results_folder / "annual_dalys_across_scenarios.pdf",
            bbox_inches="tight",
        )

        fig_2.savefig(
            results_folder / "percent_dalys_averted_vs_baseline_2027_2034.pdf",
            bbox_inches="tight",
        )

        # fig_3.savefig(
        #     results_folder / "percent_dalys_averted_by_district_2027_2034.pdf",
        #     bbox_inches="tight",
        # )

        fig_3.savefig(
            results_folder / "annual_deaths_across_scenarios.pdf",
            bbox_inches="tight",
        )

        fig_4.savefig(
            results_folder / "percent_deaths_averted_vs_baseline_2027_2034.pdf",
            bbox_inches="tight",
        )

        # Sensitivity-analysis DALY figures
        fig_5.savefig(
            results_folder /
            "annual_dalys_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_6.savefig(
            results_folder /
            "percent_dalys_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        # Sensitivity-analysis death figures
        fig_7.savefig(
            results_folder /
            "annual_deaths_across_scenarios_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_8.savefig(
            results_folder /
            "percent_deaths_averted_vs_baseline_2027_2034_improved_healthsystem.pdf",
            bbox_inches="tight",
        )

        fig_9.savefig(
            results_folder /
            "percent_dalys_averted_by_cause_national_level.pdf",
            bbox_inches="tight",
        )

        fig_10.savefig(
            results_folder /
            "percent_deaths_averted_by_cause_national_level.pdf",
            bbox_inches="tight",
        )

        fig_11.savefig(
            results_folder /
            "percent_dalys_averted_by_age_group_national_level.pdf",
            bbox_inches="tight",
        )

        fig_12.savefig(
            results_folder /
            "percent_deaths_averted_by_age_group_national_level.pdf",
            bbox_inches="tight",
        )
