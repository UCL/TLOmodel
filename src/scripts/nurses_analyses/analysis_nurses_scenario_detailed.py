"""This file uses the results of the results of running `nurse_analyses/nurses_scenario_analyses.py` to make plots of
nurse counts over time and appointments over time for each scenario/draw name from 2010 to 2034."""

import argparse
from pathlib import Path
from typing import Tuple, Dict
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo import Date
from tlo.analysis.utils import (
    load_pickled_dataframes,
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


def load_data_manually(results_folder: Path) -> Dict:
    """
    Manually load data from the folder structure we observed.
    Folder structure: draw_folder/run_folder/pickle_files
    """
    data_by_draw = {}

    # Find all draw folders (0, 1, 2, 3, 4, 5)
    draw_folders = [d for d in results_folder.iterdir() if d.is_dir() and d.name.isdigit()]
    draw_folders.sort(key=lambda x: int(x.name))

    print(f"\nFound {len(draw_folders)} draw folders: {[d.name for d in draw_folders]}")

    for draw_folder in draw_folders:
        draw_num = int(draw_folder.name)
        data_by_draw[draw_num] = {}

        # Find run folders (0, 1)
        run_folders = [r for r in draw_folder.iterdir() if r.is_dir() and r.name.isdigit()]
        run_folders.sort(key=lambda x: int(x.name))

        print(f"\nDraw {draw_num} - Found {len(run_folders)} run folders: {[r.name for r in run_folders]}")

        for run_folder in run_folders:
            run_num = int(run_folder.name)

            # Load all pickle files in this run folder
            pickle_files = list(run_folder.glob("*.pickle"))

            run_data = {}
            for pickle_file in pickle_files:
                try:
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f)

                    # Store by module name (filename without extension)
                    module_name = pickle_file.stem
                    run_data[module_name] = data
                    print(f"    Loaded {module_name} from run {run_num}")

                except Exception as e:
                    print(f"    Error loading {pickle_file.name}: {e}")

            data_by_draw[draw_num][run_num] = run_data

    return data_by_draw


def extract_nurse_counts_from_run(run_data: Dict, target_years=range(2010, 2035)) -> pd.Series:
    """
    Extract nurse counts from a single run's data.
    Looking for the right data source - probably not number_of_hcw_staff directly.
    """
    # Look for healthsystem summary data
    for module_name, data in run_data.items():
        if 'healthsystem.summary' in module_name:
            if isinstance(data, dict):
                print(f"    Examining {module_name} - keys: {list(data.keys())}")

                # First, let's check what DataFrames are available
                for key in data.keys():
                    if isinstance(data[key], pd.DataFrame):
                        df = data[key]
                        print(f"      DataFrame '{key}' has columns: {list(df.columns)}")

                        # Check if this might have nurse count data
                        if 'date' in df.columns:
                            # Look for columns that might contain nurse counts
                            for col in df.columns:
                                if 'Nursing' in str(col) or 'Midwifery' in str(col) or 'staff' in str(col).lower():
                                    print(f"        Found potential nurse column: {col}")

                            # If we find a promising DataFrame, try to extract
                            if 'Capacity' in key or 'staff' in key.lower():
                                df['year'] = pd.to_datetime(df['date']).dt.year
                                df_filtered = df[df['year'].isin(target_years)]

                                if not df_filtered.empty:
                                    # Look for nursing columns
                                    nursing_cols = [col for col in df_filtered.columns
                                                  if 'Nursing' in str(col) or 'Midwifery' in str(col)]

                                    if nursing_cols:
                                        # Sum across all nursing columns
                                        result = df_filtered.groupby('year')[nursing_cols].sum().sum(axis=1)
                                        print(f"        Found nursing columns: {nursing_cols}")
                                        print(f"        Sample values: {result.head()}")
                                        return result

                                    # If no nursing columns, look for staff columns
                                    staff_cols = [col for col in df_filtered.columns
                                                if 'staff' in str(col).lower() or 'count' in str(col).lower()]

                                    if staff_cols and len(staff_cols) > 0:
                                        # Try to get the first staff column
                                        result = df_filtered.groupby('year')[staff_cols[0]].mean()
                                        print(f"        Using staff column: {staff_cols[0]}")
                                        print(f"        Sample values: {result.head()}")
                                        return result

    return pd.Series(dtype=float)


def extract_appointments_from_run(run_data: Dict, target_years=range(2010, 2035)) -> pd.Series:
    """
    Extract appointments from a single run's data.
    """
    # Look for healthsystem summary data
    for module_name, data in run_data.items():
        if 'healthsystem.summary' in module_name:
            if isinstance(data, dict):
                # Look for HSI_Event data
                for key in ['HSI_Event', 'HSI_Event_non_blank_appt_footprint']:
                    if key in data:
                        df = data[key]
                        if isinstance(df, pd.DataFrame):
                            if 'date' in df.columns and 'Number_By_Appt_Type_Code' in df.columns:
                                df['year'] = pd.to_datetime(df['date']).dt.year

                                # Filter to target years
                                df_filtered = df[df['year'].isin(target_years)]

                                if not df_filtered.empty:
                                    # Expand appointment counts
                                    appts_expanded = df_filtered['Number_By_Appt_Type_Code'].apply(pd.Series)

                                    # Group by year and sum
                                    appts_expanded['year'] = df_filtered['year'].values
                                    yearly = appts_expanded.groupby('year').sum()

                                    return yearly.sum(axis=1)

    return pd.Series(dtype=float)


def process_all_draws(data_by_draw: Dict, target_years=range(2010, 2035)):
    """
    Process all draws to get nurse counts and appointments.
    Returns DataFrames with draws as columns and years as index.
    """
    nurse_data = {}
    appt_data = {}

    for draw_num, run_data_dict in data_by_draw.items():
        draw_nurse_series = []
        draw_appt_series = []

        for run_num, run_data in run_data_dict.items():
            print(f"\n  Processing Draw {draw_num}, Run {run_num}")

            # Extract nurse counts for this run
            nurse_series = extract_nurse_counts_from_run(run_data, target_years)
            if not nurse_series.empty:
                draw_nurse_series.append(nurse_series)
                print(f"    ✓ Found nurse data with years: {list(nurse_series.index)[:5]}...")
                print(f"    ✓ Sample values: {list(nurse_series.values)[:5]}...")

            # Extract appointments for this run
            appt_series = extract_appointments_from_run(run_data, target_years)
            if not appt_series.empty:
                draw_appt_series.append(appt_series)
                print(f"    ✓ Found appointment data with years: {list(appt_series.index)[:5]}...")

        # Average across runs for this draw
        if draw_nurse_series:
            # Convert list of Series to DataFrame and compute mean
            nurse_df = pd.DataFrame(draw_nurse_series)
            nurse_data[draw_num] = nurse_df.mean()
            print(f"  Draw {draw_num}: Averaged nurse data from {len(draw_nurse_series)} runs")

        if draw_appt_series:
            appt_df = pd.DataFrame(draw_appt_series)
            appt_data[draw_num] = appt_df.mean()
            print(f"  Draw {draw_num}: Averaged appointment data from {len(draw_appt_series)} runs")

    # Convert to DataFrames with draws as columns
    nurse_df = pd.DataFrame(nurse_data) if nurse_data else pd.DataFrame()
    appt_df = pd.DataFrame(appt_data) if appt_data else pd.DataFrame()

    return nurse_df, appt_df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_nurse_counts(nurse_df, param_names, output_folder, target_period_str):
    """
    Plot nurse counts over time for all scenarios.
    """
    if nurse_df.empty:
        print("No nurse count data to plot")
        return None, None

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors and line styles
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Use different markers to distinguish overlapping lines
    markers = ['o', 's', '^', 'D', 'v', '<']

    plots_made = False

    for draw_idx, scenario in enumerate(param_names):
        if draw_idx in nurse_df.columns:
            series = nurse_df[draw_idx]

            if series is not None and not series.empty:
                plots_made = True

                # Determine label, color, and line style based on scenario name
                if 'Baseline' in scenario:
                    level = 'Baseline'
                elif 'Fewer' in scenario:
                    level = 'Fewer'
                elif 'More' in scenario:
                    level = 'More'
                else:
                    level = 'Unknown'

                if 'Default' in scenario:
                    hs_type = 'Default'
                    color = colors[0]  # Blue for Default
                else:  # Improved
                    hs_type = 'Improved'
                    color = colors[1]  # Orange for Improved

                # Line styles based on nurse level
                if level == 'Baseline':
                    linestyle = '-'
                elif level == 'Fewer':
                    linestyle = '--'
                elif level == 'More':
                    linestyle = ':'
                else:
                    linestyle = '-'

                label = f"{level} - {hs_type}"

                # Use different markers for each draw to see if lines are overlapping
                marker = markers[draw_idx % len(markers)]

                ax.plot(
                    series.index,
                    series.values,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=6,
                    markevery=3,
                    linewidth=2
                )
                print(f"  ✓ Plotted Draw {draw_idx}: {label}")

    if not plots_made:
        print("  No data to plot")
        plt.close(fig)
        return None, None

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Nurses', fontsize=12)
    ax.set_title(f'Nurse Counts Over Time by Scenario ({target_period_str})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)

    # Set x-ticks
    all_years = []
    for col in nurse_df.columns:
        all_years.extend(nurse_df[col].index)

    if all_years:
        all_years = sorted(set(all_years))
        tick_years = all_years[::2] if len(all_years) > 10 else all_years
        ax.set_xticks(tick_years)
        ax.set_xticklabels(tick_years, rotation=45)

    fig.tight_layout()

    # Save figures
    fig.savefig(output_folder / "nurse_counts_over_time.pdf", bbox_inches='tight')
    fig.savefig(output_folder / "nurse_counts_over_time.png", bbox_inches='tight', dpi=300)

    return fig, ax


def plot_appointments(appt_df, param_names, output_folder, target_period_str):
    """
    Plot appointments over time for all scenarios.
    """
    if appt_df.empty:
        print("No appointment data to plot")
        return None, None

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors and line styles
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Use different markers to distinguish overlapping lines
    markers = ['o', 's', '^', 'D', 'v', '<']

    plots_made = False

    for draw_idx, scenario in enumerate(param_names):
        if draw_idx in appt_df.columns:
            series = appt_df[draw_idx]

            if series is not None and not series.empty:
                plots_made = True

                # Determine label, color, and line style based on scenario name
                if 'Baseline' in scenario:
                    level = 'Baseline'
                elif 'Fewer' in scenario:
                    level = 'Fewer'
                elif 'More' in scenario:
                    level = 'More'
                else:
                    level = 'Unknown'

                if 'Default' in scenario:
                    hs_type = 'Default'
                    color = colors[0]  # Blue for Default
                else:  # Improved
                    hs_type = 'Improved'
                    color = colors[1]  # Orange for Improved

                # Line styles based on nurse level
                if level == 'Baseline':
                    linestyle = '-'
                elif level == 'Fewer':
                    linestyle = '--'
                elif level == 'More':
                    linestyle = ':'
                else:
                    linestyle = '-'

                label = f"{level} - {hs_type}"

                # Use different markers for each draw
                marker = markers[draw_idx % len(markers)]

                # Convert to millions for plotting
                values_millions = series.values / 1_000_000

                ax.plot(
                    series.index,
                    values_millions,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=6,
                    markevery=3,
                    linewidth=2
                )
                print(f"  ✓ Plotted Draw {draw_idx}: {label}")

    if not plots_made:
        print("  No data to plot")
        plt.close(fig)
        return None, None

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Appointments (millions)', fontsize=12)
    ax.set_title(f'Total Appointments Delivered Over Time by Scenario ({target_period_str})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)

    # Set x-ticks
    all_years = []
    for col in appt_df.columns:
        all_years.extend(appt_df[col].index)

    if all_years:
        all_years = sorted(set(all_years))
        tick_years = all_years[::2] if len(all_years) > 10 else all_years
        ax.set_xticks(tick_years)
        ax.set_xticklabels(tick_years, rotation=45)

    fig.tight_layout()

    # Save figures
    fig.savefig(output_folder / "appointments_over_time.pdf", bbox_inches='tight')
    fig.savefig(output_folder / "appointments_over_time.png", bbox_inches='tight', dpi=300)

    return fig, ax


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot nurse counts and appointments from nurses scenario"
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
        help="Whether to save figures",
    )
    args = parser.parse_args()

    results_folder = args.scenario_outputs_folder

    print(f"\n{'='*60}")
    print(f"Loading results from: {results_folder}")
    print(f"{'='*60}")

    # Get scenario names
    param_names = tuple(StaffingScenario()._scenarios.keys())
    print(f"\nFound {len(param_names)} scenarios:")
    for i, name in enumerate(param_names):
        print(f"  {i}: {name}")

    # Create output folder
    output_folder = results_folder / "analysis_output"
    output_folder.mkdir(exist_ok=True)

    # Define target period
    target_years = range(2010, 2035)
    target_period_str = "2010-2034"

    # Manually load all data
    print(f"\n{'='*60}")
    print("MANUALLY LOADING DATA FROM FOLDER STRUCTURE")
    print(f"{'='*60}")

    data_by_draw = load_data_manually(results_folder)

    # Process all draws to extract nurse counts and appointments
    print(f"\n{'='*60}")
    print("EXTRACTING NURSE COUNTS AND APPOINTMENTS")
    print(f"{'='*60}")

    nurse_df, appt_df = process_all_draws(data_by_draw, target_years)

    # Print summary of extracted data
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")

    if not nurse_df.empty:
        print(f"\n✓ Nurse count data shape: {nurse_df.shape}")
        print(f"Draws with nurse data: {list(nurse_df.columns)}")
        for col in nurse_df.columns:
            print(f"  Draw {col}: years {list(nurse_df[col].index)[:5]}...")
            print(f"    Values: {list(nurse_df[col].values)[:5]}...")
    else:
        print("\n✗ No nurse count data found")

    if not appt_df.empty:
        print(f"\n✓ Appointment data shape: {appt_df.shape}")
        print(f"Draws with appointment data: {list(appt_df.columns)}")
        for col in appt_df.columns:
            print(f"  Draw {col}: years {list(appt_df[col].index)[:5]}...")
    else:
        print("\n✗ No appointment data found")

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    if not nurse_df.empty:
        print("\nPlotting nurse counts...")
        fig1, ax1 = plot_nurse_counts(nurse_df, param_names, output_folder, target_period_str)
        if fig1 is not None:
            print(f"✓ Nurse counts plot saved to {output_folder}/nurse_counts_over_time.png")
    else:
        print("\n✗ Cannot plot nurse counts - no data available")

    if not appt_df.empty:
        print("\nPlotting appointments...")
        fig2, ax2 = plot_appointments(appt_df, param_names, output_folder, target_period_str)
        if fig2 is not None:
            print(f"✓ Appointments plot saved to {output_folder}/appointments_over_time.png")
    else:
        print("\n✗ Cannot plot appointments - no data available")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

    if args.show_figures:
        plt.show()
