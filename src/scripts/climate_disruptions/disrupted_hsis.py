import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, summarize, get_color_short_treatment_id

min_year = 2025
max_year = 2027
PREFIX_ON_FILENAME = "1"
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = True
main_text = False
mode_2 = False

if climate_sensitivity_analysis:
    scenario_names = [
        "Baseline",
        "SSP 1.26 High",
        "SSP 1.26 Low",
        "SSP 1.26 Mean",
        "SSP 2.45 High",
        "SSP 2.45 Low",
        "SSP 2.45 Mean",
        "SSP 5.85 High",
        "SSP 5.85 Low",
        "SSP 5.85 Mean",
    ]
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
if parameter_sensitivity_analysis:
    num_draws = 200  # Total number of parameter scan draws
    scenario_names_all = range(num_draws)
    scenario_names = scenario_names_all
    suffix = "parameter_SA"

    # Specify draws to skip (e.g., corrupted data, failed runs)
    skip_draws = [140]

    scenarios_of_interest = [i for i in range(num_draws) if i not in skip_draws]

    print(f"\nConfiguration:")
    print(f"Total draws: {num_draws}")
    print(f"Draws to skip: {skip_draws if skip_draws else 'None'}")
    print(f"Draws to process: {len(scenarios_of_interest)}")

if main_text:
    scenario_names = [
        "Baseline",
        "SSP 2.45 Mean",
    ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]

precipitation_files = {
    "Baseline": "/Users/rem76/Desktop/Climate_change_health/Data/historical_weather_by_smaller_facilities_with_ANC_lm.csv",
    "SSP 1.26 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 1.26 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 1.26 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp126/mean_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 2.45 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp245/mean_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 High": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/highest_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 Low": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/lowest_monthly_prediction_weather_by_facility.csv",
    "SSP 5.85 Mean": "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/ssp585/mean_monthly_prediction_weather_by_facility.csv",
}

scenario_colours = [
    "#823038",  # Baseline
    # SSP 1.26 (Teal)
    "#00566f",  # High
    "#0081a7",  # Low
    "#5ab4c6",  # Mean
    # SSP 2.45 (Purple/Lavender - more distinct)
    "#5b3f8c",  # High
    "#8e7cc3",  # Low
    "#c7b7ec",  # Mean
    # SSP 5.85 (Coral)
    "#c65a52",  # High
    "#f07167",  # Low
    "#f59e96",  # Mean
]


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the healthcare system utilization across scenarios.
    - FACILITY-LEVEL AVERAGING: Calculates disruption % for each facility, then averages
    - This gives equal weight to each facility regardless of volume
    - Optimized for parameter sensitivity analysis with 200 draws.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # ========================================================================
    # FACILITY-LEVEL HELPER FUNCTIONS
    # ========================================================================

    def get_hsi_counts_by_facility(_df):
        """Get HSI counts by facility"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        if len(_df) == 0:
            return pd.Series(dtype=int)

        facility_totals = {}
        for _, row in _df.iterrows():
            counts_dict = row['hsi_event_key_to_counts']
            for key, value in counts_dict.items():
                if ':' in key:
                    facility_id, _ = key.split(':', 1)
                    facility_totals[facility_id] = facility_totals.get(facility_id, 0) + value

        return pd.Series(facility_totals)

    def get_never_ran_by_facility(_df):
        """Get never-ran counts by facility"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        if len(_df) == 0:
            return pd.Series(dtype=int)

        facility_totals = {}
        for _, row in _df.iterrows():
            counts_dict = row['never_ran_hsi_event_key_to_counts']
            for key, value in counts_dict.items():
                if ':' in key:
                    facility_id, _ = key.split(':', 1)
                    facility_totals[facility_id] = facility_totals.get(facility_id, 0) + value

        return pd.Series(facility_totals)

    def get_delayed_by_facility(_df):
        """Count delayed HSIs by facility"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        if len(_df) == 0:
            return pd.Series(dtype=int)

        return _df.groupby('Facility_ID').size()

    def get_cancelled_by_facility(_df):
        """Count cancelled HSIs by facility"""
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

        if len(_df) == 0:
            return pd.Series(dtype=int)

        return _df.groupby('Facility_ID').size()

    # ========================================================================
    # STEP 1: Extract baseline facility-level data ONCE (as denominator)
    # ========================================================================

    print("Extracting baseline facility-level data...")

    baseline_hsi_by_facility = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='hsi_event_counts',
            custom_generate_series=get_hsi_counts_by_facility,
            do_scaling=False,
        ),
        only_mean=True,
        collapse_columns=False,
    )[0]  # Draw 0 is baseline

    # Fix index to be int
    baseline_hsi_by_facility.index = pd.Index(
        [int(x) for x in baseline_hsi_by_facility.index],
        name='Facility_ID'
    )

    print(f"Baseline data: {len(baseline_hsi_by_facility)} facilities")

    # ========================================================================
    # STEP 2: Process each draw - calculate facility-level proportions
    # ========================================================================

    # Storage for all draws - NOW storing percentages (facility-averaged)
    all_draws_pct_delayed = []
    all_draws_pct_cancelled = []
    all_draws_pct_disrupted = []

    # Also store absolute totals for reporting
    all_draws_total_treatments = []
    all_draws_total_delayed = []
    all_draws_total_cancelled = []
    all_draws_total_never_ran = []

    processed_draws = []

    print(f"\nProcessing {len(scenarios_of_interest)} draws...")
    if parameter_sensitivity_analysis and skip_draws:
        print(f"Skipping draws: {skip_draws}")

    for idx, draw in enumerate(scenarios_of_interest):
        if idx % 10 == 0:
            print(f"Processing draw {idx}/{len(scenarios_of_interest)} (draw number: {draw})...")

        try:
            # Check if this is baseline
            is_baseline = (draw == 0 and parameter_sensitivity_analysis) or \
                          (climate_sensitivity_analysis and scenario_names[idx] == 'Baseline') or \
                          (main_text and scenario_names[idx] == 'Baseline')

            if is_baseline:
                # Baseline: no weather disruptions
                pct_delayed = 0.0
                pct_cancelled = 0.0
                pct_disrupted = 0.0

                # Total counts
                total_treatments = baseline_hsi_by_facility.sum()
                total_delayed = 0
                total_cancelled = 0

                # Get never-ran
                never_ran_by_facility = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='never_ran_hsi_event_counts',
                        custom_generate_series=get_never_ran_by_facility,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=False,
                )[draw]
                never_ran_by_facility.index = pd.Index([int(x) for x in never_ran_by_facility.index])
                total_never_ran = never_ran_by_facility.sum()

            else:
                # Get delayed counts by facility
                delayed_by_facility = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='Weather_delayed_HSI_Event_full_info',
                        custom_generate_series=get_delayed_by_facility,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=False,
                )[draw]

                # Get cancelled counts by facility
                cancelled_by_facility = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='Weather_cancelled_HSI_Event_full_info',
                        custom_generate_series=get_cancelled_by_facility,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=False,
                )[draw]

                # Get HSI counts for this draw
                hsi_by_facility = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='hsi_event_counts',
                        custom_generate_series=get_hsi_counts_by_facility,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=False,
                )[draw]
                hsi_by_facility.index = pd.Index([int(x) for x in hsi_by_facility.index])

                # Get never-ran
                never_ran_by_facility = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.healthsystem.summary',
                        key='never_ran_hsi_event_counts',
                        custom_generate_series=get_never_ran_by_facility,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=False,
                )[draw]
                never_ran_by_facility.index = pd.Index([int(x) for x in never_ran_by_facility.index])

                # ============================================================
                # KEY CHANGE: FACILITY-LEVEL CALCULATION
                # ============================================================

                # Align facility data (ensures same facilities in same order)
                baseline_aligned, delayed_aligned = baseline_hsi_by_facility.align(
                    delayed_by_facility, fill_value=0
                )
                baseline_aligned, cancelled_aligned = baseline_hsi_by_facility.align(
                    cancelled_by_facility, fill_value=0
                )

                # Calculate proportion for EACH FACILITY (avoiding division by zero)
                delayed_proportions = np.where(
                    baseline_aligned > 0,
                    delayed_aligned / baseline_aligned,
                    0
                )
                cancelled_proportions = np.where(
                    baseline_aligned > 0,
                    cancelled_aligned / baseline_aligned,
                    0
                )
                disrupted_proportions = np.where(
                    baseline_aligned > 0,
                    (delayed_aligned + cancelled_aligned) / baseline_aligned,
                    0
                )

                # AVERAGE across facilities (equal weight per facility)
                pct_delayed = np.mean(delayed_proportions) * 100
                pct_cancelled = np.mean(cancelled_proportions) * 100
                pct_disrupted = np.mean(disrupted_proportions) * 100

                # Total counts (still useful for absolute reporting)
                total_treatments = hsi_by_facility.sum()
                total_delayed = delayed_by_facility.sum()
                total_cancelled = cancelled_by_facility.sum()
                total_never_ran = never_ran_by_facility.sum()

            # Store results (FIX FOR BROADCASTING ERROR - ensure all are scalars)
            all_draws_pct_delayed.append(float(pct_delayed))
            all_draws_pct_cancelled.append(float(pct_cancelled))
            all_draws_pct_disrupted.append(float(pct_disrupted))

            all_draws_total_treatments.append(float(total_treatments))
            all_draws_total_delayed.append(float(total_delayed))
            all_draws_total_cancelled.append(float(total_cancelled))
            all_draws_total_never_ran.append(float(total_never_ran))

            processed_draws.append(draw)

        except Exception as e:
            print(f"\nWARNING: Error processing draw {draw}: {e}")
            print(f"Skipping draw {draw} and continuing...")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll draws processed.")
    print(f"Successfully processed: {len(processed_draws)} draws")
    if len(processed_draws) < len(scenarios_of_interest):
        failed_draws = [d for d in scenarios_of_interest if d not in processed_draws]
        print(f"Failed draws: {failed_draws}")
        print(f"\nTo skip these draws in future runs, add to skip_draws list:")
        print(f"skip_draws = {failed_draws}")

    print("Creating summary statistics and figures...")

    # Convert to arrays with explicit dtype (FIX FOR BROADCASTING ERROR)
    pct_delayed_values = np.array(all_draws_pct_delayed, dtype=float)
    pct_cancelled_values = np.array(all_draws_pct_cancelled, dtype=float)
    pct_disrupted_values = np.array(all_draws_pct_disrupted, dtype=float)

    total_treatments_values = np.array(all_draws_total_treatments, dtype=float)
    total_delayed_values = np.array(all_draws_total_delayed, dtype=float)
    total_cancelled_values = np.array(all_draws_total_cancelled, dtype=float)
    total_never_ran_values = np.array(all_draws_total_never_ran, dtype=float)

    # Create comprehensive summary DataFrame
    summary_df = pd.DataFrame({
        'Draw': processed_draws,
        'total_treatments': total_treatments_values,
        'total_delayed': total_delayed_values,
        'total_cancelled': total_cancelled_values,
        'total_never_ran': total_never_ran_values,
        'pct_delayed_facility_avg': pct_delayed_values,
        'pct_cancelled_facility_avg': pct_cancelled_values,
        'pct_disrupted_facility_avg': pct_disrupted_values,
    })

    summary_df.to_csv(output_folder / f"summary_all_draws_facility_level_{suffix}.csv", index=False)

    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Treatments', 'Total Delayed', 'Total Cancelled', 'Total Never Ran',
            '% Delayed (facility-avg)', '% Cancelled (facility-avg)', '% Disrupted (facility-avg)'
        ],
        'Mean': [
            total_treatments_values.mean(), total_delayed_values.mean(),
            total_cancelled_values.mean(), total_never_ran_values.mean(),
            pct_delayed_values.mean(), pct_cancelled_values.mean(), pct_disrupted_values.mean()
        ],
        'Median': [
            np.median(total_treatments_values), np.median(total_delayed_values),
            np.median(total_cancelled_values), np.median(total_never_ran_values),
            np.median(pct_delayed_values), np.median(pct_cancelled_values),
            np.median(pct_disrupted_values)
        ],
        'Std': [
            total_treatments_values.std(), total_delayed_values.std(),
            total_cancelled_values.std(), total_never_ran_values.std(),
            pct_delayed_values.std(), pct_cancelled_values.std(), pct_disrupted_values.std()
        ],
        'Min': [
            total_treatments_values.min(), total_delayed_values.min(),
            total_cancelled_values.min(), total_never_ran_values.min(),
            pct_delayed_values.min(), pct_cancelled_values.min(), pct_disrupted_values.min()
        ],
        'Max': [
            total_treatments_values.max(), total_delayed_values.max(),
            total_cancelled_values.max(), total_never_ran_values.max(),
            pct_delayed_values.max(), pct_cancelled_values.max(), pct_disrupted_values.max()
        ],
    })

    summary_stats.to_csv(output_folder / f"summary_statistics_facility_level_{suffix}.csv", index=False)

    # ============================================================================
    # SUMMARY FIGURES
    # ============================================================================

    print("Creating summary visualizations...")

    # 1. Distribution of absolute counts (box plots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    bp1 = axes[0, 0].boxplot([total_treatments_values], labels=['Total Treatments'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#823038')
    bp1['boxes'][0].set_alpha(0.7)
    axes[0, 0].set_ylabel('Number of HSIs')
    axes[0, 0].set_title(
        f'Total Health System Interactions ({min_year}–{max_year})\nAcross {len(processed_draws)} Parameter Draws')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].text(-0.05, 1.05, "(A)", transform=axes[0, 0].transAxes, fontsize=14, va="top", ha="right")

    bp2 = axes[0, 1].boxplot([total_delayed_values], labels=['Weather Delayed'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#FEB95F')
    bp2['boxes'][0].set_alpha(0.7)
    axes[0, 1].set_ylabel('Number of HSIs')
    axes[0, 1].set_title(f'Total Weather-Delayed Appointments ({min_year}–{max_year})')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].text(-0.05, 1.05, "(B)", transform=axes[0, 1].transAxes, fontsize=14, va="top", ha="right")

    bp3 = axes[1, 0].boxplot([total_cancelled_values], labels=['Weather Cancelled'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('#f07167')
    bp3['boxes'][0].set_alpha(0.7)
    axes[1, 0].set_ylabel('Number of HSIs')
    axes[1, 0].set_title(f'Total Weather-Cancelled Appointments ({min_year}–{max_year})')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].text(-0.05, 1.05, "(C)", transform=axes[1, 0].transAxes, fontsize=14, va="top", ha="right")

    bp4 = axes[1, 1].boxplot([total_never_ran_values], labels=['Never Ran'], patch_artist=True)
    bp4['boxes'][0].set_facecolor('#5b3f8c')
    bp4['boxes'][0].set_alpha(0.7)
    axes[1, 1].set_ylabel('Number of HSIs')
    axes[1, 1].set_title(f'Total Never-Ran Appointments ({min_year}–{max_year})')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].text(-0.05, 1.05, "(D)", transform=axes[1, 1].transAxes, fontsize=14, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(output_folder / f"distributions_absolute_counts_{suffix}.png", dpi=300)
    plt.close(fig)

    # 2. Histograms of absolute counts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].hist(total_treatments_values, bins=30, color='#823038', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(total_treatments_values.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {total_treatments_values.mean():.0f}')
    axes[0, 0].set_xlabel('Total HSIs')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Total Treatments')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].text(-0.05, 1.05, "(A)", transform=axes[0, 0].transAxes, fontsize=14, va="top", ha="right")

    axes[0, 1].hist(total_never_ran_values, bins=30, color='#5b3f8c', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(total_never_ran_values.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {total_never_ran_values.mean():.0f}')
    axes[0, 1].set_xlabel('Never Ran HSIs')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Never-Ran Appointments')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].text(-0.05, 1.05, "(B)", transform=axes[0, 1].transAxes, fontsize=14, va="top", ha="right")

    axes[1, 0].hist(total_delayed_values, bins=30, color='#FEB95F', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(total_delayed_values.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {total_delayed_values.mean():.0f}')
    axes[1, 0].set_xlabel('Weather Delayed HSIs')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Weather-Delayed Appointments')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].text(-0.05, 1.05, "(C)", transform=axes[1, 0].transAxes, fontsize=14, va="top", ha="right")

    axes[1, 1].hist(total_cancelled_values, bins=30, color='#f07167', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(total_cancelled_values.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {total_cancelled_values.mean():.0f}')
    axes[1, 1].set_xlabel('Weather Cancelled HSIs')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Weather-Cancelled Appointments')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].text(-0.05, 1.05, "(D)", transform=axes[1, 1].transAxes, fontsize=14, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(output_folder / f"histograms_absolute_counts_{suffix}.png", dpi=300)
    plt.close(fig)

    # 3. Percentage disruptions (FACILITY-AVERAGED)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].hist(pct_delayed_values, bins=30, color='#FEB95F', alpha=0.7, edgecolor='black')
    axes[0].axvline(pct_delayed_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {pct_delayed_values.mean():.2f}%')
    axes[0].set_xlabel('% Delayed (facility-averaged)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of % Appointments Delayed\n(averaged across facilities)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].text(-0.05, 1.05, "(A)", transform=axes[0].transAxes, fontsize=14, va="top", ha="right")

    axes[1].hist(pct_cancelled_values, bins=30, color='#f07167', alpha=0.7, edgecolor='black')
    axes[1].axvline(pct_cancelled_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {pct_cancelled_values.mean():.2f}%')
    axes[1].set_xlabel('% Cancelled (facility-averaged)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of % Appointments Cancelled\n(averaged across facilities)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].text(-0.05, 1.05, "(B)", transform=axes[1].transAxes, fontsize=14, va="top", ha="right")

    axes[2].hist(pct_disrupted_values, bins=30, color='#c65a52', alpha=0.7, edgecolor='black')
    axes[2].axvline(pct_disrupted_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {pct_disrupted_values.mean():.2f}%')
    axes[2].set_xlabel('% Disrupted (facility-averaged)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of % Total Appointments Disrupted\n(averaged across facilities)')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].text(-0.05, 1.05, "(C)", transform=axes[2].transAxes, fontsize=14, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(output_folder / f"percentage_disruptions_facility_averaged_{suffix}.png", dpi=300)
    plt.close(fig)

    # 4. Coefficient of variation
    cv_data = pd.DataFrame({
        'Metric': ['Total Treatments', 'Weather Delayed', 'Weather Cancelled', 'Never Ran',
                   '% Delayed (fac-avg)', '% Cancelled (fac-avg)', '% Disrupted (fac-avg)'],
        'CV': [
            (total_treatments_values.std() / total_treatments_values.mean()) * 100,
            (total_delayed_values.std() / total_delayed_values.mean()) * 100,
            (total_cancelled_values.std() / total_cancelled_values.mean()) * 100,
            (total_never_ran_values.std() / total_never_ran_values.mean()) * 100,
            (pct_delayed_values.std() / pct_delayed_values.mean()) * 100,
            (pct_cancelled_values.std() / pct_cancelled_values.mean()) * 100,
            (pct_disrupted_values.std() / pct_disrupted_values.mean()) * 100,
        ]
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#823038', '#FEB95F', '#f07167', '#5b3f8c', '#FEB95F', '#f07167', '#c65a52']
    ax.barh(cv_data['Metric'], cv_data['CV'], color=colors, alpha=0.7)
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title(f'Parameter Uncertainty Across Metrics\n(CV across {len(processed_draws)} draws)')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / f"cv_by_metric_{suffix}.png", dpi=300)
    plt.close(fig)

    print("\nAnalysis complete!")
    print(f"\nSummary Statistics (n={len(processed_draws)} draws):")
    print(
        f"Total Treatments - Mean: {total_treatments_values.mean():.0f}, Range: {total_treatments_values.min():.0f} to {total_treatments_values.max():.0f}")
    print(
        f"Weather Delayed - Mean: {total_delayed_values.mean():.0f}, Range: {total_delayed_values.min():.0f} to {total_delayed_values.max():.0f}")
    print(
        f"Weather Cancelled - Mean: {total_cancelled_values.mean():.0f}, Range: {total_cancelled_values.min():.0f} to {total_cancelled_values.max():.0f}")
    print(
        f"\n% Delayed (facility-averaged) - Mean: {pct_delayed_values.mean():.2f}%, Range: {pct_delayed_values.min():.2f}% to {pct_delayed_values.max():.2f}%")
    print(
        f"% Cancelled (facility-averaged) - Mean: {pct_cancelled_values.mean():.2f}%, Range: {pct_cancelled_values.min():.2f}% to {pct_cancelled_values.max():.2f}%")
    print(
        f"% Disrupted (facility-averaged) - Mean: {pct_disrupted_values.mean():.2f}%, Range: {pct_disrupted_values.min():.2f}% to {pct_disrupted_values.max():.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
