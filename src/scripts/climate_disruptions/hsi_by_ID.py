import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
    get_color_short_treatment_id,
    load_pickled_dataframes
)

min_year = 2025
max_year = 2041
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = False
main_text = True
# get scale factor from pre-suspend logs
log = load_pickled_dataframes(
    Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/baseline_run_with_pop-2026-03-03T092729Z/"), 0,
    0)
population_scaling_factor = log['tlo.methods.demography']['scaling_factor']['scaling_factor'].iloc[0]

scenario_names_all = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low",
                      "SSP 2.45 Mean", "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]

if climate_sensitivity_analysis:
    scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low",
                      "SSP 2.45 Mean", "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))
if parameter_sensitivity_analysis:
    scenario_names_all = range(0, 10, 1)
    scenario_names = scenario_names_all
    suffix = "parameter_SA"

if main_text:
    scenario_names = ["No disruption", "Baseline", "Worst Case"]
    suffix = "main_text"
    scenarios_of_interest = [0, 1, 2]

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

scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167'] * 4


def add_significance_stars(ax, x_positions, baseline_data, climate_data, y_offset_factor=0.05):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_offset = y_range * y_offset_factor

    for i, x_pos in enumerate(x_positions):
        baseline_mean = baseline_data['mean'].iloc[i]
        baseline_lower = baseline_data['lower'].iloc[i]
        baseline_upper = baseline_data['upper'].iloc[i]

        climate_mean = climate_data['mean'].iloc[i]
        climate_lower = climate_data['lower'].iloc[i]
        climate_upper = climate_data['upper'].iloc[i]

        baseline_se = (baseline_upper - baseline_lower) / (2 * 1.96)
        climate_se = (climate_upper - climate_lower) / (2 * 1.96)

        diff = abs(climate_mean - baseline_mean)
        pooled_se = np.sqrt(baseline_se ** 2 + climate_se ** 2)

        if pooled_se > 0:
            z_score = diff / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(z_score))
        else:
            p_value = 1.0

        if p_value < 0.001:
            star = '***'
        elif p_value < 0.01:
            star = '**'
        elif p_value < 0.05:
            star = '*'
        else:
            star = ''

        if star:
            y_pos = max(baseline_mean, climate_mean) + y_offset
            ax.text(x_pos, y_pos, star,
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    color='red')


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    def sum_event_counts(_df, column_name):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        total = {}
        for d in _df[column_name]:
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
        return pd.Series(sum(total.values()), name="total")

    def get_num_treatments_total(_df):
        return sum_event_counts(_df, "hsi_event_key_to_counts")

    def get_num_treatments_total_delayed(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        return pd.Series(len(_df), name="total")

    def get_num_treatments_total_cancelled(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        return pd.Series(len(_df), name="total")

    def get_population_total(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        population_mean = numeric_df.sum(numeric_only=True).mean()
        return pd.Series(population_mean, name="population")

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_treatments_mean = []
    all_draws_treatments_lower = []
    all_draws_treatments_upper = []

    all_draws_weather_delayed_mean = []
    all_draws_weather_delayed_lower = []
    all_draws_weather_delayed_upper = []

    all_draws_weather_cancelled_mean = []
    all_draws_weather_cancelled_lower = []
    all_draws_weather_cancelled_upper = []

    all_draws_treatments_mean_1000 = []
    all_draws_treatments_lower_1000 = []
    all_draws_treatments_upper_1000 = []

    all_draws_weather_delayed_mean_1000 = []
    all_draws_weather_cancelled_mean_1000 = []

    all_years_by_draw = {}

    for draw in range(len(scenario_names_all)):
        if draw not in scenarios_of_interest:
            continue

        all_years_data_treatments_mean = {}
        all_years_data_treatments_upper = {}
        all_years_data_treatments_lower = {}

        all_years_data_weather_delayed_mean = {}
        all_years_data_weather_delayed_upper = {}
        all_years_data_weather_delayed_lower = {}

        all_years_data_weather_cancelled_mean = {}
        all_years_data_weather_cancelled_upper = {}
        all_years_data_weather_cancelled_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            num_treatments_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=False
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_treatments_mean[target_year] = num_treatments_total['mean']
            all_years_data_treatments_lower[target_year] = num_treatments_total['lower']
            all_years_data_treatments_upper[target_year] = num_treatments_total['upper']

            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_total,
                do_scaling=False
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_population_mean[target_year] = result_data_population['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']

            if scenario_names[draw] == 'No disruption':
                all_years_data_weather_delayed_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_delayed_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_delayed_upper[target_year] = pd.Series([0], name='upper')

                all_years_data_weather_cancelled_mean[target_year] = pd.Series([0], name='mean')
                all_years_data_weather_cancelled_lower[target_year] = pd.Series([0], name='lower')
                all_years_data_weather_cancelled_upper[target_year] = pd.Series([0], name='upper')
            elif main_text:
                num_weather_delayed_appointments = summarize(extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event_full_info",
                    custom_generate_series=get_num_treatments_total_delayed,
                    do_scaling=False,
                ),
                    only_mean=True,
                    collapse_columns=True,
                )[draw]

                all_years_data_weather_delayed_mean[target_year] = num_weather_delayed_appointments['mean']
                all_years_data_weather_delayed_lower[target_year] = num_weather_delayed_appointments['lower']
                all_years_data_weather_delayed_upper[target_year] = num_weather_delayed_appointments['upper']

                num_weather_cancelled_appointments = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.healthsystem.summary",
                        key="Weather_cancelled_HSI_Event_full_info",
                        custom_generate_series=get_num_treatments_total_cancelled,
                        do_scaling=False,
                    ),
                    only_mean=True,
                    collapse_columns=True,
                )[draw]
                all_years_data_weather_cancelled_mean[target_year] = num_weather_cancelled_appointments['mean']
                all_years_data_weather_cancelled_lower[target_year] = num_weather_cancelled_appointments['lower']
                all_years_data_weather_cancelled_upper[target_year] = num_weather_cancelled_appointments['upper']

            # Population data for normalization (second call kept for consistency)
            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_total,
                do_scaling=False
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data_population_mean[target_year] = result_data_population['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']

        df_all_years_treatments_mean = pd.DataFrame(all_years_data_treatments_mean)
        print(df_all_years_treatments_mean)
        df_all_years_treatments_lower = pd.DataFrame(all_years_data_treatments_lower)
        df_all_years_treatments_upper = pd.DataFrame(all_years_data_treatments_upper)

        df_all_years_weather_delayed_mean = pd.DataFrame(all_years_data_weather_delayed_mean)
        df_all_years_weather_delayed_lower = pd.DataFrame(all_years_data_weather_delayed_lower)
        df_all_years_weather_delayed_upper = pd.DataFrame(all_years_data_weather_delayed_upper)

        df_all_years_weather_cancelled_mean = pd.DataFrame(all_years_data_weather_cancelled_mean)
        df_all_years_weather_cancelled_lower = pd.DataFrame(all_years_data_weather_cancelled_lower)
        df_all_years_weather_cancelled_upper = pd.DataFrame(all_years_data_weather_cancelled_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)

        all_years_by_draw[draw] = {
            'treatments': {
                'mean': df_all_years_treatments_mean.sum(),
                'lower': df_all_years_treatments_lower.sum(),
                'upper': df_all_years_treatments_upper.sum()
            },
            'weather_delayed': {
                'mean': df_all_years_weather_delayed_mean.sum(),
                'lower': df_all_years_weather_delayed_lower.sum(),
                'upper': df_all_years_weather_delayed_upper.sum()
            },
            'weather_cancelled': {
                'mean': df_all_years_weather_cancelled_mean.sum(),
                'lower': df_all_years_weather_cancelled_lower.sum(),
                'upper': df_all_years_weather_cancelled_upper.sum()
            },
            'population': df_all_years_data_population_mean
        }

        fig, axes = plt.subplots(2, 2, figsize=(25, 20))

        df_treatments_per_1000_mean = df_all_years_treatments_mean / df_all_years_data_population_mean.iloc[0, 0] * 1000
        df_weather_delayed_per_1000_mean = df_all_years_weather_delayed_mean / df_all_years_data_population_mean.iloc[
            0, 0] * 1000
        df_weather_cancelled_per_1000_mean = df_all_years_weather_cancelled_mean / \
                                             df_all_years_data_population_mean.iloc[0, 0] * 1000

        df_all_years_treatments_mean.to_csv(output_folder / f"treatments_by_type_{draw}.csv")
        df_all_years_weather_delayed_mean.to_csv(output_folder / f"weather_delayed_by_type_{draw}.csv")
        df_all_years_weather_cancelled_mean.to_csv(output_folder / f"weather_cancelled_by_type_{draw}.csv")

        all_draws_treatments_mean.append(pd.Series(df_all_years_treatments_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_mean.append(pd.Series(df_all_years_weather_delayed_mean.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean.append(
            pd.Series(df_all_years_weather_cancelled_mean.sum(), name=f'Draw {draw}'))

        all_draws_treatments_lower.append(pd.Series(df_all_years_treatments_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_lower.append(pd.Series(df_all_years_weather_delayed_lower.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_lower.append(
            pd.Series(df_all_years_weather_cancelled_lower.sum(), name=f'Draw {draw}'))

        all_draws_treatments_upper.append(pd.Series(df_all_years_treatments_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_delayed_upper.append(pd.Series(df_all_years_weather_delayed_upper.sum(), name=f'Draw {draw}'))
        all_draws_weather_cancelled_upper.append(
            pd.Series(df_all_years_weather_cancelled_upper.sum(), name=f'Draw {draw}'))

        all_draws_treatments_mean_1000.append(pd.Series(df_treatments_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_delayed_mean_1000.append(
            pd.Series(df_weather_delayed_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_weather_cancelled_mean_1000.append(
            pd.Series(df_weather_cancelled_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))

        if draw == 0:
            baseline_treatments_by_year = df_all_years_treatments_mean.copy()
            baseline_weather_delayed_by_year = df_all_years_weather_delayed_mean.copy()
            baseline_weather_cancelled_by_year = df_all_years_weather_cancelled_mean.copy()
            baseline_population = df_all_years_data_population_mean.copy()

        plt.close(fig)

    # Combine all draws
    df_treatments_all_draws_mean = pd.concat(all_draws_treatments_mean, axis=1)
    df_weather_delayed_all_draws_mean = pd.concat(all_draws_weather_delayed_mean, axis=1)
    df_weather_cancelled_all_draws_mean = pd.concat(all_draws_weather_cancelled_mean, axis=1)

    df_treatments_all_draws_lower = pd.concat(all_draws_treatments_lower, axis=1)
    df_weather_delayed_all_draws_lower = pd.concat(all_draws_weather_delayed_lower, axis=1)
    df_weather_cancelled_all_draws_lower = pd.concat(all_draws_weather_cancelled_lower, axis=1)

    df_treatments_all_draws_upper = pd.concat(all_draws_treatments_upper, axis=1)
    df_weather_delayed_all_draws_upper = pd.concat(all_draws_weather_delayed_upper, axis=1)
    df_weather_cancelled_all_draws_upper = pd.concat(all_draws_weather_cancelled_upper, axis=1)

    df_treatments_all_draws_mean_1000 = pd.concat(all_draws_treatments_mean_1000, axis=1)
    df_weather_delayed_all_draws_mean_1000 = pd.concat(all_draws_weather_delayed_mean_1000, axis=1)
    df_weather_cancelled_all_draws_mean_1000 = pd.concat(all_draws_weather_cancelled_mean_1000, axis=1)

    treatments_totals_mean = df_treatments_all_draws_mean.sum()
    weather_delayed_totals_mean = df_weather_delayed_all_draws_mean.sum()
    weather_cancelled_totals_mean = df_weather_cancelled_all_draws_mean.sum()

    treatments_totals_lower = df_treatments_all_draws_lower.sum()
    treatments_totals_upper = df_treatments_all_draws_upper.sum()
    weather_delayed_totals_lower = df_weather_delayed_all_draws_lower.sum()
    weather_delayed_totals_upper = df_weather_delayed_all_draws_upper.sum()
    weather_cancelled_totals_lower = df_weather_cancelled_all_draws_lower.sum()
    weather_cancelled_totals_upper = df_weather_cancelled_all_draws_upper.sum()

    treatments_totals_err = np.array([
        treatments_totals_mean - treatments_totals_lower,
        treatments_totals_upper - treatments_totals_mean
    ])

    weather_delayed_totals_err = np.array([
        weather_delayed_totals_mean - weather_delayed_totals_lower,
        weather_delayed_totals_upper - weather_delayed_totals_mean
    ])

    weather_cancelled_totals_err = np.array([
        weather_cancelled_totals_mean - weather_cancelled_totals_lower,
        weather_cancelled_totals_upper - weather_cancelled_totals_mean
    ])

    # ── Final treatment-type breakdown plot ───────────────────────────────────

    target_year_final = max_year
    target_period_final = (Date(2025, 1, 1), Date(target_year_final, 12, 31))
    scenario_labels_final = ["No disruption", "Baseline", "Worst Case"]
    scenario_indices_final = [0, 1, 2]

    def get_counts_of_hsi_by_treatment_id(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(
            pd.Timestamp(str(target_period_final[0])),
            pd.Timestamp(str(target_period_final[1]))
        )]
        _counts_by_treatment_id = _df['TREATMENT_ID'].apply(pd.Series).sum().astype(int)
        return _counts_by_treatment_id.groupby(level=0).sum()

    def get_counts_of_hsi_by_short_treatment_id(_df):
        _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

    # 1. Extract HSI data for all scenarios
    final_data = {}
    final_data_with_ci = {}
    for i, draw in enumerate(scenario_indices_final):
        result_data_full = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
                do_scaling=False
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]
        final_data[scenario_labels_final[i]] = result_data_full['mean']
        final_data_with_ci[scenario_labels_final[i]] = {
            'mean': result_data_full['mean'].copy(),
            'lower': result_data_full['lower'].copy(),
            'upper': result_data_full['upper'].copy()
        }

    # 2. Extract population for each scenario
    def get_population_for_scaling(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(
            pd.Timestamp(str(target_period_final[0])),
            pd.Timestamp(str(target_period_final[1]))
        )]
        numeric_df = _df.drop(columns=["female", "male"], errors="ignore")
        population_mean = numeric_df.sum(numeric_only=True).mean()
        return pd.Series(population_mean, name="population")

    population_by_scenario = {}
    for i, draw in enumerate(scenario_indices_final):
        pop = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_scaling,
                do_scaling=False,
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]
        population_by_scenario[scenario_labels_final[i]] = float(pop['mean'].iloc[0])  # extract scalar


    # 3. Normalise to per 100,000 population
    for scenario_label in scenario_labels_final:
        pop = population_by_scenario[scenario_label]
        for stat in ['mean', 'lower', 'upper']:
            final_data_with_ci[scenario_label][stat] = (
                final_data_with_ci[scenario_label][stat] / pop * 100_000
            )
        final_data[scenario_label] = final_data_with_ci[scenario_label]['mean']
        print(final_data)

    # 4. Build df_final from normalised data
    df_final = pd.DataFrame(final_data).fillna(0)
    df_final = df_final[df_final.index.map(lambda x: isinstance(x, str) and (x.endswith('*') or '_' in x))]
    df_final.to_csv(output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_{suffix}.csv")
    scenario_colors = {'No disruption': '#0081a7', 'Baseline': '#FEB95F', 'Worst Case': '#f07167'}
    offsets = {'No disruption': -0.25, 'Baseline': 0.0, 'Worst Case': 0.25}

    # ── Cancelled and delayed HSIs by treatment type ──────────────────────────

    def get_cancelled_by_short_treatment_id(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(
            pd.Timestamp(str(target_period_final[0])),
            pd.Timestamp(str(target_period_final[1]))
        )]
        if len(_df) == 0 or 'TREATMENT_ID' not in _df.columns:
            return pd.Series(dtype=float)
        counts = _df['TREATMENT_ID'].value_counts()
        short_id = counts.index.map(lambda x: x.split('_')[0] + "*")
        return counts.groupby(by=short_id).sum()

    def get_delayed_by_short_treatment_id(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(
            pd.Timestamp(str(target_period_final[0])),
            pd.Timestamp(str(target_period_final[1]))
        )]
        if len(_df) == 0 or 'TREATMENT_ID' not in _df.columns:
            return pd.Series(dtype=float)
        counts = _df['TREATMENT_ID'].value_counts()
        short_id = counts.index.map(lambda x: x.split('_')[0] + "*")
        return counts.groupby(by=short_id).sum()

    cancelled_data = {}
    delayed_data = {}
    for i, draw in enumerate(scenario_indices_final):
        if scenario_labels_final[i] == 'No disruption':
            continue

        cancelled_result = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_cancelled_HSI_Event_full_info',
                custom_generate_series=get_cancelled_by_short_treatment_id,
                do_scaling=False,
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]
        cancelled_data[scenario_labels_final[i]] = cancelled_result['mean'].copy() / float(
            population_by_scenario[scenario_labels_final[i]]) * 100_000

        delayed_result = summarize(
            extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_delayed_HSI_Event_full_info',
                custom_generate_series=get_delayed_by_short_treatment_id,
                do_scaling=False,
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]
        delayed_data[scenario_labels_final[i]] = delayed_result['mean'].copy() / float(
            population_by_scenario[scenario_labels_final[i]]) * 100_000

    df_cancelled = pd.DataFrame(cancelled_data).fillna(0)
    df_cancelled = df_cancelled[df_cancelled.index.map(lambda x: isinstance(x, str) and (x.endswith('*') or '_' in x))]
    df_delayed = pd.DataFrame(delayed_data).fillna(0)
    df_delayed = df_delayed[df_delayed.index.map(lambda x: isinstance(x, str) and (x.endswith('*') or '_' in x))]

    # Plot cancelled and delayed side by side
    fig_cd, axes_cd = plt.subplots(1, 2, figsize=(18, len(df_cancelled.index) * 0.5 + 2), sharey=True)

    for ax, df_plot, title in zip(
        axes_cd,
        [df_cancelled, df_delayed],
        ['Weather-cancelled HSIs per 100,000', 'Weather-delayed HSIs per 100,000']
    ):
        y_positions_cd = np.arange(len(df_plot.index))
        treatment_labels_cd = [t.replace("*", "") for t in df_plot.index]
        offsets_cd = {'Baseline': -0.2, 'Worst Case': 0.2}

        for scenario_label in ['Baseline', 'Worst Case']:
            if scenario_label not in df_plot.columns:
                continue
            color = scenario_colors[scenario_label]
            offset = offsets_cd[scenario_label]
            values = df_plot[scenario_label].reindex(df_plot.index).fillna(0)

            ax.barh(
                y_positions_cd + offset,
                values,
                height=0.35,
                color=color,
                label=scenario_label,
                alpha=0.8,
            )

        ax.set_yticks(y_positions_cd)
        ax.set_yticklabels(treatment_labels_cd, fontsize=10)
        ax.set_xlabel(title, fontsize=11)
        ax.legend(title='Scenario', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    fig_cd.tight_layout()
    fig_cd.savefig(
        output_folder / f"{PREFIX_ON_FILENAME}_Cancelled_Delayed_by_TreatmentType_{suffix}.png",
        dpi=300)
    plt.close(fig_cd)
    # 5. Calculate significance vs "No disruption" for each treatment type
    treatment_significance = {}
    for scenario_label in ["Baseline", "Worst Case"]:
        treatment_significance[scenario_label] = {}
        for treatment in df_final.index:
            ref_mean = final_data_with_ci['No disruption']['mean'].get(treatment, 0)
            ref_lower = final_data_with_ci['No disruption']['lower'].get(treatment, 0)
            ref_upper = final_data_with_ci['No disruption']['upper'].get(treatment, 0)

            cmp_mean = final_data_with_ci[scenario_label]['mean'].get(treatment, 0)
            cmp_lower = final_data_with_ci[scenario_label]['lower'].get(treatment, 0)
            cmp_upper = final_data_with_ci[scenario_label]['upper'].get(treatment, 0)

            ref_se = (ref_upper - ref_lower) / (2 * 1.96) if ref_upper > ref_lower else 0
            cmp_se = (cmp_upper - cmp_lower) / (2 * 1.96) if cmp_upper > cmp_lower else 0

            diff = abs(cmp_mean - ref_mean)
            pooled_se = np.sqrt(ref_se ** 2 + cmp_se ** 2)

            if pooled_se > 0 and diff > 0:
                z_score = diff / pooled_se
                p_value = 2 * (1 - stats.norm.cdf(z_score))
                if p_value < 0.001:
                    star = '***'
                elif p_value < 0.01:
                    star = '**'
                elif p_value < 0.05:
                    star = '*'
                else:
                    star = ''
            else:
                star = ''
            treatment_significance[scenario_label][treatment] = star

    # 6. Stacked bar chart
    fig_final, ax_final = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(len(scenario_labels_final))

    for treatment in df_final.index:
        color = get_color_short_treatment_id(str(treatment))
        if not isinstance(color, str):
            color = '#888888'
        values = df_final.loc[treatment]
        ax_final.bar(scenario_labels_final, values, bottom=bottom,
                     color=color,
                     label=treatment)
        bottom += values.values

    ax_final.set_ylabel("HSIs per 100,000 population", fontsize=12)
    ax_final.set_xlabel("Scenario", fontsize=12)

    handles, labels = ax_final.get_legend_handles_labels()
    clean_labels = [l.replace("*", "") for l in labels]
    original_index = df_final.index
    clean_index = original_index.str.replace("*", "", regex=False)
    clean_to_original = dict(zip(clean_index, original_index))

    new_labels = []
    for label in clean_labels:
        original = clean_to_original.get(label, label)
        stars_by_scenario = [
            f"{sc}: {treatment_significance[sc].get(original, '')}"
            for sc in ["Baseline", "Worst Case"]
            if treatment_significance[sc].get(original, '')
        ]
        if stars_by_scenario:
            new_labels.append(f"{label} ({', '.join(stars_by_scenario)})")
        else:
            new_labels.append(label)

    ax_final.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left',
                    title='Treatment Type\nvs No disruption\n(* p<0.05, ** p<0.01, *** p<0.001)',
                    fontsize=9)
    fig_final.tight_layout()
    fig_final.savefig(
        output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_StackedBar_{suffix}.png",
        dpi=300)
    plt.close(fig_final)

    # 7. Difference dot plot vs "No disruption"
    fig_diff, ax_diff = plt.subplots(figsize=(12, len(df_final.index) * 0.5 + 2))
    scenario_colors = {'No disruption': '#0081a7', 'Baseline': '#FEB95F', 'Worst Case': '#f07167'}
    offsets = {'No disruption': -0.25, 'Baseline': 0.0, 'Worst Case': 0.25}
    y_positions = np.arange(len(df_final.index))
    treatment_labels = [t.replace("*", "") for t in df_final.index]

    for scenario_label in ['Baseline', 'Worst Case']:
        color = scenario_colors[scenario_label]
        offset = offsets[scenario_label]

        ref_means = final_data_with_ci['No disruption']['mean'].reindex(df_final.index).fillna(0)
        cmp_means = final_data_with_ci[scenario_label]['mean'].reindex(df_final.index).fillna(0)
        cmp_lowers = final_data_with_ci[scenario_label]['lower'].reindex(df_final.index).fillna(0)
        cmp_uppers = final_data_with_ci[scenario_label]['upper'].reindex(df_final.index).fillna(0)

        diffs = cmp_means - ref_means
        xerr = np.array([
            np.clip(cmp_means.values - cmp_lowers.values, 0, None),
            np.clip(cmp_uppers.values - cmp_means.values, 0, None)
        ])

        ax_diff.errorbar(
            x=diffs.values,
            y=y_positions + offset,
            xerr=xerr,
            fmt='o', color=color, label=scenario_label,
            capsize=3, markersize=5, linewidth=1.2,
        )

        # Add significance stars next to each point
        x_offset = max(abs(diffs)) * 0.02 if max(abs(diffs)) > 0 else 0.01
        for j, treatment in enumerate(df_final.index):
            star = treatment_significance[scenario_label].get(treatment, '')
            if star:
                ax_diff.text(
                    diffs.iloc[j] + x_offset,
                    y_positions[j] + offset,
                    star,
                    ha='left', va='center', fontsize=9,
                    color=color, fontweight='bold'
                )

    ax_diff.axvline(0, color='black', linewidth=1, linestyle='--', label='No disruption')
    ax_diff.set_yticks(y_positions)
    ax_diff.set_yticklabels(treatment_labels, fontsize=10)
    ax_diff.set_xlabel("Difference in HSIs per 100,000 population vs No Disruption", fontsize=12)
    ax_diff.legend(title='Scenario', fontsize=10)
    ax_diff.grid(axis='x', alpha=0.3)
    ax_diff.invert_yaxis()
    fig_diff.tight_layout()
    fig_diff.savefig(
        output_folder / f"{PREFIX_ON_FILENAME}_Final_Treatments_DiffPlot_{suffix}.png",
        dpi=300)
    plt.close(fig_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
