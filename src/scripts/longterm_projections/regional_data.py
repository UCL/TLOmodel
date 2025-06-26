import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
    _standardize_short_treatment_id,
    bin_hsi_event_details,
    compute_mean_across_runs,
    extract_results,
    get_coarse_appt_type,
    get_color_short_treatment_id,
    get_scenario_info,
    load_pickled_dataframes,
    order_of_short_treatment_ids,
    plot_stacked_bar_chart,
    squarify_neat,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)
min_year = 2020
max_year = 2070
spacing_of_years = 1


scenario_names = ["Status Quo", "Maximal Healthcare \nProvision", "HTM Scale-up", "Negative Lifestyle Change", "Positive Lifestyle Change"]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167', '#9A348E']
def drop_outside_period(_df, target_period):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*target_period)])

rename_dict = {  # For legend labels
    'ALRI': 'Lower respiratory infections',
    'Bladder Cancer': 'Cancer (Bladder)',
    'Breast Cancer': 'Cancer (Breast)',
    'COPD': 'COPD',
    'Depression': 'Depression / Self-harm',
    'Diarrhoea': 'Diarrhoea',
    'Epilepsy': 'Epilepsy',
    'HIV': 'AIDS',
    'Malaria': 'Malaria',
    'Measles': 'Measles',
    'Oesophageal Cancer': 'Cancer (Oesophagus)',
    'Other Adult Cancers': 'Cancer (Other)',
    'Prostate Cancer': 'Cancer (Prostate)',
    'RTI': 'Transport Injuries',
    'Schisto': 'Schistosomiasis',
    'TB': 'TB',
    'chronic_ischemic_hd': 'Heart Disease',
    'chronic_kidney_disease': 'Kidney Disease',
    'chronic_lower_back_pain': 'Lower Back Pain',
    'diabetes': 'Diabetes',
    'hypertension': 'Hypertension'
}


def table1_description_of_hsi_events(
    results_folder: Path,
    output_folder: Path, year_range, n_runs_per_draw,
):
    """ `Table 1`: A summary table of all the HSI Events seen in the simulation.
    This is similar to that created by `hsi_events.py` but records all the different forms (levels/appt-type) that
    an HSI Event can take.
    It averages over runs within each draw and saves one CSV per draw.
    """
    for draw in range(len(scenario_names)):
        per_run_summaries = []

        for run in range(n_runs_per_draw):
            log = load_pickled_dataframes(results_folder, draw, run)
            h = pd.DataFrame(
                log['tlo.methods.healthsystem.summary']['hsi_event_details'].iloc[0]['hsi_event_key_to_event_details']
            ).T
            print(h.columns)
            # Re-order columns & sort; Remove 'HSI_' prefix from event name
            h = h[['module_name', 'treatment_id', 'event_name', 'facility_level', 'appt_footprint', 'beddays_footprint']]
            h = h.sort_values(['module_name', 'treatment_id', 'event_name', 'facility_level']).reset_index(drop=True)
            h['event_name'] = h['event_name'].str.replace('HSI_', '')

            # Rename columns
            h = h.rename(columns={
                "module_name": 'Module',
                "treatment_id": 'TREATMENT_ID',
                "event_name": 'HSI Event',
                "facility_level": 'Facility Level',
                "appt_footprint": 'Appointment Types',
                "beddays_footprint": 'Bed-Days',
            })

            # Reformat 'Appointment Types' and 'Bed-types' column
            def reformat_col(col):
                return col.apply(pd.Series) \
                    .applymap(lambda x: x[0], na_action='ignore') \
                    .apply(lambda row: ', '.join(_r for _r in row.sort_values() if not pd.isnull(_r)), axis=1)

            h['Appointment Types'] = h['Appointment Types'].pipe(reformat_col)
            h["Bed-Days"] = h["Bed-Days"].pipe(reformat_col)

            # Group by key columns and combine unique Appointment Types / Bed-Days
            def combine_unique(series):
                return ', '.join(sorted(set(filter(None, sum([s.split(', ') for s in series if s], [])))))

            h_summary = h.groupby(['Module', 'TREATMENT_ID', 'HSI Event', 'Facility Level']).agg(
                Occurrences=('Module', 'count'),
                Appointment_Types=('Appointment Types', combine_unique),
                Bed_Days=('Bed-Days', combine_unique)
            ).reset_index()

            per_run_summaries.append(h_summary)
            real_population_scaling_factor = \
            load_pickled_dataframes(results_folder, draw, run, 'tlo.methods.population')['tlo.methods.population'][
                'scaling_factor']['scaling_factor'].values[0]
        hsi_event_counts_per_draw = []

        for draw in range(len(scenario_names)):
            per_run_event_counts = defaultdict(list)  # store per-run event counts per year

            for run in range(n_runs_per_draw):
                real_population_scaling_factor = load_pickled_dataframes(
                    results_folder, draw, run, 'tlo.methods.population'
                )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]

                hsi_event_counts = load_pickled_dataframes(
                    results_folder, draw, run, "tlo.methods.healthsystem.summary"
                )["tlo.methods.healthsystem.summary"]["hsi_event_counts"]

                # Loop through years
                for year in year_range:
                    target_period = (Date(year, 1, 1), Date(year + spacing_of_years, 12, 31))

                    # Subset for this period
                    subset = hsi_event_counts[hsi_event_counts['date'].between(target_period[0], target_period[1])]

                    if not subset.empty:
                        # sum counts within this period
                        summed_counts = defaultdict(float)
                        for event_counts_dict in subset['hsi_event_key_to_counts']:
                            for key, count in event_counts_dict.items():
                                summed_counts[key] += count * real_population_scaling_factor

                        # Store this run's scaled summed counts for this year
                        per_run_event_counts[year].append(summed_counts)

            # Now average over runs per year for this draw
            averaged_event_counts_per_year = {}
            for year, run_counts_list in per_run_event_counts.items():
                if run_counts_list:  # if any runs
                    # collect all unique event keys
                    all_event_keys = set(k for d in run_counts_list for k in d.keys())
                    averaged_counts = {
                        key: np.mean([d.get(key, 0) for d in run_counts_list])
                        for key in all_event_keys
                    }
                    averaged_event_counts_per_year[year] = averaged_counts

            hsi_event_counts_per_draw.append(averaged_event_counts_per_year)

        # Combine runs for this draw
        combined_runs = pd.concat(per_run_summaries)

        # Average Occurrences over runs
        h_summary_avg = combined_runs.groupby(['Module', 'TREATMENT_ID', 'HSI Event', 'Facility Level']).agg(
            Mean_Occurrences=('Occurrences', 'mean'),
            Appointment_Types=('Appointment_Types', lambda s: ', '.join(sorted(set(filter(None, sum([x.split(', ') for x in s if x], [])))))),
            Bed_Days=('Bed_Days', lambda s: ', '.join(sorted(set(filter(None, sum([x.split(', ') for x in s if x], []))))))
        ).reset_index()

        # Put something in for blanks/nan (helps with importing into Excel/Word)
        h_summary_avg = h_summary_avg.fillna('-').replace('', '-')

        # Save table as csv for this draw (averaged over its runs)
        h_summary_avg.to_csv(
            output_folder / f"{PREFIX_ON_FILENAME}_Treatment_ID_{year_range}_draw{draw}.csv",
            index=False
        )

def figure9_distribution_of_hsi_event_all_years_line_graph(results_folder: Path, output_folder: Path,
                                                           resourcefilepath: Path, min_year, max_year):
    """ 'Figure 9': The Trend of HSI_Events that occur by date."""

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig9_{stub}_{draw}.png"  # noqa: E731

        all_years_data = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1),
                Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

            def get_counts_of_hsi_by_treatment_id(_df):
                """Get the counts of the short TREATMENT_IDs occurring"""
                _counts_by_treatment_id = _df \
                    .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
                    .apply(pd.Series) \
                    .sum() \
                    .astype(int)
                return _counts_by_treatment_id.groupby(level=0).sum()

            result_data = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='HSI_Event',
                    custom_generate_series=get_counts_of_hsi_by_treatment_id,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data[target_year] = result_data['mean']

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years = pd.DataFrame(all_years_data)
        df_all_years.to_csv(output_folder/f"HSI_events_treatment_ID_2020_2070_{draw}.csv")


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Description of the usage of healthcare system resources."""
    table1_description_of_hsi_events(
        results_folder=results_folder, output_folder=output_folder,
        year_range=range(min_year,max_year), n_runs_per_draw = 10)

    figure9_distribution_of_hsi_event_all_years_line_graph(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath,
        min_year=min_year, max_year=max_year)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
