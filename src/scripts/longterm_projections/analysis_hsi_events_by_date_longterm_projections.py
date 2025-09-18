import argparse
from pathlib import Path
import difflib

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_color_short_treatment_id,
    get_scenario_info,
    load_pickled_dataframes,
    get_filtered_treatment_ids,
    summarize,
)

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)
min_year = 2020
max_year = 2070
spacing_of_years = 1
number_of_runs = 10

scenario_names = ["Status Quo", "HTM Scale-up", "Worsening Lifestyle Factors", "Improving Lifestyle Factors", "Maximal Healthcare \nProvision",]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167', '#9A348E']
def drop_outside_period(_df, target_period):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*target_period)])
def create_non_overlapping_positions(y_values, min_gap_ratio=0.01):
    """
    Adjust y positions to prevent overlaps while minimizing displacement
    Uses relative gap based on data range to handle large value ranges
    """
    y_range = max(y_values) - min(y_values)
    min_gap = max(y_range * min_gap_ratio, 0.05)

    indexed_values = [(y, i) for i, y in enumerate(y_values)]
    indexed_values.sort()

    adjusted_positions = [0] * len(y_values)

    for i, (original_y, original_idx) in enumerate(indexed_values):
        if i == 0:
            adjusted_positions[original_idx] = original_y
        else:
            prev_y = max(adjusted_positions[indexed_values[j][1]] for j in range(i))
            adjusted_positions[original_idx] = max(original_y, prev_y + min_gap)

    return adjusted_positions
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



def figure9_distribution_of_hsi_event_all_years_line_graph(results_folder: Path, output_folder: Path,
                                                           resourcefilepath: Path, min_year, max_year):
    """ 'Figure 9': The Trend of HSI_Events that occur by date."""

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig9_{stub}_{draw}.png"  # noqa: E731

        all_years_data = {}
        all_years_data_population_mean = {}

        for target_year in target_year_sequence:
            target_period = (
                Date(target_year, 1, 1),
                Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

            def get_counts_of_hsi_by_treatment_id(_df):
                """Get the counts of the short TREATMENT_IDs occurring"""
                _counts_by_treatment_id = _df \
                    .loc[pd.to_datetime(_df['date']).between(*target_period), 'TREATMENT_ID'] \
                    .apply(pd.Series) \
                    .sum() \
                    .astype(int)
                return _counts_by_treatment_id.groupby(level=0).sum()

            def get_counts_of_hsi_by_short_treatment_id(_df):
                """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
                _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
                _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
                return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

            result_data = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='HSI_Event',
                    custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            all_years_data[target_year] = result_data['mean']
            def get_population_for_year(_df):
                """Returns the population in the year of interest"""
                _df['date'] = pd.to_datetime(_df['date'])

                # Filter the DataFrame based on the target period
                filtered_df = _df.loc[_df['date'].between(*target_period)]
                numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')
                population_sum = numeric_df.sum(numeric_only=True)

                return population_sum
            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population['mean']

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years = pd.DataFrame(all_years_data)
        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_normalized = df_all_years.div(df_all_years.iloc[:, 0], axis=0)
        df_normalized_population = df_all_years_data_population_mean / \
                                   df_all_years_data_population_mean.iloc[:, 0].values[0]

        # Plotting
        causes = list(df_normalized.index)
        group_1 = ["Hiv*", "Tb*", "Malaria*"]
        group_2 = [cause for cause in causes if "Cancer" in cause]
        group_3 = ["CardioMetabolicDisorders*", "Copd*", "Depression*", "Epilepsy*", "Epi*", "FirstAttendance*"]
        group_4 = ["Alri*", "Measles*", "Schisto*"]
        group_5 = ["AntenatalCare*", "Contraception*", "DeliveryCare*", "PostnatalCare*"]
        other_causes = [cause for cause in causes if cause not in group_1 + group_2 + group_3 + group_4 + group_5]
        new_order = group_1 + group_2 + group_3 + group_4 + group_5 + other_causes
        df_all_years_ordered = df_normalized.loc[new_order]
        df_all_years_ordered = df_all_years_ordered.drop(index = "Schisto*")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')
        # Panel A: Raw counts = stacked
        df_all_years_ordered.T.plot.bar(
            stacked=True,
            ax=axes[0],
            color=[get_color_short_treatment_id(_label) for _label in df_all_years_ordered.index]
        )

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Counts of HSI Events', fontsize=12)
        axes[0].legend().set_visible(False)

        labels = [label.get_text() for label in axes[0].get_xticklabels()]
        new_labels = [label if i % 10 == 0 else '' for i, label in enumerate(labels)]
        new_labels.append('2070')
        tick_positions = list(range(len(new_labels)))

        axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_xticks(tick_positions)
        axes[0].set_xticklabels(new_labels, rotation=0)

        for i, treatment in enumerate(df_all_years_ordered.index):
            color = get_color_short_treatment_id(treatment)
            print(df_all_years_ordered)
            axes[1].plot(
                df_all_years_ordered.columns,
                df_all_years_ordered.loc[treatment],
                marker='o',
                color=color
            )

        final_y_values = [df_all_years_ordered.loc[treatment].iloc[-1]
                          for treatment in df_all_years_ordered.index]

        adjusted_y_positions = create_non_overlapping_positions(final_y_values, min_gap_ratio=0.01)
        for i, treatment in enumerate(df_all_years_ordered.index):
            color = get_color_short_treatment_id(treatment)
            original_y = df_all_years_ordered.loc[treatment].iloc[-1]
            adjusted_y = adjusted_y_positions[i]
            final_x = df_all_years_ordered.columns[-1]

            threshold = max(abs(original_y) * 0.01, 0.025)
            if abs(adjusted_y - original_y) > threshold:
                axes[1].plot([final_x, final_x + 0.5],
                             [original_y, adjusted_y],
                             color=color, linestyle='--', alpha=0.5, linewidth=0.8)

            axes[1].text(
                x=final_x + 1.0,
                y=adjusted_y,
                s=treatment,
                color=color,
                fontsize=8,
                va='center'
            )
        axes[1].hlines(
            y=1,
            xmin=min(axes[1].get_xlim()),
            xmax=max(axes[1].get_xlim()),
            color='black'
        )

        axes[1].plot(
            df_normalized_population.columns,
            df_normalized_population.iloc[0],
            color='black',
            linestyle='--',
            linewidth=4,
        )

        axes[1].text(
            x=df_normalized_population.columns[-1] + 1.0,
            y=df_normalized_population.iloc[0, -1],
            s='Population',
            color='black',
            fontsize=8,
            va='center'
        )

        axes[1].hlines(
            y=df_normalized_population.iloc[0, 0],
            xmin=min(axes[1].get_xlim()),
            xmax=max(axes[1].get_xlim()),
            color='black'
        )

        axes[1].tick_params(axis='both', which='major', labelsize=12)
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Fold change in demand', fontsize=12)

        df_all_years.to_csv(output_folder / f"HSI_events_treatment_ID_2020_2070_{draw}.csv")
        df_normalized.to_csv(output_folder / f"HSI_events_treatment_ID_normalized_2020_2070_{draw}.csv")

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_HSI_Events_by_TREATMENT_ID_All_Years_Panel_A_and_B'))
        plt.close(fig)


def figure10_minutes_per_cadre_and_treatment(results_folder: Path, output_folder: Path,
                                             resourcefilepath: Path, min_year, max_year):
    """ ' The Fraction of the time of each HCW used by each TREATMENT_ID (Short)"""
    target_year_sequence = range(min_year, max_year, spacing_of_years)
    all_draws_cadre = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_cadre_normalised = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_cadre_normalised_lower = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_cadre_normalised_upper = pd.DataFrame(columns=range(len(scenario_names)))

    all_draws_treatment = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_treatment_normalised = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_population = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_population_normalised = pd.DataFrame(columns=range(len(scenario_names)))
    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig10_{stub}_{draw}.png"  # noqa: E731
        appointment_time_table = pd.read_csv(
            resourcefilepath
            / 'healthsystem'
            / 'human_resources'
            / 'definitions'
            / 'ResourceFile_Appt_Time_Table.csv',
            index_col=["Appt_Type_Code", "Facility_Level", "Officer_Category"]
        )

        officer_cadres = appointment_time_table.index.levels[
            appointment_time_table.index.names.index("Officer_Category")
        ].to_list()
        appt_type_facility_level_officer_category_to_appt_time = (
            appointment_time_table.Time_Taken_Mins.to_dict()
        )

        all_years_data_cadre = {}
        all_years_data_cadre_lower = {}
        all_years_data_cadre_upper = {}

        all_years_data_treatment = {}
        all_years_data_population_mean = {}

        scenario_info = get_scenario_info(results_folder)
        for target_year in target_year_sequence:
            target_period = (
                Date(target_year, 1, 1), Date(target_year + spacing_of_years, 12, 31))

            def get_population_for_year(_df):
                """Returns the population in the year of interest"""
                _df['date'] = pd.to_datetime(_df['date'])

                # Filter the DataFrame based on the target period
                filtered_df = _df.loc[_df['date'].between(*target_period)]
                numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')
                population_sum = numeric_df.sum(numeric_only=True)

                return population_sum
            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population['mean']

            # Initialize aggregation variables
            cadre_to_total_time = {}
            cadre_to_total_time_lower = {}
            cadre_to_total_time_upper = {}
            module_id_to_total_time = {}
            total_runs = scenario_info["number_of_draws"] * scenario_info["runs_per_draw"]

            # Loop through all draws and runs
            for run in range(scenario_info["runs_per_draw"]):

                    real_population_scaling_factor = load_pickled_dataframes(results_folder, draw, run, 'tlo.methods.population')['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]
                    hsi_event_key_to_event_details = load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.healthsystem.summary"
                    )["tlo.methods.healthsystem.summary"]["hsi_event_details"]
                    hsi_event_key_to_event_details = hsi_event_key_to_event_details["hsi_event_key_to_event_details"]
                    hsi_event_key_to_counts = load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.healthsystem.summary"
                    )["tlo.methods.healthsystem.summary"]["hsi_event_counts"]
                    hsi_event_key_to_counts = hsi_event_key_to_counts[
                        hsi_event_key_to_counts['date'].between(target_period[0], target_period[1])
                    ]
                    hsi_event_key_to_counts['hsi_event_key_to_counts'] = {
                        key: {inner_key: inner_value * real_population_scaling_factor
                              for inner_key, inner_value in value.items()}
                        for key, value in hsi_event_key_to_counts['hsi_event_key_to_counts'].items()
                    }
                    hsi_event_key_to_counts = hsi_event_key_to_counts['hsi_event_key_to_counts']

                    # Loop through all hsi_event details
                    for hsi_event_code, hsi_event_details in hsi_event_key_to_event_details[0].items():
                        appt_footprint = hsi_event_details["appt_footprint"]
                        if len(appt_footprint) != 0:
                            for appt_type, appt_number in appt_footprint:  # could be more than one per footprint
                                module_name = hsi_event_details["module_name"]
                                facility_level = hsi_event_details["facility_level"]
                                treatment_id = hsi_event_details["treatment_id"].split("_")[0]
                                hsi_count = hsi_event_key_to_counts.iloc[1].get(str(hsi_event_code), 0)

                                # Calculate the time for each officer cadre
                                for cadre in officer_cadres:
                                    time_for_appointment_officer_facility = appt_type_facility_level_officer_category_to_appt_time.get(
                                        (appt_type, facility_level, cadre),
                                        0  # default to 0 if not found
                                    )

                                    # Add time to cadre dictionary
                                    if cadre not in cadre_to_total_time:
                                        cadre_to_total_time[cadre] = 0
                                    cadre_to_total_time[cadre] += (time_for_appointment_officer_facility * hsi_count
                                                                   * appt_number)

                                    # Add time to module dictionary
                                    if treatment_id not in module_id_to_total_time:
                                        module_id_to_total_time[treatment_id] = 0
                                    module_id_to_total_time[treatment_id] += (time_for_appointment_officer_facility
                                                                              * hsi_count * appt_number)

            # Average the results over all runs and draws
            for cadre in cadre_to_total_time:
                # get std
                std_deviation = cadre_to_total_time[cadre].std()
                std_error = std_deviation/np.sqrt(total_runs)
                z_value = st.norm.ppf(1 - (1. - 0.95) / 2.)

                cadre_to_total_time[cadre] /= total_runs

                cadre_to_total_time_lower[cadre] = cadre_to_total_time[cadre] - z_value * std_error
                cadre_to_total_time_upper[cadre] = cadre_to_total_time[cadre] + z_value * std_error
            for module_name in module_id_to_total_time:
                module_id_to_total_time[module_name] /= total_runs

            all_years_data_cadre[target_year] = cadre_to_total_time
            all_years_data_cadre_lower[target_year] = cadre_to_total_time_lower
            all_years_data_cadre_upper[target_year] = cadre_to_total_time_upper

            all_years_data_treatment[target_year] = module_id_to_total_time

            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population['mean']

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_cadre = pd.DataFrame(all_years_data_cadre)
        df_all_years_cadre_lower = pd.DataFrame(all_years_data_cadre_lower)
        df_all_years_cadre_upper = pd.DataFrame(all_years_data_cadre_upper)
        df_all_years_treatment = pd.DataFrame(all_years_data_treatment)
        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)

        # Normalizing by the first column (first year in the sequence)
        df_normalized_cadre = df_all_years_cadre.div(df_all_years_cadre.iloc[:, 0], axis=0)
        df_normalized_cadre_lower = df_all_years_cadre_lower.div(df_all_years_cadre_lower.iloc[:, 0], axis=0)
        df_normalized_cadre_upper = df_all_years_cadre_lower.div(df_all_years_cadre_upper.iloc[:, 0], axis=0)
        df_normalized_treatment = df_all_years_treatment.div(df_all_years_treatment.iloc[:, 0], axis=0)
        df_normalized_population = df_all_years_data_population_mean.div(df_all_years_data_population_mean.iloc[:, 0],
                                                                         axis=0)

        # save final year
        all_draws_cadre[draw] = df_all_years_cadre.iloc[:, -1]
        all_draws_cadre_normalised[draw] = df_normalized_cadre.iloc[:, -1]
        all_draws_cadre_normalised_lower[draw] = df_normalized_cadre_lower.iloc[:, -1]
        all_draws_cadre_normalised_upper[draw] = df_normalized_cadre_upper.iloc[:, -1]
        all_draws_treatment[draw] = df_all_years_treatment.iloc[:, -1]
        all_draws_treatment_normalised[draw] = df_normalized_treatment.iloc[:, -1]
        all_draws_population[draw] = df_all_years_data_population_mean.iloc[:, -1]
        all_draws_population_normalised[draw] = df_normalized_population.iloc[:, -1]
        # Plotting

        def create_non_overlapping_positions_cadre(y_values, min_gap=0.05):
            indexed_values = [(y, i) for i, y in enumerate(y_values)]
            indexed_values.sort()

            adjusted_positions = [0] * len(y_values)

            for i, (original_y, original_idx) in enumerate(indexed_values):
                if i == 0:
                    adjusted_positions[original_idx] = original_y
                else:
                    prev_y = max(adjusted_positions[indexed_values[j][1]] for j in range(i))
                    adjusted_positions[original_idx] = max(original_y, prev_y + min_gap)

            return adjusted_positions

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        df_all_years_cadre = df_all_years_cadre.drop("Dental")
        df_normalized_cadre = df_normalized_cadre.drop("Dental")

        cadres = list(df_normalized_cadre.index)
        cmap = plt.get_cmap("tab10")
        colors = {cadre: cmap(i % cmap.N) for i, cadre in enumerate(cadres)}

        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                     fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                     fontsize=14, va='top', ha='right')
        df_all_years_cadre.T.plot.bar(stacked=True, ax=axes[0])

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Time Spent (Minutes)', fontsize=12)
        axes[0].legend(title='Cadre', loc='upper left',fontsize='small', title_fontsize='small' )
        axes[0].grid(False)

        labels = [label.get_text() for label in axes[0].get_xticklabels()]
        new_labels = [label if i % 10 == 0 else '' for i, label in enumerate(labels)]
        new_labels.append('2070')
        tick_positions = list(range(len(new_labels)))

        axes[0].set_xticks(tick_positions)
        axes[0].set_xticklabels(new_labels, rotation=0)
        axes[0].tick_params(axis='both', which='major', labelsize=12)

        for i, cadre in enumerate(df_normalized_cadre.index):
            axes[1].plot(
                df_normalized_cadre.columns,
                df_normalized_cadre.loc[cadre],
                marker='o',
                label=cadre,
                color = colors[cadre],

            )

        axes[1].plot(
            df_normalized_population.columns,
            df_normalized_population.iloc[0],
            color='black',
            linestyle='--',
            linewidth=4,
            label='Population'
        )

        final_y_values = [df_normalized_cadre.loc[cadre].iloc[-1]
                          for cadre in df_normalized_cadre.index]
        population_final_y = df_normalized_population.iloc[0, -1]
        final_y_values.append(population_final_y)

        adjusted_y_positions = create_non_overlapping_positions_cadre(final_y_values, min_gap=0.15)
        cadres = list(df_normalized_cadre.index)
        cmap = plt.get_cmap("tab10")
        colors = {cadre: cmap(i % cmap.N) for i, cadre in enumerate(cadres)}

        for i, cadre in enumerate(df_normalized_cadre.index):
            original_y = df_normalized_cadre.loc[cadre].iloc[-1]
            adjusted_y = adjusted_y_positions[i]
            final_x = df_normalized_cadre.columns[-1]

            axes[1].plot([final_x, final_x + 3.5],
                         [original_y, adjusted_y],
                         linestyle='--', alpha=0.6, linewidth=1.0, color = colors[cadre])

            axes[1].text(
                final_x + 4,
                adjusted_y,
                cadre,
                color = colors[cadre],
                fontsize=9,
                va='center'
            )

        population_adjusted_y = adjusted_y_positions[-1]
        population_original_y = df_normalized_population.iloc[0, -1]
        pop_final_x = df_normalized_population.columns[-1]

        axes[1].plot([pop_final_x, pop_final_x + 3.5],
                     [population_original_y, population_adjusted_y],
                     color='black', linestyle='--', alpha=0.6, linewidth=1.0)

        axes[1].text(
            pop_final_x + 4,
            population_adjusted_y,
            "Population",
            color='black',
            fontsize=9,
            va='center'
        )

        current_xlim = axes[1].get_xlim()
        axes[1].set_xlim(current_xlim[0], current_xlim[1] + 4)

        axes[1].tick_params(axis='both', which='major', labelsize=12)
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Fold change in demand', fontsize=12)
        axes[1].legend().set_visible(False)
        axes[1].grid(False)
        df_all_years_cadre.to_csv(output_folder / f"HSI_time_per_cadre_2020_2070_{draw}.csv")
        df_normalized_cadre.to_csv(output_folder / f"HSI_time_per_cadre_normalized_2020_2070_{draw}.csv")
        fig.tight_layout()
        fig.savefig(make_graph_file_name(f"Time_HSI_Events_by_Cadre_All_Years_Panel_A_and_B_{draw}"))
    # Plotting - treatments
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    all_draws_treatment_per_1000 = all_draws_treatment.div(all_draws_population.loc['total'], axis=1) * 1000

    treatments = list(all_draws_treatment_per_1000.index)
    cmap = plt.get_cmap("tab20")
    colors = {treatment: cmap(i % cmap.N) for i, treatment in enumerate(treatments)}
    all_draws_treatment_per_1000.T.plot.bar(
        stacked=True, ax=axes[0], legend=False,
        color=[colors[treatment] for treatment in treatments]
    )
    axes[0].set_ylabel('Time spent (Minutes) per 1,000 population', fontsize=12)
    axes[0].set_xlabel('Scenario', fontsize=12)
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    def create_non_overlapping_positions_treatment(y_values, min_gap=0.05):
        """
        Adjust y positions to prevent overlaps while minimizing displacement
        """
        # Create pairs of (original_y, index)
        indexed_values = [(y, i) for i, y in enumerate(y_values)]
        indexed_values.sort()  # Sort by y value

        adjusted_positions = [0] * len(y_values)

        for i, (original_y, original_idx) in enumerate(indexed_values):
            if i == 0:
                # First label keeps its position
                adjusted_positions[original_idx] = original_y
            else:
                # For subsequent labels, ensure minimum gap from previous
                prev_y = max(adjusted_positions[indexed_values[j][1]] for j in range(i))
                adjusted_positions[original_idx] = max(original_y, prev_y + min_gap)

        return adjusted_positions

    # Plot all lines and scatter points first
    for treatment in all_draws_treatment_normalised.index:
        color = colors[treatment]
        axes[1].scatter(all_draws_treatment_normalised.columns, all_draws_treatment_normalised.loc[treatment],
                        marker='o', color=color, label=treatment)
        axes[1].plot(all_draws_treatment_normalised.columns,
                     all_draws_treatment_normalised.loc[treatment],
                     color=color, alpha=0.5)

    # # Plot population line
    # axes[1].plot(
    #     all_draws_population_normalised.columns,
    #     all_draws_population_normalised.iloc[0, :],
    #     color='black',
    #     linewidth=4,
    #     linestyle='--',
    #     label='Population'
    # )

    final_y_values = [all_draws_treatment_normalised.loc[treatment].iloc[-1]
                      for treatment in all_draws_treatment_normalised.index]
    population_final_y = all_draws_population_normalised.iloc[0, -1]
    final_y_values.append(population_final_y)

    adjusted_y_positions = create_non_overlapping_positions_treatment(final_y_values, min_gap=0.08)

    for i, treatment in enumerate(all_draws_treatment_normalised.index):
        color = colors[treatment]
        original_y = all_draws_treatment_normalised.loc[treatment].iloc[-1]
        adjusted_y = adjusted_y_positions[i]
        final_x = all_draws_treatment_normalised.columns[-1]

        axes[1].plot([final_x, final_x + 0.3],
                     [original_y, adjusted_y],
                     color=color, linestyle='--', alpha=0.6, linewidth=1.0)

        axes[1].text(
            x=final_x + 0.4,
            y=adjusted_y,
            s=treatment,
            color=color,
            fontsize=8,
            va='center'
        )

    # population_adjusted_y = adjusted_y_positions[-1]
    # population_original_y = all_draws_population_normalised.iloc[0, -1]
    # population_final_x = all_draws_population_normalised.columns[-1]
    #
    # axes[1].plot([population_final_x, population_final_x + 0.3],
    #              [population_original_y, population_adjusted_y],
    #              color='black', linestyle='--', alpha=0.6, linewidth=1.0)
    #
    # axes[1].text(
    #     x=population_final_x + 0.4,
    #     y=population_adjusted_y,
    #     s='Population',
    #     color='black',
    #     fontsize=8,
    #     va='center'
    # )

    axes[1].hlines(y=1, xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()), color='black')

    axes[1].set_ylabel('Fold change in time spent', fontsize=12)
    axes[1].set_xlabel('Scenario', fontsize=12)
    axes[1].set_xticks(all_draws_treatment_normalised.columns)
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout()
    fig.savefig(output_folder / "Time_HSI_Events_by_Treatment_combined.png")

    ## cadre
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    cols = list(all_draws_cadre.columns)
    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    all_draws_cadre = all_draws_cadre[cols]
    all_draws_cadre.rename(
        columns=dict(zip(all_draws_cadre.columns, range(len(scenario_names)))),
        inplace=True
    )

    cols = list(all_draws_cadre_normalised.columns)
    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    all_draws_cadre_normalised = all_draws_cadre_normalised[cols]
    all_draws_cadre_normalised.rename(
        columns=dict(zip(all_draws_cadre_normalised.columns, range(len(scenario_names)))),
        inplace=True
    )
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')


    all_draws_cadre_per_1000 = all_draws_cadre.div(all_draws_population.loc['total'], axis=1) * 1000
    all_draws_cadre_per_1000 = all_draws_cadre_per_1000.drop("Dental")
    all_draws_cadre_normalised = all_draws_cadre_normalised.drop("Dental")
    cadres = list(all_draws_cadre_normalised.index)
    cmap = plt.get_cmap("tab10")
    colors = {cadre: cmap(i % cmap.N) for i, cadre in enumerate(cadres)}

    all_draws_cadre_per_1000.T.plot.bar(stacked=True, ax=axes[0], legend=False)
    axes[0].set_ylabel('Time Spent (Minutes) per 1,000 population', fontsize=12)
    axes[0].set_xlabel('Scenario', fontsize=12)
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend(title='Cadre', loc='upper left', fontsize='small', title_fontsize='small')

    for cadre in cadres:
        color = colors[cadre]
        axes[1].scatter(all_draws_cadre_normalised.columns, all_draws_cadre_normalised.loc[cadre],
                        marker='o', color=color, label=cadre)
        axes[1].plot(all_draws_cadre_normalised.columns, all_draws_cadre_normalised.loc[cadre],
                     alpha=0.5, color=color)

    final_y_values = [all_draws_cadre_normalised.loc[cadre].iloc[-1]
                          for cadre in all_draws_cadre_normalised.index]

    adjusted_y_positions = create_non_overlapping_positions(final_y_values, min_gap_ratio=0.1)

    for i, cadre in enumerate(all_draws_cadre_normalised.index):
        color = colors[cadre]
        original_y = all_draws_cadre_normalised.loc[cadre].iloc[-1]
        adjusted_y = adjusted_y_positions[i]
        final_x = all_draws_cadre_normalised.columns[-1]

        axes[1].plot([final_x, final_x + 0.3],
                         [original_y, adjusted_y],
                         color=color, linestyle='--', alpha=0.5, linewidth=0.8)

        axes[1].text(
            x=final_x + 0.4,
            y=adjusted_y,
            s=cadre,
            color=color,
            fontsize=8,
            va='center'
        )

    axes[1].set_ylabel('Fold change in time spent', fontsize=12)
    axes[1].set_xlabel('Scenario', fontsize=12)
    axes[1].set_xticks(all_draws_cadre_normalised.columns)
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].hlines(y=1, xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()), color='black')

    fig.tight_layout()
    fig.savefig(output_folder / "Time_HSI_Events_by_Cadre_combined.png")



    # Plot for WHO AFRO

    y_min = all_draws_cadre_normalised.min().min() * 0.5
    y_max = 11.9

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]}, figsize=(8, 6))

    y_err = [
        all_draws_cadre_normalised - all_draws_cadre_normalised_lower,
        all_draws_cadre_normalised_upper - all_draws_cadre_normalised
    ]

    # Define the break in the Y-axis (skip range 5-10)
    break_low, break_high = 9.5, 11.8
    ax1.set_xticks([])
    for cadre in all_draws_cadre_normalised.index:
        if cadre == "Dental":
            continue

        x_vals = all_draws_cadre_normalised.columns
        y_vals = all_draws_cadre_normalised.loc[cadre]
        y_err_low = abs(y_err[0].loc[cadre])
        y_err_high = abs(y_err[1].loc[cadre])

        mask_low = y_vals < break_low
        ax2.scatter(x_vals[mask_low], y_vals[mask_low], label=cadre, alpha=0.5)
        ax2.errorbar(x_vals[mask_low], y_vals[mask_low], yerr=[y_err_low[mask_low], y_err_high[mask_low]], fmt='none',
                     capsize=3)

        mask_high = y_vals > break_high
        ax1.scatter(x_vals[mask_high], y_vals[mask_high], label=cadre, alpha=0.5)
        ax1.errorbar(x_vals[mask_high], y_vals[mask_high], yerr=[y_err_low[mask_high], y_err_high[mask_high]],
                     fmt='none', capsize=3)

    ax1.set_ylim(break_high, y_max)  # Top subplot (skipping 5-10)
    ax2.set_ylim(y_min, break_low)  # Bottom subplot

    # Hide top and bottom spines where the break occurs
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    d = 0.01
    kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False)
    ax1.plot((-d, d), (-d, d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, d), **kwargs)
    ax1.xaxis.set_visible(False)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Fold change in time spent")
    ax2.set_ylabel("")

    ax2.axhline(y=1, color='black', linestyle='--')
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(scenario_names, rotation=45)

    # Combine legends
    ax1.legend(ncol=2, fontsize='small')

    plt.tight_layout()
    fig.savefig(output_folder / "WHO_AFRO_Relative_2070_to_2020.png")


def table2_description_of_coarse_hsi_events(
    results_folder: Path,
    output_folder: Path,
    min_year: int,
    max_year: int,
):
    """Table 1 extended: Compute mean HSI event counts over time and plot bar and normalized trends"""
    treatment_group = {
        'Alri*': 'RMNCH',
        'AntenatalCare*': 'RMNCH',
        'BladderCancer*': 'NCDs',
        'BreastCancer*': 'NCDs',
        'CardioMetabolicDisorders*': 'NCDs',
        'Contraception*': 'RMNCH',
        'Copd*': 'NCDs',
        'DeliveryCare*': 'RMNCH',
        'Depression*': 'NCDs',
        'Diarrhoea*': 'RMNCH',
        'Epi*': 'RMNCH',
        'Epilepsy*': 'NCDs',
        'FirstAttendance*': 'First Attendance',
        'Hiv*': 'HIV/AIDS',
        'Inpatient*': 'Inpatient',
        'Malaria*': 'Malaria',
        'Measles*': 'RMNCH',
        'OesophagealCancer*': 'NCDs',
        'OtherAdultCancer*': 'NCDs',
        'PostnatalCare*': 'RMNCH',
        'ProstateCancer*': 'NCDs',
        'Rti*': 'Transport Injuries',
        'Schisto*': 'RMNCH',
        'Tb*': 'TB (non-AIDS)',
        'Undernutrition*': 'RMNCH',
    }
    target_year_sequence = range(min_year, max_year, spacing_of_years)
    event_counts_by_year = {draw: {} for draw in range(len(scenario_names))}
    all_draws_population = pd.DataFrame(columns=range(len(scenario_names)))
    all_draws_population_normalised = pd.DataFrame(columns=range(len(scenario_names)))
    for draw in range(len(scenario_names)):
        all_years_data_population_mean = {}

        for target_year in target_year_sequence:
            target_period = (
                Date(target_year, 1, 1),
                Date(target_year, 12, 31)
            )

            def get_population_for_year(_df):
                """Returns the population in the year of interest"""
                _df['date'] = pd.to_datetime(_df['date'])

                # Filter the DataFrame based on the target period
                filtered_df = _df.loc[_df['date'].between(*target_period)]
                numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')
                population_sum = numeric_df.sum(numeric_only=True)

                return population_sum

            def get_num_treatments_group(_df):
                    """Return the number of treatments by short treatment id (total within the TARGET_PERIOD)"""
                    _df = _df.loc[pd.to_datetime(_df.date).between(*target_period), 'TREATMENT_ID'].apply(
                        pd.Series).sum()
                    print( _df.index)
                    _df.index = _df.index.map(lambda x: "_".join(x.split('_')[:2]) + "*")
                    #_df = _df.rename(index=treatment_group)
                    _df = _df.groupby(level=0).sum()
                    return _df

            num_treatments_group = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event_non_blank_appt_footprint',
                custom_generate_series=get_num_treatments_group,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population['mean']
            event_counts_by_year[draw][target_year] = (num_treatments_group['mean'] / result_data_population[
                'mean'].values) * 1000
        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)


        df_normalized_population = df_all_years_data_population_mean.div(df_all_years_data_population_mean.iloc[:, 0],
                                                                         axis=0)

        all_draws_population[draw] = df_all_years_data_population_mean.iloc[:, -1]
        all_draws_population_normalised[draw] = df_normalized_population.iloc[:, -1]
        df_event_counts = pd.DataFrame(event_counts_by_year[draw]).fillna(0)

        df_event_counts.to_csv(output_folder / f"{PREFIX_ON_FILENAME}_EventCountsOverTime_draw{draw}.csv")

        # Select top 10 events in the final year
        top_events = df_event_counts.iloc[:, -1].sort_values(ascending=False).head(10).index
        df_top_events = df_event_counts.loc[top_events]

        # Panel A: Raw counts (bar plot)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')

        event_list = list(df_top_events.index)
        cmap = plt.get_cmap("tab10")
        colors = {event: cmap(i % cmap.N) for i, event in enumerate(event_list)}


        df_top_events.T.plot.bar(stacked=True, ax=axes[0], color=colors)
        axes[0].set_xlabel("Year", fontsize=12)
        axes[0].set_ylabel("Number of Appointments per 1000", fontsize=12)
        labels = [label.get_text() for label in axes[0].get_xticklabels()]
        new_labels = [label if i % 10 == 0 else '' for i, label in enumerate(labels)]
        new_labels.append('2070')
        tick_positions = list(range(len(new_labels)))
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_xticks(tick_positions)
        axes[0].set_xticklabels(new_labels, rotation=0)

        axes[0].tick_params(axis='both', labelsize=11)
        axes[0].legend(title='Treatment ID', loc='upper left',bbox_to_anchor=(0, 1.25), fontsize=8, title_fontsize=9, ncol = 2)

        # Panel B: Normalized (line plot)
        df_top_events_normalized = df_top_events.div(df_top_events.iloc[:, 0], axis=0)

        for event in df_top_events_normalized.index:
            savgol_filter(df_top_events_normalized.loc[event].to_numpy(), window_length=5, polyorder=2),

            color =colors[event]
            axes[1].plot(df_top_events_normalized.columns,  savgol_filter(df_top_events_normalized.loc[event].to_numpy(), window_length=5, polyorder=2),
                         marker='o', color=color)
        final_y_values = [df_top_events_normalized.loc[event].iloc[-1] for event in df_top_events_normalized.index]
        adjusted_y_positions = create_non_overlapping_positions(final_y_values, min_gap_ratio=0.1)

        for i, event in enumerate(df_top_events_normalized.index):
                original_y = df_top_events_normalized.loc[event].iloc[-1]
                adjusted_y = adjusted_y_positions[i]
                final_x = df_top_events_normalized.columns[-1]

                axes[1].plot([final_x, final_x + 3.5],
                             [original_y, adjusted_y],
                             linestyle='--', alpha=0.6, linewidth=1.0, color=colors[event])

                axes[1].text(
                    final_x + 4,
                    adjusted_y,
                    event,
                    color=colors[event],
                    fontsize=9,
                    va='center'
                )

        print(df_normalized_population)
        axes[1].plot(
            df_normalized_population.columns.astype(int),
            df_normalized_population.loc['total'].values,
            color='black',
            linestyle='--',
            alpha=0.6,
            linewidth=1.0
        )
        axes[1].text(
            x=df_normalized_population.columns[-1] + 1.0,
            y=df_normalized_population.iloc[0, -1],
            s='Population',
            color='black',
            fontsize=8,
            va='center'
        )
        print(df_top_events_normalized.columns[0])
        axes[1].hlines(
                y=1,
                xmin=int(df_top_events_normalized.columns[0]),
                xmax=int(df_top_events_normalized.columns[-1]),
                color='black'
            )
        axes[1].set_xlabel("Year", fontsize=12)
        axes[1].set_ylabel("Fold Change", fontsize=12)
        axes[1].legend().set_visible(False)
        axes[1].tick_params(axis='both', labelsize=11)

        fig.tight_layout()
        fig.savefig(output_folder / f"{PREFIX_ON_FILENAME}_HSI_EventCounts_OverTime_Draw{draw}.png")
        plt.show()




def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = '/Users/rem76/PycharmProjects/TLOmodel/resources'):
    """Description of the usage of healthcare system resources."""

    # figure9_distribution_of_hsi_event_all_years_line_graph(
    #     results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath,
    #     min_year=min_year, max_year=max_year)
    #
    # figure10_minutes_per_cadre_and_treatment(
    #     results_folder=results_folder,
    #     output_folder=output_folder,
    #     resourcefilepath=resourcefilepath,
    #     min_year=min_year,
    #     max_year=max_year
    # ),
    # table1_description_of_hsi_events(
    #     results_folder= results_folder,
    #     output_folder= output_folder,
    #     min_year=min_year,
    #     max_year=max_year
    # ),
    table2_description_of_coarse_hsi_events(
        results_folder=results_folder,
        output_folder=output_folder,
        min_year=min_year,
        max_year=max_year
    ),

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()
    #results_folder = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-04T082106Z")

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
