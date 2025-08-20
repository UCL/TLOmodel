import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from longterm_projections import LongRun
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)

min_year = 2020
max_year = 2070
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Status Quo", "HTM Scale-up", "Worsening Lifestyle Factors", "Improving Lifestyle Factors", "Maximal Healthcare \nProvision",]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167', '#9A348E']

def create_non_overlapping_positions(y_values, min_gap_ratio=0.08):
    """
    Adjust y positions to prevent overlaps while minimizing displacement
    Uses relative gap based on data range to handle large value ranges
    """
    y_range = max(y_values) - min(y_values)
    min_gap = max(y_range * min_gap_ratio, 0.1)

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
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """
    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_mode.scenario_impact_of_mode import (
            ImpactOfHealthSystemMode,
        )
        e = LongRun()
        return tuple(e._scenarios.keys())

    param_names = get_parameter_names_from_scenario_file()
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()


    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    def get_num_deaths_by_cause_label_female(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .loc[_df['sex'] == 'F'] \
            .groupby(_df['label']) \
            .size()

    def get_num_deaths_by_cause_label_male(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .loc[_df['sex'] == 'M'] \
            .groupby(_df['label']) \
            .size()
    def get_num_dalys_by_cause_label(_df):
        """Return total number of DALYS (Stacked) by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
            .drop(columns=['date', 'sex', 'age_range', 'year']) \
            .sum()

    def get_population_for_year(_df):
        """Returns the population in the year of interest"""
        _df['date'] = pd.to_datetime(_df['date'])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')
        population_sum = numeric_df.sum(numeric_only=True)

        return population_sum

    def get_population_for_year_female(_df):
        """Returns the population in the year of interest"""
        _df['date'] = pd.to_datetime(_df['date'])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=['male', 'total'], errors='ignore')
        population_sum = numeric_df.sum(numeric_only=True)
        return population_sum

    def get_population_for_year_male(_df):
        """Returns the population in the year of interest"""
        _df['date'] = pd.to_datetime(_df['date'])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=['female', 'total'], errors='ignore')
        population_sum = numeric_df.sum(numeric_only=True)

        return population_sum

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    all_draws_deaths_mean = []
    all_draws_deaths_lower = []
    all_draws_deaths_upper = []

    all_draws_dalys_mean = []
    all_draws_dalys_lower = []
    all_draws_dalys_upper = []

    normalized_DALYs = []
    all_draws_deaths_mean_1000 = []
    all_draws_deaths_lower_1000 = []
    all_draws_deaths_upper_1000 = []

    all_draws_dalys_mean_1000 = []
    all_draws_dalys_lower_1000 = []
    all_draws_dalys_upper_1000 = []

    all_draws_deaths_mean_1000_male = []
    all_draws_deaths_mean_1000_female = []

    for draw in range(len(scenario_names)):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_deaths_mean = {}
        all_years_data_deaths_upper= {}
        all_years_data_deaths_lower = {}

        all_years_data_dalys_mean = {}
        all_years_data_dalys_upper = {}
        all_years_data_dalys_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        all_years_data_deaths_mean_male= {}
        all_years_data_deaths_mean_female = {}

        all_years_data_population_mean_female = {}
        all_years_data_population_mean_male = {}


        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

            # %% Quantify the health gains associated with all interventions combined.

            # Absolute Number of Deaths and DALYs
            result_data_deaths = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='death',
                custom_generate_series=get_num_deaths_by_cause_label,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_deaths_mean[target_year] = result_data_deaths['mean']
            all_years_data_deaths_lower[target_year] = result_data_deaths['lower']
            all_years_data_deaths_upper[target_year] = result_data_deaths['upper']

            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthburden',
                    key='dalys_stacked_by_age_and_time',
                    custom_generate_series=get_num_dalys_by_cause_label,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            result_data_dalys = result_data_dalys.drop(index='Schistosomiasis')
            all_years_data_dalys_mean[target_year] = result_data_dalys['mean']
            all_years_data_dalys_lower[target_year] = result_data_dalys['lower']
            all_years_data_dalys_upper[target_year] = result_data_dalys['upper']

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
            all_years_data_population_lower[target_year] = result_data_population['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']


            # females deaths and population
            result_data_deaths_female = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='death',
                custom_generate_series=get_num_deaths_by_cause_label_female,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            result_data_population_female = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year_female,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean_female[target_year] = result_data_population_female['mean']
            all_years_data_deaths_mean_female[target_year] = result_data_deaths_female['mean']

            # males deaths and population
            result_data_deaths_male = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='death',
                custom_generate_series=get_num_deaths_by_cause_label_male,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

            result_data_population_male = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year_male,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean_male[target_year] = result_data_population_male['mean']
            all_years_data_deaths_mean_male[target_year] = result_data_deaths_male['mean']

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)

        df_all_years_deaths_mean = pd.DataFrame(all_years_data_deaths_mean)
        df_all_years_deaths_lower = pd.DataFrame(all_years_data_deaths_lower)
        df_all_years_deaths_upper = pd.DataFrame(all_years_data_deaths_upper)
        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        df_all_years_data_population_mean_female = pd.DataFrame(all_years_data_population_mean_female)
        df_all_years_data_population_mean_male = pd.DataFrame(all_years_data_population_mean_male)

        df_all_years_data_deaths_mean_female = pd.DataFrame(all_years_data_deaths_mean_female)
        df_all_years_data_deaths_mean_male = pd.DataFrame(all_years_data_deaths_mean_male)


        df_death_per_1000_mean_female = df_all_years_data_deaths_mean_female.div(df_all_years_data_population_mean_female.iloc[0, 0], axis = 0) * 1000
        df_death_per_1000_mean_male = df_all_years_data_deaths_mean_male.div(df_all_years_data_population_mean_male.iloc[0, 0], axis=0) * 1000

        df_normalized_population = df_all_years_data_population_mean.div(df_all_years_data_population_mean.iloc[:, 0],
                                                                         axis=0)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')
        # Panel A: Deaths
        for i, condition in enumerate(df_all_years_deaths_mean.index):
            axes[0].plot(df_all_years_deaths_mean.columns, df_all_years_deaths_mean.loc[condition], marker='o',
                         label=condition, color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                 df_all_years_deaths_mean.index][i])
        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].grid(False)

        # Panel B: DALYs
        for i, condition in enumerate(df_all_years_DALYS_mean.index):
            axes[1].plot(df_all_years_DALYS_mean.columns, df_all_years_DALYS_mean.loc[condition], marker='o', label=condition,
                         color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                df_all_years_DALYS_mean.index][i])

        axes[1].plot(
                df_normalized_population.columns,
                df_normalized_population.iloc[0],
                color='black',
                linestyle='--',
                linewidth=4,
            )

        axes[1].text(
                x=df_normalized_population.columns[-1] + 0.5,
                y=df_normalized_population.iloc[0, -1],
                s='Population',
                color='black',
                fontsize=8,
                va='center'
            )
        axes[1].set_title('Panel B: DALYs by cause')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].legend(title='Condition', bbox_to_anchor=(0.9, 1), loc='upper left')
        axes[1].grid()

        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B'))
        plt.close(fig)

        # NORMALIZED DEATHS AND DALYS - TO 2020
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')
        df_death_normalized_mean = df_all_years_deaths_mean.div(df_all_years_deaths_mean.iloc[:, 0], axis=0)
        df_DALY_normalized_mean = df_all_years_DALYS_mean.div(df_all_years_DALYS_mean.iloc[:, 0], axis=0)

        df_all_years_deaths_mean.to_csv(output_folder / f"cause_of_death_2020_2070_{draw}.csv")
        df_all_years_DALYS_mean.to_csv(output_folder / f"cause_of_dalys_2020_2070_{draw}.csv")

        df_death_normalized_mean.to_csv(output_folder / f"cause_of_death_normalized_2020_{draw}.csv")
        df_DALY_normalized_mean.to_csv(output_folder / f"cause_of_dalys_normalized_2020_{draw}.csv")

        for i, condition in enumerate(df_death_normalized_mean.index):
            axes[0].plot(df_death_normalized_mean.columns, df_death_normalized_mean.loc[condition], marker='o',
                         label=condition, color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                 df_all_years_deaths_mean.index][i])
        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Fold change in deaths compared to 2020')
        axes[0].grid()

        # Panel B: DALYs
        for i, condition in enumerate(df_DALY_normalized_mean.index):
            axes[1].plot(df_DALY_normalized_mean.columns, df_DALY_normalized_mean.loc[condition], marker='o', label=condition,
                         color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                df_DALY_normalized_mean.index][i])
        axes[1].set_title('Panel B: DALYs by cause')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Fold change in DALYs compared to 2020')
        axes[1].legend(title='Condition', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid()

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Normalized_Panel_A_and_B'))
        plt.close(fig)

        ## BARPLOTS STACKED PER 1000
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))  # Changed to 3 subplots
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                     fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                     fontsize=14, va='top', ha='right')

        axes[2].text(-0.1, 1.05, '(C)', transform=axes[2].transAxes,
                     fontsize=14, va='top', ha='right')

        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        # Panel A: Deaths (Stacked bar plot)
        causes = list(df_daly_per_1000_mean.index)
        group_1 = ["AIDS", "TB (non-AIDS)", "Malaria"]
        group_2 = [cause for cause in causes if "Cancer" in cause]
        group_3 = ["Depression / Self-harm", "Diabetes", "Epilepsy", "Lower Back Pain", "Heart Disease",
                   "Kidney Disease", "COPD"]
        group_4 = ["Lower respiratory infections", "Measles"]
        other_causes = [cause for cause in causes if cause not in group_1 + group_2 + group_3 + group_4]
        new_order = group_1 + group_2 + group_3 + group_4 + other_causes
        df_daly_per_1000_mean_ordered = df_daly_per_1000_mean.loc[new_order]

        df_daly_per_1000_mean_ordered.T.plot.bar(
            stacked=True,
            ax=axes[0],
            color=[get_color_cause_of_death_or_daly_label(_label) for _label in new_order]
        )

        handles, labels = axes[0].get_legend_handles_labels()

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_xticks(axes[0].get_xticks()[::10])
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].set_ylabel('Number of DALYs per 1000 people', fontsize=12)
        axes[0].legend().set_visible(False)
        axes[0].tick_params(axis='both', which='major', labelsize=12)

        # Panel B: NCDs
        panel_b_groups = group_2 + group_3 + other_causes

        line_handles_b = []

        for condition in panel_b_groups:
            if condition in df_DALY_normalized_mean.index:
                color = get_color_cause_of_death_or_daly_label(condition)
                (line,) = axes[1].plot(
                    df_DALY_normalized_mean.columns,
                    df_DALY_normalized_mean.loc[condition],
                    marker='o',
                    label=condition,
                    color=color,
                    markersize=4,
                )
                line_handles_b.append((condition, line))

        # Adjust labels for Panel B
        final_y_values_b = [df_DALY_normalized_mean.loc[condition].iloc[-1] for condition in panel_b_groups
                            if condition in df_DALY_normalized_mean.index]

        adjusted_y_positions_b = create_non_overlapping_positions(final_y_values_b, min_gap_ratio=0.07)

        # Add labels with adjusted positions for Panel B
        for i, condition in enumerate([c for c in panel_b_groups if c in df_DALY_normalized_mean.index]):
            original_y = df_DALY_normalized_mean.loc[condition].iloc[-1]
            adjusted_y = adjusted_y_positions_b[i]
            x = df_DALY_normalized_mean.columns[-1]
            color = get_color_cause_of_death_or_daly_label(condition)

            axes[1].plot([x, x + 4], [original_y, adjusted_y],
                         color=color, linestyle='--', alpha=0.8, linewidth=1.5)

            axes[1].text(
                x + 5,
                adjusted_y,
                condition,
                color=color,
                fontsize=9,
                va='center'
            )

        # Add population line to Panel B
        population_adjusted_y_b = adjusted_y * 1.07
        population_original_y = df_normalized_population.iloc[0, -1]
        population_final_x = df_normalized_population.columns[-1]
        axes[1].plot(
            df_normalized_population.columns,
            df_normalized_population.iloc[0, :],
            color='black',
            linewidth=4,
            linestyle='--',
        )
        axes[1].plot([population_final_x, population_final_x + 4],
                     [population_original_y, population_adjusted_y_b],
                     color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        axes[1].text(
            population_final_x + 5,
            population_adjusted_y_b,
            'Population',
            color='black',
            fontsize=9,
            va='center'
        )

        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Normalized DALYs', fontsize=12)
        axes[1].tick_params(axis='both', which='major', labelsize=12)

        # Panel C: IDs
        panel_c_groups = group_1 + group_4
        line_handles_c = []

        for condition in panel_c_groups:
            if condition in df_DALY_normalized_mean.index:
                color = get_color_cause_of_death_or_daly_label(condition)
                (line,) = axes[2].plot(
                    df_DALY_normalized_mean.columns,
                    df_DALY_normalized_mean.loc[condition],
                    marker='o',
                    label=condition,
                    color=color,
                    markersize=4,
                )
                line_handles_c.append((condition, line))

        # Adjust labels for Panel C
        final_y_values_c = [df_DALY_normalized_mean.loc[condition].iloc[-1] for condition in panel_c_groups
                            if condition in df_DALY_normalized_mean.index]

        adjusted_y_positions_c = create_non_overlapping_positions(final_y_values_c, min_gap_ratio=0.07)

        # Add labels with adjusted positions for Panel C
        for i, condition in enumerate([c for c in panel_c_groups if c in df_DALY_normalized_mean.index]):
            original_y = df_DALY_normalized_mean.loc[condition].iloc[-1]
            adjusted_y = adjusted_y_positions_c[i]
            x = df_DALY_normalized_mean.columns[-1]
            color = get_color_cause_of_death_or_daly_label(condition)

            axes[2].plot([x, x + 4], [original_y, adjusted_y],
                         color=color, linestyle='--', alpha=0.8, linewidth=1.5)

            axes[2].text(
                x + 5,
                adjusted_y,
                condition,
                color=color,
                fontsize=9,
                va='center'
            )

        # Add population line to Panel C
        population_adjusted_y_c = df_normalized_population.iloc[0, -1]
        axes[2].plot(
            df_normalized_population.columns,
            df_normalized_population.iloc[0, :],
            color='black',
            linewidth=4,
            linestyle='--',
        )
        axes[2].plot([population_final_x, population_final_x + 4],
                     [population_original_y, population_adjusted_y_c],
                     color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        axes[2].text(
            population_final_x + 5,
            population_adjusted_y_c,
            'Population',
            color='black',
            fontsize=9,
            va='center'
        )

        axes[2].set_xlabel('Year', fontsize=12)
        axes[2].set_ylabel('Normalized DALYs', fontsize=12)
        axes[2].tick_params(axis='both', which='major', labelsize=12)

        # Legend for Panel A (keeping the original legend structure)
        ordered_names = new_order
        name_to_handle = dict(line_handles_b + line_handles_c)
        ordered_handles = [name_to_handle[name] for name in ordered_names if name in name_to_handle]
        axes[0].legend(reversed(ordered_handles), reversed(ordered_names), title="Cause", loc='upper left', fontsize=8,
                       title_fontsize=9, ncol=2)

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_DALYs_and_normalized_by_condition_All_Years_Panel_A_B_C_Stacked_Rate'))
        ## BARPLOTS STACKED PER 1000
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                   fontsize=14, va='top', ha='right')

        axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                   fontsize=14,  va='top', ha='right')
        df_death_per_1000_mean = df_all_years_deaths_mean.div(df_all_years_data_population_mean.iloc[0, 0],
                                                              axis=0) * 1000
        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_death_per_1000_lower = df_all_years_deaths_lower.div(df_all_years_data_population_lower.iloc[0, 0],
                                                                axis=0) * 1000
        df_daly_per_1000_lower = df_all_years_DALYS_lower.div(df_all_years_data_population_lower.iloc[0, 0],
                                                              axis=0) * 1000
        df_death_per_1000_upper = df_all_years_deaths_upper.div(df_all_years_data_population_upper.iloc[0, 0],
                                                                axis=0) * 1000
        df_daly_per_1000_upper = df_all_years_DALYS_upper.div(df_all_years_data_population_upper.iloc[0, 0],
                                                              axis=0) * 1000


        #save cause of death to csv
        df_all_years_DALYS_mean.to_csv(output_folder/f"dalys_by_cause_rate_{draw}.csv")
        df_all_years_deaths_mean.to_csv(output_folder/f"deaths_by_cause_rate_{draw}.csv")

        # get proportion due to CMD, mental, and cancer
        target_conditions = [
            "COPD", "Cancer (Bladder)", "Cancer (Breast)", "Cancer (Oesophagus)",
            "Cancer (Other)", "Cancer (Prostate)", "Depression / Self-harm",
            "Diabetes", "Epilepsy", "Heart Disease", "Kidney Disease", "Lower Back Pain"
        ]
        df_all_years_DALYS_mean_filtered = df_all_years_DALYS_mean[df_all_years_DALYS_mean.index.isin(target_conditions)]
        proportion_DALYS_NCD = df_all_years_DALYS_mean_filtered.iloc[:, 1:].sum() / df_all_years_DALYS_mean.iloc[:, 1:].sum()
        proportion_DALYS_NCD.to_csv(output_folder/f"prop_DALYs_NCD_{draw}.csv", index=True)

        df_all_years_deaths_mean_filtered = df_all_years_deaths_mean[df_all_years_deaths_mean.index.isin(target_conditions)]
        proportion_deaths_NCD = df_all_years_deaths_mean_filtered.iloc[:, 1:].sum() / df_all_years_deaths_mean.iloc[:, 1:].sum()
        proportion_deaths_NCD.to_csv(output_folder/f"prop_deaths_NCD_{draw}.csv", index=True)


        normalized_DALYs.append(pd.Series(df_DALY_normalized_mean.iloc[:,-1], name=f'Draw {draw}'))
        all_years_data_dalys_mean = df_all_years_DALYS_mean.sum()
        all_years_data_deaths_mean = df_all_years_deaths_mean.sum()
        all_years_data_dalys_lower = df_all_years_DALYS_lower.sum()
        all_years_data_deaths_lower = df_all_years_deaths_lower.sum()
        all_years_data_dalys_upper = df_all_years_DALYS_upper.sum()
        all_years_data_deaths_upper = df_all_years_deaths_upper.sum()
        all_draws_deaths_mean.append(pd.Series(all_years_data_deaths_mean, name=f'Draw {draw}'))
        all_draws_dalys_mean.append(pd.Series(all_years_data_dalys_mean, name=f'Draw {draw}'))
        all_draws_deaths_lower.append(pd.Series(all_years_data_deaths_lower, name=f'Draw {draw}'))
        all_draws_dalys_lower.append(pd.Series(all_years_data_dalys_lower, name=f'Draw {draw}'))
        all_draws_deaths_upper.append(pd.Series(all_years_data_deaths_upper, name=f'Draw {draw}'))
        all_draws_dalys_upper.append(pd.Series(all_years_data_dalys_upper, name=f'Draw {draw}'))
        # only include 2070 as can't have cumulative per 1000?
        all_draws_dalys_mean_1000.append(
            df_daly_per_1000_mean.iloc[:, [0, -1]].rename(
                columns={df_daly_per_1000_mean.columns[0]: 'First', df_daly_per_1000_mean.columns[-1]: 'Last'}).assign(
                Draw=f'Draw {draw}')
        )
        all_draws_dalys_lower_1000.append(
            df_daly_per_1000_lower.iloc[:, [0, -1]].rename(columns={df_daly_per_1000_lower.columns[0]: 'First',
                                                                    df_daly_per_1000_lower.columns[-1]: 'Last'}).assign(
                Draw=f'Draw {draw}')
        )
        all_draws_dalys_upper_1000.append(
            df_daly_per_1000_upper.iloc[:, [0, -1]].rename(columns={df_daly_per_1000_upper.columns[0]: 'First',
                                                                    df_daly_per_1000_upper.columns[-1]: 'Last'}).assign(
                Draw=f'Draw {draw}')
        )

        all_draws_deaths_mean_1000.append(
            df_death_per_1000_mean.iloc[:, [0, -1]].rename(columns={df_death_per_1000_mean.columns[0]: 'First',
                                                                    df_death_per_1000_mean.columns[-1]: 'Last'}).assign(
                Draw=f'Draw {draw}')
        )
        all_draws_deaths_lower_1000.append(
            df_death_per_1000_lower.iloc[:, [0, -1]].rename(columns={df_death_per_1000_lower.columns[0]: 'First',
                                                                     df_death_per_1000_lower.columns[
                                                                         -1]: 'Last'}).assign(Draw=f'Draw {draw}')
        )
        all_draws_deaths_upper_1000.append(
            df_death_per_1000_upper.iloc[:, [0, -1]].rename(columns={df_death_per_1000_upper.columns[0]: 'First',
                                                                     df_death_per_1000_upper.columns[
                                                                         -1]: 'Last'}).assign(Draw=f'Draw {draw}')
        )

        # deaths by sex
        all_draws_deaths_mean_1000_male.append(
            df_death_per_1000_mean_male.iloc[:, [0, -1]].rename(
                columns={df_death_per_1000_mean_male.columns[0]: 'First',
                         df_death_per_1000_mean_male.columns[-1]: 'Last'}).assign(Draw=f'Draw {draw}')
        )
        all_draws_deaths_mean_1000_female.append(
            df_death_per_1000_mean_female.iloc[:, [0, -1]].rename(
                columns={df_death_per_1000_mean_female.columns[0]: 'First',
                         df_death_per_1000_mean_female.columns[-1]: 'Last'}).assign(Draw=f'Draw {draw}')
        )
    df_deaths_all_draws_mean = pd.concat(all_draws_deaths_mean, axis=1)
    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_deaths_all_draws_lower = pd.concat(all_draws_deaths_lower, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_deaths_all_draws_upper = pd.concat(all_draws_deaths_upper, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    df_deaths_all_draws_mean_1000 = pd.concat(all_draws_deaths_mean_1000, axis=1)
    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_deaths_all_draws_mean_1000_female = pd.concat(all_draws_deaths_mean_1000_female, axis=1)
    df_deaths_all_draws_mean_1000_male = pd.concat(all_draws_deaths_mean_1000_male, axis=1)

    normalized_DALYs = pd.concat(normalized_DALYs, axis = 1)

    # Plotting as bar charts
    deaths_totals_mean = df_deaths_all_draws_mean.sum()
    dalys_totals_mean = df_dalys_all_draws_mean.sum()
    deaths_totals_lower = df_deaths_all_draws_lower.sum()
    deaths_totals_upper = df_deaths_all_draws_upper.sum()
    dalys_totals_lower = df_dalys_all_draws_lower.sum()
    dalys_totals_upper = df_dalys_all_draws_upper.sum()

    # Convert sums to DataFrames to keep column structure
    deaths_totals_mean = df_deaths_all_draws_mean.sum().to_frame().T
    dalys_totals_mean = df_dalys_all_draws_mean.sum().to_frame().T
    deaths_totals_lower = df_deaths_all_draws_lower.sum().to_frame().T
    dalys_totals_lower = df_dalys_all_draws_lower.sum().to_frame().T
    deaths_totals_upper = df_deaths_all_draws_upper.sum().to_frame().T
    dalys_totals_upper = df_dalys_all_draws_upper.sum().to_frame().T

    # Function to reorder columns (move second column to the end) and rename
    def reorder_and_rename(df, scenario_names):
        cols = list(df.columns)
        second_col = cols[1]
        new_cols = cols[:1] + cols[2:] + [second_col]
        df = df[new_cols]
        df.columns = range(len(scenario_names))
        return df

    # Apply to all DataFrames
    deaths_totals_mean = reorder_and_rename(deaths_totals_mean, scenario_names)
    dalys_totals_mean = reorder_and_rename(dalys_totals_mean, scenario_names)
    deaths_totals_lower = reorder_and_rename(deaths_totals_lower, scenario_names)
    dalys_totals_lower = reorder_and_rename(dalys_totals_lower, scenario_names)
    deaths_totals_upper = reorder_and_rename(deaths_totals_upper, scenario_names)
    dalys_totals_upper = reorder_and_rename(dalys_totals_upper, scenario_names)

    deaths_totals_err = np.array([
        (deaths_totals_mean - deaths_totals_lower).to_numpy().flatten(),
        (deaths_totals_upper - deaths_totals_mean).to_numpy().flatten()
    ])
    dalys_totals_err = np.array([
        (dalys_totals_mean - dalys_totals_lower).to_numpy().flatten(),
        (dalys_totals_upper - dalys_totals_mean).to_numpy().flatten()
    ])
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].text(-0.05, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.05, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    # Panel A: Total Deaths
    axes[0].bar(
        range(len(deaths_totals_mean.columns)),
        deaths_totals_mean.to_numpy().flatten(),
        color=scenario_colours,
        yerr=deaths_totals_err,
        capsize=20
    )
    axes[0].set_title('Total Deaths (2020-2070)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total Deaths')
    axes[0].set_xticks(range(len(scenario_names)))
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # Panel B: Total DALYs
    axes[1].bar(
        range(len(dalys_totals_mean.columns)),
        dalys_totals_mean.to_numpy().flatten(),
        color=scenario_colours,
        yerr=dalys_totals_err,
        capsize=20
    )
    axes[1].set_title('Total DALYs (2020-2070)')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Total DALYs')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "total_deaths_and_dalys_all_draws.png")
    plt.close(fig)


    ## Per 1000 in 2020 and 2070
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    df_dalys_all_draws_mean_1000.drop(df_dalys_all_draws_mean_1000.columns[-1], axis=1, inplace=True)
    # Panel A: DALYs per 1,000 in 2070
    df_dalys_all_draws_mean_1000.T.plot.bar(
        stacked=True,
        ax=axes[0],
        color=[get_color_cause_of_death_or_daly_label(_label) for _label in df_dalys_all_draws_mean_1000.index],
        label=[label for label in df_all_years_DALYS_mean.index]
    )

    axes[0].set_title('Panel A: DALYs per 1,000 (2070)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('DALYs per 1,000')
    axes[0].set_xticks(range(len(scenario_names)))
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend()

    # Panel B: Fold change in DALYs compared to 2020
    label_positions = []
    y_offset = 0.05

    for i, condition in enumerate(df_DALY_normalized_mean.index):
        color = get_color_cause_of_death_or_daly_label(condition)
        y_values = df_DALY_normalized_mean.loc[condition]
        x_values = df_DALY_normalized_mean.columns

        axes[1].scatter(x_values, y_values, marker='o', label=condition, color=color)
        axes[1].plot(x_values, y_values, color=color, alpha=0.7)

        final_x = x_values[-1] + 5
        final_y = y_values.iloc[-1]

        # Adjust label y position to avoid overlap
        while any(abs(final_y - existing_y) < y_offset for existing_y in label_positions):
            final_y += y_offset
            final_x += 4

        label_positions.append(final_y)

        axes[1].text(
            x=final_x,
            y=final_y,
            s=condition,
            color=color,
            fontsize=8,
            va='center'
        )

    axes[1].set_ylabel('Fold change in DALYs compared to 2020')
    axes[1].set_xlabel('Scenario')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    #axes[1].legend(title='Cause', bbox_to_anchor=(1., 1), loc='upper left')
    axes[1].legend().set_visible(False)
    axes[1].grid(False)
    axes[1].set_title('Panel B: Relative DALYs (2070 vs 2020)')

    fig.savefig(make_graph_file_name(
        f'dalys_by_cause_and_total_dalys_all_cause_all_draws_relative_2070_2020'))
    plt.close(fig)



    #### DALYS COMBINED PLOT
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    df_dalys_all_draws_mean_1000_2070 = df_dalys_all_draws_mean_1000[['Last']]
    df_dalys_all_draws_mean_1000_2070.columns = range(len(scenario_names))
    cols = list(df_dalys_all_draws_mean_1000_2070.columns)

    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    df_dalys_all_draws_mean_1000_2070 = df_dalys_all_draws_mean_1000_2070[cols]
    df_dalys_all_draws_mean_1000_2070.columns = range(len(scenario_names))

    # df_dalys_all_draws_mean_1000_2070.rename(
    #     columns=dict(zip(df_dalys_all_draws_mean_1000_2070.columns, range(len(scenario_names)))),
    #     inplace=True
    # )
    #

    cols = list(normalized_DALYs.columns)
    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    normalized_DALYs = normalized_DALYs[cols]
    normalized_DALYs.columns = range(len(scenario_names))
    #
    # normalized_DALYs.rename(
    #     columns=dict(zip(normalized_DALYs.columns, range(len(scenario_names)))),
    #     inplace=True
    # )
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')
    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    axes[2].text(-0.1, 1.05, '(C)', transform=axes[2].transAxes,
                 fontsize=14, va='top', ha='right')

    causes = list(df_dalys_all_draws_mean_1000_2070.index)
    group_1 = ["AIDS", "TB (non-AIDS)", "Malaria"]
    group_2 = [cause for cause in causes if "Cancer" in cause]
    group_3 = ["Depression / Self-harm", "Diabetes", "Epilepsy", "Lower Back Pain", "Heart Disease", "Kidney Disease",
               "COPD", "Stroke"]
    group_4 = ["Lower respiratory infections", "Measles"]
    other_causes = [cause for cause in causes if cause not in group_1 + group_2 + group_3 + group_4]
    new_order = group_1 + group_2 + group_3 + group_4 + other_causes
    df_dalys_all_draws_mean_1000_2070 = df_dalys_all_draws_mean_1000_2070.loc[new_order]
    df_dalys_all_draws_mean_1000_2070.T.plot.bar(
        stacked=True,
        ax=axes[0],
        color=[get_color_cause_of_death_or_daly_label(_label) for _label in new_order],
        legend=False)

    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].set_ylabel('DALYs per 1,000 population')
    axes[0].set_xlabel('Scenario')
    axes[0].set_xticks(range(len(scenario_names)))
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend(ordered_handles, new_order, title="Cause",  bbox_to_anchor=(1.05, 1), ncol = 1, fontsize=8, title_fontsize=9)

    subset_b = group_2 + group_3 + other_causes
    final_y_values_b = [normalized_DALYs.loc[cause].iloc[-1] for cause in subset_b if cause in normalized_DALYs.index]
    adjusted_y_positions_b = create_non_overlapping_positions(final_y_values_b, min_gap_ratio=0.05)

    for i, cause in enumerate(subset_b):
        if cause in normalized_DALYs.index:
            if cause == 'AIDS':
                break
            color = get_color_cause_of_death_or_daly_label(cause)
            y_values = normalized_DALYs.loc[cause]
            axes[1].plot(normalized_DALYs.columns, y_values, marker='o', color=color)
            original_y = y_values.iloc[-1]
            adjusted_y = adjusted_y_positions_b[i]
            final_x = normalized_DALYs.columns[-1]

            axes[1].plot([final_x, final_x + 0.5], [original_y, adjusted_y],
                         color=color, linestyle='--', alpha=0.5, linewidth=0.8)
            axes[1].text(final_x + 0.75, adjusted_y, s=cause, color=color,
                         fontsize=8, va='center')

    axes[1].hlines(
        y=normalized_DALYs.loc['TB (non-AIDS)'][0],
        xmin=min(axes[1].get_xlim()),
        xmax=max(axes[1].get_xlim()),
        color='black'
    )

    axes[1].set_ylabel('Fold Change in DALYs per 1,000 Compared to 2020')
    axes[1].set_xlabel('Scenario')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].legend().set_visible(False)

    subset_c = group_1 + group_4
    final_y_values_c = [normalized_DALYs.loc[cause].iloc[-1] for cause in subset_c if cause in normalized_DALYs.index]
    adjusted_y_positions_c = create_non_overlapping_positions(final_y_values_c, min_gap_ratio=0.08)

    for i, cause in enumerate(subset_c):
        if cause in normalized_DALYs.index:
            color = get_color_cause_of_death_or_daly_label(cause)
            y_values = normalized_DALYs.loc[cause]
            axes[2].plot(normalized_DALYs.columns, y_values, marker='o', color=color)

            # connector + adjusted label
            original_y = y_values.iloc[-1]
            adjusted_y = adjusted_y_positions_c[i]
            final_x = normalized_DALYs.columns[-1]

            axes[2].plot([final_x, final_x + 0.5], [original_y, adjusted_y],
                         color=color, linestyle='--', alpha=0.5, linewidth=0.8)
            axes[2].text(final_x + 0.75, adjusted_y, s=cause, color=color,
                         fontsize=8, va='center')

    axes[2].hlines(
        y=1,
        xmin=min(axes[1].get_xlim()),
        xmax=max(axes[1].get_xlim()),
        color='black'
    )
    axes[2].set_ylabel('Fold Change in DALYs per 1,000 Compared to 2020')
    axes[2].set_xlabel('Scenario')
    axes[2].set_xticks(range(len(scenario_names)))
    axes[2].set_xticklabels(scenario_names, rotation=45)
    axes[2].legend().set_visible(False)
    fig.tight_layout()

    fig.savefig(output_folder / "DALYs_combined_plot.png")
    normalized_DALYs.to_csv(output_folder / f"relative_of_dalys_normalized_2020_2070.csv")

    ####### DALYs NCDs
    selected_causes = [
        'Cancer (Bladder)',
        'Cancer (Breast)',
        'Cancer (Oesophagus)',
        'Cancer (Other)',
        'Cancer (Prostate)',
        'Depression / Self-harm',
        'Diabetes',
        'Epilepsy',
        'Heart Disease',
        'Kidney Disease',
        'Lower Back Pain'
    ]

    df_selected_causes = df_dalys_all_draws_mean_1000[['Last']].loc[selected_causes]
    df_selected_causes = df_selected_causes[['Last']]
    dalys_total_per_scenario = df_dalys_all_draws_mean_1000[['Last']].sum(axis=0)
    df_selected_causes_pct = df_selected_causes.div(dalys_total_per_scenario) * 100

    df_selected_causes_pct.columns = range(len(scenario_names))
    cols = list(df_selected_causes_pct.columns)

    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    df_selected_causes_pct = df_selected_causes_pct[cols]

    # Plot: Stacked bar chart of percentage DALYs for selected causes
    fig, ax = plt.subplots(figsize=(12, 8))
    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    df_selected_causes_pct.T.plot.bar(stacked=True,
                                      ax=ax,
                                      color=[get_color_cause_of_death_or_daly_label(cause) for cause in
                                             selected_causes])

    ax.set_xlabel('Scenario')
    ax.set_ylabel('% of total DALYs')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_folder /'NCD_proportion_DALYS_combined_plot.png')
    plt.close(fig)

    ###### Female vs Male deaths in 2070

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))


    axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes,
                 fontsize=14, va='top', ha='right')

    axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes,
                 fontsize=14, va='top', ha='right')
    #df_deaths_all_draws_mean_1000_male.drop(df_deaths_all_draws_mean_1000_male.columns[-1], axis=1, inplace=True)
    df_deaths_all_draws_mean_1000_male = df_deaths_all_draws_mean_1000_male[['Last']]
    df_deaths_all_draws_mean_1000_male.columns = range(len(scenario_names))
    cols = list(df_deaths_all_draws_mean_1000_male.columns)

    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    df_deaths_all_draws_mean_1000_male = df_deaths_all_draws_mean_1000_male[cols]
    df_deaths_all_draws_mean_1000_male.columns = range(len(scenario_names))
    # Panel A: Males
    df_deaths_all_draws_mean_1000_male.T.plot.bar(stacked=True, ax=axes[0],
                                            color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                   df_deaths_all_draws_mean_1000_male.index],
                                            label=[label for label in df_all_years_DALYS_mean.index])

    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Deaths per 1,000: Males')
    axes[0].set_xticks(range(len(scenario_names)))
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].set_ylim(0, 25)
    axes[0].legend_.remove()
    axes[0].legend(title='Cause', bbox_to_anchor=(1.05, 1), ncol  = 1)

    # Panel B: Females
    df_deaths_all_draws_mean_1000_female = df_deaths_all_draws_mean_1000_female[['Last']]
    df_deaths_all_draws_mean_1000_female.columns = range(len(scenario_names))
    cols = list(df_deaths_all_draws_mean_1000_female.columns)

    second_col = cols[1]
    cols = cols[:1] + cols[2:] + [second_col]
    df_deaths_all_draws_mean_1000_female = df_deaths_all_draws_mean_1000_female[cols]
    df_deaths_all_draws_mean_1000_female.columns = range(len(scenario_names))

    df_deaths_all_draws_mean_1000_female.T.plot.bar(stacked=True, ax=axes[1],
                                            color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                   df_deaths_all_draws_mean_1000_female.index],
                                            label=[label for label in df_all_years_DALYS_mean.index])

    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Deaths per 1,000: Females')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].set_ylim(0, 25)
    axes[1].legend_.remove()

    plt.tight_layout()
    fig.savefig(make_graph_file_name(
        f'deaths_all_cause_all_draws_2070_males_vs_females'))
    plt.close(fig)
    # Save data as CSV
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    # Needed the first time as pickles were not created on Azure side:
    # from tlo.analysis.utils import create_pickles_locally
    # create_pickles_locally(
    #     scenario_output_dir=args.results_folder,
    #     compressed_file_name_prefix=args.results_folder.name.split('-')[0],
    # )

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
