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
max_year = 2068
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Status Quo", "Maximal Healthcare \nProvision", "HTM Scale-up", "Lifestyle: CMD"]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167']
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
    print(param_names)
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
        numeric_df = filtered_df.drop(columns=['male'], errors='ignore')
        population_sum = numeric_df.sum(numeric_only=True)

        return population_sum

    def get_population_for_year_male(_df):
        """Returns the population in the year of interest"""
        _df['date'] = pd.to_datetime(_df['date'])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=['female'], errors='ignore')
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

    for draw in range(4):
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

        df_death_per_1000_mean_female = df_all_years_data_deaths_mean_female.div(df_all_years_data_population_mean_female.iloc[0, 0], axis=0) * 1000
        df_death_per_1000_mean_male = df_all_years_data_deaths_mean_male.div(df_all_years_data_population_mean_male.iloc[0, 0], axis=0) * 1000

        # Extract total population

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

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
        axes[1].set_title('Panel B: DALYs by cause')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].legend(title='Condition', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid()

        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B'))
        plt.close(fig)

        # NORMALIZED DEATHS AND DALYS - TO 2020
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        df_death_normalized_mean = df_all_years_deaths_mean.div(df_all_years_deaths_mean.iloc[:, 0], axis=0)
        df_DALY_normalized_mean = df_all_years_DALYS_mean.div(df_all_years_DALYS_mean.iloc[:, 0], axis=0)
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

        # BARPLOT STACKED DEATHS AND DALYS OVER TIME
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        df_all_years_deaths_mean.T.plot.bar(stacked=True, ax=axes[1],
                                       color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                              df_all_years_deaths_mean.index])

        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid()


        # Panel B: DALYs (Stacked bar plot)
        df_all_years_DALYS_mean.T.plot.bar(stacked=True, ax=axes[1],
                                      color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                             df_all_years_DALYS_mean.index])
        axes[1].axhline(0.0, color='black')
        axes[1].set_title('Panel B: DALYs')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].set_xlabel('Year')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].legend(ncol=3, fontsize=8, loc='upper right')
        axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid()

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Stacked'))

        # Stacked area graph for DALYS and Deaths
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        # Panel A: Deaths (Stacked area plot)
        years_deaths = df_all_years_deaths_mean.columns
        conditions_deaths = df_all_years_deaths_mean.index


        axes[0].stackplot(years_deaths, df_all_years_deaths_mean.values, labels=conditions_deaths,
                          colors=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                  df_all_years_deaths_mean.index])
        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].grid()

        # Panel B: DALYs (Stacked area plot)
        years_dalys = df_all_years_DALYS_mean.columns
        conditions_dalys = df_all_years_DALYS_mean.index

        axes[1].stackplot(years_dalys, df_all_years_DALYS_mean.values, labels=conditions_dalys,
                          colors=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                  df_all_years_DALYS_mean.index])
        axes[1].set_title('Panel B: DALYs by Cause')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid()

        fig.tight_layout()

        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Area'))


        ## BARPLOTS STACKED PER 1000
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        df_death_per_1000_mean = df_all_years_deaths_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000

        # Panel A: Deaths (Stacked bar plot)
        df_death_per_1000_mean.T.plot.bar(stacked=True, ax=axes[0],
                                     color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                            df_death_per_1000_mean.index])
        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths per 1000 people')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend().set_visible(False)

        # Panel B: DALYs (Stacked bar plot)
        df_daly_per_1000_mean.T.plot.bar(stacked=True, ax=axes[1],
                                    color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                           df_daly_per_1000_mean.index], label = [label for label in df_daly_per_1000_mean.index])
        axes[1].axhline(0.0, color='black')
        axes[1].set_title('Panel B: DALYs')
        axes[1].set_ylabel('Number of DALYs per 1000 people')
        axes[1].set_xlabel('Year')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].legend(ncol=3, fontsize=8, loc='upper right')
        axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Stacked_Rate'))

        ## BARPLOTS STACKED PER 1000
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

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
        print(df_all_years_DALYS_mean.index)
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
        all_draws_dalys_mean_1000.append(pd.Series(df_daly_per_1000_mean.iloc[:,-1], name=f'Draw {draw}'))
        all_draws_dalys_lower_1000.append(pd.Series(df_daly_per_1000_lower.iloc[:,-1], name=f'Draw {draw}'))
        all_draws_dalys_upper_1000.append(pd.Series(df_daly_per_1000_upper.iloc[:,-1], name=f'Draw {draw}'))
        all_draws_deaths_mean_1000.append(pd.Series(df_death_per_1000_mean.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_deaths_lower_1000.append(pd.Series(df_death_per_1000_lower.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_deaths_upper_1000.append(pd.Series(df_death_per_1000_upper.iloc[:, -1], name=f'Draw {draw}'))

        # deaths by sex
        all_draws_deaths_mean_1000_male.append(pd.Series(df_death_per_1000_mean_male.iloc[:, -1], name=f'Draw {draw}'))
        all_draws_deaths_mean_1000_female.append(pd.Series(df_death_per_1000_mean_female.iloc[:, -1], name=f'Draw {draw}'))

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
    deaths_totals_err = np.array([
        deaths_totals_mean - deaths_totals_lower,
        deaths_totals_upper - deaths_totals_mean
    ])

    dalys_totals_err = np.array([
        dalys_totals_mean - dalys_totals_lower,
        dalys_totals_upper - dalys_totals_mean
    ])
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Panel A: Total Deaths
    axes[0].bar(deaths_totals_mean.index, deaths_totals_mean.values, color=scenario_colours, yerr = deaths_totals_err, capsize=20)
    axes[0].set_title('Total Deaths (2020-2070)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total Deaths')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # Panel B: Total DALYs
    axes[1].bar(dalys_totals_mean.index, dalys_totals_mean.values, color=scenario_colours, yerr = dalys_totals_err, capsize=20)
    axes[1].set_title('Total DALYs (2020-2070)')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Total DALYs')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "total_deaths_and_dalys_all_draws.png")
    plt.close(fig)


    # per 1000 in 2070
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Panel A: Total Deaths
    #axes[0].bar(df_deaths_all_draws_mean_1000.index, df_deaths_all_draws_mean_1000.values, color=scenario_colours, yerr = deaths_totals_err, capsize=20)
    df_deaths_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[0],
                                     color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                            df_deaths_all_draws_mean_1000.index])
    axes[0].set_title('Deaths per 1,000 (2070)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Deaths per 1,000')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend().set_visible(False)
    # Panel B: Total DALYs
    #axes[1].bar(df_dalys_all_draws_mean_1000.index, df_dalys_all_draws_mean_1000.values, color=scenario_colours, yerr = dalys_totals_err, capsize=20)
    df_dalys_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[1],
                                     color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                            df_dalys_all_draws_mean_1000.index], label = [label for label in df_all_years_DALYS_mean.index])
    axes[1].set_title('DALYS per 1,000 (2070)')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('DALYS per 1,000')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].legend(title='Cause', bbox_to_anchor=(1., 1), loc='upper left')

    fig.tight_layout()
    fig.savefig(output_folder / "deaths_and_dalys_per_1000_all_cause_all_draws_2070.png")
    plt.close(fig)


    ## Normalize to 2020
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel A: DALYs per 1,000 in 2070
    df_dalys_all_draws_mean_1000.T.plot.bar(stacked=True, ax=axes[0],
                                            color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                   df_dalys_all_draws_mean_1000.index],
                                            label=[label for label in df_all_years_DALYS_mean.index])

    axes[0].set_title('Panel A: DALYs per 1,000 (2070)')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('DALYs per 1,000')
    axes[0].set_xticks(range(len(scenario_names)))
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].legend()

    # Panel B: Fold change in DALYs compared to 2020
    for i, condition in enumerate(normalized_DALYs.index):
        axes[1].scatter(normalized_DALYs.columns, normalized_DALYs.loc[condition],
                        marker='o',
                        label=condition,
                        color=[get_color_cause_of_death_or_daly_label(_label) for _label in normalized_DALYs.index][i])

    axes[1].set_ylabel('Fold change in DALYs compared to 2020')
    axes[1].set_xlabel('Scenario')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].legend(title='Cause', bbox_to_anchor=(1., 1), loc='upper left')
    axes[1].grid(False)
    axes[1].set_title('Panel B: Relative DALYs (2070 vs 2020)')

    fig.savefig(make_graph_file_name(
        f'dalys_by_cause_and_total_dalys_all_cause_all_draws_relative_2070_2020'))
    plt.close(fig)

    normalized_DALYs.to_csv(output_folder / f"relative_of_dalys_normalized_2020_2070.csv")

    # List of selected causes
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

    # Subset to those causes
    df_selected_causes = df_dalys_all_draws_mean_1000.loc[selected_causes]

    # Calculate percentage of total DALYs per scenario for those causes
    dalys_total_per_scenario = df_dalys_all_draws_mean_1000.sum(axis=0)
    df_selected_causes_pct = df_selected_causes.div(dalys_total_per_scenario) * 100

    # Plot: Stacked bar chart of percentage DALYs for selected causes
    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Save figure
    fig.tight_layout()
    fig.savefig(make_graph_file_name(f'dalys_selected_conditions_percentage_of_total_2070'))
    plt.close(fig)

    ## Female vs Male deaths in 2070

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

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
    # Panel B: Females
    df_deaths_all_draws_mean_1000_female.T.plot.bar(stacked=True, ax=axes[1],
                                            color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                   df_deaths_all_draws_mean_1000_female.index],
                                            label=[label for label in df_all_years_DALYS_mean.index])

    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Deaths per 1,000: Females')
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_ylim(0, 25)

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
