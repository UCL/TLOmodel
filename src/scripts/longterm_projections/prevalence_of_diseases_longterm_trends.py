import argparse
from pathlib import Path
from types import MappingProxyType
from typing import Union

import numpy as np
import pandas as pd
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
scenario_names = ["Baseline", "Perfect World", "HTM Scale-up", "Lifestyle: CMD", "Lifestyle: Cancer"]

CONDITION_TO_COLOR_MAP_PREVALENCE = MappingProxyType(
    {
        '*': 'black',
        'Lower respiratory infections*': 'darkorange',
        'Diarrhoea*': 'tan',
        'Epilepsy*': 'darkgoldenrod',

        'AIDS*': 'deepskyblue',
        'Malaria*': 'lightsteelblue',
        'Measles*': 'cornflowerblue',
        'TB*': 'mediumslateblue',
        'Schistosomiasis*': 'skyblue',

        'CardioMetabolicDisorders*': 'brown',
        'Heart Disease*': 'sienna',
        'Kidney Disease*': 'chocolate',
        'Lower Back Pain*': 'slategray',
        'Diabetes*': 'peru',
        'Stroke*': 'darkkhaki',
        'Hypertension*': 'firebrick',

        'Cancer (Bladder)*': 'orchid',
        'Cancer (Breast)*': 'mediumvioletred',
        'Cancer (Oesophagus)*': 'deeppink',
        'Cancer (Prostate)*': 'hotpink',
        'Cancer (Other)*': 'palevioletred',

        'Depression / Self-harm*': 'indianred',
        'Epilepsy*': 'red',
        'COPD*': 'lightcoral',

        'Transport Injuries*': 'lightsalmon',

        'Antenatal Stillbirth*': "#C8E9A0",
        'Intrapartum Stillbirth*': "#B2E07B"
    }
)

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


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the prevalence of each disease
    """
    # Set period of interest needed for helper functions
    all_draws_prevalence_normalized = pd.DataFrame(columns = range(5)) # to save 2069 results
    all_draws_prevalence = pd.DataFrame(columns = range(5))
    all_draws_prevalence_50_years = pd.DataFrame(columns = range(5))
    for draw in range(5):
        TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))
        # Definitions of general helper functions
        make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}_{draw}.png"  # noqa: E731

        _, age_grp_lookup = make_age_grp_lookup()

        def _standardize_short_treatment_id(short_treatment_id):
            return short_treatment_id.replace('_*', '*').rstrip('*') + '*'

        def get_color_cause_of_prevalence_label(prevalence_condition_label: str) -> str:
            """Return the colour (as matplotlib string) assigned to this Prevalence Label.

            Returns `np.nan` if label is not recognised.
            """
            return CONDITION_TO_COLOR_MAP_PREVALENCE.get(_standardize_short_treatment_id(prevalence_condition_label),
                                                         np.nan)

        def get_prevalence_by_cause_label(_df):
            """Return total number of Prevalence by label (total by age-group within the TARGET_PERIOD)
            """
            _df['date'] = pd.to_datetime(_df['date'])
            # Filter the DataFrame based on the target period
            filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
            prevalence_sum = filtered_df.sum(numeric_only=True)
            return prevalence_sum

        def population_over_fifty_for_year(_df):
            _df['date'] = pd.to_datetime(_df['date'])

            # Filter the DataFrame based on the target period
            filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

            population_over_fifty = (
                filtered_df.drop(columns=['date'], errors='ignore')
                .melt(var_name='age_grp')
                .set_index('age_grp')['value']
            )
            return population_over_fifty


        def get_population_for_year(_df):
            """Returns the population in the year of interest"""
            _df['date'] = pd.to_datetime(_df['date'])

            # Filter the DataFrame based on the target period
            filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

            # Drop non-numeric columns (assume 'female' and 'male' are non-numeric)
            numeric_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')

            # Sum only numeric columns
            population_sum = numeric_df.sum(numeric_only=True)

            return population_sum




        target_year_sequence = range(min_year, max_year, spacing_of_years)
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

        all_years_data_prevalence = {}
        all_years_data_population = {}
        all_years_data_prevalence_50_years = {}
        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1),
                Date(target_year + spacing_of_years, 12, 31))

            # Prevalence of diseases
            result_data_prevalence = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthburden',
                key='prevalence_of_diseases',
                custom_generate_series=get_prevalence_by_cause_label,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_prevalence[target_year] = result_data_prevalence['mean']
            #Total population
            def get_mean_pop_by_age_for_sex_and_year(draw):
                    num_by_age_F = summarize(
                        extract_results(results_folder,
                                        module="tlo.methods.demography",
                                        key='age_range_f',
                                        custom_generate_series=population_over_fifty_for_year,
                                        do_scaling=True
                                        ),
                        collapse_columns=True,
                        only_mean=True
                    )
                    num_by_age_M = summarize(
                        extract_results(results_folder,
                                        module="tlo.methods.demography",
                                        key='age_range_m',
                                        custom_generate_series=population_over_fifty_for_year,
                                        do_scaling=True
                                        ),
                        collapse_columns=True,
                        only_mean=True
                    )
                    num_by_age = num_by_age_F + num_by_age_M
                    num_by_age = num_by_age[draw]

                    num_by_age_filtered = num_by_age[num_by_age.index.to_series().apply(
                        lambda x: int(x.split('-')[0].replace('+', '')) >= 50
                    )]

                    num_by_age = num_by_age.sum()
                    num_by_age.reset_index(drop=True)
                    num_by_age_filtered.reset_index(inplace=True)
                    num_by_age_filtered = num_by_age_filtered.sum()
                    return num_by_age_filtered/num_by_age

            result_data_over_50 = get_mean_pop_by_age_for_sex_and_year(draw)
            all_years_data_population[target_year] = result_data_over_50['mean']
            all_years_data_prevalence_50_years[target_year] = result_data_prevalence['mean']/result_data_over_50['mean']
        df_all_years_prevalence = pd.DataFrame(all_years_data_prevalence)
        df_prevalence_50_years = pd.DataFrame(all_years_data_prevalence_50_years)
        # Drop rows only if they exist
        rows_to_drop = [
            'live_births', 'population',
            'PostnatalSupervisor', 'PregnancySupervisor', 'CardioMetabolicDisorders',
            'NewbornOutcomes', 'Labour',
            'Intrapartum stillbirth', 'Antenatal stillbirth', 'NMR', 'MMR'
        ]
        df_all_years_prevalence = df_all_years_prevalence.drop(index=rows_to_drop, errors='ignore')
        df_prevalence_50_years = df_prevalence_50_years.drop(index=rows_to_drop, errors='ignore')
        # Rename index labels
        df_all_years_prevalence = df_all_years_prevalence.rename(index=rename_dict)
        all_draws_prevalence[draw] = df_all_years_prevalence.iloc[:,-1]

        df_prevalence_50_years = df_prevalence_50_years.rename(index=rename_dict)
        all_draws_prevalence_50_years[draw] = df_prevalence_50_years.iloc[:,-1]

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
        # Panel A: Prevalence - general - stacked

        df_prevalence_50_years.T.plot.bar(stacked=True, ax=axes[0],
                                           color=[get_color_cause_of_prevalence_label(_label) for _label in
                                                  df_prevalence_50_years.index])
        axes[0].set_title('Panel A: Prevalence by Condition')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Prevalence in population')
        axes[0].grid(True)

        axes[0].legend().set_visible(False)

        # NORMALIZED Prevalence - normalized to 2010
        df_prevalence_50_years = df_prevalence_50_years.rename(index=rename_dict)
        df_all_years_prevalence_normalized = df_prevalence_50_years.div(df_prevalence_50_years.iloc[:, 0], axis=0)
        for i, condition in enumerate(df_all_years_prevalence_normalized.index):
            axes[1].plot(df_all_years_prevalence_normalized.columns, df_all_years_prevalence_normalized.loc[condition],
                         marker='o',
                         label=condition, color=[get_color_cause_of_prevalence_label(_label) for _label in
                                                 df_all_years_prevalence_normalized.index][i])
        axes[1].set_title('Panel B: Normalized Prevalence by Condition')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Fold change in deaths compared to 2020')

        axes[1].legend(title='Condition', bbox_to_anchor=(1, 1), loc='upper left')
        axes[1].grid(True)
        axes[1].set_ylim(0, 4.5)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(f'Trend_Prevalence_by_Condition_All_Years_Raw_and_Normalized_Panel_A_and_B_{draw}_over_50'))
        plt.close(fig)
        df_all_years_prevalence_normalized.to_csv(output_folder/f"Prevalence_by_condition_normalized_2020_{draw}_over_50.csv")
        all_draws_prevalence_normalized[draw] = df_all_years_prevalence_normalized.iloc[:,-1]

    # Plot across scenarios

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    all_draws_prevalence_50_years.T.plot.bar(
        stacked=True, ax=axes[0],
        color=[get_color_cause_of_prevalence_label(_label) for _label in all_draws_prevalence.index], legend=False
    )
    axes[0].set_ylabel('Prevalence per 1,000')
    axes[0].set_xlabel('Scenario')
    axes[0].set_xticklabels(scenario_names, rotation=45)


    for i, condition in enumerate(all_draws_prevalence_normalized.index):
        axes[1].scatter(all_draws_prevalence_normalized.columns, all_draws_prevalence_normalized.loc[condition],
                     marker='o',
                     label=condition, color=[get_color_cause_of_prevalence_label(_label) for _label in
                                             all_draws_prevalence_normalized.index][i])

    axes[1].hlines(y=1, xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()), color = 'black')

    axes[1].legend(bbox_to_anchor=(1.05, 1.05), ncol=1)
    axes[1].set_ylabel('Fold change in condition prevalence compared to 2020')
    axes[1].set_xlabel('Scenario')
    axes[1].set_xticks(all_draws_cadre_normalised.columns)
    axes[1].set_xticklabels(scenario_names, rotation=45)
    fig.tight_layout()
    fig.savefig(output_folder / "Prevalence_by_condition_combined_50_years.png")
    plt.show()

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
