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
max_year = 2070
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'
scenario_names = ["Status Quo", "Maximal Healthcare \nProvision", "HTM Scale-up", "Negative Lifestyle Change", "Positive Lifestyle Change"]
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167', '#9A348E']

CONDITION_TO_COLOR_MAP_yll = MappingProxyType(
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
    'BladderCancer': 'Cancer (Bladder)',
    'BreastCancer': 'Cancer (Breast)',
    'COPD': 'COPD',
    'Depression': 'Depression / Self-harm',
    'Diarrhoea': 'Diarrhoea',
    'Epilepsy': 'Epilepsy',
    'HIV': 'AIDS',
    'Malaria': 'Malaria',
    'Measles': 'Measles',
    'OesophagealCancer': 'Cancer (Oesophagus)',
    'OtherAdultCancer': 'Cancer (Other)',
    'ProstateCancer': 'Cancer (Prostate)',
    'RTI': 'Transport Injuries',
    'Schisto': 'Schistosomiasis',
    'Tb': 'TB',
    'chronic_ischemic_hd': 'Heart Disease',
    'chronic_kidney_disease': 'Kidney Disease',
    'chronic_lower_back_pain': 'Lower Back Pain',
    'diabetes': 'Diabetes',
    'hypertension': 'Hypertension'
}

standard_population = None


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the yll of each disease
    """
    # Set period of interest needed for helper functions
    all_draws_yll_normalized = pd.DataFrame(columns = range(len(scenario_names))) # to save 2069 results

    all_draws_yll = pd.DataFrame(columns = range(len(scenario_names)))

    all_draws_yll_standard_years = pd.DataFrame(columns = range(len(scenario_names)))

    all_draws_population = pd.DataFrame(columns = range(len(scenario_names)))

    for draw in range(len(scenario_names)):
            # Definitions of general helper functions

        _, age_grp_lookup = make_age_grp_lookup()

        def _standardize_short_treatment_id(short_treatment_id):
                return short_treatment_id.replace('_*', '*').rstrip('*') + '*'


        def population_by_agegroup_for_year(_df):
                _df['date'] = pd.to_datetime(_df['date'])

                # Filter the DataFrame based on the target period
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

                population_by_agegroup = (
                    filtered_df.drop(columns=['date'], errors='ignore')
                    .melt(var_name='age_grp')
                    .set_index('age_grp')['value']
                )
                return population_by_agegroup


        spacing_of_years = 1
        target_year_sequence = range(min_year, max_year + 1, spacing_of_years)

        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"

        all_years_data_population = {}
        all_years_data_yll_standard_years = {}


        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1),
                Date(target_year, 12, 31)
            )

            # yll of diseases
            result_data_yld = summarize(
                extract_results(
                    results_folder, "tlo.methods.healthburden", "yld_by_causes_of_disability",
                    custom_generate_series=lambda df: df.loc[pd.to_datetime(df.date).between(*TARGET_PERIOD)]
                    .drop(columns=['date', 'sex', 'age_range']).sum(),
                    do_scaling=True),
                only_mean=True, collapse_columns=True
            )[draw].loc[lambda df: df.index.str.contains('cancer', case=False)]

            num_by_age_F = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key='age_range_f',
                    custom_generate_series=population_by_agegroup_for_year,
                    do_scaling=True
                ),
                collapse_columns=True,
                only_mean=True
            )[draw]

            num_by_age_M = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key='age_range_m',
                    custom_generate_series=population_by_agegroup_for_year,
                    do_scaling=True
                ),
                collapse_columns=True,
                only_mean=True
            )[draw]

            num_by_age = num_by_age_F + num_by_age_M
            num_by_age[num_by_age == 0] = np.nan

            # Mask for rows where cause is NOT in the excluded list
            # Correct the yll values by multiplying by total alive / alive in age group

            # Store results
            all_years_data_population[target_year] = num_by_age['mean'].sum(axis=0)
            all_years_data_yll_standard_years[target_year] = (result_data_yld['mean'])/num_by_age['mean'].sum(axis=0) * 1000

        df_yll_standard_years = pd.DataFrame(all_years_data_yll_standard_years)
        df_yll_standard_years.to_csv(output_folder / f"YLL_diseases_2020_2070_{draw}.csv")

    # Drop rows only if they exist
        rows_to_drop = [
            'live_births', 'population',
            'PostnatalSupervisor', 'PregnancySupervisor', 'CardioMetabolicDisorders',
            'NewbornOutcomes', 'Labour',
            'Intrapartum stillbirth', 'Antenatal stillbirth', 'NMR', 'MMR'
        ]

        df_yll_standard_years = df_yll_standard_years.drop(index=rows_to_drop, errors='ignore')

        # Rename index labels

        df_yll_standard_years = df_yll_standard_years.rename(index=rename_dict)
        all_draws_yll_standard_years[draw] = df_yll_standard_years.iloc[:,-1]
        df_yll_standard_years = pd.DataFrame(df_yll_standard_years)
        df_all_years_yll_normalized = df_yll_standard_years.div(
                df_yll_standard_years.iloc[:, 0], axis=0)
        df_normalized_population = {year: value / all_years_data_population[2020]
                                        for year, value in all_years_data_population.items()}
        df_normalized_population = pd.Series(df_normalized_population)

        all_draws_population.loc[0,draw] = df_normalized_population.iloc[ -1]



    # Plotting

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        print(df_yll_standard_years.index)
        df_yll_standard_years.T.plot.bar(stacked=True, ax=axes[0],
                                                    color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                           df_yll_standard_years.index])

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Age-standardised YLL in population (per 1,000)', fontsize=12)
        axes[0].legend().set_visible(False)

        labels = [label.get_text() for label in axes[0].get_xticklabels()]
        new_labels = [label if i % 5 == 0 else '' for i, label in enumerate(labels)]
        new_labels.append('2070')
        axes[0].set_xticks(range(len(new_labels)))
        axes[0].set_xticklabels(new_labels, rotation=0)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        for condition in df_all_years_yll_normalized.index:
                axes[1].plot(df_all_years_yll_normalized.columns,
                             df_all_years_yll_normalized.loc[condition],
                             marker='o',
                             label=condition,
                             color=get_color_cause_of_death_or_daly_label(condition))
        axes[1].scatter(df_normalized_population.index,
                            df_normalized_population,
                            color='black', marker='s', label='Population')
        axes[1].plot(
                df_normalized_population.index,
                df_normalized_population,
                alpha=0.5
            )
        axes[1].hlines(y=df_normalized_population.loc[2020], xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()),
                           color='black')  # just want it to be at 1
        axes[1].set_xlabel('Year',  fontsize=12)
        axes[1].set_ylabel('Fold Change in YLL',  fontsize=12)
        axes[1].legend(title='Condition', bbox_to_anchor=(1, 1), loc='upper left')
        axes[1].tick_params(axis='both', which='major', labelsize=12)

        fig.tight_layout()
        fig.savefig(make_graph_file_name(f'Trend_yll_by_Condition_All_Years_Raw_and_Normalized_Panel_A_and_B_{draw}_age_standardisation'))
        plt.close(fig)
        df_all_years_yll_normalized.to_csv(output_folder/f"yll_by_condition_normalized_2020_{draw}_age_standardisation.csv")
        all_draws_yll_normalized[draw] = df_all_years_yll_normalized.iloc[:,-1]

    # Plot across scenarios

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    all_draws_yll_standard_years.T.plot.bar(
            stacked=True, ax=axes[0],
            color=[get_color_cause_of_death_or_daly_label(_label) for _label in all_draws_yll_standard_years.index], legend=False
        )
    axes[0].set_ylabel('Age-standardised yll per 1,000')
    axes[0].set_xlabel('Scenario')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    for i, condition in enumerate(all_draws_yll_normalized.index):
            print(all_draws_yll_normalized)
            axes[1].scatter(all_draws_yll_normalized.columns, all_draws_yll_normalized.loc[condition],
                         marker='o',s = 10,
                         label=condition, color=[get_color_cause_of_death_or_daly_label(_label) for _label in
                                                 all_draws_yll_normalized.index][i])
            axes[1].plot(
                all_draws_yll_normalized.columns,
                all_draws_yll_normalized.loc[condition],
                color=[get_color_cause_of_death_or_daly_label(_label) for _label in all_draws_yll_normalized.index][i],
                alpha=0.5
            )

    axes[1].hlines(y=1, xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()), color = 'black')
    axes[1].scatter(all_draws_population.columns,
                    all_draws_population,
                    color='black', marker='s', label='Population')
    axes[1].legend(title='Condition', bbox_to_anchor=(1., 1), loc='upper left')
    axes[1].set_ylabel('Fold change in YLL', fontsize=12)
    axes[1].set_xlabel('Scenario', fontsize=12)
    axes[1].set_xticks(range(len(scenario_names)))
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].set_xlim(-0.5, len(scenario_names) - 0.5)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout()
    fig.savefig(output_folder / f"yll_by_condition_combined_years.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )

