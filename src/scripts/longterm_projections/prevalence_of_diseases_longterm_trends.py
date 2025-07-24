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
    'Tb': 'TB',
    'chronic_ischemic_hd': 'Heart Disease',
    'chronic_kidney_disease': 'Kidney Disease',
    'chronic_lower_back_pain': 'Lower Back Pain',
    'diabetes': 'Diabetes',
    'hypertension': 'Hypertension'
}

standard_population = None


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the prevalence of each disease
    """
    # Set period of interest needed for helper functions
    all_draws_prevalence_normalized = pd.DataFrame(columns = range(len(scenario_names))) # to save 2069 results

    all_draws_prevalence = pd.DataFrame(columns = range(len(scenario_names)))

    all_draws_prevalence_standard_years = pd.DataFrame(columns = range(len(scenario_names)))

    all_draws_population = pd.DataFrame(columns = range(len(scenario_names)))

    for draw in range(len(scenario_names)):
            # Definitions of general helper functions

        _, age_grp_lookup = make_age_grp_lookup()

        def _standardize_short_treatment_id(short_treatment_id):
                return short_treatment_id.replace('_*', '*').rstrip('*') + '*'

        def get_color_cause_of_prevalence_label(prevalence_condition_label: str) -> str:
                """Return the colour (as matplotlib string) assigned to this Prevalence Label.

                Returns np.nan if label is not recognised.
                """
                return CONDITION_TO_COLOR_MAP_PREVALENCE.get(
                    _standardize_short_treatment_id(prevalence_condition_label),
                    np.nan)

        def get_prevalence_by_cause_by_age(_df):
                """Return total prevalence by cause label (total across age-groups and sexes within the TARGET_PERIOD)."""
                _df['date'] = pd.to_datetime(_df['date'])
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]

                prevalence_sums = {}
                for col in filtered_df.columns:
                    if col == 'date':
                        continue
                    if pd.api.types.is_numeric_dtype(filtered_df[col]):
                        prevalence_sums[col] = filtered_df[col].sum()
                    elif isinstance(filtered_df[col].iloc[0], list):
                        prevalence_sums[col] = filtered_df[col].apply(lambda x: x[0]).sum()
                    elif isinstance(filtered_df[col].iloc[0], dict):
                        age_group_sums = {}
                        for d in filtered_df[col]:
                            for age_group, sexes in d.items():
                                age_group_sums.setdefault(age_group, 0)
                                age_group_sums[age_group] += sum(sexes.values())
                        prevalence_sums[col] = age_group_sums
                return pd.DataFrame.from_dict(prevalence_sums).stack()

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
        all_years_data_prevalence_standard_years = {}


        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1),
                Date(target_year, 12, 31)
            )

            # Prevalence of diseases
            result_data_prevalence_by_age = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthburden',
                    key='prevalence_of_diseases',
                    custom_generate_series=get_prevalence_by_cause_by_age,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]

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
            if target_year == 2020:
                standard_population_structure_weights = num_by_age['mean']/num_by_age['mean'].sum()

            result_data_prevalence_by_age.index.names = ['age_grp', 'cause']
            excluded_causes = [
                'chronic_ischemic_hd',
                'chronic_kidney_disease',
                'chronic_lower_back_pain',
                'diabetes',
                'hypertension',
                'live_births',
                'population'
            ]

            # Mask for rows where cause is NOT in the excluded list
            age_grps = result_data_prevalence_by_age.index.get_level_values('age_grp')
            mask = ~result_data_prevalence_by_age.index.get_level_values('cause').isin(excluded_causes)
            crude_prevalence_by_agegroup = result_data_prevalence_by_age['mean'].copy()

            # Correct the prevalence values by multiplying by total alive / alive in age group
            total_alive = num_by_age['mean'].sum()

            correction_factor = total_alive/ num_by_age['mean'] # want to multiply by the total population to correct for mistake, but also divide by the number in each age group to get per age group cases
            # Apply correction factor to your prevalence values
            crude_prevalence_by_agegroup.loc[mask] = (
                result_data_prevalence_by_age.loc[mask, 'mean'].values *
                correction_factor.loc[age_grps[mask]].values
            )

            result_data_prevalence_by_age['crude_prevalence'] = crude_prevalence_by_agegroup
            age_standardised_prevalence = ((crude_prevalence_by_agegroup.unstack(level='age_grp') * standard_population_structure_weights)).sum(axis=1)
            age_standardised_prevalence = age_standardised_prevalence.drop(index = 'Schisto')
            # Store results
            all_years_data_population[target_year] = num_by_age['mean'].sum(axis=0)
            all_years_data_prevalence_standard_years[target_year] = (age_standardised_prevalence)

        df_prevalence_standard_years = pd.DataFrame(all_years_data_prevalence_standard_years)
        df_prevalence_standard_years.to_csv(output_folder / f"Prevalence_diseases_2020_2070_{draw}.csv")

    # Drop rows only if they exist
        rows_to_drop = [
            'live_births', 'population',
            'PostnatalSupervisor', 'PregnancySupervisor', 'CardioMetabolicDisorders',
            'NewbornOutcomes', 'Labour',
            'Intrapartum stillbirth', 'Antenatal stillbirth', 'NMR', 'MMR',
            'Schistosomiasis' # because it is out of date
        ]

        df_prevalence_standard_years = df_prevalence_standard_years.drop(index=rows_to_drop, errors='ignore')

        # Rename index labels

        df_prevalence_standard_years = df_prevalence_standard_years.rename(index=rename_dict)
        all_draws_prevalence_standard_years[draw] = df_prevalence_standard_years.iloc[:,-1]
        df_prevalence_standard_years = pd.DataFrame(df_prevalence_standard_years)
        df_all_years_prevalence_normalized = df_prevalence_standard_years.div(
                df_prevalence_standard_years.iloc[:, 0], axis=0)
        df_normalized_population = {year: value / all_years_data_population[2020]
                                        for year, value in all_years_data_population.items()}
        df_normalized_population = pd.Series(df_normalized_population)
        all_draws_population.loc[0,draw] = df_normalized_population.iloc[-1]


    # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        df_prevalence_standard_years.T.plot.bar(stacked=True, ax=axes[0],
                                                    color=[get_color_cause_of_prevalence_label(_label) for _label in
                                                           df_prevalence_standard_years.index])

        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Age-standardised prevalence in population (per 1,000)', fontsize=12)
        axes[0].legend().set_visible(False)

        labels = [label.get_text() for label in axes[0].get_xticklabels()]
        new_labels = [label if i % 5 == 0 else '' for i, label in enumerate(labels)]
        axes[0].set_xticks(range(len(new_labels)))
        axes[0].set_xticklabels(new_labels, rotation=0)
        axes[0].tick_params(axis='both', which='major', labelsize=12)

        for condition in df_all_years_prevalence_normalized.index:
                axes[1].plot(df_all_years_prevalence_normalized.columns,
                             df_all_years_prevalence_normalized.loc[condition],
                             marker='o',
                             label=condition,
                             color=get_color_cause_of_prevalence_label(condition))
        # axes[1].scatter(df_normalized_population.index,
        #                     df_normalized_population,
        #                     color='black', marker='s', label='Population')
        axes[1].hlines(y=df_normalized_population.loc[2020], xmin=min(axes[1].get_xlim()), xmax=max(axes[1].get_xlim()),
                           color='black')  # just want it to be at 1
        axes[1].set_xlabel('Year',  fontsize=12)
        axes[1].set_ylabel('Fold change in prevalence',  fontsize=12)
        axes[1].legend(title='Condition', bbox_to_anchor=(1, 1), loc='upper left')
        axes[1].tick_params(axis='both', which='major', labelsize=12)

        fig.tight_layout()
        fig.savefig(make_graph_file_name(f'Trend_Prevalence_by_Condition_All_Years_Raw_and_Normalized_Panel_A_and_B_{draw}_age_standardisation'))
        plt.close(fig)
        df_all_years_prevalence_normalized.to_csv(output_folder/f"Prevalence_by_condition_normalized_2020_{draw}_age_standardisation.csv")
        all_draws_prevalence_normalized[draw] = df_all_years_prevalence_normalized.iloc[:,-1]

        fig, axes = plt.subplots(1, 1, figsize=(10, 10))

        label_positions = []
        y_offset = 0.02

        for condition in df_all_years_prevalence_normalized.index:
                color = get_color_cause_of_prevalence_label(condition)

                axes.plot(
                    df_all_years_prevalence_normalized.columns,
                    df_all_years_prevalence_normalized.loc[condition],
                    marker='o',
                    color=color
                )

                final_x = df_all_years_prevalence_normalized.columns[-1] + 0.5
                final_y = df_all_years_prevalence_normalized.loc[condition].iloc[-1]

                while any(abs(final_y - existing_y) < y_offset for existing_y in label_positions):
                    final_y += y_offset

                label_positions.append(final_y)

                axes.text(
                    x=final_x,
                    y=final_y,
                    s=condition,
                    color=color,
                    fontsize=8,
                    va='center')

        axes.hlines(
                y=df_normalized_population.loc[2020],
                xmin=min(axes.get_xlim()),
                xmax=max(axes.get_xlim()),
                color='black'
            )

        axes.set_xlabel('Year', fontsize=12)
        axes.set_ylabel('Fold change in age-standardised prevalence', fontsize=12)
        axes.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        fig.savefig(make_graph_file_name(
                f'Trend_Prevalence_by_Condition_All_Years_Normalized_{draw}_age_standardisation'))
        plt.show()
    # Plot across scenarios

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    #
    # all_draws_prevalence_standard_years.T.plot.bar(
    #         stacked=True, ax=axes[0],
    #         color=[get_color_cause_of_prevalence_label(_label) for _label in all_draws_prevalence_standard_years.index], legend=False
    #     )
    # axes[0].set_ylabel('Age-standardised prevalence',  fontsize=12)
    # axes[0].set_xlabel('Scenario', fontsize=12)
    # axes[0].set_xticklabels(scenario_names, rotation=45)
    # axes[0].tick_params(axis='both', which='major', labelsize=12)

    for i, condition in enumerate(all_draws_prevalence_normalized.index):
        axes.scatter(all_draws_prevalence_normalized.columns, all_draws_prevalence_normalized.loc[condition],
                     marker='o', s=10,
                     label=condition, color=[get_color_cause_of_prevalence_label(_label) for _label in
                                             all_draws_prevalence_normalized.index][i])
        axes.plot(
            all_draws_prevalence_normalized.columns,
            all_draws_prevalence_normalized.loc[condition],
            color=[get_color_cause_of_prevalence_label(_label) for _label in all_draws_prevalence_normalized.index][i],
            alpha=0.5
        )

    axes.hlines(y=1, xmin=min(axes.get_xlim()), xmax=max(axes.get_xlim()), color='black')
    # axes[1].scatter(all_draws_population.columns,
    #                 all_draws_population,
    #                 color='black', marker='s', label='Population')
    axes.legend(title='Condition', bbox_to_anchor=(1., 1), loc='upper left')
    axes.set_ylabel('Fold change in prevalence', fontsize=12)
    axes.set_xlabel('Scenario', fontsize=12)
    axes.set_xticks(range(len(scenario_names)))
    axes.set_xticklabels(scenario_names, rotation=45)
    axes.set_xlim(-0.5, len(scenario_names) - 0.5)
    axes.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.savefig(output_folder / f"Prevalence_by_condition_combined_years.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
