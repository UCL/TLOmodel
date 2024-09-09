import argparse
from pathlib import Path
from typing import Tuple
from types import MappingProxyType

import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

min_year = 2010
max_year = 2024  # have to censure for dummy runs
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

SHORT_TREATMENT_ID_TO_COLOR_MAP_PREVALENCE = MappingProxyType({ # based on same colours as CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP but differnet labels

    '*': 'black',

    'FirstAttendance*': 'darkgrey',
    'Inpatient*': 'silver',

    'Contraception*': 'darkseagreen',
    'AntenatalCare*': 'green',
    'DeliveryCare*': 'limegreen',
    'PostnatalCare*': 'springgreen',

    'Alri*': 'darkorange',
    'Diarrhoea*': 'tan',
    'Undernutrition*': 'gold',
    'Epi*': 'darkgoldenrod',

    'Hiv*': 'deepskyblue',
    'Malaria*': 'lightsteelblue',
    'Measles*': 'cornflowerblue',
    'Tb*': 'mediumslateblue',
    'Schisto*': 'skyblue',

    'CardioMetabolicDisorders*': 'brown',
    'chronic_ischemic_hd*': 'sienna',
    'chronic_kidney_disease*': 'chocolate',
    'chronic_lower_back_pain*': 'slategray',
    'diabetes*': 'peru',
    'hypertension*': 'darkkhaki',

    'BladderCancer*': 'orchid',
    'BreastCancer*': 'mediumvioletred',
    'OesophagealCancer*': 'deeppink',
    'ProstateCancer*': 'hotpink',
    'OtherAdultCancer*': 'palevioletred',

    'Depression*': 'indianred',
    'Epilepsy*': 'red',
    'Copd*': 'lightcoral',

    'RTI*': 'lightsalmon',
})

rename_dict = { # FOr legend labels
    'Alri': 'Lower respiratory infections',
    'BladderCancer': 'Cancer (Bladder)',
    'BreastCancer': 'Cancer (Breast)',
    'Copd': 'COPD',
    'Depression': 'Depression / Self-harm',
    'Diarrhoea': 'Diarrhoea',
    'Epilepsy': 'Epilepsy',
    'Hiv': 'AIDS',
    'Malaria': 'Malaria',
    'Measles': 'Measles',
    'OesophagealCancer': 'Cancer (Oesophagus)',
    'OtherAdultCancer': 'Cancer (Other)',
    'ProstateCancer': 'Cancer (Prostate)',
    'RTI': 'Transport Injuries',
    'Schisto': 'Schistosomiasis',
    'Tb': 'TB (non-AIDS)',
    'chronic_ischemic_hd': 'Heart Disease',
    'chronic_kidney_disease': 'Kidney Disease',
    'chronic_lower_back_pain': 'Lower Back Pain',
    'diabetes': 'Diabetes',
    'hypertension': 'Stroke'
}


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the prevalence of each disease
    """

    def _standardize_short_treatment_id(short_treatment_id):
        return short_treatment_id.replace('_*', '*').rstrip('*') + '*'

    def get_color_cause_of_prevalence_label(prevalence_condition_label: str) -> str:
        """Return the colour (as matplotlib string) assigned to this Prevalence Label.

        Returns `np.nan` if label is not recognised.
        """
        return SHORT_TREATMENT_ID_TO_COLOR_MAP_PREVALENCE.get(_standardize_short_treatment_id(prevalence_condition_label), np.nan)

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def get_prevalence_by_cause_label(_df):
        """Return total number of Prevalence by label (total by age-group within the TARGET_PERIOD)
        """
        _df['date'] = pd.to_datetime(_df['date'])
        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        prevalence_sum = filtered_df.sum(numeric_only=True)
        return prevalence_sum

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
    for target_year in target_year_sequence:
        TARGET_PERIOD = (
            Date(target_year, 1, 1),
            Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

        # Prevalence of diseases
        result_data_deaths = summarize(extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='prevalence_of_diseases',
            custom_generate_series=get_prevalence_by_cause_label,
            do_scaling=True
        ),
            only_mean=True,
            collapse_columns=True,
        )
        all_years_data_prevalence[target_year] = result_data_deaths
        # Total population
        result_data_population = summarize(extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=get_population_for_year,
            do_scaling=True
        ),
            only_mean=True,
            collapse_columns=True,
        )
        all_years_data_population[target_year] = result_data_population

    # Convert the accumulated data into a DataFrame for plotting

    df_all_years_prevalence = pd.DataFrame(all_years_data_prevalence)
    df_all_years_prevalence = df_all_years_prevalence.drop(['live_births', 'population'], axis=0)  # extra data
    df_all_years_prevalence = df_all_years_prevalence.drop(['PostnatalSupervisor', 'PregnancySupervisor',
                                                            'CardioMetabolicDisorders', 'NewbornOutcomes', 'Labour'],
                                                           axis=0)  # empty or duplicated with actual label
    df_all_years_prevalence = df_all_years_prevalence.drop(['NMR', 'MMR',
                                                            'Intrapartum stillbirth', 'Antenatal stillbirth'],
                                                           axis=0)  # not prevalence

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    # Panel A: Prevalence - general - stacked
    df_all_years_prevalence.T.plot.bar(stacked=True, ax=axes[0],
                                       color=[get_color_cause_of_prevalence_label(_label) for _label in
                                              df_all_years_prevalence.index])
    axes[0].set_title('Panel A: Prevalence by Condition')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Prevalence in population')
    axes[0].grid(True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend().set_visible(False)

    # NORMALIZED Prevalence - normalized to 2010
    df_all_years_prevalence = df_all_years_prevalence.rename(index=rename_dict)
    df_all_years_prevalence_normalized = df_all_years_prevalence.div(df_all_years_prevalence.iloc[:, 0], axis=0)
    for i, condition in enumerate(df_all_years_prevalence_normalized.index):
        axes[1].plot(df_all_years_prevalence_normalized.columns, df_all_years_prevalence_normalized.loc[condition],
                     marker='o',
                     label=condition, color=[get_color_cause_of_prevalence_label(_label) for _label in
                                             df_all_years_prevalence_normalized.index][i])
    axes[1].set_title('Panel B: Normalized Prevalence by Condition')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Fold change in deaths compared to 2010')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True)

    fig.savefig(make_graph_file_name('Trend_Prevalence_by_Condition_All_Years_Raw_and_Normalized_Panel_A_and_B'))
    plt.close(fig)


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
