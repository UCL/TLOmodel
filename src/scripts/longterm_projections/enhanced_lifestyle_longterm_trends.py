import argparse
from pathlib import Path
from types import MappingProxyType
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
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

li_factors = ['li_urban', '']

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """
    Produce standard set of plots describing each li_factor across all scenario draws.
    All li_factors are plotted on the same graph for comparison.
    """
    target_year_sequence = range(min_year, max_year + 1, spacing_of_years)
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    for li_factor in li_factors:
        all_draws_li_factor_standard_years = pd.DataFrame(columns=range(len(scenario_names)))
        all_years_data_population = {}
        all_years_data_li_factor_standard_years = {}

        for draw in range(len(scenario_names)):
            _, age_grp_lookup = make_age_grp_lookup()

            # Helper functions
            def get_li_factor_totals(_df):
                _df = _df.copy()
                _df['date'] = pd.to_datetime(_df['date'])
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
                true_columns = [col for col in filtered_df.columns if f'{li_factor}=True' in col]
                total = filtered_df.loc[:, true_columns].sum().sum()
                return pd.Series({'total': total})

            def population_by_agegroup_for_year(_df):
                _df['date'] = pd.to_datetime(_df['date'])
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
                population_by_agegroup = (
                    filtered_df.drop(columns=['date'], errors='ignore')
                    .melt(var_name='age_grp')
                    .set_index('age_grp')['value']
                )
                return population_by_agegroup

            for target_year in target_year_sequence:
                TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

                result_data_li_factor = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.enhanced_lifestyle',
                        key=li_factor,
                        custom_generate_series=get_li_factor_totals,
                        do_scaling=True
                    ),
                    only_mean=True,
                    collapse_columns=True
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
                    standard_population_structure_weights = num_by_age['mean'] / num_by_age['mean'].sum()

                all_years_data_population[target_year] = num_by_age['mean'].sum(axis=0)
                all_years_data_li_factor_standard_years[target_year] = result_data_li_factor['mean']

            df_li_factor_standard_years = pd.DataFrame(all_years_data_li_factor_standard_years)
            all_draws_li_factor_standard_years[draw] = df_li_factor_standard_years.iloc[:, -1]

        # Normalize by first year
        df_all_years_li_factor_normalized = df_li_factor_standard_years.div(df_li_factor_standard_years.iloc[:, 0],
                                                                            axis=0)

        # Plot line for this li_factor
        mean_values = df_all_years_li_factor_normalized.mean(axis=0)
        axes.plot(
            mean_values.index,
            savgol_filter(mean_values.to_numpy(), window_length=5, polyorder=2),
            marker='o',
            label=li_factor
        )

    axes.set_xlabel('Year', fontsize=12)
    axes.set_ylabel('Fold change in factor', fontsize=12)
    axes.tick_params(axis='both', which='major', labelsize=12)
    axes.legend()
    plt.tight_layout()
    fig.savefig(make_graph_file_name('all_li_factors_normalized'))
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
