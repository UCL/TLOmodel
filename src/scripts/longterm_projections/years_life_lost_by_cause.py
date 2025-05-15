import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from longterm_projections import LongRun
import numpy as np
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

min_year = 2067
max_year = 2068
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'
age_standardisation = 50

scenario_names = ["Status Quo", "Maximal Healthcare \nProvision", "HTM Scale-up", "Lifestyle: CMD"]
scenario_colours = ['#0081a7', '#00afb9',  '#fed9b7', '#f07167']
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
    def get_mean_pop_by_age_for_sex_and_year(draw):
        num_by_age_F = summarize(
            extract_results(results_folder,
                            module="tlo.methods.demography",
                            key='age_range_f',
                            custom_generate_series=population_by_agegroup_for_year,
                            do_scaling=True
                            ),
            collapse_columns=True,
            only_mean=True
        )
        num_by_age_M = summarize(
            extract_results(results_folder,
                            module="tlo.methods.demography",
                            key='age_range_m',
                            custom_generate_series=population_by_agegroup_for_year,
                            do_scaling=True
                            ),
            collapse_columns=True,
            only_mean=True
        )
        num_by_age = num_by_age_F + num_by_age_M
        num_by_age = num_by_age[draw]

        num_by_age_filtered = num_by_age[num_by_age.index.to_series().apply(
            lambda x: int(x.split('-')[0].replace('+', '')) >= age_standardisation
        )]

        num_by_age = num_by_age.sum()
        num_by_age.reset_index(drop=True)
        num_by_age_filtered.reset_index(inplace=True)
        num_by_age_filtered = num_by_age_filtered.sum()
        return num_by_age_filtered / num_by_age

    param_names = get_parameter_names_from_scenario_file()
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()
    def get_num_yld_by_cause_label(_df):
        """Return total number of YLL by label (total by age-group within the TARGET_PERIOD)"""
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

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_yll_mean_1000_2070 = []
    all_draws_yll_lower_1000_2070 = []
    all_draws_yll_upper_1000_2070 = []
    all_draws_yld_mean_1000_2070 = []
    all_draws_yld_lower_1000_2070 = []
    all_draws_yld_upper_1000_2070 = []

    all_draws_yll_mean_1000_2020 = []
    all_draws_yll_lower_1000_2020 = []
    all_draws_yll_upper_1000_2020 = []
    all_draws_yld_mean_1000_2020 = []
    all_draws_yld_lower_1000_2020 = []
    all_draws_yld_upper_1000_2020 = []

    for draw in range(4):
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_yll_mean = {}
        all_years_data_yll_upper= {}
        all_years_data_yll_lower = {}

        all_years_data_yld_mean = {}
        all_years_data_yld_upper = {}
        all_years_data_yld_lower = {}

        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

            # %% Quantify the health gains associated with all interventions combined.

            # Absolute Number of Deaths and ylls
            result_data_yll = summarize(extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range', 'year']).sum()),
            do_scaling=True),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            result_data_yll = result_data_yll[result_data_yll.index.str.contains('cancer', case=False)]

            all_years_data_yll_mean[target_year] = result_data_yll['mean']
            all_years_data_yll_lower[target_year] = result_data_yll['lower']
            all_years_data_yll_upper[target_year] = result_data_yll['upper']

            result_data_yld = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthburden",
                    key="yld_by_causes_of_disability",
                    custom_generate_series=(
                        lambda df: df.loc[pd.to_datetime(df.date).between(*TARGET_PERIOD)] \
                        .drop(columns=['date', 'sex', 'age_range']).sum()),
                    do_scaling=True),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            result_data_yld = result_data_yld[result_data_yld.index.str.contains('cancer', case=False)]

            all_years_data_yld_mean[target_year] = result_data_yld['mean']
            all_years_data_yld_lower[target_year] = result_data_yld['lower']
            all_years_data_yld_upper[target_year] = result_data_yld['upper']

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
            result_data_over_standard = get_mean_pop_by_age_for_sex_and_year(draw)
            all_years_data_population_mean[target_year] = result_data_population['mean']/result_data_over_standard['mean']
            all_years_data_population_lower[target_year] = result_data_population['lower']/result_data_over_standard['lower']
            all_years_data_population_upper[target_year] = result_data_population['upper']/result_data_over_standard['upper']
        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_yll_mean = pd.DataFrame(all_years_data_yll_mean)
        df_all_years_yll_lower = pd.DataFrame(all_years_data_yll_lower)
        df_all_years_yll_upper = pd.DataFrame(all_years_data_yll_upper)

        df_all_years_yld_mean = pd.DataFrame(all_years_data_yld_mean)
        df_all_years_yld_lower = pd.DataFrame(all_years_data_yld_lower)
        df_all_years_yld_upper = pd.DataFrame(all_years_data_yld_upper)


        print(all_years_data_population_mean)
        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        print(df_all_years_yll_mean)
        df_yll_per_1000_mean_2070 = df_all_years_yll_mean.iloc[:,-1].div(df_all_years_data_population_mean.iloc[-1, 0], axis=0) * 1000
        df_yld_per_1000_mean_2070 = df_all_years_yld_mean.iloc[:,-1].div(df_all_years_data_population_mean.iloc[-1, 0], axis=0) * 1000
        df_yll_per_1000_lower_2070 = df_all_years_yll_lower.iloc[:,-1].div(df_all_years_data_population_lower.iloc[-1, 0], axis=0) * 1000
        df_yld_per_1000_lower_2070 = df_all_years_yld_lower.iloc[:,-1].div(df_all_years_data_population_lower.iloc[-1, 0], axis=0) * 1000
        df_yll_per_1000_upper_2070 = df_all_years_yld_upper.iloc[:,-1].div(df_all_years_data_population_upper.iloc[-1, 0], axis=0) * 1000
        df_yld_per_1000_upper_2070 = df_all_years_yll_upper.iloc[:,-1].div(df_all_years_data_population_upper.iloc[-1, 0], axis=0) * 1000

        df_yll_per_1000_mean_2020 = df_all_years_yll_mean.iloc[:,0].div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_yld_per_1000_mean_2020 = df_all_years_yld_mean.iloc[:,0].div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_yll_per_1000_lower_2020 = df_all_years_yll_lower.iloc[:,0].div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        df_yld_per_1000_lower_2020 = df_all_years_yld_lower.iloc[:,0].div(df_all_years_data_population_lower.iloc[0, 0], axis=0) * 1000
        df_yll_per_1000_upper_2020 = df_all_years_yld_upper.iloc[:,0].div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        df_yld_per_1000_upper_2020 = df_all_years_yll_upper.iloc[:,0].div(df_all_years_data_population_upper.iloc[0, 0], axis=0) * 1000
        print("df_yld_per_1000_lower_2020", df_yld_per_1000_lower_2020)
        print("df_yld_per_1000_lower_2070", df_yld_per_1000_lower_2070)

        # Extract total population

        # Plotting
        # NORMALIZED DEATHS AND yllS - TO 2020
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        df_yll_normalized_mean = df_all_years_yll_mean.div(df_all_years_yll_mean.iloc[:, 0], axis=0)
        df_yld_normalized_mean = df_all_years_yld_mean.div(df_all_years_yld_mean.iloc[:, 0], axis=0)
        df_yll_normalized_mean.to_csv(output_folder / f"cause_of_yll_normalized_2020_{draw}.csv")
        df_yld_normalized_mean.to_csv(output_folder / f"cause_of_yld_normalized_2020_{draw}.csv")


        #data_ylls_mean.to_csv(output_folder/f"ylls_by_cause_rate_2020_{draw}.csv")

        #data_deaths.to_csv(output_folder/f"deaths_by_cause_rate_2020_{draw}.csv")

        all_draws_yld_mean_1000_2070.append(pd.Series(df_yld_per_1000_mean_2070, name=f'Draw {draw}'))
        all_draws_yld_lower_1000_2070.append(pd.Series(df_yld_per_1000_lower_2070, name=f'Draw {draw}'))
        all_draws_yld_upper_1000_2070.append(pd.Series(df_yld_per_1000_upper_2070, name=f'Draw {draw}'))
        all_draws_yll_mean_1000_2070.append(pd.Series(df_yll_per_1000_mean_2070, name=f'Draw {draw}'))
        all_draws_yll_lower_1000_2070.append(pd.Series(df_yll_per_1000_lower_2070, name=f'Draw {draw}'))
        all_draws_yll_upper_1000_2070.append(pd.Series(df_yll_per_1000_upper_2070, name=f'Draw {draw}'))

        all_draws_yld_mean_1000_2020.append(pd.Series(df_yld_per_1000_mean_2020, name=f'Draw {draw}'))
        all_draws_yld_lower_1000_2020.append(pd.Series(df_yld_per_1000_lower_2020, name=f'Draw {draw}'))
        all_draws_yld_upper_1000_2020.append(pd.Series(df_yld_per_1000_upper_2020, name=f'Draw {draw}'))
        all_draws_yll_mean_1000_2020.append(pd.Series(df_yll_per_1000_mean_2020, name=f'Draw {draw}'))
        all_draws_yll_lower_1000_2020.append(pd.Series(df_yll_per_1000_lower_2020, name=f'Draw {draw}'))
        all_draws_yll_upper_1000_2020.append(pd.Series(df_yll_per_1000_upper_2020, name=f'Draw {draw}'))


    df_yll_all_draws_mean_1000_2070 = pd.concat(all_draws_yll_mean_1000_2070, axis=1)
    df_yld_all_draws_mean_1000_2070 = pd.concat(all_draws_yld_mean_1000_2070, axis=1)

    df_yll_all_draws_mean_1000_2020 = pd.concat(all_draws_yll_mean_1000_2020, axis=1)
    df_yld_all_draws_mean_1000_2020 = pd.concat(all_draws_yld_mean_1000_2020, axis=1)

    # Save to CSV
    # df_deaths_all_draws_mean.to_csv(output_folder / "total_deaths_all_draws.csv")
    # df_ylls_all_draws_mean.to_csv(output_folder / "total_ylls_all_draws.csv")


    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    print(df_yld_all_draws_mean_1000_2070.values)
    # Panel A: 2020

    axes[0].bar(df_yld_all_draws_mean_1000_2020.T.index, df_yld_all_draws_mean_1000_2020.T.values.sum(), color=scenario_colours)


    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total YLD in 2020')
    axes[0].set_ylim(0,0.7)
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # # Panel B: 2070
    axes[1].bar(df_yld_all_draws_mean_1000_2070.T.index, df_yld_all_draws_mean_1000_2070.T.values.sum(), color=scenario_colours)
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Total YLD in 2070')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)
    axes[1].set_ylim(0,0.7)
    fig.tight_layout()
    fig.savefig(output_folder / "total_ylds_and_ylls_all_draws_age_standardized_2070.png")
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


