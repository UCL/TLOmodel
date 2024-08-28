

import argparse
import textwrap
from pathlib import Path
from typing import Tuple

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
min_year = 2010
max_year = 2040
PREFIX_ON_FILENAME = '1'

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)
        """
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def get_num_dalys(_df):
        """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
        Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
        results from runs that crashed mid-way through the simulation.
        """
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return pd.Series(
            data=_df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    def get_num_dalys_by_cause_label(_df):
        """Return total number of DALYS (Stacked) by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
            .drop(columns=['date', 'sex', 'age_range', 'year']) \
            .sum()

    target_year_sequence = range(min_year, max_year, 5)
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731


    all_years_data_deaths = {}
    all_years_data_dalys = {}

    for target_year in target_year_sequence:
        TARGET_PERIOD = (
        Date(target_year, 1, 1), Date(target_year + 4, 12, 31))  # Corrected the year range to cover 5 years.

    # %% Quantify the health gains associated with all interventions combined.

        # Absolute Number of Deaths and DALYs
        result_data_deaths = summarize(extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=get_num_deaths,
            do_scaling=True
        ),
            only_mean=True,
            collapse_columns=True,
        )
        all_years_data_deaths[target_year] = result_data_deaths

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
        )
        all_years_data_dalys[target_year] = result_data_dalys
        # Convert the accumulated data into a DataFrame for plotting

    df_all_years_DALYS = pd.DataFrame(all_years_data_dalys)
    df_all_years_deaths = pd.DataFrame(all_years_data_deaths)

    num_colors = len(df_all_years_DALYS.index)
    colors = plt.cm.get_cmap('tab20', num_colors)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

    # Panel A: Deaths
    for i, condition in enumerate(df_all_years_deaths.index):
        axes[0].plot(df_all_years_deaths.columns, df_all_years_deaths.loc[condition], marker='o',
                     label=condition, color=colors(i / num_colors))
    axes[0].set_title('Panel A: Deaths by Cause')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of deaths')
    axes[0].grid(True)

    # Panel B: DALYs
    for i, condition in enumerate(df_all_years_DALYS.index):
        axes[1].plot(df_all_years_DALYS.columns, df_all_years_DALYS.loc[condition], marker='o', label=condition,
                     color=colors(i / num_colors))
    axes[1].set_title('Panel B: DALYs by cause')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Number of DALYs')
    axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True)

    # Save the figure with both panels
    fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B'))
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


