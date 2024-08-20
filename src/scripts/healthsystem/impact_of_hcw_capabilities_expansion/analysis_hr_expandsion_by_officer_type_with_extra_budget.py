"""
This file analyses and plots the services, DALYs, Deaths within different scenarios of expanding current hr by officer
type given some extra budget. Return on investment and marginal productivity of each officer type will be examined.

The scenarios are defined in scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget.py.
"""

import argparse
import textwrap
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.healthsystem.impact_of_hcw_capabilities_expansion.scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget import (
    HRHExpansionByCadreWithExtraBudget,
)
from tlo import Date
from tlo.analysis.utils import extract_results, make_age_grp_lookup, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None,
          the_target_period: Tuple[Date, Date] = None):
    """
    Extract results of number of services by appt type, number of DALYs, number of Deaths in the target period.
    (To see whether to extract these results by short treatment id and/or disease.)
    Calculate the extra budget allocated, extra staff by cadre, return on investment and marginal productivity by cadre.
    """
    TARGET_PERIOD = the_target_period

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        e = HRHExpansionByCadreWithExtraBudget()
        return tuple(e._scenarios.keys())

    def get_num_appts(_df):
        """Return the number of appointments per appt type (total within the TARGET_PERIOD)"""
        return (_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code']
                .apply(pd.Series).sum())

    def get_num_services(_df):
        """Return the number of appointments in total of all appt types (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code']
            .apply(pd.Series).sum().sum()
        )

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)"""
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

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    def find_difference_relative_to_comparison_series(
        _ser: pd.Series,
        comparison: str,
        scaled: bool = False,
        drop_comparison: bool = True,
    ):
        """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1), relative to where draw = `comparison`.
        The comparison is `X - COMPARISON`."""
        return _ser \
            .unstack(level=0) \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
            .drop(columns=([comparison] if drop_comparison else [])) \
            .stack()

    def find_difference_relative_to_comparison_series_dataframe(_df: pd.DataFrame, **kwargs):
        """Apply `find_difference_relative_to_comparison_series` to each row in a dataframe"""
        return pd.concat({
            _idx: find_difference_relative_to_comparison_series(row, **kwargs)
            for _idx, row in _df.iterrows()
        }, axis=1).T

    # Get parameter/scenario names
    param_names = get_parameter_names_from_scenario_file()

    # Absolute Number of Deaths and DALYs and Services
    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_appts = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_num_appts,
        do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)

    num_services = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_num_services,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names)
    num_deaths_summarized = summarize(num_deaths).loc[0].unstack().reindex(param_names)
    num_appts_summarized = summarize(num_appts).T.unstack().reindex(param_names)
    num_services_summarize = summarize(num_services).loc[0].unstack().reindex(param_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/bshe@ic.ac.uk/scenario_run_for_hcw_expansion_analysis-2024-08-16T160132Z
    args = parser.parse_args()

    # Produce results for short-term analysis: 5 years

    # 2015-2019
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2015, 1, 1), Date(2019, 12, 31))
    )

    # 2020-2024
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2024, 12, 31))
    )

    # Produce results for long-term analysis: 10 years
    # 2020-2029
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2029, 12, 31))
    )
