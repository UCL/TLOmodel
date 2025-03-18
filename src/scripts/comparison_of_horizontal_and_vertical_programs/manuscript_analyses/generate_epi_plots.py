"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)

job ID:
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_and_hss_runs-2025-01-16T135243Z

results_folder=Path("/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_and_hss_runs-2025-01-16T135243Z")
output_folder=Path("/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_and_hss_runs-2025-01-16T135243Z")


"""

import argparse
import textwrap
from pathlib import Path
from typing import Tuple
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    compare_number_of_deaths,
    compute_summary_statistics,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))
    scenario_info = get_scenario_info(results_folder)

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"Paper_{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.comparison_of_horizontal_and_vertical_programs.manuscript_analyses.scenario_hss_htm_paper import (
            HTMWithAndWithoutHSS,
        )
        e = HTMWithAndWithoutHSS()
        return tuple(e._scenarios.keys())

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)"""
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    param_names = get_parameter_names_from_scenario_file()

    Summary_scenarios = [
        'HSS Expansion Package',
        'HTM Program Scale-up Without HSS Expansion',
        'HTM Programs Scale-up With HSS Expansion Package',
    ]

    color_map = {
        'HSS Expansion Package': '#9e0142',
        'HTM Program Scale-up Without HSS Expansion': '#fdae61',
        'HTM Programs Scale-up With HSS Expansion Package': '#66c2a5',
    }

    # %% Quantify the deaths

    # get numbers of deaths by cause
    def summarise_deaths_by_cause(results_folder):
        """ returns mean deaths for each year of the simulation
        values are aggregated across the runs of each draw
        for the specified cause
        """

        def get_num_deaths_by_label(_df):
            """Return total number of Deaths by label within the TARGET_PERIOD
            values are summed for all ages
            df returned: rows=COD, columns=draw
            """
            return _df \
                .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
                .groupby(_df['label']) \
                .size()

        num_deaths_by_label = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=get_num_deaths_by_label,
            do_scaling=True,
        ).pipe(set_param_names_as_column_index_level_0)

        causes = {
            'AIDS': 'HIV/AIDS',
            'TB (non-AIDS)': 'TB',
            'Malaria': 'Malaria',
            '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
        }

        causes_relabels = num_deaths_by_label.index.map(causes).fillna('Other')

        grouped_deaths = num_deaths_by_label.groupby(causes_relabels).sum()
        # Reorder based on the causes keys that are in the grouped data
        ordered_causes = [cause for cause in causes.values() if cause in grouped_deaths.index]
        test = grouped_deaths.reindex(ordered_causes)

        return compute_summary_statistics(test, central_measure='median')

    num_deaths_by_cause = summarise_deaths_by_cause(results_folder)

    # %% Extract the incidence

    hiv_inc = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.hiv',
        key='summary_inc_and_prev_for_adults_and_children_and_fsw',
        column="hiv_adult_inc_15plus",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    # todo this needs to be divided by PY
    tb_inc = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.tb',
        key='tb_incidence',
        column="num_new_active_tb",
        index='date',
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )


    mal_inc = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.malaria',
        key='incidence',
        column="inc_1000py",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    # %% Extract the treatment coverage

    hiv_tx = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.hiv',
        key='hiv_program_coverage',
        column="art_coverage_adult",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    tb_tx = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.tb',
        key='tb_treatment',
        column="tbTreatmentCoverage",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    mal_tx = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.malaria',
        key='tx_coverage',
        column="treatment_coverage",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )



















