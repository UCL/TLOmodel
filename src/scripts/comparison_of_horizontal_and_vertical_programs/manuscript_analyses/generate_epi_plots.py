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
from scipy.ndimage import uniform_filter1d

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

    # this measure is per-person
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
    hiv_inc_per_1000py = hiv_inc * 1000
    hiv_inc_per_1000py.index = hiv_inc_per_1000py.index.year

    # todo this needs to be divided by PY
    tb_inc = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.tb',
        key='tb_incidence',
        column="num_new_active_tb",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    def get_person_years(_df):
        """ extract person-years for each draw/run
        sums across men and women
        will skip column if particular run has failed
        """
        years = pd.to_datetime(_df["date"]).dt.year
        py = pd.Series(dtype="int64", index=years)
        for year in years:
            tot_py = (
                (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
                (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
            ).transpose()
            py[year] = tot_py.sum().values[0]

        py.index = pd.to_datetime(years, format="%Y")

        return py

    personyears = compute_summary_statistics(extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )

    tb_inc_per_1000py = tb_inc.divide(personyears) * 1000
    tb_inc_per_1000py.index = tb_inc_per_1000py.index.year
    tb_inc_per_1000py = tb_inc_per_1000py.drop(tb_inc_per_1000py.index[0])  # drop first empty row

    mal_inc_per_1000py = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.malaria',
        key='incidence',
        column="inc_1000py",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    mal_inc_per_1000py.index = mal_inc_per_1000py.index.year

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
    hiv_tx = hiv_tx * 100
    hiv_tx.index = hiv_tx.index.year

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
    tb_tx = tb_tx * 100
    tb_tx.index = tb_tx.index.year

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
    mal_tx = mal_tx * 100
    mal_tx.index = mal_tx.index.year


    # %% Extract the numbers of deaths
    def summarise_deaths_for_one_cause(results_folder, cause):

        results_deaths = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df["date"].dt.year).groupby(
                    ["year", "cause"])["person_id"].count()
            ),
            do_scaling=True,
        ).pipe(set_param_names_as_column_index_level_0)

        # removes multi-index
        results_deaths = results_deaths.reset_index()

        # select only cause specified
        if cause == 'HIV':
            tmp = results_deaths.loc[
                (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
                ]
        else:
            tmp = results_deaths.loc[
                (results_deaths.cause == cause)
            ]

        # group deaths by year
        tmp = pd.DataFrame(tmp.groupby(["year"]).sum())
        # drop first column listing cause
        tmp = tmp.drop('cause', axis=1)

        # get median and UI
        tmp2 = pd.concat({
            'central': tmp.groupby(level=0, axis=1).median(0.5),
            'lower': tmp.groupby(level=0, axis=1).quantile(0.025),
            'upper': tmp.groupby(level=0, axis=1).quantile(0.975)
        }, axis=1).swaplevel(axis=1)

        return tmp2

    aids_deaths = summarise_deaths_for_one_cause(results_folder, 'HIV') / 1000
    tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB') / 1000
    malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria') / 1000

    # %% Extract the numbers of deaths

    # Function to apply smoothing using a moving average
    def smooth_data(data, window_size=3):
        return uniform_filter1d(data, size=window_size)

    # Function to plot multiple dataframes efficiently
    def plot_multiple_dfs(dfs, row_labels, titles, legend_text=None, legend_fontsize=10, row_label_fontsize=12, y_label_fontsize=10):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        sns.set_style("white")
        name_of_plot = 'epi_outputs'

        column_y_labels = [
            'Incidence per 1000 person-years',  # Column 0
            'Treatment coverage (%)',  # Column 1
            'Numbers of deaths, thousands',  # Column 2
        ]

        if legend_text is None:
            legend_text = Summary_scenarios

        legend_lines = []

        # Loop through the DataFrames and plot each one
        for i, df in enumerate(dfs):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            ax.grid(False)  # Explicitly remove grid lines

            # Plot each scenario in the DataFrame
            for scenario in Summary_scenarios:
                color = color_map.get(scenario, "#000000")  # Default to black if not in color_map

                # Apply smoothing to the data
                central_smoothed = smooth_data(df[(scenario, "central")].values)
                lower_smoothed = smooth_data(df[(scenario, "lower")].values)
                upper_smoothed = smooth_data(df[(scenario, "upper")].values)

                line, = ax.plot(df.index, central_smoothed, label=scenario, color=color)

                # Only the first time through should add the line to the legend
                if col == 0 and row == 0:
                    legend_lines.append(line)

                ax.fill_between(df.index, lower_smoothed, upper_smoothed, color=color, alpha=0.2)

            # Set titles and labels
            if row == 0:
                ax.set_title(titles[col])

            # Add row labels only for the first column
            if col == 0:  # Only add row label to the first column of each row
                ax.annotate(row_labels[row], xy=(-0.3, 0.5), xycoords='axes fraction', fontsize=row_label_fontsize,
                            ha='center', va='center', rotation=90, fontweight='bold')

            # Add column-specific y-axis labels
            ax.set_ylabel(column_y_labels[col], fontsize=y_label_fontsize, labelpad=10)  # y-axis label for each column

            # Add custom y-axis limits for the second column if needed
            if col == 1:
                ax.set_ylim(0, 100)  # Constraint for second column

            # Adjust x-tick labels to display every other year
            if row == 2:
                ax.set_xticks(df.index[::2])  # Label every other year
                ax.set_xticklabels(df.index[::2], rotation=45)
                ax.tick_params(axis='x', which='both', bottom=True)  # Ensure the tick marks are shown

        axes[2, 2].legend(legend_lines, legend_text, loc="upper right", fontsize=legend_fontsize, frameon=False)

        # Adjust the spacing for row labels and overall layout
        plt.tight_layout(rect=[0, 1, 1, 1])  # Ensure that everything fits within the figure
        plt.subplots_adjust(hspace=0.1, wspace=0.3)  # Increase vertical and horizontal space between subplots

        # Save and display the plot
        file_name = make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', ''))
        fig.savefig(file_name)
        plt.show()

    # Plot the epi outputs
    dfs = [hiv_inc_per_1000py, hiv_tx, aids_deaths,
           tb_inc_per_1000py, tb_tx, tb_deaths,
           mal_inc_per_1000py, mal_tx, malaria_deaths]
    plot_multiple_dfs(dfs, ["HIV", "Tuberculosis", "Malaria"],
                      ["Incidence", "Treatment Coverage", "Numbers of deaths"],
        legend_text=["HSS Package", "HTM Scale-up", "HTM Scale-up + HSS Package"])














