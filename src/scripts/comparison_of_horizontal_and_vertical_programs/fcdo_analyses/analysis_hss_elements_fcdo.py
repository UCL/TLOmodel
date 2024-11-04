"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)

test run for developing plots:
outputs/hss_elements-2024-08-21T125348Z

full run:
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/hss_elements-2024-09-04T142900Z

updated run 22nd Oct 2024
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk//hss_elements-2024-10-22T163857Z

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
from tlo.analysis.utils import (extract_results, make_age_grp_lookup, summarize)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.comparison_of_horizontal_and_vertical_programs.global_fund_analyses.scenario_hss_elements_gf import (
            HSSElements,
        )
        e = HSSElements()
        return tuple(e._scenarios.keys())

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

    def find_difference_relative_to_comparison(_ser: pd.Series,
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

    def sort_order_of_columns(_df, include_Baseline: bool = True):

        level_0_columns_results = total_num_dalys_by_label_results.columns.get_level_values(0).unique()
        if not include_Baseline:
            filtered_columns = [col for col in level_0_columns_results if col != 'Baseline']
        else:
            filtered_columns = level_0_columns_results

        # Reindex total_num_dalys_by_label_results_averted_vs_baseline to match the order of Level 0 columns
        reordered_df = _df.reindex(
            columns=filtered_columns
        )

        return reordered_df

    def do_bar_plot_with_ci(_df, set_colors=None, annotations=None,
                            xticklabels_horizontal_and_wrapped=False,
                            put_labels_in_legend=True,
                            offset=1e6):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""

        substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        yerr = np.array([
            (_df['median'] - _df['lower']).values,
            (_df['upper'] - _df['median']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        # Define colormap (used only with option `put_labels_in_legend=True`)
        # cmap = plt.get_cmap("tab20")
        # cmap = sns.color_palette('Spectral', as_cmap=True)
        # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        # colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None
        if set_colors:
            colors = [color_map.get(series, 'grey') for series in _df.index]
        else:
            cmap = sns.color_palette('Spectral', as_cmap=True)
            rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
            colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            xticks.keys(),
            _df['median'].values,
            yerr=yerr,
            # alpha=0.8,
            ecolor='black',
            color=colors,
            capsize=10,
            label=xticks.values()
        )

        if annotations:
            # for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
            #     ax.text(xpos, ypos * 1.15, '\n'.join(text.split(' ', 1)),
            #             horizontalalignment='center', rotation='horizontal', fontsize='x-small')
            for xpos, (ypos, text) in zip(xticks.keys(), zip(_df['upper'].values.flatten(), annotations)):
                # Set annotation position with fixed offset
                annotation_y = ypos + offset

                ax.text(
                    xpos,
                    annotation_y,
                    '\n'.join(text.split(' ', 1)),
                    horizontalalignment='center',
                    verticalalignment='bottom',  # Aligns text at the bottom of the annotation position
                    fontsize='x-small',
                    rotation='horizontal'
                )

        ax.set_xticks(list(xticks.keys()))

        if put_labels_in_legend:
            # Update xticks label with substitute labels
            # Insert legend with updated labels that shows correspondence between substitute label and original label
            xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]
            xtick_legend = [f'{letter}: {label}' for letter, label in zip(substitute_labels, xticks.values())]
            h, legs = ax.get_legend_handles_labels()
            ax.legend(h, xtick_legend, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
            ax.set_xticklabels(list(xtick_values))
        else:
            if not xticklabels_horizontal_and_wrapped:
                # xticklabels will be vertical and not wrapped
                ax.set_xticklabels(list(xticks.values()), rotation=90)
            else:
                wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
                ax.set_xticklabels(wrapped_labs)

        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig, ax


    def do_line_plot_with_ci(_df, set_colors=None,
                             xticklabels_horizontal_and_wrapped=False,
                             put_labels_in_legend=True):
        """
        Make a line plot with median values and shaded confidence intervals using a
        DataFrame with MultiIndex columns.
        """

        # Extract median, lower, and upper values from the MultiIndex columns
        median_df = _df.xs('median', level=0, axis=1)
        lower_df = _df.xs('lower', level=0, axis=1)
        upper_df = _df.xs('upper', level=0, axis=1)

        # Ensure that the x-axis is the row index (years)
        xticks = {i: k for i, k in enumerate(median_df.index)}

        # Define colormap
        # cmap = sns.color_palette('Spectral', as_cmap=True)
        # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        # colors = list(map(cmap, rescale(np.arange(len(median_df.columns))))) if put_labels_in_legend else None

        if set_colors:
            colors = [color_map.get(series, 'grey') for series in median_df.columns]
        else:
            cmap = sns.color_palette('Spectral', as_cmap=True)
            rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
            colors = list(map(cmap, rescale(np.arange(len(median_df.columns))))) if put_labels_in_legend else None

        fig, ax = plt.subplots(figsize=(10, 5))

        lines = []
        for i, column in enumerate(median_df.columns):
            # Plot the median line
            line, = ax.plot(
                xticks.keys(),
                median_df[column],
                color=colors[i] if colors is not None else 'black',  # Line color
                marker='o',  # Marker at each point
                label=f'{column}'
            )
            lines.append(line)

            # Fill the confidence intervals
            ax.fill_between(
                xticks.keys(),
                lower_df[column],
                upper_df[column],
                color=colors[i] if colors is not None else 'gray',  # Shaded area color
                alpha=0.3,  # Transparency of the shaded area
                label=f'{column} - CI'
            )

        if put_labels_in_legend:
            # Update legend to only include median lines
            ax.legend(handles=lines, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

        if not xticklabels_horizontal_and_wrapped:
            ax.set_xticks(list(xticks.keys()))
            ax.set_xticklabels(list(xticks.values()), rotation=0)
        else:
            wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in xticks.values()]
            ax.set_xticks(list(xticks.keys()))
            ax.set_xticklabels(wrapped_labs)

        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig, ax



    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # %% Quantify the health gains associated with all interventions combined.

    # Absolute Number of Deaths and DALYs
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

    # Filter scenarios
    # Define the multi-index column labels to drop
    exclude_labels = ['Increase Capacity of CHW', 'HSS PACKAGE: Realistic expansion, no change in HSB']

    # Drop specified columns from the DataFrame
    filtered_num_deaths = num_deaths.loc[:, ~num_deaths.columns.get_level_values(0).isin(exclude_labels)]
    filtered_num_dalys = num_deaths.loc[:, ~num_dalys.columns.get_level_values(0).isin(exclude_labels)]
    filtered_param_names = tuple(label for label in param_names if label not in exclude_labels)


    # %% Charts of total numbers of deaths / DALYS
    num_dalys_summarized = summarize(filtered_num_dalys).loc[0].unstack().reindex(filtered_param_names)
    num_deaths_summarized = summarize(filtered_num_deaths).loc[0].unstack().reindex(filtered_param_names)
    num_dalys_summarized.to_csv(results_folder / 'num_dalys_summarized.csv')
    num_deaths_summarized.to_csv(results_folder / 'num_deaths_summarized.csv')

    # color_map = {
    #     'Baseline': '#9e0142',
    #     'HRH Scale-up Following Historical Growth': '#dd4a4c',
    # 'HRH Moderate Scale-up (1%)': '#f98e52',
    # 'HRH Accelerated Scale-up (6%)': '#fed481',
    # 'CHW Scale-up Following Historical Growth': '#ffffbe',
    # 'Increase Capacity at Primary Care Levels': '#d6ee9b',
    # 'Consumables Increased to 75th Percentile':  '#86cfa5',
    # 'Consumables Available at HIV levels': '#3d95b8',
    #     'Consumables Available at EPI levels': '',
    #     'FULL PACKAGE': '#5e4fa2',
    # }
    # todo need to update
    color_map = {
        'Baseline': '#a50026',
        'HRH Moderate Scale-up (1%)': '#d73027',
        'HRH Scale-up Following Historical Growth': '#f46d43',
        'HRH Accelerated Scale-up (6%)': '#fdae61',
        'Increase Capacity at Primary Care Levels': '#fee08b',
        # 'Increase Capacity of CHW': '#ffffbf',
        'Consumables Increased to 75th Percentile': '#d9ef8b',
        'Consumables Available at HIV levels': '#a6d96a',
        'Consumables Available at EPI levels': '#66bd63',
        'Perfect Consumables Availability': '#1a9850',
        'HSS PACKAGE: Perfect': '#5e4fa2',
        'HSS PACKAGE: Realistic expansion': '#3288bd'
    }


    # DEATHS: all scenarios
    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6, set_colors=color_map)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Baseline', 'median']/1e6, color='black', linestyle='--', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    # DALYS: all scenarios
    name_of_plot = f'All Scenarios: DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6, set_colors=color_map)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc['Baseline', 'median']/1e6, color='black', alpha=0.5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)



    # %% Deaths and DALYS averted relative to Status Quo
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                filtered_num_deaths.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(filtered_param_names).drop(['Baseline'])
    num_deaths_averted.to_csv(results_folder / 'num_deaths_averted.csv')

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                filtered_num_deaths.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(filtered_param_names).drop(['Baseline'])
    pc_deaths_averted.to_csv(results_folder / 'pc_deaths_averted.csv')

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                filtered_num_dalys.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(filtered_param_names).drop(['Baseline'])
    num_dalys_averted.to_csv(results_folder / 'num_dalys_averted.csv')

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                filtered_num_dalys.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(filtered_param_names).drop(['Baseline'])
    pc_dalys_averted.to_csv(results_folder / 'pc_dalys_averted.csv')

    # DEATHS AVERTED
    name_of_plot = f'Deaths Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in pc_deaths_averted.clip(lower=0.0).iterrows()
        ],
        offset=10_000, set_colors=color_map,
    )
    fig.subplots_adjust(left=0.15, top=0.85, bottom=0.15)
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 500_000)
    ax.set_ylabel('Deaths Averted')
    # fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    # DALYS AVERTED
    name_of_plot = f'DALYs Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['median'])}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in pc_dalys_averted.clip(lower=0.0).iterrows()
        ],
        offset=0.02, set_colors=color_map,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel('DALYS Averted \n(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)




    # %% DALYS averted relative to Baseline - broken down by major cause (HIV, TB, MALARIA)

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

    def get_total_num_dalys_by_label(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        y = _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'year', 'li_wealth']) \
            .sum(axis=0)

        # define course cause mapper for HIV, TB, MALARIA and OTHER
        causes = {
            'AIDS': 'HIV/AIDS',
            'TB (non-AIDS)': 'TB',
            'Malaria': 'Malaria',
            '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
        }
        causes_relabels = y.index.map(causes).fillna('Other')

        return y.groupby(by=causes_relabels).sum()[list(causes.values())]

    total_num_dalys_by_label_results = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_wealth_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)

    summarise_total_num_dalys_by_label_results = summarize(extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_wealth_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label,
        do_scaling=True,
    ), only_median=True
    )
    summarise_total_num_dalys_by_label_results.to_csv(results_folder / 'summarise_total_num_dalys_by_label_results.csv')

    total_num_dalys_by_label_results_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison='Baseline'
        ),
        only_median=True
    )
    total_num_dalys_by_label_results_averted_vs_baseline.to_csv(results_folder / 'total_num_dalys_by_label_results_averted_vs_baseline.csv')


    # PLOT break down by cause

    # def plot_dalys_averted_by_cause(_df, put_labels_in_legend=True,
    #                                 xticklabels_horizontal_and_wrapped=False):
    #     """
    #     Plot DALYs averted by cause with annotations and save the plot.
    #
    #     """
    #     # Define annotations (A, B, C, ...)
    #     substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #     xticks = {i: label for i, label in enumerate(_df.columns)}
    #
    #     fig, ax = plt.subplots(figsize=(9, 6))  # Increase the figure size for better layout
    #     num_categories = len(_df.index)
    #     colours = sns.color_palette('Set1', num_categories)
    #
    #     # Plot stacked bar chart
    #     bars = _df.T.plot.bar(
    #         stacked=True,
    #         ax=ax,
    #         rot=0,
    #         color=colours
    #     )
    #
    #     # Set x-ticks and labels
    #     ax.set_xticks(list(xticks.keys()))
    #     ax.set_xticklabels([substitute_labels[i] for i in xticks.keys()])
    #
    #     ax.grid(axis="y")
    #     fig.tight_layout(pad=2.0)
    #     plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins
    #
    #     return fig, ax

    def plot_dalys_averted_by_cause(_df, color_map, put_labels_in_legend=True,
                                    xticklabels_horizontal_and_wrapped=False):
        """
        Plot DALYs averted by cause with annotations and save the plot.

        """
        # Define annotations (A, B, C, ...)
        substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        xticks = {i: label for i, label in enumerate(_df.columns)}

        # Reorder DataFrame based on the color map keys
        ordered_columns = [key for key in color_map.keys() if key in _df.columns]
        _df = _df[ordered_columns]

        fig, ax = plt.subplots(figsize=(9, 6))  # Increase the figure size for better layout
        num_categories = len(_df.index)
        colours = sns.color_palette('Set1', num_categories)

        # Plot stacked bar chart
        bars = _df.T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            color=colours
        )

        # Set x-ticks and labels
        ax.set_xticks(list(xticks.keys()))
        ax.set_xticklabels([substitute_labels[i] for i in xticks.keys()])

        ax.grid(axis="y")
        fig.tight_layout(pad=2.0)
        plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins

        return fig, ax

    filtered_total_num_dalys_by_label_results_averted_vs_baseline = total_num_dalys_by_label_results_averted_vs_baseline.drop(
        columns=['Increase Capacity of CHW', 'HSS PACKAGE: Realistic expansion, no change in HSB']
    )
    name_of_plot = f'DALYS Averted vs Baseline by Cause, {target_period()}'
    fig, ax = plot_dalys_averted_by_cause(filtered_total_num_dalys_by_label_results_averted_vs_baseline / 1e6, color_map)
    ax.set_title(name_of_plot)
    ax.set_ylim([0, 50])
    ax.set_ylabel('DALYs Averted vs Baseline (Millions)')
    ax.set_xlabel('')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()
    plt.close(fig)



    def get_num_dalys_by_year(_df):
        """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
        Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
        results from runs that crashed mid-way through the simulation.
        """
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return pd.Series(
            data=_df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex', 'age_range'])
            .groupby(['year']).sum().stack()
        )

    num_dalys_by_year = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_year,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    summed_by_year = num_dalys_by_year.groupby('year').sum()

    median_dalys = summed_by_year.groupby(level=0, axis=1).median(0.5)
    lower_dalys = summed_by_year.groupby(level=0, axis=1).quantile(0.025)
    upper_dalys = summed_by_year.groupby(level=0, axis=1).quantile(0.975)

    result_df = pd.concat(
        {
            'median': median_dalys,
            'lower': lower_dalys,
            'upper': upper_dalys
        },
        axis=1
    )

    # PLOT DALYS over target period with CI
    # Drop specified columns from the DataFrame
    filtered_dalys_by_year = result_df.loc[:, ~result_df.columns.get_level_values(1).isin(exclude_labels)]

    name_of_plot = f'DALYS, {target_period()}'
    fig, ax = do_line_plot_with_ci(
        filtered_dalys_by_year / 1e6,
        put_labels_in_legend=True,
        set_colors=color_map,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(6, 18)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    def get_total_num_dalys_by_wealth(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        wealth_cats = {5: '0-19%', 4: '20-39%', 3: '40-59%', 2: '60-79%', 1: '80-100%'}

        return _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'year']) \
            .assign(
            li_wealth=lambda x: x['li_wealth'].map(wealth_cats).astype(
                pd.CategoricalDtype(wealth_cats.values(), ordered=True)
            )
        ).melt(id_vars=['li_wealth']) \
            .groupby(by=['li_wealth'])['value'] \
            .sum()

    total_num_dalys_by_wealth = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_by_wealth_stacked_by_age_and_time",
            custom_generate_series=get_total_num_dalys_by_wealth,
            do_scaling=True,
        ).pipe(set_param_names_as_column_index_level_0),
        collapse_columns=True,
        only_mean=True,
    ).unstack()


    def format_to_plot(_df):
        """Format the total DALYs data for plotting."""

        # Reset index to convert multi-index series to a DataFrame
        df = _df.reset_index()

        # Pivot the DataFrame to have 'li_wealth' as rows and 'stat/draw' as columns
        df_pivot = df.pivot_table(index=['li_wealth'], columns=['draw', 'stat'], values=0, aggfunc='sum').fillna(0)
        filtered_columns = df_pivot.columns[df_pivot.columns.get_level_values(1) == 'mean']
        df_mean = df_pivot.loc[:, filtered_columns]
        df_mean.columns = df_mean.columns.droplevel(1)

        # Reorder columns
        df_formatted = sort_order_of_columns(df_mean,  include_Baseline=True)

        return df_formatted


    formatted_data = format_to_plot(total_num_dalys_by_wealth)

    name_of_plot = f'DALYS, {target_period()}'
    fig, ax = plot_dalys_averted_by_cause(formatted_data / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylim([0, 100])
    ax.set_ylabel('DALYs (Millions)')
    ax.set_xlabel('')
    legend = ax.get_legend()
    legend.set_title('Wealth Category')  # Set the title for the legend
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()
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
