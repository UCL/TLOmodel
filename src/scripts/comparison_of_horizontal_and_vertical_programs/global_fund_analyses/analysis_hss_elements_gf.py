"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)


with reduced consumables logging
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/hss_elements-2024-11-12T172311Z


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
from tlo.analysis.utils import extract_results, make_age_grp_lookup, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"GF_{stub.replace('*', '_star_')}.png"  # noqa: E731

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
                color=colors[i] if colors is not None else 'black',
                marker='o',
                label=f'{column}'
            )
            lines.append(line)

            # Fill the confidence intervals
            ax.fill_between(
                xticks.keys(),
                lower_df[column],
                upper_df[column],
                color=colors[i] if colors is not None else 'gray',
                alpha=0.3,
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


    # %% Charts of total numbers of deaths / DALYS
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names)
    num_deaths_summarized = summarize(num_deaths).loc[0].unstack().reindex(param_names)
    num_dalys_summarized.to_csv(results_folder / f'GF_num_dalys_summarized_{target_period()}.csv')
    num_deaths_summarized.to_csv(results_folder / f'GF_num_deaths_summarized_{target_period()}.csv')

    # Make a separate plot for the scale-up of each program/programs
    plots = {
        'HRH scenarios': [
            'Baseline',
            'HRH Moderate Scale-up (1%)',
            'HRH Scale-up Following Historical Growth',
            'HRH Accelerated Scale-up (6%)',
            'Increase Capacity at Primary Care Levels',
        ],
        'Supply chain scenarios': [
            'Baseline',
            'Consumables Increased to 75th Percentile',
            'Consumables Available at HIV levels',
            'Consumables Available at EPI levels',
            'HSS PACKAGE: Realistic expansion'
        ],
        'Combined scenarios': [
            'Baseline',
            'HRH Moderate Scale-up (1%)',
            'HRH Scale-up Following Historical Growth',
            'HRH Accelerated Scale-up (6%)',
            'Increase Capacity at Primary Care Levels',
            'Consumables Increased to 75th Percentile',
            'Consumables Available at HIV levels',
            'Consumables Available at EPI levels',
            'HSS PACKAGE: Realistic expansion'
        ]
    }

    color_map = {
        'Baseline': '#a50026',
        'HRH Moderate Scale-up (1%)': '#d73027',
        'HRH Scale-up Following Historical Growth': '#f46d43',
        'HRH Accelerated Scale-up (6%)': '#fdae61',
        'Increase Capacity at Primary Care Levels': '#fee08b',
        'Consumables Increased to 75th Percentile': '#d9ef8b',
        'Consumables Available at HIV levels': '#a6d96a',
        'Consumables Available at EPI levels': '#66bd63',
        'HSS PACKAGE: Realistic expansion': '#3288bd'
    }

    # DEATHS: all scenarios
    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Baseline', 'median']/1e6, color='black', linestyle='--', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DEATHS: Split by HRH scenarios and supply chain scenarios
    for plot_name, scenario_names in plots.items():
        name_of_plot = f'Deaths, {target_period()}, {plot_name}'
        fig, ax = do_bar_plot_with_ci(num_deaths_summarized.loc[scenario_names] / 1e6, set_colors=color_map)
        ax.set_title(name_of_plot)
        ax.set_ylabel('Deaths, (Millions)')
        fig.tight_layout()
        ax.axhline(num_deaths_summarized.loc['Baseline', 'median'] / 1e6, color='black',  linestyle='--', alpha=0.5)
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    # DALYS: all scenarios
    name_of_plot = f'All Scenarios: DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc['Baseline', 'median']/1e6, color='black', alpha=0.5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS: Split by HRH scenarios and supply chain scenarios
    for plot_name, scenario_names in plots.items():
        name_of_plot = f'DALYS, {target_period()}, {plot_name}'
        fig, ax = do_bar_plot_with_ci(num_dalys_summarized.loc[scenario_names] / 1e6, set_colors=color_map)
        ax.set_title(name_of_plot)
        ax.set_ylabel('DALYS, (Millions)')
        fig.tight_layout()
        ax.axhline(num_dalys_summarized.loc['Baseline', 'median'] / 1e6, color='black', alpha=0.5)
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    # %% Deaths and DALYS averted relative to Status Quo
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_deaths_averted.to_csv(results_folder / f'GF_num_deaths_averted_{target_period()}csv')

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_deaths_averted.to_csv(results_folder / f'GF_pc_deaths_averted_{target_period()}.csv')

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_dalys_averted.to_csv(results_folder / f'GF_num_dalys_averted_{target_period()}.csv')

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_dalys_averted.to_csv(results_folder / f'GF_pc_dalys_averted_{target_period()}.csv')

    # DEATHS AVERTED
    name_of_plot = f'Deaths Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in pc_deaths_averted.clip(lower=0.0).iterrows()
        ],
        offset=10_000
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 550_000)
    ax.set_ylabel('Deaths Averted')
    # fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DEATHS AVERTED: HRH scenarios and supply chain scenarios separately
    for plot_name, scenario_names in plots.items():
        filtered_scenario_names = [name for name in scenario_names if name != 'Baseline']
        name_of_plot = f'Deaths Averted vs Baseline, {target_period()}, {plot_name}'
        data = num_deaths_averted.loc[filtered_scenario_names]
        data_pc = pc_deaths_averted.loc[filtered_scenario_names]
        fig, ax = do_bar_plot_with_ci(
        data.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in data_pc.clip(lower=0.0).iterrows()
        ],
        offset=10_000, set_colors=color_map,
        )
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 400_000)
        ax.set_ylabel('Deaths Averted')
        fig.tight_layout()
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
        offset=0.5,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 50)
    ax.set_ylabel('DALYS Averted \n(Millions)')
    # fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS AVERTED: HRH scenarios and supply chain scenarios separately
    for plot_name, scenario_names in plots.items():
        filtered_scenario_names = [name for name in scenario_names if name != 'Baseline']
        name_of_plot = f'Additional DALYS Averted vs Baseline, {target_period()}, {plot_name}'
        data = num_dalys_averted.loc[filtered_scenario_names]
        data_pc = pc_dalys_averted.loc[filtered_scenario_names]
        fig, ax = do_bar_plot_with_ci(
            (data / 1e6).clip(lower=0.0),
            annotations=[
                f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
                for _, row in data_pc.clip(lower=0.0).iterrows()
            ],
            offset=0.25, set_colors=color_map,
        )
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 40)
        ax.set_ylabel('Additional DALYS Averted \n (Millions)')
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
    summarise_total_num_dalys_by_label_results.to_csv(results_folder / f'GF_num_dalys_by_label_{target_period()}.csv')

    total_num_dalys_by_label_results_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison='Baseline'
        ),
        only_median=True
    )
    total_num_dalys_by_label_results_averted_vs_baseline.to_csv(results_folder / f'GF_num_dalys_by_label_averted_vs_baseline_{target_period()}.csv')

    # DALYS averted by cause, HRH and supply chain scenarios separately
    for plot_name, scenario_names in plots.items():
        filtered_scenario_names = [name for name in scenario_names if name != 'Baseline']
        name_of_plot = f'DALYS Averted vs Baseline by Cause, {target_period()}, {plot_name}'
        fig, ax = plt.subplots()

        # Plot each bar stack with the specified color
        num_categories = len(total_num_dalys_by_label_results_averted_vs_baseline.index)
        colours = sns.color_palette('Set1', num_categories)

        scaled_data = total_num_dalys_by_label_results_averted_vs_baseline[filtered_scenario_names] / 1e6

        scaled_data.T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            # alpha=0.75,
            color=colours
        )
        ax.set_ylim([0, 30])
        ax.set_title(name_of_plot)
        ax.set_ylabel(f'DALYs Averted vs Baseline, (Millions)')
        wrapped_labs = ["\n".join(textwrap.wrap(_lab.get_text(), 13)) for _lab in ax.get_xticklabels()]
        ax.set_xticklabels(wrapped_labs)
        ax.set_xlabel('')
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
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
    name_of_plot = f'DALYS, {target_period()}'
    fig, ax = do_line_plot_with_ci(
        result_df / 1e6,
        put_labels_in_legend=True,
        set_colors=color_map,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(6, 14)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS over time: Split by HRH scenarios and supply chain scenarios
    for plot_name, scenario_names in plots.items():
        # filtered_df = result_df.xs(key=scenario_names, level=1, axis=1) / 1e6

        name_of_plot = f'DALYS, {target_period()}, {plot_name}'
        fig, ax = do_line_plot_with_ci(
            result_df.loc[:, pd.IndexSlice[:, scenario_names]] / 1e6,
            put_labels_in_legend=True,
            set_colors=color_map,
        )
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 20)
        ax.set_ylabel('DALYS, (Millions)')
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
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
