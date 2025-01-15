"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)

job ID:
results_folder = Path('/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_with_and_without_hss-2024-09-04T143044Z')

updated runs 22nd Oct 2024
results_folder = Path('/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_with_and_without_hss-2024-10-22T163743Z')

with reduced consumables logging
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_with_and_without_hss-2024-11-12T172503Z

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
    make_graph_file_name = lambda stub: output_folder / f"FCDO_{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.comparison_of_horizontal_and_vertical_programs.global_fund_analyses.scenario_vertical_programs_with_and_without_hss_gf import (
            HTMWithAndWithoutHSS,
        )
        e = HTMWithAndWithoutHSS()
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
            ecolor='black',
            color=colors,
            capsize=10,
            label=xticks.values()
        )

        if annotations:
            for xpos, (ypos, text) in zip(xticks.keys(), zip(_df['upper'].values.flatten(), annotations)):
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
        fig.tight_layout(pad=2.0)
        plt.subplots_adjust(left=0.15, right=0.85)  # Adjust left and right margins

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

    color_map = {
        'Baseline': '#9e0142',
        'HSS PACKAGE: Realistic': '#d8434e',
        'HIV Programs Scale-up WITHOUT HSS PACKAGE': '#f36b48',
        'HIV Programs Scale-up WITH REALISTIC HSS PACKAGE': '#fca45c',
        'TB Programs Scale-up WITHOUT HSS PACKAGE': '#fddc89',
        'TB Programs Scale-up WITH REALISTIC HSS PACKAGE': '#e7f7a0',
        'Malaria Programs Scale-up WITHOUT HSS PACKAGE': '#a5dc97',
        'Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE': '#6dc0a6',
        'HTM Programs Scale-up WITHOUT HSS PACKAGE': '#438fba',
        'HTM Programs Scale-up WITH REALISTIC HSS PACKAGE': '#5e4fa2',
        'HTM Programs Scale-up WITH SUPPLY CHAINS': '#3c71aa',  # New color
        'HTM Programs Scale-up WITH HRH': '#2f6094',  # New color
    }

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # %% Quantify the health gains associated with all interventions combined.


    HTM_scenarios = [
        'Baseline',
        'HIV Programs Scale-up WITHOUT HSS PACKAGE',
        'TB Programs Scale-up WITHOUT HSS PACKAGE',
        'Malaria Programs Scale-up WITHOUT HSS PACKAGE',
        'HTM Programs Scale-up WITHOUT HSS PACKAGE',
    ]

    HTM_and_HSS_scenarios = [
        'HSS PACKAGE: Realistic',
        'HIV Programs Scale-up WITHOUT HSS PACKAGE',
        'HIV Programs Scale-up WITH REALISTIC HSS PACKAGE',
        'TB Programs Scale-up WITHOUT HSS PACKAGE',
        'TB Programs Scale-up WITH REALISTIC HSS PACKAGE',
        'Malaria Programs Scale-up WITHOUT HSS PACKAGE',
        'Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE',
        'HTM Programs Scale-up WITHOUT HSS PACKAGE',
        'HTM Programs Scale-up WITH REALISTIC HSS PACKAGE',
        'HTM Programs Scale-up WITH SUPPLY CHAINS',
        'HTM Programs Scale-up WITH HRH',
    ]

     # Make a separate plot for the scale-up of each program/programs
    program_plots = {
        'HIV programs': [
            'HIV Programs Scale-up WITHOUT HSS PACKAGE',
            'HIV Programs Scale-up WITH REALISTIC HSS PACKAGE',
        ],
        'TB programs': [
            'TB Programs Scale-up WITHOUT HSS PACKAGE',
            'TB Programs Scale-up WITH REALISTIC HSS PACKAGE',
        ],
        'Malaria programs': [
            'Malaria Programs Scale-up WITHOUT HSS PACKAGE',
            'Malaria Programs Scale-up WITH REALISTIC HSS PACKAGE',
        ],
        'Summary': [
            'HSS PACKAGE: Realistic',
            'HTM Programs Scale-up WITHOUT HSS PACKAGE',
             'HTM Programs Scale-up WITH REALISTIC HSS PACKAGE'
        ]

    }

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
    num_dalys_summarized.to_csv(results_folder / f'fcdo_num_dalys_summarized_{target_period()}.csv')
    num_deaths_summarized.to_csv(results_folder / f'fcdo_num_deaths_summarized_{target_period()}.csv')

    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Baseline', 'median'] / 1e6, color='black', linestyle='--', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'All Scenarios: DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc['Baseline', 'median'] / 1e6, color='black', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    # %% Deaths and DALYS averted relative to Status Quo
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_deaths_averted.to_csv(results_folder / f'fcdo_num_deaths_averted_{target_period()}.csv')


    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_deaths_averted.to_csv(results_folder / f'fcdo_pc_deaths_averted_{target_period()}.csv')

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_dalys_averted.to_csv(results_folder / f'fcdo_num_dalys_averted_{target_period()}.csv')

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_dalys_averted.to_csv(results_folder / f'fcdo_pc_dalys_averted_{target_period()}.csv')

    # DEATHS

    # PLOT ONLY HTM SCENARIOS
    filtered_num_deaths_averted_htm = num_deaths_averted.loc[
        num_deaths_averted.index.isin(HTM_scenarios)]
    filtered_pc_deaths_averted_htm = pc_deaths_averted.loc[
        pc_deaths_averted.index.isin(HTM_scenarios)]

    name_of_plot = f'Deaths Averted vs Baseline, {target_period()}, HTM programs'
    fig, ax = do_bar_plot_with_ci(
        filtered_num_deaths_averted_htm.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in filtered_pc_deaths_averted_htm.clip(lower=0.0).iterrows()
        ], set_colors=color_map, offset=500
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 160_000)
    ax.set_ylabel('Deaths Averted vs Baseline')
    fig.tight_layout()
    plt.subplots_adjust(right=0.55)  # Increase the right margin
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # PLOT HTM and HSS SCENARIOS
    filtered_num_deaths_averted_htm_hss = num_deaths_averted.loc[
        num_deaths_averted.index.isin(HTM_and_HSS_scenarios)]
    filtered_pc_deaths_averted_htm_hss = pc_deaths_averted.loc[
        pc_deaths_averted.index.isin(HTM_and_HSS_scenarios)]

    name_of_plot = f'Deaths Averted vs Baseline, {target_period()}, HTM and HSS programs'
    fig, ax = do_bar_plot_with_ci(
        filtered_num_deaths_averted_htm_hss.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in filtered_pc_deaths_averted_htm_hss.clip(lower=0.0).iterrows()
        ], set_colors=color_map, offset=5000
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 450_000)
    ax.set_ylabel('Deaths Averted vs Baseline')
    fig.tight_layout()
    plt.subplots_adjust(right=0.55)  # Increase the right margin
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # PLOT HTM and HSS SCENARIOS
    filtered_num_dalys_averted_htm_hss = num_dalys_averted.loc[
        num_dalys_averted.index.isin(HTM_and_HSS_scenarios)]
    filtered_pc_dalys_averted_htm_hss = pc_dalys_averted.loc[
        pc_dalys_averted.index.isin(HTM_and_HSS_scenarios)]

    name_of_plot = f'DALYs Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (filtered_num_dalys_averted_htm_hss / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['median'])}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in filtered_pc_dalys_averted_htm_hss.clip(lower=0.0).iterrows()
        ], set_colors=color_map, offset=1,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 40)
    ax.set_ylabel('Additional DALYS Averted vs Baseline \n(Millions)')
    fig.tight_layout()
    plt.subplots_adjust(right=0.55)  # Increase the right margin
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # PLOT ONLY HTM SCENARIOS
    filtered_num_dalys_averted_htm = num_dalys_averted.loc[
        num_dalys_averted.index.isin(HTM_scenarios)]
    filtered_pc_dalys_averted_htm = pc_dalys_averted.loc[
        pc_dalys_averted.index.isin(HTM_scenarios)]

    name_of_plot = f'DALYs Averted vs Baseline, {target_period()}, HTM programs'
    fig, ax = do_bar_plot_with_ci(
        (filtered_num_dalys_averted_htm / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['median'])}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in filtered_pc_dalys_averted_htm.clip(lower=0.0).iterrows()
        ], set_colors=color_map, offset=1,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 12)
    ax.set_ylabel('Additional DALYS Averted vs Baseline \n(Millions)')
    fig.tight_layout()
    plt.subplots_adjust(right=0.55)  # Increase the right margin
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    # PLOT ONLY Baseline, HTM WITH AND WITHOUT HSS SCENARIOS
    # plot dalys averted for each program and summary
    for plot_name, scenario_names in program_plots.items():
        # Define the plot title and filename for each scenario
        name_of_plot = f'DALYs Averted vs Baseline, {target_period()}, {plot_name}'

        filtered_num_dalys_averted = num_dalys_averted.loc[
            num_dalys_averted.index.isin(scenario_names)]

        filtered_pc_dalys_averted = pc_dalys_averted.loc[
            pc_dalys_averted.index.isin(scenario_names)]

        # Create figure and axis
        fig, ax = plt.subplots()

        # Generate the bar plot with confidence intervals
        fig, ax = do_bar_plot_with_ci(
            (filtered_num_dalys_averted / 1e6).clip(lower=0.0),
            annotations=[
                f"{round(row['median'])}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
                for _, row in filtered_pc_dalys_averted.clip(lower=0.0).iterrows()
            ],
            set_colors=color_map,
            offset=1,
        )

        # Set plot title, labels, and limits
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 35)
        ax.set_ylabel('Additional DALYs Averted vs Baseline \n(Millions)')

        # Adjust layout and save each plot
        fig.tight_layout()
        plt.subplots_adjust(right=0.55)  # Increase right margin for annotations/legend
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)


    # %% DALYS averted relative to Baseline - broken down by major cause (HIV, TB, MALARIA)

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
    ).pipe(set_param_names_as_column_index_level_0), only_median=True
    )
    summarise_total_num_dalys_by_label_results.to_csv(results_folder / 'fcdo_summarise_num_dalys_by_label.csv')

    pc_dalys_averted_by_label = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series_dataframe(
                total_num_dalys_by_label_results,
                comparison='Baseline',
                scaled=True)
        )
    )
    pc_dalys_averted_by_label.to_csv(results_folder / 'fcdo_pc_dalys_averted_by_label.csv')

    total_num_dalys_by_label_results_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison='Baseline'
        ),
        only_median=True
    )
    total_num_dalys_by_label_results_averted_vs_baseline.to_csv(results_folder / 'fcdo_num_dalys_by_label_averted_vs_baseline.csv')

    def sort_order_of_columns(_df):

        level_0_columns_results = total_num_dalys_by_label_results.columns.get_level_values(0).unique()
        filtered_level_0_columns_results = [col for col in level_0_columns_results if col != 'Baseline']

        # Reindex total_num_dalys_by_label_results_averted_vs_baseline to match the order of Level 0 columns
        reordered_df = _df.reindex(
            columns=filtered_level_0_columns_results
        )

        return reordered_df

    for plot_name, scenario_names in program_plots.items():
        name_of_plot = f'{plot_name}'
        fig, ax = plt.subplots()

        # Plot each bar stack with the specified color
        num_categories = len(total_num_dalys_by_label_results_averted_vs_baseline.index)
        colours = sns.color_palette('Set1', num_categories)
        data = total_num_dalys_by_label_results_averted_vs_baseline[scenario_names] / 1e6
        data.clip(lower=0.0).T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            color=colours,
        )
        # ax.set_ylim([0, 1.75e7])
        ax.set_title(name_of_plot)
        ax.set_ylabel(f'DALYs Averted vs Baseline, {target_period()}\n(Millions)')
        ax.set_xlabel('')
        wrapped_labs = ["\n".join(textwrap.wrap(_lab.get_text(), 20)) for _lab in ax.get_xticklabels()]
        ax.set_xticklabels(wrapped_labs)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    def plot_combined_programs_scale_up(_df):
        """
        Generate combined plot, DALYs averted broken down by cause, exclude 'HIV_TB programs'
        """
        combined_plot_name = 'Combined Programs Scale-up'
        fig, ax = plt.subplots(figsize=(12, 6))

        colours = sns.color_palette('Set1', 4)  # We have 4 categories to stack
        x_labels = [
            'HSS \nPACKAGE \nONLY',
            'WITHOUT \nHSS',
            'WITH \nHSS',
            'WITHOUT \nHSS',
            'WITH \nHSS',
            'WITHOUT \nHSS',
            'WITH \nHSS',
            'WITHOUT \nHSS',
            'WITH \nHSS',
            'WITH \nSUPPLY \nCHAINS',
            'WITH HRH',
        ]
        shared_labels = [
            '',  # No shared label for the first bar
            'HIV Scale-up',  # Shared label for the second and third bars
            '',  # Shared label for the second and third bars
            'TB Scale-up',  # Shared label for the fourth and fifth bars
            '',  # Shared label for the fourth and fifth bars
            'Malaria Scale-up',  # Shared label for the sixth and seventh bars
            '',  # Shared label for the sixth and seventh bars
            'HTM Scale-up',  # Shared label for the eighth and ninth bars
            '',  # Shared label for the eighth and ninth bars
        ]

        # Transpose the DataFrame to get each program as a bar (columns become x-axis categories)
        _df.T.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=colours,
            rot=0
        )

        # Set the title and labels
        ax.set_title(combined_plot_name)
        ax.set_ylabel(f'DALYs Averted vs Baseline, {target_period()}\n(Millions)')
        ax.set_ylim([0, 3e7])
        ax.set_xlabel("")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, ha="center")

        # Add shared second-line labels
        for i, label in enumerate(shared_labels):
            if label:  # Only add text if there's a label
                ax.text(i, _df.sum().max() * 1.05, label, ha='left', va='bottom', fontsize=10, rotation=0,
                        color='black')

        ax.legend(title="Cause", labels=_df.index, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add vertical grey lines
        line_positions = [0, 2, 4, 6]
        for pos in line_positions:
            ax.axvline(x=pos + 0.5, color='grey', linestyle='--', linewidth=1)

        # Adjust layout and save
        # fig.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.2)  # Adjust margins for better spacing
        fig.savefig(make_graph_file_name(combined_plot_name.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    data_for_plot = sort_order_of_columns(total_num_dalys_by_label_results_averted_vs_baseline)

    filtered_df_htm_hss = data_for_plot.loc[:, data_for_plot.columns.intersection(HTM_and_HSS_scenarios)]

    plot_combined_programs_scale_up(filtered_df_htm_hss)
    #
    # def plot_pc_DALYS_combined_programs_scale_up(_df):
    #     """
    #     Generate combined plot, DALYs averted broken down by cause, exclude 'HIV_TB programs'
    #     """
    #     combined_plot_name = 'Percentage Change in DALYS vs Baseline'
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #
    #     colours = sns.color_palette('Set1', 4)  # We have 4 categories to stack
    #     x_labels = [
    #         'HSS \nPACKAGE \nONLY',
    #         'WITHOUT \nHSS',
    #         'WITH \nHSS',
    #         'WITHOUT \nHSS',
    #         'WITH \nHSS',
    #         'WITHOUT \nHSS',
    #         'WITH \nHSS',
    #         'WITHOUT \nHSS',
    #         'WITH \nHSS',
    #         'WITH \nSUPPLY \nCHAINS',
    #         'WITH HRH',
    #     ]
    #     shared_labels = [
    #         '',  # No shared label for the first bar
    #         'HIV Scale-up',  # Shared label for the second and third bars
    #         '',  # Shared label for the second and third bars
    #         'TB Scale-up',  # Shared label for the fourth and fifth bars
    #         '',  # Shared label for the fourth and fifth bars
    #         'Malaria Scale-up',  # Shared label for the sixth and seventh bars
    #         '',  # Shared label for the sixth and seventh bars
    #         'HTM Scale-up',  # Shared label for the eighth and ninth bars
    #         '',  # Shared label for the eighth and ninth bars
    #     ]
    #
    #     # Transpose the DataFrame to get each program as a bar (columns become x-axis categories)
    #     _df.T.plot(
    #         kind='bar',
    #         stacked=True,
    #         ax=ax,
    #         color=colours,
    #         rot=0
    #     )
    #
    #     # Set the title and labels
    #     ax.set_title(combined_plot_name)
    #     ax.set_ylabel(f'Percentage Change in DALYs vs Baseline, \n{target_period()}(%)')
    #     ax.set_ylim([0, 250])
    #     ax.set_xlabel("")
    #     ax.set_xticks(range(len(x_labels)))
    #     ax.set_xticklabels(x_labels, ha="center")
    #
    #     # Add shared second-line labels
    #     for i, label in enumerate(shared_labels):
    #         if label:  # Only add text if there's a label
    #             ax.text(i, _df.sum().max() * 1.05, label, ha='left', va='bottom', fontsize=10, rotation=0,
    #                     color='black')
    #
    #     ax.legend(title="Cause", labels=_df.index, bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    #     # Add vertical grey lines
    #     line_positions = [0, 2, 4, 6]
    #     for pos in line_positions:
    #         ax.axvline(x=pos + 0.5, color='grey', linestyle='--', linewidth=1)
    #
    #     # Adjust layout and save
    #     fig.tight_layout()
    #     fig.savefig(make_graph_file_name(combined_plot_name.replace(' ', '_').replace(',', '')))
    #     fig.show()
    #     plt.close(fig)
    #
    # data_for_plot = pc_dalys_averted_by_label.loc[:, pc_dalys_averted_by_label.columns.get_level_values(1) == 'median']
    # data_for_plot.columns = data_for_plot.columns.droplevel(1)
    # data_for_plot = sort_order_of_columns(data_for_plot)
    # filtered_df_htm_hss = data_for_plot.loc[:, data_for_plot.columns.intersection(HTM_and_HSS_scenarios)]
    #
    # plot_pc_DALYS_combined_programs_scale_up(filtered_df_htm_hss)


    # plot only HTM programs WITHOUT HSS

    filtered_df_htm = filtered_df_htm_hss.loc[:, filtered_df_htm_hss.columns.intersection(HTM_scenarios)]

    plot_name = 'Percentage Change in DALYS vs Baseline, HTM Programs'
    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors for each cause
    colours = sns.color_palette('Set1', len(filtered_df_htm.index))  # One color per cause
    x_labels = [
        'HIV Scale-up',
        'TB Scale-up',
        'Malaria Scale-up',
        'HTM Scale-up',
    ]

    # Transpose and plot each cause as separate bars (clustered)
    filtered_df_htm.T.plot(
        kind='bar',
        stacked=False,  # No stacking to allow clustering
        ax=ax,
        color=colours,
        rot=0,
        width=0.8  # Adjust width to ensure bars don't overlap
    )

    # Set title and labels
    ax.set_title(plot_name)
    ax.set_ylabel(f'Percentage Change in DALYs vs Baseline, \n{target_period()}')
    ax.set_ylim([0, 80])
    ax.set_xlabel("")

    # Adjust x-ticks for clustered layout
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, ha="center")

    # Add legend with cause labels
    ax.legend(title="Cause", labels=filtered_df_htm.index, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(make_graph_file_name(plot_name.replace(' ', '_').replace(',', '')))
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

    median_dalys = summed_by_year.groupby(level=0, axis=1).quantile(0.5)
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
        put_labels_in_legend=True
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.set_ylim(0, 15)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS over time
    for plot_name, scenario_names in program_plots.items():

        name_of_plot = f'DALYS, {target_period()}, {plot_name}'
        fig, ax = do_line_plot_with_ci(
            result_df.loc[:, pd.IndexSlice[:, scenario_names]] / 1e6,
            put_labels_in_legend=True)
        ax.set_title(name_of_plot)
        ax.set_ylabel('DALYS, (Millions)')
        ax.set_ylim(0, 18)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)



    def get_num_dalys_by_label_and_year(_df):

        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        y = _df \
            .loc[_df['year'].between(*[d.year for d in extended_TARGET_PERIOD])] \
            .drop(columns=['date', 'sex', 'age_range']).set_index('year')

        # define course cause mapper for HIV, TB, MALARIA and OTHER
        causes = {
            'AIDS': 'HIV/AIDS',
            'TB (non-AIDS)': 'TB',
            'Malaria': 'Malaria',
            '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
        }

        y.columns = pd.Series(y.columns).map(causes).fillna('Other')
        df_grouped = y.groupby(level=0).sum().assign(Other=lambda x: x.filter(like='Other').sum(axis=1))

        # Drop duplicate 'Other' columns
        df_grouped = df_grouped.loc[:, ~df_grouped.columns.duplicated(keep='last')]

        # Reshape the DataFrame to a Series with MultiIndex (year, cause)
        result_df = df_grouped.stack()
        result = result_df.reset_index(name='DALYS').set_index(['year', 'level_1'])['DALYS']

        return result

    extended_TARGET_PERIOD = (Date(2015, 1, 1), Date(2035, 12, 31))

    num_dalys_by_label_and_year = summarize(extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=get_num_dalys_by_label_and_year,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0), only_median=False)

    # swap multi-index columns labels around for plot function
    num_dalys_by_label_and_year = num_dalys_by_label_and_year.swaplevel(axis=1)
    num_dalys_by_label_and_year.to_csv(results_folder / 'fcdo_num_dalys_by_label_and_year.csv')


    # plot DALYS incurred by each cause over time
    years = num_dalys_by_label_and_year.index.get_level_values(0).unique()  # Extract unique years for x-tick labels
    x_ticks = range(len(years))  # Create a range for the x-ticks based on the number of unique years

    for cause in ['HIV/AIDS', 'TB', 'Malaria', 'Other']:
        if cause == 'HIV/AIDS':
            columns = ['Baseline', 'HIV Programs Scale-up WITHOUT HSS PACKAGE']
        elif cause == 'TB':
            columns = ['Baseline', 'TB Programs Scale-up WITHOUT HSS PACKAGE']
        elif cause == 'Malaria':
            columns = ['Baseline', 'Malaria Programs Scale-up WITHOUT HSS PACKAGE']

        data = num_dalys_by_label_and_year.loc[:, pd.IndexSlice[:, columns]]

        name_of_plot = f'DALYS, {cause}, 2015-2035'
        fig, ax = do_line_plot_with_ci(
            data.loc[pd.IndexSlice[:, cause], :] / 1e6,
            put_labels_in_legend=True)
        ax.set_title(name_of_plot)
        ax.set_ylabel('DALYS, (Millions)')
        ax.set_xticks(x_ticks)
        ax.set_ylim(0, 1.6)
        ax.set_xticklabels(years, rotation=45, ha='right')
        fig.tight_layout()
        # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
