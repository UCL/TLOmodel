"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)

job ID:
results for FCDO and GF presentations Sept 2024:
htm_with_and_without_hss-2024-09-04T143044Z

results for updates 30Sept2024 (IRS in high-risk distr and reduced gen pop RDT):
htm_with_and_without_hss-2024-09-17T083150Z


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
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

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

    # color_map = {
    #     'Baseline': '#9e0142',
    #      'FULL HSS PACKAGE':  '#d8434e',
    #      'HIV Programs Scale-up WITHOUT HSS PACKAGE': '#f67a49',
    #      'HIV Programs Scale-up WITH HSS PACKAGE': '#fdbf6f',
    #      'TB Programs Scale-up WITHOUT HSS PACKAGE': '#feeda1',
    #      'TB Programs Scale-up WITH HSS PACKAGE': '#f1f9a9',
    #      'Malaria Programs Scale-up WITHOUT HSS PACKAGE': '#bfe5a0',
    #      'Malaria Programs Scale-up WITH HSS PACKAGE': '#74c7a5',
    #      'HIV/Tb/Malaria Programs Scale-up WITHOUT HSS PACKAGE': '#378ebb',
    #      'HIV/Tb/Malaria Programs Scale-up WITH HSS PACKAGE': '#5e4fa2',
    # }
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
        'HTM Programs Scale-up WITH SUPPLY CHAINS': '#3c71aa',
        'HTM Programs Scale-up WITH HRH': '#2f6094',
    }

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
    num_dalys_summarized.to_csv(results_folder / 'num_dalys_summarized.csv')
    num_deaths_summarized.to_csv(results_folder / 'num_deaths_summarized.csv')


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

    # remove the HIV/TB joint scenarios
    # Filter out rows where level 0 of the multi-index is 'Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE' or 'Hiv/Tb Programs Scale-up WITH HSS PACKAGE'
    filtered_num_deaths_summarized = num_deaths_summarized.drop(
        index=['Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE', 'Hiv/Tb Programs Scale-up WITH HSS PACKAGE']
    )
    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(filtered_num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Baseline', 'median'] / 1e6, color='black', linestyle='--', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    filtered_num_dalys_summarized = num_dalys_summarized.drop(
        index=['Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE', 'Hiv/Tb Programs Scale-up WITH HSS PACKAGE']
    )
    name_of_plot = f'All Scenarios: DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(filtered_num_dalys_summarized / 1e6)
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
    num_deaths_averted.to_csv(results_folder / 'num_deaths_averted.csv')

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_deaths_averted.to_csv(results_folder / 'pc_deaths_averted.csv')

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_dalys_averted.to_csv(results_folder / 'num_dalys_averted.csv')

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_dalys_averted.to_csv(results_folder / 'pc_dalys_averted.csv')

    # DEATHS
    name_of_plot = f'Additional Deaths Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=[
            f"{round(row['median'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in pc_deaths_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('Additional Deaths Averted vs Baseline')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS
    name_of_plot = f'DALYs Averted vs Baseline, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['median'])}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in pc_dalys_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('Additional DALYS Averted vs Baseline \n(Millions)')
    fig.tight_layout()
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
    summarise_total_num_dalys_by_label_results.to_csv(results_folder / 'summarise_total_num_dalys_by_label_results.csv')

    pc_dalys_averted_by_label = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series_dataframe(
                total_num_dalys_by_label_results,
                comparison='Baseline',
                scaled=True)
        )
    )
    pc_dalys_averted_by_label.to_csv(results_folder / 'pc_dalys_averted_by_label.csv')

    total_num_dalys_by_label_results_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison='Baseline'
        ),
        only_median=True
    )
    total_num_dalys_by_label_results_averted_vs_baseline.to_csv(results_folder / 'total_num_dalys_by_label_results_averted_vs_baseline.csv')

    def sort_order_of_columns(_df):

        level_0_columns_results = total_num_dalys_by_label_results.columns.get_level_values(0).unique()
        filtered_level_0_columns_results = [col for col in level_0_columns_results if col != 'Baseline']

        # Reindex total_num_dalys_by_label_results_averted_vs_baseline to match the order of Level 0 columns
        reordered_df = _df.reindex(
            columns=filtered_level_0_columns_results
        )

        return reordered_df

    # Check that when we sum across the causes, we get the same total as calculated when we didn't split by cause.
    # this assertion fails since moving from mean to median
    assert (
        (total_num_dalys_by_label_results_averted_vs_baseline.sum(axis=0).sort_index()
         - num_dalys_averted['median'].sort_index()
         ) < 1e-6
    ).all()

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

    for plot_name, scenario_names in program_plots.items():
        name_of_plot = f'{plot_name}'
        fig, ax = plt.subplots()

        # Plot each bar stack with the specified color
        num_categories = len(total_num_dalys_by_label_results_averted_vs_baseline.index)
        colours = sns.color_palette('Set1', num_categories)

        total_num_dalys_by_label_results_averted_vs_baseline[scenario_names].T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            color=colours,
        )
        ax.set_ylim([0, 3e7])
        ax.set_title(name_of_plot)
        ax.set_ylabel(f'DALYs Averted vs Baseline, {target_period()}\n(Millions)')
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


    # def plot_combined_programs_scale_up(_df):
    #     """
    #     Generate combined plot, DALYs averted broken down by cause, exclude 'HIV_TB programs'
    #     """
    #     combined_plot_name = 'Combined Programs Scale-up'
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #
    #     colours = sns.color_palette('Set1', 4)  # We have 4 categories to stack
    #     x_labels = [
    #         'FULL HSS',
    #         'WITHOUT \nRSSH',
    #         'WITH \nRSSH',
    #         'WITHOUT \nRSSH',
    #         'WITH \nRSSH',
    #         'WITHOUT \nRSSH',
    #         'WITH \nRSSH',
    #         'WITHOUT \nRSSH',
    #         'WITH \nRSSH',
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
    #     filtered_df.T.plot(
    #         kind='bar',
    #         stacked=True,
    #         ax=ax,
    #         color=colours,
    #         rot=0
    #     )
    #
    #     # Set the title and labels
    #     ax.set_title(combined_plot_name)
    #     ax.set_ylabel(f'DALYs Averted vs Baseline, {target_period()}\n(Millions)')
    #     ax.set_ylim([0, 2e7])
    #     ax.set_xlabel("")
    #     ax.set_xticks(range(len(x_labels)))
    #     ax.set_xticklabels(x_labels, ha="center")
    #
    #     # Add shared second-line labels
    #     for i, label in enumerate(shared_labels):
    #         if label:  # Only add text if there's a label
    #             ax.text(i, filtered_df.sum().max() * 1.05, label, ha='left', va='bottom', fontsize=10, rotation=0,
    #                     color='black')
    #
    #     ax.legend(title="Cause", labels=filtered_df.index, bbox_to_anchor=(1.05, 1), loc='upper left')
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
    # data_for_plot = sort_order_of_columns(total_num_dalys_by_label_results_averted_vs_baseline)
    # filtered_df = data_for_plot.drop(
    #     columns=[
    #         'Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE',
    #         'Hiv/Tb Programs Scale-up WITH HSS PACKAGE'
    #     ]
    # )
    # plot_combined_programs_scale_up(filtered_df)



    def plot_pc_DALYS_combined_programs_scale_up(_df):
        """
        Generate combined plot, DALYs averted broken down by cause, exclude 'HIV_TB programs'
        """
        combined_plot_name = 'Percentage Change in DALYS vs Baseline'
        fig, ax = plt.subplots(figsize=(12, 6))

        colours = sns.color_palette('Set1', 4)  # We have 4 categories to stack
        x_labels = [
            'FULL HSS',
            'WITHOUT \nRSSH',
            'WITH \nRSSH',
            'WITHOUT \nRSSH',
            'WITH \nRSSH',
            'WITHOUT \nRSSH',
            'WITH \nRSSH',
            'WITHOUT \nRSSH',
            'WITH \nRSSH',
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
        filtered_df.T.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=colours,
            rot=0
        )

        # Set the title and labels
        ax.set_title(combined_plot_name)
        ax.set_ylabel(f'Percentage Change in DALYs vs Baseline, \n{target_period()}(%)')
        ax.set_ylim([0, 300])
        ax.set_xlabel("")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, ha="center")

        # Add shared second-line labels
        for i, label in enumerate(shared_labels):
            if label:  # Only add text if there's a label
                ax.text(i, filtered_df.sum().max() * 1.05, label, ha='left', va='bottom', fontsize=10, rotation=0,
                        color='black')

        ax.legend(title="Cause", labels=filtered_df.index, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add vertical grey lines
        line_positions = [0, 2, 4, 6]
        for pos in line_positions:
            ax.axvline(x=pos + 0.5, color='grey', linestyle='--', linewidth=1)

        # Adjust layout and save
        fig.tight_layout()
        fig.savefig(make_graph_file_name(combined_plot_name.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)


    # Define  custom x_labels and shared_labels
    x_labels = [
        'FULL HSS',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
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

    def plot_percentage_dalys_averted(percentage_df, x_labels, shared_labels):
        """
        Plots the percentage of dalys averted by cause for each draw.

        Parameters:
        percentage_df (pd.DataFrame): DataFrame containing percentage values with MultiIndex (draw, label).
        x_labels (list): Custom labels for the x-axis.
        shared_labels (list): Shared labels for groups of x-axis categories.
        """
        combined_plot_name = 'Percentage DALYS Averted vs Baseline'
        # Reset index to convert MultiIndex DataFrame to a suitable format for plotting
        percentage_data = percentage_df.reset_index()

        # Melt the DataFrame for easier plotting
        percentage_data = percentage_data.melt(id_vars=percentage_data.columns[0], var_name='draw',
                                               value_name='Percentage')

        # Reorder the draws according to the color_map keys
        draw_order = list(color_map.keys())[1:]  # Exclude the first label
        percentage_data['draw'] = pd.Categorical(percentage_data['draw'], categories=draw_order, ordered=True)

        # Sort the DataFrame based on the draw category
        percentage_data.sort_values('draw', inplace=True)

        # Get colors for causes using Set1 palette
        cause_colors = sns.color_palette('Set1', 4)  # We have 4 categories
        cause_color_map = {label: cause_colors[i] for i, label in enumerate(percentage_data['index'].unique())}

        # Create a bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=percentage_data, x='draw', y='Percentage', hue='index', palette=cause_color_map)

        # Add vertical dashed grey lines between draws
        vertical_lines = [1, 3, 5, 7, 9]  # Positions for vertical lines
        for line in vertical_lines:
            plt.axvline(x=line - 0.5, color='grey', linestyle='--', linewidth=1)

        # Set y-axis limit
        plt.ylim(0, 110)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=0)

        # Set custom x-axis labels
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels)

        for i, label in enumerate(shared_labels):
            if label:  # Only add text if there's a label
                plt.text(i, 105, label, ha='left', va='bottom', fontsize=10, rotation=0,
                        color='black')


        # Add labels and title
        plt.xlabel('')
        plt.ylabel('Percentage of DALYs Averted')
        plt.title(combined_plot_name)
        plt.legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

        # Show the plot
        plt.tight_layout()
        plt.savefig(make_graph_file_name(combined_plot_name.replace(' ', '_').replace(',', '')))
        plt.show()

    data_for_plot = pc_dalys_averted_by_label.loc[:, pc_dalys_averted_by_label.columns.get_level_values(1) == 'median']
    data_for_plot.columns = data_for_plot.columns.droplevel(1)
    data_for_plot = sort_order_of_columns(data_for_plot)
    filtered_df = data_for_plot.drop(
        columns=[
            'Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE',
            'Hiv/Tb Programs Scale-up WITH HSS PACKAGE'
        ]
    )
    plot_percentage_dalys_averted(filtered_df, x_labels, shared_labels)


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
        put_labels_in_legend=True
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
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
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)


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

        return summarize(test)

    num_deaths_by_cause = summarise_deaths_by_cause(results_folder)

    def calculate_deaths_averted(_df):
        """Calculate the number of deaths averted compared to the Baseline for each draw."""

        # Extract the median values
        median_values = _df.xs('median', level=1, axis=1)

        # Get the median values for the 'Baseline' draw
        baseline_values = median_values['Baseline']

        # Calculate diff in number of deaths
        deaths_diff = median_values.subtract(baseline_values, axis=0)

        # Multiply by -1 to get deaths averted
        return deaths_diff * -1

    # Use the function with your data
    deaths_averted_by_cause = calculate_deaths_averted(num_deaths_by_cause)
    deaths_averted_by_cause.to_csv(results_folder / 'deaths_averted_by_cause.csv')

    def calculate_percentage_deaths_averted(_df):
        """Calculate the percentage of deaths averted compared to the Baseline for each draw."""

        # Extract the median values
        median_values = _df.xs('median', level=1, axis=1)
        baseline_values = median_values['Baseline']

        # Calculate diff in number of deaths
        deaths_diff = median_values.subtract(baseline_values, axis=0)

        # Multiply by -1 to get deaths averted
        percentage_deaths_averted = -1 * (deaths_diff.iloc[:, 1:].divide(baseline_values, axis=0) * 100)

        return percentage_deaths_averted

    percentage_deaths_averted = calculate_percentage_deaths_averted(num_deaths_by_cause)
    percentage_deaths_averted.to_csv(results_folder / 'percentage_deaths_averted.csv')

    def plot_percentage_deaths_averted(percentage_df, x_labels, shared_labels):
        """
        Plots the percentage of deaths averted by cause for each draw.

        Parameters:
        percentage_df (pd.DataFrame): DataFrame containing percentage values with MultiIndex (draw, label).
        x_labels (list): Custom labels for the x-axis.
        shared_labels (list): Shared labels for groups of x-axis categories.
        """
        # Reset index to convert MultiIndex DataFrame to a suitable format for plotting
        percentage_data = percentage_df.reset_index()

        # Melt the DataFrame for easier plotting
        percentage_data = percentage_data.melt(id_vars='label', var_name='draw', value_name='Percentage')
        # Reorder the draws according to the color_map keys
        draw_order = list(color_map.keys())[1:]  # Exclude the first label
        percentage_data['draw'] = pd.Categorical(percentage_data['draw'], categories=draw_order, ordered=True)

        # Sort the DataFrame based on the draw category
        percentage_data.sort_values('draw', inplace=True)

        # Get colors for causes using Set1 palette
        cause_colors = sns.color_palette('Set1', 4)  # We have 4 categories
        cause_color_map = {label: cause_colors[i] for i, label in enumerate(percentage_data['label'].unique())}

        # Create a bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=percentage_data, x='draw', y='Percentage', hue='label', palette=cause_color_map)

        # Add vertical dashed grey lines between draws
        vertical_lines = [1, 3, 5, 7, 9]  # Positions for vertical lines
        for line in vertical_lines:
            plt.axvline(x=line - 0.5, color='grey', linestyle='--', linewidth=1)

        # Set y-axis limit
        plt.ylim(0, 100)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=0)

        # Set custom x-axis labels
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels)

        for i, label in enumerate(shared_labels):
            if label:  # Only add text if there's a label
                plt.text(i, 95, label, ha='left', va='bottom', fontsize=10, rotation=0,
                        color='black')


        # Add labels and title
        plt.xlabel('')
        plt.ylabel('Percentage of Deaths Averted')
        plt.title('Percentage of Deaths Averted by Cause')
        plt.legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot

        # Show the plot
        plt.tight_layout()
        plt.show()

    # Define your custom x_labels and shared_labels
    x_labels = [
        'FULL HSS',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
        'WITHOUT \nRSSH',
        'WITH \nRSSH',
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

    filtered_percentage_deaths_averted = percentage_deaths_averted.drop(
        columns=['Hiv/Tb Programs Scale-up WITHOUT HSS PACKAGE', 'Hiv/Tb Programs Scale-up WITH HSS PACKAGE']
    )
    plot_percentage_deaths_averted(filtered_percentage_deaths_averted, x_labels, shared_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )


# tmp checks
log_BASELINE = load_pickled_dataframes(results_folder, draw=0, run=0)
log_HTM_WITHOUT_HSS = load_pickled_dataframes(results_folder, draw=10, run=0)

summarise_hiv_prev = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_prev_adult_1549",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0), only_median=True)
summarise_hiv_prev.to_csv(results_folder / 'summarise_hiv_prev.csv')

summarise_hiv_inc = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="summary_inc_and_prev_for_adults_and_children_and_fsw",
    column="hiv_adult_inc_15plus",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0), only_median=True)
summarise_hiv_inc.to_csv(results_folder / 'summarise_hiv_inc.csv')

summarise_hiv_tx = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="hiv_program_coverage",
    column="art_coverage_adult_VL_suppression",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0), only_median=True)
summarise_hiv_tx.to_csv(results_folder / 'summarise_hiv_tx.csv')


summarise_hiv_tx_num = summarize(extract_results(
    results_folder,
    module="tlo.methods.hiv",
    key="hiv_program_coverage",
    column="n_on_art_total",
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0), only_median=True)
summarise_hiv_tx_num.to_csv(results_folder / 'summarise_hiv_tx_num.csv')


########################################
# extract numbers of appts
# years_of_simulation = 20
#
# def summarise_appt_outputs(df_list, treatment_id):
#     """ summarise the treatment counts across all draws/runs for one results folder
#         requires a list of dataframes with all treatments listed with associated counts
#     """
#     number_runs = len(df_list)
#     number_HSI_by_run = pd.DataFrame(index=np.arange(years_of_simulation), columns=np.arange(number_runs))
#     column_names = [
#         treatment_id + "_mean",
#         treatment_id + "_lower",
#         treatment_id + "_upper"]
#     out = pd.DataFrame(columns=column_names)
#
#     for i in range(number_runs):
#         if treatment_id in df_list[i].columns:
#             number_HSI_by_run.iloc[:, i] = pd.Series(df_list[i].loc[:, treatment_id])
#
#     out.iloc[:, 0] = number_HSI_by_run.quantile(q=0.5, axis=1)
#     out.iloc[:, 1] = number_HSI_by_run.quantile(q=0.025, axis=1)
#     out.iloc[:, 2] = number_HSI_by_run.quantile(q=0.975, axis=1)
#
#     return out
#
# def extract_appt_details(results_folder, module, key, column, draw):
#     """
#     extract list of dataframes with all treatments listed with associated counts
#     """
#
#     info = get_scenario_info(results_folder)
#
#     df_list = list()
#
#     for run in range(info['runs_per_draw']):
#         df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
#
#         new = df[['date', column]].copy()
#         df_list.append(pd.DataFrame(new[column].to_list()))
#
#     # for column in each df, get median
#     # list of treatment IDs
#     list_tx_id = list(df_list[0].columns)
#     results = pd.DataFrame(index=np.arange(years_of_simulation))
#
#     # produce a list of numbers of every treatment_id
#     for treatment_id in list_tx_id:
#         tmp = summarise_appt_outputs(df_list, treatment_id)
#
#         # append output to dataframe
#         results = results.join(tmp)
#
#     return results



def sum_appt_by_id(results_folder, module, key, column, draw):
    """
    sum occurrences of each treatment_id over the simulation period for every run within a draw
    """

    info = get_scenario_info(results_folder)
    # create emtpy dataframe
    results = pd.DataFrame()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        tmp = pd.DataFrame(new[column].to_list())

        # sum each column to get total appts of each type over the simulation
        tmp2 = pd.DataFrame(tmp.sum())
        # add results to dataframe for output
        results = pd.concat([results, tmp2], axis=1)

    return results


module = "tlo.methods.healthsystem.summary"
key = 'Never_ran_HSI_Event'
column = 'TREATMENT_ID'

baseline_never_ran_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=0)
baseline_never_ran_hsi['mean'] = baseline_never_ran_hsi.mean(axis=1)

hiv_scaleup_WITHOUT_HSS_never_ran_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=2)
hiv_scaleup_WITHOUT_HSS_never_ran_hsi['mean'] = hiv_scaleup_WITHOUT_HSS_never_ran_hsi.mean(axis=1)

htm_scaleup_WITHOUT_HSS_never_ran_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=10)
htm_scaleup_WITHOUT_HSS_never_ran_hsi['mean'] = htm_scaleup_WITHOUT_HSS_never_ran_hsi.mean(axis=1)

combined_df = pd.DataFrame({
    'Baseline': baseline_never_ran_hsi['mean'],
    'HIV Scale-up WITHOUT HSS': hiv_scaleup_WITHOUT_HSS_never_ran_hsi['mean'],
    'HTM Scale-up WITHOUT HSS': htm_scaleup_WITHOUT_HSS_never_ran_hsi['mean']
})

# Display the combined DataFrame
combined_df.head()
combined_df.to_csv(results_folder / 'hsi_never_ran.csv')




module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'TREATMENT_ID'

baseline_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=0)
baseline_hsi['mean'] = baseline_hsi.mean(axis=1)

hiv_scaleup_WITHOUT_HSS_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=2)
hiv_scaleup_WITHOUT_HSS_hsi['mean'] = hiv_scaleup_WITHOUT_HSS_hsi.mean(axis=1)

mal_scaleup_WITHOUT_HSS_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=8)
mal_scaleup_WITHOUT_HSS_hsi['mean'] = mal_scaleup_WITHOUT_HSS_hsi.mean(axis=1)


htm_scaleup_WITHOUT_HSS_hsi = sum_appt_by_id(results_folder,
                      module=module, key=key, column=column, draw=10)
htm_scaleup_WITHOUT_HSS_hsi['mean'] = htm_scaleup_WITHOUT_HSS_hsi.mean(axis=1)

combined_df = pd.DataFrame({
    'Baseline': baseline_hsi['mean'],
    'HIV Scale-up WITHOUT HSS': hiv_scaleup_WITHOUT_HSS_hsi['mean'],
    'Malaria Scale-up WITHOUT HSS': mal_scaleup_WITHOUT_HSS_hsi['mean'],
    'HTM Scale-up WITHOUT HSS': htm_scaleup_WITHOUT_HSS_hsi['mean']
})

# Display the combined DataFrame
combined_df.head()
combined_df.to_csv(results_folder / 'hsi_ran.csv')





