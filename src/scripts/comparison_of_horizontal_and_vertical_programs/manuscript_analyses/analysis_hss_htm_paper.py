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

        # Rename 'central' to 'mean' if it exists
        if 'central' in _df.columns:
            _df = _df.rename(columns={'central': 'mean'})

        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        # Updated color logic
        if set_colors:
            # def match_color(index_name):
            #     for key, colour in color_map.items():
            #         if key in index_name:
            #             return colour
            #     return 'grey'  # Default colour if no match found
            def match_color(index_name):
                return color_map.get(index_name, 'grey')  # Ensure exact name matching

            colors = [match_color(series) for series in _df.index]
        else:
            cmap = sns.color_palette('Spectral', as_cmap=True)
            rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
            colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

        # if set_colors:
        #     colors = [color_map.get(series, 'grey') for series in _df.index]
        # else:
        #     cmap = sns.color_palette('Spectral', as_cmap=True)
        #     rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        #     colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
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
        median_df = _df.xs('central', level=0, axis=1)
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
        'HRH Scale-up (1%)': '#d8434e',
        'HRH Scale-up (4%)': '#f36b48',
        'HRH Scale-up (6%)': '#fca45c',
        'Increase Capacity at Primary Care Levels': '#fddc89',
        'Consumables Increased to 75th Percentile': '#e7f7a0',
        'Consumables Available at HIV levels': '#a5dc97',
        'Consumables Available at EPI levels': '#6dc0a6',
        'HSS Expansion Package': '#438fba',
        'HIV Program Scale-up Without HSS Expansion': '#5e4fa2',
        'TB Program Scale-up Without HSS Expansion': '#3c71aa',  # New color
        'Malaria Program Scale-up Without HSS Expansion': '#2f6094',  # New color
    }

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # %% Group the scenarios

    HSS_scenarios = [
        'Baseline',
        'HRH Scale-up (1%)',
        'HRH Scale-up (4%)',
        'HRH Scale-up (6%)',
        'Increase Capacity at Primary Care Levels',
        'Consumables Increased to 75th Percentile',
        'Consumables Available at HIV levels',
        'Consumables Available at EPI levels',
        'HSS Expansion Package',
    ]

    # All HIV Program scenarios
    HIV_scenarios = [
        'HIV Program Scale-up Without HSS Expansion',
        'HIV Program Scale-up With HRH Scale-up (1%)',
        'HIV Program Scale-up With HRH Scale-up (4%)',
        'HIV Program Scale-up With HRH Scale-up (6%)',
        'HIV Program Scale-up With Increased HRH at Primary Care Levels',
        'HIV Program Scale-up With Consumables at 75th Percentile',
        'HIV Program Scale-up With Consumables at HIV levels',
        'HIV Program Scale-up With Consumables at EPI levels',
        'HIV Programs Scale-up With HSS Expansion Package',
    ]

    # All TB Program scenarios
    TB_scenarios = [
        'TB Program Scale-up Without HSS Expansion',
        'TB Program Scale-up With HRH Scale-up (1%)',
        'TB Program Scale-up With HRH Scale-up (4%)',
        'TB Program Scale-up With HRH Scale-up (6%)',
        'TB Program Scale-up With Increased HRH at Primary Care Levels',
        'TB Program Scale-up With Consumables at 75th Percentile',
        'TB Program Scale-up With Consumables at HIV levels',
        'TB Program Scale-up With Consumables at EPI levels',
        'TB Programs Scale-up With HSS Expansion Package',
    ]

    # All Malaria Program scenarios
    Malaria_scenarios = [
        'Malaria Program Scale-up Without HSS Expansion',
        'Malaria Program Scale-up With HRH Scale-up (1%)',
        'Malaria Program Scale-up With HRH Scale-up (4%)',
        'Malaria Program Scale-up With HRH Scale-up (6%)',
        'Malaria Program Scale-up With Increased HRH at Primary Care Levels',
        'Malaria Program Scale-up With Consumables at 75th Percentile',
        'Malaria Program Scale-up With Consumables at HIV levels',
        'Malaria Program Scale-up With Consumables at EPI levels',
        'Malaria Programs Scale-up With HSS Expansion Package',
    ]

    # All HTM Program scenarios
    HTM_scenarios = [
        'HTM Program Scale-up Without HSS Expansion',
        'HTM Program Scale-up With HRH Scale-up (1%)',
        'HTM Program Scale-up With HRH Scale-up (4%)',
        'HTM Program Scale-up With HRH Scale-up (6%)',
        'HTM Program Scale-up With Increased HRH at Primary Care Levels',
        'HTM Program Scale-up With Consumables at 75th Percentile',
        'HTM Program Scale-up With Consumables at HIV levels',
        'HTM Program Scale-up With Consumables at EPI levels',
        'HTM Programs Scale-up With HSS Expansion Package',
    ]

    HSS_vs_HTM = [
        'HIV Program Scale-up Without HSS Expansion',
        'HIV Programs Scale-up With HSS Expansion Package',
        'TB Program Scale-up Without HSS Expansion',
        'TB Programs Scale-up With HSS Expansion Package',
        'Malaria Program Scale-up Without HSS Expansion',
        'Malaria Programs Scale-up With HSS Expansion Package',
        'HTM Program Scale-up Without HSS Expansion',
        'HTM Programs Scale-up With HSS Expansion Package',
    ]

    Summary_scenarios = [
        'HSS Expansion Package',
        'HTM Program Scale-up Without HSS Expansion',
        'HTM Programs Scale-up With HSS Expansion Package',
    ]

    scenario_groups = {
        'HSS_scenarios': HSS_scenarios,
        'HIV_scenarios': HIV_scenarios,
        'TB_scenarios': TB_scenarios,
        'Malaria_scenarios': Malaria_scenarios,
        'HTM_scenarios': HTM_scenarios,
        'HSS_vs_HTM': HSS_vs_HTM,
        'Summary_scenarios': Summary_scenarios,
    }

    # Make a separate plot for the scale-up of each program/programs
    program_plots = {
        'HIV programs': [
            'HIV Programs Scale-up Without HSS Expansion',
            'HIV Programs Scale-up With HSS Expansion Package',
        ],
        'TB programs': [
            'TB Programs Scale-up Without HSS Expansion',
            'TB Programs Scale-up With HSS Expansion Package',
        ],
        'Malaria programs': [
            'Malaria Programs Scale-up Without HSS Expansion',
            'Malaria Programs Scale-up With HSS Expansion Package',
        ],
        'Summary': [
            'HSS PACKAGE: Realistic',
            'HTM Programs Scale-up Without HSS Expansion',
            'HTM Programs Scale-up With HSS Expansion Package'
        ]

    }

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
    num_dalys_summarized = compute_summary_statistics(num_dalys, central_measure='median').loc[0].unstack().reindex(
        param_names)
    num_deaths_summarized = compute_summary_statistics(num_deaths, central_measure='median').loc[0].unstack().reindex(
        param_names)
    num_dalys_summarized.to_csv(results_folder / f'num_dalys_summarized_{target_period()}.csv')
    num_deaths_summarized.to_csv(results_folder / f'num_deaths_summarized_{target_period()}.csv')

    # PLOT DEATHS FOR EACH SCENARIO GROUP
    # Iterate through each scenario group
    for group_name, scenarios in scenario_groups.items():
        # Filter or extract data for the current group
        # Assuming `num_deaths_summarized` contains all scenario data
        group_data = num_deaths_summarized.loc[scenarios]

        # Define the plot name
        name_of_plot = f'Deaths, {group_name}, {target_period()}'

        # Generate the plot for the entire group
        fig, ax = do_bar_plot_with_ci(group_data / 1e6)

        # Customise the plot
        ax.set_title(name_of_plot)
        ax.set_ylabel('(Millions)')
        fig.tight_layout()

        # Add a reference line for the Baseline (optional, if Baseline is common across groups)
        baseline_value = num_deaths_summarized.loc['Baseline', 'central'] / 1e6
        ax.axhline(baseline_value, color='black', linestyle='--', alpha=0.5)

        # Save the plot
        file_name = make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', ''))
        fig.savefig(file_name)

        # Show and close the plot
        fig.show()
        plt.close(fig)

    # PLOT DALYS FOR EACH SCENARIO GROUP
    for group_name, scenarios in scenario_groups.items():
        # Filter or extract data for the current group
        # Assuming `num_deaths_summarized` contains all scenario data
        group_data = num_dalys_summarized.loc[scenarios]

        # Define the plot name
        name_of_plot = f'DALYS, {group_name}, {target_period()}'

        # Generate the plot for the entire group
        fig, ax = do_bar_plot_with_ci(group_data / 1e6)

        # Customise the plot
        ax.set_title(name_of_plot)
        ax.set_ylabel('(Millions)')
        fig.tight_layout()

        # Add a reference line for the Baseline (optional, if Baseline is common across groups)
        baseline_value = num_dalys_summarized.loc['Baseline', 'central'] / 1e6
        ax.axhline(baseline_value, color='black', linestyle='--', alpha=0.5)

        # Save the plot
        file_name = make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', ''))
        fig.savefig(file_name)

        # Show and close the plot
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
    num_deaths_averted.to_csv(results_folder / f'num_deaths_averted_{target_period()}.csv')

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_deaths_averted.to_csv(results_folder / f'pc_deaths_averted_{target_period()}.csv')

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    num_dalys_averted.to_csv(results_folder / f'num_dalys_averted_{target_period()}.csv')

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='Baseline',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['Baseline'])
    pc_dalys_averted.to_csv(results_folder / f'pc_dalys_averted_{target_period()}.csv')

    # PLOT DEATHS AVERTED
    # todo this is using spectral colours not color_map
    for group_name, scenarios in scenario_groups.items():
        # Filter or extract data for the current group
        include_scenarios = [scenario for scenario in scenarios if scenario in num_deaths_averted.index]

        # Extract data for the valid scenarios
        group_data = num_deaths_averted.loc[include_scenarios]
        group_data_percent = pc_deaths_averted.loc[include_scenarios]

        # Define the plot name
        name_of_plot = f'Deaths Averted vs Baseline, {group_name}, {target_period()}'

        fig, ax = do_bar_plot_with_ci(
            group_data.clip(lower=0.0),
            annotations=[
                f"{round(row['mean'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
                for _, row in group_data_percent.clip(lower=0.0).iterrows()
            ], set_colors=None, offset=500
        )
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 400_000)
        ax.set_ylabel('Deaths Averted vs Baseline')
        fig.tight_layout()
        plt.subplots_adjust(right=0.55)  # Increase the right margin
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    # PLOT DALYS AVERTED
    # todo this is using spectral colours not color_map
    for group_name, scenarios in scenario_groups.items():
        # Filter or extract data for the current group
        include_scenarios = [scenario for scenario in scenarios if scenario in num_deaths_averted.index]

        # Extract data for the valid scenarios
        group_data = num_dalys_averted.loc[include_scenarios]
        group_data_percent = pc_dalys_averted.loc[include_scenarios]

        # Define the plot name
        name_of_plot = f'DALYS Averted vs Baseline, {group_name}, {target_period()}'

        fig, ax = do_bar_plot_with_ci(
            (group_data / 1e6).clip(lower=0.0),
            annotations=[
                f"{round(row['mean'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
                for _, row in group_data_percent.clip(lower=0.0).iterrows()
            ], set_colors=None, offset=3,
        )
        ax.set_title(name_of_plot)
        ax.set_ylim(0, 40)
        ax.set_ylabel('DALYS Averted vs Baseline \n(Millions)')
        # ax.grid(axis='y', linestyle='--', linewidth=0.5)  # Optional: horizontal lines if needed, adjust style
        ax.grid(axis='x', visible=False)  # Turn off vertical gridlines
        fig.tight_layout()
        plt.subplots_adjust(right=0.55)  # Increase the right margin
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)


    # todo plot DALYs averted vs baseline for HSS vs HTM - for paper figure
    include_scenarios = ['HSS Expansion Package',
                         'HTM Program Scale-up Without HSS Expansion',
                         'HTM Programs Scale-up With HSS Expansion Package']
    color_map = {
        'HSS Expansion Package': '#9e0142',
        'HTM Program Scale-up Without HSS Expansion': '#fdae61',
        'HTM Programs Scale-up With HSS Expansion Package': '#66c2a5',
    }
    # Extract data for the valid scenarios
    group_data = num_dalys_averted.loc[include_scenarios]
    group_data_percent = pc_dalys_averted.loc[include_scenarios]

    # Define the plot name
    name_of_plot = f'DALYS Averted vs Baseline, {group_name}, {target_period()}'

    fig, ax = do_bar_plot_with_ci(
        (group_data / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 0)}% ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
            for _, row in group_data_percent.clip(lower=0.0).iterrows()
        ], set_colors=True, offset=1.5
    )
    ax.set_title('')
    ax.set_ylim(0, 40)
    ax.set_ylabel('DALYS Averted vs Baseline \n(Millions)')
    ax.grid(axis='y', visible=False)
    # ax.grid(axis='y', linestyle='--', linewidth=0.5)  # Optional: horizontal lines if needed, adjust style
    ax.grid(axis='x', visible=False)  # Turn off vertical gridlines
    fig.tight_layout()
    plt.subplots_adjust(right=0.55)  # Increase the right margin
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

    summarise_total_num_dalys_by_label_results = compute_summary_statistics(extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_wealth_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0), central_measure='median')

    summarise_total_num_dalys_by_label_results.to_csv(results_folder / 'summarise_num_dalys_by_label.csv')

    pc_dalys_averted_by_label = 100.0 * compute_summary_statistics(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series_dataframe(
                total_num_dalys_by_label_results,
                comparison='Baseline',
                scaled=True)
        ), central_measure='median'
    )
    pc_dalys_averted_by_label.to_csv(results_folder / 'pc_dalys_averted_by_label.csv')

    total_num_dalys_by_label_results_averted_vs_baseline = compute_summary_statistics(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison='Baseline'
        ),
        central_measure='median'
    )
    total_num_dalys_by_label_results_averted_vs_baseline.to_csv(
        results_folder / 'num_dalys_by_label_averted_vs_baseline.csv')

    def sort_order_of_columns(_df):

        level_0_columns_results = total_num_dalys_by_label_results.columns.get_level_values(0).unique()
        filtered_level_0_columns_results = [col for col in level_0_columns_results if col != 'Baseline']

        # Reindex total_num_dalys_by_label_results_averted_vs_baseline to match the order of Level 0 columns
        reordered_df = _df.reindex(
            columns=filtered_level_0_columns_results
        )

        return reordered_df


    # stacked barplot of DALYs averted by cause
    for group_name, scenarios in scenario_groups.items():
        name_of_plot = f'{group_name}, DALYS Averted by Cause, {target_period()}'
        fig, ax = plt.subplots()

        # Filter the columns to select only those where 'stat' == 'central' and match the scenarios
        central_data = total_num_dalys_by_label_results_averted_vs_baseline.xs('central', level='stat', axis=1)
        include_scenarios = [scenario for scenario in scenarios if scenario in central_data.columns]
        data = central_data[include_scenarios] / 1e6  # Convert to millions

        # Plot each bar stack with the specified color
        num_categories = len(total_num_dalys_by_label_results_averted_vs_baseline.index)
        colours = sns.color_palette('cubehelix', num_categories)
        data.clip(lower=0.0).T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            color=colours,
        )
        ax.set_ylim([0, 30])
        ax.set_title(name_of_plot)
        ax.set_ylabel(f'DALYs Averted vs Baseline, {target_period()}\n(Millions)')
        ax.set_xlabel('')
        wrapped_labs = ["\n".join(textwrap.wrap(_lab.get_text(), 20)) for _lab in ax.get_xticklabels()]
        ax.set_xticklabels(wrapped_labs)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    # %% 5-panel multiplot of DALYS averted by cause for every scenario
    sns.set(style="whitegrid")

    def plot_horizontal_stacked_barpanels(scenario_groups, data, target_period, make_graph_file_name):
        """
        Create a 5-panel horizontal stacked bar plot for scenario groups.

        Parameters:
            scenario_groups (dict): Dictionary of scenario groups.
            data (DataFrame): Data for plotting, filtered by 'central'.
            target_period (callable): Function returning the target period string.
            make_graph_file_name (callable): Function to generate filenames for saving plots.
        """
        # Exclude 'Summary_scenarios'
        scenario_groups = {key: val for key, val in scenario_groups.items() if key != 'Summary_scenarios'}

        # Create the figure and axes
        fig, axes = plt.subplots(1, 5, figsize=(20, 10), sharey=False, constrained_layout=True)

        # Set the colour palette
        num_categories = len(data.index)
        colours = sns.color_palette('cubehelix', num_categories)

        for ax, (group_name, scenarios) in zip(axes, scenario_groups.items()):
            # name_of_plot = f'{group_name}, DALYS Averted by Cause, {target_period()}'

            # Filter the columns to select only those where 'stat' == 'central' and match the scenarios
            central_data = data.xs('central', level='stat', axis=1)
            include_scenarios = [scenario for scenario in scenarios if scenario in central_data.columns]
            plot_data = central_data[include_scenarios] / 1e6  # Convert to millions

            if group_name == 'HSS_scenarios':
                plot_data.insert(0, 'Vertical Programs Only', 0)

            # Plot the horizontal stacked bar chart
            plot_data.clip(lower=0.0).T.plot.barh(
                stacked=True,
                ax=ax,
                color=colours,
                edgecolor='none',
            )

            # Adjust the axis and title
            ax.set_xlim([0, 30])
            if group_name == 'HSS_scenarios':
                ax.set_title('Health System Strengthening')
            if group_name == 'HIV_scenarios':
                ax.set_title('HIV Program')
            if group_name == 'TB_scenarios':
                ax.set_title('TB Program')
            if group_name == 'Malaria_scenarios':
                ax.set_title('Malaria Program')
            if group_name == 'HTM_scenarios':
                ax.set_title('HTM Program')

            # ax.set_title(panel_titles.get(group_name, group_name), fontsize=12)  # Set the new title
            ax.set_xlabel(f'DALYs Averted (Millions)')
            ax.set_ylabel('')

            # Wrap scenario names to improve readability
            wrapped_labels = ["\n".join(textwrap.wrap(label, 25)) for label in plot_data.columns]
            ax.set_yticks(range(len(wrapped_labels)))
            ax.set_yticklabels(wrapped_labels, fontsize=10)

            if group_name == 'HSS_scenarios':
                ax.legend(loc='lower right')
            else:
                ax.get_legend().remove()
                ax.set_yticklabels([])

        # Save and show the plot
        file_name = make_graph_file_name(f'5_Panel_Horizontal_Stacked_Bar_Plot_{target_period().replace(" ", "_")}')
        fig.savefig(file_name, dpi=300)
        plt.show()
        plt.close(fig)

    plot_horizontal_stacked_barpanels(scenario_groups, total_num_dalys_by_label_results_averted_vs_baseline, target_period, make_graph_file_name)

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

    def calculate_deaths_averted(_df):
        """Calculate the number of deaths averted compared to the Baseline for each draw."""

        # Extract the median values
        median_values = _df.xs('central', level=1, axis=1)

        # Get the median values for the 'Baseline' draw
        baseline_values = median_values['Baseline']

        # Calculate diff in number of deaths
        deaths_diff = median_values.subtract(baseline_values, axis=0)

        # Multiply by -1 to get deaths averted
        return deaths_diff * -1

    # Use the function with your data
    deaths_averted_by_cause = calculate_deaths_averted(num_deaths_by_cause)
    deaths_averted_by_cause.to_csv(results_folder / f'num_deaths_averted_by_cause_{target_period()}.csv')


    # %% disease-specific outputs

    def compute_percentage_difference_in_indicator_across_runs(_df):
        """
        Computes the percentage difference between each scenario and the baseline for each run.
        The comparison is made run-by-run.
        negative values indicate lower value in scenario compared with baseline
        """
        baseline_cols = [col for col in _df.columns if "Baseline" in col]
        scenario_cols = [col for col in _df.columns if col not in baseline_cols]

        percentage_differences = _df.copy()
        for scenario_col in scenario_cols:
            run_number = scenario_col[1]  # Extract run number from multi-index
            corresponding_baseline = next(col for col in baseline_cols if col[1] == run_number)
            percentage_differences[scenario_col] = ((_df[scenario_col] - _df[corresponding_baseline]) / _df[
                                                       corresponding_baseline]) * 100

        return percentage_differences

    def summarise_percentage_diffs_by_scenario(_df):
        """
        Computes the median, 2.5th percentile, and 97.5th percentile for each scenario's last row values across runs (excluding 'Baseline').
        """
        # Identify scenario columns
        scenario_cols = [name for name in param_names if "Baseline" not in name]

        summary_stats = {}

        # Compute the summary statistics for each scenario
        for scenario in scenario_cols:
            # Extract the last row for this scenario (across all runs)
            scenario_data = _df[scenario].iloc[-1]  # Get data for the last row

            # Compute median and percentiles over the 5 runs (columns)
            summary_stats[scenario] = {
                "Median": scenario_data.median(),  # Median of the last row values for each run
                "2.5%": scenario_data.quantile(0.025),  # 2.5th percentile
                "97.5%": scenario_data.quantile(0.975)  # 97.5th percentile
            }

        # Return the results as a DataFrame
        return pd.DataFrame(summary_stats).T

    def extract_and_process_results(results_folder, module, key, column, do_scaling):
        """
        Extracts results from the specified module and key, processes the data, and computes summary statistics.
        """
        # Assuming `extract_results` function is available to extract the required data
        _df = extract_results(
            results_folder,
            module=module,
            key=key,
            column=column,
            index='date',
            do_scaling=do_scaling
        ).pipe(set_param_names_as_column_index_level_0)

        # Compute percentage differences
        percentage_diff = compute_percentage_difference_in_indicator_across_runs(_df)
        percentage_diff.to_csv(results_folder / f'{column}_percentage_diff.csv')

        # Compute summary statistics
        summary_stats = summarise_percentage_diffs_by_scenario(percentage_diff)
        summary_stats.to_csv(results_folder / f'{column}_percentage_diff_summary.csv')

        return percentage_diff, summary_stats

    # HIV
    module = 'tlo.methods.hiv'
    key = 'summary_inc_and_prev_for_adults_and_children_and_fsw'
    column = "hiv_adult_inc_15plus"
    hiv_inc_percentage_diff, hiv_inc_summary_stats = extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    column = "hiv_prev_adult_15plus"
    hiv_prev_percentage_diff, hiv_prev_summary_stats = extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    key = 'hiv_program_coverage'
    column = "dx_adult"
    hiv_dx_percentage_diff, hiv_dx_summary_stats = extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    # simple extraction
    hiv_inc = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key='summary_inc_and_prev_for_adults_and_children_and_fsw',
        column='hiv_adult_inc_15plus',
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                 central_measure='median'
                                                 )
    hiv_inc.to_csv(results_folder / 'hiv_inc.csv')

    hiv_num_cases = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key='summary_inc_and_prev_for_adults_and_children_and_fsw',
        column='n_new_infections_adult_1549',
        index='date',
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0),
                                                 central_measure='median'
                                                 )
    hiv_num_cases.to_csv(results_folder / 'hiv_num_cases.csv')

    hiv_dx_coverage = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key='hiv_program_coverage',
        column='dx_adult',
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    hiv_dx_coverage.to_csv(results_folder / 'hiv_dx_coverage.csv')

    column = "art_coverage_adult"
    hiv_art_percentage_diff, hiv_art_summary_stats = extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    hiv_art_coverage = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key='hiv_program_coverage',
        column=column,
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    hiv_art_coverage.to_csv(results_folder / 'hiv_art_coverage.csv')

    column = "art_coverage_adult_VL_suppression"
    hiv_art_VLsuppr_percentage_diff, hiv_art_VLsuppr_summary_stats = extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    # TB
    module = 'tlo.methods.tb'
    key = 'tb_prevalence'
    column = "tbPrevActive"
    extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    # simple extraction
    tb_prevalence = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key=key,
        column=column,
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    tb_prevalence.to_csv(results_folder / 'tb_prevalence.csv')

    key = 'tb_treatment'
    column = "tbTreatmentCoverage"
    extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    tb_tx_coverage = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key=key,
        column=column,
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    tb_tx_coverage.to_csv(results_folder / 'tb_tx_coverage.csv')

    # MALARIA
    module = 'tlo.methods.malaria'
    key = 'incidence'
    column = "inc_1000py"
    extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    key = 'coinfection_prevalence'
    column = "prev_malaria_in_hiv_population"
    extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    key = 'tx_coverage'
    column = "treatment_coverage"
    extract_and_process_results(
        results_folder, module, key, column, do_scaling=False)

    # simple extraction
    mal_tx_coverage = compute_summary_statistics(extract_results(
        results_folder,
        module=module,
        key=key,
        column=column,
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0),
                                                central_measure='median'
                                                )
    mal_tx_coverage.to_csv(results_folder / 'mal_tx_coverage.csv')
    mal_tx_coverage.to_csv(results_folder / 'mal_tx_coverage.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
