"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)"""

import argparse
import textwrap
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.impact_of_historical_changes_in_hr.scenario_historical_changes_in_hr import HistoricalChangesInHRH
from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, the_target_period: Tuple[Date, Date] = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = the_target_period

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        e = HistoricalChangesInHRH()
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

    def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False, put_labels_in_legend=True):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""

        substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        # Define colormap (used only with option `put_labels_in_legend=True`)
        cmap = plt.get_cmap("tab20")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
            alpha=0.8,
            ecolor='black',
            color=colors,
            capsize=10,
            label=xticks.values(),
            zorder=100,
        )
        if annotations:
            for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
                ax.text(xpos, ypos*1.15, text, horizontalalignment='center', rotation='vertical', fontsize='x-small')
        ax.set_xticks(list(xticks.keys()))

        if put_labels_in_legend:
            # Update xticks label with substitute labels
            # Insert legend with updated labels that shows correspondence between substitute label and original label
            xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]
            xtick_legend = [f'{letter}: {label}' for letter, label in zip(substitute_labels, xticks.values())]
            h, l = ax.get_legend_handles_labels()
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

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()
    counterfactual_scenario = 'Counterfactual (No Scale-up)'
    actual_scenario = 'Actual (Scale-up)'

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

    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6, xticklabels_horizontal_and_wrapped=True, put_labels_in_legend=False)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc[counterfactual_scenario, 'mean']/1e6, color='black', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6, xticklabels_horizontal_and_wrapped=True, put_labels_in_legend=False)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc[counterfactual_scenario, 'mean']/1e6, color='black', alpha=0.5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


    # %% Deaths and DALYS averted relative to Counterfactual
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison=counterfactual_scenario)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([counterfactual_scenario])

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison=counterfactual_scenario,
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([counterfactual_scenario])

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison=counterfactual_scenario)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([counterfactual_scenario])

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison=counterfactual_scenario,
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([counterfactual_scenario])

    # DEATHS
    name_of_plot = f'Deaths Averted vs Counterfactual, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=None,
        put_labels_in_legend=False,
        xticklabels_horizontal_and_wrapped=True,
    )
    annotation = (f"{int(round(num_deaths_averted.loc[actual_scenario,'mean'], -3))} ({int(round(num_deaths_averted.loc[actual_scenario, 'lower'], -3))} - {int(round(num_deaths_averted.loc[actual_scenario,'upper'], -3))})\n"
                  f"{round(pc_deaths_averted.loc[actual_scenario, 'mean'])} ({round(pc_deaths_averted.loc[actual_scenario,'lower'], 1)} - {round(pc_deaths_averted.loc[actual_scenario, 'upper'], 1)})% of that in Counterfactual"
                  )
    ax.set_title(f"{name_of_plot}\n{annotation}")
    ax.set_ylabel('Deaths Averted vs Counterfactual')
    fig.set_figwidth(5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS
    name_of_plot = f'DALYs Averted vs Counterfactual, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=None,
        put_labels_in_legend=False,
        xticklabels_horizontal_and_wrapped=True,
    )
    annotation = (f"{int(round(num_dalys_averted.loc[actual_scenario,'mean'], -4))} ({int(round(num_dalys_averted.loc[actual_scenario, 'lower'], -4))} - {int(round(num_dalys_averted.loc[actual_scenario,'upper'], -4))})\n"
                  f"{round(pc_dalys_averted.loc[actual_scenario, 'mean'])} ({round(pc_dalys_averted.loc[actual_scenario,'lower'], 1)} - {round(pc_dalys_averted.loc[actual_scenario, 'upper'], 1)})% of that in Counterfactual"
                  )
    ax.set_title(f"{name_of_plot}\n{annotation}")
    ax.set_ylabel('DALYS Averted \n(Millions)')
    fig.set_figwidth(5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # Graphs showing difference by disease (HTM/OTHER and split by age/sex)
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
            '': 'Other',    # defined in order to use this dict to determine ordering of the causes in output
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

    total_num_dalys_by_label_results_averted_vs_baseline = summarize(
        -1.0 * find_difference_relative_to_comparison_series_dataframe(
            total_num_dalys_by_label_results,
            comparison=counterfactual_scenario,
        ),
        only_mean=True
    )

    # Check that when we sum across the causes, we get the same total as calculated when we didn't split by cause.
    assert (
        (total_num_dalys_by_label_results_averted_vs_baseline.sum(axis=0).sort_index()
         - num_dalys_averted['mean'].sort_index()
         ) < 1e-6
    ).all()

    yerr = np.array([
        (num_dalys_averted['mean'].values - num_dalys_averted['lower']).values,
        (num_dalys_averted['upper'].values - num_dalys_averted['mean']).values,
    ])/1e6

    make_string_number = lambda row: f"{round(row['mean']/1e6,1)} ({round(row['lower']/1e6, 1)}-{round(row['upper']/1e6, 1)}) Million"
    str_num_dalys_averted = f'{make_string_number(num_dalys_averted.loc[actual_scenario])}'

    make_string_percent = lambda row: f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)})"
    str_pc_dalys_averted = f'{make_string_percent(pc_dalys_averted.loc[actual_scenario])}% of DALYS in Counterfactual'

    def make_daly_split_by_cause_graph(df):
        name_of_plot = f'DALYS Averted: Actual vs Counterfactual, {target_period()}'
        fig, ax = plt.subplots()
        (df.iloc[::-1] /1e6).T.plot.bar(
            stacked=True,
            ax=ax,
            rot=0,
            alpha=0.75,
            zorder=3,
            legend=False,
            color=['orange', 'teal', 'purple', 'red']
        )
        ax.errorbar(0, num_dalys_averted['mean'].values/1e6, yerr=yerr, fmt="o", color="black", zorder=4)
        ax.set_title(name_of_plot + '\n' + str_num_dalys_averted + '\n' + str_pc_dalys_averted)
        ax.set_ylabel(f'DALYs Averted\n(Millions)')
        ax.set_xlabel('')
        ax.set_xlim(-0.5, 0.65)
        ax.set_ylim(bottom=0)
        ax.get_xaxis().set_ticks([])
        wrapped_labs = ["\n".join(textwrap.wrap(_lab.get_text(), 20)) for _lab in ax.get_xticklabels()]
        ax.set_xticklabels(wrapped_labs)
        ax.grid(axis='y', zorder=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Cause of DALYS', loc='center right')
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
        fig.show()
        plt.close(fig)

    # Make graph - separating H/T/M/Other
    make_daly_split_by_cause_graph(total_num_dalys_by_label_results_averted_vs_baseline)

    # Make graph - separating HTM-Combined/Other
    total_num_dalys_by_label_results_averted_vs_baseline_grouping_htm = total_num_dalys_by_label_results_averted_vs_baseline.groupby(
        total_num_dalys_by_label_results_averted_vs_baseline.index == 'Other').sum().rename(
        index={False: "H/T/M", True: "Other"})
    make_daly_split_by_cause_graph(total_num_dalys_by_label_results_averted_vs_baseline_grouping_htm)

    # percent of DALYS averted in HTM
    1.0 - (total_num_dalys_by_label_results_averted_vs_baseline.loc['Other'] / total_num_dalys_by_label_results_averted_vs_baseline.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    # Produce results for short-term analysis
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2017, 1, 1), Date(2024, 12, 31))
    )

    # Produce results for long-term analysis
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2030, 12, 31))
    )
