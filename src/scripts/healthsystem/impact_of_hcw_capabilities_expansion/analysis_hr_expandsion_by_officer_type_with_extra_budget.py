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
from tlo.analysis.utils import (
    APPT_TYPE_TO_COARSE_APPT_TYPE_MAP,
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    extract_results,
    summarize,
)

# rename scenarios
substitute_labels = {
    's_1': 'no_expansion',
    's_2': 'CDNP_expansion_current',
    's_3': 'C_expansion',
    's_4': 'D_expansion',
    's_5': 'N_expansion',
    's_6': 'P_expansion',
    's_7': 'CD_expansion',
    's_8': 'CN_expansion',
    's_9': 'CP_expansion',
    's_10': 'DN_expansion',
    's_11': 'DP_expansion',
    's_12': 'NP_expansion',
    's_13': 'CDN_expansion',
    's_14': 'CDP_expansion',
    's_15': 'CNP_expansion',
    's_16': 'DNP_expansion',
    's_17': 'CDNP_expansion_equal'
}


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
                .apply(pd.Series)
                .rename(columns=APPT_TYPE_TO_COARSE_APPT_TYPE_MAP)
                .groupby(level=0, axis=1).sum()
                .sum())

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

    def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False, put_labels_in_legend=True):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""

        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        # Define colormap (used only with option `put_labels_in_legend=True`)
        cmap = plt.get_cmap("tab20")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
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
            xtick_values = [substitute_labels[v] for v in xticks.values()]
            xtick_legend = [f'{v}: {substitute_labels[v]}' for v in xticks.values()]
            h, _ = ax.get_legend_handles_labels()
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
    num_appts_summarized = summarize(num_appts, only_mean=True).T.reindex(param_names)
    num_services_summarized = summarize(num_services).loc[0].unstack().reindex(param_names)

    # plot absolute numbers for scenarios

    name_of_plot = f'Services, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_services_summarized / 1e6, xticklabels_horizontal_and_wrapped=True,
                                  put_labels_in_legend=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6, xticklabels_horizontal_and_wrapped=True,
                                  put_labels_in_legend=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6, xticklabels_horizontal_and_wrapped=True,
                                  put_labels_in_legend=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Services by appointment type, {target_period()}'
    num_appts_summarized_in_millions = num_appts_summarized / 1e6
    appt_color = {
        appt: COARSE_APPT_TYPE_TO_COLOR_MAP.get(appt, np.nan) for appt in num_appts_summarized_in_millions.columns
    }
    yerr_services = np.array([
        (num_services_summarized['mean'].values - num_services_summarized['lower']).values,
        (num_services_summarized['upper'].values - num_services_summarized['mean']).values,
    ])/1e6
    fig, ax = plt.subplots()
    num_appts_summarized_in_millions.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    ax.errorbar([0, 1], num_services_summarized['mean'].values / 1e6, yerr=yerr_services,
                fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_appts_summarized_in_millions.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # todo
    # get data of extra budget, extra staff
    # calculate return on investment
    # plot DALYS by causes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/bshe@ic.ac.uk/scenario_run_for_hcw_expansion_analysis-2024-08-16T160132Z
    args = parser.parse_args()

    # Produce results for short-term analysis: 5 years

    # # 2015-2019, before change, incl. mode, hr expansion, etc.
    # apply(
    #     results_folder=args.results_folder,
    #     output_folder=args.results_folder,
    #     resourcefilepath=Path('./resources'),
    #     the_target_period=(Date(2015, 1, 1), Date(2019, 12, 31))
    # )
    #
    # # 2020-2024
    # apply(
    #     results_folder=args.results_folder,
    #     output_folder=args.results_folder,
    #     resourcefilepath=Path('./resources'),
    #     the_target_period=(Date(2020, 1, 1), Date(2024, 12, 31))
    # )

    # Produce results for long-term analysis: 10 years
    # 2020-2029
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2029, 12, 31))
    )
