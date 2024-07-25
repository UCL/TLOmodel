"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)"""

import argparse
import textwrap
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.overview_paper.C_impact_of_healthsystem_assumptions.scenario_impact_of_healthsystem import (
            ImpactOfHealthSystemAssumptions,
        )
        e = ImpactOfHealthSystemAssumptions()
        return tuple(e._scenarios.keys())

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)
        """
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

    def do_bar_plot_with_ci(_df, annotations=None, xticklabels_horizontal_and_wrapped=False):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""
        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        fig, ax = plt.subplots()
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
            alpha=0.5,
            ecolor='black',
            capsize=10,
            label=xticks.values()
        )
        if annotations:
            for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
                ax.text(xpos, ypos*1.05, text, horizontalalignment='center')
        ax.set_xticks(list(xticks.keys()))
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

    def rename_scenarios_in_index(ser: pd.Series) -> pd.Series:
        """Update the index of a pd.Series to reflect updated names of each scenario"""
        scenario_renaming = {
            '+ Perfect Clinical Practice': '+ Perfect Healthcare System Function'
        }
        ser.index = pd.Series(ser.index).replace(scenario_renaming)
        return ser

    num_deaths_summarized = rename_scenarios_in_index(num_deaths_summarized)
    num_dalys_summarized = rename_scenarios_in_index(num_dalys_summarized)

    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Status Quo', 'mean']/1e6, color='green', alpha=0.5)
    ax.containers[1][2].set_color('green')
    ax.containers[1][0].set_color('k')
    ax.containers[1][1].set_color('red')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'All Scenarios: DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc['Status Quo', 'mean']/1e6, color='green', alpha=0.5)
    ax.containers[1][2].set_color('green')
    ax.containers[1][0].set_color('k')
    ax.containers[1][1].set_color('red')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_dalys_summarized.loc[['No Healthcare System', 'With Hard Constraints', 'Status Quo']] / 1e6
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.containers[1][2].set_color('green')
    ax.containers[1][0].set_color('k')
    ax.containers[1][1].set_color('red')
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS Averted vs No Healthcare System
    print(
        summarize(
            -1.0 *
            pd.DataFrame(
                find_difference_relative_to_comparison(
                    num_dalys.loc[0],
                    comparison='No Healthcare System')
            ).T
        ).iloc[0].unstack().reindex(param_names) / 1e6
    )

    # DALYS Averted Due to the Squeezing Being Allowable (= Dalys Averted in Hard Constraints vs Status Quo Scenario)
    print(
        summarize(
            -1.0 *
            pd.DataFrame(
                find_difference_relative_to_comparison(
                    num_dalys.loc[0],
                    comparison="With Hard Constraints")
            ).T
        ).iloc[0].unstack().reindex(param_names).loc["Status Quo"] / 1e6
    )



    # %% Deaths and DALYS averted relative to Status Quo
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Status Quo')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Status Quo',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Status Quo')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Status Quo',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    # rename scenarios
    num_deaths_averted = rename_scenarios_in_index(num_deaths_averted)
    pc_deaths_averted = rename_scenarios_in_index(pc_deaths_averted)
    num_dalys_averted = rename_scenarios_in_index(num_dalys_averted)
    pc_dalys_averted = rename_scenarios_in_index(pc_dalys_averted)

    # DEATHS
    name_of_plot = f'Additional Deaths Averted vs Status Quo, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
            for _, row in pc_deaths_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('Additional Deaths Averted')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS
    name_of_plot = f'Additional DALYs Averted vs Status Quo, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
            for _, row in pc_dalys_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 16)
    ax.set_yticks(np.arange(0, 18, 2))
    ax.set_ylabel('Additional DALYS Averted \n(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS (with xtickabels horizontal and wrapped)
    name_of_plot = f'Additional DALYs Averted vs Status Quo, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
            for _, row in pc_dalys_averted.clip(lower=0.0).iterrows()
        ],
        xticklabels_horizontal_and_wrapped=True,
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 16)
    ax.set_yticks(np.arange(0, 18, 2))
    ax.set_ylabel('Additional DALYS Averted \n(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # Plot DALYS incurred wrt wealth under selected scenarios (No HealthSystem, SQ, Perfect Healthcare seeking)
    #  in order to understand the root causes of the observation that more DALYS are averted under SQ in higher
    #  wealth quantiles.

    def get_total_num_dalys_by_wealth_and_label(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        wealth_cats = {5: "0-19%", 4: "20-39%", 3: "40-59%", 2: "60-79%", 1: "80-100%"}

        return (
            _df.loc[_df["year"].between(*[d.year for d in TARGET_PERIOD])]
            .drop(columns=["date", "year"])
            .assign(
                li_wealth=lambda x: x["li_wealth"]
                .map(wealth_cats)
                .astype(pd.CategoricalDtype(wealth_cats.values(), ordered=True))
            )
            .melt(id_vars=["li_wealth"], var_name="label")
            .groupby(by=["li_wealth", "label"])["value"]
            .sum()
        )

    total_num_dalys_by_wealth_and_label = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_by_wealth_stacked_by_age_and_time",
            custom_generate_series=get_total_num_dalys_by_wealth_and_label,
            do_scaling=True,
        ),
    ).pipe(set_param_names_as_column_index_level_0)[
        ['No Healthcare System', 'Status Quo', 'Perfect Healthcare Seeking']
    ].loc[:, (slice(None), 'mean')].droplevel(axis=1, level=1)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    name_of_plot = f'DALYS Incurred by Wealth and Cause {target_period()}'
    for _ax, _scenario_name, in zip(ax, total_num_dalys_by_wealth_and_label.columns):
        format_to_plot = total_num_dalys_by_wealth_and_label[_scenario_name].unstack()
        format_to_plot = format_to_plot \
            .sort_index(axis=0) \
            .reindex(columns=CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys(), fill_value=0.0) \
            .sort_index(axis=1, key=order_of_cause_of_death_or_daly_label)
        (
            format_to_plot / 1e6
        ).plot.bar(stacked=True,
                   ax=_ax,
                   color=[get_color_cause_of_death_or_daly_label(_label) for _label in format_to_plot.columns],
                   )
        _ax.axhline(0.0, color='black')
        _ax.set_title(f'{_scenario_name}')
        if _scenario_name == 'Status Quo':
            _ax.set_ylabel('Number of DALYs Averted (/1e6)')
        _ax.set_ylim(0, 20)
        _ax.set_xlabel('Wealth Percentile')
        _ax.grid()
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.legend(ncol=3, fontsize=8, loc='upper right')
        _ax.legend().set_visible(False)
    fig.suptitle(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # Normalised Total DALYS (where DALYS in the highest wealth class are 100)
    tots = total_num_dalys_by_wealth_and_label.groupby(axis=0, level=0).sum()
    normalised_tots = tots.div(tots.loc['80-100%'])

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False)
    name_of_plot = f'DALYS Incurred by Wealth Normalises {target_period()}'
    (tots / 1e6).plot(
        ax=ax[0],
        color=('black', 'green', (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)),
        alpha=0.5)
    ax[0].set_ylabel('Total DALYS (/million)')
    ax[0].set_xlabel('Wealth Percentile')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[0].set_ylim(0, 15)
    ax[0].legend(fontsize=8)
    ax[0].set_title('DALYS by Wealth (Totals)')
    (normalised_tots * 100.0).plot(
        ax=ax[1],
        color=('black', 'green', (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)),
        alpha=0.5)
    ax[1].set_ylabel('Normalised DALYS\n100 = Highest Wealth quantile')
    ax[1].set_xlabel('Wealth Percentile')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    ax[1].axhline(100.0, color='k', linestyle='--')
    ax[1].legend().set_visible(False)
    ax[1].set_title('DALYS by Wealth (Normalised)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # Which disease are over-represented in the highest wealth categories, compared to others
    #  in No Healthcare System scenario)...?

    dalys_no_hcs = total_num_dalys_by_wealth_and_label['No Healthcare System'].unstack()
    dalys_no_hcs = dalys_no_hcs.sort_index(axis=1, key=order_of_cause_of_death_or_daly_label).drop(columns=['Other'])

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    name_of_plot = 'Distribution of health benefits'
    # Excess DALYS in highest compared to middle
    (
        (dalys_no_hcs.loc['80-100%'] - dalys_no_hcs.loc['40-59%']) / 1e6
    ).plot.bar(
        ax=axs[0],
        color=[get_color_cause_of_death_or_daly_label(_label) for _label in dalys_no_hcs.columns],
    )
    axs[0].set_title('Excess DALYS in Wealth Group 80-100% versus 40-59%')
    axs[0].set_xlabel('Cause')
    axs[0].set_ylabel('DALYS (/millions)')
    axs[0].axhline(0.0, color='black')
    axs[0].set_ylim(-1.5, 2.0)
    axs[0].set_yticks(np.arange(-1.0, 2.5, 1.0))
    axs[0].grid()
    # Excess DALYS in lowest compared to middle
    (
        (dalys_no_hcs.loc['0-19%'] - dalys_no_hcs.loc['40-59%']) / 1e6
    ).plot.bar(
        ax=axs[1],
        color=[get_color_cause_of_death_or_daly_label(_label) for _label in dalys_no_hcs.columns],
    )
    axs[1].set_title('Excess DALYS in Wealth Group 0-19% versus 40-59%')
    axs[1].set_xlabel('Cause')
    axs[1].set_ylabel('DALYS (/millions)')
    axs[1].axhline(0.0, color='black')
    axs[1].set_ylim(-1.5, 2.0)
    axs[1].set_yticks(np.arange(-1.0, 2.5, 1.0))
    axs[1].grid()

    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    # Needed the first time as pickles were not created on Azure side:
    # from tlo.analysis.utils import create_pickles_locally
    # create_pickles_locally(
    #     scenario_output_dir=args.results_folder,
    #     compressed_file_name_prefix=args.results_folder.name.split('-')[0],
    # )

    apply(
        results_folder=args.results_folder,
        output_folder=Path(
            '/Users/tbh03/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/TLM/Papers/Introductory Model Paper/2024_07_11_RESUBMISSION_TO_THE_LANCET_GLOBAL_HEALTH/Comparing results between mean and median/3 - upated using median/healthsystem_under_different_assumptions-2023-11-13T194641Z'
        ),
        resourcefilepath=Path('./resources')
    )
