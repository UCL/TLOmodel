"""
Extract results of HRH staff counts, DALYs and Deaths across multiple historical HRH growth scenarios.
"""

import argparse
import textwrap
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.impact_of_historical_changes_in_hr.scenario_historical_changes_in_hr_extended import (
    HistoricalChangesInHRH,
)
from tlo import Date
from tlo.analysis.utils import extract_results, make_age_grp_lookup, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, the_target_period: Tuple[Date, Date] = None):

    TARGET_PERIOD = the_target_period
    hrh_check_period = (Date(2018, 1, 1), Date(2026, 1, 1))

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        e = HistoricalChangesInHRH()
        return tuple(e._scenarios.keys())

    def get_num_deaths_by_year_cause(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)"""
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]
        _df['year'] = _df['date'].dt.year
        _df = _df.groupby(['year', 'cause']) \
            .agg({'person_id': 'count'}) \
            .rename(columns={'person_id': 'num_deaths'})['num_deaths']

        return _df

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    def get_total_num_dalys_by_label_all_causes(_df):
        """Return the total number of DALYS in the TARGET_PERIOD cause label."""
        _df = _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'age_range', 'sex']) \
            .groupby('year') \
            .sum() \
            .reset_index() \
            .melt(id_vars='year', var_name='cause', value_name='dalys') \
            .set_index(['year', 'cause'])['dalys']

        return _df

    def get_staff_counts(_df):
        _df['year'] = _df['date'].dt.year
        _df = _df.loc[pd.to_datetime(_df['date']).between(*hrh_check_period), ['year', 'GenericClinic']
                      ].set_index('year').rename(columns={'GenericClinic': 'facility_officer'})
        _df_staff = _df['facility_officer'].apply(pd.Series).stack().reset_index()
        _df_staff.columns = ['year', 'facility_officer', 'staff_count']
        _df_staff[['facility_id', 'officer_type']] = _df_staff['facility_officer'].str.extract(
            r'FacilityID_(\d+)_Officer_(.*)'
        )
        _df_staff['facility_id'] = _df_staff['facility_id'].astype(int)

        main_cadres = ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA']
        _df_staff.loc[
            ~_df_staff["officer_type"].isin(main_cadres),
            "officer_type"
        ] = "Other"

        _df_staff = _df_staff.groupby(['year', 'officer_type'])['staff_count'].sum()
        return _df_staff

    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

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

    def get_num_treatments_by_year_treatment(_df):
        """Return the number of treatments by short treatment id and year (within the TARGET_PERIOD)"""
        _df['year'] = _df['date'].dt.year
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), ['year', 'TREATMENT_ID']].set_index('year')
        _df = _df['TREATMENT_ID'].apply(pd.Series)
        _df.columns = _df.columns.map(lambda x: x.split('_')[0] + "*")
        _df = _df.T.groupby(level=0).sum().T
        _df = _df.stack()
        _df.index = _df.index.set_names(["year", "treatment_type"])
        _df.name = "count"
        return _df

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
            xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]
            xtick_legend = [f'{letter}: {label}' for letter, label in zip(substitute_labels, xticks.values())]
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

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # Check HRH staff counts
    hcw_count = (extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_staff_counts,
        do_scaling=False
    )).pipe(set_param_names_as_column_index_level_0)
    hcw_count = hcw_count.loc[:, hcw_count.columns.get_level_values("run") == 0]
    hcw_count.columns = hcw_count.columns.get_level_values(0)

    hcw_count = (
        hcw_count.stack()
        .reset_index(name='value')
        .rename(columns={'draw': 'scenario'})
    )

    hcw_count = hcw_count.sort_values(['officer_type', 'scenario', 'year'])

    hcw_count['scale_factor'] = (
        hcw_count['value'] /
        hcw_count.groupby(['officer_type', 'scenario'])['value'].shift(1)
    )

    from matplotlib.lines import Line2D
    # marker for each officer type
    markers = {
        'Clinical': 'o',
        'Nursing_and_Midwifery': '*',
        'Pharmacy': '^',
        'DCSA': 'd',
        'Other': 'X',
    }
    marker_sizes = {
        'Clinical': 6,
        'Nursing_and_Midwifery': 8,
        'Pharmacy': 6,
        'DCSA': 6,
        'Other': 6,
    }

    # color for each scenario
    cmap = plt.cm.get_cmap('tab10', len(param_names))
    colors = {
        scenario: cmap(i)
        for i, scenario in enumerate(param_names)
    }

    name_of_plot = "Number of healthcare workers"
    fig, ax = plt.subplots(figsize=(12, 5))

    for officer in hcw_count['officer_type'].unique():
        for scenario in hcw_count['scenario'].unique():
            subset = hcw_count[
                (hcw_count['officer_type'] == officer) &
                (hcw_count['scenario'] == scenario)
                ]

            ax.plot(
                subset['year'],
                subset['value'],
                linestyle="--",
                marker=markers[officer],
                markersize=marker_sizes[officer],
                color=colors[scenario],
                label=f'{officer} - {scenario}',
            )

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of staff')
    ax.set_xticks(sorted(hcw_count['year'].unique()))

    officer_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[officer],
            color='black',
            linestyle='None',
            markersize=8,
            label=officer
        )
        for officer in markers.keys()
    ]

    legend_officer = ax.legend(
        handles=officer_handles,
        title='Officer type',
        loc='center left',
        bbox_to_anchor=(1.02, 0.65),
        fontsize=8,
        title_fontsize=9
    )
    ax.add_artist(legend_officer)

    scenario_handles = [
        Line2D(
            [0],
            [0],
            color=colors[scenario],
            linestyle='--',
            linewidth=2,
            label=scenario
        )
        for scenario in colors.keys()
    ]

    ax.legend(
        handles=scenario_handles,
        title='Scenario',
        loc='center left',
        bbox_to_anchor=(1.02, 0.25),
        fontsize=8,
        title_fontsize=9
    )

    plt.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()
    plt.close(fig)

    # Check total DALYs
    # %% Define parameter names
    counterfactual_scenario = 'Main Counterfactual'
    actual_scenario = 'Main Actual'

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
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6, xticklabels_horizontal_and_wrapped=True,
                                  put_labels_in_legend=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc[counterfactual_scenario, 'mean'] / 1e6, color='black', alpha=0.5)
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6, xticklabels_horizontal_and_wrapped=True,
                                  put_labels_in_legend=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc[counterfactual_scenario, 'mean'] / 1e6, color='black', alpha=0.5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # %% Deaths and DALYS averted relative to Actual
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison=actual_scenario)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([actual_scenario])

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison=actual_scenario,
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([actual_scenario])

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison=actual_scenario)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([actual_scenario])

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison=actual_scenario,
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop([actual_scenario])

    # DEATHS
    name_of_plot = f'Deaths Averted vs Historical growth (uniform), {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        pc_deaths_averted,  # num_deaths_averted
        annotations=None,
        put_labels_in_legend=True,
        xticklabels_horizontal_and_wrapped=True,
    )
    # annotation = (
    #     f"{int(round(num_deaths_averted.loc[actual_scenario, 'mean'], -3))} ({int(round(num_deaths_averted.loc[actual_scenario, 'lower'], -3))} - {int(round(num_deaths_averted.loc[actual_scenario, 'upper'], -3))})\n"
    #     f"{round(pc_deaths_averted.loc[actual_scenario, 'mean'])} ({round(pc_deaths_averted.loc[actual_scenario, 'lower'], 1)} - {round(pc_deaths_averted.loc[actual_scenario, 'upper'], 1)})% of that in Counterfactual"
    #     )
    ax.set_title(f"{name_of_plot}")
    ax.set_ylabel('Deaths Averted vs Historical growth (uniform)')
    # fig.set_figwidth(5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS
    name_of_plot = f'DALYs Averted vs Historical growth (uniform), {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        pc_dalys_averted,  # (num_dalys_averted / 1e6),
        annotations=None,
        put_labels_in_legend=True,
        xticklabels_horizontal_and_wrapped=True,
    )
    # annotation = (
    #     f"{int(round(num_dalys_averted.loc[actual_scenario, 'mean'], -4))} ({int(round(num_dalys_averted.loc[actual_scenario, 'lower'], -4))} - {int(round(num_dalys_averted.loc[actual_scenario, 'upper'], -4))})\n"
    #     f"{round(pc_dalys_averted.loc[actual_scenario, 'mean'])} ({round(pc_dalys_averted.loc[actual_scenario, 'lower'], 1)} - {round(pc_dalys_averted.loc[actual_scenario, 'upper'], 1)})% of that in Counterfactual"
    #     )
    ax.set_title(f"{name_of_plot}")
    ax.set_ylabel('DALYS Averted vs Historical growth (uniform)')
    # fig.set_figwidth(5)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # Prepare Absolute Number of Deaths and DALYs for Izzy
    num_deaths_by_year_cause = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_year_cause,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).stack(['draw', 'run']).reset_index(name='num_deaths')

    num_dalys_by_year_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label_all_causes,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0).stack(['draw', 'run']).reset_index(name='num_dalys')

    # And absolute Number of treatments upon analysis needs
    num_treatments_by_year_treatment = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments_by_year_treatment,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).stack(['draw', 'run']).reset_index(name='num_treatments')

    num_dalys_by_year_cause.to_csv(output_folder / 'num_dalys_by_year_cause (for Izzy).csv', index=False)
    num_deaths_by_year_cause.to_csv(output_folder / 'num_deaths_by_year_cause (for Izzy).csv', index=False)
    num_treatments_by_year_treatment.to_csv(
        output_folder / 'num_treatments_by_year_treatment (for Izzy).csv', index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    # # Produce results for short-term analysis - 2020 - 2024 (incl.)
    # apply(
    #     results_folder=args.results_folder,
    #     output_folder=args.results_folder,
    #     resourcefilepath=Path('./resources'),
    #     the_target_period=(Date(2020, 1, 1), Date(2024, 12, 31))
    # )
    # Produce results for only later period 2025-2030 (incl.)
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2030, 12, 31))
    )
