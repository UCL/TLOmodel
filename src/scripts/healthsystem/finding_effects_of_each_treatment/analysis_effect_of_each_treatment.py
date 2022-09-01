"""Produce plots to show the impact each set of treatments."""

import argparse
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.calibration_analyses.analysis_scripts import plot_legends
from scripts.healthsystem.finding_effects_of_each_treatment import plot_org_chart_treatment_ids
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_coarse_appt_type,
    get_color_cause_of_death_label,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    make_age_grp_lookup,
    make_age_grp_types,
    order_of_cause_of_death_label,
    order_of_coarse_appt,
    order_of_short_treatment_ids,
    squarify_neat,
    summarize,
    to_age_group,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2010, 1, 1), Date(2014, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.finding_effects_of_each_treatment.scenario_effect_of_each_treatment import (
            EffectOfEachTreatment,
        )
        e = EffectOfEachTreatment()
        return tuple(e._scenarios.keys())

    def format_scenario_name(_sn: str) -> str:
        """Return a reformatted scenario name ready for plotting.
        - Remove prefix of "No "
        - Remove suffix of "*"
        """

        if _sn == "Everything":
            # This is when every TREATMENT_ID is allowed to occur.
            return _sn

        elif _sn == "Nothing":
            return "*"
            # In the scenario called "Nothing", all interventions are off. (So, the difference relative to "Everything"
            # reflects the effects of all the interventions.)

        else:
            return _sn.lstrip("No ")

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])

        reformatted_names = map(format_scenario_name, names_of_cols_level0)
        _df.columns = _df.columns.set_levels(reformatted_names, level=0)
        return _df

    def find_difference_extra_relative_to_comparison(_ser: pd.Series,
                                                     comparison: str,
                                                     scaled: bool = False,
                                                     drop_comparison: bool = True
                                                     ):
        """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1). Drop the comparison entries. The comparison is made: DIFF(X) = X - COMPARISON. """
        return _ser \
            .unstack() \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=0) \
            .drop(index=([comparison] if drop_comparison else [])) \
            .stack()

    def find_mean_difference_in_appts_relative_to_comparison(_df: pd.DataFrame,
                                                             comparison: str,
                                                             drop_comparison: bool = True
                                                             ):
        """Find the mean difference in the number of appointments between each draw and the comparison draw (within each
        run). We are looking for the number FEWER appointments that occur when treatment does not happen, so we flip the
         sign (as `find_extra_difference_relative_to_comparison` gives the number extra relative the comparison)."""
        return - summarize(pd.concat({
            _idx: find_difference_extra_relative_to_comparison(row,
                                                               comparison=comparison,
                                                               drop_comparison=drop_comparison)
            for _idx, row in _df.iterrows()
        }, axis=1).T, only_mean=True)

    def find_mean_difference_extra_relative_to_comparison_dataframe(_df: pd.DataFrame,
                                                                    comparison: str,
                                                                    drop_comparison: bool = True,
                                                                    ):
        """Same as `find_difference_extra_relative_to_comparison` but for pd.DataFrame, which is the same as
        `find_mean_difference_in_appts_relative_to_comparison`.
        """
        # todo factorize these three functions more -- it's the same operation for a pd.Series or a pd.DataFrame
        return summarize(pd.concat({
            _idx: find_difference_extra_relative_to_comparison(row,
                                                               comparison=comparison,
                                                               drop_comparison=drop_comparison)
            for _idx, row in _df.iterrows()
        }, axis=1).T, only_mean=True)

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # %% Quantify the health gains associated with all interventions combined.

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    def get_num_dalys_by_cause_label(_df):
        """Return total number of DALYS (Stacked) by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
            .drop(columns=['date', 'sex', 'age_range', 'year']) \
            .sum()

    num_deaths_by_cause_label = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=get_num_deaths_by_cause_label,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)[['Everything', '*']]
    )

    num_dalys_by_cause_label = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_cause_label,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)[['Everything', '*']]
    )

    # Plots.....
    def do_bar_plot_with_ci(_df, _ax):
        """Make a vertical bar plot for each Cause-of-Death Label for the _df onto axis _ax"""
        _df_sorted = _df.loc[order_of_cause_of_death_label(_df.index)]  # sort cause-of-death labels

        for i, cause_label in enumerate(_df_sorted.index):
            # plot bar for one cause
            color = get_color_cause_of_death_label(cause_label)
            one_cause = _df.loc[cause_label]

            mean_deaths = one_cause.loc[(slice(None), "mean")]
            lower_bar = mean_deaths["Everything"]  # (When all interventions are on)
            full_height_of_bar = mean_deaths["*"]  # (When all interventions are off)
            upper_bar = full_height_of_bar - lower_bar
            lower_bar_yerr = np.array([
                one_cause.loc[("Everything", "mean")] - one_cause.loc[("Everything", "lower")],
                one_cause.loc[("Everything", "upper")] - one_cause.loc[("Everything", "mean")]
            ]).reshape(2, 1)
            full_height_bar_yerr = np.array([
                one_cause.loc[("*", "mean")] - one_cause.loc[("*", "lower")],
                one_cause.loc[("*", "upper")] - one_cause.loc[("*", "mean")]
            ]).reshape(2, 1)

            lb, = ax.bar(i, lower_bar, yerr=lower_bar_yerr, bottom=0, label="All TREATMENT_IDs", color=color)
            ub, = _ax.bar(i, upper_bar, yerr=full_height_bar_yerr, bottom=lower_bar, label="No TREATMENT_IDs",
                          color=color, alpha=0.5)
        _ax.set_xticks(range(len(_df_sorted.index)))
        _ax.set_xticklabels(_df_sorted.index, rotation=90)
        _ax.legend([lb, ub], ['All TREATMENT_IDs', 'No TREATMENT_IDs'], loc='upper right')

    fig, ax = plt.subplots()
    name_of_plot = f'Deaths With None or All TREATMENT_IDs, {target_period()}'
    do_bar_plot_with_ci(num_deaths_by_cause_label / 1e3, ax)
    ax.set_title(name_of_plot)
    ax.set_xlabel('Cause of Death')
    ax.set_ylabel('Number of Deaths (/1000)')
    ax.set_ylim(0, 400)
    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    fig, ax = plt.subplots()
    name_of_plot = f'DALYS With None or All TREATMENT_IDs, {target_period()}'
    do_bar_plot_with_ci(num_dalys_by_cause_label / 1e6, ax)
    ax.set_title(name_of_plot)
    ax.set_xlabel('Cause of Disability/Death')
    ax.set_ylabel('Number of DALYS Averted (1/1e6)')
    ax.set_ylim(0, 15)
    ax.set_yticks(range(0, 18, 3))
    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    # %%  Quantify the health gais associated with each TREATMENT_ID (short) individually (i.e., the
    # difference in deaths and DALYS between each scenario and the 'Everything' scenario.)

    def get_num_deaths_by_age_group(_df):
        """Return total number of deaths (total by age-group within the TARGET_PERIOD)"""
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['age'].map(age_grp_lookup).astype(make_age_grp_types())) \
            .size()

    def do_barh_plot_with_ci(_df, _ax):
        """Make a horizontal bar plot for each TREATMENT_ID for the _df onto axis _ax"""
        errors = pd.concat([
            _df['mean'] - _df['lower'],
            _df['upper'] - _df['mean']
        ], axis=1).T.to_numpy()
        _df.plot.barh(
            ax=_ax, y='mean', xerr=errors, legend=False, color=[get_color_short_treatment_id(_id) for _id in _df.index]
        )

    def do_label_barh_plot(_df, _ax):
        """Add text annotation from values in _df onto _ax"""
        y_cords = {ylabel.get_text(): ytick for ytick, ylabel in zip(_ax.get_yticks(), _ax.get_yticklabels())}

        pos_on_rhs = _ax.get_xticks()[-1]

        for label, row in _df.iterrows():
            if row['mean'] > 0:
                annotation = f"{round(row['mean'], 1)} ({round(row['lower'])}-{round(row['upper'])}) %"
                _ax.annotate(annotation,
                             xy=(pos_on_rhs, y_cords.get(label)),
                             xycoords='data',
                             horizontalalignment='left',
                             verticalalignment='center',
                             size=7
                             )

    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_age_group,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across age-groups)

    num_deaths_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Everything')).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True).drop(['FirstAttendance*'])

    pc_deaths_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Everything', scaled=True)).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True).drop(['FirstAttendance*'])

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_cause_label,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across causes)

    num_dalys_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Everything')).T
    ).iloc[0].unstack().drop(['FirstAttendance*']).sort_values(by='mean', ascending=True)

    pc_dalys_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Everything', scaled=True)).T
    ).iloc[0].unstack().drop(['FirstAttendance*']).sort_values(by='mean', ascending=True)

    # PLOTS FOR EACH TREATMENT_ID (Short)
    fig, ax = plt.subplots()
    name_of_plot = f'Deaths Averted by Each TREATMENT_ID, {target_period()}'
    do_barh_plot_with_ci(num_deaths_averted.drop(['*']) / 1e3, ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Number of Deaths Averted (/1000)')
    ax.set_xlim(0, 140)
    do_label_barh_plot(pc_deaths_averted.drop(['*']), ax)
    ax.grid()
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    fig, ax = plt.subplots()
    name_of_plot = f'DALYS Averted by Each TREATMENT_ID, {target_period()}'
    do_barh_plot_with_ci(num_dalys_averted.drop(['*']) / 1e6, ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Number of DALYS Averted (1/1e6)')
    ax.set_xlim(0, 6)
    do_label_barh_plot(pc_dalys_averted.drop(['*']), ax)
    ax.grid()
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    # %% Quantify the health associated with each TREATMENT_ID (short) SPLIT BY AGE and WEALTH

    def get_total_num_death_by_agegrp_and_label(_df):
        """Return the total number of deaths in the TARGET_PERIOD by age-group and cause label."""
        age_group = to_age_group(_df['age'])
        return _df \
            .loc[_df['date'].between(*TARGET_PERIOD)] \
            .groupby([age_group, 'label'])['person_id'].size()

    total_num_death_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_total_num_death_by_agegrp_and_label,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    deaths_averted_by_agegrp_and_label = find_mean_difference_extra_relative_to_comparison_dataframe(
        total_num_death_by_agegrp_and_label, comparison='Everything'
    ).drop(columns=['FirstAttendance*'])

    for _scenario_name, _deaths_av in deaths_averted_by_agegrp_and_label.T.iterrows():
        format_to_plot = _deaths_av.unstack()
        format_to_plot.index = format_to_plot.index.astype(make_age_grp_types())
        format_to_plot = format_to_plot.sort_index(axis=0)
        format_to_plot = format_to_plot[order_of_cause_of_death_label(format_to_plot.columns)]

        fig, ax = plt.subplots()
        name_of_plot = f'Deaths Averted by {_scenario_name} by Age and Cause {target_period()}'
        (
            format_to_plot / 1000
        ).plot.bar(stacked=True, ax=ax,
                   color=[get_color_cause_of_death_label(_label) for _label in format_to_plot.columns],
                   )
        ax.axhline(0.0, color='black')
        ax.set_title(name_of_plot)
        ax.set_ylabel('Number of Deaths Averted (/1000)')
        ax.set_ylim(-50, 150)
        ax.set_xlabel('Age-group')
        ax.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(ncol=3, fontsize=8, loc='upper right')
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        fig.show()

    def get_total_num_death_by_wealth_and_label(_df):
        """Return the total number of deaths in the TARGET_PERIOD by wealth and cause label."""
        wealth_cats = {5: '0-19%', 4: '20-39%', 3: '40-59%', 2: '60-79%', 1: '80-100%'}
        wealth_group = _df['li_wealth'] \
            .map(wealth_cats) \
            .astype(pd.CategoricalDtype(wealth_cats.values(), ordered=True))

        return _df \
            .loc[_df['date'].between(*TARGET_PERIOD)] \
            .groupby([wealth_group, 'label'])['person_id'].size()

    total_num_death_by_wealth_and_label = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_total_num_death_by_wealth_and_label,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    deaths_averted_by_wealth_and_label = find_mean_difference_extra_relative_to_comparison_dataframe(
        total_num_death_by_wealth_and_label, comparison='Everything'
    ).drop(columns=['FirstAttendance*'])

    for _scenario_name, _deaths_av in deaths_averted_by_wealth_and_label.T.iterrows():
        format_to_plot = _deaths_av.unstack()
        format_to_plot = format_to_plot.sort_index(axis=0)
        format_to_plot = format_to_plot[order_of_cause_of_death_label(format_to_plot.columns)]

        fig, ax = plt.subplots()
        name_of_plot = f'Deaths Averted by {_scenario_name} by Wealth and Cause {target_period()}'
        (
            format_to_plot / 1000
        ).plot.bar(stacked=True, ax=ax,
                   color=[get_color_cause_of_death_label(_label) for _label in format_to_plot.columns],
                   )
        ax.axhline(0.0, color='black')
        ax.set_title(name_of_plot)
        ax.set_ylabel('Number of Deaths Averted (/1000)')
        ax.set_ylim(-50, 150)
        ax.set_xlabel('Wealth Percentile')
        ax.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(ncol=3, fontsize=8, loc='upper right')
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        fig.show()

    # %% Quantify the healthcare system resources used with each TREATMENT_ID (short) (The difference in the number of
    # appointments between each scenario and the 'Everything' scenario.)

    # 1) Examine the HSI that are occurring by TREATMENT_ID

    def get_counts_of_hsi_by_short_treatment_id(_df):
        """Get the counts of the short TREATMENT_IDs occurring (up to first underscore)"""
        _counts_by_treatment_id = _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

    counts_of_hsi_by_short_treatment_id = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index().drop(columns=['FirstAttendance*'])

    mean_num_hsi_by_short_treatment_id = summarize(counts_of_hsi_by_short_treatment_id, only_mean=True)

    for scenario_name, _counts in mean_num_hsi_by_short_treatment_id.T.iterrows():
        _counts_non_zero = _counts[_counts > 0]

        if len(_counts_non_zero):
            fig, ax = plt.subplots()
            name_of_plot = f'HSI Events Occurring, {scenario_name}, {target_period()}'
            squarify_neat(
                sizes=_counts_non_zero.values,
                label=_counts_non_zero.index,
                colormap=get_color_short_treatment_id,
                alpha=1,
                pad=True,
                ax=ax,
                text_kwargs={'color': 'black', 'size': 8},
            )
            ax.set_axis_off()
            ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
            fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
            fig.show()

    # 2) Examine the Difference in the number/type of appointments occurring

    def get_counts_of_appts(_df):
        """Get the counts of appointments of each type being used."""
        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)

    counts_of_appts = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_appts,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index().drop(columns=['FirstAttendance*'])

    delta_appts = find_mean_difference_in_appts_relative_to_comparison(counts_of_appts, comparison='Everything')

    fig, ax = plt.subplots()
    name_of_plot = f'Additional Appointments With Intervention, {target_period()}'
    (
        delta_appts / 1e6
    ).T.plot.bar(
        stacked=True, legend=True, ax=ax
    )
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    ax.set_ylabel('(/1e6)')
    ax.set_xlabel('TREATMENT_ID (Short)')
    ax.axhline(0, color='grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(ncol=3, fontsize=5, loc='upper left')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    # VERSION WITH COARSE APPOINTMENTS, CONFORMING TO STANDARD ORDERING/COLORS AND ORDER
    fig, ax = plt.subplots()
    name_of_plot = f'Additional Appointments [Coarse] With Intervention, {target_period()}'
    delta_appts_coarse = delta_appts \
        .groupby(axis=0, by=delta_appts.index.map(get_coarse_appt_type)) \
        .sum() \
        .sort_index(key=order_of_coarse_appt)
    delta_appts_coarse = delta_appts_coarse[order_of_short_treatment_ids(delta_appts_coarse.columns)]
    (
        delta_appts_coarse / 1e6
    ).T.plot.bar(
        stacked=True, legend=True, ax=ax, color=[get_color_coarse_appt(_a) for _a in delta_appts_coarse.index]
    )
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    ax.set_ylabel('(/1e6)')
    ax.set_xlabel('TREATMENT_ID (Short)')
    ax.axhline(0, color='grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(ncol=2, fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()


if __name__ == "__main__":
    rfp = Path('resources')

    parser = argparse.ArgumentParser(
        description="Produce plots to show the impact each set of treatments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Directory to write outputs to. If not specified (set to None) outputs "
            "will be written to value of --results-path argument."
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Directory containing results from running src/scripts/healthsystem/"
            "finding_effects_of_each_treatment/scenario_effect_of_each_treatment.py "
            "script. If not specified (set to None) the last (sorting in alphabetical "
            "order) directory matching either of the glob patterns outputs/"
            "*effect_of_each_treatment* and outputs/*/*effect_of_each_treatment* will "
            "be used if any, or an error raised if there are no matches."
        ),
        default=None,
        required=False
    )
    args = parser.parse_args()
    if args.results_path is None:
        candidate_paths = glob.glob(
            str(Path("outputs") / "*effect_of_each_treatment*"), recursive=True
        )
        candidate_paths += glob.glob(
            str(Path("outputs") / "*" / "*effect_of_each_treatment*"), recursive=True
        )
        if len(candidate_paths) == 0:
            raise FileNotFoundError(
                "Could not find any directories matching pattern outputs/[*/]"
                "*effect_of_each_treatment* to use as results path, directory "
                "to use should be specified explicitly using --results-path argument."
            )
        else:
            results_path = Path(sorted(candidate_paths)[-1])
    else:
        results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path
    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path
    )

    # Plot the legends
    plot_legends.apply(
        results_folder=None, output_folder=results_path, resourcefilepath=rfp)

    # Plot the organisation chart of the TREATMENT_IDs
    plot_org_chart_treatment_ids.apply(
        results_folder=None, output_folder=results_path, resourcefilepath=None)
