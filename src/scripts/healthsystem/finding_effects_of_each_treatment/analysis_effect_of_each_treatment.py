"""Produce plots to show the impact each set of treatments."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, make_age_grp_lookup, make_age_grp_types, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2010, 1, 1), Date(2019, 12, 31))

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
        - Remove prefix of No
        - Remove suffix of *
        """

        if _sn == "Everything":
            return _sn

        elif _sn == "Nothing":
            return "All"
            # In the scenario called "Nothing", all interventions are off, so the difference relative to "Everything"
            # reflects the effects of all the interventions.

        else:
            return _sn.lstrip("No ").rstrip("*")

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])

        reformatted_names = map(format_scenario_name, names_of_cols_level0)
        _df.columns = _df.columns.set_levels(reformatted_names, level=0)
        return _df

    def get_colors(x):
        cmap = plt.cm.get_cmap('jet')
        return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]

    def find_difference_extra_relative_to_comparison(_ser: pd.Series, comparison: str, scaled=False):
        """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1). Drop the comparison entries. The comparison is made: DIFF(X) = X - COMPARISON. """
        return _ser \
            .unstack() \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=0) \
            .drop(index=[comparison]) \
            .stack()

    def find_mean_difference_in_appts_relative_to_comparison(_df: pd.DataFrame, comparison: str):
        """Find the mean difference in the number of appointments between each draw and the comparison draw (within each
        run). We are looking for the number FEWER appointments that occur when treatment does not happen, so we flip the
         sign (as `find_extra_difference_relative_to_comparison` gives the number extra relative the comparison)."""
        return - summarize(pd.concat({
            _idx: find_difference_extra_relative_to_comparison(row, comparison=comparison)
            for _idx, row in _df.iterrows()
        }, axis=1).T, only_mean=True)

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # %% Quantify the health associated with each TREATMENT_ID (short) (The difference in deaths and DALYS between each
    # scenario and the 'Everything' scenario.)

    def num_deaths_by_age_group(_df):
        """Return total number of deaths (total by age-group within the TARGET_PERIOD)"""
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['age'].map(age_grp_lookup).astype(make_age_grp_types())) \
            .size()

    def num_dalys_by_cause(_df):
        """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
        return _df \
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
            .drop(columns=['date', 'sex', 'age_range', 'year']) \
            .sum()

    def do_barh_plot_with_ci(_df, _ax):
        """Make a horizontal bar plot of the _df onto axis _ax"""
        errors = pd.concat([
            _df['mean'] - _df['lower'],
            _df['upper'] - _df['mean']
        ], axis=1).T.to_numpy()
        _df.plot.barh(ax=_ax, y='mean', xerr=errors, legend=False)

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
        custom_generate_series=num_deaths_by_age_group,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across age-groups)

    num_deaths_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Everything')).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True).drop(['All', 'FirstAttendance'])

    pc_deaths_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Everything', scaled=True)).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True).drop(['All', 'FirstAttendance'])

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across causes)

    num_dalys_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Everything')).T
    ).iloc[0].unstack().drop(['All', 'FirstAttendance']).sort_values(by='mean', ascending=True)

    pc_dalys_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Everything', scaled=True)).T
    ).iloc[0].unstack().drop(['All', 'FirstAttendance']).sort_values(by='mean', ascending=True)

    fig, ax = plt.subplots()
    name_of_plot = f'Deaths Averted by Each TREATMENT_ID, {target_period()}'
    do_barh_plot_with_ci(num_deaths_averted / 1e3, ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Number of Deaths Averted (/1000)')
    ax.set_xlim(0, 140)
    do_label_barh_plot(pc_deaths_averted, ax)
    ax.grid()
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    fig, ax = plt.subplots()
    name_of_plot = f'DALYS Averted by Each TREATMENT_ID, {target_period()}'
    do_barh_plot_with_ci(num_dalys_averted / 1e6, ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Number of DALYS Averted (1/1e6)')
    ax.set_xlim(0, 6)
    do_label_barh_plot(pc_dalys_averted, ax)
    ax.grid()
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    # %% Quantify the healthcare system resources used with each TREATMENT_ID (short) (The difference in the number of
    # appointments between each scenario and the 'Everything' scenario.)

    # 1) Examine the HSI that are occurring by TREATMENT_ID

    def get_counts_of_hsi_by_treatment_id(_df):
        """Get the counts of the TREATMENT_IDs occurring."""
        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)

    counts_of_hsi_by_treatment_id = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsi_by_treatment_id,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index().drop(columns=['All', 'FirstAttendance'])

    mean_appts = summarize(counts_of_hsi_by_treatment_id, only_mean=True)

    for scenario_name, _counts in mean_appts.T.iterrows():
        _counts_non_zero = _counts[_counts > 0]

        if len(_counts_non_zero):
            fig, ax = plt.subplots()
            name_of_plot = f'HSI Events Occurring: {scenario_name}, {target_period()}'
            squarify.plot(
                sizes=_counts_non_zero.values,
                label=_counts_non_zero.index,
                color=get_colors(_counts_non_zero.values),
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
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index().drop(columns=['All', 'FirstAttendance'])

    delta_appts = find_mean_difference_in_appts_relative_to_comparison(counts_of_appts, comparison='Everything')

    fig, ax = plt.subplots()
    name_of_plot = f'Additional Appointments With Intervention, {target_period()}'
    (delta_appts / 1e6).T.plot.bar(stacked=True, legend=True, ax=ax)
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
    # todo - this plot would benefit from colormap for appointment types


if __name__ == "__main__":
    # Declare usual paths:
    outputspath = Path('./outputs/tbh03@ic.ac.uk')
    rfp = Path('./resources')

    # Find results folder (the results should have arisne from running `scenario_effect_of_each_treatment.py`.)

    # Most Recent:
    # results_folder = get_scenario_outputs('scenario_effect_of_each_treatment.py', outputspath)[-1]

    # TREATMENT_IDs split by module; consumables not always available
    results_folder = Path('outputs/tbh03@ic.ac.uk/scenario_effect_of_each_treatment-2022-06-13T181214Z')

    # TREATMENT_IDs split by module: consumables always available and healthsystem in mode 0
    # when it's done, it will be: scenario_effect_of_each_treatment-2022-06-14T133746Z

    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
