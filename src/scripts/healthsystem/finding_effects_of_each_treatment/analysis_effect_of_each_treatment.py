"""Produce plots to show the impact of removing each set of Treatments from the healthcare system"""

from pathlib import Path

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):


    # Definitions of general helper functions

    make_graph_file_name = lambda stub: output_folder / f"{stub}.png"  # noqa: E731

    TARGET_PERIOD = (Date(2010, 1, 1), Date(2019, 12, 31))

    _, age_grp_lookup = make_age_grp_lookup()


    def get_parameter_names_from_scenario_file() -> tuple:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.finding_effects_of_each_treatment.scenario_effect_of_each_treatment import (
            EffectOfEachTreatment,
        )
        e = EffectOfEachTreatment()
        return tuple(e._scenarios.keys())


    def drop_outside_period(_df):
        """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
        return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df


    def get_colors(x):
        cmap = plt.cm.get_cmap('jet')
        return [cmap(i) for i in np.arange(0, 1, 1.0 / len(x))]


    def find_difference_relative_to_comparison(_ser: pd.Series, comparison: str, scaled=False):
        """Find the percentage difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1). Drop the comparison entries."""
        return _ser \
            .unstack() \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=0) \
            .drop(index=[comparison]) \
            .stack()
        # todo - is this wrong way around?!?!?!!? Better commenting needed about how we're estimating "deaths averted"


    def find_mean_difference_in_appts_relative_to_comparison(_df: pd.DataFrame, comparison: str):
        """Find the mean difference in the number of appointments between each draw and the comparison draw (within each
        run)."""
        return pd.concat({
            _idx: row.unstack().apply(lambda x: (x[comparison] - x), axis=0).mean(axis=1)
            for _idx, row in _df.iterrows()
        }, axis=1).T.drop(columns=comparison)


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


    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=num_deaths_by_age_group,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across age-groups)
    # todo - UNCERTAINTY
    # todo - labelling - strip "No's" and replace "Nothing" --> "Everything"
    # todo - version for numbers and version for percent

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).sum()  # (Summing across causes)

    fig, ax = plt.subplots()
    name_of_plot = 'Deaths Averted by Each TREATMENT_ID (Short)'
    pc_deaths_averted = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison(num_deaths, comparison='Everything')).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)
    pc_deaths_averted['mean'].plot.barh(ax=ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Percent of Deaths Averted')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()

    fig, ax = plt.subplots()
    name_of_plot = 'DALYS Averted by Each TREATMENT_ID (Short)'
    pc_dalys_averted = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison(num_dalys, comparison='Everything')).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)
    pc_dalys_averted['mean'].plot.barh(ax=ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('TREATMENT_ID (Short)')
    ax.set_xlabel('Percent of DALYS Averted')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()


    # %% Quantify the healthcare system resources used with each TREATMENT_ID (short) (The difference in the number of
    # appointments between each scenario and the 'Everything' scenario.)

    def get_counts_of_hsi_by_treatment_id(_df):
        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)


    def find_mean_appts(_df: pd.DataFrame):
        """Find the mean difference in the number of appointments between each draw and the comparison draw (within each
        run)."""
        return pd.concat({
            _idx: row.unstack().mean(axis=1)
            for _idx, row in _df.iterrows()
        }, axis=1).T


    def get_counts_of_appts(_df):
        """Get the counts of appointments of each type being used."""
        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)


    counts_of_hsi_by_treatment_id = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_hsi_by_treatment_id,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index()

    mean_appts = find_mean_appts(counts_of_hsi_by_treatment_id)

    # 1) Examine the HSI that are occurring by TREATMENT_ID
    for scenario_name, _counts in mean_appts.T.iterrows():
        _counts_non_zero = _counts[_counts > 0]

        fig, ax = plt.subplots()
        name_of_plot = f'HSI Events Occurring: {scenario_name}'
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

    # Look at differences in terms of TREATMENT_IDS
    delta = find_mean_difference_in_appts_relative_to_comparison(counts_of_hsi_by_treatment_id, comparison='Everything')
    big_difference = delta.apply(lambda x: np.isclose(1.0, x))

    # 2) Examine the Difference in the number/type of appointments occurring

    counts_of_appts = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event',
        custom_generate_series=get_counts_of_appts,
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0).fillna(0.0).sort_index()

    delta_appts = find_mean_difference_in_appts_relative_to_comparison(counts_of_appts, comparison='Everything')

    for scenario_name, _appt_counts in delta_appts.T.iterrows():
        fig, ax = plt.subplots()
        name_of_plot = f'Difference in Appointments: {scenario_name}*'
        _appt_counts.plot.bar(ax=ax)
        ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        fig.show()


if __name__ == "__main__":

    # Declare usual paths:
    outputspath = Path('./outputs/tbh03@ic.ac.uk')
    rfp = Path('./resources')

    # Declare the name of the file that specified the scenarios used in this run.
    scenario_filename = 'scenario_effect_of_each_treatment.py'

    # Find results folder (most recent run generated using that scenario_filename)
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
