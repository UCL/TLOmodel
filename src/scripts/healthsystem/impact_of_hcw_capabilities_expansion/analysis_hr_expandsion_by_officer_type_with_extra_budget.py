"""
This file analyses and plots the services, DALYs, Deaths within different scenarios of expanding current hr by officer
type given some extra budget. Return on investment and marginal productivity of each officer type will be examined.

The scenarios are defined in scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget.py.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.healthsystem.impact_of_hcw_capabilities_expansion.prepare_minute_salary_and_extra_budget_frac_data import (
    extra_budget_fracs, Minute_Salary_by_Cadre_Level,
)

from scripts.healthsystem.impact_of_hcw_capabilities_expansion.scenario_of_expanding_current_hcw_by_officer_type_with_extra_budget import (
    HRHExpansionByCadreWithExtraBudget,
)
import statsmodels.api as sm
import statsmodels.stats as ss
from tlo import Date
from tlo.analysis.utils import (
    APPT_TYPE_TO_COARSE_APPT_TYPE_MAP,
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
    extract_results,
    summarize,
)

# rename scenarios
substitute_labels = {
    's_1': 'no_expansion',
    's_2': 'all_cadres_current_allocation',
    's_3': 'all_cadres_equal_allocation',
    's_4': 'Clinical (C)', 's_5': 'DCSA (D)', 's_6': 'Nursing_and_Midwifery (N&M)', 's_7': 'Pharmacy (P)',
    's_8': 'Other (O)',
    's_9': 'C + D', 's_10': 'C + N&M', 's_11': 'C + P', 's_12': 'C + O', 's_13': 'D + N&M',
    's_14': 'D + P', 's_15': 'D + O', 's_16': 'N&M + P', 's_17': 'N&M + O', 's_18': 'P + O',
    's_19': 'C + D + N&M', 's_20': 'C + D + P', 's_21': 'C + D + O', 's_22': 'C + N&M + P', 's_23': 'C + N&M + O',
    's_24': 'C + P + O', 's_25': 'D + N&M + P', 's_26': 'D + N&M + O', 's_27': 'D + P + O', 's_28': 'N&M + P + O',
    's_29': 'C + D + N&M + P', 's_30': 'C + D + N&M + O', 's_31': 'C + D + P + O', 's_32': 'C + N&M + P + O',
    's_33': 'D + N&M + P + O',
}

# group scenarios for presentation
scenario_groups_init = {
    'no_expansion': {'s_1'},
    'all_cadres_expansion': {'s_2', 's_3'},
    'one_cadre_expansion': {'s_4', 's_5', 's_6', 's_7', 's_8'},
    'two_cadres_expansion': {'s_9', 's_10', 's_11', 's_12', 's_13',
                             's_14', 's_15', 's_16', 's_17', 's_18'},
    'three_cadres_expansion': {'s_19', 's_20', 's_21', 's_22', 's_23',
                               's_24', 's_25', 's_26', 's_27', 's_28'},
    'four_cadres_expansion': {'s_29', 's_30', 's_31', 's_32', 's_33'}
}

# group scenarios based on whether expand Clinical/Pharmacy
scenario_groups = {
    'C + P + D/N&M/O/None': {'s_2', 's_3', 's_11', 's_20', 's_22', 's_24', 's_29', 's_31', 's_32'},
    'C + D/N&M/O/None': {'s_4', 's_9', 's_10', 's_12', 's_19', 's_21', 's_23', 's_30'},
    'P + D/N&M/O/None': {'s_7', 's_14', 's_16', 's_18', 's_25', 's_27', 's_28', 's_33'},
    'D/N&M/O/None': {'s_5', 's_6', 's_8', 's_13', 's_15', 's_17', 's_26', 's_1'}
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
        """Return the number of services by appt type (total within the TARGET_PERIOD)"""
        return (_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code']
                .apply(pd.Series)
                .rename(columns=APPT_TYPE_TO_COARSE_APPT_TYPE_MAP)
                .groupby(level=0, axis=1).sum()
                .sum())

    def get_num_appts_by_level(_df):
        """Return the number of services by appt type and facility level (total within the TARGET_PERIOD)"""
        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.items()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .sum(axis=0)

    def get_num_services(_df):
        """Return the number of services in total of all appt types (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code']
            .apply(pd.Series).sum().sum()
        )

    def get_num_treatments(_df):
        """Return the number of treatments by short treatment id (total within the TARGET_PERIOD)"""
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
        _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
        _df = _df.groupby(level=0).sum()
        return _df

    def get_num_treatments_total(_df):
        """Return the number of treatments in total of all treatments (total within the TARGET_PERIOD)"""
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
        _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
        _df = _df.groupby(level=0).sum().sum()
        return pd.Series(_df)

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)"""
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def get_num_dalys(_df):
        """Return total number of DALYS (Stacked) (total within the TARGET_PERIOD).
        Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
        results from runs that crashed mid-way through the simulation).
        """
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return pd.Series(
            data=_df
            .loc[_df.year.between(*years_needed)]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )

    def get_num_dalys_yearly(_df):
        """Return total number of DALYS (Stacked) for every year in the TARGET_PERIOD.
        Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
        results from runs that crashed mid-way through the simulation).
        """
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        _df = (_df.loc[_df.year.between(*years_needed)]
               .drop(columns=['date', 'sex', 'age_range'])
               .groupby('year').sum()
               .sum(axis=1)
               )
        return _df

    def get_num_dalys_by_cause(_df):
        """Return total number of DALYS by cause (Stacked) (total within the TARGET_PERIOD).
        Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
        results from runs that crashed mid-way through the simulation).
        """
        years_needed = [i.year for i in TARGET_PERIOD]
        assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
        return (_df
                .loc[_df.year.between(*years_needed)].drop(columns=['date', 'year', 'li_wealth'])
                .sum(axis=0)
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
        return (_ser
                .unstack(level=0)
                .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1)
                .drop(columns=([comparison] if drop_comparison else []))
                .stack()
                )

    def find_difference_relative_to_comparison_dataframe(_df: pd.DataFrame, **kwargs):
        """Apply `find_difference_relative_to_comparison_series` to each row in a dataframe"""
        return pd.concat({
            _idx: find_difference_relative_to_comparison_series(row, **kwargs)
            for _idx, row in _df.iterrows()
        }, axis=1).T

    def do_bar_plot_with_ci(_df, _df_percent=None, annotation=False):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar.
         Annotated with percent statistics from _df_percent, if annotation=True and _df_percent not None."""

        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        colors = [scenario_color[s] for s in _df.index]

        fig, ax = plt.subplots(figsize=(18, 6))
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

        if annotation:
            assert (_df.index == _df_percent.index).all()
            for xpos, ypos, text1, text2, text3 in zip(xticks.keys(), _df['upper'].values,
                                                       _df_percent['mean'].values,
                                                       _df_percent['lower'].values,
                                                       _df_percent['upper'].values):
                text = f"{int(round(text1 * 100, 2))}%\n{[round(text2, 2),round(text3, 2)]}"
                ax.text(xpos, ypos * 1.05, text, horizontalalignment='center', fontsize='xx-small')

        ax.set_xticks(list(xticks.keys()))

        xtick_label_detail = [substitute_labels[v] for v in xticks.values()]
        ax.set_xticklabels(xtick_label_detail, rotation=90)

        legend_labels = list(scenario_groups_color.keys())
        legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                        color=scenario_groups_color[label]) for label in legend_labels]
        ax.legend(legend_handles, legend_labels, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))

        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig, ax

    # def get_scale_up_factor(_df):
    #     """
    #     Return a series of yearly scale up factors for all cadres,
    #     with index of year and value of list of scale up factors.
    #     """
    #     _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), ['year_of_scale_up', 'scale_up_factor']
    #                   ].set_index('year_of_scale_up')
    #     _df = _df['scale_up_factor'].apply(pd.Series)
    #     assert (_df.columns == cadres).all()
    #     _dict = {idx: [list(_df.loc[idx, :])] for idx in _df.index}
    #     _df_1 = pd.DataFrame(data=_dict).T
    #     return pd.Series(
    #         _df_1.loc[:, 0], index=_df_1.index
    #     )

    def get_total_cost(_df):
        """
        Return a series of yearly total cost for all cadres,
        with index of year and values of list of total cost.
        """
        _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), ['year_of_scale_up', 'total_hr_salary']
                      ].set_index('year_of_scale_up')
        _df = _df['total_hr_salary'].apply(pd.Series)
        assert (_df.columns == cadres).all()
        _dict = {idx: [list(_df.loc[idx, :])] for idx in _df.index}
        _df_1 = pd.DataFrame(data=_dict).T
        return pd.Series(
            _df_1.loc[:, 0], index=_df_1.index
        )

    def get_current_hr(cadres):
        """
        Return current (year of 2018/2019) staff counts and capabilities for the cadres specified.
        """
        curr_hr_path = Path(resourcefilepath
                            / 'healthsystem' / 'human_resources' / 'actual' / 'ResourceFile_Daily_Capabilities.csv')
        curr_hr = pd.read_csv(curr_hr_path).groupby('Officer_Category').agg(
            {'Staff_Count': 'sum', 'Total_Mins_Per_Day': 'sum'}).reset_index()
        curr_hr['Total_Minutes_Per_Year'] = curr_hr['Total_Mins_Per_Day'] * 365.25
        curr_hr.drop(['Total_Mins_Per_Day'], axis=1, inplace=True)
        curr_hr = curr_hr.loc[
            curr_hr['Officer_Category'].isin(cadres), ['Officer_Category', 'Staff_Count']
        ].set_index('Officer_Category').T
        return curr_hr[cadres]

    def get_hr_salary(cadres):
        """
        Return annual salary for the cadres specified.
        """
        salary_path = Path(resourcefilepath
                           / 'costing' / 'ResourceFile_Annual_Salary_Per_Cadre.csv')
        salary = pd.read_csv(salary_path, index_col=False)
        salary = salary.loc[
            salary['Officer_Category'].isin(cadres), ['Officer_Category', 'Annual_Salary_USD']
        ].set_index('Officer_Category').T
        return salary[cadres]

    def format_appt_time_and_cost():
        """
        Return the formatted appointment time requirements and costs per cadre
        """
        file_path = Path(resourcefilepath
                         / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Time_Table.csv')
        _df = pd.read_csv(file_path, index_col=False)

        time = _df.pivot(index=['Facility_Level', 'Appt_Type_Code'], columns='Officer_Category',
                         values='Time_Taken_Mins').fillna(0.0).T
        minute_salary = Minute_Salary_by_Cadre_Level
        cost = _df.merge(minute_salary, on=['Facility_Level', 'Officer_Category'], how='left')
        cost['cost_USD'] = cost['Time_Taken_Mins'] * cost['Minute_Salary_USD']
        cost = cost.pivot(index=['Facility_Level', 'Appt_Type_Code'], columns='Officer_Category',
                          values='cost_USD').fillna(0.0).T

        return time, cost

    # def get_hcw_time_usage(_df):
    #     """Return the number of treatments by short treatment id (total within the TARGET_PERIOD)"""
    #     _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
    #     _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
    #     _df = _df.groupby(level=0).sum()
    #     return _df

    # Get parameter/scenario names
    param_names = get_parameter_names_from_scenario_file()

    # Define cadres in order
    cadres = ['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy',
              'Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography']

    # Get appointment time and cost requirement
    appt_time, appt_cost = format_appt_time_and_cost()

    # # Get current (year of 2018/2019) hr counts
    # curr_hr = get_current_hr(cadres)

    # # Get scale up factors for all scenarios
    # scale_up_factors = extract_results(
    #     results_folder,
    #     module='tlo.methods.healthsystem.summary',
    #     key='HRScaling',
    #     custom_generate_series=get_scale_up_factor,
    #     do_scaling=False
    # ).pipe(set_param_names_as_column_index_level_0).stack(level=0)
    # # check that the scale up factors are all most the same between each run within each draw
    # # assert scale_up_factors.eq(scale_up_factors.iloc[:, 0], axis=0).all().all()
    # # keep scale up factors of only one run within each draw
    # scale_up_factors = scale_up_factors.iloc[:, 0].unstack().reset_index().melt(id_vars='index')
    # scale_up_factors[cadres] = scale_up_factors.value.tolist()
    # scale_up_factors.drop(columns='value', inplace=True)

    # Get salary
    salary = get_hr_salary(cadres)

    # Get total cost for all scenarios
    total_cost = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HRScaling',
        custom_generate_series=get_total_cost,
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0).stack(level=0)
    total_cost = total_cost.iloc[:, 0].unstack().reset_index().melt(id_vars='index')
    total_cost[cadres] = total_cost.value.tolist()
    total_cost.drop(columns='value', inplace=True)
    total_cost['all_cadres'] = total_cost[[c for c in total_cost.columns if c in cadres]].sum(axis=1)
    total_cost.rename(columns={'index': 'year'}, inplace=True)

    # total cost of all expansion years
    total_cost_all_yrs = total_cost.groupby('draw').sum().drop(columns='year')

    # total extra cost of all expansion years
    extra_cost_all_yrs = total_cost_all_yrs.copy()
    for s in param_names[1:]:
        extra_cost_all_yrs.loc[s, :] = total_cost_all_yrs.loc[s, :] - total_cost_all_yrs.loc['s_1', :]
    extra_cost_all_yrs.drop(index='s_1', inplace=True)

    # get staff count = total cost / salary
    staff_count = total_cost.copy()
    for c in cadres:
        staff_count.loc[:, c] = total_cost.loc[:, c] / salary[c].values[0]
    staff_count.loc[:, 'all_cadres'] = staff_count[[c for c in staff_count.columns if c in cadres]].sum(axis=1)

    # get extra count = staff count - staff count of no expansion s_1
    # note that annual staff increase rate = scale up factor - 1
    extra_staff = staff_count.copy()
    for i in staff_count.index:
        extra_staff.iloc[i, 2:] = staff_count.iloc[i, 2:] - staff_count.iloc[0, 2:]

    extra_staff_2029 = extra_staff.loc[extra_staff.year == 2029, :].drop(columns='year').set_index('draw').drop(
        index='s_1'
    )
    staff_count_2029 = staff_count.loc[staff_count.year == 2029, :].drop(columns='year').set_index('draw')

    # check total cost calculated is increased as expected
    years = range(2019, the_target_period[1].year + 1)
    for s in param_names[1:]:
        assert (abs(
            total_cost.loc[(total_cost.year == 2029) & (total_cost.draw == s), 'all_cadres'].values[0] -
            (1 + 0.042) ** len(years) * total_cost.loc[(total_cost.year == 2019) & (total_cost.draw == 's_1'),
                                                       'all_cadres'].values[0]
        ) < 1e6).all()

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

    num_dalys_yearly = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_yearly,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys_by_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_wealth_stacked_by_age_and_time",
        custom_generate_series=get_num_dalys_by_cause,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)

    num_appts = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_appts,
        do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)

    num_services = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_services,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_treatments = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_treatments_total = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments_total,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_never_ran_appts = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Never_ran_HSI_Event',
        custom_generate_series=get_num_appts,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_never_ran_appts_by_level = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Never_ran_HSI_Event',
        custom_generate_series=get_num_appts_by_level,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_never_ran_services = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Never_ran_HSI_Event',
        custom_generate_series=get_num_services,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_never_ran_treatments_total = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Never_ran_HSI_Event',
        custom_generate_series=get_num_treatments_total,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_never_ran_treatments = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Never_ran_HSI_Event',
        custom_generate_series=get_num_treatments,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    # get total service demand
    assert len(num_services) == len(num_never_ran_services) == 1
    assert (num_services.columns == num_never_ran_services.columns).all()
    num_services_demand = num_services + num_never_ran_services
    ratio_services = num_services / num_services_demand

    assert (num_appts.columns == num_never_ran_appts.columns).all()
    num_never_ran_appts.loc['Lab / Diagnostics', :] = 0
    num_never_ran_appts = num_never_ran_appts.reindex(num_appts.index).fillna(0.0)
    assert (num_appts.index == num_never_ran_appts.index).all()
    num_appts_demand = num_appts + num_never_ran_appts

    # hcw_time_usage = extract_results(
    #     results_folder,
    #     module='tlo.methods.healthsystem.summary',
    #     key='Capacity',#'Capacity_By_OfficerType_And_FacilityLevel',
    #     custom_generate_series=get_hcw_time_usage,
    #     do_scaling=False
    # ).pipe(set_param_names_as_column_index_level_0)

    # get absolute numbers for scenarios
    # sort the scenarios according to their DALYs values, in ascending order
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names).sort_values(by='mean')
    num_dalys_by_cause_summarized = summarize(num_dalys_by_cause, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )

    num_dalys_yearly_summarized = (summarize(num_dalys_yearly)
                                   .stack([0, 1])
                                   .rename_axis(['year', 'scenario', 'stat'])
                                   .reset_index(name='count'))

    num_deaths_summarized = summarize(num_deaths).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )

    num_services_summarized = summarize(num_services).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_appts_summarized = summarize(num_appts, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_never_ran_appts_by_level_summarized = summarize(num_never_ran_appts_by_level, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    ).fillna(0.0)
    num_appts_demand_summarized = summarize(num_appts_demand, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_treatments_summarized = summarize(num_treatments, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_treatments_total_summarized = summarize(num_treatments_total).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )

    num_never_ran_services_summarized = summarize(num_never_ran_services).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_never_ran_appts_summarized = summarize(num_never_ran_appts, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_never_ran_treatments_summarized = summarize(num_never_ran_treatments, only_mean=True).T.reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_never_ran_treatments_total_summarized = summarize(num_never_ran_treatments_total).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    num_service_demand_summarized = summarize(num_services_demand).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )
    ratio_service_summarized = summarize(ratio_services).loc[0].unstack().reindex(param_names).reindex(
        num_dalys_summarized.index
    )

    # get relative numbers for scenarios, compared to no_expansion scenario: s_1
    num_services_increased = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_services.loc[0],
                comparison='s_1')
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_services_increased_percent = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_services.loc[0],
                comparison='s_1',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='s_1')
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_deaths_averted_percent = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_deaths.loc[0],
                comparison='s_1',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='s_1')
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_dalys_averted_percent = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_dalys.loc[0],
                comparison='s_1',
                scaled=True
            )
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_dalys_by_cause_averted = summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            num_dalys_by_cause,
            comparison='s_1',
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_dalys_by_cause_averted_percent = summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            num_dalys_by_cause,
            comparison='s_1',
            scaled=True
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_appts_increased = summarize(
        find_difference_relative_to_comparison_dataframe(
            num_appts,
            comparison='s_1',
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_never_ran_appts_reduced = summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            num_never_ran_appts,
            comparison='s_1',
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_never_ran_treatments_reduced = summarize(
        -1.0 * find_difference_relative_to_comparison_dataframe(
            num_never_ran_treatments,
            comparison='s_1',
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_appts_increased_percent = summarize(
        find_difference_relative_to_comparison_dataframe(
            num_appts,
            comparison='s_1',
            scaled=True
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_treatments_increased = summarize(
        find_difference_relative_to_comparison_dataframe(
            num_treatments,
            comparison='s_1',
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_treatments_increased_percent = summarize(
        find_difference_relative_to_comparison_dataframe(
            num_treatments,
            comparison='s_1',
            scaled=True
        ),
        only_mean=True
    ).T.reindex(num_dalys_summarized.index).drop(['s_1'])

    num_treatments_total_increased = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_treatments_total.loc[0],
                comparison='s_1')
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    num_treatments_total_increased_percent = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                num_treatments_total.loc[0],
                comparison='s_1',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    service_ratio_increased = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                ratio_services.loc[0],
                comparison='s_1')
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    service_ratio_increased_percent = summarize(
        pd.DataFrame(
            find_difference_relative_to_comparison_series(
                ratio_services.loc[0],
                comparison='s_1',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).reindex(num_dalys_summarized.index).drop(['s_1'])

    # Check that when we sum across the causes/appt types,
    # we get the same total as calculated when we didn't split by cause/appt type.
    assert (
        (num_appts_increased.sum(axis=1).sort_index()
         - num_services_increased['mean'].sort_index()
         ) < 1e-6
    ).all()

    assert (
        (num_dalys_by_cause_averted.sum(axis=1).sort_index()
         - num_dalys_averted['mean'].sort_index()
         ) < 1e-6
    ).all()

    assert (
        (num_treatments_increased.sum(axis=1).sort_index()
         - num_treatments_total_increased['mean'].sort_index()
         ) < 1e-6
    ).all()

    # get HCW time and cost needed to run the never run appts
    def hcw_time_or_cost_gap(_df=appt_time):
        cols_1 = num_never_ran_appts_by_level_summarized.columns
        cols_2 = _df.columns
        # check that never ran appts (at a level) not in appt_time (as defined) have count 0 and drop them
        assert (num_never_ran_appts_by_level_summarized[list(set(cols_1) - set(cols_2))] == 0).all().all()
        num_never_ran_appts_by_level_summarized.drop(columns=list(set(cols_1) - set(cols_2)), inplace=True)
        assert set(num_never_ran_appts_by_level_summarized.columns).issubset(set(cols_2))
        # calculate hcw time gap
        gap = pd.DataFrame(index=num_never_ran_appts_by_level_summarized.index,
                           columns=_df.index)
        for i in gap.index:
            for j in gap.columns:
                gap.loc[i, j] = num_never_ran_appts_by_level_summarized.loc[i, :].mul(
                    _df.loc[j, num_never_ran_appts_by_level_summarized.columns]
                ).sum()
        # reorder columns to be consistent with cadres
        gap = gap[['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy',
                   'Dental', 'Laboratory', 'Mental', 'Radiography']]

        return gap

    hcw_time_gap = hcw_time_or_cost_gap(appt_time)
    hcw_cost_gap = hcw_time_or_cost_gap(appt_cost)

    # cost gap proportions of cadres within each scenario
    hcw_cost_gap_percent = pd.DataFrame(index=hcw_cost_gap.index, columns=hcw_cost_gap.columns)
    for i in hcw_cost_gap_percent.index:
        hcw_cost_gap_percent.loc[i, :] = hcw_cost_gap.loc[i, :] / hcw_cost_gap.loc[i, :].sum()
    # add a column of 'other' to sum up other cadres
    hcw_cost_gap_percent['Other'] = hcw_cost_gap_percent[
        ['Dental', 'Laboratory', 'Mental', 'Radiography']
    ].sum(axis=1)

    # find appts that need Clinical + Pharmacy
    # then calculate hcw time needed for these appts (or treatments, need treatment and their appt footprint) in never run set
    # also consider plot service demand by appt of all scenarios to see if they are similar
    # so we can explain that expand C+P is reducing the never run appts and bring health benefits across scenarios
    # then the next question is what proportion for C and P?
    appts_need_C_P = []
    for col in appt_time.columns:
        if (appt_time.loc[['Clinical', 'Pharmacy'], col] > 0).all():
            appts_need_C_P.append(col)

    # get Return (in terms of DALYs averted) On Investment (extra cost) for all expansion scenarios, excluding s_1
    # get Cost-Effectiveness, i.e., cost of every daly averted, for all expansion scenarios
    ROI = pd.DataFrame(index=num_dalys_averted.index, columns=num_dalys_averted.columns)
    # todo: for the bad scenarios (s_5, s_8, s_15), the dalys averted are negative
    #  (maybe only due to statistical variation; relative difference to s_1 are close to 0%),
    #  thus CE does not make sense.
    # CE = pd.DataFrame(index=num_dalys_averted.index, columns=num_dalys_averted.columns)
    for i in ROI.index:
        ROI.loc[i, :] = num_dalys_averted.loc[i, :] / extra_cost_all_yrs.loc[i, 'all_cadres']
    #     CE.loc[i, 'mean'] = extra_cost_all_yrs.loc[i, 'all_cadres'] / num_dalys_averted.loc[i, 'mean']
    #     CE.loc[i, 'lower'] = extra_cost_all_yrs.loc[i, 'all_cadres'] / num_dalys_averted.loc[i, 'upper']
    #     CE.loc[i, 'upper'] = extra_cost_all_yrs.loc[i, 'all_cadres'] / num_dalys_averted.loc[i, 'lower']

    # prepare colors for plots
    appt_color = {
        appt: COARSE_APPT_TYPE_TO_COLOR_MAP.get(appt, np.nan) for appt in num_appts_summarized.columns
    }
    treatment_color = {
        treatment: SHORT_TREATMENT_ID_TO_COLOR_MAP.get(treatment, np.nan)
        for treatment in num_treatments_summarized.columns
    }
    cause_color = {
        cause: CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.get(cause, np.nan)
        for cause in num_dalys_by_cause_summarized.columns
    }
    officer_category_color = {
        'Clinical': 'blue',
        'DCSA': 'orange',
        'Nursing_and_Midwifery': 'red',
        'Pharmacy': 'green',
        'Dental': 'purple',
        'Laboratory': 'orchid',
        'Mental': 'plum',
        'Nutrition': 'thistle',
        'Radiography': 'lightgray',
        'Other': 'gray'
    }
    scenario_groups_color_init = {
        'no_expansion': 'gray',
        'one_cadre_expansion': 'lightpink',
        'two_cadres_expansion': 'violet',
        'three_cadres_expansion': 'darkorchid',
        'four_cadres_expansion': 'paleturquoise',
        'all_cadres_expansion': 'darkturquoise'
    }
    scenario_groups_color = {
        'D/N&M/O/None': 'lightpink',
        'P + D/N&M/O/None': 'violet',
        'C + D/N&M/O/None': 'darkorchid',
        'C + P + D/N&M/O/None': 'darkturquoise',
    }

    scenario_color = {}
    for s in param_names:
        for k in scenario_groups_color.keys():
            if s in scenario_groups[k]:
                scenario_color[s] = scenario_groups_color[k]

    best_scenarios_color = {'s_1': 'black'}
    cmap_list = list(map(plt.get_cmap("Set1"), range(9)))
    for i in range(9):
        best_scenarios_color[num_dalys_summarized.index[i]] = cmap_list[i]

    # plot 4D data: relative increases of Clinical, Pharmacy, and Nursing_and_Midwifery as three coordinates,\
    # percentage of DALYs averted decides the color of that scatter point
    extra_budget_allocation = extra_budget_fracs.T.reindex(num_dalys_summarized.index)
    extra_budget_allocation['Other'] = extra_budget_allocation[
        ['Dental', 'Laboratory', 'Mental', 'Radiography']
    ].sum(axis=1)
    name_of_plot = f'Dalys averted (%) against no expansion, {target_period()}'
    heat_data = pd.merge(num_dalys_averted_percent['mean'],
                         extra_budget_allocation[['Clinical', 'Pharmacy', 'Nursing_and_Midwifery']],
                         left_index=True, right_index=True, how='inner')
    # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    colors = [scenario_color[s] for s in heat_data.index]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(heat_data['Clinical'], heat_data['Pharmacy'], heat_data['Nursing_and_Midwifery'],
                     alpha=0.8, marker='o', s=heat_data['mean'] * 2000,
                     #c=heat_data['mean'] * 100, cmap='viridis',
                     c=colors)
    # plot lines from the best point to three axes panes
    # ax.plot3D([heat_data['Clinical'][0], heat_data['Clinical'][0]],
    #           [heat_data['Pharmacy'][0], heat_data['Pharmacy'][0]],
    #           [0, heat_data['Nursing_and_Midwifery'][0]],
    #           linestyle='--', color='gray', alpha=0.8)
    # ax.plot3D([heat_data['Clinical'][0], heat_data['Clinical'][0]],
    #           [0, heat_data['Pharmacy'][0]],
    #           [heat_data['Nursing_and_Midwifery'][0], heat_data['Nursing_and_Midwifery'][0]],
    #           linestyle='--', color='gray', alpha=0.8)
    # ax.plot3D([0, heat_data['Clinical'][0]],
    #           [heat_data['Pharmacy'][0], heat_data['Pharmacy'][0]],
    #           [heat_data['Nursing_and_Midwifery'][0], heat_data['Nursing_and_Midwifery'][0]],
    #           linestyle='--', color='gray', alpha=0.8)
    ax.set_xlabel('Fraction of extra budget allocated to \nClinical cadre (C)')
    ax.set_ylabel('Pharmacy cadre (P)')
    #ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_zlabel('Nursing and Midwifery (N&M)')
    ax.plot3D([0, 1], [0, 1], [0, 1], linestyle='-', color='orange', alpha=1.0, linewidth=2)
    legend_labels = list(scenario_groups_color.keys()) + ['line of C = P = N&M']
    legend_handles = [plt.Line2D([0, 0], [0, 0],
                                 linestyle='none', marker='o', color=scenario_groups_color[label]
                                 ) for label in legend_labels[0:len(legend_labels) - 1]
                      ] + [plt.Line2D([0, 1], [0, 0], linestyle='-', color='orange')]
    plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.25)
    plt.title(name_of_plot)
    plt.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'3D Dalys averted, Services increased and Treatment increased, {target_period()}'
    heat_data = pd.concat([num_dalys_averted_percent['mean'], num_services_increased_percent['mean'],
                           num_treatments_total_increased_percent['mean']], axis=1)
    # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    colors = [scenario_color[s] for s in heat_data.index]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(100 * heat_data.iloc[:, 1], 100 * heat_data.iloc[:, 2], 100 * heat_data.iloc[:, 0],
               alpha=0.8, marker='o',
               c=colors)
    ax.set_xlabel('Services increased %')
    ax.set_ylabel('Treatments increased %')
    ax.set_zlabel('DALYs averted %')
    legend_labels = list(scenario_groups_color.keys())
    legend_handles = [plt.Line2D([0, 0], [0, 0],
                                 linestyle='none', marker='o', color=scenario_groups_color[label]
                                 ) for label in legend_labels
                      ]
    plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.title(name_of_plot)
    plt.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'2D Dalys averted, Services increased and Treatment increased, {target_period()}'
    heat_data = pd.concat([num_dalys_averted_percent['mean'], num_services_increased_percent['mean'],
                           num_treatments_total_increased_percent['mean']], axis=1)
    # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    colors = [scenario_color[s] for s in heat_data.index]
    fig, ax = plt.subplots()
    ax.scatter(100 * heat_data.iloc[:, 1], 100 * heat_data.iloc[:, 2],
               alpha=0.8, marker='o', s=2000 * heat_data.iloc[:, 0],
               c=colors)
    ax.set_xlabel('Services increased %')
    ax.set_ylabel('Treatments increased %')
    legend_labels = list(scenario_groups_color.keys())
    legend_handles = [plt.Line2D([0, 0], [0, 0],
                                 linestyle='none', marker='o', color=scenario_groups_color[label]
                                 ) for label in legend_labels
                      ]
    plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.title(name_of_plot)
    plt.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Dalys averted and Services increased, {target_period()}'
    heat_data = pd.concat([num_dalys_averted_percent['mean'], num_services_increased_percent['mean'],
                           num_treatments_total_increased_percent['mean']], axis=1)
    # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    colors = [scenario_color[s] for s in heat_data.index]
    fig, ax = plt.subplots()
    ax.scatter(100 * heat_data.iloc[:, 1], 100 * heat_data.iloc[:, 0],
               alpha=0.8, marker='o', c=colors)
    ax.set_xlabel('Services increased %')
    ax.set_ylabel('DLAYs averted %')
    legend_labels = list(scenario_groups_color.keys())
    legend_handles = [plt.Line2D([0, 0], [0, 0],
                                 linestyle='none', marker='o', color=scenario_groups_color[label]
                                 ) for label in legend_labels
                      ]
    plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.title(name_of_plot)
    plt.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Dalys averted and Treatments increased, {target_period()}'
    # heat_data = pd.concat([num_dalys_averted_percent['mean'], num_services_increased_percent['mean'],
    #                        num_treatments_total_increased_percent['mean']], axis=1)
    # # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    # colors = [scenario_color[s] for s in heat_data.index]
    # fig, ax = plt.subplots()
    # ax.scatter(100 * heat_data.iloc[:, 2], 100 * heat_data.iloc[:, 0],
    #            alpha=0.8, marker='o', c=colors)
    # ax.set_xlabel('Treatments increased %')
    # ax.set_ylabel('DLAYs averted %')
    # legend_labels = list(scenario_groups_color.keys())
    # legend_handles = [plt.Line2D([0, 0], [0, 0],
    #                              linestyle='none', marker='o', color=scenario_groups_color[label]
    #                              ) for label in legend_labels
    #                   ]
    # plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.title(name_of_plot)
    # plt.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Dalys averted and Services ratio increased, {target_period()}'
    # heat_data = pd.concat([num_dalys_averted_percent['mean'], service_ratio_increased_percent['mean']], axis=1)
    # # scenarios_with_CNP_only = ['s_4', 's_6', 's_7', 's_10', 's_11', 's_16', 's_22']
    # # heat_data = heat_data.loc[heat_data.index.isin(scenarios_with_CNP_only)]
    # colors = [scenario_color[s] for s in heat_data.index]
    # fig, ax = plt.subplots()
    # ax.scatter(100 * heat_data.iloc[:, 1], 100 * heat_data.iloc[:, 0],
    #            alpha=0.8, marker='o', c=colors)
    # ax.set_xlabel('Service delivery ratio increased %')
    # ax.set_ylabel('DLAYs averted %')
    # legend_labels = list(scenario_groups_color.keys())
    # legend_handles = [plt.Line2D([0, 0], [0, 0],
    #                              linestyle='none', marker='o', color=scenario_groups_color[label]
    #                              ) for label in legend_labels
    #                   ]
    # plt.legend(legend_handles, legend_labels, loc='upper center', fontsize='small', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.title(name_of_plot)
    # plt.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # do some linear regression to see the marginal effects of individual cadres and combined effects of C, N, P cadres
    outcome_data = num_dalys_averted_percent['mean']
    # outcome = num_services_increased_percent['mean']
    # outcome = num_treatments_total_increased_percent['mean']
    regression_data = pd.merge(outcome_data,
                               extra_budget_allocation,
                               left_index=True, right_index=True, how='inner')
    regression_data['C*P'] = regression_data['Clinical'] * regression_data['Pharmacy']
    regression_data['C*N'] = regression_data['Clinical'] * regression_data['Nursing_and_Midwifery']
    regression_data['N*P'] = regression_data['Pharmacy'] * regression_data['Nursing_and_Midwifery']
    regression_data['C*N*P'] = (regression_data['Clinical'] * regression_data['Pharmacy']
                                * regression_data['Nursing_and_Midwifery'])
    cadres_to_drop_due_to_multicollinearity = ['Dental', 'Laboratory', 'Mental', 'Nutrition', 'Radiography', 'Other']
    regression_data.drop(columns=cadres_to_drop_due_to_multicollinearity, inplace=True)
    predictor = regression_data[regression_data.columns[1:]]
    outcome = regression_data['mean']
    predictor = sm.add_constant(predictor)
    est = sm.OLS(outcome.astype(float), predictor.astype(float)).fit()
    print(est.summary())

    # do anova analysis to test the difference of scenario groups
    def anova_oneway(df=num_dalys_averted_percent):
        best = df.loc[list(scenario_groups['C + P + D/N&M/O/None']), 'mean']
        middle_C = df.loc[list(scenario_groups['C + D/N&M/O/None']), 'mean']
        middle_P = df.loc[list(scenario_groups['P + D/N&M/O/None']), 'mean']
        worst = df.loc[df.index.isin(scenario_groups['D/N&M/O/None']), 'mean']

        return ss.oneway.anova_oneway((best, middle_C, middle_P, worst),
                                      groups=None, use_var='unequal', welch_correction=True, trim_frac=0)

    anova_dalys = anova_oneway()
    anova_services = anova_oneway(num_services_increased_percent)
    anova_treatments = anova_oneway(num_treatments_total_increased_percent)

    # plot absolute numbers for scenarios

    # name_of_plot = f'Deaths, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('(Millions)')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'DALYs, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('(Millions)')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Service demand, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(num_service_demand_summarized / 1e6)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('(Millions)')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Service delivery ratio, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(ratio_service_summarized)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('services delivered / demand')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # plot yearly DALYs for best 9 scenarios
    name_of_plot = f'Yearly DALYs, {target_period()}'
    fig, ax = plt.subplots(figsize=(9, 6))
    best_scenarios = list(num_dalys_summarized.index[0:9]) + ['s_1']
    for s in best_scenarios:
        data = (num_dalys_yearly_summarized.loc[num_dalys_yearly_summarized.scenario == s, :]
                .drop(columns='scenario')
                .pivot(index='year', columns='stat')
                .droplevel(0, axis=1))
        ax.plot(data.index, data['mean'] / 1e6, label=substitute_labels[s], color=best_scenarios_color[s], linewidth=2)
        # ax.fill_between(data.index.to_numpy(),
        #                 (data['lower'] / 1e6).to_numpy(),
        #                 (data['upper'] / 1e6).to_numpy(),
        #                 color=best_scenarios_color[s],
        #                 alpha=0.2)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.set_xticks(data.index)
    legend_labels = [substitute_labels[v] for v in best_scenarios]
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                    color=best_scenarios_color[v]) for v in best_scenarios]
    ax.legend(legend_handles, legend_labels, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # plot yearly staff count (Clinical/Pharmacy/Nursing and Midwifery) for best 9 scenarios
    best_cadres = ['Clinical', 'Pharmacy', 'Nursing_and_Midwifery']
    name_of_plot = f'Yearly staff count for C+P+N, {target_period()}'
    fig, ax = plt.subplots(figsize=(9, 6))
    best_scenarios = list(num_dalys_summarized.index[0:9]) + ['s_1']
    for s in best_scenarios:
        data = staff_count.loc[staff_count.draw == s].set_index('year').drop(columns='draw').loc[:, best_cadres].sum(
            axis=1)
        ax.plot(data.index, data.values / 1e3, label=substitute_labels[s], color=best_scenarios_color[s])
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Thousands)')
    ax.set_xticks(data.index)
    legend_labels = [substitute_labels[v] for v in best_scenarios]
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                    color=best_scenarios_color[v]) for v in best_scenarios]
    ax.legend(legend_handles, legend_labels, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Services by appointment type, {target_period()}'
    # num_appts_summarized_in_millions = num_appts_summarized / 1e6
    # yerr_services = np.array([
    #     (num_services_summarized['mean'] - num_services_summarized['lower']).values,
    #     (num_services_summarized['upper'] - num_services_summarized['mean']).values,
    # ])/1e6
    # fig, ax = plt.subplots(figsize=(9, 6))
    # num_appts_summarized_in_millions.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)), num_services_summarized['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_appts_summarized_in_millions.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Services demand by appointment type, {target_period()}'
    # num_appts_demand_to_plot = num_appts_demand_summarized / 1e6
    # yerr_services = np.array([
    #     (num_service_demand_summarized['mean'] - num_service_demand_summarized['lower']).values,
    #     (num_service_demand_summarized['upper'] - num_service_demand_summarized['mean']).values,
    # ]) / 1e6
    # fig, ax = plt.subplots(figsize=(9, 6))
    # num_appts_demand_to_plot.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)), num_service_demand_summarized['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_appts_demand_to_plot.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    name_of_plot = f'Never ran services by appointment type, {target_period()}'
    num_never_ran_appts_summarized_in_millions = num_never_ran_appts_summarized / 1e6
    yerr_services = np.array([
        (num_never_ran_services_summarized['mean'] - num_never_ran_services_summarized['lower']).values,
        (num_never_ran_services_summarized['upper'] - num_never_ran_services_summarized['mean']).values,
    ])/1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    num_never_ran_appts_summarized_in_millions.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    ax.errorbar(range(len(param_names)), num_never_ran_services_summarized['mean'].values / 1e6, yerr=yerr_services,
                fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_never_ran_appts_summarized_in_millions.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Services by treatment type, {target_period()}'
    # num_treatments_summarized_in_millions = num_treatments_summarized / 1e6
    # yerr_services = np.array([
    #     (num_treatments_total_summarized['mean'] - num_treatments_total_summarized['lower']).values,
    #     (num_treatments_total_summarized['upper'] - num_treatments_total_summarized['mean']).values,
    # ]) / 1e6
    # fig, ax = plt.subplots(figsize=(10, 6))
    # num_treatments_summarized_in_millions.plot(kind='bar', stacked=True, color=treatment_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)), num_treatments_total_summarized['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_treatments_summarized_in_millions.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.4), title='Treatment type', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Never ran services by treatment type, {target_period()}'
    # num_never_ran_treatments_summarized_in_millions = num_never_ran_treatments_summarized / 1e6
    # yerr_services = np.array([
    #     (num_never_ran_treatments_total_summarized['mean'] - num_never_ran_treatments_total_summarized['lower']).values,
    #     (num_never_ran_treatments_total_summarized['upper'] - num_never_ran_treatments_total_summarized['mean']).values,
    # ]) / 1e6
    # fig, ax = plt.subplots(figsize=(10, 6))
    # num_never_ran_treatments_summarized_in_millions.plot(kind='bar', stacked=True, color=treatment_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)), num_never_ran_treatments_total_summarized['mean'].values / 1e6,
    #             yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_never_ran_treatments_summarized_in_millions.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.4), title='Treatment type', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'Number of staff by cadre, {TARGET_PERIOD[1].year}'
    # total_staff_to_plot = (staff_count_2029 / 1000).drop(columns='all_cadres').reindex(num_dalys_summarized.index)
    # column_dcsa = total_staff_to_plot.pop('DCSA')
    # total_staff_to_plot.insert(3, "DCSA", column_dcsa)
    # fig, ax = plt.subplots(figsize=(9, 6))
    # total_staff_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    # ax.set_ylabel('Thousands', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in total_staff_to_plot.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    name_of_plot = f'HCW time needed to deliver never ran appointments, {target_period()}'
    hcw_time_gap_to_plot = (hcw_time_gap / 1e6).reindex(num_dalys_summarized.index)
    column_dcsa = hcw_time_gap_to_plot.pop('DCSA')
    hcw_time_gap_to_plot.insert(3, "DCSA", column_dcsa)
    fig, ax = plt.subplots(figsize=(9, 6))
    hcw_time_gap_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    ax.set_ylabel('Minutes in Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in hcw_time_gap_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'HCW cost needed to deliver never ran appointments, {target_period()}'
    hcw_cost_gap_to_plot = (hcw_cost_gap / 1e6).reindex(num_dalys_summarized.index)
    column_dcsa = hcw_cost_gap_to_plot.pop('DCSA')
    hcw_cost_gap_to_plot.insert(3, "DCSA", column_dcsa)
    fig, ax = plt.subplots(figsize=(9, 6))
    hcw_cost_gap_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    ax.set_ylabel('USD in Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in hcw_cost_gap_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'HCW cost gap distribution among cadres, {target_period()}'
    cadres_to_plot = ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA', 'Other']
    hcw_cost_gap_percent_to_plot = hcw_cost_gap_percent[cadres_to_plot] * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    hcw_cost_gap_percent_to_plot.plot(kind='bar', color=officer_category_color, rot=0, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage %')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in hcw_cost_gap_percent_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category')
    # plot the average proportions of all scenarios
    for c in cadres_to_plot:
        plt.axhline(y=hcw_cost_gap_percent_to_plot[c].mean(),
                    linestyle='--', color=officer_category_color[c], alpha=0.8,
                    label=c)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Extra budget allocation among cadres, {target_period()}'
    cadres_to_plot = ['Clinical', 'Nursing_and_Midwifery', 'Pharmacy', 'DCSA', 'Other']
    extra_budget_allocation_to_plot = extra_budget_allocation[cadres_to_plot] * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    extra_budget_allocation_to_plot.plot(kind='bar', color=officer_category_color, rot=0, ax=ax)
    ax.set_ylabel('Percentage %')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in extra_budget_allocation_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category')
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Total budget in USD dollars by cadre, {target_period()}'
    # total_cost_to_plot = (total_cost_all_yrs / 1e6).drop(columns='all_cadres').reindex(num_dalys_summarized.index)
    # column_dcsa = total_cost_to_plot.pop('DCSA')
    # total_cost_to_plot.insert(3, "DCSA", column_dcsa)
    # fig, ax = plt.subplots(figsize=(9, 6))
    # total_cost_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in total_cost_to_plot.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # name_of_plot = f'DALYs by cause, {target_period()}'
    # num_dalys_by_cause_summarized_in_millions = num_dalys_by_cause_summarized / 1e6
    # yerr_dalys = np.array([
    #     (num_dalys_summarized['mean'] - num_dalys_summarized['lower']).values,
    #     (num_dalys_summarized['upper'] - num_dalys_summarized['mean']).values,
    # ])/1e6
    # fig, ax = plt.subplots(figsize=(9, 6))
    # num_dalys_by_cause_summarized_in_millions.plot(kind='bar', stacked=True, color=cause_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)), num_dalys_summarized['mean'].values / 1e6, yerr=yerr_dalys,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_dalys_by_cause_summarized_in_millions.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # fig.subplots_adjust(right=0.7)
    # ax.legend(
    #     loc="center left",
    #     bbox_to_anchor=(0.750, 0.6),
    #     bbox_transform=fig.transFigure,
    #     title='Cause of death or injury',
    #     title_fontsize='x-small',
    #     fontsize='x-small',
    #     reverse=True,
    #     ncol=1
    # )
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # plot relative numbers for scenarios
    name_of_plot = f'DALYs averted against no expansion, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_averted / 1e6, num_dalys_averted_percent, annotation=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Deaths averted against no expansion, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_averted / 1e6, num_deaths_averted_percent, annotation=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Service delivery ratio against no expansion, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(service_ratio_increased * 100, service_ratio_increased_percent, annotation=True)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('Percentage')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    name_of_plot = f'Extra staff by cadre against no expansion, {TARGET_PERIOD[1].year}'
    extra_staff_by_cadre_to_plot = extra_staff_2029.drop(columns='all_cadres').reindex(
        num_dalys_summarized.index).drop(['s_1']) / 1e3
    column_dcsa = extra_staff_by_cadre_to_plot.pop('DCSA')
    extra_staff_by_cadre_to_plot.insert(3, "DCSA", column_dcsa)
    fig, ax = plt.subplots(figsize=(9, 6))
    extra_staff_by_cadre_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    ax.set_ylabel('Thousands', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in extra_staff_by_cadre_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Extra budget by cadre against no expansion, {target_period()}'
    extra_cost_by_cadre_to_plot = extra_cost_all_yrs.drop(columns='all_cadres').reindex(
        num_dalys_summarized.index).drop(index='s_1') / 1e6
    column_dcsa = extra_cost_by_cadre_to_plot.pop('DCSA')
    extra_cost_by_cadre_to_plot.insert(3, "DCSA", column_dcsa)
    fig, ax = plt.subplots(figsize=(9, 6))
    extra_cost_by_cadre_to_plot.plot(kind='bar', stacked=True, color=officer_category_color, rot=0, ax=ax)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in extra_cost_by_cadre_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Officer category', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'Services increased by appointment type \nagainst no expansion, {target_period()}'
    num_appts_increased_in_millions = num_appts_increased / 1e6
    yerr_services = np.array([
        (num_services_increased['mean'] - num_services_increased['lower']).values,
        (num_services_increased['upper'] - num_services_increased['mean']).values,
    ]) / 1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    num_appts_increased_in_millions.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    ax.errorbar(range(len(param_names)-1), num_services_increased['mean'].values / 1e6, yerr=yerr_services,
                fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_appts_increased_in_millions.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(
        name_of_plot.replace(' ', '_').replace(',', '').replace('\n', ''))
    )
    fig.show()
    plt.close(fig)

    name_of_plot = f'Never ran services reduced by appointment type \nagainst no expansion, {target_period()}'
    num_never_ran_appts_reduced_to_plot = num_never_ran_appts_reduced / 1e6
    # yerr_services = np.array([
    #     (num_services_increased['mean'] - num_services_increased['lower']).values,
    #     (num_services_increased['upper'] - num_services_increased['mean']).values,
    # ]) / 1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    num_never_ran_appts_reduced_to_plot.plot(kind='bar', stacked=True, color=appt_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names) - 1), num_services_increased['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_never_ran_appts_reduced_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(
        name_of_plot.replace(' ', '_').replace(',', '').replace('\n', ''))
    )
    fig.show()
    plt.close(fig)

    name_of_plot = f'Never ran services reduced by treatment type \nagainst no expansion, {target_period()}'
    num_never_ran_treatments_reduced_to_plot = num_never_ran_treatments_reduced / 1e6
    # yerr_services = np.array([
    #     (num_services_increased['mean'] - num_services_increased['lower']).values,
    #     (num_services_increased['upper'] - num_services_increased['mean']).values,
    # ]) / 1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    num_never_ran_treatments_reduced_to_plot.plot(kind='bar', stacked=True, color=treatment_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names) - 1), num_services_increased['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_never_ran_treatments_reduced_to_plot.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Treatment type', title_fontsize='small',
               fontsize='small', reverse=True)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(
        name_of_plot.replace(' ', '_').replace(',', '').replace('\n', ''))
    )
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Services increased by treatment type \nagainst no expansion, {target_period()}'
    # num_treatments_increased_in_millions = num_treatments_increased / 1e6
    # yerr_services = np.array([
    #     (num_treatments_total_increased['mean'] - num_treatments_total_increased['lower']).values,
    #     (num_treatments_total_increased['upper'] - num_treatments_total_increased['mean']).values,
    # ]) / 1e6
    # fig, ax = plt.subplots(figsize=(10, 6))
    # num_treatments_increased_in_millions.plot(kind='bar', stacked=True, color=treatment_color, rot=0, ax=ax)
    # ax.errorbar(range(len(param_names)-1), num_treatments_total_increased['mean'].values / 1e6, yerr=yerr_services,
    #             fmt=".", color="black", zorder=100)
    # ax.set_ylabel('Millions', fontsize='small')
    # ax.set(xlabel=None)
    # xtick_labels = [substitute_labels[v] for v in num_treatments_increased_in_millions.index]
    # ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.4), title='Treatment type', title_fontsize='small',
    #            fontsize='small', reverse=True)
    # plt.title(name_of_plot)
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(
    #     name_of_plot.replace(' ', '_').replace(',', '').replace('\n', ''))
    # )
    # fig.show()
    # plt.close(fig)

    name_of_plot = f'DALYs averted by cause against no expansion, {target_period()}'
    num_dalys_by_cause_averted_in_millions = num_dalys_by_cause_averted / 1e6
    yerr_dalys = np.array([
        (num_dalys_averted['mean'] - num_dalys_averted['lower']).values,
        (num_dalys_averted['upper'] - num_dalys_averted['mean']).values,
    ]) / 1e6
    fig, ax = plt.subplots(figsize=(9, 6))
    num_dalys_by_cause_averted_in_millions.plot(kind='bar', stacked=True, color=cause_color, rot=0, ax=ax)
    ax.errorbar(range(len(param_names)-1), num_dalys_averted['mean'].values / 1e6, yerr=yerr_dalys,
                fmt=".", color="black", zorder=100)
    ax.set_ylabel('Millions', fontsize='small')
    ax.set(xlabel=None)
    xtick_labels = [substitute_labels[v] for v in num_dalys_by_cause_averted.index]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize='small')
    fig.subplots_adjust(right=0.7)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.750, 0.6),
        bbox_transform=fig.transFigure,
        title='Cause of death or injury',
        title_fontsize='x-small',
        fontsize='x-small',
        ncol=1,
        reverse=True
    )
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # plot ROI and CE for all expansion scenarios

    name_of_plot = f'DALYs averted per extra USD dollar invested, {target_period()}'
    fig, ax = do_bar_plot_with_ci(ROI)
    ax.set_title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # name_of_plot = f'Cost per DALY averted, {target_period()}'
    # fig, ax = do_bar_plot_with_ci(CE)
    # ax.set_title(name_of_plot)
    # ax.set_ylabel('USD dollars')
    # fig.tight_layout()
    # fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    # fig.show()
    # plt.close(fig)

    # todo
    # As we have 33 scenarios in total, \
    # design comparison groups of scenarios to examine marginal/combined productivity of cadres.
    # To vary the HRH budget growth rate (default: 4.2%) and do sensitivity analysis \
    # (around the best possible extra budget allocation scenario)?
    # As it is analysis of 10 year results, it would be better to consider increasing annual/minute salary? The \
    # inflation rate of GDP and health workforce budget and the increase rate of salary could be assumed to be \
    # the same, thus no need to consider the increase rate of salary if GDP inflation is not considered.
    # To plot time series of staff and budget in the target period to show \
    # how many staff and how much budget to increase yearly (choose the best scenario to illustrate)?
    # Get and plot services by short treatment id?
    # To design more scenarios so that Pharmacy cadre can be expanded more than the 33 scenarios
    # and so that each cadre has different scale up factor (the one in more shortage will need to be scaled up more)?
    # Later, to explain the cause of differences in scenarios, might consider hcw time flow?
    # Before submit a run, merge in the remote master.
    # Think about a measure of Universal Health Service Coverage for the scenarios.


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
        the_target_period=(Date(2019, 1, 1), Date(2029, 12, 31))
    )
