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

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # HRH staff counts
    hcw_count = (extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_staff_counts,
        do_scaling=False
    )).pipe(set_param_names_as_column_index_level_0)
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

    # marker for each officer type
    markers = {
        'Clinical': 'o',
        'Nursing_and_Midwifery': '*',
        'Pharmacy': '^',
        'DCSA': 'd',
        'Other': 'X',
    }

    # color for each scenario
    cmap = plt.cm.get_cmap('tab10', len(param_names))
    colors = {
        scenario: cmap(i)
        for i, scenario in enumerate(param_names)
    }

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
                color=colors[scenario],
                label=f'{officer} - {scenario}',
            )

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of staff')
    ax.set_xticks(sorted(hcw_count['year'].unique()))

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8
    )

    plt.tight_layout()

    plt.show()

    # Absolute Number of Deaths and DALYs
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

    num_dalys_by_year_cause.to_csv(output_folder / 'num_dalys_by_year_cause (for Izzy).csv', index=False)
    num_deaths_by_year_cause.to_csv(output_folder / 'num_deaths_by_year_cause (for Izzy).csv', index=False)


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
