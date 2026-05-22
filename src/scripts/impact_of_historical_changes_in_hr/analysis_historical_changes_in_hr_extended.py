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

from scripts.impact_of_historical_changes_in_hr.scenario_historical_changes_in_hr import (
    HistoricalChangesInHRH,
)
from tlo import Date
from tlo.analysis.utils import extract_results, make_age_grp_lookup, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, the_target_period: Tuple[Date, Date] = None):

    TARGET_PERIOD = the_target_period
    hrh_check_period = (Date(2020, 1, 1), Date(2030, 12, 31))

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

    def get_total_num_dalys_by_label_htm(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        y = _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'year', 'sex', 'age_range']) \
            .sum(axis=0)

        # define course cause mapper for HIV, TB, MALARIA and OTHER
        causes = {
            'AIDS': 'HIV/AIDS',
            'TB (non-AIDS)': 'TB',
            'Malaria': 'Malaria',
            'Lower respiratory infections': 'Lower respiratory infections',
            'Neonatal Disorders': 'Neonatal Disorders',
            'Maternal Disorders': 'Maternal Disorders',
            '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
        }
        causes_relabels = y.index.map(causes).fillna('Other')

        return y.groupby(by=causes_relabels).sum()[list(causes.values())]

    def get_total_num_dalys_by_label_all_causes(_df):
        """Return the total number of DALYS in the TARGET_PERIOD cause label."""
        return _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'year', 'age_range', 'sex']) \
            .sum(axis=0)

    # todo: to get HRH counts by cadre group and year
    def get_staff_counts(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]
        _df_staff = (
            pd.Series(_df.GenericClinic[0], name="staff_count")
            .rename_axis("facility_officer")
            .reset_index()
        )

        _df_staff[["facility_id", "officer_type"]] = _df_staff["facility_officer"].str.extract(
            r"FacilityID_(\d+)_Officer_(.*)"
        )

        _df_staff["facility_id"] = _df_staff["facility_id"].astype(int)

        _df_staff = _df_staff[["facility_id", "officer_type", "staff_count"]]

        _df_staff = _df_staff.loc[_df_staff.officer_type != 'DCSA']

        _df_staff = pd.Series(_df_staff.staff_count.sum())

        _df_staff.index = [pd.to_datetime(_df["date"].iloc[0])]
        _df_staff.name = 'yearly_staff_count'

        return _df_staff

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

    # HRH staff counts
    hcw_count = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_staff_counts,
        do_scaling=False
    )

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

    # %% Total numbers of deaths / DALYS
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names)
    num_deaths_summarized = summarize(num_deaths).loc[0].unstack().reindex(param_names)

    # Results by disease (HTM/OTHER and split by age/sex)
    total_num_dalys_by_label_results = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label_htm,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)

    total_num_dalys_by_label_results_all_causes = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=get_total_num_dalys_by_label_all_causes,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)  # outputs/horizontal_and_vertical_programs-2024-05-16
    args = parser.parse_args()

    # Produce results for short-term analysis - 2020 - 2024 (incl.)
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2020, 1, 1), Date(2024, 12, 31))
    )
    # Produce results for only later period 2025-2030 (incl.)
    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2025, 1, 1), Date(2030, 12, 31))
    )
