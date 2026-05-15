"""
This file analyses and plots the patient load and mix in service areas including outpatient (including ANC) and
emergency care.

The scenarios are defined in XXX.py.
"""
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    load_pickled_dataframes,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None,
          the_target_period: Tuple[Date, Date] = None,
          district_resolution: Optional[bool] = False,
          level_resolution: Optional[bool] = False,
          year_resolution: Optional[bool] = False,
          month_resolution: Optional[bool] = False,
          date_resolution: Optional[bool] = False):

    TARGET_PERIOD = the_target_period

    def get_patient_counts_in_emerg(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # include HSIs of generic first attendance (nonemerg) and first and subsequent ANC visits
        _df = _df.loc[_df['TREATMENT_ID'].isin(['FirstAttendance_Emergency',
                                                'FirstAttendance_SpuriousEmergencyCare'])]

        # count patient volume, dropping duplicated person_ids between first attendance and anc on the same day
        _daily_person_counts = _df.groupby("date")["Person_ID"].nunique()
        _daily_person_counts.index = pd.to_datetime(_daily_person_counts.index)
        _daily_person_counts.name = "daily_patient_volume"
        return _daily_person_counts

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

    def get_patient_counts_in_opd(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # include HSIs of generic first attendance (nonemerg) and first and subsequent ANC visits
        _df = _df.loc[_df['TREATMENT_ID'].isin(['FirstAttendance_NonEmergency',
                                                'AntenatalCare_Outpatient'])]

        # count patient volume, dropping duplicated person_ids between first attendance and anc on the same day
        _daily_person_counts = _df.groupby("date")["Person_ID"].nunique()
        _daily_person_counts.index = pd.to_datetime(_daily_person_counts.index)
        _daily_person_counts.name = "daily_patient_volume"

        return _daily_person_counts

    # log = load_pickled_dataframes(results_folder, 0, 0)
    # h = pd.DataFrame(
    #     log['tlo.methods.healthsystem.summary']['hsi_event_details'].iloc[0]['hsi_event_key_to_event_details']
    # ).T

    hcw_count = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_staff_counts,
        do_scaling=False
    )
    hcw_count = hcw_count[(0, 0)]

    patient_volume_opd = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=get_patient_counts_in_opd,
        do_scaling=True
    )

    patient_volume_emerg = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=get_patient_counts_in_emerg,
        do_scaling=True
    )

    patient_load_per_hcw_in_opd = patient_volume_opd / hcw_count.values[0]
    patient_load_per_hcw_in_emerg = patient_volume_emerg / hcw_count.values[0]

    opd = patient_load_per_hcw_in_opd[(0, 0)].rename("Outpatient care")
    emerg = patient_load_per_hcw_in_emerg[(0, 0)].rename("Emergency/Intensive care")

    patient_load_per_hcw = pd.concat([opd, emerg], axis=1)
    patient_load_per_hcw = patient_load_per_hcw.fillna(0)

    patient_load_per_hcw = (
        patient_load_per_hcw
        .reset_index()
        .melt(
            id_vars="date",
            var_name="coarse_loc_cat_str",
            value_name="daily_patient_load_per_hcw"
        )
    )

    path_to_tlm_folder = (
        resourcefilepath
        / "healthsystem"
        / "human_resources"
        / "TLM_2024"
    )

    patient_load_per_hcw.to_stata(
        path_to_tlm_folder / "tlo_pat_load.dta",
        write_index=False,
        convert_dates={"date": "td"}
    )

    # issue: do not know how many staff members are in outpatient, emergency care, and inpatient
    # issue: how to treat nan entries for emerg -> default 0?


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources'),
        the_target_period=(Date(2010, 1, 1), Date(2010, 5, 31))
    )


