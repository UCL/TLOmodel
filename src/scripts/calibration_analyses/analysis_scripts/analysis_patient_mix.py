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

    def get_hcw_count(_df):
        the_hrh_target_period = (Date(2024, 1, 1), Date(2024, 1, 1))
        _df = _df.loc[pd.to_datetime(_df['date']).between(*the_hrh_target_period), :]
        _df_staff = (
            pd.Series(_df.GenericClinic.iloc[0], name="staff_count")
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

    def get_patient_count(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # count overall patient volume, dropping duplicated person_ids on the same day
        _daily_person_counts = _df.groupby("date")["Person_ID"].nunique()
        _daily_person_counts.index = pd.to_datetime(_daily_person_counts.index)
        _daily_person_counts.name = "daily_patient_volume"

        return _daily_person_counts

    def get_hcw_count_facility_id(_df):
        the_hrh_target_period = (Date(2010, 1, 1), Date(2010, 5, 1))
        _df = _df.loc[pd.to_datetime(_df['date']).between(*the_hrh_target_period), :]

        _df_staff = (
            pd.Series(_df.GenericClinic.iloc[0], name="Staff_Count")
            .rename_axis("facility_officer")
            .reset_index()
        )

        _df_staff[["Facility_ID", "Officer_Type"]] = _df_staff["facility_officer"].str.extract(
            r"FacilityID_(\d+)_Officer_(.*)"
        )

        _df_staff["Facility_ID"] = _df_staff["Facility_ID"].astype(int)

        _df_staff = _df_staff[["Facility_ID", "Officer_Type", "Staff_Count"]]

        _df_staff = _df_staff.groupby("Facility_ID")["Staff_Count"].sum()

        return _df_staff

    def get_patient_count_facility_id(_df):
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # count overall patient volume, dropping duplicated person_ids on the same day
        _daily_person_counts = _df.groupby(["date", "Facility_ID"])["Person_ID"].nunique().reset_index()
        _daily_person_counts["date"] = pd.to_datetime(_daily_person_counts.date)
        _daily_person_counts = _daily_person_counts.set_index(["date", "Facility_ID"])["Person_ID"]
        _daily_person_counts.name = "daily_patient_volume"

        return _daily_person_counts

    def merge_info_from_mfl(_df):
        # should merge on "left" as original designed level 1b is now merged in to level 2
        _df = _df.merge(mfl[["Facility_ID", "District", "Facility_Level", "Region"]], on="Facility_ID", how="left")

        return _df

    def get_patient_mix_total_period(_df):
        # keep only months in TLM for comparison consistency
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # map Event_Name with TLM service area
        _df["loc_cat"] = _df["Event_Name"].map(hsi_loc_cat_map)

        # check that all events are mapped
        unmapped = _df.loc[_df["loc_cat"].isna(), "Event_Name"].unique()
        assert set(unmapped).issubset(
            {"HSI_Alri_Treatment", "_BaseHSIGenericFirstAppt", "HSI_GenericEmergencyFirstAppt",
             "HSI_GenericNonEmergencyFirstAppt", "HSI_Hiv_SelfTest", "HSI_Malaria_rdt_community", "HSI_Schisto_MDA",
             "Inpatient_Care"})

        # todo: drop duplicated persons in the target period?
        # Duplicated case 1: same person id received multiple HSIs on a day, including generic fist appt
        # Duplicated case 2: same person id received multiple HSIs due to the same episode of condition
        # in the target period, such as inpatient postnatal care
        # Rough solution: drop nan loc_cat entries () and then
        # drop duplicated person ids receiving care in the same tlm service area on the same day, consistent with
        # TLM data collection method that is based on daily collection
        # One possible issue is that this drop may drop more than necessary,
        # such as a patient visiting OPD clinic may have multiple diseases to see, whereas TLM patient exit shows
        # no duplicated patients and that each patient has only visited one clinic on a day
        _df = _df.dropna(subset=["loc_cat"])
        _df = _df.drop_duplicates(subset=["date", "Person_ID", "loc_cat"])

        # get patient counts per target subgroups
        _df = _df.groupby(
            ["Facility_ID", "Sex", "Age_Range", "Wealth", "loc_cat"]
        )["Person_ID"].count().reset_index().rename(columns={"Person_ID": "patient_count"})

        # merge info from mfl and format
        _df = merge_info_from_mfl(_df)

        # drop HQ/Facility_Level= 5 and Community Level/Facility_Level=0
        _df.drop(index=_df[_df["Facility_Level"].isin(["0", "5"])].index, inplace=True)

        # fill NANs
        _df.loc[
            _df["Facility_Level"] == "4", ["patient_count", "District", "Region"]
        ] = [0, "Central Hospitals (Southern)", "Southern"]  # ZMH
        _df.loc[
            _df["Facility_ID"] == 128, "District"
        ] = "Central Hospitals (Southern)"
        _df.loc[
            _df["Facility_ID"] == 129, "District"
        ] = "Central Hospitals (Northern)"
        _df.loc[
            _df["Facility_ID"] == 130, "District"
        ] = "Central Hospitals (Central)"

        # keep only districts in TLM
        _df = _df.loc[_df["District"].isin(common_districts)]

        # group up by subgroups and get patient proportions across subgroups
        group_list = ["Facility_Level", "Age_Range", "Wealth", "Sex", "loc_cat"]
        _df_mix = pd.DataFrame(columns=["category", "subgroup", "patient_proportion"])
        for sg in group_list:
            _df_sg = _df.groupby(sg)["patient_count"].sum().reset_index().rename(
                columns={sg: "subgroup"}
            )
            _df_sg["category"] = sg
            _df_sg["patient_proportion"] = _df_sg["patient_count"] / _df_sg["patient_count"].sum()
            _df_mix = pd.concat([_df_mix, _df_sg], ignore_index=True)

        # create series
        _df_mix = _df_mix.set_index(["category", "subgroup"])["patient_proportion"]

        return _df_mix

    def get_cons_access_mix_total_period(_df):
        # keep only months in TLM
        _df = _df.loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), :]

        # merge in facility information from mfl and format
        _df.rename(columns={
            "event_name": "Event_Name",
            "facility_id": "Facility_ID",
            "person_id": "Person_ID"
        }, inplace=True)
        _df = merge_info_from_mfl(_df)

        # drop HQ/Facility_Level= 5 and Community Level/Facility_Level=0
        _df.drop(index=_df[_df["Facility_Level"].isin(["0", "5"])].index, inplace=True)

        # fill NANs
        _df.loc[
            _df["Facility_Level"] == "4", ["District", "Region"]
        ] = ["Central Hospitals (Southern)", "Southern"]  # ZMH
        _df.loc[
            _df["Facility_ID"] == 128, "District"
        ] = "Central Hospitals (Southern)"
        _df.loc[
            _df["Facility_ID"] == 129, "District"
        ] = "Central Hospitals (Northern)"
        _df.loc[
            _df["Facility_ID"] == 130, "District"
        ] = "Central Hospitals (Central)"

        # keep only districts in TLM
        _df = _df.loc[_df["District"].isin(common_districts)]

        # todo: get the proportions of patients not getting the prescribed cons. by subgroup (overall, facility level)
        # need to identify the HSIs involves prescribing:
        # 1. approximated by item_requested = item_available + item_not_available != {}, but TLO items may not be
        # medicines prescribed for the patients to take home as assumed in TLM data collection
        # 2. indicator of Prescription involvement (assume test, investigation, check HSIs do not involve prescribing)

        # add in column of prescription involvement
        _df["Prescription involvement"] = _df["Event_Name"].map(hsi_prescription_map)

        # check that all events are mapped
        unmapped = _df.loc[_df["Prescription involvement"].isna(), "Event_Name"].unique()
        assert set(unmapped).issubset({"Inpatient_Care"})

        # add in column of item_requested
        # change item columns to dicts from string
        import ast
        _df["Item_NotAvailable"] = _df["Item_NotAvailable"].apply(ast.literal_eval)
        _df["Item_Available"] = _df["Item_Available"].apply(ast.literal_eval)
        _df["Item_Used"] = _df["Item_Used"].apply(ast.literal_eval)

        _df["Item_Requested"] = _df.apply(
            lambda row: dict(
                Counter(row["Item_Available"])
                + Counter(row["Item_NotAvailable"])
            ),
            axis=1
        )

        # add the column of access_meds, consistent with TLM data; (Yes, No, Non prescribed)
        # label event_name with item_requested = {} as "Non prescribed"
        # label event_name assigned to "No" in the "Prescription involvement" column as "Non prescribed"
        # label event_name with item_used = !{} as "Yes" (if not "Non prescribed")
        # label event_name with item_used = {} as "No" (if not "Non prescribed")
        _df["access_meds"] = np.where(
            (_df["Item_Requested"].apply(lambda x: not x))
            | (_df["Prescription involvement"] == "No"),
            "Non prescribed",
            np.where(
                _df["Item_Used"].apply(lambda x: bool(x)),
                "Yes",
                "No",
            ),
        )

        # add loc_cat column; map Event_Name with TLM service area
        _df["loc_cat"] = _df["Event_Name"].map(hsi_loc_cat_map)

        # check that all events are mapped
        unmapped = _df.loc[_df["loc_cat"].isna(), "Event_Name"].unique()
        assert set(unmapped).issubset(
            {"HSI_Alri_Treatment", "_BaseHSIGenericFirstAppt", "HSI_GenericEmergencyFirstAppt",
             "HSI_GenericNonEmergencyFirstAppt", "HSI_Hiv_SelfTest", "HSI_Malaria_rdt_community", "HSI_Schisto_MDA",
             "Inpatient_Care"})

        # may not drop duplicated person_id + loc_cat + day

        # percent meds access, by subgroup ["overall", "loc_cat", "fac_level", "district"]
        def meds_access_by_subgroup_tlo(__df, subgroup=None):
            if subgroup == "overall":
                _df_yn = __df[__df["access_meds"].isin(["Yes", "No"])]
                access_meds_percent = (_df_yn["access_meds"] == "Yes").mean() * 100

                access_meds_df = pd.DataFrame({
                    "category": ["overall"],
                    "subgroup": ["overall"],
                    "source": ["TLO"],
                    "access_meds_percent": [access_meds_percent]
                })
            else:  # subgroup == ["fac_level", "district", "loc_cat"]
                access_meds_df = (
                    __df[__df["access_meds"].isin(["Yes", "No"])]
                    .groupby(subgroup, dropna=False)["access_meds"]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .reset_index()
                    .rename(columns={subgroup: "subgroup", "access_meds": "access_meds_percent"})
                    .assign(category=subgroup, source="TLO")
                    [["category", "subgroup", "source", "access_meds_percent"]]
                )

            return access_meds_df

        subgroups = ["overall", "loc_cat", "Facility_Level", "District"] #

        _access_meds = pd.concat(
            [meds_access_by_subgroup_tlo(_df, subgroup=s) for s in subgroups],
            ignore_index=True
        )

        _access_meds["category"] = _access_meds["category"].replace(
            {"overall": "Overall", "loc_cat": "Service_Area"}
        )

        # creat series
        _access_meds = _access_meds.set_index(["category", "subgroup", "source"])["access_meds_percent"]

        return _access_meds

    # log = load_pickled_dataframes(results_folder, 0, 0)
    # h = pd.DataFrame(
    #     log['tlo.methods.healthsystem.']['hsi_event_details'].iloc[0]['hsi_event_key_to_event_details']
    # ).T

    mfl = pd.read_csv(resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

    ## patient volume
    patient_volume_facility_id = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=get_patient_count_facility_id,
        do_scaling=True
    ).loc[:, [(0, 0)]]  # draw=0, run=0

    patient_volume_facility_id.columns = patient_volume_facility_id.columns.droplevel('run')
    patient_volume_facility_id = patient_volume_facility_id.reset_index().rename(
        columns={0: "Patient_Volume", "date": "Date"})
    patient_volume_facility_id = merge_info_from_mfl(patient_volume_facility_id)

    ## hcw count
    hcw_count_facility_id = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_hcw_count_facility_id,
        do_scaling=False
    ).loc[:, [(0, 0)]]  # draw=0, run=0

    hcw_count_facility_id.columns = hcw_count_facility_id.columns.droplevel('run')
    hcw_count_facility_id = hcw_count_facility_id.reset_index().rename(columns={0: 'Staff_Count'})
    hcw_count_facility_id = merge_info_from_mfl(hcw_count_facility_id)

    # fix levels in hcw_count: 1b has no staff now as merged to 2; ZMH at 4 to be merged to 3; drop HQ at 5
    hcw_count_facility_id.drop(index=hcw_count_facility_id[hcw_count_facility_id["Facility_Level"] == "5"].index,
                               inplace=True)
    assert (hcw_count_facility_id.loc[hcw_count_facility_id["Facility_Level"] == "1b", "Staff_Count"] == 0).all()
    hcw_count_facility_id.drop(index=hcw_count_facility_id[hcw_count_facility_id["Facility_Level"] == "1b"].index,
                               inplace=True)
    ## patient load
    assert set(patient_volume_facility_id.Facility_ID.drop_duplicates()).issubset(
        set(hcw_count_facility_id.Facility_ID.drop_duplicates())
    )
    daily_patient_load_per_hcw = patient_volume_facility_id[["Date", "Facility_ID", "Patient_Volume"]].merge(
        hcw_count_facility_id[["Facility_ID", "District", "Facility_Level", "Region", "Staff_Count"]],
        on=["Facility_ID"], how="right")
    # fill NAN entries
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_Level"] == "4", ["Patient_Volume", "District", "Region", "Date"]
    ] = [0, "Central Hospitals (Southern)", "Southern", patient_volume_facility_id.loc[0, "Date"]]  # ZMH
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 128, "District"
    ] = "Central Hospitals (Southern)"
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 129, "District"
    ] = "Central Hospitals (Northern)"
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 130, "District"
    ] = "Central Hospitals (Central)"

    # # check the TLO outputs sample size
    # tab = pd.crosstab(
    #     daily_patient_load_per_hcw["District"],
    #     daily_patient_load_per_hcw["Facility_Level"],
    #     dropna=False
    # )
    #
    # print(tab)

    def daily_pat_load_per_hcw_per_resolution(_df, resolution=["District", "Facility_Level"], adjust_hcw=True):
        res_plus_date = resolution + ["Date"]
        _df = daily_patient_load_per_hcw.groupby(res_plus_date).agg(
            {"Staff_Count": "sum", "Patient_Volume": "sum"}
        ).reset_index()
        if adjust_hcw:
            # Adjust available HCWs on duty every day by a ratio of 0.5649,
            # which is the prob. that any HCW is on duty on any day (estimates from CHAI data: 206.3381/365.25),
            # given TLO assumes the same HCWs in the HS every day in a year.
            # (TLO also assumes the patients seek care independently of the availability of HCWs and seek care every day;
            # so no need to adjust patient volumes)
            _df['Daily_Patient_Load_Per_HCW'] = _df["Patient_Volume"] / (_df["Staff_Count"] * 206.3381 / 365.25)
        else:
            _df['Daily_Patient_Load_Per_HCW'] = _df["Patient_Volume"] / _df["Staff_Count"]

        return _df

    daily_patient_load_per_hcw = daily_pat_load_per_hcw_per_resolution(daily_patient_load_per_hcw)

    # path to TLM data sources
    path_to_tlm_folder = (
        resourcefilepath
        / "healthsystem"
        / "human_resources"
        / "TLM_2024"
    )

    # HSI list is from https://www.tlomodel.org/hsi_events.html
    hsi_loc_cat_map = pd.read_csv(
        path_to_tlm_folder / 'hsi_tlm_service_area_map.csv',
        usecols=["Event", "TLM service area"]
    ).rename(columns={"Event": "Event_Name", "TLM service area": "loc_cat"}).set_index("Event_Name")["loc_cat"]

    hsi_prescription_map = pd.read_csv(
        path_to_tlm_folder / 'hsi_tlm_service_area_map.csv',
        usecols=["Event", "Prescription involvement"]
    ).rename(columns={"Event": "Event_Name"}).set_index("Event_Name")["Prescription involvement"]

    # read in TLM estimates
    hcw_tms_pat_load = pd.read_stata(path_to_tlm_folder/"tool_3_pat_load.dta", convert_categoricals=True)
    fac_tms_pat_load = pd.read_stata(path_to_tlm_folder / "tool_6_pat_load.dta", convert_categoricals=True)
    pat_exit = pd.read_stata(path_to_tlm_folder / "tool_2_pat_exit.dta", convert_categoricals=True)

    # check that districts and facility levels in the two tools are a subset of TLO output;
    # district and facility level consistency in the two tools already checked in Stata
    hcw_tms_pat_load["district"] = hcw_tms_pat_load["district"].replace({"Mzuzu": "Mzuzu City"})
    fac_tms_pat_load["district"] = fac_tms_pat_load["district"].replace({"Mzuzu": "Mzuzu City"})
    pat_exit["district"] = pat_exit["district"].replace({"Mzuzu": "Mzuzu City"})

    # assert set(hcw_tms_pat_load['district'].unique()).issubset(
    #     set(daily_patient_load_per_hcw['District'].unique())
    # )
    # assert set(hcw_tms_pat_load['fac_level'].unique()).issubset(
    #     set(daily_patient_load_per_hcw['Facility_Level'].unique())
    # )
    # assert set(pat_exit['district'].unique()).issubset(
    #     set(daily_patient_load_per_hcw['District'].unique())
    # )
    # assert set(pat_exit['fac_level'].unique()).issubset(
    #     set(daily_patient_load_per_hcw['Facility_Level'].unique())
    # )

    common_districts = hcw_tms_pat_load["district"].drop_duplicates().tolist()

    # *** patient mix comparisons ***
    # from patient exit
    def pat_prop_per_subgroup_total_period_tool_2(_df, subgroup="fac_level"):
        _df = _df[["respondent_id", subgroup]].groupby(subgroup).count().reset_index().rename(
            columns={"respondent_id": "pat_volume", subgroup: "subgroup"})
        _df["category"] = subgroup
        _df["pat_proportion"] = _df["pat_volume"] / _df["pat_volume"].sum()
        return _df

    subgroups = ["fac_level", "age_group_tlo", "wealth_tlo", "sex", "loc_cat"]
    pat_mix = pd.concat(
        [pat_prop_per_subgroup_total_period_tool_2(pat_exit, subgroup=s) for s in subgroups],
        ignore_index=True
    )
    pat_mix["source"] = "Patient Exit"

    # from facility summary
    def pat_prop_per_subgroup_total_period_tool_6(_df, subgroup="fac_level"):
        _df = _df[["num_of_patients", subgroup]].groupby(subgroup).sum().reset_index().rename(
            columns={"num_of_patients": "pat_volume", subgroup: "subgroup"})
        _df["category"] = subgroup
        _df["pat_proportion"] = _df["pat_volume"] / _df["pat_volume"].sum()
        return _df

    subgroups_fac_tms = ["fac_level", "loc_cat"]
    pat_mix_fac_tms = pd.concat(
        [pat_prop_per_subgroup_total_period_tool_6(fac_tms_pat_load, subgroup=s) for s in subgroups_fac_tms],
        ignore_index=True
    )
    pat_mix_fac_tms["source"] = "Facility Summary"

    pat_mix = pd.concat([pat_mix, pat_mix_fac_tms], ignore_index=True)

    # format to be consistent to TLO output
    pat_mix["category"] = pat_mix["category"].replace({"fac_level": "Facility_Level",
                                                       "age_group_tlo": "Age_Range",
                                                       "wealth_tlo": "Wealth",
                                                       "sex": "Sex",
                                                       "loc_cat": "Service_Area"})
    pat_mix = pat_mix[["category", "subgroup", "pat_proportion", "source"]].rename(
        columns={"pat_proportion": "mean"}
    ).copy()
    pat_mix["lower"] = pat_mix["mean"].copy()
    pat_mix["upper"] = pat_mix["mean"].copy()

    patient_mix_tlo = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=get_patient_mix_total_period,
        do_scaling=False
    ).fillna(0)
    patient_mix_tlo = summarize(patient_mix_tlo)
    patient_mix_tlo.columns = patient_mix_tlo.columns.droplevel('draw')
    patient_mix_tlo["source"] = "TLO"
    patient_mix_tlo.reset_index(inplace=True)

    assert set(pat_mix.columns) == set(patient_mix_tlo.columns)
    pat_mix = pd.concat([pat_mix, patient_mix_tlo], ignore_index=True)

    # todo: plot; note axis ticks should be the same across sources - assert needed

    # todo: *** prescribed cons. access comparison  ***

    ## from TLM
    def meds_access_by_subgroup(_df, subgroup=None):
        if subgroup == "overall":
            _df_yn = _df[_df["access_meds"].isin(["Yes", "No"])]
            access_meds_percent = (_df_yn["access_meds"] == "Yes").mean() * 100

            access_meds_df = pd.DataFrame({
                "category": ["overall"],
                "subgroup": ["overall"],
                "source": ["Patient Exit"],
                "access_meds_percent": [access_meds_percent]
            })
        else:  # subgroup == ["fac_level", "district", "loc_cat"]
            access_meds_df = (
                _df[_df["access_meds"].isin(["Yes", "No"])]
                .groupby(subgroup, dropna=False)["access_meds"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
                .rename(columns={subgroup: "subgroup", "access_meds": "access_meds_percent"})
                .assign(category=subgroup, source="Patient Exit")
                [["category", "subgroup", "source", "access_meds_percent"]]
            )

        return access_meds_df

    subgroups = ["overall", "loc_cat", "fac_level", "district"]

    access_meds = pd.concat(
        [meds_access_by_subgroup(pat_exit, subgroup=s) for s in subgroups],
        ignore_index=True
    )

    access_meds["category"] = access_meds["category"].replace(
        {"overall": "Overall", "loc_cat": "Service_Area", "fac_level": "Facility_Level", "district": "District"}
    )

    access_meds.rename(columns={"access_meds_percent": "mean"}, inplace=True)
    access_meds["lower"] = access_meds["mean"].copy()
    access_meds["upper"] = access_meds["mean"].copy()

    ## from TLO
    access_meds_tlo = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="Consumables",
        custom_generate_series=get_cons_access_mix_total_period,
        do_scaling=False
    )
    # ignore NAN entries
    access_meds_tlo = summarize(access_meds_tlo)
    access_meds_tlo.columns = access_meds_tlo.columns.droplevel('draw')
    access_meds_tlo .reset_index(inplace=True)

    assert set(access_meds.columns) == set(access_meds_tlo.columns)
    access_meds = pd.concat([access_meds, access_meds_tlo], ignore_index=True)

    # todo: plot access_meds comparison

    # *** patient load per hcw per day comparison ***
    # merge all three patient load estimates at the same resolution in one dataframe,
    # noting the source and keeping all observations
    hcw_tms_pat_load = hcw_tms_pat_load.rename(columns={
        "fac_level": "Facility_Level",
        "district": "District",
        "pat_day_tms_nr_adj": "Daily_Patient_Load_Per_HCW",
    })
    fac_tms_pat_load = fac_tms_pat_load.rename(columns={
        "fac_level": "Facility_Level",
        "district": "District",
        "pat_load_per_hcw": "Daily_Patient_Load_Per_HCW",
    })

    hcw_tms_pat_load["Source"] = "HCW TMS"
    fac_tms_pat_load["Source"] = "Facility Summary"
    daily_patient_load_per_hcw["Source"] = "TLO"

    pat_load_comparison = pd.concat([
        hcw_tms_pat_load[["District", "Facility_Level", "Daily_Patient_Load_Per_HCW", "Source",]],
        fac_tms_pat_load[["District", "Facility_Level", "Daily_Patient_Load_Per_HCW", "Source"]],
        daily_patient_load_per_hcw[["District", "Facility_Level", "Daily_Patient_Load_Per_HCW", "Source"]],
    ], ignore_index=True)

    assert len(hcw_tms_pat_load) + len(fac_tms_pat_load) + len(daily_patient_load_per_hcw) == len(pat_load_comparison)

    # drop level 0/community service by DCSA, level 5/HQ
    pat_load_comparison = pat_load_comparison.loc[~pat_load_comparison["Facility_Level"].isin(["0", "5"])]

    # *** make comparison plots ***
    from matplotlib.ticker import MultipleLocator

    # shared settings and helper function

    facility_levels = ["1a", "2", "3", "4"]
    sources = ["TLO", "HCW TMS", "Facility Summary"]

    markers = {
        "HCW TMS": "o",
        "Facility Summary": "^",
        "TLO": "d"
    }

    source_colors = {
        "TLO": "green",
        "HCW TMS": "blue",
        "Facility Summary": "orange"
    }

    PLOT_STYLE = {
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16
    }

    plt.rcParams.update(PLOT_STYLE)

    patient_load_col = "Daily_Patient_Load_Per_HCW"

    def annotate_small_n_horizontal(
        ax,
        y_positions,
        upper_bounds,
        counts,
        threshold=10,
        symbol="*"
    ):
        mask = counts < threshold

        for yy, xx in zip(
            np.asarray(y_positions)[mask],
            np.asarray(upper_bounds)[mask]
        ):
            ax.text(
                xx,
                yy,
                symbol,
                ha="left",
                va="center",
                fontsize=15,
                color="red"
            )

    def annotate_small_n_vertical(
        ax,
        x_positions,
        upper_bounds,
        counts,
        threshold=10,
        symbol="*"
    ):
        mask = counts < threshold

        for xx, yy in zip(
            np.asarray(x_positions)[mask],
            np.asarray(upper_bounds)[mask]
        ):
            ax.text(
                xx,
                yy,
                symbol,
                ha="center",
                va="bottom",
                fontsize=14,
                color="red"
            )

    # prepare data: keep districts appearing in HCW TMS

    common_districts = (
        pat_load_comparison
        .loc[pat_load_comparison["Source"] == "HCW TMS", "District"]
        .unique()
    )

    df_plot = pat_load_comparison[
        pat_load_comparison["District"].isin(common_districts)
    ].copy()

    # ** median + IQR plots **

    def summarise_median_iqr(data, group_cols):
        return (
            data
            .groupby(group_cols)[patient_load_col]
            .agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                n="count"
            )
            .reset_index()
        )

    # Plot 1: by facility level and district
    # y-axis = district, x-axis = patient load

    summary = summarise_median_iqr(
        df_plot,
        ["District", "Facility_Level", "Source"]
    )

    offset = 0.2

    fig, axes = plt.subplots(
        1,
        len(facility_levels),
        figsize=(20, 12),
        sharex="all"
    )

    for ax, fac_level in zip(axes, facility_levels):

        temp = summary[
            summary["Facility_Level"] == fac_level
            ].copy()

        districts = sorted(temp["District"].unique())
        y = np.arange(len(districts))

        for i, src in enumerate(sources):
            dat = (
                temp[temp["Source"] == src]
                .set_index("District")
                .reindex(districts)
            )

            xerr = np.vstack([
                dat["median"] - dat["q25"],
                dat["q75"] - dat["median"]
            ])

            ax.errorbar(
                x=dat["median"],
                y=y + (i - 1) * offset,
                xerr=xerr,
                fmt=markers[src],
                color=source_colors[src],
                markersize=8,
                elinewidth=1.5,
                capsize=2,
                capthick=1.5,
                linestyle="none",
                label=src
            )

            annotate_small_n_horizontal(
                ax=ax,
                y_positions=y + (i - 1) * offset,
                upper_bounds=dat["q75"],
                counts=dat["n"]
            )

        ax.set_title(f"Facility Level {fac_level}")

        ax.set_yticks(y)
        ax.set_yticklabels(districts)
        ax.invert_yaxis()

        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.grid(axis="x", which="major", alpha=0.5)
        ax.grid(axis="x", which="minor", alpha=0.25, linestyle=":")

        ax.tick_params(axis="both")
        ax.tick_params(axis="x", labelrotation=90)

    # axes[0].set_ylabel("District")
    fig.supxlabel(
        "Median Daily Patient Load per HCW",
        y=0.04
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    # Plot 2: overall facility levels
    # y-axis = district, x-axis = patient load

    summary = summarise_median_iqr(
        df_plot,
        ["District", "Source"]
    )

    fig, ax = plt.subplots(figsize=(10, 12))

    districts_ordered = (
        summary
        .loc[summary["Source"] == "HCW TMS"]
        .sort_values("median", ascending=False)["District"]
        .tolist()
    )
    y = np.arange(len(districts_ordered))

    offset = 0.2

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
            .set_index("District")
            .reindex(districts_ordered)
        )

        xerr = np.vstack([
            dat["median"] - dat["q25"],
            dat["q75"] - dat["median"]
        ])

        ax.errorbar(
            x=dat["median"],
            y=y + (i - 1) * offset,
            xerr=xerr,
            fmt=markers[src],
            color=source_colors[src],
            markersize=8,
            capsize=2,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="none",
            label=src
        )

        annotate_small_n_horizontal(
            ax=ax,
            y_positions=y + (i - 1) * offset,
            upper_bounds=dat["q75"],
            counts=dat["n"]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(districts_ordered)
    ax.invert_yaxis()

    ax.set_xlabel("Median Daily Patient Load per HCW")
    ax.set_ylabel("District or Central Hospital")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.grid(axis="x", which="major", alpha=0.5)
    ax.grid(axis="x", which="minor", alpha=0.25, linestyle=":")

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    # Plot 3: over all districts
    # x-axis = facility level, y-axis = patient load

    summary = summarise_median_iqr(
        df_plot,
        ["Facility_Level", "Source"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(facility_levels))
    offset = 0.12

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
            .set_index("Facility_Level")
            .reindex(facility_levels)
        )

        y = dat["median"]

        yerr = np.vstack([
            dat["median"] - dat["q25"],
            dat["q75"] - dat["median"]
        ])

        ax.errorbar(
            x=x + (i - 1) * offset,
            y=y,
            yerr=yerr,
            fmt=markers[src],
            color=source_colors[src],
            markersize=8,
            capsize=2,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="none",
            label=src
        )

        annotate_small_n_vertical(
            ax=ax,
            x_positions=x + (i - 1) * offset,
            upper_bounds=dat["q75"],
            counts=dat["n"]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(facility_levels)

    ax.grid(axis="y", which="major", alpha=0.5)
    ax.grid(axis="y", which="minor", alpha=0.25, linestyle=":")

    ax.set_xlabel("Facility Level")
    ax.set_ylabel("Median Daily Patient Load per HCW")

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Plot 4: over all districts and facility levels
    # x-axis = source, y-axis = patient load

    colors = [source_colors[src] for src in sources]

    fig, ax = plt.subplots(figsize=(7, 6))

    box_data = [
        df_plot.loc[
            df_plot["Source"] == src,
            patient_load_col
        ].dropna()
        for src in sources
    ]

    box = ax.boxplot(
        box_data,
        positions=np.arange(len(sources)),
        widths=0.7,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    summary = (
        df_plot
        .groupby("Source")[patient_load_col]
        .agg(
            n="count",
            q75=lambda x: x.quantile(0.75)
        )
        .reindex(sources)
        .reset_index()
    )

    annotate_small_n_vertical(
        ax=ax,
        x_positions=np.arange(len(sources)),
        upper_bounds=summary["q75"],
        counts=summary["n"]
    )

    ax.set_xticks(np.arange(len(sources)))
    ax.set_xticklabels(sources)

    ax.grid(axis="y", which="major", alpha=0.5)
    ax.grid(axis="y", which="minor", alpha=0.25, linestyle=":")

    ax.set_xlabel("Source")
    ax.set_ylabel("Median Daily Patient Load per HCW")

    plt.tight_layout()
    plt.show()

    # ** mean + 95%CI plots **

    def summarise_mean_ci95(data, group_cols):
        summary = (
            data
            .groupby(group_cols)[patient_load_col]
            .agg(
                mean="mean",
                sd="std",
                n="count"
            )
            .reset_index()
        )

        summary["se"] = summary["sd"] / np.sqrt(summary["n"])
        summary["ci95"] = 1.96 * summary["se"]

        summary["ci95"] = summary["ci95"].fillna(0)

        return summary

    # Plot 1: by facility level and district
    # y-axis = district, x-axis = patient load

    summary = summarise_mean_ci95(
        df_plot,
        ["District", "Facility_Level", "Source"]
    )

    offset = 0.2

    fig, axes = plt.subplots(
        1,
        len(facility_levels),
        figsize=(20, 12),
        sharex="all"
    )

    for ax, fac_level in zip(axes, facility_levels):

        temp = summary[
            summary["Facility_Level"] == fac_level
            ].copy()

        districts = sorted(temp["District"].unique())
        y = np.arange(len(districts))

        for i, src in enumerate(sources):
            dat = (
                temp[temp["Source"] == src]
                .set_index("District")
                .reindex(districts)
            )

            lower_err = np.minimum(dat["ci95"], dat["mean"])
            upper_err = dat["ci95"]

            xerr = np.vstack([
                lower_err,
                upper_err
            ])

            ax.errorbar(
                x=dat["mean"],
                y=y + (i - 1) * offset,
                xerr=xerr,
                fmt=markers[src],
                color=source_colors[src],
                markersize=8,
                elinewidth=1.5,
                capsize=2,
                capthick=1.5,
                linestyle="none",
                label=src
            )

            annotate_small_n_horizontal(
                ax=ax,
                y_positions=y + (i - 1) * offset,
                upper_bounds=dat["mean"] + dat["ci95"],
                counts=dat["n"]
            )

        ax.set_title(f"Facility Level {fac_level}")

        ax.set_yticks(y)
        ax.set_yticklabels(districts)
        ax.invert_yaxis()

        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.grid(axis="x", which="major", alpha=0.5)
        ax.grid(axis="x", which="minor", alpha=0.25, linestyle=":")

        ax.tick_params(axis="both")
        ax.tick_params(axis="x", labelrotation=90)

    fig.supxlabel(
        "Mean Daily Patient Load per HCW",
        y=0.04
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    # Plot 2: overall facility levels
    # y-axis = district, x-axis = patient load

    summary = summarise_mean_ci95(
        df_plot,
        ["District", "Source"]
    )

    fig, ax = plt.subplots(figsize=(10, 12))

    districts_ordered = (
        summary
        .loc[summary["Source"] == "HCW TMS"]
        .sort_values("mean", ascending=False)["District"]
        .tolist()
    )
    y = np.arange(len(districts_ordered))

    offset = 0.2

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
            .set_index("District")
            .reindex(districts_ordered)
        )

        lower_err = np.minimum(dat["ci95"], dat["mean"])
        upper_err = dat["ci95"]

        xerr = np.vstack([
            lower_err,
            upper_err
        ])

        ax.errorbar(
            x=dat["mean"],
            y=y + (i - 1) * offset,
            xerr=xerr,
            fmt=markers[src],
            color=source_colors[src],
            markersize=8,
            capsize=2,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="none",
            label=src
        )

        annotate_small_n_horizontal(
            ax=ax,
            y_positions=y + (i - 1) * offset,
            upper_bounds=dat["mean"] + dat["ci95"],
            counts=dat["n"]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(districts_ordered)
    ax.invert_yaxis()

    ax.set_xlabel("Mean Daily Patient Load per HCW")
    ax.set_ylabel("District or Central Hospital")

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.grid(axis="x", which="major", alpha=0.5)
    ax.grid(axis="x", which="minor", alpha=0.25, linestyle=":")

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    # Plot 3: over all districts
    # x-axis = facility level, y-axis = patient load

    summary = summarise_mean_ci95(
        df_plot,
        ["Facility_Level", "Source"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(facility_levels))
    offset = 0.12

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
            .set_index("Facility_Level")
            .reindex(facility_levels)
        )

        y = dat["mean"]

        lower_err = np.minimum(dat["ci95"], dat["mean"])
        upper_err = dat["ci95"]

        yerr = np.vstack([
            lower_err,
            upper_err
        ])

        ax.errorbar(
            x=x + (i - 1) * offset,
            y=y,
            yerr=yerr,
            fmt=markers[src],
            color=source_colors[src],
            markersize=8,
            capsize=2,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="none",
            label=src
        )

        annotate_small_n_vertical(
            ax=ax,
            x_positions=x + (i - 1) * offset,
            upper_bounds=dat["mean"] + dat["ci95"],
            counts=dat["n"]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(facility_levels)

    ax.grid(axis="y", which="major", alpha=0.5)
    ax.grid(axis="y", which="minor", alpha=0.25, linestyle=":")

    ax.set_xlabel("Facility Level")
    ax.set_ylabel("Mean Daily Patient Load per HCW")

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Plot 4: over all districts and facility levels
    # x-axis = source, y-axis = patient load

    summary = (
        summarise_mean_ci95(
            df_plot,
            ["Source"]
        )
        .set_index("Source")
        .reindex(sources)
        .reset_index()
    )

    lower_err = np.minimum(summary["ci95"], summary["mean"])
    upper_err = summary["ci95"]

    yerr = np.vstack([
        lower_err,
        upper_err
    ])

    colors = [source_colors[src] for src in sources]

    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(
        x=np.arange(len(sources)),
        height=summary["mean"],
        yerr=yerr,
        capsize=5,
        color=colors,
        alpha=0.7,
        width=0.7
    )

    annotate_small_n_vertical(
        ax=ax,
        x_positions=np.arange(len(sources)),
        upper_bounds=summary["mean"] + summary["ci95"],
        counts=summary["n"]
    )

    ax.set_xticks(np.arange(len(sources)))
    ax.set_xticklabels(sources)

    ax.grid(axis="y", which="major", alpha=0.5)
    ax.grid(axis="y", which="minor", alpha=0.25, linestyle=":")

    ax.set_xlabel("Source")
    ax.set_ylabel("Mean Daily Patient Load per HCW")

    plt.tight_layout()
    plt.show()

    # hcw_count = extract_results(
    #     results_folder,
    #     module="tlo.methods.healthsystem.summary",
    #     key="number_of_hcw_staff",
    #     custom_generate_series=get_hcw_count,
    #     do_scaling=False
    # )
    # hcw_count = hcw_count[(0, 0)]
    #
    # patient_volume = extract_results(
    #     results_folder,
    #     module="tlo.methods.healthsystem",
    #     key="HSI_Event",
    #     custom_generate_series=get_patient_count,
    #     do_scaling=True
    # )

    # patient_volume_opd = extract_results(
    #     results_folder,
    #     module="tlo.methods.healthsystem",
    #     key="HSI_Event",
    #     custom_generate_series=get_patient_counts_in_opd,
    #     do_scaling=True
    # )
    #
    # patient_volume_emerg = extract_results(
    #     results_folder,
    #     module="tlo.methods.healthsystem",
    #     key="HSI_Event",
    #     custom_generate_series=get_patient_counts_in_emerg,
    #     do_scaling=True
    # )

    # patient_load_per_hcw_in_opd = patient_volume_opd / hcw_count.values[0]
    # patient_load_per_hcw_in_emerg = patient_volume_emerg / hcw_count.values[0]
    #
    # opd = patient_load_per_hcw_in_opd[(0, 0)].rename("Outpatient care")
    # emerg = patient_load_per_hcw_in_emerg[(0, 0)].rename("Emergency/Intensive care")
    #
    # patient_load_per_hcw = pd.concat([opd, emerg], axis=1)

    # patient_load_per_hcw = patient_volume / hcw_count.values[0]
    # patient_load_per_hcw = patient_load_per_hcw.fillna(0)
    #
    # patient_load_per_hcw = (
    #     patient_load_per_hcw
    #     .reset_index()
    #     .melt(
    #         id_vars="date",
    #         var_name="coarse_loc_cat_str",
    #         value_name="daily_patient_load_per_hcw"
    #     )
    # )
    # patient_load_per_hcw["coarse_loc_cat_str"] = "All service area"
    #
    # path_to_tlm_folder = (
    #     resourcefilepath
    #     / "healthsystem"
    #     / "human_resources"
    #     / "TLM_2024"
    # )
    #
    # patient_load_per_hcw.to_stata(
    #     path_to_tlm_folder / "tlo_pat_load.dta",
    #     write_index=False,
    #     convert_dates={"date": "td"}
    # )

    # issue: do not know how many staff members are in outpatient, emergency care, and inpatient


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


