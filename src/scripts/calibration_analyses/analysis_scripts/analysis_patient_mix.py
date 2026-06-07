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
        the_hrh_target_period = (Date(2024, 1, 1), Date(2024, 1, 1))
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
    )

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
    )
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
    daily_patient_load_per_hcw = patient_volume_facility_id[["Facility_ID", "Patient_Volume"]].merge(
        hcw_count_facility_id[["Facility_ID", "District", "Facility_Level", "Region", "Staff_Count"]],
        on=["Facility_ID"], how="right")
    # fill NAN entries
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_Level"] == "4", ["Patient_Volume", "District", "Region"]
    ] = [0, "Central Hospitals (Southern)", "Southern"]  # ZMH
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 128, "District"
    ] = "Central Hospitals (Southern)"
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 129, "District"
    ] = "Central Hospitals (Northern)"
    daily_patient_load_per_hcw.loc[
        daily_patient_load_per_hcw["Facility_ID"] == 130, "District"
    ] = "Central Hospitals (Central)"

    # 0.5649 is the prob. that any HCW is on duty on any day (estimates from CHAI data)
    # given TLO assume the same HCWs in the HS
    # alongside, we assume the patient seeking care independently of the availability of HCWs?
    # otherwise, would need to adjust patient volume in the nominator, too; by which, the adjustments may cancel out.
    # daily_patient_load_per_hcw['Daily_Patient_Load_Per_HCW'] = (daily_patient_load_per_hcw['Patient_Volume']
    #                                                             / (daily_patient_load_per_hcw['Staff_Count'] * 0.5649)
    #                                                             )

    daily_patient_load_per_hcw['Daily_Patient_Load_Per_HCW'] = (daily_patient_load_per_hcw['Patient_Volume']
                                                                / (daily_patient_load_per_hcw['Staff_Count'] * 1.0)
                                                                )

    tab = pd.crosstab(
        daily_patient_load_per_hcw["District"],
        daily_patient_load_per_hcw["Facility_Level"],
        dropna=False
    )

    print(tab)

    # read in TLM estimates
    path_to_tlm_folder = (
        resourcefilepath
        / "healthsystem"
        / "human_resources"
        / "TLM_2024"
    )

    hcw_tms_pat_load = pd.read_stata(path_to_tlm_folder/"tool_3_pat_load.dta", convert_categoricals=True)
    fac_tms_pat_load = pd.read_stata(path_to_tlm_folder / "tool_6_pat_load.dta", convert_categoricals=True)

    # check that districts and facility levels in the two tools are a subset of TLO output;
    # district and facility level consistency in the two tools already checked in Stata
    hcw_tms_pat_load["district"] = hcw_tms_pat_load["district"].replace({"Mzuzu": "Mzuzu City"})
    fac_tms_pat_load["district"] = fac_tms_pat_load["district"].replace({"Mzuzu": "Mzuzu City"})

    assert set(hcw_tms_pat_load['district'].unique()).issubset(
        set(daily_patient_load_per_hcw['District'].unique())
    )
    assert set(hcw_tms_pat_load['fac_level'].unique()).issubset(
        set(daily_patient_load_per_hcw['Facility_Level'].unique())
    )

    # merge all three estimates in one dataframe, noting the source and keeping all observations
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

    # Plot 1: by facility level
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

    districts = sorted(summary["District"].unique())
    y = np.arange(len(districts))

    offset = 0.2

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
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
    ax.set_yticklabels(districts)
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
    ax.set_ylabel("Daily Patient Load per HCW")

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

    # Plot 1: by facility level
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

    fig.supxlabel(
        "Daily Patient Load per HCW",
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

    districts = sorted(summary["District"].unique())
    y = np.arange(len(districts))

    offset = 0.2

    for i, src in enumerate(sources):
        dat = (
            summary[summary["Source"] == src]
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
    ax.set_yticklabels(districts)
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
        the_target_period=(Date(2024, 1, 1), Date(2024, 5, 31))
    )


