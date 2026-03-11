import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
    _standardize_short_treatment_id,
    # DON'T import bin_hsi_event_details from utils
    compute_mean_across_runs,
    extract_results,
    get_coarse_appt_type,
    get_color_short_treatment_id,
    load_pickled_dataframes,
    order_of_short_treatment_ids,
    plot_stacked_bar_chart,
    squarify_neat,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)
import re
from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2034, 12, 31))


def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def figure4_hr_use_overall(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 4': The level of usage of the HealthSystem HR Resources """

    make_graph_file_name = lambda stub: output_folder / f"Fig4_{stub}.png"  # noqa: E731

    def get_share_of_time_for_hw_in_each_facility_by_short_treatment_id(_df):

        _df = drop_outside_period(_df)
        _df = _df.set_index("date")

        nurse_cols = [
            c for c in _df.columns
            if "Officer_Nursing_and_Midwifery" in c
        ]

        if len(nurse_cols) == 0:
            return None

        nurse_df = _df[nurse_cols]

        # Mean usage across all nurse facilities
        nurse_df = nurse_df.copy()
        nurse_df.loc[:, "All"] = nurse_df.mean(axis=1)
        # nurse_df["All"] = nurse_df.mean(axis=1)

        return nurse_df.resample("M").mean().stack()

    def get_share_of_time_used_for_each_officer_at_each_level(_df):

        _df = drop_outside_period(_df)
        _df = _df.set_index("date")

        # Columns look like:
        # clinic=GenericClinic|facID_and_officer=FacilityID_0_Officer_Nursing_and_Midwifery

        officer_cols = [
            c for c in _df.columns if "FacilityID_" in c and "Officer_" in c
        ]

        if len(officer_cols) == 0:
            return None

        officer_df = _df[officer_cols].copy()

        # Load Master Facility List
        mfl = pd.read_csv(
            Path("./resources/healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")
        ).set_index("Facility_ID")

        results = []

        for col in officer_cols:

            col_string = str(col)

            # Extract facility ID
            fac_match = re.search(r'FacilityID_(\d+)', col_string)
            if fac_match is None:
                continue
            fid = int(fac_match.group(1))

            # Extract cadre
            officer_match = re.search(r'Officer_(.*)', col_string)
            if officer_match is None:
                continue
            cadre = officer_match.group(1)

            # Get facility level
            if fid not in mfl.index:
                continue

            level = mfl.loc[fid, "Facility_Level"]
            level = "2" if level == "1b" else level

            # Compute mean usage
            mean_val = officer_df[col].mean()

            results.append((cadre, level, mean_val))

        if len(results) == 0:
            return None

        result_df = pd.DataFrame(results, columns=["Cadre", "Facility_Level", "Usage"])

        return result_df.groupby(["Cadre", "Facility_Level"])["Usage"].mean()

    capacity_by_facility = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Capacity_By_FacID_and_Officer',
            custom_generate_series=get_share_of_time_for_hw_in_each_facility_by_short_treatment_id,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    capacity_by_officer = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Capacity_By_FacID_and_Officer',
            custom_generate_series=get_share_of_time_used_for_each_officer_at_each_level,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    # Find the levels of each facility
    mfl = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv'
    ).set_index('Facility_ID')

    def find_level_for_facility(col_name):
        # Skip aggregated column
        if col_name == "All":
            return None

        match = re.search(r'FacilityID_(\d+)', str(col_name))

        if match is None:
            return None

        fid = int(match.group(1))

        level = mfl.loc[fid, "Facility_Level"]

        return "2" if level == "1b" else level

    # def find_level_for_facility(col_tuple):
    #     # Extract the text part
    #     col_string = col_tuple[2]
    #
    #     # Extract facility ID number
    #     match = re.search(r'FacilityID_(\d+)', col_string)
    #     fid = int(match.group(1))
    #
    #     level = mfl.loc[fid, "Facility_Level"]
    #     return "2" if level == "1b" else level
    # def find_level_for_facility(id):
    #     return mfl.loc[id].Facility_Level if mfl.loc[id].Facility_Level != '1b' else '2'
    # def find_level_for_facility(fid):
    #     level = mfl.loc[fid, "Facility_Level"]
    #     return "2" if level == "1b" else level

    color_for_level = {'0': 'blue', '1a': 'yellow', '1b': 'green', '2': 'grey', '3': 'orange', '4': 'black',
                       '5': 'white'}

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time By Month'
    capacity_unstacked = capacity_by_facility.unstack()
    for i in capacity_unstacked.columns:

        level = find_level_for_facility(i)

        if level is None:
            continue

        h1, = ax.plot(
            capacity_unstacked[i].index,
            capacity_unstacked[i].values,
            color=color_for_level[level],
            linewidth=0.5,
            label=f'Facility_Level {level}'
        )
    # for i in capacity_unstacked.columns:
    #     if i != 'All':
    #         level = find_level_for_facility(i)
    #         h1, = ax.plot(capacity_unstacked[i].index, capacity_unstacked[i].values,
    #                       color=color_for_level[level], linewidth=0.5, label=f'Facility_Level {level}')

    if 'All' in capacity_unstacked.columns:
        h2, = ax.plot(
            capacity_unstacked['All'].index,
            capacity_unstacked['All'].values,
            color='red',
            linewidth=1.5
        )
        ax.legend([h1, h2], ['Each Facility', 'All Facilities'])
    else:
        ax.legend([h1], ['Each Facility'])

    ax.set_title(name_of_plot)
    ax.set_xlabel('Month')
    ax.set_ylabel('Fraction of all time used\n(Average for the month)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time (Average)'
    capacity_unstacked_average = capacity_by_facility.unstack().mean()
    # levels = [find_level_for_facility(i) if i != 'All' else 'All' for i in capacity_unstacked_average.index]
    xpos_for_level = dict(zip((color_for_level.keys()), range(len(color_for_level))))
    xpos_for_level.update({'1b': 2, '2': 2, '3': 3, '4': 4, '5': 5})
    for id, val in capacity_unstacked_average.items():
        if id != 'All':
            _level = find_level_for_facility(id)

            # Skip if facility level could not be determined
            if _level is None:
                continue

            if _level != '5':
                xpos = xpos_for_level[_level]
                scatter = (np.random.rand() - 0.5) * 0.25
                h1, = ax.plot(xpos + scatter, val * 100, color=color_for_level[_level],
                              marker='.', markersize=15, label='Each Facility', linestyle='none')
    if 'All' in capacity_unstacked_average.index:
        h2 = ax.axhline(
            y=capacity_unstacked_average['All'] * 100,
            color='red',
            linestyle='--',
            label='Average'
        )
        ax.set_title(name_of_plot)
        ax.set_xlabel('Facility_Level')
        ax.set_xticks(list(xpos_for_level.values()))
        ax.set_xticklabels(xpos_for_level.keys())
        ax.set_ylabel('Percent of Time Available That is Used\n')
        ax.legend(handles=[h1, h2])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        plt.close(fig)

        fig, ax = plt.subplots()
        name_of_plot = 'Usage of Healthcare Worker Time by Cadre and Facility_Level'
        (100.0 * capacity_by_officer.unstack()).T.plot.bar(ax=ax)
        ax.legend()
        ax.set_xlabel('Facility_Level')
        ax.set_ylabel('Percent of time that is used')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(name_of_plot)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        plt.close(fig)


def get_yearly_hr_count(_df):

    if 'GenericClinic' not in _df.columns:
        return None

    _df['year'] = _df['date'].dt.year

    # Expand facility dictionary
    staff_df = _df['GenericClinic'].apply(pd.Series)

    # Extract cadre names
    staff_df.columns = [c.split('Officer_')[-1] for c in staff_df.columns]

    # Sum facilities within cadre
    staff_df = staff_df.groupby(level=0, axis=1).sum()

    # Add year
    staff_df['year'] = _df['year']

    # Sum within year
    staff_df = staff_df.groupby('year').sum()
    POP_SCALE = 145.39609
    # POP_SCALE = 1000
    staff_df = staff_df * POP_SCALE

    # Convert to stacked series (year,cadre → value)
    return staff_df.stack()


def extract_staff_counts(results_folder):
    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_yearly_hr_count,
        do_scaling=False
    )


def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def plot_staff_counts_by_cadre_across_scenarios(staff_counts_summary, output_folder):
    scenario_names = staff_counts_summary.columns.get_level_values(0).unique()
    cadres = staff_counts_summary.index.get_level_values(1).unique()

    for cadre in cadres:

        fig, ax = plt.subplots()

        for scenario in scenario_names:

            central = staff_counts_summary[(scenario, "mean")].xs(cadre, level=1)
            lower = staff_counts_summary[(scenario, "lower")].xs(cadre, level=1)
            upper = staff_counts_summary[(scenario, "upper")].xs(cadre, level=1)

            years = central.index

            ax.plot(
                years,
                central.values,
                label=scenario
            )

            # ax.fill_between(
            #     years,
            #     lower.values,
            #     upper.values,
            #     alpha=0.25
            # )
            ax.fill_between(
                years,
                np.maximum(lower.values, 0),
                upper.values,
                alpha=0.25
            )

        ax.set_title(f"{cadre} Staff Counts Across Scenarios")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Health Workers")

        ax.legend()

        fig.tight_layout()

        fig.savefig(output_folder / f"{cadre}_staff_counts_across_scenarios.png")

        plt.close(fig)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Description of the usage of healthcare system resources."""

    # figure2_appointments_used(
    #     results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    # )
    log = load_pickled_dataframes(results_folder, 0, 0)
    print(log.keys())

    print(log['tlo.methods.healthsystem.summary'].keys())

    # STEP 1: extract staff counts
    staff_counts = extract_staff_counts(results_folder)

    # STEP 2: rename draws to scenario names
    param_names = tuple(StaffingScenario()._scenarios.keys())

    staff_counts = staff_counts.pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names
    )

    # STEP 3: summarize runs
    print(type(staff_counts))
    print(staff_counts.head())
    staff_counts_summary = summarize(staff_counts)

    print("\n=== Staff counts from 2025–2034 ===")

    # Select years 2025–2034
    years_to_check = range(2025, 2035)

    export_df = staff_counts_summary.reset_index()

    # Filter the years
    export_df = export_df[export_df["year"].isin(years_to_check)]

    # Save to Excel
    export_path = output_folder / "debug_staff_counts_2025_2034.xlsx"
    export_df.to_excel(export_path)

    print(f"Staff counts exported to: {export_path}")

    # STEP 4: plot
    plot_staff_counts_by_cadre_across_scenarios(staff_counts_summary, output_folder)

    figure4_hr_use_overall(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save figures",
    )
    args = parser.parse_args()

    # Use the command-line argument instead of hardcoded path
    results_folder = args.scenario_outputs_folder
    # results_folder = Path(
    #     './outputs/wamulwafu@kuhes.ac.mw/nurses_scenario_outputs-2026-02-09T110530Z'
    # )

    apply(
        results_folder=results_folder,  # or directly: args.scenario_outputs_folder
        output_folder=results_folder,
        resourcefilepath=Path('./resources')
    )
