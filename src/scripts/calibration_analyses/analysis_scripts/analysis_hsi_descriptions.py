import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    COARSE_APPT_TYPE_TO_COLOR_MAP,
    bin_hsi_event_details,
    compute_mean_across_runs,
    extract_results,
    get_coarse_appt_type,
    get_color_short_treatment_id,
    order_of_short_treatment_ids,
    plot_stacked_bar_chart,
    squarify_neat,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))


def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def formatting_hsi_df(_df):
    """Standard formatting for the HSI_Event log."""
    _df = _df.pipe(drop_outside_period) \
        .drop(_df.index[~_df.did_run]) \
        .reset_index(drop=True) \
        .drop(columns=['Person_ID', 'Squeeze_Factor', 'Facility_ID', 'did_run'])

    # Unpack the dictionary in `Number_By_Appt_Type_Code`.
    _df = _df.join(_df['Number_By_Appt_Type_Code'].apply(pd.Series).fillna(0.0)).drop(
        columns='Number_By_Appt_Type_Code')

    # Produce coarse version of TREATMENT_ID (just first level, which is the module)
    _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])

    return _df


def figure1_distribution_of_hsi_event_by_treatment_id(results_folder: Path, output_folder: Path,
                                                      resourcefilepath: Path):
    """ 'Figure 1': The Distribution of HSI_Events that occur by TREATMENT_ID.
    N.B. This uses the summary logger for speed. All other figures use the full logger as that is necessary."""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig1_{stub}.png"  # noqa: E731

    def get_counts_of_hsi_by_treatment_id(_df):
        """Get the counts of the short TREATMENT_IDs occurring"""
        _counts_by_treatment_id = _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)
        return _counts_by_treatment_id.groupby(level=0).sum()

    def get_counts_of_hsi_by_short_treatment_id(_df):
        """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
        _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()

    counts_of_hsi_by_treatment_id = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    )

    counts_of_hsi_by_treatment_id_short = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    )

    fig, ax = plt.subplots()
    name_of_plot = 'Proportion of HSI Events by TREATMENT_ID'
    squarify.plot(
        sizes=counts_of_hsi_by_treatment_id.values,
        label=counts_of_hsi_by_treatment_id.index,
        alpha=1,
        pad=True,
        ax=ax,
        text_kwargs={'color': 'black', 'size': 8},
    )
    ax.set_axis_off()
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'HSI Events by TREATMENT_ID (Short)'
    squarify_neat(
        sizes=counts_of_hsi_by_treatment_id_short.values,
        label=counts_of_hsi_by_treatment_id_short.index,
        colormap=get_color_short_treatment_id,
        alpha=1,
        pad=True,
        ax=ax,
        text_kwargs={'color': 'black', 'size': 8}
    )
    ax.set_axis_off()
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def figure2_appointments_used(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 2': The Appointments Used"""
    # Get counts of number of HSI events run for each treatment ID and coarse
    # appointment type pair, taking mean across scenario runs and scaling by population
    # scale factor
    counts_by_treatment_id_and_coarse_appt_type = compute_mean_across_runs(
        bin_hsi_event_details(
            results_folder,
            lambda event_details, count: sum(
                [
                    Counter({
                        (
                            event_details["treatment_id"].split("_")[0],
                            get_coarse_appt_type(appt_type)
                        ):
                        count * appt_number
                    })
                    for appt_type, appt_number in event_details["appt_footprint"]
                ],
                Counter()
            ),
            *TARGET_PERIOD,
            True
        )
    )[0]
    name_of_plot = 'Appointment Types Used'
    fig, ax = plt.subplots()
    plot_stacked_bar_chart(
        ax,
        counts_by_treatment_id_and_coarse_appt_type,
        COARSE_APPT_TYPE_TO_COLOR_MAP,
        count_scale=1e-6
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(ncol=2, prop={'size': 8}, loc='upper left')
    ax.set_ylabel('Number of appointments (millions)')
    ax.set_xlabel('TREATMENT_ID (Short)')
    ax.set_ylim(0, 80)
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.tight_layout()
    fig.savefig(
        output_folder
        / f"{PREFIX_ON_FILENAME}_Fig2_{name_of_plot.replace(' ', '_')}.png"
    )
    plt.close(fig)


def figure3_fraction_of_time_of_hcw_used_by_treatment(results_folder: Path, output_folder: Path,
                                                      resourcefilepath: Path):
    """ 'Figure 3': The Fraction of the time of each HCW used by each TREATMENT_ID (Short)"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig3_{stub}.png"  # noqa: E731

    appointment_time_table = pd.read_csv(
        resourcefilepath
        / 'healthsystem'
        / 'human_resources'
        / 'definitions'
        / 'ResourceFile_Appt_Time_Table.csv',
        index_col=["Appt_Type_Code", "Facility_Level", "Officer_Category"]
    )

    appt_type_facility_level_officer_category_to_appt_time = (
        appointment_time_table.Time_Taken_Mins.to_dict()
    )

    officer_categories = appointment_time_table.index.levels[
        appointment_time_table.index.names.index("Officer_Category")
    ].to_list()

    times_by_officer_category_treatment_id_per_run = bin_hsi_event_details(
        results_folder,
        lambda event_details, count: sum(
            [
                Counter({
                    (
                        officer_category,
                        event_details["treatment_id"].split("_")[0]
                    ):
                    count
                    * appt_number
                    * appt_type_facility_level_officer_category_to_appt_time.get(
                        (
                            appt_type,
                            event_details["facility_level"],
                            officer_category
                        ),
                        0
                    )
                    for officer_category in officer_categories
                })
                for appt_type, appt_number in event_details["appt_footprint"]
            ],
            Counter()
        ),
        *TARGET_PERIOD,
        False
    )

    proportions_per_treatment_id_by_officer_category_for_all_runs = defaultdict(list)

    for _, times_by_officer_category_treatment_id in (
        times_by_officer_category_treatment_id_per_run.items()
    ):
        total_times_by_officer_category = Counter()
        times_per_treatment_id_by_officer_category = defaultdict(dict)
        for (cat, treatment_id), time in times_by_officer_category_treatment_id.items():
            total_times_by_officer_category[cat] += time
            times_per_treatment_id_by_officer_category[cat][treatment_id] = time
        for cat, times_per_treatment_id in (
            times_per_treatment_id_by_officer_category.items()
        ):
            proportions_per_treatment_id_by_officer_category_for_all_runs[cat].append(
                Counter(
                    {
                        treatment_id: time / total_times_by_officer_category[cat]
                        for treatment_id, time in times_per_treatment_id.items()
                    }
                )
            )

    def mean_of_counters(counters):
        sum_counters = sum(counters, Counter())
        len_counters = len(counters)
        return {key: value / len_counters for key, value in sum_counters.items()}

    proportions_per_treatment_id_by_officer_category = {
        officer_category: mean_of_counters(proportions_per_treatment_id_for_all_runs)
        for officer_category, proportions_per_treatment_id_for_all_runs
        in proportions_per_treatment_id_by_officer_category_for_all_runs.items()
    }

    for officer_category, proportions_per_treatment_id in (
        proportions_per_treatment_id_by_officer_category.items()
    ):
        proportions_per_treatment_id_by_officer_category[officer_category] = dict(
            sorted(
                proportions_per_treatment_id.items(),
                key=lambda key_value: order_of_short_treatment_ids(key_value[0])
            )
        )

    cadres_to_plot = ['DCSA', 'Nursing_and_Midwifery', 'Clinical', 'Pharmacy']

    fig, ax = plt.subplots(nrows=2, ncols=2)
    name_of_plot = 'Proportion of Time Used For Selected Cadre by TREATMENT_ID (Short)'
    for cadre, ax in zip(cadres_to_plot, ax.flat):
        p_by_treatment_id = proportions_per_treatment_id_by_officer_category[cadre]
        squarify_neat(
            sizes=list(p_by_treatment_id.values()),
            label=list(p_by_treatment_id.keys()),
            colormap=get_color_short_treatment_id,
            numlabels=4,
            alpha=1,
            pad=True,
            ax=ax,
            text_kwargs={'color': 'black', 'size': 8},
        )
        ax.set_axis_off()
        ax.set_title(f'{cadre}', {'size': 10, 'color': 'black'})
    fig.suptitle(name_of_plot, fontproperties={'size': 12})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def figure4_hr_use_overall(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 4': The level of usage of the HealthSystem HR Resources """

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig4_{stub}.png"  # noqa: E731

    def get_share_of_time_for_hw_in_each_facility_by_short_treatment_id(_df):
        _df = drop_outside_period(_df)
        _df = _df.set_index('date')
        _all = _df['Frac_Time_Used_Overall']
        _df = _df['Frac_Time_Used_By_Facility_ID'].apply(pd.Series)
        _df.columns = _df.columns.astype(int)
        _df = _df.reindex(columns=sorted(_df.columns))
        _df['All'] = _all
        return _df.groupby(pd.Grouper(freq="M")).mean().stack()  # find monthly averages and stack into series

    def get_share_of_time_used_for_each_officer_at_each_level(_df):
        _df = drop_outside_period(_df)
        _df = _df.set_index('date')
        _df = _df['Frac_Time_Used_By_OfficerType'].apply(pd.Series).mean()  # find mean over the period
        _df.index = unflatten_flattened_multi_index_in_logging(_df.index)
        return _df

    capacity_by_facility = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Capacity',
            custom_generate_series=get_share_of_time_for_hw_in_each_facility_by_short_treatment_id,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    capacity_by_officer = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Capacity',
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

    def find_level_for_facility(id):
        return mfl.loc[id].Facility_Level

    color_for_level = {'0': 'blue', '1a': 'yellow', '1b': 'green', '2': 'grey', '3': 'orange', '4': 'black',
                       '5': 'white'}

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time By Month'
    capacity_unstacked = capacity_by_facility.unstack()
    for i in capacity_unstacked.columns:
        if i != 'All':
            level = find_level_for_facility(i)
            h1, = ax.plot(capacity_unstacked[i].index, capacity_unstacked[i].values,
                          color=color_for_level[level], linewidth=0.5, label=f'Facility_Level {level}')

    h2, = ax.plot(capacity_unstacked['All'].index, capacity_unstacked['All'].values, color='red', linewidth=1.5)
    ax.set_title(name_of_plot)
    ax.set_xlabel('Month')
    ax.set_ylabel('Fraction of all time used\n(Average for the month)')
    ax.legend([h1, h2], ['Each Facility', 'All Facilities'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Healthcare Worker Time (Average)'
    capacity_unstacked_average = capacity_by_facility.unstack().mean()
    # levels = [find_level_for_facility(i) if i != 'All' else 'All' for i in capacity_unstacked_average.index]
    xpos_for_level = dict(zip((color_for_level.keys()), range(len(color_for_level))))
    for id, val in capacity_unstacked_average.iteritems():
        if id != 'All':
            _level = find_level_for_facility(id)
            if _level != '5':
                xpos = xpos_for_level[_level]
                scatter = (np.random.rand() - 0.5) * 0.25
                h1, = ax.plot(xpos + scatter, val * 100, color=color_for_level[_level],
                              marker='.', markersize=15, label='Each Facility', linestyle='none')
    h2 = ax.axhline(y=capacity_unstacked_average['All'] * 100,
                    color='red', linestyle='--', label='Average')
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
    fig.show()
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
    fig.show()
    plt.close(fig)


def figure5_bed_use(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 5': The level of usage of the Beds in the HealthSystem"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig5_{stub}.png"  # noqa: E731

    def get_frac_of_beddays_used(_df):
        _df = drop_outside_period(_df)
        _df = _df.set_index('date')
        return _df.mean()

    frac_of_beddays_used = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='FractionOfBedDaysUsed',
            custom_generate_series=get_frac_of_beddays_used,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True
    )

    fig, ax = plt.subplots()
    name_of_plot = 'Usage of Bed-Days Used'
    (100.0 * frac_of_beddays_used).plot.bar(ax=ax)
    ax.legend()
    ax.set_xlabel('Type of Bed')
    ax.set_ylabel('Percent of time that is used')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(name_of_plot)
    ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def figure6_cons_use(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 6': Usage of consumables in the HealthSystem"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig6_{stub}.png"  # noqa: E731

    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)

        counts_of_available = defaultdict(int)
        counts_of_not_available = defaultdict(int)

        for _, row in _df.iterrows():
            for item, num in eval(row['Item_Available']).items():
                counts_of_available[item] += num
            for item, num in eval(row['Item_NotAvailable']).items():
                counts_of_not_available[item] += num

        return pd.concat(
            {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
            axis=1
        ).fillna(0).astype(int).stack()

    cons_req = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True
    )

    # Merge in item names and prepare to plot:
    cons = cons_req.unstack()
    cons_names = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv'
    )[['Item_Code', 'Items']].set_index('Item_Code').drop_duplicates()
    cons = cons.merge(cons_names, left_index=True, right_index=True, how='left').set_index('Items').astype(int)
    cons = cons.assign(total=cons.sum(1)).sort_values('total').drop(columns='total')

    fig, ax = plt.subplots()
    name_of_plot = 'Demand For Consumables'
    (cons / 1e6).head(20).plot.barh(ax=ax, stacked=True)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Item (20 most requested)')
    ax.set_xlabel('Number of requests (Millions)')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Consumables Not Available'
    (cons['Not_Available'] / 1e6).sort_values().head(20).plot.barh(ax=ax)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Item (20 most frequently not available when requested)')
    ax.set_xlabel('Number of requests (Millions)')
    ax.yaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # HSI affected by missing consumables

    def get_treatment_id_affecting_by_missing_consumables(_df):
        """Return frequency that a (short) TREATMENT_ID suffers from consumables not being available."""
        _df = drop_outside_period(_df)
        _df = _df.loc[(_df['Item_NotAvailable'] != '{}'), ['TREATMENT_ID', 'Item_NotAvailable']]
        _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].map(lambda x: x.split('_')[0])
        return _df['TREATMENT_ID_SHORT'].value_counts()

    treatment_id_affecting_by_missing_consumables = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='Consumables',
            custom_generate_series=get_treatment_id_affecting_by_missing_consumables,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True
    )

    fig, ax = plt.subplots()
    name_of_plot = 'HSI Affected by Unavailable Consumables (by Short TREATMENT_ID)'
    squarify_neat(
        sizes=treatment_id_affecting_by_missing_consumables.values,
        label=treatment_id_affecting_by_missing_consumables.index,
        colormap=get_color_short_treatment_id,
        alpha=1,
        ax=ax,
        text_kwargs={'color': 'black', 'size': 8}
    )
    ax.set_axis_off()
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Description of the usage of healthcare system resources."""

    figure1_distribution_of_hsi_event_by_treatment_id(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )

    figure2_appointments_used(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )

    figure3_fraction_of_time_of_hcw_used_by_treatment(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )

    figure4_hr_use_overall(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )

    figure5_bed_use(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )

    figure6_cons_use(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Analyse logged HSI event data from scenario run")
    parser.add_argument(
        "--results-folder",
        type=Path,
        help="Path to folder containing results of scenario to perform analysis for"
    )
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
