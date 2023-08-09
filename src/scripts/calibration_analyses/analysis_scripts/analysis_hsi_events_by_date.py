import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_coarse_appt_type,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    load_pickled_dataframes,
    order_of_coarse_appt,
    summarize,
)

PREFIX_ON_FILENAME = '3'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))


def drop_outside_period(_df):
    """Return a dataframe for which the date is within the limits defined by TARGET_PERIOD"""
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

    # values_sum = _df.sum()values_sum=_df.sum()

    # Produce coarse version of TREATMENT_ID (just first level, which is the module)
    _df['TREATMENT_ID_SHORT'] = _df['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])

    return _df


def format_dataframe(result_folder):
    result = load_pickled_dataframes(result_folder, draw=0, run=0, name="tlo.methods.healthsystem")
    print(result['tlo.methods.healthsystem']['HSI_Event'].keys())
    date = result['tlo.methods.healthsystem']['HSI_Event']['date']
    hsi_event = result['tlo.methods.healthsystem']['HSI_Event'] \
        .drop(columns=['Person_ID', 'Squeeze_Factor', 'did_run', 'Facility_ID'])
    # Produce coarse version of TREATMENT_ID (just first level, which is the module)
    hsi_event['TREATMENT_ID_SHORT'] = hsi_event['TREATMENT_ID'].str.split('_').apply(lambda x: x[0])
    return date, hsi_event


def figure1_distribution_of_hsi_event_by_date(results_folder: Path, output_folder: Path):
    """ 'Figure 1': The Distribution of HSI_Events that occur by date."""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig1_{stub}.png"  # noqa: E731

    def _define_facility_ids() -> pd.Series:
        """Define the order of the facility_ids and the color for each.
        Names of colors are selected with reference to: https://matplotlib.org/stable/gallery/color/named_colors.html"""
        return pd.Series({
            '0': 'burlywood',
            '1a': 'cornflowerblue',
            '1b': 'darkseagreen',
            '2': 'darkorange',
            '3': 'orchid',
        })

    def get_color_facility_id(facility_id: str) -> str:
        """Return the colour (as matplotlib string) assigned to this shorted facility_id.
         Returns `np.nan` if facility_id is not recognised."""
        colors = _define_facility_ids()
        if facility_id in colors.index:
            return colors.loc[facility_id]
        else:
            return np.nan

    [date, hsi_events] = format_dataframe(results_folder)

    hsi_events['Facility_Level'].value_counts()
    facility_id = hsi_events.groupby("Facility_Level").groups

    fig, ax = plt.subplots()
    name_of_plot = 'Proportion of HSI Events by date'
    plt.hist(date, bins=365)
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Proportion of HSI Events by Facility level'
    for key, value in facility_id.items():
        date.iloc[value]. \
            value_counts(normalize=True). \
            plot(kind='line', color=get_color_facility_id(key), stacked=True, label=key)
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    ax.legend(ncol=2, prop={'size': 8}, loc='upper left')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def figure2_distribution_of_hsi_event_by_treatment(results_folder: Path, output_folder: Path):
    """ 'Figure 2': Distribution of treatment id by date."""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig2_{stub}.png"  # noqa: E731

    def create_figure(name_of_plot, key_variable, s_variable, plot_kind, colour_kind):
        fig, ax = plt.subplots()
        s_variable.iloc[key_variable].value_counts(). \
            plot(kind=plot_kind, color=get_color_short_treatment_id(colour_kind), stacked=True, label=colour_kind)
        ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
        ax.legend(ncol=2, prop={'size': 8}, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        fig.show()
        plt.close(fig)

    [date, hsi_events] = format_dataframe(results_folder)

    treatment = hsi_events.groupby("TREATMENT_ID_SHORT").groups

    # del treatment['Tb'],treatment['Malaria']
    fig, ax = plt.subplots()
    name_of_plot = 'Proportion of HSI Events by Treatment Type'
    for key, value in treatment.items():
        date.iloc[value]. \
            value_counts(normalize=True). \
            plot(kind='line', color=get_color_short_treatment_id(key), stacked=True, label=key)
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    ax.legend(ncol=2, prop={'size': 8}, loc='upper left')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    plt.ylim(0, 0.05)
    fig.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    name_of_plot = 'Proportion of HSI Events by selected Treatment Type'
    date.iloc[treatment["Tb"]].value_counts(normalize=True). \
        plot(kind='line', color=get_color_short_treatment_id("Tb"), stacked=True, label="Tb")
    date.iloc[treatment["Malaria"]].value_counts(normalize=True). \
        plot(kind='line', color=get_color_short_treatment_id("Malaria"), stacked=True, label="Malaria")
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    ax.legend(ncol=2, prop={'size': 8}, loc='upper left')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    create_figure('Tb over Facility Level', treatment["Tb"], hsi_events['Facility_Level'],
                  'bar', "Tb")
    create_figure('Malaria over Facility Level', treatment["Malaria"], hsi_events['Facility_Level'],
                  'bar', "Malaria")
    create_figure('Tb over Appointment', treatment["Tb"], hsi_events['Number_By_Appt_Type_Code'],
                  'bar', "Tb")
    create_figure('Malaria over Appointment', treatment["Malaria"], hsi_events['Number_By_Appt_Type_Code'],
                  'bar', "Malaria")


def figure3_appointments_used(results_folder: Path, output_folder: Path):
    """ 'Figure 2': The Appointments Used"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig2_{stub}.png"  # noqa: E731

    def get_counts_of_appt_type_by_date(_df):
        return formatting_hsi_df(_df) \
            .drop(columns=['TREATMENT_ID_SHORT', 'TREATMENT_ID', 'Facility_Level']) \
            .melt(id_vars=['date'], var_name='Appt_Type', value_name='Num') \
            .groupby(by=['date', 'Appt_Type'])['Num'].sum()

    counts_of_appt_by_date = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_appt_type_by_date,
            do_scaling=False
        ),
        only_mean=True,
        collapse_columns=True,
    )
    # PLOT TOTALS BY COARSE APPT_TYPE

    counts_of_coarse_appt_by_date = \
        counts_of_appt_by_date.unstack(). \
        groupby(axis=1, by=counts_of_appt_by_date.index.levels[1].map(get_coarse_appt_type)).sum()

    counts_of_coarse_appt_by_date = counts_of_coarse_appt_by_date[
        sorted(counts_of_coarse_appt_by_date.columns, key=order_of_coarse_appt)
    ]

    # del counts_of_coarse_appt_by_date['Con w/ DCSA']
    fig, ax = plt.subplots()
    name_of_plot = 'Appointment Types Used'
    counts_of_coarse_appt_by_date.plot.bar(
        ax=ax, stacked=True,
        color=[get_color_coarse_appt(_appt) for _appt in counts_of_coarse_appt_by_date.columns]
    )
    ax.legend(ncol=2, prop={'size': 8}, loc='upper right')
    ax.set_ylabel('Number of appointments')
    ax.set_xlabel('Date')
    ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def apply(results_folder: Path, output_folder: Path):
    """Description of the usage of healthcare system resources."""

    figure1_distribution_of_hsi_event_by_date(
       results_folder=results_folder, output_folder=output_folder
    )
    figure2_distribution_of_hsi_event_by_treatment(
        results_folder=results_folder, output_folder=output_folder
    )
    figure3_appointments_used(
       results_folder=results_folder, output_folder=output_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
    )
