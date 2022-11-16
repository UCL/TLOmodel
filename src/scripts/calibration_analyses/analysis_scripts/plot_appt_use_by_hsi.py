from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_coarse_appt_type,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    order_of_coarse_appt,
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


def figure2_appointments_used(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ 'Figure 2': The Appointments Used"""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_Fig2_{stub}.png"  # noqa: E731

    def get_counts_of_appt_type_by_treatment_id(_df):
        return formatting_hsi_df(_df) \
            .drop(columns=['date', 'TREATMENT_ID_SHORT', 'Facility_Level']) \
            .melt(id_vars=['TREATMENT_ID'], var_name='Appt_Type', value_name='Num') \
            .groupby(by=['TREATMENT_ID', 'Appt_Type'])['Num'].sum()

    counts_of_appt_by_treatment_id = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem',
            key='HSI_Event',
            custom_generate_series=get_counts_of_appt_type_by_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    )

    counts_of_appt_by_treatment_id = counts_of_appt_by_treatment_id.to_frame().reset_index().rename(
        columns={'mean': 'Count'})

    # rename some appts to be consistent in comparison with real data
    appt_dict = {'Under5OPD': 'OPD',
                 'Over5OPD': 'OPD',
                 'AntenatalFirst': 'AntenatalTotal',
                 'ANCSubsequent': 'AntenatalTotal',
                 'NormalDelivery': 'Delivery',
                 'CompDelivery': 'Delivery',
                 'EstMedCom': 'EstAdult',
                 'EstNonCom': 'EstAdult',
                 'VCTPositive': 'VCTTests',
                 'VCTNegative': 'VCTTests',
                 'DentAccidEmerg': 'DentalAll',
                 'DentSurg': 'DentalAll',
                 'DentU5': 'DentalAll',
                 'DentO5': 'DentalAll',
                 'MentOPD': 'MentalAll',
                 'MentClinic': 'MentalAll'
                 }
    counts_of_appt_by_treatment_id.Appt_Type = counts_of_appt_by_treatment_id.Appt_Type.replace(appt_dict)
    counts_of_appt_by_treatment_id = counts_of_appt_by_treatment_id.groupby(
        ['TREATMENT_ID', 'Appt_Type'])['Count'].sum().reset_index()

    # plot a square plot of appt use by treatment_id for each appt
    # appts to be plot
    appts = ['AccidentsandEmerg', 'AntenatalTotal', 'Csection',
             'Delivery', 'EPI', 'EstAdult', 'FamPlan',
             'IPAdmission', 'InpatientDays', 'MaleCirc',
             'MentalAll', 'NewAdult', 'OPD', 'Peds', 'TBNew',
             'U5Malnutr', 'VCTTests']
    fig, ax = plt.subplots(len(appts), sharey=False)
    name_of_figure = 'Proportion of Appointment Use by TREATMENT_ID per Appointment Type'
    for idx in range(len(appts)):
        df_to_plot = counts_of_appt_by_treatment_id[
            (counts_of_appt_by_treatment_id.Appt_Type == appts[idx]) &
            (counts_of_appt_by_treatment_id.Count > 0)].copy()
        name_of_subplot = appts[idx]
        ax[idx] = squarify.plot(
            sizes=df_to_plot.Count,
            label=df_to_plot.TREATMENT_ID,
            alpha=1,
            pad=True,
            ax=ax[idx],
            text_kwargs={'color': 'black', 'size': 8},
        )
        ax[idx].set_axis_off()
        ax[idx].set_title(name_of_subplot, {'size': 12, 'color': 'black'})
    fig.savefig(make_graph_file_name(name_of_figure.replace(' ', '_')))
    fig.show()
    plt.close(fig)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Description of the usage of healthcare system resources."""

    figure2_appointments_used(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )


if __name__ == "__main__":

    results_folder = Path('./outputs') / 'tlo_output_from_server'  # small run created for test purposes (locally)

    apply(
        results_folder=results_folder,
        output_folder=results_folder,
        resourcefilepath=Path('./resources')
    )
