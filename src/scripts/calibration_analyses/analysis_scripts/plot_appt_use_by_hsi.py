import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import squarify
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import bin_hsi_event_details, compute_mean_across_runs

PREFIX_ON_FILENAME = '5'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))


def figure5_proportion_of_hsi_events_per_appt_type(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """ Figure 5: Proportion of hsi events for each appointment type """

    # get the data frame of counts of hsi events by treatment id and appt type
    counts_by_treatment_id_and_appt_type = compute_mean_across_runs(
        bin_hsi_event_details(
            results_folder,
            lambda event_details, count: sum(
                [
                    Counter({
                        (
                            event_details["treatment_id"],
                            appt_type
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

    # change counter to dataframe
    counter_to_df = pd.DataFrame.from_dict(counts_by_treatment_id_and_appt_type, orient='index').reset_index()
    counter_to_df['TREATMENT_ID'], counter_to_df['Appt_Type'] = zip(*counter_to_df['index'])
    counter_to_df = counter_to_df.rename(columns={0: 'Count'}).drop(columns='index').copy()

    # get avg annual count
    counter_to_df['Count'] = counter_to_df['Count']/5

    # plot per appointment type
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
    counter_to_df.Appt_Type = counter_to_df.Appt_Type.replace(appt_dict)
    counter_to_df = counter_to_df.groupby(
        ['TREATMENT_ID', 'Appt_Type'])['Count'].sum().reset_index()

    # plot a square plot of appt use by treatment_id for each appt
    # appts to be plot
    appts = ['AccidentsandEmerg', 'AntenatalTotal', 'Csection',
             'Delivery', 'EPI', 'EstAdult', 'FamPlan',
             'IPAdmission', 'InpatientDays', 'MaleCirc',
             'MentalAll', 'NewAdult', 'OPD', 'Peds', 'TBNew',
             'U5Malnutr', 'VCTTests']
    fig, axs = plt.subplots(len(appts), 1, figsize=(11, 40))
    name_of_figure = 'Proportion of Appointment Use by TREATMENT_ID per Appointment Type'
    for idx in range(len(appts)):
        df_to_plot = counter_to_df[counter_to_df.Appt_Type == appts[idx]].copy()
        df_to_plot = df_to_plot.sort_values(by=['Count'], ascending=False)  # sort count in descending order
        name_of_subplot = appts[idx]
        axs[idx] = squarify.plot(sizes=df_to_plot.Count, label=df_to_plot.TREATMENT_ID[:3],  # label top 3
                                 alpha=0.6, pad=True, ax=axs[idx],
                                 text_kwargs={'color': 'black', 'size': 12})
        axs[idx].axis('off')
        axs[idx].invert_xaxis()
        axs[idx].legend(handles=axs[idx].containers[0][3:10],  # legend for top 10
                        labels=list(df_to_plot.TREATMENT_ID[3:10]),
                        handlelength=1, handleheight=1, fontsize=10,
                        title='Unlabelled Treatment ID', title_fontsize=11,
                        ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
        axs[idx].set_title(name_of_subplot, {'size': 13, 'color': 'black'})
    fig.tight_layout()
    fig.savefig(
        output_folder
        / f"{PREFIX_ON_FILENAME}_{name_of_figure.replace(' ', '_')}.png"
    )
    fig.show()
    plt.close(fig)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Description of the usage of each appointment type by treatment id."""

    figure5_proportion_of_hsi_events_per_appt_type(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
    )
