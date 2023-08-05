from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import (
    bin_hsi_event_details,
    compute_mean_across_runs,
    extract_results,
    get_scenario_outputs,
    summarize,
)

PREFIX_ON_FILENAME = '6'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

# appointment dict to match model and data
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


def get_annual_num_hsi_by_appt_and_level(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual count of hsi
    per treatment id per each appt type per level."""
    hsi_count = compute_mean_across_runs(
        bin_hsi_event_details(
            results_folder,
            lambda event_details, count: sum(
                [
                    Counter({
                        (
                            event_details['treatment_id'],
                            appt_type,
                            event_details['facility_level'],
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

    hsi_count = pd.DataFrame.from_dict(hsi_count, orient='index').reset_index().rename(columns={0: 'Count'})
    hsi_count[['Treatment_ID', 'Appt_Type_Code', 'Facility_Level']] = pd.DataFrame(hsi_count['index'].tolist(),
                                                                                   index=hsi_count.index)
    # average annual count by treatment id, appt type and facility level
    yr_count = TARGET_PERIOD[1].year - TARGET_PERIOD[0].year + 1
    hsi_count = hsi_count.groupby(['Treatment_ID', 'Appt_Type_Code', 'Facility_Level'])['Count'].sum()/yr_count
    hsi_count = hsi_count.to_frame().reset_index()

    # drop dummy PharmDispensing for HCW paper results and plots
    hsi_count = hsi_count.drop(index=hsi_count[hsi_count['Appt_Type_Code'] == 'PharmDispensing'].index)

    return hsi_count


def get_annual_hcw_time_used_with_confidence_interval(results_folder: Path, resourcefilepath: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual hcw time used per cadre across all levels,
    with 95% confidence interval."""

    def get_annual_hcw_time_used(_df) -> pd.Series:
        """Get the annual hcw time used per cadre across all levels"""

        # get annual counts of appt per level
        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  _idx: pd.DataFrame.from_dict(mydict) for _idx, mydict in _raw.items()
                 }
             ).unstack().fillna(0.0).astype(int)

        annual_counts_of_appts_per_level = _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .groupby(level=[0, 1], axis=1).sum() \
            .mean(axis=0) \
            .to_frame().reset_index() \
            .rename(columns={'level_0': 'Facility_Level', 'level_1': 'Appt_Type_Code', 0: 'Count'}) \
            .pivot(index='Facility_Level', columns='Appt_Type_Code', values='Count') \
            .drop(columns='PharmDispensing')  # do not include this dummy appt for HCW paper results and plots

        # get appt time definitions
        appt_time = get_expected_appt_time(resourcefilepath)

        appts_def = set(appt_time.Appt_Type_Code)
        appts_sim = set(annual_counts_of_appts_per_level.columns.values)
        assert appts_sim.issubset(appts_def)

        # get hcw time used per cadre per level
        _hcw_usage = appt_time.drop(index=appt_time[~appt_time.Appt_Type_Code.isin(appts_sim)].index).reset_index(
            drop=True)
        for idx in _hcw_usage.index:
            fl = _hcw_usage.loc[idx, 'Facility_Level']
            appt = _hcw_usage.loc[idx, 'Appt_Type_Code']
            _hcw_usage.loc[idx, 'Total_Mins_Used_Per_Year'] = (_hcw_usage.loc[idx, 'Time_Taken_Mins'] *
                                                               annual_counts_of_appts_per_level.loc[fl, appt])

        # get hcw time used per cadre
        _hcw_usage = _hcw_usage.groupby(['Officer_Category'])['Total_Mins_Used_Per_Year'].sum()

        return _hcw_usage

    # get hcw time used per cadre with CI
    hcw_usage = summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_annual_hcw_time_used,
                do_scaling=True
            ),
        only_mean=False,
        collapse_columns=True,
        ).unstack()

    # reformat
    hcw_usage = hcw_usage.to_frame().reset_index() \
        .rename(columns={'stat': 'Value_Type', 0: 'Value'}) \
        .pivot(index='Officer_Category', columns='Value_Type', values='Value')

    return hcw_usage


def get_expected_appt_time(resourcefilepath) -> pd.DataFrame:
    """This is to return the expected time requirements per appointment type per coarse cadre per facility level."""
    expected_appt_time = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Time_Table.csv')
    appt_type = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv')
    expected_appt_time = expected_appt_time.merge(
        appt_type[['Appt_Type_Code', 'Appt_Cat']], on='Appt_Type_Code', how='left')
    # rename Appt_Cat
    appt_cat = {'GENERAL_INPATIENT_AND_OUTPATIENT_CARE': 'IPOP',
                'Nutrition': 'NUTRITION',
                'Misc': 'MISC',
                'Mental_Health': 'MENTAL'}
    expected_appt_time['Appt_Cat'] = expected_appt_time['Appt_Cat'].replace(appt_cat)
    expected_appt_time.rename(columns={'Appt_Cat': 'Appt_Category'}, inplace=True)

    # modify time for dummy ConWithDCSA so that no overworking/underworking
    expected_appt_time.loc[expected_appt_time['Appt_Category'] == 'ConWithDCSA', 'Time_Taken_Mins'] = 20.0

    return expected_appt_time


def get_hcw_capability(resourcefilepath, hcwscenario='actual') -> pd.DataFrame:
    """This is to return the annual hcw capabilities per cadre per facility level.
       Argument hcwscenario can be actual, funded_plus."""
    hcw_capability = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / hcwscenario / 'ResourceFile_Daily_Capabilities.csv'
    )
    hcw_capability = hcw_capability.groupby(['Facility_Level', 'Officer_Category']
                                            )['Total_Mins_Per_Day'].sum().reset_index()
    hcw_capability['Total_Mins_Per_Year'] = hcw_capability['Total_Mins_Per_Day'] * 365.25
    hcw_capability.drop(columns='Total_Mins_Per_Day', inplace=True)

    return hcw_capability


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Compare appointment usage from model output with real appointment usage.
    The real appointment usage is collected from DHIS2 system and HIV Dept."""

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

    # format data and plot bar chart for hcw working time per cadre
    def format_hcw_usage(hcwscenario='actual'):
        """format data for bar plot"""
        # get hcw capability in actual or establishment (funded_plus) scenarios
        hcw_capability = get_hcw_capability(resourcefilepath, hcwscenario=hcwscenario) \
            .groupby('Officer_Category')['Total_Mins_Per_Year'].sum().to_frame() \
            .rename(columns={'Total_Mins_Per_Year': 'capability'})

        # calculate hcw time usage ratio against capability with CI
        hcw_usage = get_annual_hcw_time_used_with_confidence_interval(results_folder, resourcefilepath)
        hcw_usage_ratio = hcw_usage.join(hcw_capability)
        hcw_usage_ratio.loc['All'] = hcw_usage_ratio.sum()
        hcw_usage_ratio['mean'] = hcw_usage_ratio['mean'] / hcw_usage_ratio['capability']
        hcw_usage_ratio['lower'] = hcw_usage_ratio['lower'] / hcw_usage_ratio['capability']
        hcw_usage_ratio['upper'] = hcw_usage_ratio['upper'] / hcw_usage_ratio['capability']

        hcw_usage_ratio['lower_error'] = (hcw_usage_ratio['mean'] - hcw_usage_ratio['lower'])
        hcw_usage_ratio['upper_error'] = (hcw_usage_ratio['upper'] - hcw_usage_ratio['mean'])

        asymmetric_error = [hcw_usage_ratio['lower_error'].values, hcw_usage_ratio['upper_error'].values]
        hcw_usage_ratio = pd.DataFrame(hcw_usage_ratio['mean']) \
            .clip(lower=0.1, upper=10.0)

        # reduce the mean ratio by 1.0, for the bar plot that starts from y=1.0 instead of y=0.0
        hcw_usage_ratio['mean'] = hcw_usage_ratio['mean'] - 1.0

        # rename cadre Nursing_and_Midwifery
        hcw_usage_ratio.rename(index={'Nursing_and_Midwifery': 'Nursing and Midwifery'}, inplace=True)

        return hcw_usage_ratio, asymmetric_error

    hcw_usage_ratio_actual, error_actual = format_hcw_usage(hcwscenario='actual')
    hcw_usage_ratio_establishment, error_establishment = format_hcw_usage(hcwscenario='funded_plus')

    name_of_plot = 'Simulated average annual working time (95% CI) vs Capability per cadre'
    fig, ax = plt.subplots(figsize=(8, 5))
    hcw_usage_ratio_establishment.plot(kind='bar', yerr=error_establishment, width=0.4,
                                       ax=ax, position=0, bottom=1.0,
                                       legend=False, color='c')
    hcw_usage_ratio_actual.plot(kind='bar', yerr=error_actual, width=0.4,
                                ax=ax, position=1, bottom=1.0,
                                legend=False, color='y')
    ax.axhline(1.0, color='r')
    ax.set_xlim(right=len(hcw_usage_ratio_establishment) - 0.3)
    ax.set_yscale('log')
    ax.set_ylim(1 / 20, 20)
    ax.set_yticks([1 / 10, 1.0, 10])
    ax.set_yticklabels(("<= 1/10", "1.0", ">= 10"))
    ax.set_ylabel('Working time / Capability')
    ax.set_xlabel('Cadre Category')
    plt.xticks(rotation=60, ha='right')
    ax.xaxis.grid(True, which='major', linestyle='--')
    ax.yaxis.grid(True, which='both', linestyle='--')
    ax.set_title(name_of_plot)
    patch_establishment = matplotlib.patches.Patch(facecolor='c', label='Establishment capability')
    patch_actual = matplotlib.patches.Patch(facecolor='y', label='Actual capability')
    legend = plt.legend(handles=[patch_actual, patch_establishment], loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.add_artist(legend)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(',', '').replace('\n', '_').replace(' ', '_')))
    plt.show()

    # hcw usage per cadre per appt per hsi
    hsi_count = get_annual_num_hsi_by_appt_and_level(results_folder)

    # first compare appts defined and appts in simulation/model
    appt_time = get_expected_appt_time(resourcefilepath)
    appts_def = set(appt_time.Appt_Type_Code)
    appts_sim = set(hsi_count.Appt_Type_Code)
    assert appts_sim.issubset(appts_def)

    # then calculate the hcw working time per treatment id, appt type and cadre
    hcw_usage_hsi = appt_time.drop(index=appt_time[~appt_time.Appt_Type_Code.isin(appts_sim)].index
                                   ).reset_index(drop=True)
    hcw_usage_hsi = hsi_count.merge(hcw_usage_hsi, on=['Facility_Level', 'Appt_Type_Code'], how='left')
    hcw_usage_hsi['Total_Mins_Used_Per_Year'] = hcw_usage_hsi['Count'] * hcw_usage_hsi['Time_Taken_Mins']
    hcw_usage_hsi = hcw_usage_hsi.groupby(['Treatment_ID', 'Appt_Category', 'Officer_Category']
                                          )['Total_Mins_Used_Per_Year'].sum().reset_index()

    # rename Nursing_and_Midwifery
    hcw_usage_hsi.Officer_Category = hcw_usage_hsi.Officer_Category.replace(
        {'Nursing_and_Midwifery': 'Nursing and Midwifery'})

    # save the data to draw sankey diagram
    hcw_usage_hsi.to_csv(output_folder/'hcw_working_time_per_hsi.csv', index=False)


if __name__ == "__main__":
    outputspath = Path('./outputs/bshe@ic.ac.uk')
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = '10_year_scale_run.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-4]

    # Test dataset:
    # results_folder = Path('/Users/tbh03/GitHub/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-small')

    # If needed -- in the case that pickles were not created remotely during batch
    # create_pickles_locally(results_folder)

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
