from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import (
    bin_hsi_event_details,
    compute_mean_across_runs,
    extract_results,
    get_coarse_appt_type,
    get_scenario_outputs,
    summarize)

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


def get_annual_num_appts_by_level(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level."""

    def get_counts_of_appts(_df):
        """Get the mean number of appointments of each type being used each year at each level.
        Need to rename appts to match standardized categories from the DHIS2 data."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.items()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .groupby(level=[0, 1], axis=1).sum() \
            .mean(axis=0)  # mean over each year (row)

    return summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_appts,
                do_scaling=True
            ),
        only_mean=True,
        collapse_columns=True,
        ).unstack().astype(int)


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

    return hsi_count


def get_simulation_usage_by_level(results_folder: Path) -> pd.DataFrame:
    """Returns the simulated MEAN USAGE PER YEAR DURING THE TIME_PERIOD, by appointment type and level.
    """

    # Get model outputs
    model_output = get_annual_num_appts_by_level(results_folder=results_folder)

    return model_output


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
            .pivot(index='Facility_Level', columns='Appt_Type_Code', values='Count')

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


def adjust_real_usage_on_mentalall(real_usage_df) -> pd.DataFrame:
    """This is to adjust the annual MentalAll usage in real usage dataframe.
    The MentalAll usage was not adjusted in the preprocessing stage considering individual facilities and very low
    reporting rates.
    We now directly adjust its annual usage by facility level using the aggregated annual reporting rates by
    facility level. The latter is calculated based on DHIS2 Mental Health Report reporting rates."""
    # the annual reporting rates for Mental Health Report by facility level (%), 2015-2019
    # could turn the reporting rates data into a ResourceFile if necessary
    rr = pd.DataFrame(index=['1a', '1b', '2', '3'], columns=list(range(2015, 2020)),
                      data=[[44.00, 39.33, 79.00, 97.33, 95.00],
                            [10.42, 12.50, 25.00, 40.00, 68.33],
                            [36.67, 39.44, 37.22, 63.89, 56.67],
                            [50.00, 45.83, 45.83, 50.00, 45.83]])
    # make the adjustment assuming 100% reporting rates for each year
    for level in ['1a', '1b', '2', '3']:
        for y in range(2015, 2020):
            real_usage_df.loc[(real_usage_df.Facility_Level == level)
                              & (real_usage_df.Year == y)
                              & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] = (
                real_usage_df.loc[(real_usage_df.Facility_Level == level)
                                  & (real_usage_df.Year == y)
                                  & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] * 100 / rr.loc[level, y]
            )

    return real_usage_df


def get_real_usage(resourcefilepath, adjusted=True) -> pd.DataFrame:
    """
    Returns the adjusted (default) or unadjusted real data on the (MEAN) USAGE PER YEAR DURING THE TIME_PERIOD
    for each appointment at each level and all levels.
    """

    # add facility level and district columns to both real and simulation usage
    mfl = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

    # Get real usage data
    # For the details of adjustment of real usage data, refer to Paper
    # "The Changes in Health Service Utilisation in Malawi during the COVID-19 Pandemic"
    if adjusted:
        real_usage = pd.read_csv(
            resourcefilepath / 'healthsystem' / 'real_appt_usage_data' /
            'real_monthly_usage_of_appt_type.csv')
    else:
        real_usage = pd.read_csv(
            resourcefilepath / 'healthsystem' / 'real_appt_usage_data' /
            'unadjusted_real_monthly_usage_of_appt_type.csv')

    # add Csection usage to Delivery, as Delivery has excluded Csection in real data file (to avoid overlap)
    # whereas Delivery in tlo output has included Csection
    real_delivery = real_usage.loc[(real_usage.Appt_Type == 'Delivery') | (real_usage.Appt_Type == 'Csection')
                                   ].groupby(['Year', 'Month', 'Facility_ID']).agg({'Usage': 'sum'}).reset_index()
    real_delivery['Appt_Type'] = 'Delivery'
    real_usage = pd.concat([real_usage.drop(real_usage[real_usage.Appt_Type == 'Delivery'].index),
                            real_delivery])

    # get facility_level for each record
    real_usage = real_usage.merge(mfl[['Facility_ID', 'Facility_Level']], left_on='Facility_ID', right_on='Facility_ID')

    # adjust annual MentalAll usage using annual reporting rates
    if adjusted:
        real_usage = adjust_real_usage_on_mentalall(real_usage)

    # assign date to each record
    real_usage['date'] = pd.to_datetime({'year': real_usage['Year'], 'month': real_usage['Month'], 'day': 1})

    # Produce table of the AVERAGE NUMBER PER YEAR DURING THE TIME_PERIOD of appointment type by level
    # limit to date
    totals_by_year = real_usage \
        .loc[real_usage['date'].between(*TARGET_PERIOD)] \
        .groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()

    # Combine the TB data [which is yearly] (after dropping period outside 2017-2019 according to data consistency
    # and pandemic) with the rest of the data.
    # Note that TB data is not adjusted considering comparability with NTP reports.
    real_usage_TB = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'real_appt_usage_data' / 'real_yearly_usage_of_TBNotifiedAll.csv')
    real_usage_TB = real_usage_TB.loc[real_usage_TB['Year'].isin([2017, 2018, 2019])]
    real_usage_TB = real_usage_TB.merge(mfl[['Facility_ID', 'Facility_Level']],
                                        left_on='Facility_ID', right_on='Facility_ID')
    totals_by_year_TB = real_usage_TB.groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()

    annual_usage_by_level = pd.concat([totals_by_year.reset_index(), totals_by_year_TB.reset_index()], axis=0)

    # group levels 1b and 2 into 2
    # annual_usage_by_level['Facility_Level'] = annual_usage_by_level['Facility_Level'].replace({'1b': '2'})
    annual_usage_by_level = annual_usage_by_level.groupby(
        ['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum().reset_index()

    # prepare annual usage by level with mean, 97.5% percentile, and 2.5% percentile
    annual_usage_by_level_with_ci = annual_usage_by_level.drop(columns='Year').groupby(
        ['Appt_Type', 'Facility_Level']
    ).describe(percentiles=[0.025, 0.975]
               ).stack(level=[0])[['mean', '2.5%', '97.5%']].reset_index().drop(columns='level_2')

    average_annual_by_level = annual_usage_by_level_with_ci[['Appt_Type', 'Facility_Level', 'mean']].set_index(
        ['Appt_Type', 'Facility_Level']).unstack()
    average_annual_by_level.columns = average_annual_by_level.columns.get_level_values(1)
    average_annual_by_level = average_annual_by_level.T

    annual_usage_by_level_with_ci = pd.melt(annual_usage_by_level_with_ci,
                                            id_vars=['Appt_Type', 'Facility_Level'], var_name='value_type')
    annual_usage_by_level_with_ci.value_type = annual_usage_by_level_with_ci.value_type.replace({'2.5%': 'lower',
                                                                                                 '97.5%': 'upper'})

    # prepare annual usage at all levels with mean, 97.5% percentile, and 2.5% percentile
    annual_usage_with_ci = annual_usage_by_level.groupby(
        ['Year', 'Appt_Type'])['Usage'].sum().reset_index().drop(columns='Year').groupby(
        'Appt_Type').describe(percentiles=[0.025, 0.975]
                              ).stack(level=[0])[['mean', '2.5%', '97.5%']].reset_index().drop(columns='level_1')
    annual_usage_with_ci = pd.melt(annual_usage_with_ci,
                                   id_vars='Appt_Type', var_name='value_type')
    annual_usage_with_ci.value_type = annual_usage_with_ci.value_type.replace({'2.5%': 'lower', '97.5%': 'upper'})

    return average_annual_by_level, annual_usage_by_level_with_ci, annual_usage_with_ci


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

        return hcw_usage_ratio, asymmetric_error

    hcw_usage_ratio_actual, error_actual = format_hcw_usage(hcwscenario='actual')
    hcw_usage_ratio_establishment, error_establishment = format_hcw_usage(hcwscenario='funded_plus')

    name_of_plot = 'Simulated annual working time vs Capability per cadre'
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

    # todo: get actual hcw time used derived from DHIS2 data and plot

    # get average annual usage by level for Simulation and Real
    simulation_usage = get_simulation_usage_by_level(results_folder)

    real_usage = get_real_usage(resourcefilepath)[0]

    # get expected appt time
    appt_time = get_expected_appt_time(resourcefilepath)

    # check that appts in simulation_usage are in appt_time
    appts_def = set(appt_time.Appt_Type_Code)
    appts_sim = set(simulation_usage.columns.values)
    assert appts_sim.issubset(appts_def)

    # hcw usage per cadre per appointment type
    hcw_usage = appt_time.drop(index=appt_time[~appt_time.Appt_Type_Code.isin(appts_sim)].index).reset_index(drop=True)
    for idx in hcw_usage.index:
        hcw_usage.loc[idx, 'Total_Mins_Used_Per_Year'] = (hcw_usage.loc[idx, 'Time_Taken_Mins'] *
                                                          simulation_usage.loc[hcw_usage.loc[idx, 'Facility_Level'],
                                                                               hcw_usage.loc[idx, 'Appt_Type_Code']])
    hcw_usage = hcw_usage.groupby(['Officer_Category', 'Appt_Category']
                                  )['Total_Mins_Used_Per_Year'].sum().reset_index()

    # check that hcw time simulated derived from different methods are equal (or with negligible difference)
    hcw_time_used_1 = hcw_usage.groupby(['Officer_Category'])['Total_Mins_Used_Per_Year'].sum().to_frame()
    hcw_time_used_0 = get_annual_hcw_time_used_with_confidence_interval(results_folder, resourcefilepath)
    assert (hcw_time_used_1.index == hcw_time_used_0.index).all()
    assert (abs(
        (hcw_time_used_1['Total_Mins_Used_Per_Year'] - hcw_time_used_0['mean']) / hcw_time_used_0['mean']) < 1e-4
            ).all()

    # hcw usage per cadre per appt per hsi
    hsi_count = get_annual_num_hsi_by_appt_and_level(results_folder)

    # first check that hsi count by different methods are equal (or with small difference)
    hsi_count_alt = hsi_count.groupby(['Appt_Type_Code', 'Facility_Level'])['Count'].sum().reset_index().pivot(
        index='Facility_Level', columns='Appt_Type_Code', values='Count').fillna(0.0)
    assert (hsi_count_alt - simulation_usage.drop(index='4') < 1.0).all().all()

    # then calculate the hcw working time per treatment id, appt type and cadre
    hcw_usage_hsi = appt_time.drop(index=appt_time[~appt_time.Appt_Type_Code.isin(appts_sim)].index
                                   ).reset_index(drop=True)
    hcw_usage_hsi = hsi_count.merge(hcw_usage_hsi, on=['Facility_Level', 'Appt_Type_Code'], how='left')
    hcw_usage_hsi['Total_Mins_Used_Per_Year'] = hcw_usage_hsi['Count'] * hcw_usage_hsi['Time_Taken_Mins']
    hcw_usage_hsi = hcw_usage_hsi.groupby(['Treatment_ID', 'Appt_Category', 'Officer_Category']
                                          )['Total_Mins_Used_Per_Year'].sum().reset_index()

    # also check that the hcw time from different methods are equal (or with small difference)
    hcw_usage_alt = hcw_usage_hsi.groupby(['Officer_Category', 'Appt_Category']
                                          )['Total_Mins_Used_Per_Year'].sum().reset_index()
    assert (hcw_usage_alt.Officer_Category == hcw_usage.Officer_Category).all
    assert (hcw_usage_alt.Appt_Category == hcw_usage.Appt_Category).all()
    assert ((abs(hcw_usage_alt.Total_Mins_Used_Per_Year - hcw_usage.Total_Mins_Used_Per_Year) /
            hcw_usage.Total_Mins_Used_Per_Year) < 1e-4
            ).all().all()

    # save the data to draw sankey diagram
    hcw_usage_hsi.to_csv(output_folder/'hcw_working_time_per_hsi.csv', index=False)


if __name__ == "__main__":
    outputspath = Path('./outputs/bshe@ic.ac.uk')
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = '10_year_scale_run.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Test dataset:
    # results_folder = Path('/Users/tbh03/GitHub/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-small')

    # If needed -- in the case that pickles were not created remotely during batch
    # create_pickles_locally(results_folder)

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
