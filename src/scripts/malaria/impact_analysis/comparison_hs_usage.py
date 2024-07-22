"""
Read in the output files generated by analysis_scenarios and plot outcomes for comparison

from baseline outputs, extract numbers of treatments and sum by group, test, treatment etc.
"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("exclude_HTM_services.py", outputspath)[-1]
# results_folder = get_scenario_outputs("remove_treatment_effects.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# extract scaling factor
scaling_factor = extract_results(
    results_folder,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


# %% -------------------------------------------------------------------------------------------------------
# EXTRACT GROUPED APPT TYPES FOR EACH SCENARIO

# group together appts
opd = ['MaleCirc', 'FamPlan', 'VCTNegative', 'Over5OPD', 'VCTPositive', 'MinorSurg',
       'Under5OPD', 'NewAdult', 'Peds', 'EstNonCom', 'EPI', 'U5Malnutr',
       'ConWithDCSA', 'TBNew', 'MentOPD', 'TBFollowUp', 'AntenatalFirst',
       'ANCSubsequent', 'NormalDelivery']
inpat = ['IPAdmission', 'InpatientDays', 'MajorSurg', 'CompDelivery', 'Csection', 'AccidentsandEmerg']
lab = ['DiagRadio', 'Tomography', 'Mammography', 'LabTBMicro', 'LabMolec']
pharm = ['PharmDispensing']


# extract numbers of appts delivered for every run within a specified draw
def sum_appt_by_id(results_folder, module, key, column, draw):
    """
    sum occurrences of each treatment_id over the simulation period for every run within a draw

    produces dataframe: rows=treatment_id, columns=counts for every run

    results are scaled to true population size
    """

    info = get_scenario_info(results_folder)
    # create emtpy dataframe
    results = pd.DataFrame()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        tmp = pd.DataFrame(new[column].to_list())

        # sum each column to get total appts of each type over the simulation
        tmp2 = pd.DataFrame(tmp.sum())
        # add results to dataframe for output
        results = pd.concat([results, tmp2], axis=1)

    # multiply appt numbers by scaling factor
    results = results.mul(scaling_factor.values[0][0])

    # make row index a column
    results.reset_index(level=0, inplace=True)

    # groupby and sum if row is in opd
    # Create a mapping dictionary
    appointment_type_mapping = {appointment: category for category, appointments in
                                zip(['opd', 'inpat', 'lab', 'pharm'], [opd, inpat, lab, pharm])
                                for appointment in appointments}

    # Map values in the DataFrame column to appointment types
    results['AppointmentCategory'] = results['index'].map(appointment_type_mapping)

    # sum appointments for each run by appointment category
    df2 = results.groupby('AppointmentCategory').sum()
    df2 = df2.drop('index', axis=1)

    return df2


# extract numbers of appts
module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'Number_By_Appt_Type_Code'

# get total counts of every appt type for each scenario
# these counts are scaled
appt_sums0 = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=0)
appt_sums1 = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=1)
appt_sums2 = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=2)
appt_sums3 = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=3)
appt_sums4 = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=4)


baseline_median = appt_sums0.median(axis='columns')
baseline_lower = appt_sums0.quantile(0.025, axis='columns')
baseline_upper = appt_sums0.quantile(0.975, axis='columns')

sc4_median = appt_sums4.median(axis='columns')
sc4_lower = appt_sums4.quantile(0.025, axis='columns')
sc4_upper = appt_sums4.quantile(0.975, axis='columns')


diffs0_1 = appt_sums1 - appt_sums0
diffs0_2 = appt_sums2 - appt_sums0
diffs0_3 = appt_sums3 - appt_sums0
diffs0_4 = appt_sums4 - appt_sums0


percent_diffs0_1 = (appt_sums1 - appt_sums0)/appt_sums0
percent_diffs0_2 = (appt_sums2 - appt_sums0)/appt_sums0
percent_diffs0_3 = (appt_sums3 - appt_sums0)/appt_sums0
percent_diffs0_4 = (appt_sums4 - appt_sums0)/appt_sums0

lower_percentdiffs0_1 = percent_diffs0_1.quantile(0.025, axis='columns')
lower_percentdiffs0_2 = percent_diffs0_2.quantile(0.025, axis='columns')
lower_percentdiffs0_3 = percent_diffs0_3.quantile(0.025, axis='columns')
lower_percentdiffs0_4 = percent_diffs0_4.quantile(0.025, axis='columns')

upper_percentdiffs0_1 = percent_diffs0_1.quantile(0.975, axis='columns')
upper_percentdiffs0_2 = percent_diffs0_2.quantile(0.975, axis='columns')
upper_percentdiffs0_3 = percent_diffs0_3.quantile(0.975, axis='columns')
upper_percentdiffs0_4 = percent_diffs0_4.quantile(0.975, axis='columns')




# for the diffs, if value negative scenario count lower than baseline
# if value is positive, scenario count higher than baseline
print(diffs0_1.median(axis='columns'))

# Calculate medians for each DataFrame
# values are median difference in hs usage run by run
median_diffs0_1 = diffs0_1.median(axis='columns')
median_diffs0_2 = diffs0_2.median(axis='columns')
median_diffs0_3 = diffs0_3.median(axis='columns')
median_diffs0_4 = diffs0_4.median(axis='columns')
# median_diffs0_5 = diffs0_5.median(axis='columns')

lower_diffs0_1 = diffs0_1.quantile(0.025, axis='columns')
lower_diffs0_2 = diffs0_2.quantile(0.025, axis='columns')
lower_diffs0_3 = diffs0_3.quantile(0.025, axis='columns')
lower_diffs0_4 = diffs0_4.quantile(0.025, axis='columns')
# lower_diffs0_5 = diffs0_5.quantile(0.025, axis='columns')

upper_diffs0_1 = diffs0_1.quantile(0.975, axis='columns')
upper_diffs0_2 = diffs0_2.quantile(0.975, axis='columns')
upper_diffs0_3 = diffs0_3.quantile(0.975, axis='columns')
upper_diffs0_4 = diffs0_4.quantile(0.975, axis='columns')
# upper_diffs0_5 = diffs0_5.quantile(0.975, axis='columns')


# Create a DataFrame using the median values
out = pd.DataFrame({
    'Baseline': baseline_median,
    'Baseline_lower': baseline_lower,
    'Baseline_upper': baseline_upper,
    'Exclude HIV': median_diffs0_1,
    'lower_diffs0_1': lower_diffs0_1,
    'upper_diffs0_1': upper_diffs0_1,
    'Exclude TB': median_diffs0_2,
    'lower_diffs0_2': lower_diffs0_2,
    'upper_diffs0_2': upper_diffs0_2,
    'Exclude malaria': median_diffs0_3,
    'lower_diffs0_3': lower_diffs0_3,
    'upper_diffs0_3': upper_diffs0_3,
    'Exclude HTM': median_diffs0_4,
    'lower_diffs0_4': lower_diffs0_4,
    'upper_diffs0_4': upper_diffs0_4,
    # 'Remove HTM': median_diffs0_5,
    # 'lower_diffs0_5': lower_diffs0_5,
    # 'upper_diffs0_5': upper_diffs0_5
})

def round_to_nearest_100(x):
    return round(x, -2)

out = out.applymap(round_to_nearest_100)
# Convert all values to integers
out = out.astype(int, errors='ignore')

out.to_csv(outputspath / ('comparison_hs_usage_excl_HTM_4Dec' + '.csv'))






