


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    unflatten_flattened_multi_index_in_logging,
)

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs/t.mangal@imperial.ac.uk")

# results_folder = Path("outputs/schisto_calibration-2024-06-05T133315Z")

# azure runs
# results_folder = Path("outputs/t.mangal@imperial.ac.uk/calibration-2024-05-24T110817Z")

# local runs
results_folder = Path("outputs/t.mangal@imperial.ac.uk/schisto_calibration-2024-06-07T171443Z")


# results_folder = Path("outputs/t.mangal@imperial.ac.uk/schisto_tmp")

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Name of species that being considered:
species = ('mansoni', 'haematobium')


# extract mean worm burden from equilibrium phase
def get_mean_worm_burden(_df):
    """Get the prevalence every year of the simulation """

    # select the last entry for each year
    _df.set_index('date', inplace=True)

    df = _df.filter(like='Likoma')

    # select last 10 years (last 10*12 rows)
    last_120_values = df['Likoma'].tail(120)
    mean_last_120 = last_120_values.mean()

    return pd.Series(mean_last_120)


# get prevalence
def get_model_prevalence_by_district_over_time(_df):
    """Get the prevalence every year of the simulation """

    # select the last entry for each year
    _df.set_index('date', inplace=True)
    df = _df.resample('Y').last()

    df = df.filter(like='Likoma')

    # todo limit to SAC
    if age == 'SAC':
        df = df.filter(like='SAC')
    if age == 'adult':
        df = df.filter(like='Adult')

    # Aggregate the sums of infection statuses by district_of_residence and year
    district_sum = df.sum(axis=1)

    if inf == 'HML':
        df_filtered = df.filter(regex='(High-infection|Moderate-infection|Low-infection)')
    if inf == 'HM':
        df_filtered = df.filter(regex='(Moderate-infection|High-infection)')
    if inf == 'ML':
        df_filtered = df.filter(regex='(Moderate-infection|Low-infection)')
    if inf == 'M':
        df_filtered = df.filter(regex='(Moderate-infection)')
    if inf == 'L':
        df_filtered = df.filter(regex='(Low-infection)')

    infected = df_filtered.sum(axis=1)

    prop_infected = infected.div(district_sum)

    return prop_infected


age = 'SAC'  # SAC, adult, all
inf = 'HM'
prev = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="infection_status_haematobium",
        custom_generate_series=get_model_prevalence_by_district_over_time,
        do_scaling=False,
)
prev.index = prev.index.year

# select last 10 years
# prev = prev[prev.index >= (prev.index.max() - 9)]

eq_value_H = prev.iloc[-10:, :4].mean().mean()


# Plot the columns of values as lines in one figure
prev.plot(kind='line')  # You can customize the plot further if needed
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('')
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend(title='Columns')
plt.show()




age = 'SAC'  # SAC, adult, all
inf = 'HM'
prev = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="infection_status_mansoni",
        custom_generate_series=get_model_prevalence_by_district_over_time,
        do_scaling=False,
    )
prev.index = prev.index.year

# select last 10 years
# prev = prev[prev.index >= (prev.index.max() - 9)]

eq_value_M = prev.iloc[-10:, 4:].mean().mean()

# Plot the columns of values as lines in one figure
prev.plot(kind='line')  # You can customize the plot further if needed
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('')
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend(title='Columns')
plt.show()



mwb_H = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="mean_worm_burden_by_district_haematobium",
        custom_generate_series=get_mean_worm_burden,
        do_scaling=False,
    )
mwb_H.columns = mwb_H.columns.get_level_values(0)
mean_excluding_zero_H = mwb_H.iloc[0][mwb_H.iloc[0] != 0].mean()


mwb_M = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="mean_worm_burden_by_district_mansoni",
        custom_generate_series=get_mean_worm_burden,
        do_scaling=False,
    )
mwb_M.columns = mwb_M.columns.get_level_values(0)
mean_excluding_zero_M = mwb_M.iloc[0][mwb_M.iloc[0] != 0].mean()

