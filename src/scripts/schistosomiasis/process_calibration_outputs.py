


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

results_folder = Path("outputs/schisto_calibration-2024-06-05T133315Z")

# azure runs
# results_folder = Path("outputs/t.mangal@imperial.ac.uk/calibration-2024-05-24T110817Z")

# local runs
results_folder = Path("outputs/t.mangal@imperial.ac.uk/schisto_calibration-2024-05-30T125223Z")

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Name of species that being considered:
species = ('mansoni', 'haematobium')


# get prevalence
def get_model_prevalence_by_district_over_time(_df):
    """Get the prevalence every year of the simulation """

    # select the last entry for each year
    _df.set_index('date', inplace=True)
    df = _df.resample('Y').last()

    df = df.filter(like='Blantyre')

    # todo limit to SAC
    # df = df.filter(like='Adult')
    df = df.filter(like='SAC')

    # Aggregate the sums of infection statuses by district_of_residence and year
    district_sum = df.sum(axis=1)

    # todo limit to high or low-infection only
    # df_filtered = df.filter(regex='(Low-infection)')
    # df_filtered = df.filter(regex='(High-infection)')
    df_filtered = df.filter(regex='(High-infection|Low-infection)')
    # df_filtered = df.filter(regex='(High-infection|Moderate-infection|Low-infection)')

    infected = df_filtered.sum(axis=1)

    prop_infected = infected.div(district_sum)

    return prop_infected


prev = extract_results(
        results_folder,
        module="tlo.methods.schisto",
        key="infection_status_haematobium",
        custom_generate_series=get_model_prevalence_by_district_over_time,
        do_scaling=False,
    )

prev.index = prev.index.year

# Plot the columns of values as lines in one figure
prev.plot(kind='line')  # You can customize the plot further if needed
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('')
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend(title='Columns')
plt.show()

