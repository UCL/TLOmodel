


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


results_folder = Path("outputs/t.mangal@imperial.ac.uk/calibration-2024-05-24T110817Z")

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# Districts that are high prevalence and for which this model has been calibrated:
fitted_districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

# Name of species that being considered:
species = ('mansoni', 'haematobium')




# get prevalence

def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(log['tlo.methods.schisto'])


def get_model_prevalence_by_district_over_time(_df):
    """Get the prevalence every year of the simulation """

    _df = dfs[f'infection_status_haematobium']
    # select the last entry for each year
    df = _df.resample('Y').last()

    df = df.loc[:, df.columns.get_level_values(1) == 'Blantyre']
    df = df.loc[:, df.columns.get_level_values(2) == 'SAC']

    # Aggregate the sums of infection statuses by district_of_residence and year
    district_sum = df.sum(axis=1)

    filtered_columns = df.columns.get_level_values('infection_status').isin(['High-infection', 'Low-infection'])
    infected = df.loc[:, filtered_columns].sum(axis=1)

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


# ----------- PLOTS -----------------


# All Districts = prevalence by year
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    data = get_model_prevalence_by_district_over_time(_spec)
    data.plot(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('')
    ax.set_ylabel('End of year prevalence')
    ax.set_ylim(0, 1.00)
    ax.get_legend().remove()
    # data.to_csv(outputpath / (f"{_spec}" + '.csv'))

    # Plot legend only for the last subplot
    # if i == len(species) -1:
    #     handles, labels = ax.get_legend_handles_labels()  # Get handles and labels for legend
    #     ax.legend(handles, labels, bbox_to_anchor =(1.44,-0.10), loc='lower right')

fig.tight_layout()
# fig.savefig(make_graph_file_name('annual_prev_in_districts'))
fig.show()
