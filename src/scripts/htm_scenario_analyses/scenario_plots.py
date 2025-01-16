""" this reads in the outputs generates through analysis_htm_scaleup.py
and produces plots for HIV, TB and malaria incidence
"""


import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs")
# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")


# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("scaleup_tests", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# DEATHS


def get_num_deaths_by_cause_label(_df):
    """Return total number of Deaths by label within the TARGET_PERIOD
    values are summed for all ages
    df returned: rows=COD, columns=draw
    """
    return _df \
        .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
        .groupby(_df['label']) \
        .size()


TARGET_PERIOD = (Date(2020, 1, 1), Date(2025, 1, 1))

num_deaths_by_cause_label = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_cause_label,
        do_scaling=False
    )


def summarise_deaths_for_one_cause(results_folder, label):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "label"])["person_id"].count()
        ),
        do_scaling=True,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    tmp = results_deaths.loc[
        (results_deaths.label == label)
    ]

    # group deaths by year
    tmp = pd.DataFrame(tmp.groupby(["year"]).sum())

    # get mean for each draw
    mean_deaths = pd.concat({'mean': tmp.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_deaths


aids_deaths = summarise_deaths_for_one_cause(results_folder, 'AIDS')
tb_deaths = summarise_deaths_for_one_cause(results_folder, 'TB (non-AIDS)')
malaria_deaths = summarise_deaths_for_one_cause(results_folder, 'Malaria')

draw_labels = ['No scale-up', 'HIV, scale-up', 'TB scale-up', 'Malaria scale-up', 'HTM scale-up']

colors = sns.color_palette("Set1", 5) # Blue, Orange, Green, Red


# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 10))

# Plot for df1
for i, col in enumerate(aids_deaths.columns):
    axs[0].plot(aids_deaths.index, aids_deaths[col], label=draw_labels[i], color=colors[i])
axs[0].set_title('HIV/AIDS')
axs[0].legend()
axs[0].axvline(x=2019, color='gray', linestyle='--')

# Plot for df2
for i, col in enumerate(tb_deaths.columns):
    axs[1].plot(tb_deaths.index, tb_deaths[col], color=colors[i])
axs[1].set_title('TB')
axs[1].axvline(x=2019, color='gray', linestyle='--')

# Plot for df3
for i, col in enumerate(malaria_deaths.columns):
    axs[2].plot(malaria_deaths.index, malaria_deaths[col], color=colors[i])
axs[2].set_title('Malaria')
axs[2].axvline(x=2019, color='gray', linestyle='--')

for ax in axs:
    ax.set_xlabel('Years')
    ax.set_ylabel('Number deaths')

plt.tight_layout()
plt.show()

