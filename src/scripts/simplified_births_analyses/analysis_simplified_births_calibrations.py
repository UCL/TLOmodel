"""
Plot to demonstrate calibration of simplified births module
"""

# %% Import Statements and initial declarations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import (
    make_calendar_period_lookup,
    make_calendar_period_type,
)
from tlo.methods import (
    demography,
    simplified_births,
)

# Path to the resource files
resources = Path("./resources")


def run():
    # Setting the seed for the Simulation instance.
    seed = 0

    # configuring outputs
    log_config = {
        "filename": "simplified_births_calibrations",
    }

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2018, 1, 2)
    pop_size = 200

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # registering all required modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# date-stamp to label log files and any other outputs
datestamp = "__2020_06_16"
resourcefilepath = resources
outputpath = Path("./outputs")

# %% Run the Simulation
sim = run()

# Births over time (Model)
# define the dataframe
df = sim.population.props

# select babies born during simulation
number_of_ever_newborns = df.loc[df.date_of_birth.notna() & (df.mother_id >= 0)]

# getting total number of newborns per year
total_births_per_year = pd.DataFrame(data=number_of_ever_newborns['date_of_birth'].dt.year.value_counts())
total_births_per_year.sort_index(inplace=True)
total_births_per_year.rename(columns={'date_of_birth': 'total_births'}, inplace=True)
births_model = total_births_per_year.reset_index()
births_model.rename(columns={'index': 'year'}, inplace=True)

# Aggregate the model outputs into five year periods:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births_model["Period"] = births_model["year"].map(calendar_period_lookup)
births_model = births_model.groupby(by='Period')['total_births'].sum()
births_model.index = births_model.index.astype(make_calendar_period_type())

# Births over time (WPP)
wpp = pd.read_csv(Path(resourcefilepath) / "ResourceFile_TotalBirths_WPP.csv")
wpp = wpp.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
wpp.index = wpp.index.astype(make_calendar_period_type())
wpp.columns = 'WPP_' + wpp.columns

# Births in 2018 Census
cens = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Births_2018Census.csv")
cens_per_5y_per = cens['Count'].sum() * 5
# Merge in model results
births = wpp.copy()
births['Model'] = births_model
births['Census'] = np.nan
births.at[cens['Period'][0], 'Census'] = cens_per_5y_per

# Plot:
cens_period = cens['Period'][0]
ax = births.plot.line(y=['Model',  'WPP_Estimates', 'WPP_Medium variant'])
births.plot.line(
    y=['Census'],
    marker='^',
    color='red',
    ax=ax
)
plt.xticks(np.arange(len(births.index)), births.index)
ax.fill_between(births.index, births['WPP_Low variant'], births['WPP_High variant'], facecolor='green', alpha=0.2)
plt.xticks(rotation=90)
ax.set_title('Number of Births Per Calendar Period')
ax.legend(loc='upper left')
ax.set_xlabel('Calendar Period')
ax.set_ylabel('Number per period')
plt.savefig(outputpath / ("Births_Over_Time_" + datestamp + ".pdf"), format='pdf')
plt.tight_layout()
plt.show()
