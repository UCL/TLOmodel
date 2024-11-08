"""
Plot to demonstrate calibration of simplified births module to the Census and WPP data.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import make_calendar_period_lookup, make_calendar_period_type
from tlo.methods import demography, simplified_births

# Path to the resource files
resourcefilepath = Path("./resources")


def run():
    # configuring outputs
    log_config = {
        "filename": "simplified_births_calibrations",
        "custom_levels": {"*": logging.FATAL},
    }

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2099, 12, 31)
    pop_size = 20_000

    sim = Simulation(start_date=start_date, log_config=log_config, resourcefilepath=resourcefilepath)

    # Registering all required modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths()
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


def get_scaling_ratio(sim):
    cens_tot = pd.read_csv(Path(resourcefilepath) / 'demography' / "ResourceFile_PopulationSize_2018Census.csv")[
        'Count'].sum()
    cens_yr = 2018
    cens_date = Date(cens_yr, 7, 1)  # notional date for census at midpoint of the census year.
    assert sim.date >= cens_date, "Cannot scale if simulation does not include the census date"

    # Compute number of people alive in the year of census
    df = sim.population.props
    alive_in_cens_yr = \
        ~df.date_of_birth.isna() & \
        (df.date_of_birth <= cens_date) & \
        ~(df.date_of_death < cens_date)
    model_tot = alive_in_cens_yr.sum()

    # Calculate ratio for scaling
    ratio_data_to_model = cens_tot / model_tot

    return ratio_data_to_model


# %% Run the Simulation
sim = run()

# %% Make the plots

# date-stamp to label outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# destination for outputs
outputpath = Path("./outputs")

# Births over time (Model)

# select babies born during simulation
df = sim.population.props
newborns = df.loc[df.date_of_birth.notna() & (df.mother_id >= 0)]

# getting total number of newborns per year in the model
births_model = pd.DataFrame(data=newborns['date_of_birth'].dt.year.value_counts())
births_model.sort_index(inplace=True)
births_model.reset_index(inplace=True)
births_model.rename(columns={'index': 'year', 'date_of_birth': 'total_births'}, inplace=True)

# rescale the number of births in the model (so that the model population sizes matches actual population size)
births_model['total_births'] *= get_scaling_ratio(sim)

# Aggregate the model outputs into five year periods:
(__tmp__, calendar_period_lookup) = make_calendar_period_lookup()
births_model["Period"] = births_model["year"].map(calendar_period_lookup)
births_model = births_model.groupby(by='Period')['total_births'].sum()
births_model.index = births_model.index.astype(make_calendar_period_type())

# Births over time (WPP)
wpp = pd.read_csv(Path(resourcefilepath) / 'demography' / "ResourceFile_TotalBirths_WPP.csv")
wpp = wpp.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
wpp.index = wpp.index.astype(make_calendar_period_type())
wpp.columns = 'WPP_' + wpp.columns

# Births in 2018 Census
cens = pd.read_csv(Path(resourcefilepath) / 'demography' / "ResourceFile_Births_2018Census.csv")
cens_per_5y = cens['Count'].sum() * 5

# Merge in model results
births = wpp.copy()
births['Model'] = births_model
births['Census'] = np.nan
births.at[cens['Period'][0], 'Census'] = cens_per_5y

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
