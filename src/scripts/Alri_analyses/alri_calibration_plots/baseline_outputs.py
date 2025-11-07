"""
This is a script analysis of ALRI with default parameters, set the baseline outputs
"""
# %% Import Statements and initial declarations
import datetime
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

log_filename = outputpath / 'baseline_alri_outputs__2022-03-16T165848.log'
# <-- insert name of log file to avoid re-running the simulation // GBD_lri_comparison_50k_pop__2022-03-15T111444.log

if not os.path.exists(log_filename):
    # If logfile does not exists, re-run the simulation:
    # Do not run this cell if you already have a logfile from a simulation:

    start_date = Date(2010, 1, 1)
    end_date = Date(2025, 12, 31)
    popsize = 5000

    log_config = {
        "filename": "baseline_alri_outputs",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.DEBUG,
            "tlo.methods.demography": logging.INFO,
            "tlo.methods.healthburden": logging.INFO,
        }
    }

    seed = random.randint(0, 50000)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                     show_progress_bar=True, resourcefilepath=resourcefilepath)

    # run the simulation
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        simplified_births.SimplifiedBirths(),
        healthsystem.HealthSystem(service_availability=['*']),
        alri.Alri(),
        alri.AlriPropertiesOfOtherModules()
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # display filename
    log_filename = sim.log_filepath
    print(f"log_filename: {log_filename}")

output = parse_log_file(log_filename)

# ----------------------------------------------------------------------------------------
# Set the baseline output plots
counts = output['tlo.methods.alri']['event_counts']
counts['year'] = pd.to_datetime(counts['date']).dt.year
counts.drop(columns='date', inplace=True)
counts.set_index(
    'year',
    drop=True,
    inplace=True
)

mean_per_year = counts.sum() / counts.index.value_counts().count()

fig, ax = plt.subplots()
labels = mean_per_year.index.values
ax.bar(labels, mean_per_year)
plt.title('key events')
plt.xticks(rotation=90)
plt.ylabel('Number of cases')
fig.tight_layout()
plt.show()
