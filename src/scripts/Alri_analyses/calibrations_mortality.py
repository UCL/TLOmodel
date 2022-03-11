"""
This is the analysis script for the calibration of the ALRI model
"""
# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
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

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2019, 12, 31)
popsize = 10000

log_config = {
    "filename": "alri_model_calibartions",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.alri": logging.DEBUG,
        "tlo.methods.demography": logging.INFO,
    }
}

# Establish the simulation object
sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

# run the simulation
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
    alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=22),  # choose to log an individual
    alri.AlriPropertiesOfOtherModules()
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the output:
output = parse_log_file(sim.log_filepath)

# %% ----------------------------  Classification and treatment ----------------------------
# Map the classifications with the treatment given
classification = output['tlo.methods.alri']['classification_and_treatment']

# get model output dates in correct format
classification['date'] = pd.to_datetime(classification.date)
classification['year'] = classification['date'].dt.year
start_date = 2010
end_date = 2019

fig, ax = plt.subplots(figsize=(8, 6))
plotting = classification.groupby('classification').plot(kind='kde', ax=ax)

#
# classification['year'] = pd.to_datetime(stats_incidence['date']).dt.year
# stats_incidence.drop(columns='date', inplace=True)
# stats_incidence.set_index(
#     'year',
#     drop=True,
#     inplace=True
# )
