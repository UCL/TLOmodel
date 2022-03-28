"""
This will demonstrate the effect of different treatment.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

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
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# Define the 'service_availability' parameter for two sceanrios (without treatment, with treatment)
scenarios = dict()
scenarios['No_pulse_oximeter'] = []
scenarios['pulse_oximeter'] = ['*']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2019, 12, 31)
popsize = 5000

for label, service_avail in scenarios.items():

    log_config = {
        "filename": "alri_with_and_without_pulse_oximeter",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.INFO,
            "tlo.methods.demography": logging.INFO,
        }
    }

    if service_avail == []:
        _disable = False
        _disable_and_reject_all = True
    else:
        _disable = True
        _disable_and_reject_all = False

    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=_disable,
                                  disable_and_reject_all=_disable_and_reject_all),

        alri.Alri(resourcefilepath=resourcefilepath),
        alri.AlriPropertiesOfOtherModules()
    )
    sim.modules['Alri'].parameters['availability']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath
