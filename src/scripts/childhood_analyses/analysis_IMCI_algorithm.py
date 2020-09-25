"""
This will run the DxAlgorithmChild Module
"""
# %% Import Statements and initial declarations
import datetime
import os
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    pneumonia,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

seed = 123

log_config = {
    "filename": "imci_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.pneumonia": logging.INFO,
        "tlo.methods.dx_algorithm_child": logging.DEBUG
    }
}

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
pop_size = 1000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
resources = "./resources"

# Used to configure health system behaviour
service_availability = ["*"]

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources, service_availability=service_availability),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    pneumonia.ALRI(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resources)
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)  # output


def get_pneumonia_management_information(logfile):
    output = parse_log_file(logfile)
    # Calculate the IMCI algorithm from the output counts of ALRI episodes
    counts = output['tlo.methods.pneumonia']['pneumonia_management_child_info']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )
    # create empty dictionary of {'column_name': column_data}, then fill it with all data
    df_data = {}
    for col in ['A', 'B', 'C', 'D']:
        column_name = f'column_{col}'
        column_data = function_that_returns_list_of_data(col)
        df_data[column_name] = column_data

    # convert dictionary into pandas dataframe
    pd.DataFrame(data=df_data, index=index)  # the index here can be left out if not relevant
