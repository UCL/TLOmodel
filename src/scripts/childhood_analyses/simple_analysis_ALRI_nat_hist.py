"""
This will run the ALRI Module + create analysis plots
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
    ALRI,
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
        "tlo.methods.dx_algorithm_child": logging.INFO
    }
}

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
pop_size = 100

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
resources = Path('./resources')

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
    ALRI.ALRI(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resources)
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
output = parse_log_file(sim.log_filepath)

# model outputs -----------------------
complications_per_year = output['tlo.methods.pneumonia']['alri_complications']['count']
print(complications_per_year)
# temp_df = pd.DataFrame()
# for k, v in complications_per_year.items():
#     print(k, v)
#     for k1, v1 in v.items():
#         print(k1, v1)
#         temp_df[f'{k1}'] = [v1]
# print(temp_df)

complications_per_year_flattened = [{**{'dict_key': k}, **v} for k, v in complications_per_year.items()]
complicat_per_year_df = pd.DataFrame(complications_per_year_flattened)
complicat_per_year_df = complicat_per_year_df.set_index('dict_key')

print(complicat_per_year_df)

# create plots -----------------------
# complications_per_year_df['year'] = pd.to_datetime(complications_per_year_df['date']).dt.year

plt.style.use("ggplot")

# ALRI complications
plt.plot.bar(complicat_per_year_df)
plt.title("Complications per year")
plt.xlabel("Year")
plt.ylabel("Number of ALRI complications")
# plt.xticks(rotation=90)
# plt.legend(["Model"], loc="upper left")

plt.show()
