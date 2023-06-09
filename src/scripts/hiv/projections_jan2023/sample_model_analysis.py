"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
# import random
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    tb,
    symptommanager,
    hiv, healthburden,
    simplified_births,
    healthsystem,
    epi,
    enhanced_lifestyle,
    healthseekingbehaviour,
)

# from tlo.methods.fullmodel import fullmodel

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
popsize = 5000
scenario = 0

# set up the log config
# add deviance measure logger if needed
log_config = {
    "filename": "tb_transmission_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.labour.detail": logging.WARNING,  # this logger keeps outputting even when set to warning
    },
}

# Register the appropriate modules
seed = 2025  # set seed for reproducibility

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)

sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
   healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
   )

# set the scenario
# sim.modules["Tb"].parameters["probability_community_chest_xray"] = 0.6
sim.modules["Tb"].parameters["scenario"] = 0
sim.modules["Tb"].parameters["scenario_start_date"] = start_date

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "Tb_baseline.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

with open(outputpath / "Tb_baseline.pickle", "rb") as f:
    output = pickle.load(f)
