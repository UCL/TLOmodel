"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    deviance_measure,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 5000
# todo check parameters set below

# set up the log config
log_config = {
    "filename": "deviance_calibrated",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        # "tlo.methods.deviance_measure": logging.INFO,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 3  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=True,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        disable=True,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        store_hsi_events_that_have_run=False,  # convenience function for debugging
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
    # deviance_measure.Deviance(resourcefilepath=resourcefilepath),
)

# todo remove
sim.modules["Hiv"].parameters["beta"] = 0.127623113

# transmission rate active cases -> new latent cases
sim.modules["Tb"].parameters["transmission_rate"] = 19.5

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)


###################################################
""" load the outputs from a simulation and plot the results with comparison data """

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

cov_over_time = output["tlo.methods.hiv"]["hiv_program_coverage"]
cov_over_time = cov_over_time.set_index("date")

dx = cov_over_time["dx_adult"] * 100
art_among_dx = (cov_over_time["art_coverage_adult"] / cov_over_time["dx_adult"]) * 100
vs_among_art = (cov_over_time["art_coverage_adult_VL_suppression"]) * 100
print(dx)  # want 95% diagnosed from 2022 onwards
