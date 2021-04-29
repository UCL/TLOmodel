from pathlib import Path
import numpy as np
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# In this file I run the model with a large population twice with the same seed, storing the model run as it's
# log file and then see if the two logfiles are the same for all the logs which RTI will produce/interact with.
# Obviously all the logfiles should be the same if the seed is the same, but I don't know how best to check every
# bit of logging.
# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
# Set up large population size
pop_size = 1000000
# Run simulation n times

# Create storage for logfiles
logfiles_storage = dict()
# create a seed to be used in every logfile
Seed = int(np.random.uniform(1, 10000000, 1))

sim = Simulation(start_date=start_date, seed=Seed)
# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
    )
# name the logfile
logfile = sim.configure_logging(filename="LogFile")
# create and run the simulation
sim.make_initial_population(n=pop_size)
# run the simulation
sim.simulate(end_date=end_date)
# parse the logfile
log_df = parse_log_file(logfile)
# store the logfile
logfiles_storage = log_df
df = sim.population.props
df.to_csv('Documents/big_df.csv')
print(df.head())
