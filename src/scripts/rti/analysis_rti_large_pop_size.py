from pathlib import Path
import numpy as np
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
    dx_algorithm_child,
    dx_algorithm_adult
)
import pandas as pd

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
nsim = 2
# Create storage for logfiles
logfiles_storage = dict()
Seed = int(np.random.uniform(1, 10000000, 1))
for i in range(0, nsim):
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
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    logfile = sim.configure_logging(filename="LogFile")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = []

    sim.simulate(end_date=end_date)
    log_df = parse_log_file(logfile)
    logfiles_storage[i] = log_df
# Check the RTI logging dfs are all the same per sim
for df in logfiles_storage[0]['tlo.methods.rti'].keys():
    for sim in logfiles_storage:
        print(df)
        try:
            assert logfiles_storage[0]['tlo.methods.rti'][df].equals(
                logfiles_storage[sim]['tlo.methods.rti'][df]), \
                'Something went wrong in rti replicability'
        except AssertionError:
            print('Something went wrong in rti replicability')
            print(df)

# Check the enhanced_lifestyle logging dfs are all the same per sim
for df in logfiles_storage[0]['tlo.methods.enhanced_lifestyle'].keys():
    for sim in logfiles_storage:
        print(df)
        try:
            assert logfiles_storage[0]['tlo.methods.enhanced_lifestyle'][df].equals(
                logfiles_storage[sim]['tlo.methods.enhanced_lifestyle'][df]), \
                'Something went wrong in enhanced_lifestyle replicability'
        except AssertionError:
            print('Something went wrong in enhanced_lifestyle replicability')
            print(df)

# Check the healthsystem logging dfs are all the same per sim
for df in logfiles_storage[0]['tlo.methods.healthsystem'].keys():
    for sim in logfiles_storage:
        print(df)
        try:
            assert logfiles_storage[0]['tlo.methods.healthsystem'][df].equals(
                logfiles_storage[sim]['tlo.methods.healthsystem'][df]), \
                'Something went wrong in healthsystem replicability'
        except AssertionError:
            print('Something went wrong in healthsystem replicability')
            print(df)
# Check the healthburden logging dfs are all the same per sim
for df in logfiles_storage[0]['tlo.methods.healthburden'].keys():
    for sim in logfiles_storage:
        print(df)
        try:
            assert logfiles_storage[0]['tlo.methods.healthburden'][df].equals(
                logfiles_storage[sim]['tlo.methods.healthburden'][df]), \
                'Something went wrong in healthburden replicability'
        except AssertionError:
            print('Something went wrong in healthburden replicability')
            print(df)
# Check the demography logging dfs are all the same per sim
for df in logfiles_storage[0]['tlo.methods.demography'].keys():
    for sim in logfiles_storage:
        print(df)
        try:
            assert logfiles_storage[0]['tlo.methods.demography'][df].equals(
                logfiles_storage[sim]['tlo.methods.demography'][df]), \
                'Something went wrong in demography replicability'
        except AssertionError:
            print('Something went wrong in demography replicability')
            print(df)

# Something odd happens in the logfiles_storage[0]['tlo.methods.demography']['person_years'] dictionaries, where
# in one simulation the age range goes up to 102, and in the other it goes up to 100, meaning that the above throws up
# an assertion error
