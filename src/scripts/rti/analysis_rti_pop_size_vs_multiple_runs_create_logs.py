from pathlib import Path

from tlo import Date, Simulation, logging
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

# =============================== Analysis description ========================================================
# Here I am trying to find out if it is better to do single runs with a larger population size vs
# multiple runs with smaller population sizes

# Set up the logging
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
# Create the conditions for the simulations, i.e. how long it runs for, how many people are involved and the
# number of simulations this analysis will be run for, to try and account for some of the variation between simulations
yearsrun = 1
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
smaller_pop_size = 20000
nsim = 5
larger_pop_size = smaller_pop_size * nsim

# clear out previous logs
log_file_location = './outputs/multiple_runs_vs_population_size'

for i in range(0, nsim):
    sim = Simulation(start_date=start_date)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
    )
    # Get the logfile and store it
    logfile = sim.configure_logging(filename="LogFile_multiple_runs",
                                    directory="./outputs/multiple_runs_vs_population_size")
    # make initial population
    sim.make_initial_population(n=smaller_pop_size)
    # alter the rti parameters
    params = sim.modules['RTI'].parameters
    # run the sim
    sim.simulate(end_date=end_date)
# create a single larger run of the model
sim = Simulation(start_date=start_date)
sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
    )
# Get the logfile and store it
logfile = sim.configure_logging(filename="LogFile_large_pop_size",
                                directory="./outputs/multiple_runs_vs_population_size")
# make initial population
sim.make_initial_population(n=larger_pop_size)
# alter the RTI parameters
params = sim.modules['RTI'].parameters
# run the sim
sim.simulate(end_date=end_date)
