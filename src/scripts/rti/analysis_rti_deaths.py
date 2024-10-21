import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# To reproduce the results, you must set the seed for the Simulation instance. The Simulation
# will seed the random number generators for each module when they are registered.
# If a seed argument is not given, one is generated. It is output in the log and can be
# used to reproduce results of a run
seed = 100

log_config_no_hs = {
    "filename": "rti_analysis_no_healthsystem",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
    }
}
log_config_with_hs = {
    "filename": "rti_analysis_with_healthsystem",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
    }
}
start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
pop_size = 3000

# Path to the resource files used by the disease and intervention methods
# resourcefilepath = Path('./resources')
resourcefilepath = './resources'

# Creat simulations both with and without the health system
sim_no_health_system = Simulation(start_date=start_date, seed=seed, log_config=log_config_no_hs,
                                  resourcefilepath=resourcefilepath)

# Used to configure health system behaviour
service_availability_no_hs = []
# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
# Register modules used in each model run, specifying the availability of service from the hs
sim_no_health_system.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(service_availability=service_availability_no_hs),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        rti.RTI(),
        simplified_births.SimplifiedBirths()
        )

# create and run each simulation
sim_no_health_system.make_initial_population(n=pop_size)
sim_no_health_system.simulate(end_date=end_date)

sim_with_health_system = Simulation(start_date=start_date, seed=seed, log_config=log_config_with_hs,
                                    resourcefilepath=resourcefilepath)

service_availability_with_hs = ['*']

sim_with_health_system.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(service_availability=service_availability_with_hs),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        rti.RTI(),
        simplified_births.SimplifiedBirths()
        )

sim_with_health_system.make_initial_population(n=pop_size)
sim_with_health_system.simulate(end_date=end_date)
# parse the simulation logfile to get the output dataframes
log_df_no_hs = parse_log_file(sim_no_health_system.log_filepath)
log_df_with_hs = parse_log_file(sim_with_health_system.log_filepath)
# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #
# get the incidence of rti per 100,000 and the incidence of death per 100,000 in each simulation
inc_no_hs = log_df_no_hs["tlo.methods.rti"]["summary_1m"]["incidence of rti per 100,000"]
inc_with_hs = log_df_with_hs["tlo.methods.rti"]["summary_1m"]["incidence of rti per 100,000"]
inc_death_no_hs = log_df_no_hs["tlo.methods.rti"]["summary_1m"]["incidence of rti death per 100,000"]
inc_death_with_hs = log_df_with_hs["tlo.methods.rti"]["summary_1m"]["incidence of rti death per 100,000"]
# ------------------------------------- PLOTS  ------------------------------------- #
# calculate the mean incidence of RTI in each simulation
mean_inc_rti_no_hs = np.mean(inc_no_hs)
mean_inc_rti_with_hs = np.mean(inc_with_hs)
# calculate the mean incidence of RTI death in each simulation
mean_inc_rti_death_no_hs = np.mean(inc_death_no_hs)
mean_inc_rti_death_with_hs = np.mean(inc_death_with_hs)
plt.bar([1, 2], [mean_inc_rti_death_no_hs, mean_inc_rti_death_with_hs], color='lightsteelblue')
plt.xticks([1, 2], ['Incidence of death \nwithout healthsystem', 'Incidence of death \nwith healthsystem'])
plt.ylabel('Mean incidence of death per 100,000')
plt.title('Incidence of RTI death with and without the health system')
plt.show()
