from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
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

log_config = {
    "filename": "rti_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
pop_size = 5000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
# resources = "./resources"
resourcefilepath = Path('./resources')

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
        )

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)
# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

model_rti = log_df["tlo.methods.rti"]["summary_1m"]["incidence of rti per 100,000"]
model_date = log_df["tlo.methods.rti"]["summary_1m"]["date"]
# ------------------------------------- PLOTS  ------------------------------------- #
comparison = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=resourcefilepath)
gbd_deaths_2010_2014 = comparison.loc[('2010-2014')]['GBD_mean'].sum()
gbd_deaths_2015_2019 = comparison.loc[('2015-2019')]['GBD_mean'].sum()
gbd_deaths = gbd_deaths_2010_2014 + gbd_deaths_2015_2019
model_deaths_2010_2014 = comparison.loc[('2010-2014')]['model'].sum()
model_deaths_2015_2019 = comparison.loc[('2015-2019')]['model'].sum()
model_deaths = model_deaths_2010_2014 + model_deaths_2015_2019
plt.bar(np.arange(2), [gbd_deaths, model_deaths], color=['lightsalmon', 'lightsteelblue'])
plt.ylabel('Number of Deaths')
plt.xticks(np.arange(2), ['GBD', 'TLO'])
plt.title('RTI deaths estimated by the GBD and TLO model, 2010-2019')
plt.show()
plt.clf()
plt.style.use("ggplot")

# Measles incidence
plt.subplot(111)  # numrows, numcols, fignum
plt.plot(model_date, model_rti)
plt.title("RTI incidence")
plt.xlabel("Date")
plt.ylabel("Incidence per 100,000 person years")
plt.xticks(rotation=90)
plt.legend(["Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()
