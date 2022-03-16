# import datetime
# import os
# import time
from pathlib import Path

import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    measles,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# import pandas as pd


# To reproduce the results, you must set the seed for the Simulation instance. The Simulation
# will seed the random number generators for each module when they are registered.
# If a seed argument is not given, one is generated. It is output in the log and can be
# used to reproduce results of a run
seed = 100

log_config = {
    "filename": "measles_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.measles": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    }
}

start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
pop_size = 500

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
# resources = "./resources"
resources = Path('./resources')

# Used to configure health system behaviour
service_availability = ["*"]

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    healthsystem.HealthSystem(
        resourcefilepath=resources,
        service_availability=service_availability,
        mode_appt_constraints=0,
        ignore_cons_constraints=True,
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=True,
    ),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    healthburden.HealthBurden(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    epi.Epi(resourcefilepath=resources),
    measles.Measles(resourcefilepath=resources),
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

model_measles = log_df["tlo.methods.measles"]["incidence"]["inc_1000people"]
model_date = log_df["tlo.methods.measles"]["incidence"]["date"]
# ------------------------------------- PLOTS  ------------------------------------- #

plt.style.use("ggplot")

# Measles incidence
plt.subplot(111)  # numrows, numcols, fignum
plt.plot(model_date, model_measles)
plt.title("Measles incidence")
plt.xlabel("Date")
plt.ylabel("Incidence per 1000py")
plt.xticks(rotation=90)
plt.legend(["Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()
