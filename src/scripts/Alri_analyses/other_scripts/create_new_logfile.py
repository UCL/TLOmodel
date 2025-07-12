"""
This is a utility script to create a new logfile to use for analysis
"""
# %% Import Statements and initial declarations
import datetime
import os
import random
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

log_filename = outputpath / 'xxxx'
# <-- insert name of log file to avoid re-running the simulation // GBD_lri_comparison_50k_pop__2022-03-15T111444.log
# alri_classification_and_treatment__2022-03-15T170845.log

if not os.path.exists(log_filename):
    # If logfile does not exists, re-run the simulation:
    # Do not run this cell if you already have a logfile from a simulation:

    start_date = Date(2010, 1, 1)
    end_date = Date(2025, 12, 31)
    popsize = 5000

    log_config = {
        "filename": "logfile_for_analysis_default_hs_availability_5kpop",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.DEBUG,
            "tlo.methods.demography": logging.INFO,
            "tlo.methods.healthburden": logging.INFO,
        }
    }

    seed = random.randint(0, 50000)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                     show_progress_bar=True, resourcefilepath=resourcefilepath)

    # run the simulation
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        simplified_births.SimplifiedBirths(),
        healthsystem.HealthSystem(service_availability=['*']),
        # mode_appt_constraints=0,
        # ignore_priority=True,
        # capabilities_coefficient=1.0,
        # disable=True),
        alri.Alri(),
        alri.AlriPropertiesOfOtherModules()
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # display filename
    log_filename = sim.log_filepath
    print(f"log_filename: {log_filename}")

output = parse_log_file(log_filename)
