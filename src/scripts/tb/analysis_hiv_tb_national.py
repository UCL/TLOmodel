"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
from pathlib import Path
# import random

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
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
end_date = Date(2020, 1, 1)
popsize = 5000

# set up the log config
log_config = {
    "filename": "Logfile",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 4  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        ignore_cons_constraints=True,  # mode for consumable constraints (if ignored, all consumables available)
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
)

# change IPT high-risk districts to all districts for national-level model
# all_districts = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='all_districts')
# sim.modules['Tb'].parameters['tb_high_risk_distr'] = all_districts.district_name

sim.modules["Tb"].parameters["tb_high_risk_distr"] = pd.read_excel(
    resourcefilepath / "ResourceFile_TB.xlsx", sheet_name="all_districts"
)

# change tb mixing parameter to allow more between-district transmission
sim.modules["Tb"].parameters["mixing_parameter"] = 1
sim.modules["Hiv"].parameters["rr_test_hiv_positive"] = 0.5
sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 0.98
sim.modules["Hiv"].parameters["prob_start_art_after_hiv_test"] = 0.3
sim.modules["Hiv"].parameters["temporal_trend_art_initiation"] = 1.5


sim.modules["Hiv"].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1
sim.modules["Hiv"].parameters["probability_of_seeking_further_art_appointment_if_appointment_not_available"] = 1

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
# with open(outputpath / "poll_plus_hsi_generic.pickle", "wb") as f:
# with open(outputpath / "poll_plus_anc.pickle", "wb") as f:
# with open(outputpath / "only_hiv_poll.pickle", "wb") as f:
# with open(outputpath / "schedule_2010_tests.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
