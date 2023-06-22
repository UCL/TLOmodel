#################################################
"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path
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
end_date = Date(2013, 12, 31)
popsize = 1000

# set up the log config
log_config = {
    "filename": "DAH_run2",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    },
}
# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
#seed = 41728  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="funded_plus",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False),
    tb.Tb(resourcefilepath=resourcefilepath),
)

# set the scenario
sim.modules["Tb"].parameters["scenario"] =2
#sim.modules["Tb"].parameters["second_line_test"] = "sputum"
#sim.modules["Tb"].parameters["scenario_start_date"] = start_date
#sim.modules["Tb"].parameters["outreach_xray_start_date"] = Date(2011, 1, 1)
sim.modules["Tb"].parameters["probability_access_to_xray"] = 0.0
#sim.modules["Tb"].parameters["probability_community_chest_xray"] = 0.001

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "DAH_run2.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "DAH_run2.pickle", "rb") as f:
    output = pickle.load(f)
