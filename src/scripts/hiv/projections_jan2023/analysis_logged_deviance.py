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
from tlo.methods import (  # deviance_measure,
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
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py
end_date = Date(2012, 1, 1)
popsize = 500

# set up the log config
log_config = {
    "filename": "test_runs_0509",
========
end_date = Date(2022, 1, 1)
popsize = 5000

# scenario = 1

# set up the log config
log_config = {
    "filename": "test_runs",
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py
========
        # "tlo.methods.demography.detail": logging.WARNING,
        # "tlo.methods.healthsystem.summary": logging.INFO,
        # "tlo.methods.healthsystem": logging.INFO,
        # "tlo.methods.healthburden": logging.INFO,
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py

========
# seed = 41728  # set seed for reproducibility
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="funded_plus",  # actual: use numbers/distribution of staff available currently
========
        mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="actual",  # actual: use numbers/distribution of staff available currently
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py
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
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py
sim.modules["Tb"].parameters["probability_community_chest_xray"] = 0.001
sim.modules["Tb"].parameters["outreach_xray_start_date"] = Date(2010, 1, 1)

========
sim.modules["Hiv"].parameters["do_scaleup"] = True
sim.modules["Hiv"].parameters["scaleup_start_year"] = 2019
# sim.modules["Tb"].parameters["scenario"] = scenario
# sim.modules["Tb"].parameters["scenario_start_date"] = Date(2010, 1, 1)
# sim.modules["Tb"].parameters["scenario_SI"] = "z"

# sim.modules["Tb"].parameters["rr_tb_hiv"] = 5  # default 13
# rr relapse if HIV+ 4.7
# sim.modules["Tb"].parameters["rr_tb_aids"] = 26  # default 26

# to cluster tests in positive people
# sim.modules["Hiv"].parameters["rr_test_hiv_positive"] = 1.1  # default 1.5

# to account for people starting-> defaulting, or not getting cons
# this not used now if perfect referral testing->treatment
# affects the prob of art start once diagnosed
# sim.modules["Hiv"].parameters["treatment_initiation_adjustment"] = 1  # default 1.5

# assume all defaulting is due to cons availability
# sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 1.0
# sim.modules["Hiv"].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1.0
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# # save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
# with open(outputpath / "default_run0509.pickle", "wb") as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)
#
# # load the results
# with open(outputpath / "default_run.pickle0509", "rb") as f:
#     output = pickle.load(f)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run0509x.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
<<<<<<<< HEAD:src/scripts/hiv/analysis_test_runs.py
with open(outputpath / "default_run0509x.pickle", "rb") as f:
    output = pickle.load(f)

========
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)
>>>>>>>> master:src/scripts/hiv/projections_jan2023/analysis_logged_deviance.py
