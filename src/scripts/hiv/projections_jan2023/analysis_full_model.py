"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
# import random
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 100

scenario = 0

# set up the log config
log_config = {
    "filename": "tb_transmission_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        # "tlo.methods.healthsystem.summary": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 32  # set seed for reproducibility

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(*fullmodel(
    resourcefilepath=resourcefilepath,
    use_simplified_births=False,
    symptommanager_spurious_symptoms=True,
    healthsystem_disable=False,
    healthsystem_mode_appt_constraints=0,  # no constraints
    healthsystem_cons_availability="default",  # if "all", all cons always available
    healthsystem_beds_availability="all",  # all beds always available
    healthsystem_ignore_priority=False,  # ignore priority in HSI scheduling
    healthsystem_use_funded_or_actual_staffing="funded_plus",  # daily capabilities of staff
    healthsystem_capabilities_coefficient=1.0,  # if 'None' set to ratio of init 2010 pop
    healthsystem_record_hsi_event_details=False
))

# # set the scenario
# sim.modules["Hiv"].parameters["beta"] = 0.135999
sim.modules["Tb"].parameters["beta"] = 0.3
# sim.modules["Tb"].parameters["scenario"] = scenario
# sim.modules["Tb"].parameters["scenario_start_date"] = Date(2023, 1, 1)
#
# # to cluster tests in positive people
# sim.modules["Hiv"].parameters["rr_test_hiv_positive"] = 10  # default 1.5
# # to account for people starting-> defaulting, or not getting cons
# # this not used now if perfect referral testing->treatment
# sim.modules["Hiv"].parameters["treatment_initiation_adjustment"] = 3  # default 1.5
# # assume all defaulting is due to cons availability
# sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 1.0
# sim.modules["Hiv"].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1.0

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)
