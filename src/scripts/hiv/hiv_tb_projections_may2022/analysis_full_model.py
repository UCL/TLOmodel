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
end_date = Date(2015, 1, 1)
popsize = 1000

# todo
scenario = 0
mode_appt_constraints = 0  # HR constraints, 0: no constraints,
            # all HSI events run with no squeeze factor, 1: elastic constraints, all HSI
            # events run with squeeze factor, 2: hard constraints, only HSI events with
            # no squeeze factor run.
cons_availability = "all"  # consumable constraints, default=use cons listing, all=everything available
ignore_priority = True  # if True, do not use the priority information in HSI event to schedule


# scenario = 2
# mode_appt_constraints = 0  # HR constraints, 0=no constraints, 2=hard constraints
# cons_availability = "default"  # consumable constraints, default=use cons listing, all=everything available
# ignore_priority = True  # if True, use the priority information in HSI event to schedule


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
seed = 5  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(*fullmodel(
    resourcefilepath=resourcefilepath,
    use_simplified_births=False,
    healthsystem_disable=False,
    healthsystem_mode_appt_constraints=mode_appt_constraints,
    healthsystem_cons_availability=cons_availability,
    healthsystem_ignore_priority=ignore_priority,
    healthsystem_capabilities_coefficient=1.0,
))

# set the scenario
sim.modules["Tb"].parameters["scenario"] = scenario

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)
