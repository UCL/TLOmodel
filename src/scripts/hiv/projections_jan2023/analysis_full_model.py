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
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)
popsize = 500
# scenario = 0

# set up the log config
# add deviance measure logger if needed
log_config = {
    "filename": "hiv_test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.labour.detail": logging.WARNING,  # this logger keeps outputting even when set to warning
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 3  # set seed for reproducibility


sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(*fullmodel(
    resourcefilepath=resourcefilepath),
 ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=resourcefilepath)

)

# # set the scenario
sim.modules['HealthSystem'].parameters['year_cons_availability_switch'] = 2011
sim.modules['HealthSystem'].parameters['cons_availability_postSwitch'] = 'scenario11'

# sim.modules["Tb"].parameters["scenario"] = scenario
# sim.modules["Tb"].parameters["scenario_start_date"] = Date(2023, 1, 1)
# sim.modules["Malaria"].parameters["type_of_scaleup"] = 'target'
# sim.modules["Malaria"].parameters["scaleup_start_year"] = 2011
#
# sim.modules['ImprovedHealthSystemAndCareSeekingScenarioSwitcher'].parameters['max_healthcare_seeking'] = [False, True]
# sim.modules['ImprovedHealthSystemAndCareSeekingScenarioSwitcher'].parameters['year_of_switch'] = 2011


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
