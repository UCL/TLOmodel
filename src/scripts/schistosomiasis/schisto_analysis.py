"""
this file runs the malaria module and outputs graphs with data for comparison
"""
import datetime
import pickle
import time
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    schisto,
    simplified_births,
    symptommanager,
    epi,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore all FutureWarnings

t0 = time.time()

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2024, 1, 1)
popsize = 1000

# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.schisto": logging.INFO,
        "tlo.methods.enhanced_lifestyle": logging.INFO,
        # "tlo.methods.healthsystem.summary": logging.INFO,
    },
}
seed = 20
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Register the appropriate modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],
        mode_appt_constraints=1,
        cons_availability='default',
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=False,
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    schisto.Schisto(
        resourcefilepath=resourcefilepath,
    ),
    epi.Epi(
        resourcefilepath=resourcefilepath,
    )
)


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "schisto_test.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

with open(outputpath / "schisto_test.pickle", "rb") as f:
    output = pickle.load(f)
