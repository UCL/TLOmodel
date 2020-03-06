import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    symptommanager,
)
from tlo.methods.depression import compute_key_outputs_for_last_3_years

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2010, 7, 1)
popsize = 10000

sim = Simulation(start_date=start_date)
sim.seed_rngs(0)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(
    resourcefilepath=resourcefilepath,
    disable=True
))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(depression.Depression(resourcefilepath=resourcefilepath))

# Establish the logger
logfile = sim.configure_logging(filename="LogFile")

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
