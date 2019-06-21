import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthsystem, lifestyle, epilepsy, healthburden

# Where will output go
outputpath = './src/scripts/epilepsy_analyses/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
# logfile = outputpath + 'LogFile' + datestamp + '.log'

# if os.path.exists(logfile):
#    os.remove(logfile)
# fh = logging.FileHandler(logfile)
# fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
# fh.setFormatter(fr)
# logging.getLogger().addHandler(fh)

# logging.getLogger('tlo.methods.Depression').setLevel(logging.DEBUG)


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
             ignore_appt_constraints=True,
             ignore_cons_constraints=True
             ))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(epilepsy.Epilepsy(resourcefilepath=resourcefilepath))

# Run the simulation and flush the logger
# sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


