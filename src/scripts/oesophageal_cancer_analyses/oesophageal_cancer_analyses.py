import datetime
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophageal_cancer,
)

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

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
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(oesophageal_cancer.Oesophageal_Cancer(resourcefilepath=resourcefilepath))

# Run the simulation and flush the logger
# sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# fh.flush()
