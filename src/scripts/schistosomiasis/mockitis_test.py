import datetime
import logging
import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
)
from tlo.methods import mockitis

# Where will output go
outputpath = ""
# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
#resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 100

# Establish the simulation object
sim = Simulation(start_date = start_date)

# # Establish the logger
# logfile = outputpath + "LogFile" + datestamp + ".log"
#
# if os.path.exists(logfile):
#     os.remove(logfile)
# fh = logging.FileHandler(logfile)
# fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
# fh.setFormatter(fr)
# logging.getLogger().addHandler(fh)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(mockitis.Mockitis())

# Run the simulation and flush the logger
sim.seed_rngs(0)
# initialise the population first
sim.make_initial_population(n=popsize)
df = sim.population.props


# start the simulation
sim.simulate(end_date=end_date)
