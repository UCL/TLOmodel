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
    contraception
)
from tlo.methods import schisto

# Where will output go
outputpath = ""
# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
#resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 10000

# Establish the simulation object
sim = Simulation(start_date = start_date)

# Establish the logger
logfile = outputpath + "LogFile" + datestamp + ".log"

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
# service_availability = ["*"]

logging.getLogger("tlo.methods.demography").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.contraception").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.healthburden").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.healthsystem").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.schisto").setLevel(logging.INFO)


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(schisto.Schisto())

# Run the simulation and flush the logger
sim.seed_rngs(0)
# initialise the population first
sim.make_initial_population(n=popsize)
# start the simulation
sim.simulate(end_date=end_date)
fh.flush()

df = sim.population.props

