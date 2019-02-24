import datetime
import os
import logging

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, lifestyle, mockitis, chronicsyndrome
from tlo.methods import healthsystem


# Where will output go
outputpath = '/Users/tbh03/Dropbox (SPH Imperial College)/TLO Model Output/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
resourcefilepath = '/Users/tbh03/PycharmProjects/TLOmodel/resources/'


start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50


# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp  +'.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

logging.getLogger('tlo.methods.Demography').setLevel(logging.DEBUG)


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem())
sim.register(lifestyle.Lifestyle())
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())



# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


# %% read the results
# output = parse_log_file(logfile)


