import datetime
import os
import logging
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, lifestyle, mockitis, chronicsyndrome, qaly
from tlo.methods import healthsystem


# Where will output go
outputpath = ''

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources/'


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


# make a dataframe that contains the switches for which interventions are allowed or not allowed during this run.
# NB. These must use the exact 'registered strings' that the disease modules allow

service_availability=pd.DataFrame(data=[],columns=['Service','Available'])
service_availability.loc[0]=['Mockitis_Treatment',True]
service_availability.loc[1]=['ChronicSyndrome_Treatment',False]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
sim.register(lifestyle.Lifestyle())
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())
sim.register(qaly.QALY(resourcefilepath=resourcefilepath)) # NB.This relies on the health system module having been registered first.


# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


# %% read the results
# output = parse_log_file(logfile)


