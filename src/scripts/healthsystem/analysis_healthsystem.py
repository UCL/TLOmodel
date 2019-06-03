import datetime
import logging
import os

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthsystem, lifestyle, mockitis, qaly, chronicsyndrome

# [NB. Working directory must be set to the root of TLO: TLOmodel/]

# Where will output go
outputpath = ''

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 50

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

logging.getLogger('tlo.methods.Demography').setLevel(logging.DEBUG)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed (can also use *). Empoty list means nothing allowed
# (This can be set to 'all' or 'none'; and it will allow any treatment_id that begins with a stub)
service_availability = '*'


# -----------------------------------------------------------------

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()

# %% read the results
output = parse_log_file(logfile)
