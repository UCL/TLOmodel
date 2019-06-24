import datetime
import logging
import os

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthsystem, lifestyle, healthburden, hiv, \
    male_circumcision, hiv_behaviour_change, tb

# Where will output go
outputpath = './src/scripts/outputLogs/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources/'

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 2000

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

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ['*']

logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.lifestyle').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.healthburden').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.hiv').setLevel(logging.INFO)
logging.getLogger('tlo.methods.tb').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.male_circumcision').setLevel(logging.WARNING)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.hiv(resourcefilepath=resourcefilepath))
sim.register(tb.tb(resourcefilepath=resourcefilepath))
sim.register(male_circumcision.male_circumcision(resourcefilepath=resourcefilepath))
sim.register(hiv_behaviour_change.BehaviourChange())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
fh.close()


# # %% read the results
# output = parse_log_file(logfile)
