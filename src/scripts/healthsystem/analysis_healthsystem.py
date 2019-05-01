import datetime
import logging
import os

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthsystem, lifestyle, qaly, hiv, \
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

logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.lifestyle').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.qaly').setLevel(logging.WARNING)
logging.getLogger('tlo.methods.hiv').setLevel(logging.DEBUG)
logging.getLogger('tlo.methods.tb').setLevel(logging.DEBUG)

params = [0.2, 0.05]  # sample params for runs

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.hiv(resourcefilepath=resourcefilepath, par_est=params[0]))
sim.register(tb.tb(resourcefilepath=resourcefilepath))
sim.register(male_circumcision.male_circumcision(resourcefilepath=resourcefilepath, par_est5=params[1]))
sim.register(hiv_behaviour_change.BehaviourChange())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
fh.close()


# # %% read the results
# output = parse_log_file(logfile)
