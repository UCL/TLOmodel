import datetime
import logging
import os

import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, healthsystem, lifestyle, qaly, hiv, \
    male_circumcision, tb_hs_engagement, hiv_behaviour_change, tb

# Where will output go
outputpath = './src/scripts/output logs/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources/'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10000

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
logging.getLogger('tlo.methods.hiv').setLevel(logging.INFO)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow

service_availability = pd.DataFrame(data=[], columns=['Service', 'Available'])
service_availability.loc[0] = ['Mockitis_Treatment', True]
service_availability.loc[1] = ['ChronicSyndrome_Treatment', True]
service_availability['Service'] = service_availability.astype('object')
service_availability['Available'] = service_availability.astype('bool')

params = [0.2, 0.1, 0.05, 0.4, 0.5, 0.05]  # sample params for runs

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.hiv(resourcefilepath=resourcefilepath, par_est=params[0], par_est1=params[1], par_est2=params[2],
                     par_est3=params[3], par_est4=params[4]))
sim.register(tb.tb_baseline(resourcefilepath=resourcefilepath))
sim.register(male_circumcision.male_circumcision(resourcefilepath=resourcefilepath, par_est5=params[5]))
sim.register(hiv_behaviour_change.BehaviourChange())
sim.register(tb_hs_engagement.health_system_tb())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


# %% read the results
# output = parse_log_file(logfile)
