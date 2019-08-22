import datetime
import logging
import os
import time


from tlo import Date, Simulation
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    hiv,
    lifestyle,
    male_circumcision,
    tb,
)

# Where will output go
outputpath = './src/scripts/tb/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = "./resources/"

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 500

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
service_availability = ["*"]

t0 = time.time()

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.hiv(resourcefilepath=resourcefilepath))
sim.register(tb.tb(resourcefilepath=resourcefilepath))
sim.register(male_circumcision.male_circumcision(resourcefilepath=resourcefilepath))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger("tlo.methods.tb").setLevel(logging.INFO)
logging.getLogger("tlo.methods.hiv").setLevel(logging.INFO)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

t1 = time.time()
print('Time taken, mins', (t1 - t0) / 60)

# %% read the results
# out = sim.population.props
# out.to_csv(r'C:\Users\Tara\Documents\TLO\outputs.csv', header=True)
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
# from tlo.analysis.utils import parse_log_file
#
#
# #
# # outputpath = './src/scripts/tb/'
# # datestamp = datetime.date.today().strftime("__%Y_%m_%d")
# #
# # logfile = outputpath + 'LogFile' + datestamp + '.log'
# ##
# logfile = './src/scripts/tb/LogFile__2019_08_22.log'
#
# output = parse_log_file(logfile)
#
# tb_df = output['tlo.methods.tb']['tb_prevalence']
