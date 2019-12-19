import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, epilepsy, healthburden, healthsystem, enhanced_lifestyle

# Where will output go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")


start_date = Date(2010, 1, 1)
end_date = Date(2011, 4, 1)
popsize = 5000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath / ('LogFile' + datestamp + '.log')

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

logging.getLogger('tlo.methods.Demography').setLevel(logging.DEBUG)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       ignore_appt_constraints=True,
                                       ignore_cons_constraints=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(epilepsy.Epilepsy(resourcefilepath=resourcefilepath))


# Run the simulation and flush the logger
# sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()


# %% read the results
output = parse_log_file(logfile)

prop_seiz_stat_0 = pd.Series(
    output['tlo.methods.epilepsy']['summary_stats_per_3m']['prop_seiz_stat_0'].values,
    index=output['tlo.methods.epilepsy']['summary_stats_per_3m']['date'])

prop_seiz_stat_0.plot()
plt.show()
