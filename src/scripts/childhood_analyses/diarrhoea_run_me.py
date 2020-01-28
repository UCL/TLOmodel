"""
@ Ines - whilst we work on this, I think running this file will make it easier to do the debugging.
So let this me the file that we both use to run the model for now.

"""

# %% Import Statements and initial declarations
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import (
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    parse_log_file,
)
from tlo.methods import contraception, demography, diarrhoea, childhood_management, healthsystem, enhanced_lifestyle, \
    symptommanager
from tlo.util import create_age_range_lookup

outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
# resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
resourcefilepath = Path("./resources")

logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 2)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the outputs to file

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
# sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))  ## NB --- this is commented out -- so no health burden information wil come at the moment.
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
# sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath)) ## removing this so remove any health care seeking so Ines can focus on the 'natural history' and 'epidemiology'
sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(childhood_management.ChildhoodManagement(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()
output = parse_log_file(logfile)


# %% -----------------------------------------------------------------------------------
# %% Plot Incidence of Diarrhoea Over time:


# years = mdates.YearLocator()   # every year
# months = mdates.MonthLocator()  # every month
# years_fmt = mdates.DateFormatter('%Y')

# Load Model Results on clinical types of diarrhoea
clinical_type_df = output['tlo.methods.diarrhoea']['clinical_diarrhoea_type']
Model_Years = pd.to_datetime(clinical_type_df.date)
Model_AWD = clinical_type_df.AWD
Model_dysentery = clinical_type_df.dysentery
Model_persistent = clinical_type_df.persistent

fig1, ax = plt.subplots(figsize=(9, 7))
ax.plot(np.asarray(Model_Years), Model_AWD)
ax.plot(np.asarray(Model_Years), Model_dysentery)
ax.plot(np.asarray(Model_Years), Model_persistent)

# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)

plt.title("Total clinical diarrhoea")
plt.xlabel("Year")
plt.ylabel("Number of diarrhoea episodes")
plt.legend(['acute watery diarrhoea', 'dysentery', 'persistent diarrhoea'])
plt.savefig(outputpath / ('3 clinical diarrhoea types' + datestamp + '.pdf'))

plt.show()
