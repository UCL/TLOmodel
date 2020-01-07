# %% Import Statements
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, antenatal_care, \
    healthburden, contraception, pregnancy_supervisor

# Where will output go - by default, wherever this script is run
outputpath = ""
# outputpath = Path("./outputs")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
resourcefilepath = Path("./resources/")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file
logfile = outputpath + 'LogFile' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# These first three graphs provide yearly incidence of preterm birth, early preterm birth and late preterm birth as a
# rate per 100 births

# ============================ PRETERM BIRTH RATE ======================================================
preterm_birth_df = output['tlo.methods.newborn_outcomes']['preterm_birth']
preterm_birth_df['date'] = pd.to_datetime(preterm_birth_df['date'])
preterm_birth_df['year'] = preterm_birth_df['date'].dt.year
preterm_by_year = preterm_birth_df.groupby(['year'])['person_id'].size()

all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
birth_by_year = all_births_df.groupby(['year'])['child'].size()

ptb_df = pd.concat((preterm_by_year, birth_by_year),axis=1)
ptb_df.columns = ['preterm_by_year', 'birth_by_year']
ptb_df['PTBR'] = ptb_df['preterm_by_year']/ptb_df['birth_by_year'] * 100

ptb_df.plot.bar(y='PTBR', stacked=True)
plt.title("Yearly Preterm Birth Rate")
plt.show()

# ============================ EARLY PRETERM BIRTH RATE ==============================================
early_preterm_df = output['tlo.methods.newborn_outcomes']['early_preterm']
early_preterm_df['date'] = pd.to_datetime(early_preterm_df['date'])
early_preterm_df['year'] = early_preterm_df['date'].dt.year
eptb_by_year = early_preterm_df.groupby(['year'])['person_id'].size()

all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
birth_by_year = all_births_df.groupby(['year'])['child'].size()

eptb_df = pd.concat((eptb_by_year, birth_by_year),axis=1)
eptb_df.columns = ['eptb_by_year', 'birth_by_year']
eptb_df['EPTBR'] = eptb_df['eptb_by_year']/eptb_df['birth_by_year'] * 100

eptb_df.plot.bar(y='PTBR', stacked=True)
plt.title("Yearly Early Preterm Birth Rate")
plt.show()

# ============================ LATE PRETERM BIRTH RATE ==============================================
late_preterm_df = output['tlo.methods.newborn_outcomes']['late_preterm']
late_preterm_df['date'] = pd.to_datetime(late_preterm_df['date'])
late_preterm_df['year'] = late_preterm_df['date'].dt.year
lptb_by_year = late_preterm_df.groupby(['year'])['person_id'].size()

all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
birth_by_year = all_births_df.groupby(['year'])['child'].size()

lptb_df = pd.concat((lptb_by_year, birth_by_year),axis=1)
lptb_df.columns = ['lptb_by_year', 'birth_by_year']
lptb_df['EPTBR'] = lptb_df['lptb_by_year']/lptb_df['birth_by_year'] * 100

lptb_df.plot.bar(y='PTBR', stacked=True)
plt.title("Yearly Late Preterm Birth Rate")
plt.show()

# ======================== GESTATION AT BIRTH FOR ALL BIRTHS ========================================



