import datetime
import logging
import os
from pathlib import Path
import pandas as pd
import time

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    malaria,
    dx_algorithm_child,
    dx_algorithm_adult,
    healthseekingbehaviour,
    symptommanager,
)

t0 = time.time()

# Where will output go
outputpath = './src/scripts/outputs/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2025, 12, 31)
popsize = 100000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'Malaria_ITN0.9' + datestamp + '.log'

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
malaria_strat = 1  # levels: 0 = national; 1 = district
malaria_testing = 0.35  # adjust this to match rdt/tx levels
itn = 0.9

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=0,
                                       ignore_cons_constraints=True,
                                       ignore_priority=True,
                                       capabilities_coefficient=1.0,
                                       disable=True))  # disables the health system constraints so all HSI events run
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
sim.register(dx_algorithm_child.DxAlgorithmChild())
sim.register(dx_algorithm_adult.DxAlgorithmAdult())
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                             level=malaria_strat, testing=malaria_testing, itn=itn))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.malaria').setLevel(logging.INFO)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

t1 = time.time()
print('Time taken', t1 - t0)

# ---------------------------------------- PLOTS ---------------------------------------- #
# %% read the results
from tlo.analysis.utils import parse_log_file
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from tlo import Date
import pandas as pd
from pathlib import Path
import xlsxwriter

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# import output files
outputpath = './src/scripts/outputs/'
baseline_file = outputpath + 'Malaria_Baseline__2020_01_27' + '.log'
itn07file = outputpath + 'Malaria_ITN0.7__2020_01_27' + '.log'
itn08file = outputpath + 'Malaria_ITN0.8__2020_01_27' + '.log'
itn09file = outputpath + 'Malaria_ITN0.9__2020_01_27' + '.log'

baseline = parse_log_file(baseline_file)
itn07 = parse_log_file(itn07file)
itn08 = parse_log_file(itn08file)
itn09 = parse_log_file(itn09file)

# plot variables
inc_baseline = baseline['tlo.methods.malaria']['incidence']
inc_itn07 = itn07['tlo.methods.malaria']['incidence']
inc_itn08 = itn08['tlo.methods.malaria']['incidence']
inc_itn09 = itn09['tlo.methods.malaria']['incidence']

pfpr_baseline = baseline['tlo.methods.malaria']['prevalence']
pfpr_itn07 = itn07['tlo.methods.malaria']['prevalence']
pfpr_itn08 = itn08['tlo.methods.malaria']['prevalence']
pfpr_itn09 = itn09['tlo.methods.malaria']['prevalence']

mort_baseline = baseline['tlo.methods.malaria']['ma_mortality']
mort_itn07 = itn07['tlo.methods.malaria']['ma_mortality']
mort_itn08 = itn08['tlo.methods.malaria']['ma_mortality']
mort_itn09 = itn09['tlo.methods.malaria']['ma_mortality']

model_years = pd.to_datetime(inc_baseline.date)
model_years = model_years.dt.year
start_date = 2010
end_date = 2025

# create plots
## FIGURES
plt.figure(1, figsize=(8, 6))

# Malaria incidence per 1000py - all ages
ax = plt.subplot(311)  # numrows, numcols, fignum
plt.plot(model_years, inc_baseline.inc_clin_counter)  # baseline
plt.plot(model_years, inc_itn07.inc_clin_counter)  # itn 0.7
plt.plot(model_years, inc_itn08.inc_clin_counter)  # itn 0.8
plt.plot(model_years, inc_itn09.inc_clin_counter)  # itn 0.9
plt.title("Malaria Inc / 1000py")
plt.xlabel("Year")
plt.ylabel("Incidence (/1000py)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
           bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()

# Malaria parasite prevalence rate - 2-10 year olds 
ax2 = plt.subplot(312)  # numrows, numcols, fignum
plt.plot(model_years, pfpr_baseline.child2_10_prev)  # baseline
plt.plot(model_years, pfpr_itn07.child2_10_prev)  # itn 0.7
plt.plot(model_years, pfpr_itn08.child2_10_prev)  # itn 0.8
plt.plot(model_years, pfpr_itn09.child2_10_prev)  # itn 0.9
plt.title("Malaria PfPR 2-10 yrs")
plt.xlabel("Year")
plt.ylabel("PfPR (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
           bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()

# Malaria mortality rate - all ages 
ax3 = plt.subplot(313)  # numrows, numcols, fignum
plt.plot(model_years, mort_baseline.mort_rate)  # baseline
plt.plot(model_years, mort_itn07.mort_rate)  # itn 0.7
plt.plot(model_years, mort_itn08.mort_rate)  # itn 0.8
plt.plot(model_years, mort_itn09.mort_rate)  # itn 0.9
plt.title("Malaria Mortality Rate")
plt.xlabel("Year")
plt.ylabel("Mortality rate")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
           bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()

out_path = '//fi--san02/homes/tmangal/Thanzi la Onse/Malaria/model_outputs/ITN_projections_28Jan2010/'
figpath = out_path + "ITN_projections" + datestamp + ".png"
plt.savefig(figpath, bbox_inches='tight')

plt.show()
plt.close()
