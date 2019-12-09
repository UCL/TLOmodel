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
    malaria
)

t0 = time.time()

# Where will output go
outputpath = './src/scripts/malaria/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2025, 12, 31)
popsize = 50000

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
malaria_strat = 0  # levels: 0 = national; 1 = district

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=0,
                                       capabilities_coefficient=1.0))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                             level=malaria_strat))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.malaria').setLevel(logging.INFO)
# logging.getLogger('tlo.methods.healthsystem').setLevel(logging.INFO)


# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

t1 = time.time()
print('Time taken', t1 - t0)

# %% read the results
from tlo.analysis.utils import parse_log_file
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from tlo import Date
import pandas as pd
from pathlib import Path

outputpath = './src/scripts/malaria/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
# logfile = outputpath + 'LogFile' + datestamp + '.log'

logfile = outputpath + 'LogFile' + '__2019_12_08' + '.log'
output = parse_log_file(logfile)
#
resourcefilepath = Path("./resources")

inc = output['tlo.methods.malaria']['incidence']
pfpr = output['tlo.methods.malaria']['prevalence']
tx = output['tlo.methods.malaria']['tx_coverage']
mort = output['tlo.methods.malaria']['ma_mortality']

# get model output dates in correct format
model_years = pd.to_datetime(inc.date)
model_years = model_years.dt.year
start_date = 2010
end_date = 2025

# import malaria data
inc_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="inc1000py_MAPdata",
)
PfPR_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="PfPR_MAPdata",
)
mort_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="mortalityRate_MAPdata",
)
tx_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="txCov_MAPdata",
)

# check date format for year columns in each sheet
# inc_data_years = pd.to_datetime(inc_data.Year, format="%Y")
# PfPR_data_years = pd.to_datetime(PfPR_data.Year, format="%Y")
# mort_data_years = pd.to_datetime(mort_data.Year, format="%Y")
# tx_data_years = pd.to_datetime(tx_data.Year, format="%Y")
#
# yr_num = inc_data_years.values

## FIGURES
plt.figure(1)

# Malaria incidence per 1000py - all ages with MAP model estimates
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(inc_data.Year, inc_data.inc_1000pyMean)  # MAP data
plt.fill_between(inc_data.Year, inc_data.inc_1000py_Lower,
                 inc_data.inc_1000pyUpper)  # ,color='red',alpha=.5)
# plt.fill_between(inc_data_years, inc_data.inc_1000py_Lower, inc_data.inc_1000pyUpper)#,color='red',alpha=.5)
plt.plot(model_years, inc.inc_clin_counter)  # model - using the clinical counter for multiple episodes per person
plt.title("Malaria Inc / 1000py")
plt.xlabel("Year")
plt.ylabel("Incidence (/1000py)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"])

# Malaria parasite prevalence rate - 2-10 year olds with MAP model estimates
# expect model estimates to be slightly higher as some will have
# undetectable parasitaemia
ax2 = plt.subplot(222)  # numrows, numcols, fignum
plt.plot(PfPR_data.Year, PfPR_data.PfPR_median)  # MAP data
plt.plot(model_years, pfpr.child2_10_prev)  # model
plt.title("Malaria PfPR 2-10 yrs")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("PfPR (%)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"])

# Malaria treatment coverage - all ages with MAP model estimates
ax3 = plt.subplot(223)  # numrows, numcols, fignum
plt.plot(tx_data.Year, tx_data.ACT_coverage)  # MAP data
plt.plot(model_years, tx.treatment_coverage)  # model
plt.title("Malaria Treatment Coverage")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Treatment coverage (%)")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["Data", "Model"])

# Malaria mortality rate - all ages with MAP model estimates
ax4 = plt.subplot(224)  # numrows, numcols, fignum
plt.plot(mort_data.Year, mort_data.mortality_rate_median)  # MAP data
plt.plot(model_years, mort.mort_rate)  # model
plt.title("Malaria Mortality Rate")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality rate")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"])
plt.show()

