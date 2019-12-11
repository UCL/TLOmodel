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
end_date = Date(2024, 12, 31)
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
import xlsxwriter

outputpath = './src/scripts/malaria/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
# logfile = outputpath + 'LogFile' + datestamp + '.log'
logfile = outputpath + 'LogFile' + '__2019_12_09' + '.log'
output = parse_log_file(logfile)
resourcefilepath = Path("./resources")

inc = output['tlo.methods.malaria']['incidence']
pfpr = output['tlo.methods.malaria']['prevalence']
tx = output['tlo.methods.malaria']['tx_coverage']
mort = output['tlo.methods.malaria']['ma_mortality']

prev_district = output['tlo.methods.malaria']['prev_district']

# ----------------------------------- SAVE OUTPUTS -----------------------------------
out_path = '//fi--san02/homes/tmangal/Thanzi la Onse/Malaria/'

if malaria_strat == 0:
    savepath = out_path + "national_output_" + datestamp + ".xlsx"
else:
    savepath = out_path + "district_output_" + datestamp + ".xlsx"

writer = pd.ExcelWriter(savepath, engine='xlsxwriter')

inc_df = pd.DataFrame(inc)
inc_df.to_excel(writer, sheet_name='inc')

pfpr_df = pd.DataFrame(pfpr)
pfpr_df.to_excel(writer, sheet_name='pfpr')

tx_df = pd.DataFrame(tx)
tx_df.to_excel(writer, sheet_name='tx')

mort_df = pd.DataFrame(mort)
mort_df.to_excel(writer, sheet_name='mort')

prev_district_df = pd.DataFrame(prev_district)
prev_district_df.to_excel(writer, sheet_name='prev_district')

writer.save()

# ----------------------------------- CREATE PLOTS-----------------------------------

# get model output dates in correct format
model_years = pd.to_datetime(inc.date)
model_years = model_years.dt.year
start_date = 2010
end_date = 2025

# import malaria data
# MAP
incMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="inc1000py_MAPdata",
)
PfPRMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="PfPR_MAPdata",
)
mortMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="mortalityRate_MAPdata",
)
txMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="txCov_MAPdata",
)

# WHO
WHO_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="WHO_MalReport",
)

# check date format for year columns in each sheet
# inc_data_years = pd.to_datetime(inc_data.Year, format="%Y")
# PfPR_data_years = pd.to_datetime(PfPR_data.Year, format="%Y")
# mort_data_years = pd.to_datetime(mort_data.Year, format="%Y")
# tx_data_years = pd.to_datetime(tx_data.Year, format="%Y")
#

## FIGURES
plt.figure(1, figsize=(10, 10))

# Malaria incidence per 1000py - all ages with MAP model estimates
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(incMAP_data.Year, incMAP_data.inc_1000pyMean)  # MAP data
plt.fill_between(incMAP_data.Year, incMAP_data.inc_1000py_Lower,
                 incMAP_data.inc_1000pyUpper, alpha=.5)
plt.plot(WHO_data.Year, WHO_data.cases1000pyPoint)  # WHO data
plt.fill_between(WHO_data.Year, WHO_data.cases1000pyLower,
                 WHO_data.cases1000pyUpper, alpha=.5)
plt.plot(model_years, inc.inc_clin_counter)  # model - using the clinical counter for multiple episodes per person
plt.title("Malaria Inc / 1000py")
plt.xlabel("Year")
plt.ylabel("Incidence (/1000py)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

# Malaria parasite prevalence rate - 2-10 year olds with MAP model estimates
# expect model estimates to be slightly higher as some will have
# undetectable parasitaemia
ax2 = plt.subplot(222)  # numrows, numcols, fignum
plt.plot(PfPRMAP_data.Year, PfPRMAP_data.PfPR_median)  # MAP data
plt.fill_between(PfPRMAP_data.Year, PfPRMAP_data.PfPR_LCI,
                 PfPRMAP_data.PfPR_UCI, alpha=.5)
plt.plot(model_years, pfpr.child2_10_prev)  # model
plt.title("Malaria PfPR 2-10 yrs")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("PfPR (%)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"])
plt.tight_layout()

# Malaria treatment coverage - all ages with MAP model estimates
ax3 = plt.subplot(223)  # numrows, numcols, fignum
plt.plot(txMAP_data.Year, txMAP_data.ACT_coverage)  # MAP data
plt.plot(model_years, tx.treatment_coverage)  # model
plt.title("Malaria Treatment Coverage")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Treatment coverage (%)")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["Data", "Model"])
plt.tight_layout()

# Malaria mortality rate - all ages with MAP model estimates
ax4 = plt.subplot(224)  # numrows, numcols, fignum
plt.plot(mortMAP_data.Year, mortMAP_data.mortality_rate_median)  # MAP data
plt.fill_between(mortMAP_data.Year, mortMAP_data.mortality_rate_LCI,
                 mortMAP_data.mortality_rate_UCI, alpha=.5)
plt.plot(WHO_data.Year, WHO_data.MortRatePerPersonPoint)  # WHO data
plt.fill_between(WHO_data.Year, WHO_data.MortRatePerPersonLower,
                 WHO_data.MortRatePerPersonUpper, alpha=.5)
plt.plot(model_years, mort.mort_rate)  # model
plt.title("Malaria Mortality Rate")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality rate")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 0.005)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

if malaria_strat == 0:
    figpath = out_path + "national_output_" + datestamp + ".png"
else:
    figpath = out_path + "district_output_" + datestamp + ".png"

plt.savefig(figpath, bbox_inches='tight')
plt.show()

plt.close()

########### district plots ###########################
# get model output dates in correct format
model_years = pd.to_datetime(prev_district.date)
model_years = model_years.dt.year
start_date = 2010
end_date = 2025

plt.figure(1, figsize=(30, 20))

# Malaria parasite prevalence
ax = plt.subplot(111)  # numrows, numcols, fignum
plt.plot(model_years, prev_district.Balaka)  # model - using the clinical counter for multiple episodes per person
plt.title("Parasite prevalence in Balaka")
plt.xlabel("Year")
plt.ylabel("Parasite prevalence in Balaka")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Model"])
plt.tight_layout()
figpath = out_path + "district_output_seasonal" + datestamp + ".png"

plt.savefig(figpath, bbox_inches='tight')
plt.show()
plt.close()
