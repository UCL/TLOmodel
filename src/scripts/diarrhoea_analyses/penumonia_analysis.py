# %% Import Statements
import datetime
import logging
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, lifestyle, new_pneumonia

# Where will output go - by default, wherever this script is run
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# Work out the resource path from the path of the analysis file
resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 3000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file
logfile = outputpath + "LogFile" + datestamp + ".log"

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
# sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(new_pneumonia.NewPneumonia(resourcefilepath=resourcefilepath))

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% Plot Incidence of Diarrhoea Over time:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Model Results on attributable pathogens
diarrhoea_patho_df = output['tlo.methods.new_diarrhoea']['pneumonia_pathogens']
Model_Years = pd.to_datetime(diarrhoea_patho_df.date)
Model_RSV = diarrhoea_patho_df.rotavirus
Model_rhinovirus = diarrhoea_patho_df.rhinovirus
Model_hMPV = diarrhoea_patho_df.hMPV
Model_parainfluenza = diarrhoea_patho_df.parainfluenza
Model_strep = diarrhoea_patho_df.strep
Model_hib = diarrhoea_patho_df.hib
Model_TB = diarrhoea_patho_df.TB
Model_staph = diarrhoea_patho_df.staph
Model_influenza = diarrhoea_patho_df.influenza
Model_jirovecii = diarrhoea_patho_df.P.jirovecii
# pathogen_by_age = diarrhoea_patho_df.groupby(['years'])['person_id'].size()

ig1, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_RSV)
ax.plot(np.asarray(Model_Years), Model_rhinovirus)
ax.plot(np.asarray(Model_Years), Model_hMPV)
ax.plot(np.asarray(Model_Years), Model_parainfluenza)
ax.plot(np.asarray(Model_Years), Model_strep)
ax.plot(np.asarray(Model_Years), Model_hib)
ax.plot(np.asarray(Model_Years), Model_TB)
ax.plot(np.asarray(Model_Years), Model_staph)
ax.plot(np.asarray(Model_Years), Model_influenza)
ax.plot(np.asarray(Model_Years), Model_jirovecii)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pneumonia attributable pathogens")
plt.xlabel("Year")
plt.ylabel("Number of pathogen-attributed pneumonia cases")
plt.legend(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza',
            'streptococcus', 'hib', 'TB', 'staph', 'influenza', 'P. jirovecii'])
plt.savefig(outputpath + 'Pneumonia attributable pathogens' + datestamp + '.pdf')

plt.show()

# -----------------------------------------------------------------------------------
'''

# Load Model Results on attributable pathogens
incidence_by_patho_df = output['tlo.methods.new_diarrhoea']['diarr_incidence_by_patho']
Model_Years = pd.to_datetime(incidence_by_patho_df.date)
Model_rotavirus = incidence_by_patho_df.rotavirus
Model_shigella = incidence_by_patho_df.shigella
Model_adenovirus = incidence_by_patho_df.adenovirus
Model_crypto = incidence_by_patho_df.cryptosporidium
Model_campylo = incidence_by_patho_df.campylobacter
Model_ETEC = incidence_by_patho_df.ETEC
Model_sapovirus = incidence_by_patho_df.sapovirus
Model_norovirus = incidence_by_patho_df.norovirus
Model_astrovirus = incidence_by_patho_df.astrovirus
Model_EPEC = incidence_by_patho_df.tEPEC
# pathogen_by_age = diarrhoea_patho_df.groupby(['years'])['person_id'].size()

igf, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_rotavirus)
ax.plot(np.asarray(Model_Years), Model_shigella)
ax.plot(np.asarray(Model_Years), Model_adenovirus)
ax.plot(np.asarray(Model_Years), Model_crypto)
ax.plot(np.asarray(Model_Years), Model_campylo)
ax.plot(np.asarray(Model_Years), Model_ETEC)
ax.plot(np.asarray(Model_Years), Model_sapovirus)
ax.plot(np.asarray(Model_Years), Model_norovirus)
ax.plot(np.asarray(Model_Years), Model_astrovirus)
ax.plot(np.asarray(Model_Years), Model_EPEC)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pathogen-attributed incidence of diarrhoea")
plt.xlabel("Year")
plt.ylabel("diarrhoea incidence by pathogen per 100 child-years")
plt.legend(['Rotavirus', 'Shigella', 'Adenovirus', 'Cryptosporidium', 'Campylobacter', 'ETEC', 'sapovirus', 'norovirus',
            'astrovirus', 'tEPEC'])
plt.savefig(outputpath + 'Diarrhoea incidence by pathogens' + datestamp + '.pdf')

plt.show()
'''
