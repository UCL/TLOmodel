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
from tlo.methods import demography, enhanced_lifestyle, new_diarrhoea, contraception

# Where will output go - by default, wherever this script is run
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# Work out the resource path from the path of the analysis file
# resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
resourcefilepath = Path('./resources')


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1000

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
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
# sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(new_diarrhoea.NewDiarrhoea(resourcefilepath=resourcefilepath))

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% TH: looking at births...
num_births = len(output['tlo.methods.demography']['on_birth'])

num_under_5ys = output['tlo.methods.demography']['age_range_m']['0-4'] + \
                    output['tlo.methods.demography']['age_range_f']['0-4']

# births and under-5s



# -----------------------------------------------------------------------------------
# %% Plot Incidence of Diarrhoea Over time:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


'''
# -----------------------------------------------------------------------------------
# Load Model Results on attributable pathogens
incidence_by_patho_df = output['tlo.methods.new_diarrhoea']['number_pathogen_diarrhoea']
Model_Years = pd.to_datetime(incidence_by_patho_df.date)
Model_Years = Model_Years.dt.year
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
# -----------------------------------------------------------------------------
# Load Model Results on clinical types of diarrhoea
clinical_type_df = output['tlo.methods.new_diarrhoea']['clinical_diarrhoea_type']
Model_Years = pd.to_datetime(clinical_type_df.date)
Model_AWD = clinical_type_df.AWD
Model_dysentery = clinical_type_df.dysentery
Model_persistent = clinical_type_df.persistent

fig2, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_AWD)
ax.plot(np.asarray(Model_Years), Model_dysentery)
ax.plot(np.asarray(Model_Years), Model_persistent)

plt.title("Total clinical diarrhoea")
plt.xlabel("Year")
plt.ylabel("Number of diarrhoea episodes")
plt.legend(['acute watery diarrhoea', 'dysentery', 'persistent diarrhoea'])
plt.savefig(outputpath + '3 clinical diarrhoea types' + datestamp + '.pdf')

plt.show()

# -----------------------------------------------------------------------------
# Load Model Results on clinical types of diarrhoea
status_counts_df = output['tlo.methods.new_diarrhoea']['episodes_counts']
Model_Years = pd.to_datetime(status_counts_df.date)
Model_incidence = status_counts_df.incidence_per100cy

plt.plot(Model_Years, Model_incidence)

plt.title("Number of children under 5 over the years")
plt.xlabel("Year")
plt.ylabel("number of children ")
# plt.savefig(outputpath + 'Diarrhoea incidence per 100 child-years' + datestamp + '.pdf')

plt.show()

'''
plt.title("Overall incidence of diarrhoea per 100 child-years")
plt.xlabel("Year")
plt.ylabel("Incidence of diarrhoea per 100 child-years")
plt.legend(['Yearly diarrhoea incidence'])
plt.savefig(outputpath + 'Diarrhoea incidence per 100 child-years' + datestamp + '.pdf')

# -----------------------------------------------------------------------------------
# Load Model Results on attributable pathogens
incidence_by_age_df = output['tlo.methods.new_diarrhoea']['diarr_incidence_age']
Model_Years = pd.to_datetime(incidence_by_patho_df.date)
Model_rotavirus = incidence_by_age_df.rotavirus[0]
Model_shigella = incidence_by_age_df.shigella[0]
Model_adenovirus = incidence_by_age_df.adenovirus[0]
Model_crypto = incidence_by_age_df.cryptosporidium[0]
Model_campylo = incidence_by_age_df.campylobacter[0]
Model_ETEC = incidence_by_age_df.ETEC[0]
Model_sapovirus = incidence_by_age_df.sapovirus[0]
Model_norovirus = incidence_by_age_df.norovirus[0]
Model_astrovirus = incidence_by_age_df.astrovirus[0]
Model_EPEC = incidence_by_age_df.tEPEC[0]
# pathogen_by_age = diarrhoea_patho_df.groupby(['years'])['person_id'].size()

fig3, ax = plt.subplots()
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

plt.title("Pathogen-attributed incidence of diarrhoea by age group")
plt.xlabel("Year")
plt.ylabel("diarrhoea incidence by pathogen per 100 child-years")
plt.legend(['Rotavirus', 'Shigella', 'Adenovirus', 'Cryptosporidium', 'Campylobacter', 'ETEC', 'sapovirus', 'norovirus',
            'astrovirus', 'tEPEC'])
plt.savefig(outputpath + 'Diarrhoea incidence by age group' + datestamp + '.pdf')

plt.show()
'''
'''
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Load Model Results on Dehydration
dehydration_df = output['tlo.methods.new_diarrhoea']['dehydration_levels']
Model_Years = pd.to_datetime(dehydration_df.date)
Model_total = clinical_type_df.total
Model_any_dehydration = dehydration_df.total
# Model_some_dehydration = dehydration_df.some
# Model_severe_dehydration = dehydration_df.severe
# diarrhoea_by_year = diarrhoea_df.groupby(['year'])['person_id'].size()

fig1, ax = plt.subplots()
# ax.plot(np.asarray(Model_Years), Model_any_dehydration) # TODO: remove the 'no dehydration'
ax.plot(np.asarray(Model_Years), Model_total)
ax.plot(np.asarray(Model_Years), Model_any_dehydration)
# ax.plot(np.asarray(Model_Years), Model_severe_dehydration)

plt.title("Incidence of Diarrhoea with dehydration")
plt.xlabel("Year")
plt.ylabel("Number of diarrhoeal episodes with dehydration")
plt.legend(['total diarrhoea', 'any dehydration'])
plt.savefig(outputpath + 'Dehydration incidence' + datestamp + '.pdf')

plt.show()
'''
'''
# Load Model Results on death from diarrhoea
death_df = output['tlo.methods.new_diarrhoea']['death_diarrhoea']
deaths_df_Years = pd.to_datetime(death_df.date)
death_by_diarrhoea = death_df.death

fig3, ax = plt.subplots()
ax.plot(np.asarray(deaths_df_Years), death_by_diarrhoea)

plt.title("Diarrhoea deaths")
plt.xlabel("Year")
plt.ylabel("Death by clinical type")
plt.legend(['number of deaths'])
plt.savefig(outputpath + 'Diarrhoeal death' + datestamp + '.pdf')

plt.show()
'''
# -----------------------------------------------------------------------------------

'''death_by_cause.plot.bar(stacked=True)
plt.title(" Total diarrhoea deaths per Year")
plt.show()
'''

'''
ig2, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_death)
ax.plot(np.asarray(Model_Years), Model_death1)
ax.plot(np.asarray(Model_Years), Model_death2)

plt.title("Diarrhoea deaths")
plt.xlabel("Year")
plt.ylabel("Number of children died from diarrhoea")
plt.legend(['AWD', 'persistent', 'dehydration'])
plt.savefig(outputpath + 'Diarrhoea attributable pathogens' + datestamp + '.pdf')

plt.show()
'''
