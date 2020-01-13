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
from tlo.methods import demography, enhanced_lifestyle, pneumonia

# Where will output go - by default, wherever this script is run
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# Work out the resource path from the path of the analysis file
resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 10000

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
sim.register(enhanced_lifestyle.Lifestyle())
# sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(pneumonia.Pneumonia(resourcefilepath=resourcefilepath))

sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
# this will make sure that the logging file is complete
fh.flush()
# %% read the results
output = parse_log_file(logfile)

# ---------------------------------------------------------------------------------------
df = sim.population.props
params = sim.modules['Pneumonia'].parameters

tot_pop_under5_alive = len(df.index[df.is_alive & df.age_exact_years < 5])

def plot_incidence_age_group(incidence, age):
    fig, ax = plt.subplots()
    for i in incidence_by_patho_df:
        df_before_2019 = df[df.date <= pd.Timestamp(date(2019, 1, 1))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]



# %% Plot Incidence of Diarrhoea Over time:
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
'''
# Load Model Results on attributable pathogens
pneumonia_patho_df = output['tlo.methods.pneumonia2']['pneumonia_pathogens']
Model_Years = pd.to_datetime(pneumonia_patho_df.date)
Model_RSV = pneumonia_patho_df.RSV
Model_rhinovirus = pneumonia_patho_df.rhinovirus
Model_hMPV = pneumonia_patho_df.hMPV
Model_parainfluenza = pneumonia_patho_df.parainfluenza
Model_strep = pneumonia_patho_df.strep
Model_hib = pneumonia_patho_df.hib
Model_TB = pneumonia_patho_df.TB
Model_staph = pneumonia_patho_df.staph
Model_influenza = pneumonia_patho_df.influenza
Model_jirovecii = pneumonia_patho_df.jirovecii
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
'''
# -----------------------------------------------------------------------------------
# Load Model Results on pneumonia incidence rate by pathogen
incidence_by_patho_df = output['tlo.methods.pneumonia2']['pneumo_incidence_by_patho']
Model_Years = pd.to_datetime(incidence_by_patho_df.date)
Model_RSV = incidence_by_patho_df.RSV
Model_rhinovirus = incidence_by_patho_df.rhinovirus
Model_hMPV = incidence_by_patho_df.hMPV
Model_parainfluenza = incidence_by_patho_df.parainfluenza
Model_strep = incidence_by_patho_df.strep
Model_hib = incidence_by_patho_df.hib
Model_TB = incidence_by_patho_df.TB
Model_staph = incidence_by_patho_df.staph
Model_influenza = incidence_by_patho_df.influenza
Model_jirovecii = incidence_by_patho_df.jirovecii
# pathogen_by_age = diarrhoea_patho_df.groupby(['years'])['person_id'].size()

ig2, ax = plt.subplots()
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

plt.title("Pathogen-attributed incidence of severe pneumonia")
plt.xlabel("Year")
plt.ylabel("Incidence of severe pneumonia by pathogen per 100 child-years")
plt.legend(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza',
            'streptococcus', 'hib', 'TB', 'staph', 'influenza', 'P. jirovecii'])
plt.savefig(outputpath + 'Pneumonia incidence by pathogens' + datestamp + '.pdf')

plt.show()

# -----------------------------------------------------------------------------------
# Load Model Results on pneumonia severity
severity_df = output['tlo.methods.pneumonia2']['severity_pneumonia']

Model_total = severity_df.total
Model_pneumonia = severity_df.pneumonia
Model_severe = severity_df.severe
Model_very_severe = severity_df.very_severe

fig1, ax = plt.subplots()
ax.plot(np.asarray(Model_Years), Model_pneumonia)
ax.plot(np.asarray(Model_Years), Model_severe)
ax.plot(np.asarray(Model_Years), Model_very_severe)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pneumonia cases by severity")
plt.xlabel("Year")
plt.ylabel("Number of pneumonia cases")
plt.legend(['non-severe', 'severe', 'very severe'])
plt.savefig(outputpath + 'Pneumonia severity' + datestamp + '.pdf')

plt.show()

# -----------------------------------------------------------------------------------
# Load Model Results on pneumonia severity by pathogen
'''
severity_by_patho_df = output['tlo.methods.pneumonia2']['severity_pneumonia']

Model_pneumonia = severity_by_patho_df.pneumonia
Model_severe = severity_by_patho_df.severe
Model_very_severe = severity_by_patho_df.very_severe
labels = output['tlo.methods.pneumonia2']['pneumonia_pathogens']

fig3, ax = plt.subplots()
ax.plot(np.asarray(Model_severe), Model_RSV)
ax.plot(np.asarray(Model_Years), Model_severe)
ax.plot(np.asarray(Model_Years), Model_very_severe)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pneumonia cases by severity")
plt.xlabel("Year")
plt.ylabel("Number of pneumonia cases")
plt.legend(['non-severe', 'severe', 'very severe'])
plt.savefig(outputpath + 'Pneumonia severity' + datestamp + '.pdf')

plt.show()
'''
'''
pneumonia_death_df = output['tlo.methods.pneumonia2']['pneumo_death']
death_each_year = pd.to_datetime(pneumonia_death_df.date)
pneumonia_death_df['year'] = death_each_year.dt.year
pneumonia_death_number = pneumonia_death_df.groupby(['year', 'cause'])['child'].size()

pneumonia_death_number = pneumonia_death_number.reset_index()
pneumonia_death_number.index = pneumonia_death_number['year']
pneumonia_death_number.drop(columns="year", inplace=True)
pneumonia_death_number = pneumonia_death_number.rename(columns={'child': 'number of deaths'})

plt.plot_date(death_each_year, pneumonia_death_number)
plt.xlabel('Year')
plt.ylabel('number of deaths')

# pneumonia_death_number.plot.bar(stacked=True)
plt.show()
'''
'''
# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.xlabel("Year")
plt.ylabel("number of deaths")
plt.savefig(outputpath + "Deaths" + datestamp + ".pdf")
plt.show()
'''
'''
ig3, ax = plt.subplots()
ax.plot(np.asarray(pneumonia_death_df['date']), death_by_year)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)

plt.title("Pneumonia deaths")
plt.xlabel("Year")
plt.ylabel("Number of pneumonia deaths")
plt.legend()
plt.savefig(outputpath + 'Pneumonia deaths count' + datestamp + '.pdf')

plt.show()
'''
