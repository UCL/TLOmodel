import datetime
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    hiv,
    enhanced_lifestyle,
    malecircumcision,
    tb,
    symptommanager
)

# TODO: this sim includes symptom manager. Include dx_algorithm once it is updated by Tim

start_time = time.time()

# Where will output go
outputpath = './src/scripts/tb/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
# resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2014, 12, 31)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + "TbHiv_LogFile" + datestamp + ".log"

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
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
sim.register(tb.Tb(resourcefilepath=resourcefilepath))
sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.hiv').setLevel(logging.INFO)
logging.getLogger("tlo.methods.tb").setLevel(logging.INFO)
logging.getLogger("tlo.methods.demography").setLevel(logging.INFO)  # to get deaths
# logging.getLogger("tlo.methods.contraception").setLevel(logging.INFO)  # for births

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

print("--- %s seconds ---" % (time.time() - start_time))

# %% read the results
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from tlo import Date

from tlo.analysis.utils import parse_log_file

outputpath = './src/scripts/tb/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath + "TbHiv_LogFile" + datestamp + ".log"
output = parse_log_file(logfile)

# output = parse_log_file('./src/scripts/tb/LogFile__2019_10_04.log')

# ------------------------------------- DEMOGRAPHY OUTPUTS ------------------------------------- #

# get deaths from demography
deaths = output['tlo.methods.demography']['death']

deaths['date'] = pd.to_datetime(deaths['date'])
deaths['year'] = deaths.date.dt.year.astype(int)

# select only hiv deaths
agg_deaths = deaths.groupby(['year', 'cause']).count().unstack(fill_value=0).stack()
death_counts = agg_deaths.iloc[agg_deaths.index.get_level_values('cause') == 'hiv'].person_id

pop = output['tlo.methods.demography']['population']
pop['date'] = pd.to_datetime(pop['date'])

mortality_rate = [(x / y) * 1000 for x, y in zip(death_counts, pop['total'])]

# ------------------------------------- MODEL OUTPUTS AND DATA ------------------------------------- #

## HIV
# model outputs
m_hiv = output['tlo.methods.hiv']['hiv_infected']
m_hiv_prev_m = output['tlo.methods.hiv']['hiv_adult_prev_m']
m_hiv_prev_f = output['tlo.methods.hiv']['hiv_adult_prev_f']
m_hiv_prev_child = output['tlo.methods.hiv']['hiv_child_prev_m']
m_hiv_tx = output['tlo.methods.hiv']['hiv_treatment']
m_hiv_fsw = output['tlo.methods.hiv']['hiv_fsw']
m_hiv_mort = output['tlo.methods.hiv']['hiv_mortality']

m_hiv_years = pd.to_datetime(m_hiv.date)
# m_hiv_years = m_hiv_years.dt.year

hiv_art_cov_percent = m_hiv_tx.hiv_coverage_adult_art * 100

# import HIV data
aidsInfo_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_HIV.xlsx",
    sheet_name="aids_info",
)

data_years = pd.to_datetime(aidsInfo_data.year, format="%Y")

# TB
m_tb_inc = output['tlo.methods.tb']['tb_incidence']
m_tb_prev_m = output['tlo.methods.tb']['tb_propActiveTbMale']
m_tb_prev_f = output['tlo.methods.tb']['tb_propActiveTbFemale']
m_tb_prev = output['tlo.methods.tb']['tb_prevalence']
m_tb_treatment = output['tlo.methods.tb']['tb_treatment']
m_tb_mort = output['tlo.methods.tb']['tb_mortality']
m_tb_bcg = output['tlo.methods.tb']['tb_bcg']

m_tb_years = pd.to_datetime(m_tb_inc.date)

tb_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_TB.xlsx",
    sheet_name="WHO_estimates",
)
tb_data_years = pd.to_datetime(tb_data.year, format="%Y")

# ------------------------------------- HIV FIGURES ------------------------------------- #

plt.style.use('ggplot')
plt.figure(4, figsize=(15, 10))

# HIV prevalence
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(data_years, aidsInfo_data.prev_15_49)
plt.fill_between(data_years, aidsInfo_data.prev_15_49_lower,
                 aidsInfo_data.prev_15_49_upper, alpha=.5)
plt.plot(m_hiv_years, m_hiv.hiv_prev_adult)
plt.title("HIV adult prevalence")
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0, 15)
plt.legend(["UNAIDS", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# HIV incidence
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(data_years, aidsInfo_data.inc_15_49_percent)
plt.fill_between(data_years, aidsInfo_data.inc_15_49_percent_lower,
                 aidsInfo_data.inc_15_49_percent_upper, alpha=.5)
plt.plot(m_hiv_years, m_hiv.hiv_adult_inc_percent)
plt.title("HIV adult incidence (%)")
plt.xlabel("Year")
plt.ylabel("Incidence (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0, 1.0)
plt.legend(["UNAIDS", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# HIV treatment coverage
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(data_years, aidsInfo_data.percent15plus_on_art)
plt.fill_between(data_years, aidsInfo_data.percent15plus_on_art_lower,
                 aidsInfo_data.percent15plus_on_art_upper, alpha=.5)
plt.plot(m_hiv_years, hiv_art_cov_percent)
plt.title("ART adult coverage (%)")
plt.xlabel("Year")
plt.ylabel("Coverage (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0, 100)
plt.legend(["UNAIDS", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# AIDS mortality
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(data_years, aidsInfo_data.mort_rate100k)
plt.fill_between(data_years, aidsInfo_data.mort_rate100k_lower,
                 aidsInfo_data.mort_rate100k_upper, alpha=.5)
plt.plot(pop['date'], mortality_rate)
plt.title("Mortality rates per 100k")
plt.xlabel("Year")
plt.ylabel("Mortality rate per 100k")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0, 15)
plt.legend(["UNAIDS", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()

# plt.close()
# ------------------------------------- TB FIGURES ------------------------------------- #

plt.style.use('ggplot')
plt.figure(2, figsize=(15, 10))

# TB incidence
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(tb_data_years, tb_data.incidence_per_100k)
plt.plot(m_tb_years, m_tb_inc.tbIncActive100k)
plt.title("TB case incidence/100k")
plt.xlabel("Year")
plt.ylabel("Incidence (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# TB prevalence
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(tb_data_years, tb_data.prevalence_all_ages)
plt.plot(m_tb_years, m_tb_prev.tbPropActive)
plt.title("TB prevalence")
plt.xlabel("Year")
plt.ylabel("Prevalence")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# TB treatment coverage
plt.subplot(223)  # numrows, numcols, fignum
# plt.plot(tb_data_years, tb_data.prevalence_all_ages)
plt.plot(m_tb_years, m_tb_treatment.tbTreat)
plt.title("TB treatment coverage")
plt.xlabel("Year")
plt.ylabel("Coverage (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")

# BCG coverage
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(tb_data_years, tb_data.bcg_coverage)
plt.plot(m_tb_years, m_tb_bcg.tbBcgCoverage)
plt.title("BCG coverage")
plt.xlabel("Year")
plt.ylabel("Coverage (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Data", "Model"],
           bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()
# plt.savefig(outputpath + "hiv_inc_adult" + datestamp + ".pdf")


##########################################################################################################
## send outputs to csv files
##########################################################################################################

# create new folder with today's date
# datestamp2 = datetime.date.today().strftime("%Y_%m_%d")
# path = "Z:Thanzi la Onse/model_outputs/" + datestamp2
# if not os.path.exists(path):
#     os.makedirs(path)
#
# ## HIV
# inc = output['tlo.methods.hiv']['hiv_infected']
# prev_m = output['tlo.methods.hiv']['hiv_adult_prev_m']
# prev_f = output['tlo.methods.hiv']['hiv_adult_prev_f']
# prev_child = output['tlo.methods.hiv']['hiv_child_prev_m']
# tx = output['tlo.methods.hiv']['hiv_treatment']
# fsw = output['tlo.methods.hiv']['hiv_fsw']
# mort = output['tlo.methods.hiv']['hiv_mortality']
#
# inc_path = os.path.join(path, "hiv_inc_new.csv")
# inc.to_csv(inc_path, header=True)
#
# prev_m_path = os.path.join(path, "hiv_prev_m.csv")
# prev_m.to_csv(prev_m_path, header=True)
#
# prev_f_path = os.path.join(path, "hiv_prev_f.csv")
# prev_f.to_csv(prev_f_path, header=True)
#
# prev_child_path = os.path.join(path, "hiv_prev_child.csv")
# prev_child.to_csv(prev_child_path, header=True)
#
# tx_path = os.path.join(path, "hiv_tx_new.csv")
# tx.to_csv(tx_path, header=True)
#
# fsw_path = os.path.join(path, "hiv_fsw_new.csv")
# fsw.to_csv(fsw_path, header=True)
#
# mort_path = os.path.join(path, "hiv_mort_new.csv")
# mort.to_csv(mort_path, header=True)
#
# # TB
# tb_inc = output['tlo.methods.tb']['tb_incidence']
# tb_prev_m = output['tlo.methods.tb']['tb_propActiveTbMale']
# tb_prev_f = output['tlo.methods.tb']['tb_propActiveTbFemale']
# tb_prev = output['tlo.methods.tb']['tb_prevalence']
# tb_mort = output['tlo.methods.tb']['tb_mortality']
#
# tb_inc_path = os.path.join(path, "tb_inc.csv")
# tb_inc.to_csv(tb_inc_path, header=True)
#
# tb_prev_m_path = os.path.join(path, "tb_prev_m.csv")
# tb_prev_m.to_csv(tb_prev_m_path, header=True)
#
# tb_prev_f_path = os.path.join(path, "tb_prev_f.csv")
# tb_prev_f.to_csv(tb_prev_f_path, header=True)
#
# tb_prev_path = os.path.join(path, "tb_prev.csv")
# tb_prev.to_csv(tb_prev_path, header=True)
#
# tb_mort_path = os.path.join(path, "tb_mort.csv")
# tb_mort.to_csv(tb_mort_path, header=True)

# deaths_df = output['tlo.methods.demography']['death']
# deaths_df['date'] = pd.to_datetime(deaths_df['date'])
# deaths_df['year'] = deaths_df['date'].dt.year
# d_gp = deaths_df.groupby(['year', 'cause']).size().unstack().fillna(0)
# d_gp.to_csv(r'Z:Thanzi la Onse\HIV\Model_original\deaths_new.csv', header=True)
