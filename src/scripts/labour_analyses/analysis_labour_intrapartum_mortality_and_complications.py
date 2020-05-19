"""This analysis file produces all mortality outputs"""

import datetime
from pathlib import Path

import pandas as pd
from tlo import Date, Simulation
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, \
    healthburden, healthseekingbehaviour, symptommanager

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)
service_availability = ['*']

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
# sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
#sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=[]))

sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

stats_incidence = output['tlo.methods.labour']['summary_stats_incidence']
stats_incidence['date'] = pd.to_datetime(stats_incidence['date'])
stats_incidence['year'] = stats_incidence['date'].dt.year

stats_crude = output['tlo.methods.labour']['summary_stats_crude_cases']
stats_crude['date'] = pd.to_datetime(stats_crude['date'])
stats_crude['year'] = stats_crude['date'].dt.year

stats_deliveries = output['tlo.methods.labour']['summary_stats_deliveries']
stats_deliveries['date'] = pd.to_datetime(stats_deliveries['date'])
stats_deliveries['year'] = stats_deliveries['date'].dt.year

stats_nb = output['tlo.methods.newborn_outcomes']['summary_stats']
stats_nb['date'] = pd.to_datetime(stats_nb['date'])
stats_nb['year'] = stats_nb['date'].dt.year



# todo: set index as year? restructure to year only?
# where can i output this too for analyses

# stats.plot.bar(x='year', y='ol_incidence', stacked=True)
# plt.title("Yearly Obstructed Labour Rate")
# plt.show()

# stats.plot.bar(x='year', y='aph_incidence', stacked=True)
# plt.title("Yearly Antepartum Haemorrhage Rate")
# plt.show()

# stats.plot.bar(x='year', y='ur_incidence', stacked=True)
# plt.title("Yearly Uterine Rupture Rate")
# plt.show()

# stats.plot.bar(x='year', y='ec_incidence', stacked=True)
# plt.title("Yearly Eclampsia Rate")
# plt.show()

# stats.plot.bar(x='year', y='sep_incidence', stacked=True)
# plt.title("Yearly Sepsis Rate")
# plt.show()

# stats.plot.bar(x='year', y='pph_incidence', stacked=True)
# plt.title("Yearly PPH Rate")
# plt.show()
