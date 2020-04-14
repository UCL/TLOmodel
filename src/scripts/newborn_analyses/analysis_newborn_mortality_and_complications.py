"""This analysis file produces all mortality outputs"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, \
    healthburden

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 5000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)
service_availability = ['*']

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

stats = output['tlo.methods.newborn_outcomes']['summary_stats']
stats['date'] = pd.to_datetime(stats['date'])
stats['year'] = stats['date'].dt.year

stats.plot.bar(x='year', y='nmr_early', stacked=True)
plt.title("Yearly NMR following birth")
plt.show()

stats.plot.bar(x='year', y='total_preterm_births', stacked=True)
plt.title("Yearly Preterm Birth Incidence")
plt.show()

stats.plot.bar(x='year', y='low_birth_weight', stacked=True)
plt.title("Yearly Low Birth Weight Incidence")
plt.show()

stats.plot.bar(x='year', y='small_for_gestational_age', stacked=True)
plt.title("Yearly Small For Gestational Age Incidence")
plt.show()

stats.plot.bar(x='year', y='sepsis_incidence', stacked=True)
plt.title("Yearly Early Onset Sepsis Incidence")
plt.show()

stats.plot.bar(x='year', y='total_enceph_incidence', stacked=True)
plt.title("Yearly Encephalopathy Incidence")
plt.show()

stats.plot.bar(x='year', y='ftt_incidence', stacked=True)
plt.title("Yearly Failure To Transition Incidence")
plt.show()


