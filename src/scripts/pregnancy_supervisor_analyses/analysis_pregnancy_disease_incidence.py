import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from tlo import Date, Simulation
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, symptommanager, healthseekingbehaviour

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

stats = output['tlo.methods.pregnancy_supervisor']['summary_stats']
stats['date'] = pd.to_datetime(stats['date'])
stats['year'] = stats['date'].dt.year
stats['year'] = stats['year'] - 1
stats.set_index("year", inplace=True)

stats.plot.bar(y='antenatal_mmr', stacked=True)
plt.title("Yearly Antenatal Maternal Mortality Rate")
plt.show()

stats.plot.bar(y='antenatal_sbr', stacked=True)
plt.title("Yearly Antenatal Still Birth Rate")
plt.show()


stats.plot.bar(y='spontaneous_abortion_rate', stacked=True)
plt.title("Yearly spontaneous_abortion_rate Rate")
plt.show()

stats.plot.bar(y='induced_abortion_rate', stacked=True)
plt.title("Yearly induced_abortion_rat Rate")
plt.show()

stats.plot.bar(y='ectopic_rate', stacked=True)
plt.title("Yearly ectopic_rate Rate")
plt.show()

stats.plot.bar(y='anaemia_rate', stacked=True)
plt.title("Yearly anaemia_rate Rate")
plt.show()

stats = output['tlo.methods.pregnancy_supervisor']['summary_stats']
