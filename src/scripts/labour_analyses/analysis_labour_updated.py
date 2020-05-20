import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)

from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, symptommanager, healthseekingbehaviour

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: No Treatment
# *2: Some Treatment

scenarios = dict()
scenarios['No_Health_System'] = []
scenarios['Health_System'] = ['*']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 2)
popsize = 20000


for label, service_avail in scenarios.items():
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_avail))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))

    logfile = sim.configure_logging(filename="LogFile")
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = logfile


def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    maternal_counts = output['tlo.methods.labour']['summary_stats_incidence']
    maternal_counts['year'] = pd.to_datetime(maternal_counts['date']).dt.year
    maternal_counts.drop(columns='date', inplace=True)
    maternal_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )
    newborn_counts = output['tlo.methods.newborn_outcomes']['summary_stats']
    newborn_counts['year'] = pd.to_datetime(newborn_counts['date']).dt.year
    newborn_counts.drop(columns='date', inplace=True)
    newborn_counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    mmr = maternal_counts['intrapartum_mmr']
    sbr = maternal_counts['sbr']
    nmr = newborn_counts['nmr_early']

    return mmr, sbr, nmr


maternal_deaths = dict()
newborn_deaths = dict()
still_births = dict()
for label, file in output_files.items():
    maternal_deaths[label], newborn_deaths[label], still_births[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)

data = {}
for label in maternal_deaths.keys():
    data.update({label: maternal_deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Maternal Mortality Ratio by Year')
plt.savefig(outputpath / ("MMR_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()

for label in newborn_deaths.keys():
    data.update({label: newborn_deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Early Neonatal Mortality Ratio by Year ')
plt.savefig(outputpath / ("NMR_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()

for label in still_births.keys():
    data.update({label: still_births[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Intrapartum Stillbirth Rate by Year ')
plt.savefig(outputpath / ("NMR_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()


