import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)

from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care

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
popsize = 10000


for label, service_avail in scenarios.items():
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_avail))
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
    counts = output['tlo.methods.labour']['summary_stats_incidence']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )
    mmr = counts['intrapartum_mmr']
    uterine_rupture = counts['ur_incidence']

    return mmr, uterine_rupture


deaths = dict()
incidence = dict()
for label, file in output_files.items():
    deaths[label], incidence[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)


data = {}
for label in deaths.keys():
    data.update({label: deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Maternal Mortality Ratio by Year')
plt.savefig(outputpath / ("MMR_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()

for label in incidence.keys():
    data.update({label: incidence[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Incidence of Uterine Rupture by Year')
plt.savefig(outputpath / ("UR_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()


