"""
This will demonstrate the effect of different treatment
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import contraception, demography, diarrhoea, healthsystem, enhanced_lifestyle, \
    symptommanager, healthburden, healthseekingbehaviour, dx_algorithm_child

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: No Treatment
# *2: Some Treatment
# TODO: expand this.

scenarios = dict()
scenarios['No_Treatment'] = []
scenarios['Treatment'] = ['*']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 2)
popsize = 5000

for label, service_avail in scenarios.items():
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_avail))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
    sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

    logfile = sim.configure_logging(filename="LogFile")
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = logfile


# %% Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = output['tlo.methods.diarrhoea']['incidence_count_by_pathogen']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    # get person-years of 0 year-old, 1 year-olds and 2-4 year-old
    py_ = output['tlo.methods.demography']['person_years']
    years = pd.to_datetime(py_['date']).dt.year
    py = pd.DataFrame(index=years, columns=['0y', '1y', '2-4y'])
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) + \
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
        ).transpose()

        py.loc[year, '0y'] = tot_py.loc[0].values[0]
        py.loc[year, '1y'] = tot_py.loc[1].values[0]
        py.loc[year, '2-4y'] = tot_py.loc[2:4].sum().values[0]

    # Incidence rate among 0, 1, 2-4 year-olds
    inc_rate = dict()
    for age_grp in ['0y', '1y', '2-4y']:
        inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

    # Produce mean inicence rates of incidence rate during the simulation:
    inc_mean = pd.DataFrame()
    inc_mean['0y_model_output'] = inc_rate['0y'].mean()
    inc_mean['1y_model_output'] = inc_rate['1y'].mean()
    inc_mean['2-4y_model_output'] = inc_rate['2-4y'].mean()

    # calculate death rate
    deaths_df = output['tlo.methods.demography']['death']
    deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
    deaths = deaths_df.loc[deaths_df['cause'].str.startswith('Diarrhoea')].groupby('year').size()

    return inc_mean, deaths


inc_by_pathogen = dict()
deaths = dict()
for label, file in output_files.items():
    inc_by_pathogen[label], deaths[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)


def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    for label in results.keys():
        summary_table.update({label: results[label][column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("Diarrhoea_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()


# Plot incidence by pathogen: across the sceanrios
for column_of_interest in inc_by_pathogen[list(inc_by_pathogen.keys())[0]].columns:
    plot_for_column_of_interest(inc_by_pathogen, column_of_interest)

# Plot death rates by year: across the scenarios
data = {}
for label in deaths.keys():
    data.update({label: deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Number of Deaths Due to Diarrhoea')
plt.savefig(outputpath / ("Diarrhoea_deaths_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()
