"""
This will run the Pneumonia Module and plot the incidence rate of each pathogen by each age group.
This will then be compared with:
    * The input incidence rate for each pathogen
    * The desired incidence rate for each pathogen
There is no treatment.
"""
# %% Import Statements and initial declarations
import datetime
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    pneumonia,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)
# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# *1: No Treatment
# *2: Some Treatment

scenarios = dict()
scenarios['No_Treatment'] = []
scenarios['Treatment'] = ['*']

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2016, 1, 1)
popsize = 5000

for label, service_avail in scenarios.items():
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 pneumonia.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath


# %% Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = output['tlo.methods.pneumonia']['incidence_count_by_pathogen']
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
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
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
    alri_deaths = deaths_df.loc[deaths_df['cause'].str.startswith('ALRI')].groupby('year').size()

    return inc_mean, alri_deaths


inc_by_pathogen = dict()
alri_deaths = dict()
for label, file in output_files.items():
    inc_by_pathogen[label], alri_deaths[label] = \
        get_incidence_rate_and_death_numbers_from_logfile(file)


def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    for label in results.keys():
        summary_table.update({label: results[label][column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("ALRI_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()


# Plot incidence by pathogen: across the sceanrios
for column_of_interest in inc_by_pathogen[list(inc_by_pathogen.keys())[0]].columns:
    plot_for_column_of_interest(inc_by_pathogen, column_of_interest)

# Plot death rates by year: across the scenarios
data = {}
for label in alri_deaths.keys():
    data.update({label: alri_deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Number of Deaths Due to ALRI')
plt.savefig(outputpath / ("ALRI_deaths_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()


def get_alri_complications_from_logfile(logfile):
    output = parse_log_file(logfile)

    # %% Plot Incidence of Diarrhoea Over time:
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    # Load Model Results on ALRI complications
    complications_per_year_df = output['tlo.methods.pneumonia']['alri_complication_count']
    complications_per_year_df['year'] = pd.to_datetime(complications_per_year_df['date']).dt.year
    complications_per_year_df.drop(columns='date', inplace=True)
    complications_per_year_df.set_index(
        'year',
        drop=True,
        inplace=True
    )
    # pneumothorax_compl_per_year = complications_per_year_df.count_alri_complic_pneumothorax
    # pleural_effusion_compl_per_year = complications_per_year_df.count_alri_complic_pleural_eff
    # respiratory_failure_compl_per_year = complications_per_year_df.count_alri_complic_respiratory_failure

    # ig1, ax = plt.subplots()
    # ax.plot(np.asarray(complications_per_year_df['year']), pneumothorax_compl_per_year)
    # ax.plot(np.asarray(complications_per_year_df['year']), pleural_effusion_compl_per_year)
    # ax.plot(np.asarray(complications_per_year_df['year']), respiratory_failure_compl_per_year)
    #
    # # format the ticks
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(years_fmt)
    #
    # plt.title("ALRI complications by year")
    # plt.xlabel("Year")
    # plt.ylabel("number of cases with complication")
    # plt.legend(['pneumothorax', 'pleural effusion', 'respiratory_failure'])
    # plt.savefig(outputpath + 'ALRI complications by year' + datestamp + '.pdf')
    #
    # plt.show()

    # Incidence rate among 0, 1, 2-4 year-olds
    complications_per_year = dict()
    for complication in ['pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess',
                         'sepsis', 'meningitis', 'respiratory_failure']:
        complications_per_year[complication] = complications_per_year_df[complication].apply(pd.Series).dropna()

    # Produce mean inicence rates of incidence rate during the simulation:
    complic_mean = pd.DataFrame()
    complic_mean['pneumothorax_model_output'] = complications_per_year['pneumothorax'].mean()
    complic_mean['pleural_effusion_model_output'] = complications_per_year['pleural_effusion'].mean()
    complic_mean['empyema_model_output'] = complications_per_year['empyema'].mean()
    complic_mean['lung_abscess_model_output'] = complications_per_year['lung_abscess'].mean()
    complic_mean['sepsis_model_output'] = complications_per_year['sepsis'].mean()
    complic_mean['meningitis_model_output'] = complications_per_year['meningitis'].mean()
    complic_mean['respiratory_failure_model_output'] = complications_per_year['respiratory_failure'].mean()

    return complic_mean


complications_by_pathogen = dict()
deaths = dict()
for label, file in output_files.items():
    complications_by_pathogen[label], deaths[label] = \
        get_alri_complications_from_logfile(file)
