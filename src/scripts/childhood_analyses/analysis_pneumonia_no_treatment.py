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
end_date = Date(2013, 1, 1)
popsize = 1000

for label, service_avail in scenarios.items():
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 pneumonia.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath


# %% ----------------------------  INCIDENCE RATE OF PNEUMONIA BY PATHOGEN  ----------------------------

#  Calculate the "incidence rate" from the output counts of incidence
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)
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
        # py.loc[year, '<5y'] = tot_py.loc[0:4].sum().values[0]

    # # get population size to make a comparison
    pop = output['tlo.methods.demography']['num_children']
    pop.set_index(
        'date',
        drop=True,
        inplace=True
    )
    pop.columns = pop.columns.astype(int)
    pop['0y'] = pop[0]
    pop['1y'] = pop[1]
    pop['2-4y'] = pop[2] + pop[3] + pop[4]
    # pop['<5y'] = pop[0] + pop[1] + pop[2] + pop[3] + pop[4]
    pop.drop(columns=[x for x in range(5)], inplace=True)

    # Incidence rate among 0, 1, 2-4 year-olds
    inc_rate = dict()
    for age_grp in ['0y', '1y', '2-4y']:
        inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

    # Load the incidence rate data to which we calibrate
    calibration_incidence_rate_0_year_olds = {
        'RSV': 9.7007598 / 100.0,
        'Rhinovirus': 9.7007598 / 100.0,
        'HMPV': 9.7007598 / 100.0,
        'Parainfluenza': 9.7007598 / 100.0,
        'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
        'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
        'Hib': 9.7007598 / 100.0,
        'H.influenzae_non_type_b': 9.7007598 / 100.0,
        'Staph_aureus': 9.7007598 / 100.0,
        'Enterobacteriaceae': 9.7007598 / 100.0,
        'other_Strepto_Enterococci': 9.7007598 / 100.0,
        'Influenza': 9.7007598 / 100.0,
        'P.jirovecii': 9.7007598 / 100.0,
        'Bocavirus': 9.7007598 / 100.0,
        'Adenovirus': 9.7007598 / 100.0,
        'other_viral_pathogens': 9.7007598 / 100.0,
        'other_bacterial_pathogens': 9.7007598 / 100.0
    }

    calibration_incidence_rate_1_year_olds = {
        'RSV': 9.7007598 / 100.0,
        'Rhinovirus': 9.7007598 / 100.0,
        'HMPV': 9.7007598 / 100.0,
        'Parainfluenza': 9.7007598 / 100.0,
        'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
        'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
        'Hib': 9.7007598 / 100.0,
        'H.influenzae_non_type_b': 9.7007598 / 100.0,
        'Staph_aureus': 9.7007598 / 100.0,
        'Enterobacteriaceae': 9.7007598 / 100.0,
        'other_Strepto_Enterococci': 9.7007598 / 100.0,
        'Influenza': 9.7007598 / 100.0,
        'P.jirovecii': 9.7007598 / 100.0,
        'Bocavirus': 9.7007598 / 100.0,
        'Adenovirus': 9.7007598 / 100.0,
        'other_viral_pathogens': 9.7007598 / 100.0,
        'other_bacterial_pathogens': 9.7007598 / 100.0
    }

    calibration_incidence_rate_2_to_4_year_olds = {
        'RSV': 9.7007598 / 100.0,
        'Rhinovirus': 9.7007598 / 100.0,
        'HMPV': 9.7007598 / 100.0,
        'Parainfluenza': 9.7007598 / 100.0,
        'Strep_pneumoniae_PCV13': 9.7007598 / 100.0,
        'Strep_pneumoniae_non_PCV13': 9.7007598 / 100.0,
        'Hib': 9.7007598 / 100.0,
        'H.influenzae_non_type_b': 9.7007598 / 100.0,
        'Staph_aureus': 9.7007598 / 100.0,
        'Enterobacteriaceae': 9.7007598 / 100.0,
        'other_Strepto_Enterococci': 9.7007598 / 100.0,
        'Influenza': 9.7007598 / 100.0,
        'P.jirovecii': 9.7007598 / 100.0,
        'Bocavirus': 9.7007598 / 100.0,
        'Adenovirus': 9.7007598 / 100.0,
        'other_viral_pathogens': 9.7007598 / 100.0,
        'other_bacterial_pathogens': 9.7007598 / 100.0
    }

    # Produce a set of line plot comparing to the calibration data
    fig, axes = plt.subplots(ncols=2, nrows=5, sharey=True, figsize=(10, 20))
    for ax_num, pathogen in enumerate(sim.modules['ALRI'].pathogens):
        ax = fig.axes[ax_num]
        inc_rate['0y'][pathogen].plot(ax=ax, label='Model output')
        ax.hlines(y=calibration_incidence_rate_0_year_olds[pathogen],  # axhlines is to plot horizontal lines at each y
                  xmin=min(inc_rate['0y'].index),
                  xmax=max(inc_rate['0y'].index),
                  label='calibrating_data'
                  )
        ax.set_title(f'{pathogen}')
        ax.set_xlabel("Year")
        ax.set_ylabel("Incidence Rate")
        ax.legend()
    plt.title('Incidence Rate among <1 year old')
    plt.savefig(outputpath / ("ALRI_inc_rate_by_pathogen_0_year_olds" + datestamp + ".pdf"), format='pdf')
    plt.show()

    # Produce a bar plot for means of incidence rate during the simulation:
    inc_mean = pd.DataFrame()
    inc_mean['0y_model_output'] = inc_rate['0y'].mean()
    inc_mean['1y_model_output'] = inc_rate['1y'].mean()
    inc_mean['2-4y_model_output'] = inc_rate['2-4y'].mean()

    # put in the inputs:
    inc_mean['0y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_0_year_olds)
    inc_mean['1y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_1_year_olds)
    inc_mean['2-4y_calibrating_data'] = pd.Series(data=calibration_incidence_rate_2_to_4_year_olds)

    # 0 year-olds
    inc_mean.plot.bar(y=['0y_model_output', '0y_calibrating_data'])
    plt.title('Incidence Rate: 0 year-olds')
    plt.xlabel('Pathogen')
    plt.ylabel('Risk of pathogen causing pneumonia per year')
    plt.savefig(outputpath / ("ALRI_inc_rate_calibration_0_year_olds" + datestamp + ".pdf"), format='pdf')
    plt.tight_layout()
    plt.show()

    # 1 year-olds
    inc_mean.plot.bar(y=['1y_model_output', '1y_calibrating_data'])
    plt.title('Incidence Rate: 1 year-olds')
    plt.xlabel('Pathogen')
    plt.ylabel('Risk of pathogen causing pneumonia per year')
    plt.xlabel('Pathogen')
    plt.ylabel('Risk of pathogen causing pneumonia per year')
    plt.savefig(outputpath / ("ALRI_inc_rate_calibration_1_year_olds" + datestamp + ".pdf"), format='pdf')
    plt.tight_layout()
    plt.show()

    # 2-4 year-olds
    inc_mean.plot.bar(y=['2-4y_model_output', '2-4y_calibrating_data'])
    plt.title('Incidence Rate: 2-4 year-olds')
    plt.xlabel('Pathogen')
    plt.ylabel('Risk of pathogen causing pneumonia per year')
    plt.xlabel('Pathogen')
    plt.ylabel('Risk of pathogen causing pneumonia per year')
    plt.savefig(outputpath / ("ALRI_inc_rate_calibration_2-4_year_olds" + datestamp + ".pdf"), format='pdf')
    plt.tight_layout()
    plt.show()

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

# Plot death rates by year: across the scenarios
data = {}
print(alri_deaths)
for label in alri_deaths.keys():
    data.update({label: alri_deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Number of Deaths Due to ALRI')
plt.savefig(outputpath / ("ALRI_deaths_by_scenario" + datestamp + ".pdf"), format='pdf')
plt.show()

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

