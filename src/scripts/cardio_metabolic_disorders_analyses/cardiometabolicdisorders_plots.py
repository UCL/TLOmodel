"""
This file runs the CardioMetabolicDisorders module and outputs graphs with data for comparison for incidence,
prevalence, and deaths. It also produces a csv file of prevalence of different co-morbidities.
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    bladder_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# ------------------------------------------------- RUN THE SIMULATION -------------------------------------------------

def runsim(seed=0):
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging

    start_date = Date(2010, 1, 1)
    end_date = Date(2020, 12, 31)
    popsize = 20000

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim


sim = runsim()

output = parse_log_file(sim.log_filepath)

# ----------------------------------------------- SET UP FUNCTIONS ----------------------------------------------

# list all of the conditions in the module to loop through
conditions = sim.modules['CardioMetabolicDisorders'].conditions
age_range = sim.modules['Demography'].AGE_RANGE_CATEGORIES


# set up some functions
def transform_output(x):
    return pd.concat([pd.Series(name=x.iloc[i]['date'], data=x.iloc[i]['data']) for i in range(len(x))], axis=1,
                     sort=False).transpose()


def restore_multi_index(dat):
    """restore the multi-index that had to be flattened to pass through the logger"""
    cols = dat.columns
    index_value_list = list()
    for col in cols.str.split('__'):
        index_value_list.append(tuple(component.split('=')[1] for component in col))
    index_name_list = tuple(component.split('=')[0] for component in cols[0].split('__'))
    dat.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    return dat


def convert_output(output_path):
    output_path['year'] = pd.to_datetime(output_path['date']).dt.year
    output_path.drop(columns='date', inplace=True)
    output_path.set_index(
        'year',
        drop=True,
        inplace=True
    )
    return output_path


# ----------------------------------------------- CREATE PREVALENCE PLOTS ----------------------------------------------

for condition in conditions:
    # Capitalize and replace underscores with spaces for title
    condition_title = condition.replace("_", " ")
    condition_title = condition_title.title()

    # Plot prevalence by age and sex for each condition
    prev_condition_age_sex = restore_multi_index(
        transform_output(
            output['tlo.methods.cardio_metabolic_disorders'][f'{condition}_prevalence_by_age_and_sex']
        )
    )

    # get prevalence + lower and upper values
    prev_range = pd.read_excel("resources/cmd/ResourceFile_cmd_condition_prevalence.xlsx", sheet_name=None)
    asymptomatic_error = [(prev_range[f'{condition}']['value'].values - prev_range[f'{condition}']['lower'].values),
                          (prev_range[f'{condition}']['upper'].values - prev_range[f'{condition}']['value'].values)]

    bar_width = 0.75
    opacity = 0.25

    prev_df = pd.DataFrame(index=["M/0-4", "M/5-9", "M/10-14", "M/15-19", "M/20-24", "M/25-29", "M/30-34", "M/35-39",
                                  "M/40-44", "M/45-49", "M/50-54", "M/55-59", "M/60-64", "M/65-69", "M/70-74",
                                  "M/75-79", "M/80-84", "M/85-89", "M/90-94", "M/95-99", "M/100+", "F/0-4", "F/5-9",
                                  "F/10-14", "F/15-19", "F/20-24", "F/25-29",
                                  "F/30-34", "F/35-39", "F/40-44", "F/45-49", "F/50-54", "F/55-59", "F/60-64",
                                  "F/65-69", "F/70-74", "F/75-79", "F/80-84", "F/85-89", "F/90-94", "F/95-99",
                                  "F/100+", ])
    prev_df['model_prevalence'] = prev_condition_age_sex.iloc[-1].transpose().values

    bar = plt.bar(prev_df.index, prev_df['model_prevalence'], bar_width,
                  alpha=opacity,
                  color='b',
                  label='Model')
    scatter = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values, s=20,
                          alpha=1.0,
                          color='gray',
                          label="Data")
    plt.xticks(rotation=90)
    plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values, yerr=asymptomatic_error,
                 fmt='o', c='gray')
    plt.ylabel(f'Proportion With {condition_title}')
    plt.title(f'Prevalence of {condition_title} by Age and Sex')
    plt.legend([bar, scatter], ['Model', 'Data'])
    plt.savefig(outputpath / f'prevalence_{condition_title}_by_age_sex.pdf')
    # plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

    # Plot prevalence among all adults over time for each condition
    prev_condition_all = output['tlo.methods.cardio_metabolic_disorders'][f'{condition}_prevalence']
    plt.plot(prev_condition_all)
    plt.ylabel(f'Prevalence of {condition_title} Over Time (Ages 20+)')
    plt.xlabel('Year')
    plt.ylim([0, 0.4])
    plt.tight_layout()
    plt.savefig(outputpath / f'prevalence_{condition_title}_over_time.pdf')
    plt.show()

# plot prevalence of multi-morbidities
age_range = sim.modules['Demography'].AGE_RANGE_CATEGORIES
prev_mm_age_all = convert_output(output['tlo.methods.cardio_metabolic_disorders']['mm_prevalence_by_age_all'])
last_year = prev_mm_age_all.iloc[-1, :].to_frame(name="counts")
n_conditions_by_age = pd.DataFrame(index=last_year.index, columns=age_range)
for age_grp in age_range:
    n_conditions_by_age[age_grp] = last_year['counts'].apply(lambda x: x.get(f'{age_grp}')).dropna()

n_conditions_by_age.T.plot.bar(stacked=True)
plt.title("Prevalence of number of co-morbidities by age")
plt.ylabel("Prevalence of number of co-morbidities")
plt.xticks(rotation=90)
plt.legend(loc=(1.04, 0))
plt.tight_layout()
plt.savefig(outputpath / ("N_comorbidities_by_age_all" + datestamp + ".pdf"), format='pdf')
plt.show()

# ----------------------------------------------- CREATE DEATH PLOTS --------------------------------------------------
# calculate death rate
deaths_df = output['tlo.methods.demography']['death']
deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
deaths_df.to_csv('deaths_by_cause.csv')

pop_df = output['tlo.methods.demography']['population']
pop_df['year'] = pd.to_datetime(pop_df['date']).dt.year

deaths = pd.DataFrame(index=deaths_df['year'].unique(), columns=conditions)
rate_deaths = pd.DataFrame(index=deaths_df['year'].unique(), columns=conditions)

total_deaths = pd.DataFrame(index=deaths_df['year'].unique())
total_deaths['total_deaths'] = deaths_df.groupby('year')['cause'].count()

total_pop = pd.DataFrame(index=pop_df['year'].unique())
total_pop['total_pop'] = pop_df.groupby('year')['total'].sum()
total_pop['total_pop_adjustment_factor'] = 100000 / total_pop['total_pop']

for condition in conditions:
    deaths[condition] = deaths_df.loc[deaths_df['cause'].str.startswith(f'{condition}')].groupby('year').size()
    rate_deaths[condition] = deaths[condition] * total_pop['total_pop_adjustment_factor']  # gives rate per 100k pop

# death rates for conditions from GBD
deaths_data = [{'nc_diabetes': 19.04, 'nc_hypertension': 0, 'nc_chronic_kidney_disease': 18.45,
                'nc_lower_back_pain': 0, 'nc_chronic_ischemic_hd': 118.10}]
deaths_lower = [{'nc_diabetes': 17.73, 'nc_hypertension': 0, 'nc_chronic_kidney_disease': 16.98,
                 'nc_lower_back_pain': 0, 'nc_chronic_ischemic_hd': 108.51}]
deaths_upper = [{'nc_diabetes': 20.24, 'nc_hypertension': 0, 'nc_chronic_kidney_disease': 19.70,
                 'nc_lower_back_pain': 0, 'nc_chronic_ischemic_hd': 125.93}]
df_death_data = pd.DataFrame(deaths_data)
df_death_lower = pd.DataFrame(deaths_lower)
df_death_upper = pd.DataFrame(deaths_upper)
asymptomatic_error = [[1.31, 0, 1.47, 0, 9.59], [1.2, 0, 1.25, 0, 7.83]]

bar_width = 0.5
opacity = 0.25

bar = plt.bar(conditions, deaths.iloc[-1], bar_width,
              alpha=opacity,
              color='b',
              label='Model')
scatter = plt.scatter(conditions, df_death_data.iloc[-1], s=20,
                      alpha=1.0,
                      color='gray',
                      label="GBD Data")
plt.xticks(rotation=90)
plt.errorbar(conditions, df_death_data.iloc[-1], yerr=asymptomatic_error, fmt='o', c='gray')
plt.ylabel('Death Rate (per 100k Population)')
plt.title('Death Rate by Condition (2020)')
plt.legend([bar, scatter], ['Model', 'GBD'])
plt.savefig(outputpath / 'deaths_by_condition.pdf')
# plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# ----------------------------------------------- RETRIEVE COMBINATION OF CONDITIONS ----------------------------------

prop_combos = convert_output(output['tlo.methods.cardio_metabolic_disorders']['prop_combos'])
last_year = prop_combos.iloc[-1, :].to_frame(name="props")
props_by_age = pd.DataFrame(index=last_year.index, columns=age_range)
for age_grp in age_range:
    props_by_age[age_grp] = last_year['props'].apply(lambda x: x.get(f'{age_grp}')).dropna()

props_by_age.to_csv('condition_combos.csv')


# ----------------------------------------------- CREATE INCIDENCE PLOTS ----------------------------------------------

# Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = convert_output(output['tlo.methods.cardio_metabolic_disorders']['incidence_count_by_condition'])

    # create empty dict to store incidence rates
    inc_rate = dict()

    for condition in conditions:
        # get person-years of time lived without condition
        py_ = output['tlo.methods.cardio_metabolic_disorders'][f'person_years_{condition}']
        years = pd.to_datetime(py_['date']).dt.year
        py = pd.DataFrame(index=years, columns=age_range)

        for year in years:
            tot_py = (
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
            ).transpose()
            py.loc[year, :] = tot_py.T.iloc[0]
            py = py.replace(0, np.nan)

        condition_counts = pd.DataFrame(index=years, columns=age_range)

        for age_grp in age_range:
            # extract specific condition counts from counts df
            condition_counts[age_grp] = counts[f'{condition}'].apply(lambda x: x.get(f'{age_grp}')).dropna()
            individual_condition = condition_counts[age_grp].apply(pd.Series).div(py[age_grp],
                                                                                  axis=0).dropna()
            individual_condition.columns = [f'{condition}']
            if condition == conditions[0]:
                inc_rate[age_grp] = condition_counts[age_grp].apply(pd.Series).div(py[age_grp],
                                                                                   axis=0).dropna()
                inc_rate[age_grp].columns = [f'{condition}']
            else:
                inc_rate[age_grp] = inc_rate[age_grp].join(individual_condition)

    # Produce mean incidence rates of incidence rate during the simulation:
    inc_mean = pd.DataFrame()
    for age_grp in age_range:
        inc_mean[age_grp] = inc_rate[age_grp].mean()

    # replace NaNs and inf's with 0s
    inc_mean = inc_mean.replace([np.inf, -np.inf, np.nan], 0)

    return inc_mean


inc_by_condition = get_incidence_rate_and_death_numbers_from_logfile(sim.log_filepath)


def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    summary_table.update({'No_Treatment': results[column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("CardioMetabolicDisorders_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()


# Plot incidence by condition

for column_of_interest in inc_by_condition.columns:
    plot_for_column_of_interest(inc_by_condition, column_of_interest)
