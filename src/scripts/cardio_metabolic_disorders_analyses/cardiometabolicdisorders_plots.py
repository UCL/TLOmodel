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
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    bladder_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    simplified_births,
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
    end_date = Date(2019, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim


sim = runsim()

output = parse_log_file(sim.log_filepath)

# ---------------------------------------- COMPARE OUTPUTS OF MODEL TO DATA -------------------------------------

# Get comparison
comparison = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=resourcefilepath)
condition_names = ["Diabetes", "Heart Disease", "Kidney Disease", "Stroke"]

for cond in condition_names:

    age_cats = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',
                '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99']

    men_GBD = []
    men_model = []
    for age_cat in age_cats:
        men_GBD.append(comparison.loc[('2015-2019', 'M', f'{age_cat}', f'{cond}')]['GBD_mean'])
        men_model.append(comparison.loc[('2015-2019', 'M', f'{age_cat}', f'{cond}')]['model'])

    x = np.arange(len(age_cats))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_GBD, width, label='GBD mean')
    rects2 = ax.bar(x + width/2, men_model, width, label='Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Deaths')
    ax.set_title(f'Annual Deaths from {cond} (Men, 2015-2019)')
    ax.set_xticks(x)
    ax.set_xticklabels(age_cats, rotation=90)
    ax.legend()

    fig.tight_layout()

    plt.show()

    women_GBD = []
    women_model = []
    for age_cat in age_cats:
        women_GBD.append(comparison.loc[('2015-2019', 'F', f'{age_cat}', f'{cond}')]['GBD_mean'])
        women_model.append(comparison.loc[('2015-2019', 'F', f'{age_cat}', f'{cond}')]['model'])

    x = np.arange(len(age_cats))
    width = 0.35

    fig, ax = plt.subplots()
    rects3 = ax.bar(x - width/2, women_GBD, width, label='GBD mean')
    rects4 = ax.bar(x + width/2, women_model, width, label='Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Deaths')
    ax.set_title(f'Annual Deaths from {cond} (Women, 2015-2019)')
    ax.set_xticks(x)
    ax.set_xticklabels(age_cats, rotation=90)
    ax.legend()

    fig.tight_layout()

    plt.show()

# ----------------------------------------------- SET UP FUNCTIONS ----------------------------------------------

# list all of the conditions in the module to loop through
conditions = sim.modules['CardioMetabolicDisorders'].conditions
events = sim.modules['CardioMetabolicDisorders'].events
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
    baseline_error = [(prev_range[f'{condition}']['value'].values - prev_range[f'{condition}']['lower'].values),
                          (prev_range[f'{condition}']['upper'].values - prev_range[f'{condition}']['value'].values)]
    if 'gbd_value' in prev_range[f'{condition}']:
        gbd_error = [(prev_range[f'{condition}']['gbd_value'].values - prev_range[f'{condition}'][
            'gbd_lower'].values),
                          (prev_range[f'{condition}']['gbd_upper'].values - prev_range[
                              f'{condition}']['gbd_value'].values)]
    if 'steps_value' in prev_range[f'{condition}']:
        steps_error = [(prev_range[f'{condition}']['steps_value'].values - prev_range[f'{condition}']['steps_lower'].values),
                          (prev_range[f'{condition}']['steps_upper'].values - prev_range[
                              f'{condition}']['steps_value'].values)]

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

    if condition == 'diabetes':
        scatter = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['gbd_value'].values, s=8,
                              alpha=0.8,
                              color='gray',
                              label="GBD 2019")
        scatter_price = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values,
                                    s=8,
                                    alpha=0.8,
                                    color='hotpink',
                                    label="Price et al. 2018")
        scatter_steps = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['steps_value'].values,
                                    s=8,
                                    alpha=0.8,
                                    color='#23395d',
                                    label="STEPS Survey 2017")
        plt.xticks(rotation=90)
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['gbd_value'].values,
                     yerr=gbd_error,
                     fmt='x', c='gray')
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values,
                     yerr=baseline_error,
                     fmt='o', c='hotpink')
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['steps_value'].values,
                     yerr=steps_error,
                     fmt='*', c='#23395d')
        plt.ylabel(f'Proportion With {condition_title}')
        plt.title(f'Prevalence of {condition_title} by Age and Sex')
        plt.legend([bar, scatter, scatter_price, scatter_steps], ['Model', 'GBD 2019', 'Price et al. 2018',
                                                                  'STEPS Survey 2017'])
        plt.savefig(outputpath / f'prevalence_{condition_title}_by_age_sex.pdf')
        # plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()
    elif condition == 'hypertension':
        scatter_price = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values,
                                    s=8,
                                    alpha=0.8,
                                    color='hotpink',
                                    label="Price et al. 2018")
        scatter_steps = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['steps_value'].values,
                                    s=8,
                                    alpha=0.8,
                                    color='#23395d',
                                    label="STEPS Survey 2017")
        plt.xticks(rotation=90)
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values,
                     yerr=baseline_error,
                     fmt='o', c='hotpink')
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['steps_value'].values,
                     yerr=steps_error,
                     fmt='*', c='#23395d')
        plt.ylabel(f'Proportion With {condition_title}')
        plt.title(f'Prevalence of {condition_title} by Age and Sex')
        plt.legend([bar, scatter_price, scatter_steps], ['Model', 'Price et al. 2018', 'STEPS Survey 2017'])
        plt.savefig(outputpath / f'prevalence_{condition_title}_by_age_sex.pdf')
        # plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()
    elif condition == 'chronic_kidney_disease':
        scatter = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['gbd_value'].values, s=8,
                              alpha=0.8,
                              color='gray',
                              label="GBD 2019")
        scatter_price = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values,
                                    s=8,
                                    alpha=0.8,
                                    color='hotpink',
                                    label="Nakanga et al. 2018")
        plt.xticks(rotation=90)
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['gbd_value'].values,
                     yerr=gbd_error,
                     fmt='x', c='gray')
        plt.ylabel(f'Proportion With {condition_title}')
        plt.title(f'Prevalence of {condition_title} by Age and Sex')
        plt.legend([bar, scatter, scatter_price], ['Model', 'GBD 2019', 'Nakanga et al. 2018'])
        plt.savefig(outputpath / f'prevalence_{condition_title}_by_age_sex.pdf')
        # plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()
    else:
        scatter = plt.scatter(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values, s=8,
                              alpha=0.8,
                              color='gray',
                              label="GBD 2019")
        plt.xticks(rotation=90)
        plt.errorbar(prev_range[f'{condition}'].index, prev_range[f'{condition}']['value'].values, yerr=baseline_error,
                 fmt='x', c='gray')
        plt.ylabel(f'Proportion With {condition_title}')
        plt.title(f'Prevalence of {condition_title} by Age and Sex')
        plt.legend([bar, scatter], ['Model', 'GBD 2019'])
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

for event in events:
    # Capitalize and replace underscores with spaces for title
    event_title = event.replace("ever_", " ")
    event_title = event_title.title()

    # Plot prevalence by age and sex for each condition
    prev_event_age_sex = restore_multi_index(
        transform_output(
            output['tlo.methods.cardio_metabolic_disorders'][f'{event}_prevalence_by_age_and_sex']
        )
    )

    # get prevalence + lower and upper values
    prev_range = pd.read_excel("resources/cmd/ResourceFile_cmd_event_prevalence.xlsx", sheet_name=None)
    asymptomatic_error = [(prev_range[f'{event}']['value'].values - prev_range[f'{event}']['lower'].values),
                          (prev_range[f'{event}']['upper'].values - prev_range[f'{event}']['value'].values)]

    bar_width = 0.75
    opacity = 0.25

    prev_df = pd.DataFrame(index=["M/0-4", "M/5-9", "M/10-14", "M/15-19", "M/20-24", "M/25-29", "M/30-34", "M/35-39",
                                  "M/40-44", "M/45-49", "M/50-54", "M/55-59", "M/60-64", "M/65-69", "M/70-74",
                                  "M/75-79", "M/80-84", "M/85-89", "M/90-94", "M/95-99", "M/100+", "F/0-4", "F/5-9",
                                  "F/10-14", "F/15-19", "F/20-24", "F/25-29",
                                  "F/30-34", "F/35-39", "F/40-44", "F/45-49", "F/50-54", "F/55-59", "F/60-64",
                                  "F/65-69", "F/70-74", "F/75-79", "F/80-84", "F/85-89", "F/90-94", "F/95-99",
                                  "F/100+", ])
    prev_df['model_prevalence'] = prev_event_age_sex.iloc[-1].transpose().values

    bar = plt.bar(prev_df.index, prev_df['model_prevalence'], bar_width,
                  alpha=opacity,
                  color='b',
                  label='Model')
    scatter = plt.scatter(prev_range[f'{event}'].index, prev_range[f'{event}']['value'].values, s=20,
                          alpha=1.0,
                          color='gray',
                          label="GBD 2019")
    plt.xticks(rotation=90)
    plt.errorbar(prev_range[f'{event}'].index, prev_range[f'{event}']['value'].values, yerr=asymptomatic_error,
                 fmt='o', c='gray')
    plt.ylabel(f'Proportion With {event_title}')
    plt.title(f'Prevalence of {event_title} by Age and Sex')
    plt.legend([bar, scatter], ['Model', 'Data'])
    plt.savefig(outputpath / f'prevalence_{event_title}_by_age_sex.pdf')
    # plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

    # Plot prevalence among all adults over time for each condition
    prev_event_all = output['tlo.methods.cardio_metabolic_disorders'][f'{event}_prevalence']
    plt.plot(prev_event_all)
    plt.ylabel(f'Prevalence of {event_title} Over Time (Ages 20+)')
    plt.xlabel('Year')
    plt.ylim([0, 0.4])
    plt.tight_layout()
    plt.savefig(outputpath / f'prevalence_{event_title}_over_time.pdf')
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

# ----------------------------------------------- COMBINATIONS OF CONDITIONS ----------------------------------

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

# Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile_events(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = convert_output(output['tlo.methods.cardio_metabolic_disorders']['incidence_count_by_event'])

    # create empty dict to store incidence rates
    inc_rate = dict()

    for event in events:
        # get person-years of time lived without condition
        py_ = output['tlo.methods.cardio_metabolic_disorders'][f'person_years_{event}']
        years = pd.to_datetime(py_['date']).dt.year
        py = pd.DataFrame(index=years, columns=age_range)

        for year in years:
            tot_py = (
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
            ).transpose()
            py.loc[year, :] = tot_py.T.iloc[0]
            py = py.replace(0, np.nan)

        event_counts = pd.DataFrame(index=years, columns=age_range)

        for age_grp in age_range:
            # extract specific condition counts from counts df
            event_counts[age_grp] = counts[f'{event}'].apply(lambda x: x.get(f'{age_grp}')).dropna()
            individual_condition = event_counts[age_grp].apply(pd.Series).div(py[age_grp],
                                                                                  axis=0).dropna()
            individual_condition.columns = [f'{event}']
            if event == events[0]:
                inc_rate[age_grp] = event_counts[age_grp].apply(pd.Series).div(py[age_grp],
                                                                                   axis=0).dropna()
                inc_rate[age_grp].columns = [f'{event}']
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
inc_by_event = get_incidence_rate_and_death_numbers_from_logfile_events(sim.log_filepath)


# def plot_for_column_of_interest(results, column_of_interest):
    # summary_table = dict()
    # summary_table.update({'No_Treatment': results[column_of_interest]})
    # data = 100 * pd.concat(summary_table, axis=1)
    # data.plot.bar()
    # plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    # plt.savefig(outputpath / ("CardioMetabolicDisorders_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    # plt.tight_layout()
    # plt.show()


# Plot incidence by condition

# for column_of_interest in inc_by_condition.columns:
    # plot_for_column_of_interest(inc_by_condition, column_of_interest)

# for column_of_interest in inc_by_event.columns:
    # plot_for_column_of_interest(inc_by_event, column_of_interest)

def make_incidence_plot(condition):
    # Capitalize and replace underscores with spaces for title
    condition_title = condition.replace("_", " ")
    condition_title = condition_title.title()

    inc_range = pd.read_excel("resources/cmd/ResourceFile_cmd_condition_and_events_incidence.xlsx", sheet_name=None)
    asymptomatic_error = [(inc_range[f'{condition}']['value'].values - inc_range[f'{condition}']['lower'].values),
                          (inc_range[f'{condition}']['upper'].values - inc_range[f'{condition}']['value'].values)]

    bar_width = 0.75
    opacity = 0.25

    inc_df = pd.DataFrame(index=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
                                  "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                                  "75-79", "80-84", "85-89", "90-94", "95-99", "100+"])
    if condition == 'ever_stroke':
        inc_df['model_incidence'] = inc_by_event.loc[f'{condition}'].transpose().values
    else:
        inc_df['model_incidence'] = inc_by_condition.loc[f'{condition}'].transpose().values

    inc_df['model_incidence'] = inc_df['model_incidence'] * 100

    bar = plt.bar(inc_df.index, inc_df['model_incidence'], bar_width,
                  alpha=opacity,
                  color='b',
                  label='Model')
    scatter = plt.scatter(inc_range[f'{condition}'].index, inc_range[f'{condition}']['value'].values, s=20,
                          alpha=1.0,
                          color='gray',
                          label="Data")
    plt.xticks(rotation=90)
    plt.errorbar(inc_range[f'{condition}'].index, inc_range[f'{condition}']['value'].values, yerr=asymptomatic_error,
                 fmt='o', c='gray')
    plt.ylabel(f'Incidence of {condition_title} per 100 PY')
    plt.title(f'Incidence of {condition_title} by Age')
    plt.legend([bar, scatter], ['Model', 'GBD 2019'])
    plt.savefig(outputpath / f'incidence_{condition_title}_by_age.pdf')
    plt.tight_layout()
    plt.show()

conditions_and_events_for_incidence = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd',
                                       'chronic_lower_back_pain', 'ever_stroke']

for condition in conditions_and_events_for_incidence:
    make_incidence_plot(condition)

