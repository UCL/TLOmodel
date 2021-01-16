from pathlib import Path
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
    dx_algorithm_child,
    ncds
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
    end_date = Date(2012, 1, 2)
    popsize = 100

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim


sim = runsim()

output = parse_log_file(sim.log_filepath)

# ----------------------------------------------- CREATE PREVALENCE PLOTS ----------------------------------------------

# list all of the conditions in the module to loop through
conditions = sim.modules['Ncds'].conditions

transform_output = lambda x: pd.concat([
    pd.Series(name=x.iloc[i]['date'], data=x.iloc[i]['data']) for i in range(len(x))
], axis=1, sort=False).transpose()


def restore_multi_index(dat):
    """restore the multi-index that had to be flattened to pass through the logger"""
    cols = dat.columns
    index_value_list = list()
    for col in cols.str.split('__'):
        index_value_list.append(tuple(component.split('=')[1] for component in col))
    index_name_list = tuple(component.split('=')[0] for component in cols[0].split('__'))
    dat.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    return dat


for condition in conditions:
    # Strip leading 'nc_' from condition name
    condition_name = condition.replace('nc_', '')
    # Capitalize and replace underscores with spaces for title
    condition_title = condition_name.replace("_", " ")
    condition_title = condition_title.title()

    # Plot prevalence by age and sex for each condition
    prev_condition_age_sex = restore_multi_index(
        transform_output(
            output['tlo.methods.ncds'][f'{condition_name}_prevalence_by_age_and_sex']
        )
    )

    prev_condition_age_sex.iloc[-1].transpose().plot.bar()
    plt.ylabel(f'Proportion With {condition_title}')
    plt.title(f'Prevalence of {condition_title}')
    plt.savefig(outputpath / f'prevalence_{condition_title}_by_age.pdf')
    # plt.ylim([0, 1])
    plt.show()

    # Plot prevalence among all adults for each condition
    prev_condition_all = output['tlo.methods.ncds'][f'{condition_name}_prevalence']

    prev_condition_all['year'] = pd.to_datetime(prev_condition_all['date']).dt.year
    prev_condition_all.drop(columns='date', inplace=True)
    prev_condition_all.set_index('year', drop=True, inplace=True)

    plt.plot(prev_condition_all)
    plt.ylabel(f'Prevalence of {condition_title} Over Time')
    plt.xlabel('Year')
    plt.show()


# Plot prevalence of multi-morbidities
prev_mm_age_sex = output['tlo.methods.ncds']['mm_prevalence_by_age_sex']
prev_mm_age_sex['year'] = pd.to_datetime(prev_mm_age_sex['date']).dt.year
prev_mm_age_sex.drop(columns='date', inplace=True)
prev_mm_age_sex.set_index(
    'year',
    drop=True,
    inplace=True
)


# ----------------------------------------------- CREATE INCIDENCE PLOTS ----------------------------------------------

# Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = output['tlo.methods.ncds']['incidence_count_by_condition']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    # import conditions and age range from modules
    conditions = sim.modules['Ncds'].conditions
    age_range = sim.modules['Demography'].AGE_RANGE_CATEGORIES

    # create empty dict to store incidence rates
    inc_rate = dict()

    for condition in conditions:
        # get person-years of time lived without condition
        py_ = output['tlo.methods.ncds'][f'person_years_{condition}']
        years = pd.to_datetime(py_['date']).dt.year
        py = pd.DataFrame(index=years, columns=age_range)

        for year in years:
            tot_py = (
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
                (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
            ).transpose()
            py.loc[year, :] = tot_py.T.iloc[0]

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
    inc_mean = inc_mean.replace([np.inf, -np.inf, np.nan],0)

    return inc_mean

inc_by_condition = get_incidence_rate_and_death_numbers_from_logfile(sim.log_filepath)


def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    summary_table.update({'No_Treatment': results[column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("NCDs_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()


# Plot incidence by condition

for column_of_interest in inc_by_condition.columns:
    plot_for_column_of_interest(inc_by_condition, column_of_interest)
