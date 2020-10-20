# todo: run the model and make plots of prevalence and incidence of each conditons wrt age/sex
#  make helpfer functions to make the plots for each condition and then loop through them all

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

scenarios = dict()
scenarios['No_Treatment'] = []

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 2)
popsize = 10000

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
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath),
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
    counts = output['tlo.methods.ncds']['incidence_count_by_condition']
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
    py = pd.DataFrame(index=years, columns=['0-4y', '5-9y', '10-14y', '15-19y', '20-24y', '25-29y',
                                            '30-34y', '35-39y', '40-44y', '45-49y', '50-54y',
                                            '55-59y', '60-64y', '65-69y', '70-74y', '75-79y',
                                            '80-84y', '85-89y', '90-94y', '95-99y', '100+y'])
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
        py.loc[year, '0-4y'] = tot_py.loc[0:4].sum().values[0]
        py.loc[year, '5-9y'] = tot_py.loc[5:9].sum().values[0]
        py.loc[year, '10-14y'] = tot_py.loc[10:14].sum().values[0]
        py.loc[year, '15-19y'] = tot_py.loc[15:19].sum().values[0]
        py.loc[year, '20-24y'] = tot_py.loc[20:24].sum().values[0]
        py.loc[year, '25-29y'] = tot_py.loc[25:29].sum().values[0]
        py.loc[year, '30-34y'] = tot_py.loc[30:34].sum().values[0]
        py.loc[year, '35-39y'] = tot_py.loc[35:39].sum().values[0]
        py.loc[year, '40-44y'] = tot_py.loc[40:44].sum().values[0]
        py.loc[year, '45-49y'] = tot_py.loc[45:49].sum().values[0]
        py.loc[year, '50-54y'] = tot_py.loc[50:54].sum().values[0]
        py.loc[year, '55-59y'] = tot_py.loc[55:59].sum().values[0]
        py.loc[year, '60-64y'] = tot_py.loc[60:64].sum().values[0]
        py.loc[year, '65-69y'] = tot_py.loc[65:69].sum().values[0]
        py.loc[year, '70-74y'] = tot_py.loc[70:74].sum().values[0]
        py.loc[year, '75-79y'] = tot_py.loc[75:79].sum().values[0]
        py.loc[year, '80-84y'] = tot_py.loc[80:84].sum().values[0]
        py.loc[year, '85-89y'] = tot_py.loc[85:89].sum().values[0]
        py.loc[year, '90-94y'] = tot_py.loc[90:94].sum().values[0]
        py.loc[year, '95-99y'] = tot_py.loc[95:99].sum().values[0]
        py.loc[year, '100+y'] = tot_py.loc[100:120].sum().values[0]

    # Incidence rate among 0, 1, 2-4 year-olds
    inc_rate = dict()
    for age_grp in ['0-4y', '5-9y', '10-14y', '15-19y', '20-24y', '25-29y', '30-34y', '35-39y', '40-44y', '45-49y',
                    '50-54y', '55-59y', '60-64y', '65-69y', '70-74y', '75-79y', '80-84y', '85-89y', '90-94y',
                    '95-99y', '100+y']:
        inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp], axis=0).dropna()

    # Produce mean incidence rates of incidence rate during the simulation:
    inc_mean = pd.DataFrame()
    inc_mean['0-4y_model_output'] = inc_rate['0-4y'].mean()
    inc_mean['5-9y_model_output'] = inc_rate['5-9y'].mean()
    inc_mean['10-14y_model_output'] = inc_rate['10-14y'].mean()
    inc_mean['15-19y_model_output'] = inc_rate['15-19y'].mean()
    inc_mean['20-24y_model_output'] = inc_rate['20-24y'].mean()
    inc_mean['25-29y_model_output'] = inc_rate['25-29y'].mean()
    inc_mean['30-34y_model_output'] = inc_rate['30-34y'].mean()
    inc_mean['35-39y_model_output'] = inc_rate['35-39y'].mean()
    inc_mean['40-44y_model_output'] = inc_rate['40-44y'].mean()
    inc_mean['45-49y_model_output'] = inc_rate['45-49y'].mean()
    inc_mean['50-54y_model_output'] = inc_rate['50-54y'].mean()
    inc_mean['55-59y_model_output'] = inc_rate['55-59y'].mean()
    inc_mean['60-64y_model_output'] = inc_rate['60-64y'].mean()
    inc_mean['65-69y_model_output'] = inc_rate['65-69y'].mean()
    inc_mean['70-74y_model_output'] = inc_rate['70-74y'].mean()
    inc_mean['75-79y_model_output'] = inc_rate['75-79y'].mean()
    inc_mean['80-84y_model_output'] = inc_rate['80-84y'].mean()
    inc_mean['85-89y_model_output'] = inc_rate['85-89y'].mean()
    inc_mean['90-94y_model_output'] = inc_rate['90-94y'].mean()
    inc_mean['95-99y_model_output'] = inc_rate['95-99y'].mean()
    inc_mean['100+y_model_output'] = inc_rate['100+y'].mean()

    # calculate death rate
    # deaths_df = output['tlo.methods.demography']['death']
    # deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
    # deaths = deaths_df.loc[deaths_df['cause'].str.startswith('Diarrhoea')].groupby('year').size()

    return inc_mean

inc_by_condition = dict()
for label, file in output_files.items():
    inc_by_condition[label] = get_incidence_rate_and_death_numbers_from_logfile(file)

def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    for label in results.keys():
        summary_table.update({label: results[label][column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("NCDs_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()

# Plot incidence by condition

for column_of_interest in inc_by_condition[list(inc_by_condition.keys())[0]].columns:
    plot_for_column_of_interest(inc_by_condition, column_of_interest)

# %% Create illustrative plot of individual trajectories of glucose over time.

# Join-up values from each log to create a trace for an individual

# transform_output = lambda x: pd.concat([
    # pd.Series(name=x.iloc[i]['date'], data=x.iloc[i]['data']) for i in range(len(x))
# ], axis=1, sort=False).transpose()

# Show distribution of conditions wrt key demographic variables

# def restore_multi_index(dat):
    # """restore the multi-index that had to be flattened to pass through the logger"""
    # cols = dat.columns
    # index_value_list = list()
    # for col in cols.str.split('__'):
        # index_value_list.append( tuple(component.split('=')[1] for component in col))
    # index_name_list = tuple(component.split('=')[0] for component in cols[0].split('__'))
    # dat.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    # return dat

# Plot prevalence of diabetes by age and sex
# prev_sa = restore_multi_index(
    # transform_output(
        # output['tlo.methods.ncds']['diabetes_prevalence_by_age_and_sex']
    # )
# )

# prev_sa.iloc[-1].transpose().plot.bar()
# plt.ylabel('Proportion With Diabetes')
# plt.title('Prevalence of Diabetes by Age/Sex')
# plt.savefig(outputpath / 'diabetes_prevalence_by_age_and_sex.pdf')
# plt.show()

# Plot prevalence of diabetes by BMI and High Suagr intake:
# prev_bs = restore_multi_index(
    # transform_output(
        # output['tlo.methods.diabetes']['diabetes_prevalence_by_bmi_and_high_sugar']
    # )
# )

# prev_bs.iloc[-1].unstack().plot.bar()
# plt.ylabel('Proportion With Diabetes')
# plt.title('Prevalence of Diabetes by BMI/Sugar Intake')
# plt.savefig(outputpath / 'diabetes_prevalence_by_bmi_and_sugar.pdf')
# plt.show()

