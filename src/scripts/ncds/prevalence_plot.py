import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    ncds,
    pregnancy_supervisor,
    symptommanager,
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

def runsim(seed=0):
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging

    start_date = Date(2010, 1, 1)
    end_date = Date(2012, 1, 2)
    popsize = 500

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


# %% Create illustrative plot of individual trajectories of glucose over time.

transform_output = lambda x: pd.concat([
    pd.Series(name=x.iloc[i]['date'], data=x.iloc[i]['data']) for i in range(len(x))
], axis=1, sort=False).transpose()

# Show distribution of diabetes wrt key demographic variables

def restore_multi_index(dat):
    """restore the multi-index that had to be flattened to pass through the logger"""
    cols = dat.columns
    index_value_list = list()
    for col in cols.str.split('__'):
        index_value_list.append( tuple(component.split('=')[1] for component in col))
    index_name_list = tuple(component.split('=')[0] for component in cols[0].split('__'))
    dat.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    return dat


# Plot prevalence by age and sex
prev_diabetes = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['diabetes_prevalence_by_age_and_sex']
    )
)

prev_hypertension = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['hypertension_prevalence_by_age_and_sex']
    )
)

prev_depression = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['depression_prevalence_by_age_and_sex']
    )
)

prev_clbp = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['chronic_lower_back_pain_prevalence_by_age_and_sex']
    )
)

prev_ckd = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['chronic_kidney_disease_prevalence_by_age_and_sex']
    )
)

prev_cihd = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['cihd_prevalence_by_age_and_sex']
    )
)

prev_diabetes_all = output['tlo.methods.ncds']['diabetes_prevalence']
prev_hypertension_all = output['tlo.methods.ncds']['hypertension_prevalence']
prev_depression_all = output['tlo.methods.ncds']['depression_prevalence']
prev_clbp_all = output['tlo.methods.ncds']['chronic_lower_back_pain_prevalence']
prev_ckd_all = output['tlo.methods.ncds']['chronic_kidney_disease_prevalence']
prev_cihd_all = output['tlo.methods.ncds']['cihd_prevalence']

mm_prevalence = output['tlo.methods.ncds']['mm_prevalence']

# plot prevalence of conditions by age and sex

prev_diabetes.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Diabetes')
plt.title('Prevalence of Diabetes by Age')
plt.savefig(outputpath / 'prevalence_diabetes_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

prev_hypertension.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Hypertension')
plt.title('Prevalence of Hypertension by Age')
plt.savefig(outputpath / 'prevalence_hypertension_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

prev_depression.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Depression')
plt.title('Prevalence of Depression by Age')
plt.savefig(outputpath / 'prevalence_depression_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

prev_cihd.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Chronic Lower Back Pain')
plt.title('Prevalence of Chronic Lower Back Pain by Age')
plt.savefig(outputpath / 'prevalence_clbp_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

prev_cihd.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Chronic Kidney Disease')
plt.title('Prevalence of Chronic Kidney Disease by Age')
plt.savefig(outputpath / 'prevalence_ckd_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

prev_cihd.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Chronic Ischemic Heart Disease')
plt.title('Prevalence of Chronic Ischemic Heart Disease by Age')
plt.savefig(outputpath / 'prevalence_cihd_by_age.pdf')
#plt.ylim([0, 1])
plt.show()

# plot prevalence among all adults for each condition

prev_diabetes_all['year'] = pd.to_datetime(prev_diabetes_all['date']).dt.year
prev_diabetes_all.drop(columns='date', inplace=True)
prev_diabetes_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_diabetes_all)
plt.ylabel('Prevalence of Diabetes Over Time')
plt.xlabel('Year')
plt.show()

prev_hypertension_all['year'] = pd.to_datetime(prev_hypertension_all['date']).dt.year
prev_hypertension_all.drop(columns='date', inplace=True)
prev_hypertension_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_hypertension_all)
plt.ylabel('Prevalence of Hypertension Over Time')
plt.xlabel('Year')
plt.show()

prev_depression_all['year'] = pd.to_datetime(prev_depression_all['date']).dt.year
prev_depression_all.drop(columns='date', inplace=True)
prev_depression_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_depression_all)
plt.ylabel('Prevalence of Depression Over Time')
plt.xlabel('Year')
plt.show()

prev_clbp_all['year'] = pd.to_datetime(prev_clbp_all['date']).dt.year
prev_clbp_all.drop(columns='date', inplace=True)
prev_clbp_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_clbp_all)
plt.ylabel('Prevalence of Chronic Lower Back Pain Over Time')
plt.xlabel('Year')
plt.show()

prev_ckd_all['year'] = pd.to_datetime(prev_ckd_all['date']).dt.year
prev_ckd_all.drop(columns='date', inplace=True)
prev_ckd_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_ckd_all)
plt.ylabel('Prevalence of CKD Over Time')
plt.xlabel('Year')
plt.show()

prev_cihd_all['year'] = pd.to_datetime(prev_cihd_all['date']).dt.year
prev_cihd_all.drop(columns='date', inplace=True)
prev_cihd_all.set_index('year', drop=True, inplace=True)

plt.plot(prev_cihd_all)
plt.ylabel('Prevalence of CIHD Over Time')
plt.xlabel('Year')
plt.show()

prev_diabetes_all['hypertension'] = prev_hypertension_all['prevalence']
prev_diabetes_all['depression'] = prev_depression_all['prevalence']
prev_diabetes_all['chronic lower back pain'] = prev_clbp_all['prevalence']
prev_diabetes_all['chronic kidney disease'] = prev_ckd_all['prevalence']
prev_diabetes_all['chronic ischemic heart disease'] = prev_cihd_all['prevalence']
prev_diabetes_all.rename(columns={'prevalence': 'diabetes'}, inplace=True)

prev_diabetes_all.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion of Adults 15+ with Condition')
plt.title('Prevalence of Adults with Conditions')
plt.savefig(outputpath / 'prevalence_conditions_adults.pdf')
#plt.ylim([0, 1])
plt.show()

# Plot prevalence of multi-morbidities (no salt vs. high salt):
plotdata = pd.DataFrame({
    "0 co-morbidities":[mm_prevalence["mm_prev_0"].iloc[-1]],
    "1 co-morbidity":[mm_prevalence["mm_prev_1"].iloc[-1]],
    "2 co-morbidities":[mm_prevalence["mm_prev_2"].iloc[-1]],
    "3 co-morbidities":[mm_prevalence["mm_prev_3"].iloc[-1]]
    },
    index=[""]
)
plotdata.plot(kind="bar", stacked=True, width=0.2)
plt.title("Prevalence of number of co-morbidities in 2020")
plt.ylabel("Prevalence of number of co-morbidities")
plt.xticks(rotation=0)
plt.legend(loc='center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=2)

plt.show()

# multi-morbidity prevalence

mm_prevalence = output['tlo.methods.ncds']['mm_prevalence']
mm_prevalence['year'] = pd.to_datetime(mm_prevalence['date']).dt.year
mm_prevalence.drop(columns='date', inplace=True)
mm_prevalence.set_index('year', drop=True, inplace=True)
mm_prevalence.rename(columns={'mm_prev_0': '0  co-morbidities', 'mm_prev_1': '1 co-morbidity', 'mm_prev_2':
    '2 co-morbidities', 'mm_prev_3': '3+ co-morbidities'}, inplace=True)

mm_prevalence.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion of Adults 15+ with Co-Morbidities')
plt.title('Prevalence of Adults with Co-Morbidities')
plt.savefig(outputpath / 'prevalence_comorbidities_adults.pdf')
#plt.ylim([0, 1])
plt.show()





