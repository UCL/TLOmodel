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

def runsim(seed=0):
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging

    start_date = Date(2010, 1, 1)
    end_date = Date(2020, 1, 2)
    popsize = 5000

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

# Join-up values from each log to create a trace for an individual

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
prev_ldl_hdl = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['ldl_hdl_prevalence_by_age_and_sex']
    )
)

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

prev_cihd = restore_multi_index(
    transform_output(
        output['tlo.methods.ncds']['cihd_prevalence_by_age_and_sex']
    )
)

prev_ldl_hdl.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With LDL/HDL')
plt.title('Prevalence of LDL/HDL by Age')
plt.savefig(outputpath / 'prevalence_ldl_hdl_by_age.pdf')
plt.show()

prev_diabetes.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Diabetes')
plt.title('Prevalence of Diabetes by Age')
plt.savefig(outputpath / 'prevalence_diabetes_by_age.pdf')
plt.show()

prev_hypertension.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Hypertension')
plt.title('Prevalence of Hypertension by Age')
plt.savefig(outputpath / 'prevalence_hypertension_by_age.pdf')
plt.show()

prev_depression.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Depression')
plt.title('Prevalence of Depression by Age')
plt.savefig(outputpath / 'prevalence_depression_by_age.pdf')
plt.show()

prev_cihd.iloc[-1].transpose().plot.bar()
plt.ylabel('Proportion With Chronic Ischemic Heart Disease')
plt.title('Prevalence of Chronic Ischemic Heart Disease by Age')
plt.savefig(outputpath / 'prevalence_cihd_by_age.pdf')
plt.show()
