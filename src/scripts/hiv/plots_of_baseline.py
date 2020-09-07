"""Create a large population to simulate and compare HIV prevalence at the start of the simulation with calibrating
data"""
import datetime
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

import matplotlib.pyplot as plt
from tlo.methods.hiv import unpack_raw_output_dict, map_to_age_group

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
popsize = 100000

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=0)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )

# Initialise the population:
sim.make_initial_population(n=popsize)

# Compute Prevalence and output dict that is usually produced by the log
df = sim.population.props
log_prev_by_age_and_sex = {}
for sex in ['F', 'M']:
    n_hiv = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].sum()
    n_pop = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].count()
    log_prev_by_age_and_sex[sex] = (n_hiv / n_pop).to_dict()



# %% Create plots, using the same processing approach as when the model prevalence comes from the log

# Get model outputs of HIV prevalence by age and sex in the year 2010
prev_by_age_and_sex = pd.DataFrame()
for sex in ['F', 'M']:
    df_ = unpack_raw_output_dict(log_prev_by_age_and_sex[sex])
    df_['sex'] = sex
    prev_by_age_and_sex = pd.concat([prev_by_age_and_sex, df_])
prev_by_age_and_sex.rename(columns={'value': 'prev_model'}, inplace=True)

# Load and merge-in the data prevalence for the year 2010
data = pd.read_csv(resourcefilepath / "ResourceFile_HIV_prevalence.csv")
data2010 = data.loc[data.year == 2010].copy()
data2010['age_group'] = map_to_age_group(data['age_from'])
data2010 = pd.DataFrame(data2010.groupby(by=['sex', 'age_group'])['prev ', 'pop_size'].sum()).reset_index()
data2010['prev_data'] = data2010['prev '] / data2010['pop_size']
prev_by_age_and_sex = prev_by_age_and_sex.merge(data2010[['sex', 'age_group', 'prev_data']], left_on=['sex', 'age_group'], right_on=['sex', 'age_group'])

# Create multi-index (using groupby) for sex/age and plot:
prev_by_age_and_sex = prev_by_age_and_sex.groupby(by=['sex', 'age_group'])['prev_data', 'prev_model'].sum()
prev_by_age_and_sex.plot.bar()
plt.title('HIV Prevalence in 2010')
plt.savefig(outputpath / ("HIV_prevalence_in_2010" + datestamp + ".pdf"), format='pdf')
plt.show()


