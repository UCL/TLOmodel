import datetime
import os
# import matplotlib.pyplot as plt
# import numpy as np
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    symptommanager, healthseekingbehaviour)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000

# Establish the simulation object
sim = Simulation(start_date=start_date)
sim.seed_rngs(0)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(depression.Depression(resourcefilepath=resourcefilepath))

# Establish the logger
custom_levels = {"*": logging.INFO,
                 "tlo.methods.Depression": logging.DEBUG
}
logfile = sim.configure_logging(filename="LogFile", custom_levels=custom_levels)

logging.getLogger('tlo.methods.Depression').setLevel(logging.DEBUG)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results
outputs = parse_log_file(logfile)

# %%  Compute Key Outputs

depr = outputs['tlo.methods.depression']['summary_stats']
depr.date = (pd.to_datetime(depr['date']))

# define the period of interest for averages to be the last 3 years of the simulation
period = (max(depr.date) - pd.DateOffset(years=3)) < depr['date']

result = pd.DataFrame(columns=['Model', 'Data'])

# Overall prevalence of current moderate/severe depression in people aged 15+
# (Note that only severe depressions are modelled)
# TODO; check that

result.loc['Current prevalence of depression, aged 15+', 'Model'] = depr.loc[period, 'prop_ge15_depr'].mean()
result.loc['Current prevalence of depression, aged 15+', 'Data'] = 0.09

result.loc['Current prevalence of depression, aged 15+ males', 'Model'] = depr.loc[period, 'prop_ge15_m_depr'].mean()
result.loc['Current prevalence of depression, aged 15+ males', 'Data'] = 0.06

result.loc['Current prevalence of depression, aged 15+ females', 'Model'] = depr.loc[period, 'prop_ge15_f_depr'].mean()
result.loc['Current prevalence of depression, aged 15+ females', 'Data'] = [0.10, 0.08]


# Ever depression in people age 50:
result.loc['Ever depression, aged 50y', 'Model'] = depr.loc[period, 'prop_age_50_ever_depr'].mean()

# Prevalence of antidepressant use amongst age 15+ year olds ever depressed
result.loc['Proportion of 15+ ever depressed using anti-depressants, aged 15+y', 'Model'] = depr.loc[period, 'prop_antidepr_if_ever_depr'].mean()

# Prevalence of antidepressant use amongst people currently depressed
result.loc['Proportion of 15+ currently depressed using anti-depressants, aged 15+y', 'Model'] = depr.loc[period, 'prop_antidepr_if_curr_depr'].mean()


# Process the event outputs from the model
depr_events = outputs['tlo.methods.depression']['event_counts']
depr_events['year'] = pd.to_datetime(depr_events['date']).dt.year
depr_events = depr_events.groupby(by='year')[['SelfHarmEvents', 'SuicideEvents']].sum()

# Get population sizes for the
def get_15plus_pop_by_year(df):
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    df.drop(columns='date',inplace=True)
    df.set_index('year', drop=True, inplace=True)
    cols_for_15plus = [int(x[0]) >= 15 for x in df.columns.str.strip('+').str.split('-')]
    return df[df.columns[cols_for_15plus]].sum(axis=1)


tot_pop = get_15plus_pop_by_year(outputs['tlo.methods.demography']['age_range_m']) \
    + get_15plus_pop_by_year(outputs['tlo.methods.demography']['age_range_f'])

depr_event_rate = depr_events.div(tot_pop, axis=0)

# Rate of serious non fatal self harm incidents per 100,000 adults age 15+ per year
result.loc['Rate of non-fatal self-harm incidence per 100k persons aged 15+', 'Model'] = 1e5 * depr_event_rate['SelfHarmEvents'].mean()
result.loc['Rate of non-fatal self-harm incidence per 100k persons aged 15+', 'Data'] = 7.7

# Rate of suicide per 100,000 adults age 15+ per year
result.loc['Rate of suicide incidence per 100k persons aged 15+', 'Model'] = 1e5 * depr_event_rate['SuicideEvents'].mean()
result.loc['Rate of suicide incidence per 100k persons aged 15+', 'Data'] = [26.1, 8.0, 3.7]

