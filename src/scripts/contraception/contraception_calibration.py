import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.hiv import DummyHivModule

# Create simulation object
resourcefilepath = Path('resources')
start_date = Date(2010, 1, 1)
sim = Simulation(start_date=start_date, seed=0)

sim.register(
    # - core modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),

    # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
    contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=False),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

    # - Dummy HIV module (as contraception requires the property hv_inf)
    DummyHivModule()
)
sim.make_initial_population(n=50000)

# %% Examine trend in contraceptive use

# Run the ContraceptivePoll (with the option `run_do_pregnancy`=False to prevent pregnancies)
states = sim.modules['Contraception'].all_contraception_states
poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'], run_do_pregnancy=False)
age_update_event = demography.AgeUpdateEvent(sim.modules['Demography'], sim.modules['Demography'].AGE_RANGE_LOOKUP)
results1549 = pd.DataFrame(index=range(100), columns=states)
results1519 = pd.DataFrame(index=range(100), columns=states)
results3034 = pd.DataFrame(index=range(100), columns=states)
results_by_age = dict()


for i in range(12*20):
    results1549.loc[i] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
        sim.population.props.age_years.between(15, 49)), 'co_contraception'].value_counts()
    results1519.loc[i] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
        sim.population.props.age_years.between(15, 19)), 'co_contraception'].value_counts()
    results3034.loc[i] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
        sim.population.props.age_years.between(30, 34)), 'co_contraception'].value_counts()

    results_by_age[sim.date] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
        sim.population.props.age_years.between(15, 49))].groupby(by=['co_contraception', 'age_range']).size()

    sim.date += pd.DateOffset(months=1)
    age_update_event.apply(sim.population)
    poll.apply(sim.population)


def plot(df, title=''):
    spacing = (np.arange(len(df)) % 24) == 0
    df.loc[spacing].plot.bar(stacked=True)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Number')
    plt.show()

plot(results1549, 'All Ages')
plot(results1519, '15-19')
plot(results3034, '30-34')


# %% Compare average rates of pregnancy with rates of birth in WPP
# todo this!!! and this use this to get correct number of births in 2010 (change fertility levels) and in future years (change rates of contraceptive use)
AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = sim.modules['Demography'].AGE_RANGE_CATEGORIES, sim.modules[
    'Demography'].AGE_RANGE_LOOKUP
p = sim.modules['Contraception'].processed_params
use = p['initial_method_use']

# Get the "intrinsic fertility rates" by contraception status
fert = pd.DataFrame(index=range(15, 50), columns=states)
fert['not_using'] = 1.0 - np.exp(-p['p_pregnancy_no_contraception_per_month'] * 12)
fert.loc[:, fert.columns.drop('not_using')] = 1.0 - np.exp(-p['p_pregnancy_with_contraception_per_month'] * 12)

# Compute average fertility, grouped by age, for specific comparison dates
dates = [
    datetime.date(2010, 1, 1),
    datetime.date(2020, 1, 1),
    datetime.date(2030, 1, 1)
]

for date in dates:

    average_fert = (use * fert).sum(axis=1)
    agegrps = average_fert.index.map(AGE_RANGE_LOOKUP)
    average_fert_5y = average_fert.groupby(by=agegrps).mean().rename_axis('Age_Grp')

    # Compare to WPP ASFR (=live births, so expect to be a bit higher than model due to losses and mortality)
    wpp = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_ASFR_WPP.csv')
    wpp_fert = \
    wpp.loc[(wpp.Period == '2010-2014') & (wpp.Variant == 'WPP_Estimates'), ['Age_Grp', 'asfr']].set_index('Age_Grp')[
        'asfr']

    f = pd.concat({
        'wpp': wpp_fert,
        'model': average_fert_5y},
        axis=1
    )

    ratio = average_fert_5y / wpp_fert
