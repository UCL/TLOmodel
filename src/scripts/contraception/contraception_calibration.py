"""This file is used to do a quick check on the likely outcomes of a longer run. It simulates only the contraceptive
poll and age-update event in order to construct an age-time trend of the usage of each contraceptive. This is combined
with the assumption of fertility and failure rates and compared with the WPP estimates of age-specific fertility rates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import make_calendar_period_lookup
from tlo.methods import contraception, demography, enhanced_lifestyle, healthsystem, symptommanager
from tlo.methods.hiv import DummyHivModule

# %% Create dummy simulation object
resourcefilepath = Path('resources')
start_date = Date(2010, 1, 1)
sim = Simulation(start_date=start_date, seed=0)

sim.register(
    # - core modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),

    contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=False),
    contraception.SimplifiedPregnancyAndLabour(),

    # - Dummy HIV module (as contraception requires the property hv_inf)
    DummyHivModule()
)

# %% Over-ride the content of the resourcefile to explore use of alternative parameter sets:

# Zero-out age-effects on initiation and discontinuation
# sim.modules['Contraception'].parameters['Initiation_ByAge']['r_init1_age'] *= 0.0
# sim.modules['Contraception'].parameters['Discontinuation_ByAge']['r_discont_age'] *= 0.0

# todo -- will have to be age-specific changes in initiation and discontinuation?? **

# %% Simulate the changes in contraceptive use 'manually'

# Run the ContraceptivePoll (with the option `run_do_pregnancy`=False to prevent pregnancies)
sim.make_initial_population(n=5000)
states = sim.modules['Contraception'].all_contraception_states
poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'], run_do_pregnancy=False)
age_update_event = demography.AgeUpdateEvent(sim.modules['Demography'], sim.modules['Demography'].AGE_RANGE_LOOKUP)

usage_by_age = dict()

for date in pd.date_range(sim.date, Date(2099, 12, 1), freq=pd.DateOffset(months=1)):
    sim.date = date

    age_update_event.apply(sim.population)

    usage_by_age[date] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
        sim.population.props.age_years.between(15, 49))].groupby(by=['co_contraception', 'age_range']).size()

    poll.apply(sim.population)

    # recycle 50-years to become 15-year-olds
    df = sim.population.props
    df.loc[(df.age_exact_years > 50.0), 'date_of_birth'] = sim.date - pd.DateOffset(years=15)
    df.loc[(df.age_exact_years > 50.0), 'co_contraception'] = "not_using"


# %% Declare useful formatting functions and load WPP data
AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = sim.modules['Demography'].AGE_RANGE_CATEGORIES, sim.modules[
    'Demography'].AGE_RANGE_LOOKUP

_, period_lookup = make_calendar_period_lookup()


def format_usage_results(df):
    return df.unstack().T.apply(lambda row: row / row.sum(), axis=1).dropna()


# %% Describe patterns of contraceptive usage over time

def plot(df, title=''):
    spacing = (np.arange(len(df)) % 24) == 0

    fig, ax = plt.subplots()
    df.loc[spacing].apply(lambda row: row / row.sum(), axis=1).plot.bar(stacked=True, ax=ax, legend=False)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Proportion')

    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)
    plt.show()


for age_grp in ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']:
    plot(pd.DataFrame.from_dict({d: usage_by_age[d].unstack()[age_grp] for d in usage_by_age}, orient='index'), age_grp)

# %% Check that initial usage by age matches the input assumption (initial_method_use)
actual_init_use = format_usage_results(usage_by_age[Date(2010, 1, 1)])

assumption_init_use = sim.modules['Contraception'].processed_params['initial_method_use'].copy()
assumption_init_use.index = assumption_init_use.index.map(AGE_RANGE_LOOKUP)
assumption_init_use = assumption_init_use.groupby(by=assumption_init_use.index).mean()

fig, ax = plt.subplots(2, 4)
ax = ax.reshape(-1)
for i, agegrp in enumerate(actual_init_use.index):
    ax[i].plot(actual_init_use.loc[agegrp].index, actual_init_use.loc[agegrp].values, '--', label='actual')
    ax[i].plot(assumption_init_use.loc[agegrp].index, actual_init_use.loc[agegrp].values, '-', label='assumption')
    ax[i].set_title(f"{agegrp}")
plt.show()

# %% Get the "intrinsic fertility" rates by contraception status assumed in the TLO model
fert = pd.DataFrame(index=range(15, 50), columns=states)
p = sim.modules['Contraception'].processed_params
fert['not_using'] = 1.0 - np.exp(-p['p_pregnancy_no_contraception_per_month']['hv_inf_False'] * 12)
fert.loc[:, fert.columns.drop('not_using')] = 1.0 - np.exp(-p['p_pregnancy_with_contraception_per_month'] * 12)
fert.index = fert.index.map(AGE_RANGE_LOOKUP)
fert = fert.groupby(by=fert.index).mean()

# %% Compare Age-Specific Fertility Rates to 'approx average fertility' of women in the TLO model output in 2010
# (given patterns of contraceptive use and intrinsic fertility and the assumption of the proportion of pregnancy that
# result in live births).

# Load WPP data on live births (age-specific fertility rates)
wpp = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_ASFR_WPP.csv')

# Make assumption about the proportion of pregnancy that lead to live births (in the TLO model)
prop_preg_to_live_birth = 0.67

fig, ax = plt.subplots()
wpp_fert = wpp.loc[
    (wpp.Period == period_lookup[2010]) & (wpp.Variant == 'WPP_Estimates'), ['Age_Grp', 'asfr']
].set_index('Age_Grp')
model_fert = (format_usage_results(usage_by_age[Date(2010, 1, 1)]) * fert).sum(axis=1)
ax.plot(model_fert.index, model_fert.values, 'b-', label='Model Pregnancy')
ax.plot(model_fert.index, model_fert.values * prop_preg_to_live_birth, 'b--', label='Approx Model Live Births')
ax.plot(wpp_fert.index, wpp_fert.values, 'r-', label='WPP Live Births')
ax.set_title(f"{2010}")
ax.set_ylim([0, 0.60])
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# %% Examine trends in fertility over time.

# Compute the 'approx average fertility' of women in the TLO model over time.
av_fert_by_period = dict()
for date in usage_by_age:
    usage = format_usage_results(usage_by_age[date])
    av_fert_by_period[period_lookup[date.year]] = (usage * fert).sum(axis=1)
av_fert_by_period = pd.DataFrame(av_fert_by_period)

# Produce plot, normalising to the period 2010-2014
fig, ax = plt.subplots(2, 4)
ax = ax.reshape(-1)
for i, agegrp in enumerate(av_fert_by_period.index):
    # Get WPP fertility:
    wpp_fert = wpp.loc[
        wpp.Period.isin(av_fert_by_period.columns) & (wpp.Variant.isin(['WPP_Estimates', 'WPP_Medium variant'])) & (
                wpp.Age_Grp == agegrp), ['Period', 'asfr']].set_index('Period')

    # Get average fertility in the model:
    model_fert = av_fert_by_period.loc[agegrp]

    assert (model_fert.index == wpp_fert.index).all()

    # Comparative plot:
    l1 = ax[i].plot(model_fert.index, model_fert.values / model_fert.values[0], 'b-', label='Trend in Model Pregnancy')
    l2 = ax[i].plot(wpp_fert.index, wpp_fert.values / wpp_fert.values[0], 'r-', label='Trend in WPP Live Births')

    ax[i].set_title(f"{agegrp}")
    ax[i].set_ylim([0.0, 1.5])
    plt.setp(ax[i].get_xticklabels(), rotation=90, ha='right')

ax[-1].set_axis_off()
fig.legend((l1[0], l2[0]), ('Trend in Model Pregnancy', 'Trend in WPP Live Births'), 'lower right')
fig.tight_layout()
fig.subplots_adjust(right=0.90)
plt.show()
