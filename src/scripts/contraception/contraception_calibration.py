"""This file is used to do a quick check on the likely outcomes of a longer run. It simulates only the contraceptive
poll and age-update event in order to construct a *rought* age-time trend of the usage of each contraceptive.
This is combined with the assumption of fertility and failure rates and compared with the WPP estimates of age-specific
fertility rates. Its limitations are that it does not consider the time when a woman is pregnant (and so can't become
pregnant) and the choice of contraceptive following birth.
For a full calibration, run the "real" simulation and compute the number of age-specific births directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import make_calendar_period_lookup
from tlo.methods import contraception, demography, enhanced_lifestyle, healthsystem, symptommanager
from tlo.methods.contraception import get_medium_variant_asfr_from_wpp_resourcefile
from tlo.methods.hiv import DummyHivModule

# %% Create dummy simulation object
resourcefilepath = Path('resources')
start_date = Date(2010, 1, 1)
sim = Simulation(start_date=start_date, seed=0, resourcefilepath=resourcefilepath)

sim.register(
    # - core modules:
    demography.Demography(),
    enhanced_lifestyle.Lifestyle(),
    symptommanager.SymptomManager(),
    healthsystem.HealthSystem(disable=True),

    contraception.Contraception(use_healthsystem=False),
    contraception.SimplifiedPregnancyAndLabour(),

    # - Dummy HIV module (as contraception requires the property hv_inf)
    DummyHivModule()
)
sim.modules['Contraception'].pre_initialise_population()

# %% Preparation

# Shortcuts
states = sim.modules['Contraception'].all_contraception_states
pp = sim.modules['Contraception'].processed_params
prob_live_births = sim.modules['Labour'].parameters['prob_live_birth']

# Helper Functions
AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = sim.modules['Demography'].AGE_RANGE_CATEGORIES, sim.modules[
    'Demography'].AGE_RANGE_LOOKUP
adult_age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

_, period_lookup = make_calendar_period_lookup()

# Load WPP data on live births (age-specific fertility rates)
wpp = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_ASFR_WPP.csv')


def format_usage_results(_df):
    return _df.unstack().T.apply(lambda row: row / row.sum(), axis=1).dropna()


def get_asfr_per_month_implied_by_contraceptive_use(contraceptive_use: pd.DataFrame) -> dict:
    """Compute the age-specific fertility rate per month that is implied by a pattern of contraceptive use, given
    the model parameters."""

    # Number of pregnancies per month per method
    preg_per_month = pd.DataFrame(index=range(15, 50), columns=sorted(states))
    preg_per_month['not_using'] = pp['p_pregnancy_no_contraception_per_month'][
        'hv_inf_False']  # (for simplicity, assume all persons HIV-negative)
    preg_per_month.loc[:, sorted(states - {'not_using'})] = \
        pp['p_pregnancy_with_contraception_per_month'].loc[:, sorted(states - {'not_using'})]
    preg_per_month = preg_per_month.groupby(by=preg_per_month.index.map(AGE_RANGE_LOOKUP)).mean()

    # Total live births per month: sum across risk of pregnancy from all births and multiply by prob of live birth.
    return ((contraceptive_use * preg_per_month).sum(axis=1) * prob_live_births).to_dict()


# %% Compare the induced the age-specific fertility rates with the WPP data

assumption_init_use = pp['initial_method_use'].groupby(by=pp['initial_method_use'].index.map(AGE_RANGE_LOOKUP)).mean()
assert np.isclose(1.0, assumption_init_use.sum(axis=1)).all()

# Get the model-induced age-specific fertility rate, per month
asfr_per_month_init = get_asfr_per_month_implied_by_contraceptive_use(assumption_init_use)

# Get the WPP age-specific fertility rates, adjusted to risk per woman per month.
wpp_fert_per_month_2010 = get_medium_variant_asfr_from_wpp_resourcefile(wpp, months_exposure=1)[2010]

# Plot
plt.plot(asfr_per_month_init.keys(), asfr_per_month_init.values(), 'k', label='Model (Expectation)')
plt.plot(wpp_fert_per_month_2010.keys(), wpp_fert_per_month_2010.values(), 'r-', label='WPP')
plt.title("Age-specific fertility per month in 2010")
plt.xlabel('Age-group')
plt.ylabel('Live-births per month per woman')
plt.legend()
plt.tight_layout()
plt.show()

current_sf = (sim.modules['Contraception']).parameters['scaling_factor_on_monthly_risk_of_pregnancy']
y = np.array(list(wpp_fert_per_month_2010.values())) / (np.array(list(asfr_per_month_init.values())) / current_sf)
print(f'Approximate scaling factor on pregnancy should be (by age_group): {[round(_x, 2) for _x in y]}')
# NB. This is very approximate because it doesn't take into account the time that is actually spent being pregnant (when
#  , therefore, women are not at risk of becoming pregnant again). The best estimate for the scaling factor is created
# by using "real" runs of the simulation.


# %% Define the simulation runner
# NB. This is very approximate because it doesn't take into account the time that is actually spent being pregnant or
# the pattern of resumption of contraception following birth. It can be used as a rough guide only.

def run_sim(end_date=Date(2029, 12, 31)):
    # Run the ContraceptivePoll (with the option `run_do_pregnancy=False` to prevent pregnancies)
    popsize = 5_000  # size of population for simulation
    sim.make_initial_population(n=popsize)
    poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'], run_do_pregnancy=False)
    age_update_event = demography.AgeUpdateEvent(sim.modules['Demography'], sim.modules['Demography'].AGE_RANGE_LOOKUP)

    _usage_by_age = dict()
    sim.date = sim.start_date
    for date in pd.date_range(sim.date, end_date, freq=pd.DateOffset(months=1)):
        sim.date = date
        age_update_event.apply(sim.population)
        _usage_by_age[date] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
            sim.population.props.age_years.between(15, 49))].groupby(by=['co_contraception', 'age_range']).size()
        poll.apply(sim.population)

        # recycle 50-years to become 15-year-olds, and "not_using" (which is assigned on_birth)
        df = sim.population.props
        df.loc[(df.age_exact_years > 50.0), 'date_of_birth'] = sim.date - pd.DateOffset(years=15)
        df.loc[(df.age_exact_years > 50.0), 'co_contraception'] = "not_using"

    return _usage_by_age


# %% Simulate the changes in contraceptive use to get an idea of how age-specific fertility will change over
# time, and so calibrate the temporal changes in initiation/discontinuation

end_date_simulation = Date(2099, 12, 31)

usage_by_age = run_sim(end_date=end_date_simulation)

# %% Inspect Results


def plot(df, title=''):
    spacing = (np.arange(len(df)) % 60) == 0

    fig, ax = plt.subplots()
    df.loc[spacing].apply(lambda row: row / row.sum(), axis=1).plot.bar(stacked=True, ax=ax, legend=False)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Proportion on method')

    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)
    plt.show()


for age_grp in ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']:
    plot(
        pd.DataFrame.from_dict({_date.date(): usage_by_age[_date].unstack()[age_grp] for _date in usage_by_age},
                               orient='index'),
        title=age_grp
    )

# %% Get the approximate implied "age-specific fertility" rates (per month) by this changing pattern of contraceptive
# use.

#  WPP ASFR per month:
wpp_fert_per_month = pd.DataFrame(get_medium_variant_asfr_from_wpp_resourcefile(wpp, months_exposure=1))

# ASFR per month in model:
model = dict()
for _d in usage_by_age:
    model[_d] = get_asfr_per_month_implied_by_contraceptive_use(format_usage_results(usage_by_age[_d]))
model = pd.DataFrame(model)

# Produce plot, normalising to the period 2010-2014
fig, ax = plt.subplots(2, 4)
ax = ax.reshape(-1)
for i, agegrp in enumerate(adult_age_groups):
    # Get average fertility in the model:
    model_this_agegrp = model.loc[agegrp]
    wpp_this_agegrp = wpp_fert_per_month.loc[agegrp]

    # Comparative plot:
    l1 = ax[i].plot(model_this_agegrp.index.year, model_this_agegrp.values, 'r-', label='Model')
    l2 = ax[i].plot(wpp_this_agegrp.index, wpp_this_agegrp.values, 'k-', label='WPP')
    ax[i].plot((2010, 2010), (0, 0.03), 'b--')
    ax[i].set_title(f"{agegrp}")
    ax[i].set_xlim(2005, end_date_simulation.date().year)
    plt.setp(ax[i].get_xticklabels(), rotation=90, ha='right')

# Plot mean of all ages in last panel
l1 = ax[-1].plot(model_this_agegrp.index.year, model.mean(), 'r-', label='Model')
l2 = ax[-1].plot(wpp_this_agegrp.index, wpp_fert_per_month.mean(), 'k-', label='WPP')
ax[-1].plot((2010, 2010), (0, 0.03), 'b--')
ax[-1].set_title("* 15-49 *")
ax[-1].set_xlim(2005, 2030)
plt.setp(ax[-1].get_xticklabels(), rotation=90, ha='right')

fig.legend((l1[0], l2[0]), ('Model', 'WPP'), 'lower right')
fig.tight_layout()
fig.subplots_adjust(right=0.90)
fig.show()

# %% Check that initial usage by age matches the input assumption (initial_method_use)

actual_init_use = format_usage_results(usage_by_age[Date(2010, 1, 1)])
for i, agegrp in enumerate(actual_init_use.index):
    pd.concat({
        'actual': actual_init_use.loc[agegrp],
        'expected': assumption_init_use.loc[agegrp]
    }, axis=1).plot.bar()
    plt.title(f"Contraception use in 2010: {agegrp}")
    plt.tight_layout()
    plt.show()
