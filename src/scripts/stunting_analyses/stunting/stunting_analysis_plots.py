from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from tlo import Date, Simulation
from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize
from tlo.methods import demography, enhanced_lifestyle, healthsystem, simplified_births, stunting

# %% Preparation

scenario_filename = 'stunting_analysis_scenario.py'
outputspath = Path('./outputs/tbh03@ic.ac.uk')
resourcefilepath = './resources'

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731


# %% Define functions

def __process(x):
    """Process log of stunting in a pd.Series with multi-index age / cat/ year"""
    x = x.set_index('date')
    x.index = x.index.year
    age = [int(_.strip("()'").split(',')[0]) for _ in list(x.columns)]
    cat = [_.replace("'", "").strip("( )").split(',')[1].strip() for _ in list(x.columns)]
    x.columns = pd.MultiIndex.from_tuples(list(zip(age, cat)), names=('age', 'cat'))
    return x.stack().stack()


def __get_sim():
    """Return simulation object with Stunting and other necessary modules registered"""
    start_date = Date(2010, 1, 1)

    _sim = Simulation(start_date=start_date, seed=0, resourcefilepath=resourcefilepath)
    _sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(ignore_cons_constraints=True),
        simplified_births.SimplifiedBirths(),
        stunting.Stunting(),
        stunting.StuntingPropertiesOfOtherModules()
    )
    return _sim


# %% Extract results
results = summarize(extract_results(results_folder,
                                    module="tlo.methods.stunting",
                                    key="prevalence",
                                    custom_generate_series=__process))

# Extract the the mean distribution for the baseline case (with HealthSystem working)
mean_dist_draw0 = results.loc[:, (0, "mean")].unstack().unstack().T.reset_index()

# %%  Plot results for model in 2015 relative to calibration data (from 2015)
sim = __get_sim()


def __get_proportions_stunted_by_params(mean, stdev):
    haz_distribution = norm(loc=mean, scale=stdev)
    return haz_distribution.cdf(-2), haz_distribution.cdf(-3)


def __get_proportions_stunted_by_age(_age):
    if _age < 1:
        # get average for 0-5 months and 6-11 months
        return [
            (_a + _b) / 2.0 for _a, _b in zip(
                __get_proportions_stunted_by_params(
                    *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_0_5mo']),
                __get_proportions_stunted_by_params(
                    *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_6_11mo'])
            )
        ]
    elif _age < 2:
        return __get_proportions_stunted_by_params(
            *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_12_23mo'])
    elif _age < 3:
        return __get_proportions_stunted_by_params(
            *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_24_35mo'])
    elif _age < 4:
        return __get_proportions_stunted_by_params(
            *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_36_47mo'])
    elif _age < 5:
        return __get_proportions_stunted_by_params(
            *sim.modules['Stunting'].parameters['prev_HAZ_distribution_age_48_59mo'])
    else:
        return np.nan


years_to_compare_to_data = 2015
cats_in_order = ['HAZ>=-2', '-3<=HAZ<-2', 'HAZ<-3']
fig, ax = plt.subplots()
to_plot = \
    mean_dist_draw0[['age', 'cat', years_to_compare_to_data]]\
    .groupby(['age', 'cat'])[years_to_compare_to_data].sum()\
    .groupby(level=0).apply(lambda x: x / x.sum()).unstack()[cats_in_order]
for _age in range(5):
    # Model
    plt.bar(_age, to_plot.loc[_age]['HAZ>=-2'], color='darkgoldenrod')
    plt.bar(_age, to_plot.loc[_age]['-3<=HAZ<-2'], bottom=to_plot.loc[_age]['HAZ>=-2'],
            color='goldenrod')
    plt.bar(_age, to_plot.loc[_age]['HAZ<-3'], bottom=(
        to_plot.loc[_age]['HAZ>=-2'] + to_plot.loc[_age]['-3<=HAZ<-2']),
            color='gold')

    # Data
    prop_notseverely_stunted, prop_severely_stunted = __get_proportions_stunted_by_age(_age)
    plt.plot(_age, 1.0 - prop_notseverely_stunted, 'kx')
    plt.plot(_age, 1.0 - prop_severely_stunted, 'ko')

# Proxy artists
plt.plot([np.nan], [np.nan], 'ko', label='Data: HAZ<-3')
plt.plot([np.nan], [np.nan], 'kx', label='Data: HAZ>=-2')
plt.bar([np.nan], [np.nan], color='gold', label='Model: HAZ<-3')
plt.bar([np.nan], [np.nan], color='goldenrod', label='Model: -3<=HAZ<-2')
plt.bar([np.nan], [np.nan], color='darkgoldenrod', label='Model: HAZ>=-2')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Proportion')
ax.set_title(f'Year {years_to_compare_to_data}')
ax.legend(loc=3)
plt.tight_layout()
plt.show()

# %% Stacked-bars for selection of specific years
years_to_plot = [2015, 2020, 2025]
cats_in_order = ['HAZ>=-2', '-3<=HAZ<-2', 'HAZ<-3']
fig, axes = plt.subplots(1, len(years_to_plot), sharex=True, sharey=True)
for year, ax in zip(years_to_plot, axes):
    breakdown_this_year = mean_dist_draw0[['age', 'cat', year]].groupby(['age', 'cat'])[year].sum() \
        .groupby(level=0).apply(lambda x: x / x.sum())
    breakdown_this_year.unstack()[cats_in_order].plot.bar(stacked=True, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Proportion')
    ax.set_title(f'Year {year}')
    ax.legend()
plt.tight_layout()
plt.show()

# %% Proportion any stunting over time by age
props = mean_dist_draw0.groupby(
    by=[mean_dist_draw0.age, mean_dist_draw0.cat.isin(['HAZ<-3', '-3<=HAZ<-2'])]).sum().groupby(level=0).apply(
    lambda x: x / x.sum())
fig, ax = plt.subplots()
for _age in range(5):
    ax.plot(props.columns, props.loc[(_age, True)], label=f'Age {_age} years')
plt.xlabel('Year')
plt.legend()
plt.ylabel('Proportion Stunted (HAZ < -2)')
plt.tight_layout()
plt.show()

# %%  Examine the difference in the proportion of children stunted between the two scenarios (with/without the
# HealthSystem operating).
#
# Awaiting fix to https://github.com/UCL/TLOmodel/issues/392
