"""This file uses the results of the results of running `impact_of_cons_availability/scenarios.py` to make some summary
 graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    create_age_range_lookup,
    make_age_grp_types,
    make_age_grp_lookup,
    make_calendar_period_lookup,
    make_calendar_period_type,
)

outputspath = Path('./outputs/tbh03@ic.ac.uk')


# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('scenarios.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# %% Extracting results from run

def _extract_deaths_by_age_group_and_time_period(_df: pd.DataFrame) -> pd.Series:
    """Construct a series with index age-range/time-period and value of the number of deaths from the `death` dataframe
    logged in `tlo.methods.demography`."""

    _, agegrplookup = make_age_grp_lookup()
    _, calperiodlookup = make_calendar_period_lookup()

    _df['Age_Grp'] = _df['age'].map(agegrplookup).astype(make_age_grp_types())
    _df['Period'] = pd.to_datetime(_df['date']).dt.year.map(calperiodlookup).astype(make_calendar_period_type())
    _df = _df.rename(columns={'sex': 'Sex'})

    breakdown_by_age_sex_period = _df.groupby(['Sex', 'Age_Grp', 'Period'])['person_id'].count()
    breakdown_by_period = _df.groupby(['Period'])['person_id'].count()

    return breakdown_by_period


deaths_extracted = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=_extract_deaths_by_age_group_and_time_period,
    do_scaling=True
)

deaths_summarized = summarize(deaths_extracted, only_mean=True)


# %% Creating some plots:

param_name = 'HealthSystem:cons_availability'  # name of parameter that varies

# i) bar plot to summarize as the value at the end of the run
propinf_end = propinf.iloc[[-1]]

height = propinf_end.loc[:, (slice(None), "mean")].iloc[0].values
lower_upper = np.array(list(zip(
    propinf_end.loc[:, (slice(None), "lower")].iloc[0].values,
    propinf_end.loc[:, (slice(None), "upper")].iloc[0].values
))).transpose()

yerr = abs(lower_upper - height)

xvals = range(info['number_of_draws'])
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]

fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=propinf_end.loc[:, (slice(None), "mean")].iloc[0].values,
    yerr=yerr
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.show()

# ii) plot to show time-series (means)
for draw in range(info['number_of_draws']):
    plt.plot(
        propinf.loc[:, (draw, "mean")].index, propinf.loc[:, (draw, "mean")].values,
        label=f"{param_name}={round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)}"
    )
plt.xlabel(propinf.index.name)
plt.legend()
plt.show()

# iii) banded plot to show variation across runs
draw = 0
plt.plot(propinf.loc[:, (draw, "mean")].index, propinf.loc[:, (draw, "mean")].values, 'b')
plt.fill_between(
    propinf.loc[:, (draw, "mean")].index,
    propinf.loc[:, (draw, "lower")].values,
    propinf.loc[:, (draw, "upper")].values,
    color='b',
    alpha=0.5,
    label=f"{param_name}={round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)}"
)
plt.xlabel(propinf.index.name)
plt.legend()
plt.show()
