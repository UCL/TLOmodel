"""This file uses the results of the results of running `impact_of_cons_availability/scenarios.py` to make some summary
 graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.calibration_analyses.analysis_scripts.analysis_demography_calibrations import agegrplookup, calperiodlookup
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize, make_age_grp_types, make_calendar_period_type,
)

outputspath = Path('./outputs')


# %% Gathering basic information

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('impact_of_consumables_availability.py', outputspath)[-1]

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
    # todo add do grouping by time period too
    _df = _df.reset_index()
    _df['Age_Grp'] = _df['age'].map(agegrplookup).astype(make_age_grp_types())
    _df['Period'] = _df['year'].map(calperiodlookup).astype(make_calendar_period_type())
    _df = _df.rename(columns={'sex': 'Sex'})
    _df = _df.drop(columns=['age', 'year']).groupby(['Period', 'Sex', 'Age_Grp']).sum()
    return _df.assign(year=_df['date'].dt.year).groupby(['sex', 'year', 'age'])['person_id'].count()


deaths_extracted = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=_extract_deaths_by_age_group_and_time_period,
    do_scaling=True
)

deaths_summarized = summarize(deaths_extracted)


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
