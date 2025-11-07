"""This file uses the results of the results of running `impact_of_cons_availability/scenarios.py` to make some summary
 graphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
)

outputspath = Path('./outputs/tbh03@ic.ac.uk')


# %% Gathering basic information

# Find results_folder associated with a given batch_file and get most recent
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

    _, agegrplookup = make_age_grp_lookup()
    _, calperiodlookup = make_calendar_period_lookup()

    _df['Age_Grp'] = _df['age'].map(agegrplookup).astype(make_age_grp_types())
    _df['Period'] = pd.to_datetime(_df['date']).dt.year.map(calperiodlookup).astype(make_calendar_period_type())
    _df = _df.rename(columns={'sex': 'Sex'})

    # breakdown_by_age_sex_period = _df.groupby(['Sex', 'Age_Grp', 'Period'])['person_id'].count()
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
deaths_summarized = deaths_summarized.loc[deaths_summarized.index.isin(('2010-2014', '2015-2019'))]

# %% Creating some plots:

fig, ax = plt.subplots()
for i, _p in enumerate(params.values):
    central_val = deaths_summarized[(i, 'mean')].values / deaths_summarized[(0, 'mean')].values
    lower_val = deaths_summarized[(i, 'lower')].values/deaths_summarized[(0, 'lower')].values
    upper_val = deaths_summarized[(i, 'upper')].values/deaths_summarized[(0, 'upper')].values
    # todo - this form of constructing the intervals on the ratio is not quite right: just an approximation for now!
    #  When we have decided exactly what we want to plot, we should compute the statistic on each draw and then
    #  summmarise the distribution of those statistics.

    ax.plot(
        deaths_summarized.index, central_val,
        label=_p
    )
    ax.fill_between(
        deaths_summarized.index, lower_val, upper_val, alpha=0.5
    )
ax.set_xlabel('Time period')
ax.set_ylabel('Total deaths (Normalised to calibration)')
ax.set_ylim((0, 1.5))
ax.legend(loc='lower left')
fig.tight_layout()
fig.show()
