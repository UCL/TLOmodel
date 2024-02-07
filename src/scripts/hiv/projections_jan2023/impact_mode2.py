""" get DALYs averted for mode 2 compared with baseline

"""

import datetime
from pathlib import Path

import lacroix
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results10 = get_scenario_outputs("mode2_scenario10.py", outputspath)[-1]
results11 = get_scenario_outputs("mode2_scenario11.py", outputspath)[-1]
results12 = get_scenario_outputs("mode2_scenario12.py", outputspath)[-1]


# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2023, 1, 1), Date(2034, 1, 1))


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys10 = extract_results(
    results10,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys10.loc['Column_Total'] = full_dalys10.sum(numeric_only=True, axis=0)

full_dalys11 = extract_results(
    results11,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys11.loc['Column_Total'] = full_dalys11.sum(numeric_only=True, axis=0)

full_dalys12 = extract_results(
    results12,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys12.loc['Column_Total'] = full_dalys12.sum(numeric_only=True, axis=0)

# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden

# select first 2 runs of baseline scenario (originally 5 draws and 5 runs per draw)
f_full_dalys0 = full_dalys0.loc[:, full_dalys0.columns.get_level_values('run').isin([0, 1])]


sc10_sc0 = f_full_dalys0 - full_dalys10
sc10_sc0_median = sc10_sc0.median(axis=1)
sc10_sc0_lower = sc10_sc0.quantile(q=0.025, axis=1)
sc10_sc0_upper = sc10_sc0.quantile(q=0.975, axis=1)

sc11_sc0 = f_full_dalys0 - full_dalys11
sc11_sc0_median = sc11_sc0.median(axis=1)
sc11_sc0_lower = sc11_sc0.quantile(q=0.025, axis=1)
sc11_sc0_upper = sc11_sc0.quantile(q=0.975, axis=1)

sc12_sc0 = f_full_dalys0 - full_dalys12
sc12_sc0_median = sc12_sc0.median(axis=1)
sc12_sc0_lower = sc12_sc0.quantile(q=0.025, axis=1)
sc12_sc0_upper = sc12_sc0.quantile(q=0.975, axis=1)


# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc10_sc0_median.index
daly_averted_table['scenario10_med'] = [int(round(x, -3)) for x in sc10_sc0_median]
daly_averted_table['scenario10_low'] = [int(round(x, -3)) for x in sc10_sc0_lower]
daly_averted_table['scenario10_upp'] = [int(round(x, -3)) for x in sc10_sc0_upper]
daly_averted_table['scenario11_med'] = [int(round(x, -3)) for x in sc11_sc0_median]
daly_averted_table['scenario11_low'] = [int(round(x, -3)) for x in sc11_sc0_lower]
daly_averted_table['scenario11_upp'] = [int(round(x, -3)) for x in sc11_sc0_upper]
daly_averted_table['scenario12_med'] = [int(round(x, -3)) for x in sc12_sc0_median]
daly_averted_table['scenario12_low'] = [int(round(x, -3)) for x in sc12_sc0_lower]
daly_averted_table['scenario12_upp'] = [int(round(x, -3)) for x in sc12_sc0_upper]

daly_averted_table.iloc[:, 1:] = daly_averted_table.iloc[:, 1:] / 1_000_000


daly_averted_table.to_csv(outputspath / "daly_averted_mode2.csv")

