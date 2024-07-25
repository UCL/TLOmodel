
import datetime
from pathlib import Path

# import lacroix
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

from tlo import Date

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
# all scenarios are scale-up
# perfect cons
perfect = get_scenario_outputs("scenario2.py", outputspath)[-1]
# hiv tx cons returned to default
hiv_tx = get_scenario_outputs("scenario3.py", outputspath)[-1]
# hiv prevention cons returned to default
hiv_pre = get_scenario_outputs("scenario4.py", outputspath)[-1]
tb_tx = get_scenario_outputs("scenario5.py", outputspath)[-1]
tb_pre = get_scenario_outputs("scenario6.py", outputspath)[-1]

# get deaths - all-cause - for each scenario
TARGET_PERIOD = (Date(2023, 1, 1), Date(2034, 1, 1))


# extract total deaths
def extract_total_deaths(results_folder, do_scaling=True):
    """ sum all deaths occurring for each run of each draw
    dataframe returned: row=total deaths, column=run/draw
    """

    def get_num_deaths_by_cause(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    tmp = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_num_deaths_by_cause,
        do_scaling=do_scaling
    )

    # add total row
    tmp.loc["Total"] = tmp.sum(axis=0)

    return tmp

deaths_perfect = extract_total_deaths(perfect)
median_perfect = deaths_perfect.iloc[-1].median()
lower_perfect = deaths_perfect.iloc[-1].quantile(q=0.025)
upper_perfect = deaths_perfect.iloc[-1].quantile(q=0.975)

median_perfect_aids = deaths_perfect.loc['AIDS'].median()
lower_perfect_aids = deaths_perfect.loc['AIDS'].quantile(q=0.025)
upper_perfect_aids = deaths_perfect.loc['AIDS'].quantile(q=0.975)

median_perfect_tb = deaths_perfect.loc['TB (non-AIDS)'].median()
lower_perfect_tb = deaths_perfect.loc['TB (non-AIDS)'].quantile(q=0.025)
upper_perfect_tb = deaths_perfect.loc['TB (non-AIDS)'].quantile(q=0.975)



deaths_hivTX = extract_total_deaths(hiv_tx)
median_hivTX = deaths_hivTX.loc['AIDS'].median()
lower_hivTX = deaths_hivTX.loc['AIDS'].quantile(q=0.025)
upper_hivTX = deaths_hivTX.loc['AIDS'].quantile(q=0.975)

median_hivTX_tb = deaths_hivTX.loc['TB (non-AIDS)'].median()
lower_hivTX_tb = deaths_hivTX.loc['TB (non-AIDS)'].quantile(q=0.025)
upper_hivTX_tb = deaths_hivTX.loc['TB (non-AIDS)'].quantile(q=0.975)


deaths_hivPR = extract_total_deaths(hiv_pre)
median_hivPR = deaths_hivPR.loc['AIDS'].median()
lower_hivPR = deaths_hivPR.loc['AIDS'].quantile(q=0.025)
upper_hivPR = deaths_hivPR.loc['AIDS'].quantile(q=0.975)

print(deaths_hivPR.loc['TB (non-AIDS)'].median())
print(deaths_hivPR.loc['TB (non-AIDS)'].quantile(q=0.025))
print(deaths_hivPR.loc['TB (non-AIDS)'].quantile(q=0.975))



deaths_tbTX = extract_total_deaths(tb_tx)
median_tbTX = deaths_tbTX.loc['TB (non-AIDS)'].median()
lower_tbTX = deaths_tbTX.loc['TB (non-AIDS)'].quantile(q=0.025)
upper_tbTX = deaths_tbTX.loc['TB (non-AIDS)'].quantile(q=0.975)

print(deaths_tbTX.loc['AIDS'].median())
print(deaths_tbTX.loc['AIDS'].quantile(q=0.025))
print(deaths_tbTX.loc['AIDS'].quantile(q=0.975))



deaths_tbPR = extract_total_deaths(tb_pre)
median_tbPR = deaths_tbPR.loc['TB (non-AIDS)'].median()
lower_tbPR = deaths_tbPR.loc['TB (non-AIDS)'].quantile(q=0.025)
upper_tbPR = deaths_tbPR.loc['TB (non-AIDS)'].quantile(q=0.975)

print(deaths_tbPR.loc['AIDS'].median())
print(deaths_tbPR.loc['AIDS'].quantile(q=0.025))
print(deaths_tbPR.loc['AIDS'].quantile(q=0.975))


# comparison of TOTAL deaths, run by run
# there are 10 sub-scenarios compared with 25 in main analysis (perfect)
d1 = deaths_hivTX.iloc[-1].values - deaths_perfect.iloc[-1, 0:10].values
d2 = deaths_hivPR.iloc[-1].values - deaths_perfect.iloc[-1, 0:10].values
d3 = deaths_tbTX.iloc[-1].values - deaths_perfect.iloc[-1, 0:10].values
d4 = deaths_tbPR.iloc[-1].values - deaths_perfect.iloc[-1, 0:10].values

print(np.quantile(d1, 0.5))  # deaths averted through strengthening HIV tx
print(np.quantile(d1, 0.25))
print(np.quantile(d1, 0.75))

np.quantile(d2, 0.5)
np.quantile(d3, 0.5)
np.quantile(d4, 0.5)

# use AIDS deaths or TB deaths only
d1 = deaths_hivTX.loc['AIDS'].values - deaths_perfect.loc['AIDS'][0:10].values
d2 = deaths_hivPR.loc['AIDS'].values - deaths_perfect.loc['AIDS'][0:10].values
d3 = deaths_tbTX.loc['TB (non-AIDS)'].values - deaths_perfect.loc['TB (non-AIDS)'][6:16].values
d4 = deaths_tbPR.loc['TB (non-AIDS)'].values - deaths_perfect.loc['TB (non-AIDS)'][0:10].values


print(np.quantile(d1, 0.5))  # deaths averted through strengthening HIV tx
print(np.quantile(d1, 0.025))
print(np.quantile(d1, 0.975))


print(np.quantile(d3, 0.5))  # deaths averted through strengthening TB PR
print(np.quantile(d3, 0.025))
print(np.quantile(d3, 0.975))

# get DALYs - all-cause - for each scenario
def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


def return_daly_summary(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )

    dalys['median'] = dalys.median(axis=1)
    dalys['lower'] = dalys.quantile(q=0.025, axis=1)
    dalys['upper'] = dalys.quantile(q=0.975, axis=1)

    return dalys


dalys_perfect = return_daly_summary(perfect)
dalys_hivTX = return_daly_summary(hiv_tx)
dalys_hivPR = return_daly_summary(hiv_pre)
dalys_tbTX = return_daly_summary(tb_tx)
dalys_tbPR = return_daly_summary(tb_pre)

rounded_perfect = dalys_perfect.applymap(lambda x: np.round(x, -2))
rounded_hivTX = dalys_hivTX.applymap(lambda x: np.round(x, -2))
rounded_hivPR = dalys_hivPR.applymap(lambda x: np.round(x, -2))
rounded_tbTX = dalys_tbTX.applymap(lambda x: np.round(x, -2))
rounded_tbPR = dalys_tbPR.applymap(lambda x: np.round(x, -2))

