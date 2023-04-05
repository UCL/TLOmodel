"""This file uses the results of the scenario runs to generate plots

*1 HIV and TB treatment delay

"""

import datetime
from pathlib import Path

import lacroix
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

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]

# colour scheme
berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'


# %%:  ---------------------------------- Treatment delays -------------------------------------

def extract_tx_delay(results_folder: Path,
                     module: str,
                     key: str,
                     column: str = None,
                     ):
    """Utility function to unpack results
    edited version for utils.py
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    # Collect results from each draw/run
    res = dict()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            draw_run = (draw, run)

            try:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                test = df[column]
                test2 = test.apply(pd.to_numeric, errors="coerce")
                res[draw_run] = test2

            except KeyError:
                # Some logs could not be found - probably because this run failed.
                res[draw_run] = None

    return res


# need to collapse all draws/runs together
# set up empty list with columns for each year
# values will be variable length lists of delays
years = list((range(2010, 2035, 1)))


# todo this should be for years post-2022
def summarise_tx_delay(treatment_delay_df):
    """
    extract all treatment delays from all draws/runs
    for each scenario and collapse into lists, with
    one list per year
    """
    list_delays = [[] for i in range(25)]

    # for each row of tb_tx_delay_adult_sc0 0-14 [draws, runs]:
    for i in range(treatment_delay_df.shape[0]):

        # separate each row into its arrays 0-25 [years]
        tmp = treatment_delay_df.loc[i, 1]

        # combine them into a list, with items separated from array
        # e.g. tmp[0] has values for 2010
        for j in range(25):
            tmp2 = tmp[j]

            list_delays[j] = [*list_delays[j], *tmp2]

    return list_delays

#
# # hiv treatment delays
# hiv_tx_delay_adult_sc0_dict = extract_tx_delay(results_folder=results0,
#                                               module="tlo.methods.hiv",
#                                               key="hiv_treatment_delays",
#                                               column="HivTreatmentDelayAdults")
#
# hiv_tx_delay_adult_sc1_dict = extract_tx_delay(results_folder=results1,
#                                               module="tlo.methods.hiv",
#                                               key="hiv_treatment_delays",
#                                               column="HivTreatmentDelayAdults")
#
# hiv_tx_delay_adult_sc2_dict = extract_tx_delay(results_folder=results2,
#                                               module="tlo.methods.hiv",
#                                               key="hiv_treatment_delays",
#                                               column="HivTreatmentDelayAdults")
#
#
# # convert dict to dataframe
# hiv_tx_delay_adult_sc0 = pd.DataFrame(hiv_tx_delay_adult_sc0_dict.items())
# hiv_tx_delay_adult_sc1 = pd.DataFrame(hiv_tx_delay_adult_sc1_dict.items())
# hiv_tx_delay_adult_sc2 = pd.DataFrame(hiv_tx_delay_adult_sc2_dict.items())
#
#
# list_hiv_tx_delay0 = summarise_tx_delay(hiv_tx_delay_adult_sc0)
# list_hiv_tx_delay1 = summarise_tx_delay(hiv_tx_delay_adult_sc1)
# list_hiv_tx_delay2 = summarise_tx_delay(hiv_tx_delay_adult_sc2)
#
# # replace nan with negative number (false positive)
# list_hiv_tx_delay0 = [[-99 if np.isnan(j) else j for j in i] for i in list_hiv_tx_delay0]
# list_hiv_tx_delay1 = [[-99 if np.isnan(j) else j for j in i] for i in list_hiv_tx_delay1]
# list_hiv_tx_delay2 = [[-99 if np.isnan(j) else j for j in i] for i in list_hiv_tx_delay2]
#
# # convert lists to df
# # todo note nans are fillers for dataframe
# hiv_delay0 = pd.DataFrame(list_hiv_tx_delay0).T
# hiv_delay0.columns = years
# # convert wide to long format
# hiv_delay0 = hiv_delay0.reset_index()
# hiv_delay0_scatter = pd.melt(hiv_delay0, id_vars='index', value_vars=years)
# hiv_delay0_scatter['value_years'] = round(hiv_delay0_scatter.value / 365.25)
# hiv_delay0_scatter.loc[hiv_delay0_scatter['value_years'] >= 16, 'value_years'] = 16
# hiv_delay0_scatter = hiv_delay0_scatter[hiv_delay0_scatter['value'].notna()]
#
# hiv_delay1 = pd.DataFrame(list_hiv_tx_delay1).T
# hiv_delay1.columns = years
# # convert wide to long format
# hiv_delay1 = hiv_delay1.reset_index()
# hiv_delay1_scatter = pd.melt(hiv_delay1, id_vars='index', value_vars=years)
# hiv_delay1_scatter['value_years'] = round(hiv_delay1_scatter.value / 365.25)
# hiv_delay1_scatter.loc[hiv_delay1_scatter['value_years'] >= 16, 'value_years'] = 16
# hiv_delay1_scatter = hiv_delay1_scatter[hiv_delay1_scatter['value_years'].notna()]
#
# hiv_delay2 = pd.DataFrame(list_hiv_tx_delay2).T
# hiv_delay2.columns = years
# # convert wide to long format
# hiv_delay2 = hiv_delay2.reset_index()
# hiv_delay2_scatter = pd.melt(hiv_delay2, id_vars='index', value_vars=years)
# hiv_delay2_scatter['value_years'] = round(hiv_delay2_scatter.value / 365.25)
# hiv_delay2_scatter.loc[hiv_delay2_scatter['value_years'] >= 16, 'value_years'] = 16
# hiv_delay2_scatter = hiv_delay2_scatter[hiv_delay2_scatter['value'].notna()]
#
# # scenario delays 2023-2035
# # aggregate values over 10 years
# hiv_delay0_hist = hiv_delay0_scatter.loc[hiv_delay0_scatter['variable'] >= 2023]
# hiv_delay0_hist = hiv_delay0_hist.loc[
#     (hiv_delay0_hist['value_years'] >= 1) & (hiv_delay0_hist['value'] <= 10000)]  # exclude negative values (false +ve)
#
# hiv_delay1_hist = hiv_delay1_scatter.loc[hiv_delay1_scatter['variable'] >= 2023]
# hiv_delay1_hist = hiv_delay1_hist.loc[
#     (hiv_delay1_hist['value_years'] >= 1) & (hiv_delay1_hist['value'] <= 10000)]
#
# hiv_delay2_hist = hiv_delay2_scatter.loc[hiv_delay2_scatter['variable'] >= 2023]
# hiv_delay2_hist = hiv_delay2_hist.loc[
#     (hiv_delay2_hist['value_years'] >= 1) & (hiv_delay2_hist['value'] <= 10000)]
#

# tb treatment delays
tb_tx_delay_adult_sc0_dict = extract_tx_delay(results_folder=results0,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc1_dict = extract_tx_delay(results_folder=results1,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")

tb_tx_delay_adult_sc2_dict = extract_tx_delay(results_folder=results2,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayAdults")


# convert dict to dataframe
tb_tx_delay_adult_sc0 = pd.DataFrame(tb_tx_delay_adult_sc0_dict.items())
tb_tx_delay_adult_sc1 = pd.DataFrame(tb_tx_delay_adult_sc1_dict.items())
tb_tx_delay_adult_sc2 = pd.DataFrame(tb_tx_delay_adult_sc2_dict.items())

list_tx_delay0 = summarise_tx_delay(tb_tx_delay_adult_sc0)
list_tx_delay1 = summarise_tx_delay(tb_tx_delay_adult_sc1)
list_tx_delay2 = summarise_tx_delay(tb_tx_delay_adult_sc2)

# replace nan with negative number (false positive)
list_tx_delay0 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay0]
list_tx_delay1 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay1]
list_tx_delay2 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay2]

# convert lists to df
# todo note nans are fillers for dataframe
delay0 = pd.DataFrame(list_tx_delay0).T
delay0.columns = years
# convert wide to long format
delay0 = delay0.reset_index()
delay0_scatter = pd.melt(delay0, id_vars='index', value_vars=years)
delay0_scatter['value_weeks'] = round(delay0_scatter.value / 7)
delay0_scatter.loc[delay0_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay0_scatter = delay0_scatter[delay0_scatter['value'].notna()]

delay1 = pd.DataFrame(list_tx_delay1).T
delay1.columns = years
# convert wide to long format
delay1 = delay1.reset_index()
delay1_scatter = pd.melt(delay1, id_vars='index', value_vars=years)
delay1_scatter['value_weeks'] = round(delay1_scatter.value / 7)
delay1_scatter.loc[delay1_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay1_scatter = delay1_scatter[delay1_scatter['value'].notna()]

delay2 = pd.DataFrame(list_tx_delay2).T
delay2.columns = years
# convert wide to long format
delay2 = delay2.reset_index()
delay2_scatter = pd.melt(delay2, id_vars='index', value_vars=years)
delay2_scatter['value_weeks'] = round(delay2_scatter.value / 7)
delay2_scatter.loc[delay2_scatter['value_weeks'] >= 10, 'value_weeks'] = 10
delay2_scatter = delay2_scatter[delay2_scatter['value'].notna()]


# scenario 1 delays 2023-2035
# aggregate values over 10 weeks
delay0_hist = delay0_scatter.loc[delay0_scatter['variable'] >= 2023]
delay0_hist = delay0_hist.loc[
    (delay0_hist['value_weeks'] >= 1) & (delay0_hist['value'] <= 1095)]  # exclude negative values (false +ve)

delay1_hist = delay1_scatter.loc[delay1_scatter['variable'] >= 2023]
delay1_hist = delay1_hist.loc[
    (delay1_hist['value_weeks'] >= 1) & (delay1_hist['value'] <= 1095)]

delay2_hist = delay2_scatter.loc[delay2_scatter['variable'] >= 2023]
delay2_hist = delay2_hist.loc[
    (delay2_hist['value_weeks'] >= 1) & (delay2_hist['value'] <= 1095)]

# -------------------------------- plots -------------------------------- #
plt.style.use('ggplot')

colours = [baseline_colour, sc1_colour, sc2_colour]
# bins_hiv = [0, 2, 4, 6, 8, 10, 12, 14, 16]
# labels_hiv = ["", "2", "4", "6", "8",
#               "10", "12", "14", "≥ 16"]

bins = range(1, 12)
labels = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "≥ 10"]

## plot
plt.style.use('ggplot')
fig, ax = plt.subplots(constrained_layout=True)
fig.suptitle('')

ax.hist([list(delay0_hist.value_weeks),
                               list(delay1_hist.value_weeks),
                               list(delay2_hist.value_weeks),
          ],
         bins=bins,
         align='right',
         color=colours,
         density=True)
ax.set_xticks(bins)
ax.set_xticklabels(labels)
ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

ax.set(title='',
        ylabel='Density',
        xLabel="Treatment delay, weeks")
ax.set_ylim([0, 1.0])
ax.set(title='',
       ylabel='Proportion treated')
ax.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])

plt.tick_params(axis="both", which="major", labelsize=10)

fig.savefig(outputspath / "TB_treatment_delays.png")

plt.show()




# ## plot
# plt.style.use('ggplot')
# # fig, ax = plt.subplots()
# fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
#                                              # sharey=True,
#                                              constrained_layout=True,
#                                              figsize=(9, 4))
# fig.suptitle('')
#
# ax1.hist([list(hiv_delay0_hist.value_years),
#                                list(hiv_delay1_hist.value_years),
#                                list(hiv_delay2_hist.value_years),
#           ],
#          bins=bins_hiv,
#          align='right',
#          color=colours,
#          density=True)
#
# ax1.set_xticks(bins_hiv)
# ax1.set_xticklabels(labels_hiv)
# ax1.patch.set_edgecolor('grey')
# ax1.patch.set_linewidth('1')
#
# ax1.set(title='HIV',
#         ylabel='Density',
#         xLabel="Treatment delay, years")
# ax1.set_ylim([0, 1.0])
# plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
#
# ax2.hist([list(delay0_hist.value_weeks),
#                                list(delay1_hist.value_weeks),
#                                list(delay2_hist.value_weeks),
#           ],
#          bins=bins,
#          align='right',
#          color=colours,
#          density=True)
# ax2.set_xticks(bins)
# ax2.set_xticklabels(labels)
# ax2.patch.set_edgecolor('grey')
# ax2.patch.set_linewidth('1')
#
# ax2.set(title='TB',
#         ylabel='',
#         xLabel="Treatment delay, weeks")
# ax2.set_ylim([0, 1.0])
#
# plt.tick_params(axis="both", which="major", labelsize=10)
#
# # fig.savefig(outputspath / "Treatment_delays.png")
#
# plt.show()
#
