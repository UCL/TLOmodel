"""This file uses the results of the scenario runs to generate plots

*1 DALYs averted and HCW time required

"""

import os
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
from tlo import Date

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


# %% ---------------------------------- Fraction HCW time-------------------------------------

# fraction of HCW time
# output difference in fraction HCW time from baseline for each scenario
def summarise_frac_hcws(results_folder):
    capacity0 = extract_results(
        results0,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    tmp = (capacity.subtract(capacity0) / capacity0) * 100

    hcw = pd.DataFrame(index=capacity.index, columns=["median", "lower", "upper"])
    hcw["median"] = tmp.median(axis=1)
    hcw["lower"] = tmp.quantile(q=0.025, axis=1)
    hcw["upper"] = tmp.quantile(q=0.975, axis=1)

    return hcw


hcw1 = summarise_frac_hcws(results1)
hcw2 = summarise_frac_hcws(results2)

# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2023, 1, 1), Date(2036, 1, 1))


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
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


def return_daly_summary2(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    # dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    # dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


dalys0 = return_daly_summary(results0)
dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary2(results2)

dalys0.loc['Column_Total'] = dalys0.sum(numeric_only=True, axis=0)
dalys1.loc['Column_Total'] = dalys1.sum(numeric_only=True, axis=0)
dalys2.loc['Column_Total'] = dalys2.sum(numeric_only=True, axis=0)

# create full table for export
daly_table = pd.DataFrame()
daly_table['scenario0'] = dalys0['median'].astype(str) + \
                          " (" + dalys0['lower'].astype(str) + " - " + \
                          dalys0['upper'].astype(str) + ")"
daly_table['scenario1'] = dalys1['median'].astype(str) + \
                          " (" + dalys1['lower'].astype(str) + " - " + \
                          dalys1['upper'].astype(str) + ")"
daly_table['scenario2'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"

daly_table.to_csv(outputspath / "daly_summary.csv")

# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys0.loc['TB (non-AIDS)'] = full_dalys0.loc['TB (non-AIDS)'] + full_dalys0.loc['non_AIDS_TB']
full_dalys0.drop(['non_AIDS_TB'], inplace=True)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['TB (non-AIDS)'] = full_dalys1.loc['TB (non-AIDS)'] + full_dalys1.loc['non_AIDS_TB']
full_dalys1.drop(['non_AIDS_TB'], inplace=True)
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/t.mangal@imperial.ac.uk/full_dalys.xlsx")
full_dalys0.to_excel(writer, sheet_name='sc0')
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
writer.save()

# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden
sc1_sc0 = full_dalys0.subtract(full_dalys1, fill_value=0)
sc1_sc0_median = sc1_sc0.median(axis=1)
sc1_sc0_lower = sc1_sc0.quantile(q=0.025, axis=1)
sc1_sc0_upper = sc1_sc0.quantile(q=0.975, axis=1)

sc2_sc0 = full_dalys0.subtract(full_dalys2, fill_value=0)
sc2_sc0_median = sc2_sc0.median(axis=1)
sc2_sc0_lower = sc2_sc0.quantile(q=0.025, axis=1)
sc2_sc0_upper = sc2_sc0.quantile(q=0.975, axis=1)

# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario1_med'] = [int(round(x, -3)) for x in sc1_sc0_median]
daly_averted_table['scenario1_low'] = [int(round(x, -3)) for x in sc1_sc0_lower]
daly_averted_table['scenario1_upp'] = [int(round(x, -3)) for x in sc1_sc0_upper]
daly_averted_table['scenario2_med'] = [int(round(x, -3)) for x in sc2_sc0_median]
daly_averted_table['scenario2_low'] = [int(round(x, -3)) for x in sc2_sc0_lower]
daly_averted_table['scenario2_upp'] = [int(round(x, -3)) for x in sc2_sc0_upper]

daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

aids_dalys_diff = [sc1_sc0_median['AIDS'],
                   sc2_sc0_median['AIDS']]
tb_dalys_diff = [sc1_sc0_median['TB (non-AIDS)'],
                 sc2_sc0_median['TB (non-AIDS)']]
total_dalys_diff = [sc1_sc0_median['Column_Total'],
                    sc2_sc0_median['Column_Total']]

# -------------------------- plots ---------------------------- #
# plt.style.use('ggplot')
#
# aids_colour = "#8949ab"
# tb_colour = "#ed7e7a"
# total_colour = "#eede77"
#
# years = list((range(2010, 2036, 1)))
# years_num = pd.Series(years)
#
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
#                                figsize=(14, 6))
# # constrained_layout=True)
# fig.suptitle('')
#
# # HCW time
# # labels = ['Baseline', 'Constrained scale-up', 'Unconstrained scale-up']
# # x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
#
# ax1.bar(years_num[13:26], hcw1["median"].loc[13:25], width, color=sc1_colour)
# ax1.bar(years_num[13:26] + width, hcw2["median"].loc[13:25], width, color=sc2_colour)
#
# ax1.set_ylabel("% difference HCW time", rotation=90, labelpad=15)
# # ax1.set_ylim([-0.5, 1.5])
#
# ax1.yaxis.set_label_position("left")
# ax1.legend(["Constrained scale-up", "Unconstrained scale-up"], frameon=False)
#
# # DALYs
# labels = ['Constrained scale-up', 'Unconstrained scale-up']
# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
#
# rects1 = ax2.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)
# rects2 = ax2.bar(x, tb_dalys_diff, width, label='TB', color=tb_colour)
# rects3 = ax2.bar(x + width, total_dalys_diff, width, label='Total', color=total_colour)
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('DALYs')
# ax2.set_title('')
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.legend(["AIDS", "TB", "Total"], frameon=False)
#
# font = {'family': 'sans-serif',
#         'color': 'black',
#         'weight': 'bold',
#         'size': 11,
#         }
#
# ax1.text(-0.15, 1.05, 'A)', horizontalalignment='center',
#          verticalalignment='center', transform=ax1.transAxes, fontdict=font)
#
# ax2.text(-0.1, 1.05, 'B)', horizontalalignment='center',
#          verticalalignment='center', transform=ax2.transAxes, fontdict=font)
#
# fig.tight_layout()
# fig.savefig(outputspath / "HCW_DALYS.png")
#
# plt.show()


# -------------------------- HCW by cadre -----------------------------------------------
# PMTCT services embedded within ANC care

years_of_simulation = 26


def summarise_treatment_counts(df_list, treatment_id):
    """ summarise the treatment counts across all draws/runs for one results folder
        requires a list of dataframes with all treatments listed with associated counts
    """
    number_runs = len(df_list)
    number_HSI_by_run = pd.DataFrame(index=np.arange(years_of_simulation), columns=np.arange(number_runs))
    column_names = [
        treatment_id + "_median",
        treatment_id + "_lower",
        treatment_id + "_upper"]
    out = pd.DataFrame(columns=column_names)

    for i in range(number_runs):
        if treatment_id in df_list[i].columns:
            number_HSI_by_run.iloc[:, i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:, 0] = number_HSI_by_run.median(axis=1)
    out.iloc[:, 1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:, 2] = number_HSI_by_run.quantile(q=0.975, axis=1)

    return out


def treatment_counts(results_folder, module, key, column):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            # check if anything contained in folder (some runs failed)
            folder = results_folder / str(draw) / str(run)
            p: os.DirEntry
            pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
            if pickles:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

                new = df[['date', column]].copy()
                df_list.append(pd.DataFrame(new[column].to_list()))

    # for column in each df, get median
    # list of treatment IDs
    list_tx_id = list(df_list[0].columns)
    results = pd.DataFrame(index=np.arange(years_of_simulation))

    for treatment_id in list_tx_id:
        tmp = summarise_treatment_counts(df_list, treatment_id)

        # append output to dataframe
        results = results.join(tmp)

    return results


def treatment_counts_full(results_folder, module, key, column, treatment_id):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):

            # check if anything contained in folder (some runs failed)
            folder = results_folder / str(draw) / str(run)
            p: os.DirEntry
            pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
            if pickles:
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

                new = df[['date', column]].copy()
                df_list.append(pd.DataFrame(new[column].to_list()))

    # join all treatment_id outputs from every draw/run
    results = pd.DataFrame(index=np.arange(years_of_simulation))
    for i in range(len(df_list)):
        tmp = df_list[i][treatment_id]
        # append output to dataframe
        results.loc[:, i] = tmp

    return results


tx_id0 = treatment_counts(results_folder=results0,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id1 = treatment_counts(results_folder=results1,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

tx_id2 = treatment_counts(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID")

# extract numbers of hiv and tb appts scaled to full population size
# compare scenario 1 and 2
scaling_factor = 145.39609

# produce lists of relevant columns
tb_appts = [col for col in tx_id1 if col.startswith('Tb')]
hiv_appts = [col for col in tx_id1 if col.startswith('Hiv')]
all_appts = tb_appts + hiv_appts

data0 = tx_id0[tx_id0.columns.intersection(all_appts)]
data1 = tx_id1[tx_id1.columns.intersection(all_appts)]
data2 = tx_id2[tx_id2.columns.intersection(all_appts)]

# row 13 is 2023
# sum all appts from 2023 for each scenario
tmp0 = data0.iloc[13:26]
tmp1 = data1.iloc[13:26]
tmp2 = data2.iloc[13:26]

# total number of hiv/tb appts 2023-2035 - sum only the "_median" columns
median_appts = [col for col in tx_id1 if col.endswith('_median')]
data0_median = tmp1[tmp1.columns.intersection(median_appts)]
data1_median = tmp1[tmp1.columns.intersection(median_appts)]
data2_median = tmp2[tmp2.columns.intersection(median_appts)]


# map appts to footprint

# scale to full population

# extract minutes of clinical, nursing and pharmacy time


