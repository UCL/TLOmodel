"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
plots created:
4-panel plot HIV and TB incidence and deaths
"""

import datetime
from pathlib import Path
import os

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('tkagg')

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix
import math

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
# todo check file paths to make sure scenario12 not selected
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]


# -----------------------------------------------------------------------------------------
# %% Plots for health system usage
# -----------------------------------------------------------------------------------------


# ---------------------------------- Fraction HCW time-------------------------------------

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


# %%:  ---------------------------------- Appt usage -------------------------------------

years_of_simulation = 24


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


# get full treatment counts for each draw
tb_test_counts0 = treatment_counts_full(results_folder=results0,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Test_Screening")

tb_test_counts1 = treatment_counts_full(results_folder=results1,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Test_Screening")

tb_test_counts2 = treatment_counts_full(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Test_Screening")


# tb treatment
tb_tx_counts0 = treatment_counts_full(results_folder=results0,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Treatment")

tb_tx_counts1 = treatment_counts_full(results_folder=results1,
                                        module="tlo.methods.healthsystem.summary",
                                        key="HSI_Event",
                                        column="TREATMENT_ID",
                                        treatment_id="Tb_Treatment")

tb_tx_counts2 = treatment_counts_full(results_folder=results2,
                          module="tlo.methods.healthsystem.summary",
                          key="HSI_Event",
                          column="TREATMENT_ID",
                          treatment_id="Tb_Treatment")



# select treatments relating to TB and HIV
tx0 = tx_id0[tx_id0.columns[pd.Series(tx_id0.columns).str.startswith(('Hiv', 'Tb'))]]
# drop lower and upper columns - keep only median
tx0 = tx0.loc[:, ~tx0.columns.str.contains('lower')]
tx0 = tx0.loc[:, ~tx0.columns.str.contains('upper')]
tx0 = tx0.T  # transpose for plotting heatmap
tx0 = tx0.fillna(1)  # replce nan with 0
tx0_norm = tx0.divide(tx0.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx0_norm.loc["Tb_Prevention_Ipt_median"] = tx0_norm.loc["Tb_Prevention_Ipt_median"] / tx0_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx0_norm.loc["Hiv_Prevention_Prep_median"] = tx0_norm.loc["Hiv_Prevention_Prep_median"] / tx0_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx1 = tx_id1[tx_id1.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx1 = tx1.loc[:, ~tx1.columns.str.contains('lower')]
tx1 = tx1.loc[:, ~tx1.columns.str.contains('upper')]
tx1 = tx1.T  # transpose for plotting heatmap
tx1 = tx1.fillna(1)  # replce nan with 0
tx1_norm = tx1.divide(tx1.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx1_norm.loc["Tb_Prevention_Ipt_median"] = tx1_norm.loc["Tb_Prevention_Ipt_median"] / tx1_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx1_norm.loc["Hiv_Prevention_Prep_median"] = tx1_norm.loc["Hiv_Prevention_Prep_median"] / tx1_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

tx2 = tx_id2[tx_id2.columns[pd.Series(tx_id2.columns).str.startswith(('Hiv', 'Tb'))]]
tx2 = tx2.loc[:, ~tx2.columns.str.contains('lower')]
tx2 = tx2.loc[:, ~tx2.columns.str.contains('upper')]
tx2 = tx2.T  # transpose for plotting heatmap
tx2 = tx2.fillna(1)  # replce nan with 0
tx2_norm = tx2.divide(tx2.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx2_norm.loc["Tb_Prevention_Ipt_median"] = tx2_norm.loc["Tb_Prevention_Ipt_median"] / tx2_norm.loc[
    "Tb_Prevention_Ipt_median", 4]
tx2_norm.loc["Hiv_Prevention_Prep_median"] = tx2_norm.loc["Hiv_Prevention_Prep_median"] / tx2_norm.loc[
    "Hiv_Prevention_Prep_median", 13]

# rename treatment IDs
# appt_types = ["TB test", "HIV test", "HIV tx", "VMMC", "HIV \n infant prophylaxis", "TB X-ray",
#               "TB tx", "TB follow-up", "TB IPT", "PrEP"]
tx0_norm.index = [name.replace('_median', '').replace('_', ' ') for name in tx0_norm.index]
tx1_norm.index = [name.replace('_median', '').replace('_', ' ') for name in tx1_norm.index]
tx2_norm.index = [name.replace('_median', '').replace('_', ' ') for name in tx2_norm.index]

# insert zeros for IPT and PrEP pre-introduction (actual values are slightly above 0)
tx0_norm.loc['Tb Prevention Ipt', tx0_norm.columns[0:4]] = 0
tx1_norm.loc['Tb Prevention Ipt', tx1_norm.columns[0:4]] = 0
tx2_norm.loc['Tb Prevention Ipt', tx2_norm.columns[0:4]] = 0

tx0_norm.loc['Hiv Prevention Prep', tx0_norm.columns[0:8]] = 0
tx1_norm.loc['Hiv Prevention Prep', tx1_norm.columns[0:8]] = 0
tx2_norm.loc['Hiv Prevention Prep', tx2_norm.columns[0:8]] = 0

years = list((range(2010, 2034, 1)))

tx0_norm.columns = years
tx1_norm.columns = years
tx2_norm.columns = years

# save treatment_ID numbers
with pd.ExcelWriter(outputspath / ("Treatment_numbers_Dec2023" + ".xlsx"), engine='openpyxl') as writer:
    tx0.to_excel(writer, sheet_name='scenario0', index=True)
    tx1.to_excel(writer, sheet_name='scenario1', index=True)
    tx2.to_excel(writer, sheet_name='scenario2', index=True)
    # writer.save()
    writer.close()


# %%:  ---------------------------------- PLOTS ------------------------------------

plt.style.use('ggplot')
cmap = sns.cm.mako_r

width = 0.1
years_num = pd.Series(years)

tx0_norm.columns = years_num
tx1_norm.columns = years_num
tx2_norm.columns = years_num

# Make plot
fig, axs = plt.subplots(ncols=3, nrows=1,
                        # sharex=True,
                        # sharey=True,
                        constrained_layout=True,
                        figsize=(14, 5),
                        gridspec_kw=dict(width_ratios=[4,4,4]))
sns.heatmap(tx0_norm,
                 xticklabels=5,
                 yticklabels=1,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[0]
            )
axs[0].set_title("Baseline", size=10)

sns.heatmap(tx1_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[1]
            )
axs[1].set_title("Constrained scale-up", size=10)

sns.heatmap(tx2_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=True,
            ax=axs[2]
            )
axs[2].set_title("Unconstrained scale-up", size=10)

plt.tick_params(axis="both", which="major", labelsize=9)
fig.savefig(outputspath / "Figure_S9_NumbersAppointments.png")
plt.show()




# frac HCW time

berry = lacroix.colorList('CranRaspberry')
baseline_colour = berry[5]
sc1_colour = berry[3]
sc2_colour = berry[2]


fig, ax = plt.subplots()

ax.bar(years_num[14:24], hcw1["median"].loc[14:23], width, color=sc1_colour)
ax.bar(years_num[14:24] + width, hcw2["median"].loc[14:23], width, color=sc2_colour)

ax.set_ylabel("% difference", rotation=90, labelpad=15)
ax.set_ylim([-10, 60])

ax.yaxis.set_label_position("left")
ax.legend(["Constrained scale-up", "Unconstrained scale-up"], frameon=False)
plt.tight_layout()
# fig.savefig(outputspath / "Frac_HWC_time.png")
plt.show()



# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2024, 1, 1), Date(2034, 1, 1))


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
    # dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    # dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


dalys0 = return_daly_summary(results0)
dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary(results2)


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
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)


with pd.ExcelWriter(outputspath / ("full_dalys_Dec2023" + ".xlsx"), engine='openpyxl') as writer:
    full_dalys0.to_excel(writer, sheet_name='scenario0', index=True)
    full_dalys1.to_excel(writer, sheet_name='scenario1', index=True)
    full_dalys2.to_excel(writer, sheet_name='scenario2', index=True)
    # writer.save()
    writer.close()

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

sc2_sc1 = full_dalys1.subtract(full_dalys2, fill_value=0)
sc2_sc1_median = sc2_sc1.median(axis=1)
sc2_sc1_lower = sc2_sc1.quantile(q=0.025, axis=1)
sc2_sc1_upper = sc2_sc1.quantile(q=0.975, axis=1)

# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario1_med'] = [int(round(x,-3)) for x in sc1_sc0_median]
daly_averted_table['scenario1_low'] = [int(round(x,-3)) for x in sc1_sc0_lower]
daly_averted_table['scenario1_upp'] = [int(round(x,-3)) for x in sc1_sc0_upper]
daly_averted_table['scenario2_med'] = [int(round(x,-3)) for x in sc2_sc0_median]
daly_averted_table['scenario2_low'] = [int(round(x,-3)) for x in sc2_sc0_lower]
daly_averted_table['scenario2_upp'] = [int(round(x,-3)) for x in sc2_sc0_upper]


daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

aids_dalys_diff = [sc1_sc0_median['AIDS'],
              sc2_sc0_median['AIDS']]
tb_dalys_diff = [sc1_sc0_median['TB (non-AIDS)'],
              sc2_sc0_median['TB (non-AIDS)']]
total_dalys_diff = [sc1_sc0_median['Column_Total'],
              sc2_sc0_median['Column_Total']]


aids_dalys = [dalys0.loc['AIDS', 'median'],
              dalys1.loc['AIDS', 'median'],
              dalys2.loc['AIDS', 'median']]
tb_dalys = [dalys0.loc['TB (non-AIDS)', 'median'],
              dalys1.loc['TB (non-AIDS)', 'median'],
              dalys2.loc['TB (non-AIDS)', 'median']]
total_dalys = [dalys0.loc['Column_Total', 'median'],
              dalys1.loc['Column_Total', 'median'],
              dalys2.loc['Column_Total', 'median']]

labels = ['Baseline', 'Constrained scale-up', 'Unconstrained scale-up']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# plots of total DALYs
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, aids_dalys, width, label='AIDS', color=baseline_colour)
rects2 = ax.bar(x, tb_dalys, width, label='TB', color=sc1_colour)
rects3 = ax.bar(x + width, total_dalys, width, label='Total', color=sc2_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DALYs')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig(outputspath / "DALYS.png")

plt.show()



# plots of diff in DALYs from baseline
labels = ['Constrained scale-up', 'Constrained scale-up \n no constraints']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width, aids_dalys_diff, width, label='AIDS', color=baseline_colour)
rects2 = ax.bar(x, tb_dalys_diff, width, label='TB', color=sc1_colour)
rects3 = ax.bar(x + width, total_dalys_diff, width, label='Total', color=sc2_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DALYs')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig(outputspath / "DALYS_averted.png")

plt.show()



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

# need to collapse all draws/runs together
# set up empty list with columns for each year
# values will be variable length lists of delays
years = list((range(2010, 2034, 1)))

def summarise_tx_delay(treatment_delay_df):
    """
    extract all treatment delays from all draws/runs
    for each scenario and collapse into lists, with
    one list per year
    """
    list_delays = [[] for i in range(24)]

    # for each row of tb_tx_delay_adult_sc0 0-14 [draws, runs]:
    for i in range(treatment_delay_df.shape[0]):

        # separate each row into its arrays 0-25 [years]
        tmp = treatment_delay_df.loc[i, 1]

        # combine them into a list, with items separated from array
        # e.g. tmp[0] has values for 2010
        for j in range(23):
            tmp2 = tmp[j]

            list_delays[j] = [*list_delays[j], *tmp2]

    return list_delays


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


## TREATMENT DELAY CHILDREN

tb_tx_delay_child_sc0_dict = extract_tx_delay(results_folder=results0,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

tb_tx_delay_child_sc1_dict = extract_tx_delay(results_folder=results1,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

tb_tx_delay_child_sc2_dict = extract_tx_delay(results_folder=results2,
                                              module="tlo.methods.tb",
                                              key="tb_treatment_delays",
                                              column="tbTreatmentDelayChildren")

# convert dict to dataframe
tb_tx_delay_child_sc0 = pd.DataFrame(tb_tx_delay_child_sc0_dict.items())
tb_tx_delay_child_sc1 = pd.DataFrame(tb_tx_delay_child_sc1_dict.items())
tb_tx_delay_child_sc2 = pd.DataFrame(tb_tx_delay_child_sc2_dict.items())

list_tx_delay0 = summarise_tx_delay(tb_tx_delay_child_sc0)
list_tx_delay1 = summarise_tx_delay(tb_tx_delay_child_sc1)
list_tx_delay2 = summarise_tx_delay(tb_tx_delay_child_sc2)

# replace nan with negative number (false positive)
list_tx_delay0 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay0]
list_tx_delay1 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay1]
list_tx_delay2 = [[-99 if np.isnan(j) else j for j in i] for i in list_tx_delay2]


# convert lists to df
# todo note nans are fillers for dataframe
delay0child = pd.DataFrame(list_tx_delay0).T
delay0child.columns = years
# convert wide to long format
delay0child = delay0child.reset_index()
delay0_scatterchild = pd.melt(delay0child, id_vars='index', value_vars=years)
delay0_scatterchild['value_weeks'] = round(delay0_scatterchild.value / 7)
delay0_scatterchild.loc[delay0_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay0_scatterchild = delay0_scatterchild[delay0_scatterchild['value'].notna()]

delay1child = pd.DataFrame(list_tx_delay1).T
delay1child.columns = years
# convert wide to long format
delay1child = delay1child.reset_index()
delay1_scatterchild = pd.melt(delay1child, id_vars='index', value_vars=years)
delay1_scatterchild['value_weeks'] = round(delay1_scatterchild.value / 7)
delay1_scatterchild.loc[delay1_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay1_scatterchild = delay1_scatterchild[delay1_scatterchild['value'].notna()]

delay2child = pd.DataFrame(list_tx_delay2).T
delay2child.columns = years
# convert wide to long format
delay2child = delay2child.reset_index()
delay2_scatterchild = pd.melt(delay2child, id_vars='index', value_vars=years)
delay2_scatterchild['value_weeks'] = round(delay2_scatterchild.value / 7)
delay2_scatterchild.loc[delay2_scatterchild['value_weeks'] >= 10, 'value_weeks'] = 10
delay2_scatterchild = delay2_scatterchild[delay2_scatterchild['value'].notna()]

# aggregate values over 10 weeks
delay0_histchild = delay0_scatterchild.loc[delay0_scatterchild['variable'] >= 2023]
delay0_histchild = delay0_histchild.loc[
    (delay0_histchild['value_weeks'] >= 1) & (delay0_histchild['value'] <= 1095)]  # exclude negative values (false +ve)

delay1_histchild = delay1_scatterchild.loc[delay1_scatterchild['variable'] >= 2023]
delay1_histchild = delay1_histchild.loc[
    (delay1_histchild['value_weeks'] >= 1) & (delay1_histchild['value'] <= 1095)]

delay2_histchild = delay2_scatterchild.loc[delay2_scatterchild['variable'] >= 2023]
delay2_histchild = delay2_histchild.loc[
    (delay2_histchild['value_weeks'] >= 1) & (delay2_histchild['value'] <= 1095)]


counts, bins, bars = plt.hist(delay0_hist.value_weeks, bins=range(0,11))

colours = [baseline_colour, sc1_colour, sc2_colour]
bins = range(1, 12)
labels = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "≥ 10"]

## plot
plt.style.use('ggplot')
# fig, ax = plt.subplots()
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
                                             sharey=True,
                                             constrained_layout=True,
                                             figsize=(9, 4))
fig.suptitle('')

ax1.hist([list(delay0_hist.value_weeks),
                               list(delay1_hist.value_weeks),
                               list(delay2_hist.value_weeks),
          ],
         bins=bins,
         align='right',
         color=colours,
         density=True)

ax1.set_xticks(bins)
ax1.set_xticklabels(labels)
ax1.patch.set_edgecolor('grey')
ax1.patch.set_linewidth('1')

ax1.set(title='Adults',
        ylabel='Density',
        xlabel="Treatment delay, weeks")
ax1.set_ylim([0, 1.0])


ax2.hist([list(delay0_histchild.value_weeks),
                               list(delay1_histchild.value_weeks),
                               list(delay2_histchild.value_weeks),
          ],
         bins=bins,
         align='right',
         color=colours,
         density=True)

ax2.set_xticks(bins)
ax2.set_xticklabels(labels)
ax2.patch.set_edgecolor('grey')
ax2.patch.set_linewidth('1')

ax2.set(title='Children',
        ylabel='',
        xlabel="Treatment delay, weeks")
ax2.set_ylim([0, 1.0])

plt.tick_params(axis="both", which="major", labelsize=10)

plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
# fig.savefig(outputspath / "Tb_treatment_delay.png")

plt.show()




# %%:  ---------------------------------- PrEP IMPACT ---------------------------------- #

# HIV incidence in AGYW

def hiv_agyw_inc(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="infections_by_2age_groups_and_sex",
        column="n_new_infections_female_1524",
        index="date",
        do_scaling=True
    )

    inc.columns = inc.columns.get_level_values(0)
    inc_summary = pd.DataFrame(index=inc.index, columns=["median", "lower", "upper"])
    inc_summary["median"] = inc.median(axis=1)
    inc_summary["lower"] = inc.quantile(q=0.025, axis=1)
    inc_summary["upper"] = inc.quantile(q=0.975, axis=1)

    return inc_summary


agyw_inc0 = hiv_agyw_inc(results0)
agyw_inc1 = hiv_agyw_inc(results1)
agyw_inc2 = hiv_agyw_inc(results2)


baseline_num_infections = agyw_inc0["median"][11:26]
# multiply by scaling factor to get numbers of expected infections


## plot HIV incidence in AGYW
plt.style.use('ggplot')
fig, ax = plt.subplots()

ax.plot(years_num[12:26], agyw_inc0["median"][11:26], "-", color=baseline_colour)
ax.fill_between(years_num[12:26], agyw_inc0["lower"][11:26], agyw_inc0["upper"][11:26],
                color=baseline_colour, alpha=0.2)

ax.plot(years_num[12:26], agyw_inc1["median"][11:26], "-", color=sc1_colour)
ax.fill_between(years_num[12:26], agyw_inc1["lower"][11:26], agyw_inc1["upper"][11:26],
                color=sc1_colour, alpha=0.2)

ax.plot(years_num[12:26], agyw_inc2["median"][11:26], "-", color=sc2_colour)
ax.fill_between(years_num[12:26], agyw_inc2["lower"][11:26], agyw_inc2["upper"][11:26],
                color=sc2_colour, alpha=0.2)

ax.patch.set_edgecolor('grey')
ax.patch.set_linewidth('1')

plt.ylabel("Number new HIV infections")

plt.xlabel("Year")
plt.ylim((0, 7500))
plt.title("")
plt.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])
fig.savefig(outputspath / "Incidence_HIV_AGYW.png")
plt.show()



# %%: get consumables availability

# get consumables spreadsheet
cons_availability = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_availability_small.csv")
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

# import master facilities list to get facility levels mapped to facility ID
master_fac = pd.read_csv(
    resourcefilepath / "healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# map facility levels to facility ID in consumables spreadsheet
cons_full = pd.merge(cons_availability, master_fac,
                     left_on=["Facility_ID"], right_on=["Facility_ID"], how='left')

# groupby item code & facility level -> average availability by facility level for all items
average_cons_availability = cons_full.groupby(["item_code", "Facility_Level"])["available_prop"].mean().reset_index()

overall_cons_availability = cons_full.groupby(["item_code"])["available_prop"].mean().reset_index()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:

    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])

## TB consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["sputum_test"] = get_item_codes_from_package_name(items_list, "Microscopy Test")
item_codes_dict["xpert_test"] = get_item_codes_from_package_name(items_list, "Xpert test")
item_codes_dict["chest_xray"] = get_item_code_from_item_name(items_list, "X-ray")

# treatment
item_codes_dict["tb_tx_adult"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit A")
item_codes_dict["tb_tx_child"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit B")
item_codes_dict["tb_retx_adult"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A1")
item_codes_dict["tb_retx_child"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A2")
item_codes_dict["tb_mdrtx"] = get_item_code_from_item_name(items_list, "Treatment: second-line drugs")
item_codes_dict["tb_ipt"] = get_item_code_from_item_name(items_list, "Isoniazid/Pyridoxine, tablet 300 mg")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# selected_cons_availability = overall_cons_availability[overall_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 184, "item_code"] = "Sputum test"
selected_cons_availability.loc[selected_cons_availability.item_code == 187, "item_code"] = "GeneXpert test"
selected_cons_availability.loc[selected_cons_availability.item_code == 175, "item_code"] = "Chest X-ray"
selected_cons_availability.loc[selected_cons_availability.item_code == 176, "item_code"] = "Adult treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 178, "item_code"] = "Child treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 177, "item_code"] = "Adult retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 179, "item_code"] = "Child retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 181, "item_code"] = "MDR treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 192, "item_code"] = "IPT"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

fig, ax = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(top=1.5)  # Adjust the top margin to leave space for the title

ax = sns.heatmap(df_heatmap, annot=True,
                 vmin=0, vmax=1.0)

plt.xlabel('Facility level')
plt.ylabel('')
plt.title('TB consumables')
plt.tight_layout()

plt.savefig(outputspath / "TBcons_availability.png", bbox_inches='tight')
plt.show()


## HIV consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["HIV test"] = get_item_code_from_item_name(items_list, "Test, HIV EIA Elisa")
item_codes_dict["Viral load"] = get_item_codes_from_package_name(items_list, "Viral Load")
item_codes_dict["VMMC"] = get_item_code_from_item_name(items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(items_list, "Nevirapine, oral solution, 10 mg/ml")
item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(items_list, "First-line ART regimen: adult")
item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(items_list, "First line ART regimen: older child")
item_codes_dict['First line ART regimen: older child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(items_list, "First line ART regimen: young child")
item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 196, "item_code"] = "HIV test"
selected_cons_availability.loc[selected_cons_availability.item_code == 190, "item_code"] = "Viral load"
selected_cons_availability.loc[selected_cons_availability.item_code == 197, "item_code"] = "VMMC"
selected_cons_availability.loc[selected_cons_availability.item_code == 1191, "item_code"] = "Adult PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 198, "item_code"] = "Infant PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 2671, "item_code"] = "Adult ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 204, "item_code"] = "Adult cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2672, "item_code"] = "Child ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Child cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2673, "item_code"] = "Infant ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 202, "item_code"] = "Infant cotrimoxazole"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

fig, ax = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(top=1.5)  # Adjust the top margin to leave space for the title

ax = sns.heatmap(df_heatmap, annot=True,
                 vmin=0, vmax=1.0)

plt.xlabel('Facility level')
plt.ylabel('')
plt.title('HIV consumables')
plt.tight_layout()

plt.savefig(outputspath / "cons_availability_HIV.png", bbox_inches='tight')
plt.show()



# %%: get consumables availability - one month plots

# get consumables spreadsheet
cons_availability = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_availability_small.csv")
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

# import master facilities list to get facility levels mapped to facility ID
master_fac = pd.read_csv(
    resourcefilepath / "healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# map facility levels to facility ID in consumables spreadsheet
cons_full = pd.merge(cons_availability, master_fac,
                     left_on=["Facility_ID"], right_on=["Facility_ID"], how='left')


# select month for representation - Dec (use jan for hiv?)
cons_dec = cons_full.loc[cons_full.month == 12]

# groupby item code & facility level -> average availability by facility level for all items
average_cons_availability = cons_full.groupby(["item_code", "Facility_Level"])["available_prop"].mean().reset_index()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:

    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])

## TB consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["sputum_test"] = get_item_codes_from_package_name(items_list, "Microscopy Test")
item_codes_dict["xpert_test"] = get_item_codes_from_package_name(items_list, "Xpert test")
item_codes_dict["chest_xray"] = get_item_code_from_item_name(items_list, "X-ray")

# treatment
item_codes_dict["tb_tx_adult"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit A")
item_codes_dict["tb_tx_child"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit B")
item_codes_dict["tb_retx_adult"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A1")
item_codes_dict["tb_retx_child"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A2")
item_codes_dict["tb_mdrtx"] = get_item_code_from_item_name(items_list, "Treatment: second-line drugs")
item_codes_dict["tb_ipt"] = get_item_code_from_item_name(items_list, "Isoniazid/Pyridoxine, tablet 300 mg")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 184, "item_code"] = "Sputum test"
selected_cons_availability.loc[selected_cons_availability.item_code == 187, "item_code"] = "GeneXpert test"
selected_cons_availability.loc[selected_cons_availability.item_code == 175, "item_code"] = "Chest X-ray"
selected_cons_availability.loc[selected_cons_availability.item_code == 176, "item_code"] = "Adult treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 178, "item_code"] = "Child treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 177, "item_code"] = "Adult retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 179, "item_code"] = "Child retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 181, "item_code"] = "MDR treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 192, "item_code"] = "IPT"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('Facility level')
plt.ylabel('')
# plt.savefig(outputspath / "tb_cons_availability_dec.png", bbox_inches='tight')
plt.show()


## HIV consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["HIV test"] = get_item_code_from_item_name(items_list, "Test, HIV EIA Elisa")
item_codes_dict["Viral load"] = get_item_codes_from_package_name(items_list, "Viral Load")
item_codes_dict["VMMC"] = get_item_code_from_item_name(items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(items_list, "Nevirapine, oral solution, 10 mg/ml")
item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(items_list, "First-line ART regimen: adult")
item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(items_list, "First line ART regimen: older child")
item_codes_dict['First line ART regimen: older child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(items_list, "First line ART regimen: young child")
item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 196, "item_code"] = "HIV test"
selected_cons_availability.loc[selected_cons_availability.item_code == 190, "item_code"] = "Viral load"
selected_cons_availability.loc[selected_cons_availability.item_code == 197, "item_code"] = "VMMC"
selected_cons_availability.loc[selected_cons_availability.item_code == 1191, "item_code"] = "Adult PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 198, "item_code"] = "Infant PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 2671, "item_code"] = "Adult ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 204, "item_code"] = "Adult cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2672, "item_code"] = "Child ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Child cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2673, "item_code"] = "Infant ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 202, "item_code"] = "Infant cotrimoxazole"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('')
plt.ylabel('')
plt.savefig(outputspath / "hiv_cons_availability_dec.png", bbox_inches='tight')
plt.show()
