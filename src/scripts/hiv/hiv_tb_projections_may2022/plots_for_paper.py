"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
plots created:
4-panel plot HIV and TB incidence and deaths

"""

import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

# ---------------------------------- Fraction HCW time-------------------------------------

# fraction of HCW time
# output fraction of time by year
def summarise_frac_hcws(results_folder):
    capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    capacity.columns = capacity.columns.get_level_values(0)
    hcw = pd.DataFrame(index=capacity.index, columns=["median", "lower", "upper"])
    hcw["median"] = capacity.median(axis=1)
    hcw["lower"] = capacity.quantile(q=0.025, axis=1)
    hcw["upper"] = capacity.quantile(q=0.975, axis=1)

    return hcw


hcw0 = summarise_frac_hcws(results0)
hcw1 = summarise_frac_hcws(results1)
hcw2 = summarise_frac_hcws(results2)
hcw3 = summarise_frac_hcws(results3)
hcw4 = summarise_frac_hcws(results4)

# percentage change in HCW time used from baseline (scenario 0)
hcw_diff1 = ((hcw1["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff2 = ((hcw2["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff3 = ((hcw3["median"] - hcw0["median"]) / hcw0["median"]) * 100
hcw_diff4 = ((hcw4["median"] - hcw0["median"]) / hcw0["median"]) * 100


# ---------------------------------- Appt usage -------------------------------------

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
            number_HSI_by_run.iloc[:,i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:,0] = number_HSI_by_run.median(axis=1)
    out.iloc[:,1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:,2] = number_HSI_by_run.quantile(q=0.975, axis=1)

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

tx_id3 = treatment_counts(results_folder=results3,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

tx_id4 = treatment_counts(results_folder=results4,
               module="tlo.methods.healthsystem.summary",
               key="HSI_Event",
               column="TREATMENT_ID")

# select treatments relating to TB and HIV
tx0 = tx_id0[tx_id0.columns[pd.Series(tx_id0.columns).str.startswith(('Hiv', 'Tb'))]]
# drop lower and upper columns - keep only median
tx0 = tx0.loc[:,~tx0.columns.str.contains('lower')]
tx0 = tx0.loc[:,~tx0.columns.str.contains('upper')]
tx0 = tx0.T  # transpose for plotting heatmap
tx0 = tx0.fillna(1)  # replce nan with 0
tx0_norm = tx0.divide(tx0.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx0_norm.loc["Tb_Prevention_Ipt_median"] = tx0_norm.loc["Tb_Prevention_Ipt_median"] / tx0_norm.loc["Tb_Prevention_Ipt_median", 4]
tx0_norm.loc["Hiv_Prevention_Prep_median"] = tx0_norm.loc["Hiv_Prevention_Prep_median"] / tx0_norm.loc["Hiv_Prevention_Prep_median", 13]

tx1 = tx_id1[tx_id1.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx1 = tx1.loc[:,~tx1.columns.str.contains('lower')]
tx1 = tx1.loc[:,~tx1.columns.str.contains('upper')]
tx1 = tx1.T  # transpose for plotting heatmap
tx1 = tx1.fillna(1)  # replce nan with 0
tx1_norm = tx1.divide(tx1.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx1_norm.loc["Tb_Prevention_Ipt_median"] = tx1_norm.loc["Tb_Prevention_Ipt_median"] / tx1_norm.loc["Tb_Prevention_Ipt_median", 4]
tx1_norm.loc["Hiv_Prevention_Prep_median"] = tx1_norm.loc["Hiv_Prevention_Prep_median"] / tx1_norm.loc["Hiv_Prevention_Prep_median", 13]

tx2 = tx_id2[tx_id2.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx2 = tx2.loc[:,~tx2.columns.str.contains('lower')]
tx2 = tx2.loc[:,~tx2.columns.str.contains('upper')]
tx2 = tx2.T  # transpose for plotting heatmap
tx2 = tx2.fillna(1)  # replce nan with 0
tx2_norm = tx2.divide(tx2.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx2_norm.loc["Tb_Prevention_Ipt_median"] = tx2_norm.loc["Tb_Prevention_Ipt_median"] / tx2_norm.loc["Tb_Prevention_Ipt_median", 4]
tx2_norm.loc["Hiv_Prevention_Prep_median"] = tx2_norm.loc["Hiv_Prevention_Prep_median"] / tx2_norm.loc["Hiv_Prevention_Prep_median", 13]

tx3 = tx_id3[tx_id3.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx3 = tx3.loc[:,~tx3.columns.str.contains('lower')]
tx3 = tx3.loc[:,~tx3.columns.str.contains('upper')]
tx3 = tx3.T  # transpose for plotting heatmap
tx3 = tx3.fillna(1)  # replce nan with 0
tx3_norm = tx3.divide(tx3.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx3_norm.loc["Tb_Prevention_Ipt_median"] = tx3_norm.loc["Tb_Prevention_Ipt_median"] / tx3_norm.loc["Tb_Prevention_Ipt_median", 4]
tx3_norm.loc["Hiv_Prevention_Prep_median"] = tx3_norm.loc["Hiv_Prevention_Prep_median"] / tx3_norm.loc["Hiv_Prevention_Prep_median", 13]

tx4 = tx_id4[tx_id4.columns[pd.Series(tx_id1.columns).str.startswith(('Hiv', 'Tb'))]]
tx4 = tx4.loc[:,~tx4.columns.str.contains('lower')]
tx4 = tx4.loc[:,~tx4.columns.str.contains('upper')]
tx4 = tx4.T  # transpose for plotting heatmap
tx4 = tx4.fillna(1)  # replce nan with 0
tx4_norm = tx4.divide(tx4.iloc[:, 0].values, axis=0)  # calculate diff from baseline
# for prevention (diff start dates), calculate diff from first introduction
tx4_norm.loc["Tb_Prevention_Ipt_median"] = tx4_norm.loc["Tb_Prevention_Ipt_median"] / tx4_norm.loc["Tb_Prevention_Ipt_median", 4]
tx4_norm.loc["Hiv_Prevention_Prep_median"] = tx4_norm.loc["Hiv_Prevention_Prep_median"] / tx4_norm.loc["Hiv_Prevention_Prep_median", 13]

# rename treatment IDs
appt_types = ["TB test", "HIV test", "TB X-ray", "HIV tx", "VMMC",
              "TB tx", "TB follow-up", "TB IPT", "PrEP"]
tx0_norm.index = appt_types
tx1_norm.index = appt_types
tx2_norm.index = appt_types
tx3_norm.index = appt_types
tx4_norm.index = appt_types

years = list((range(2010, 2036, 1)))

tx0_norm.columns = years
tx1_norm.columns = years
tx2_norm.columns = years
tx3_norm.columns = years
tx4_norm.columns = years


# ---------------------------------- PLOTS ------------------------------------
plt.style.use('ggplot')
cmap = sns.cm.rocket_r

# Make plot
fig = plt.figure(figsize=(8, 6))

# heatmap scenario 0?
ax0 = plt.subplot2grid((2, 3), (0, 0))  # 2 rows, 3 cols
sns.heatmap(tx1_norm,
                 xticklabels=False,
                 yticklabels=1,
            vmin=0,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=True,
            cbar_kws={
                'pad': .02,
                'ticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            },
            )
ax0.set_title("Scenario 1", size=10)

# heatmap scenario 2?
ax1 = plt.subplot2grid((2, 3), (1, 0))
sns.heatmap(tx2_norm,
                 xticklabels=5,
                 yticklabels=1,
            vmin=0,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=True,
            cbar_kws={
                'pad': .02,
                'ticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            },
            )
ax1.set_title("Scenario 2", size=10)

# Frac HCW time
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
ax2.yaxis.tick_right()
ax2.plot(years, hcw_diff1, "-", color="C0")
# ax2.fill_between(hcw1.index, hcw_diff1["lower"], hcw_diff1["upper"], color="C0", alpha=0.2)
ax2.plot(years, hcw_diff2, "-", color="C2")
# ax2.fill_between(hcw2.index, hcw_diff2["lower"], hcw_diff2["upper"], color="C2", alpha=0.2)
ax2.plot(years, hcw_diff3, "-", color="C4")
# ax2.fill_between(hcw3.index, hcw_diff3["lower"], hcw_diff3["upper"], color="C4", alpha=0.2)
ax2.plot(years, hcw_diff4, "-", color="C6")
# ax2.fill_between(hcw4.index, hcw_diff4["lower"], hcw_diff4["upper"], color="C6", alpha=0.2)

plt.show()

