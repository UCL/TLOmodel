
import os
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tlo import Date
from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results0)

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


# Make plot
fig, ax = plt.subplots()

ax.plot(hcw1.index, hcw_diff1, "-", color="C0")
# ax.fill_between(hcw1.index, hcw_diff1["lower"], hcw_diff1["upper"], color="C0", alpha=0.2)

ax.plot(hcw2.index, hcw_diff2, "-", color="C2")
# ax.fill_between(hcw2.index, hcw_diff2["lower"], hcw_diff2["upper"], color="C2", alpha=0.2)

ax.plot(hcw3.index, hcw_diff3, "-", color="C4")
# ax.fill_between(hcw3.index, hcw_diff3["lower"], hcw_diff3["upper"], color="C4", alpha=0.2)

ax.plot(hcw4.index, hcw_diff4, "-", color="C6")
# ax.fill_between(hcw4.index, hcw_diff4["lower"], hcw_diff4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Percentage difference in healthcare worker time")
plt.ylabel("% difference in healthcare worker time")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()

# ---------------------------------- HSI Events - TREATMENT ID -------------------------------------

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


# Make plot
# tb test, x-ray, start tx, follow-up
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.suptitle('HS outputs')

# TB tests
ax1.plot(tx_id0.index, tx_id0["Tb_Treatment_median"], "-", color="C3")

ax1.plot(tx_id1.index, tx_id1["Tb_Treatment_median"], "-", color="C0")

ax1.plot(tx_id2.index, tx_id2["Tb_Treatment_median"], "-", color="C2")

ax1.plot(tx_id3.index, tx_id3["Tb_Treatment_median"], "-", color="C4")

ax1.plot(tx_id4.index, tx_id4["Tb_Treatment_median"], "-", color="C6")

ax1.set(title='Numbers of TB tests',
       ylabel='Numbers of TB tests')

# TB x-ray
ax2.plot(tx_id0.index, tx_id0["Tb_Test_Xray_median"], "-", color="C3")

ax2.plot(tx_id1.index, tx_id1["Tb_Test_Xray_median"], "-", color="C0")

ax2.plot(tx_id2.index, tx_id2["Tb_Test_Xray_median"], "-", color="C2")

ax2.plot(tx_id3.index, tx_id3["Tb_Test_Xray_median"], "-", color="C4")

ax2.plot(tx_id4.index, tx_id4["Tb_Test_Xray_median"], "-", color="C6")

ax2.set(title='Numbers of X-rays for TB',
       ylabel='Numbers of X-rays')

# TB start treatment
ax3.plot(tx_id0.index, tx_id0["Tb_Treatment_median"], "-", color="C3")

ax3.plot(tx_id1.index, tx_id1["Tb_Treatment_median"], "-", color="C0")

ax3.plot(tx_id2.index, tx_id2["Tb_Treatment_median"], "-", color="C2")

ax3.plot(tx_id3.index, tx_id3["Tb_Treatment_median"], "-", color="C4")

ax3.plot(tx_id4.index, tx_id4["Tb_Treatment_median"], "-", color="C6")

ax3.set(title='Numbers starting treatment',
       ylabel='Numbers starting treatment')

# TB treatment follow-up
ax4.plot(tx_id0.index, tx_id0["Tb_Test_FollowUp_median"], "-", color="C3")

ax4.plot(tx_id1.index, tx_id1["Tb_Test_FollowUp_median"], "-", color="C0")

ax4.plot(tx_id2.index, tx_id2["Tb_Test_FollowUp_median"], "-", color="C2")

ax4.plot(tx_id3.index, tx_id3["Tb_Test_FollowUp_median"], "-", color="C4")

ax4.plot(tx_id4.index, tx_id4["Tb_Test_FollowUp_median"], "-", color="C6")

ax4.set(title='Numbers of follow-up appointments',
       ylabel='Numbers of appointments')

fig.tight_layout()

plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
plt.show()


# ----------------------------------------------------------------------------------------
## PREVENTIVE MEASURES


# Make plot
# tb test, x-ray, start tx, follow-up
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.suptitle('HS outputs')

# VMMC
ax1.plot(tx_id0.index, tx_id0["Hiv_Prevention_Circumcision_median"], "-", color="C3")

ax1.plot(tx_id1.index, tx_id1["Hiv_Prevention_Circumcision_median"], "-", color="C0")

ax1.plot(tx_id2.index, tx_id2["Hiv_Prevention_Circumcision_median"], "-", color="C2")

ax1.plot(tx_id3.index, tx_id3["Hiv_Prevention_Circumcision_median"], "-", color="C4")

ax1.plot(tx_id4.index, tx_id4["Hiv_Prevention_Circumcision_median"], "-", color="C6")

ax1.set(title='VMMC',
       ylabel='Numbers of circumcisions')

# PrEP
ax2.plot(tx_id0.index, tx_id0["Hiv_Prevention_Prep_median"], "-", color="C3")

ax2.plot(tx_id1.index, tx_id1["Hiv_Prevention_Prep_median"], "-", color="C0")

ax2.plot(tx_id2.index, tx_id2["Hiv_Prevention_Prep_median"], "-", color="C2")

ax2.plot(tx_id3.index, tx_id3["Hiv_Prevention_Prep_median"], "-", color="C4")

ax2.plot(tx_id4.index, tx_id4["Hiv_Prevention_Prep_median"], "-", color="C6")

ax2.set(title='PrEP',
       ylabel='Numbers initiating PrEP')

# IPT
ax3.plot(tx_id0.index, tx_id0["Tb_Prevention_Ipt_median"], "-", color="C3")

ax3.plot(tx_id1.index, tx_id1["Tb_Prevention_Ipt_median"], "-", color="C0")

ax3.plot(tx_id2.index, tx_id2["Tb_Prevention_Ipt_median"], "-", color="C2")

ax3.plot(tx_id3.index, tx_id3["Tb_Prevention_Ipt_median"], "-", color="C4")

ax3.plot(tx_id4.index, tx_id4["Tb_Prevention_Ipt_median"], "-", color="C6")

ax3.set(title='IPT',
       ylabel='Numbers starting IPT')

fig.delaxes(ax4)

fig.tight_layout()

plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])
plt.show()


# -----------------------------------------------------------------------------------------------------
# HEATMAP of TREATMENTS OCCURRING

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

cmap = sns.cm.rocket_r


fig, axs = plt.subplots(ncols=4, nrows=2,
                        # sharex=True,
                        # sharey=True,
                        constrained_layout=True,
                        figsize=(10, 5),
                        gridspec_kw=dict(width_ratios=[4,4,4,0.2]))
sns.heatmap(tx0_norm,
                 xticklabels=False,
                 yticklabels=1,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[0,0]
            )
axs[0,0].set_title("Scenario 0", size=10)

sns.heatmap(tx1_norm,
                 xticklabels=False,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[0,1]
            )
axs[0,1].set_title("Scenario 1", size=10)

sns.heatmap(tx2_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[0,2]
            )
axs[0,2].set_title("Scenario 2", size=10)

cb = fig.colorbar(axs[0, 0].collections[0],
                  cax=axs[0, 3],
                  # ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                  drawedges=False)
cb.set_ticks([0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 2.95])
cb.set_ticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cb.ax.tick_params(labelsize=9)
cb.outline.set_edgecolor('white')

sns.heatmap(tx3_norm,
                 xticklabels=5,
                 # yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[1,0]
            )
axs[1,0].set_title("Scenario 3", size=10)

sns.heatmap(tx4_norm,
                 xticklabels=5,
                 yticklabels=False,
                 vmax=3,
                 linewidth=0.5,
                 cmap=cmap,
            cbar=False,
            ax=axs[1,1]
            )
axs[1,1].set_title("Scenario 4", size=10)

axs[1,2].axis("off")
axs[1,3].axis("off")

plt.tick_params(axis="both", which="major", labelsize=9)
plt.show()


#---------------------------------- CONSUMABLES NOT AVAILABLE -----------------------------------------------

cons0 = treatment_counts(results_folder=results0,
                         module="tlo.methods.healthsystem.summary",
                         key="Consumables",
                         column="Item_Available")

cons1 = treatment_counts(results_folder=results1,
                         module="tlo.methods.healthsystem.summary",
                         key="Consumables",
                         column="Item_Available")

cons2 = treatment_counts(results_folder=results2,
                         module="tlo.methods.healthsystem.summary",
                         key="Consumables",
                         column="Item_Available")

cons3 = treatment_counts(results_folder=results3,
                         module="tlo.methods.healthsystem.summary",
                         key="Consumables",
                         column="Item_Available")

cons4 = treatment_counts(results_folder=results4,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_Available")

cons0NA = treatment_counts(results_folder=results0,
                           module="tlo.methods.healthsystem.summary",
                           key="Consumables",
                           column="Item_NotAvailable")

cons1NA = treatment_counts(results_folder=results1,
                           module="tlo.methods.healthsystem.summary",
                           key="Consumables",
                           column="Item_NotAvailable")

cons2NA = treatment_counts(results_folder=results2,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_NotAvailable")

cons3NA = treatment_counts(results_folder=results3,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_NotAvailable")

cons4NA = treatment_counts(results_folder=results4,
               module="tlo.methods.healthsystem.summary",
               key="Consumables",
               column="Item_NotAvailable")

# Make plot of consumables availability
# item 175 adult tx
fig, ax = plt.subplots()
ax.plot(cons2.index, cons2["175_median"], "-", color="C3")
ax.fill_between(cons2.index, cons2["175_lower"], cons2["175_upper"], color="C3", alpha=0.2)

ax.plot(cons2NA.index, cons2NA["175_median"], "-", color="C0")
ax.fill_between(cons2NA.index, cons2NA["175_lower"], cons2NA["175_upper"], color="C0", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Numbers of TB treatment consumables requested - scenario 2")
plt.ylabel("Numbers of consumables")
plt.legend(["Cons available", "Cons not available"])

plt.show()


# item 2671 "First-line ART regimen: adult"
fig, ax = plt.subplots()
ax.plot(cons2.index, cons2["2671_median"], "-", color="C3")
ax.fill_between(cons2.index, cons2["2671_lower"], cons2["2671_upper"], color="C3", alpha=0.2)

ax.plot(cons2NA.index, cons2NA["2671_median"], "-", color="C0")
ax.fill_between(cons2NA.index, cons2NA["2671_lower"], cons2NA["2671_upper"], color="C0", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Numbers of First-line ART treatment consumables requested - scenario 2")
plt.ylabel("Numbers of consumables")
plt.legend(["Cons available", "Cons not available"])

plt.show()


# ---------------------------------- TREATMENT COVERAGE ---------------------------------- #


# tb treatment coverage
def tb_tx_coverage(results_folder):
    tx_cov = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False
    )

    tx_cov.columns = tx_cov.columns.get_level_values(0)
    tx_cov_summary = pd.DataFrame(index=tx_cov.index, columns=["median", "lower", "upper"])
    tx_cov_summary["median"] = tx_cov.median(axis=1)
    tx_cov_summary["lower"] = tx_cov.quantile(q=0.025, axis=1)
    tx_cov_summary["upper"] = tx_cov.quantile(q=0.975, axis=1)

    return tx_cov_summary


tb_tx0 = tb_tx_coverage(results0)
tb_tx1 = tb_tx_coverage(results1)
tb_tx2 = tb_tx_coverage(results2)
tb_tx3 = tb_tx_coverage(results3)
tb_tx4 = tb_tx_coverage(results4)

# Make plot
fig, ax = plt.subplots()
ax.plot(tb_tx0.index, tb_tx0["median"], "-", color="C3")
ax.fill_between(tb_tx0.index, tb_tx0["lower"], tb_tx0["upper"], color="C3", alpha=0.2)

ax.plot(tb_tx1.index, tb_tx1["median"], "-", color="C0")
ax.fill_between(tb_tx1.index, tb_tx1["lower"], tb_tx1["upper"], color="C0", alpha=0.2)

ax.plot(tb_tx2.index, tb_tx2["median"], "-", color="C2")
ax.fill_between(tb_tx2.index, tb_tx2["lower"], tb_tx2["upper"], color="C2", alpha=0.2)

ax.plot(tb_tx3.index, tb_tx3["median"], "-", color="C4")
ax.fill_between(tb_tx3.index, tb_tx3["lower"], tb_tx3["upper"], color="C4", alpha=0.2)

ax.plot(tb_tx4.index, tb_tx4["median"], "-", color="C6")
ax.fill_between(tb_tx4.index, tb_tx4["lower"], tb_tx4["upper"], color="C6", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.ylim((0, 1.0))
plt.title("TB treatment coverage")
plt.ylabel("TB treatment coverage")
plt.legend(["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


# ---------------------------------------------------------------------------------------------------------------


# ------------------------------- TB HSI vs TREATMENT COVERAGE ----------------------------------------------

propTBtx2 = cons2NA["175_median"] / (cons2["175_median"] + cons2NA["175_median"])
propTBtx2_lower = cons2NA["175_lower"] / (cons2["175_lower"] + cons2NA["175_lower"])
propTBtx2_upper = cons2NA["175_upper"] / (cons2["175_upper"] + cons2NA["175_upper"])

propTBtx3 = cons3NA["175_median"] / (cons3["175_median"] + cons3NA["175_median"])
propTBtx3_lower = cons3NA["175_lower"] / (cons3["175_lower"] + cons3NA["175_lower"])
propTBtx3_upper = cons3NA["175_upper"] / (cons3["175_upper"] + cons3NA["175_upper"])

years = list((range(2010, 2036, 1)))


# Make plot
# tb test, x-ray, start tx, follow-up
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    constrained_layout=True,
                                    figsize=(16,5))
fig.suptitle('')

# TB tests
ax1.plot(years, tx_id0["Tb_Treatment_median"], "-", color="C3")
ax1.fill_between(years, tx_id0["Tb_Treatment_lower"], tx_id0["Tb_Treatment_upper"], color="C3", alpha=0.2)

ax1.plot(years, tx_id1["Tb_Treatment_median"], "-", color="C0")
ax1.fill_between(years, tx_id1["Tb_Treatment_lower"], tx_id1["Tb_Treatment_upper"], color="C0", alpha=0.2)

ax1.plot(years, tx_id2["Tb_Treatment_median"], "-", color="C2")
ax1.fill_between(years, tx_id2["Tb_Treatment_lower"], tx_id2["Tb_Treatment_upper"], color="C2", alpha=0.2)

ax1.plot(years, tx_id3["Tb_Treatment_median"], "-", color="C4")
ax1.fill_between(years, tx_id3["Tb_Treatment_lower"], tx_id3["Tb_Treatment_upper"], color="C4", alpha=0.2)

ax1.plot(years, tx_id4["Tb_Treatment_median"], "-", color="C6")
ax1.fill_between(years, tx_id4["Tb_Treatment_lower"], tx_id4["Tb_Treatment_upper"], color="C6", alpha=0.2)

ax1.set(title='',
       ylabel='Numbers of TB tests requested')

# prop cons not available
ax2.plot(years, propTBtx2, "-", color="C2")
ax2.fill_between(years, propTBtx2_lower, propTBtx2_upper, color="C2", alpha=0.2)

ax2.plot(years, propTBtx3, "-", color="C4")
ax2.fill_between(years, propTBtx3_lower, propTBtx3_upper, color="C4", alpha=0.2)

ax2.set(title='',
       ylabel='Proportion TB treatment not available',
        ylim=(0, 1.0))

# tb tx coverage
ax3.plot(tb_tx0.index, tb_tx0["median"], "-", color="C3")
ax3.fill_between(tb_tx0.index, tb_tx0["lower"], tb_tx0["upper"], color="C3", alpha=0.2)

ax3.plot(tb_tx1.index, tb_tx1["median"], "-", color="C0")
ax3.fill_between(tb_tx1.index, tb_tx1["lower"], tb_tx1["upper"], color="C0", alpha=0.2)

ax3.plot(tb_tx2.index, tb_tx2["median"], "-", color="C2")
ax3.fill_between(tb_tx2.index, tb_tx2["lower"], tb_tx2["upper"], color="C2", alpha=0.2)

ax3.plot(tb_tx3.index, tb_tx3["median"], "-", color="C4")
ax3.fill_between(tb_tx3.index, tb_tx3["lower"], tb_tx3["upper"], color="C4", alpha=0.2)

ax3.plot(tb_tx4.index, tb_tx4["median"], "-", color="C6")
ax3.fill_between(tb_tx4.index, tb_tx4["lower"], tb_tx4["upper"], color="C6", alpha=0.2)

ax3.set(title='',
       ylabel='TB treatment coverage',
        ylim=(0, 1.0))

plt.tick_params(axis="both", which="major", labelsize=10)
plt.legend(labels=["Scenario 0", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"])

plt.show()


#---------------------------------------------------------------------------------------------------

