"""This file uses the results of the scenario runs to generate plots

*1 Care cascades (HIV and TB)

"""

import datetime
import os
from pathlib import Path

import lacroix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tlo.analysis.utils import (
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]

log = load_pickled_dataframes(results0)

# colour scheme
berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'


# -----------------------------------------------------------------------------------------
# %% Cascades
# -----------------------------------------------------------------------------------------


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


# ---------------------------------- HIV cascade ---------------------------------- #

# hiv treatment coverage
def hiv_tx_coverage(results_folder):
    hiv_cov = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False
    )

    hiv_cov.columns = hiv_cov.columns.get_level_values(0)
    hiv_cov_summary = pd.DataFrame(index=hiv_cov.index, columns=["median", "lower", "upper"])
    hiv_cov_summary["median"] = hiv_cov.median(axis=1)
    hiv_cov_summary["lower"] = hiv_cov.quantile(q=0.025, axis=1)
    hiv_cov_summary["upper"] = hiv_cov.quantile(q=0.975, axis=1)

    return hiv_cov_summary


hiv_tx0 = hiv_tx_coverage(results0)
hiv_tx1 = hiv_tx_coverage(results1)
hiv_tx2 = hiv_tx_coverage(results2)

# ---------------------------------- TB cascade ---------------------------------- #

# get scaling factor for numbers of tests performed and treatments requested
# scaling factor 145.39609
sf = extract_results(
    results0,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


# tb proportion diagnosed
def tb_proportion_diagnosed(results_folder):

    tb_case = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    tb_dx = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewDiagnosis",
        index="date",
        do_scaling=False
    )

    prop_dx = tb_dx.divide(tb_case, axis='columns')
    prop_dx_out = pd.DataFrame(index=prop_dx.index, columns=["median", "lower", "upper"])
    prop_dx_out["median"] = prop_dx.median(axis=1)
    prop_dx_out["lower"] = prop_dx.quantile(q=0.025, axis=1)
    prop_dx_out["upper"] = prop_dx.quantile(q=0.975, axis=1)

    # replace values >1 with 1
    prop_dx_out[prop_dx_out > 1] = 1

    return prop_dx_out


def tb_proportion_diagnosed_full(results_folder):

    tb_case = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    tb_dx = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewDiagnosis",
        index="date",
        do_scaling=False
    )

    prop_dx = tb_dx.divide(tb_case, axis='columns')
    prop_dx = prop_dx.T.reset_index(drop=True).T
    # replace values >1 with 1
    prop_dx[prop_dx > 1] = 1

    return prop_dx


tb_dx0 = tb_proportion_diagnosed(results0)
tb_dx1 = tb_proportion_diagnosed(results1)
tb_dx2 = tb_proportion_diagnosed(results2)


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


def tb_tx_coverage_full(results_folder):
    tx_cov = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False
    )

    tx_cov.columns = tx_cov.columns.get_level_values(0)

    return tx_cov


tb_tx0 = tb_tx_coverage(results0)
tb_tx1 = tb_tx_coverage(results1)
tb_tx2 = tb_tx_coverage(results2)


# ---------------------------------- create smoothed lines ---------------------------------- #

num_interp = 12


def create_smoothed_lines(data_x, data_y):

    xvals = np.linspace(start=data_x.min(), stop=data_x.max(), num=num_interp)
    smoothed_data = sm.nonparametric.lowess(endog=data_y, exog=data_x, frac=0.45, xvals=xvals, it=0)

    # retain original starting value (2022)
    data_y = data_y.reset_index(drop=True)
    smoothed_data[0] = data_y[0]

    return smoothed_data


data_x = hiv_tx0.index[11:].year  # 2022 onwards
xvals = np.linspace(start=data_x.min(), stop=data_x.max(), num=num_interp)

# hiv testing
hiv_test0 = create_smoothed_lines(data_x,
                                   (tx_id0["Hiv_Test_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test0_l = create_smoothed_lines(data_x,
                                    (tx_id0["Hiv_Test_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test0_u = create_smoothed_lines(data_x,
                                    (tx_id0["Hiv_Test_upper"][11:25] * sf[0][0].values[0]) / 1000000)

hiv_test1 = create_smoothed_lines(data_x,
                                   (tx_id1["Hiv_Test_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test1_l = create_smoothed_lines(data_x,
                                    (tx_id1["Hiv_Test_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test1_u = create_smoothed_lines(data_x,
                                    (tx_id1["Hiv_Test_upper"][11:25] * sf[0][0].values[0]) / 1000000)

hiv_test2 = create_smoothed_lines(data_x,
                                   (tx_id2["Hiv_Test_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test2_l = create_smoothed_lines(data_x,
                                    (tx_id2["Hiv_Test_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_test2_u = create_smoothed_lines(data_x,
                                    (tx_id2["Hiv_Test_upper"][11:25] * sf[0][0].values[0]) / 1000000)

# hiv treatment
hiv_treat0 = create_smoothed_lines(data_x,
                                   (tx_id0["Hiv_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat0_l = create_smoothed_lines(data_x,
                                    (tx_id0["Hiv_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat0_u = create_smoothed_lines(data_x,
                                    (tx_id0["Hiv_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

hiv_treat1 = create_smoothed_lines(data_x,
                                   (tx_id1["Hiv_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat1_l = create_smoothed_lines(data_x,
                                    (tx_id1["Hiv_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat1_u = create_smoothed_lines(data_x,
                                    (tx_id1["Hiv_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

hiv_treat2 = create_smoothed_lines(data_x,
                                   (tx_id2["Hiv_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat2_l = create_smoothed_lines(data_x,
                                    (tx_id2["Hiv_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
hiv_treat2_u = create_smoothed_lines(data_x,
                                    (tx_id2["Hiv_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

# hiv prop treated
hiv_prop_treat0 = create_smoothed_lines(data_x, hiv_tx0["median"][11:25])
hiv_prop_treat0_l = create_smoothed_lines(data_x, hiv_tx0["lower"][11:25])
hiv_prop_treat0_u = create_smoothed_lines(data_x, hiv_tx0["upper"][11:25])

hiv_prop_treat1 = create_smoothed_lines(data_x, hiv_tx1["median"][11:25])
hiv_prop_treat1_l = create_smoothed_lines(data_x, hiv_tx1["lower"][11:25])
hiv_prop_treat1_u = create_smoothed_lines(data_x, hiv_tx1["upper"][11:25])

hiv_prop_treat2 = create_smoothed_lines(data_x, hiv_tx2["median"][11:25])
hiv_prop_treat2_l = create_smoothed_lines(data_x, hiv_tx2["lower"][11:25])
hiv_prop_treat2_u = create_smoothed_lines(data_x, hiv_tx2["upper"][11:25])

# tb testing
tb_test0 = create_smoothed_lines(data_x,
                                   (tx_id0["Tb_Test_Screening_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test0_l = create_smoothed_lines(data_x,
                                    (tx_id0["Tb_Test_Screening_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test0_u = create_smoothed_lines(data_x,
                                    (tx_id0["Tb_Test_Screening_upper"][11:25] * sf[0][0].values[0]) / 1000000)

tb_test1 = create_smoothed_lines(data_x,
                                   (tx_id1["Tb_Test_Screening_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test1_l = create_smoothed_lines(data_x,
                                    (tx_id1["Tb_Test_Screening_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test1_u = create_smoothed_lines(data_x,
                                    (tx_id1["Tb_Test_Screening_upper"][11:25] * sf[0][0].values[0]) / 1000000)

tb_test2 = create_smoothed_lines(data_x,
                                   (tx_id2["Tb_Test_Screening_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test2_l = create_smoothed_lines(data_x,
                                    (tx_id2["Tb_Test_Screening_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_test2_u = create_smoothed_lines(data_x,
                                    (tx_id2["Tb_Test_Screening_upper"][11:25] * sf[0][0].values[0]) / 1000000)

# tb treatment
tb_treat0 = create_smoothed_lines(data_x,
                                   (tx_id0["Tb_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat0_l = create_smoothed_lines(data_x,
                                    (tx_id0["Tb_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat0_u = create_smoothed_lines(data_x,
                                    (tx_id0["Tb_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

tb_treat1 = create_smoothed_lines(data_x,
                                   (tx_id1["Tb_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat1_l = create_smoothed_lines(data_x,
                                    (tx_id1["Tb_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat1_u = create_smoothed_lines(data_x,
                                    (tx_id1["Tb_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

tb_treat2 = create_smoothed_lines(data_x,
                                   (tx_id2["Tb_Treatment_median"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat2_l = create_smoothed_lines(data_x,
                                    (tx_id2["Tb_Treatment_lower"][11:25] * sf[0][0].values[0]) / 1000000)
tb_treat2_u = create_smoothed_lines(data_x,
                                    (tx_id2["Tb_Treatment_upper"][11:25] * sf[0][0].values[0]) / 1000000)

# tb prop treated
tb_prop_treat0 = create_smoothed_lines(data_x, tb_tx0["median"][11:25])
tb_prop_treat0_l = create_smoothed_lines(data_x, tb_tx0["lower"][11:25])
tb_prop_treat0_u = create_smoothed_lines(data_x, tb_tx0["upper"][11:25])

tb_prop_treat1 = create_smoothed_lines(data_x, tb_tx1["median"][11:25])
tb_prop_treat1_l = create_smoothed_lines(data_x, tb_tx1["lower"][11:25])
tb_prop_treat1_u = create_smoothed_lines(data_x, tb_tx1["upper"][11:25])

tb_prop_treat2 = create_smoothed_lines(data_x, tb_tx2["median"][11:25])
tb_prop_treat2_l = create_smoothed_lines(data_x, tb_tx2["lower"][11:25])
tb_prop_treat2_u = create_smoothed_lines(data_x, tb_tx2["upper"][11:25])


# ---------------------------------- Plots ---------------------------------- #
plt.style.use('ggplot')

font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 11,
        }

xlabel_pos = [2022, 2024, 2026, 2028, 2030, 2032, 2034]
xlabel = ["2022", "2024", "2026", "2028", "2030", "2032", "2034"]

# Make 6-panel plot
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                             constrained_layout=True,
                                             figsize=(12, 9))
fig.suptitle('')

# HIV testing appts
ax1.plot(xvals, hiv_test0, "-", color=baseline_colour)
ax1.fill_between(xvals, hiv_test0_l, hiv_test0_u, color=baseline_colour, alpha=0.2)

ax1.plot(xvals, hiv_test1, "-", color=sc1_colour)
ax1.fill_between(xvals, hiv_test1_l, hiv_test1_u, color=sc1_colour, alpha=0.2)

ax1.plot(xvals, hiv_test2, "-", color=sc2_colour)
ax1.fill_between(xvals, hiv_test2_l, hiv_test2_u, color=sc2_colour, alpha=0.2)

ax1.set_ylim([2, 7.5])
ax1.set_xticklabels([])

ax1.set(title='',
       ylabel='No. test appts (millions)')

# HIV start treatment appts
ax2.plot(xvals, hiv_treat0, "-", color=baseline_colour)
ax2.fill_between(xvals, hiv_treat0_l, hiv_treat0_u, color=baseline_colour, alpha=0.2)

ax2.plot(xvals, hiv_treat1, "-", color=sc1_colour)
ax2.fill_between(xvals, hiv_treat1_l, hiv_treat1_u, color=sc1_colour, alpha=0.2)

ax2.plot(xvals, hiv_treat2, "-", color=sc2_colour)
ax2.fill_between(xvals, hiv_treat2_l, hiv_treat2_u, color=sc2_colour, alpha=0.2)

ax2.set_ylim([1, 5.5])
ax2.set_xticklabels([])

ax2.set(title='',
       ylabel='No. treatment appts (millions)')


# HIV proportion treated
ax3.plot(xvals, hiv_prop_treat0, "-", color=baseline_colour)
ax3.fill_between(xvals, hiv_prop_treat0_l, hiv_prop_treat0_u, color=baseline_colour, alpha=0.2)

ax3.plot(xvals, hiv_prop_treat1, "-", color=sc1_colour)
ax3.fill_between(xvals, hiv_prop_treat1_l, hiv_prop_treat1_u, color=sc1_colour, alpha=0.2)

ax3.plot(xvals, hiv_prop_treat2, "-", color=sc2_colour)
ax3.fill_between(xvals, hiv_prop_treat2_l, hiv_prop_treat2_u, color=sc2_colour, alpha=0.2)

ax3.set_ylim([0, 1.1])
ax3.set_xticklabels([])

ax3.set(title='',
       ylabel='Proportion treated')
ax3.legend(labels=["Baseline", "Constrained scale-up", "Unconstrained scale-up"])


# TB tests
ax4.plot(xvals, tb_test0, "-", color=baseline_colour)
ax4.fill_between(xvals, tb_test0_l, tb_test0_u, color=baseline_colour, alpha=0.2)

ax4.plot(xvals, tb_test1, "-", color=sc1_colour)
ax4.fill_between(xvals, tb_test1_l, tb_test1_u, color=sc1_colour, alpha=0.2)

ax4.plot(xvals, tb_test0, "-", color=sc2_colour)
ax4.fill_between(xvals, tb_test0_l, tb_test0_u, color=sc2_colour, alpha=0.2)

ax4.set_ylim([6, 15])
ax4.set_xticks(xlabel_pos)
ax4.set_xticklabels(xlabel)

ax4.set(title='',
       ylabel='No. test appts (millions)')

# TB start treatment
ax5.plot(xvals, tb_treat0, "-", color=baseline_colour)
ax5.fill_between(xvals, tb_treat0_l, tb_treat0_u, color=baseline_colour, alpha=0.2)

ax5.plot(xvals, tb_treat1, "-", color=sc1_colour)
ax5.fill_between(xvals, tb_treat1_l, tb_treat1_u, color=sc1_colour, alpha=0.2)

ax5.plot(xvals, tb_treat2, "-", color=sc2_colour)
ax5.fill_between(xvals, tb_treat2_l, tb_treat2_u, color=sc2_colour, alpha=0.2)

ax5.set_ylim([0, 0.10])
ax5.set_xticks(xlabel_pos)
ax5.set_xticklabels(xlabel)

ax5.set(title='',
       ylabel='No. treatment appts (millions)')

# TB treatment coverage
ax6.plot(xvals, tb_prop_treat0, "-", color=baseline_colour)
ax6.fill_between(xvals, tb_prop_treat0_l, tb_prop_treat0_u, color=baseline_colour, alpha=0.2)

ax6.plot(xvals, tb_prop_treat1, "-", color=sc1_colour)
ax6.fill_between(xvals, tb_prop_treat1_l, tb_prop_treat1_u, color=sc1_colour, alpha=0.2)

ax6.plot(xvals, tb_prop_treat2, "-", color=sc2_colour)
ax6.fill_between(xvals, tb_prop_treat2_l, tb_prop_treat2_u, color=sc2_colour, alpha=0.2)

ax6.set_ylim([0, 1.1])
ax6.set_xticks(xlabel_pos)
ax6.set_xticklabels(xlabel)

ax6.set(title='',
       ylabel='Proportion treated')

plt.tick_params(axis="both", which="major", labelsize=10)

ax1.text(-0.15, 1.05, 'A)', horizontalalignment='center',
    verticalalignment='center', transform=ax1.transAxes, fontdict=font)

ax2.text(-0.1, 1.05, 'B)', horizontalalignment='center',
    verticalalignment='center', transform=ax2.transAxes, fontdict=font)

ax3.text(-0.15, 1.05, 'C)', horizontalalignment='center',
    verticalalignment='center', transform=ax3.transAxes, fontdict=font)

ax4.text(-0.1, 1.05, 'D)', horizontalalignment='center',
    verticalalignment='center', transform=ax4.transAxes, fontdict=font)

ax5.text(-0.15, 1.05, 'E)', horizontalalignment='center',
    verticalalignment='center', transform=ax5.transAxes, fontdict=font)

ax6.text(-0.1, 1.05, 'F)', horizontalalignment='center',
    verticalalignment='center', transform=ax6.transAxes, fontdict=font)

# fig.savefig(outputspath / "Treatment_cascade_6panel.png")

plt.show()


# extract numbers of hiv and tb appts scaled to full population size
# compare scenario 1 and 2
scaling_factor = 145.39609

# produce lists of relevant columns
tb_appts = [col for col in tx_id1 if col.startswith('Tb')]
hiv_appts = [col for col in tx_id1 if col.startswith('Hiv')]
all_appts = tb_appts + hiv_appts

data1 = tx_id1[tx_id1.columns.intersection(all_appts)]
data2 = tx_id2[tx_id2.columns.intersection(all_appts)]

# row 13 is 2023
# sum all appts from 2023 for each scenario
tmp1 = data1.iloc[13:26]
tmp1.loc['Total'] = tmp1.sum()
tmp2 = data2.iloc[13:26]
tmp2.loc['Total'] = tmp2.sum()

# total number of hiv/tb appts 2023-2035 - sum only the "_median" columns
median_appts = [col for col in tx_id1 if col.endswith('_median')]
data1_median = tmp1[tmp1.columns.intersection(median_appts)]
data2_median = tmp2[tmp2.columns.intersection(median_appts)]

total_sc1 = data1_median.loc['Total'].sum() * scaling_factor
total_sc2 = data2_median.loc['Total'].sum() * scaling_factor

# additional appts required due to supply constraints in hiv/tb
print(total_sc1 - total_sc2)

# need to scale to full population
sc1_prev_appts = sum([tmp1.loc["Total", "Hiv_Prevention_Circumcision_median"],
                     tmp1.loc["Total", "Hiv_Prevention_Infant_median"],
                     tmp1.loc["Total", "Tb_Prevention_Ipt_median"],
                     tmp1.loc["Total", "Hiv_Prevention_Prep_median"]]) * scaling_factor

sc2_prev_appts = sum([tmp2.loc["Total", "Hiv_Prevention_Circumcision_median"],
                     tmp2.loc["Total", "Hiv_Prevention_Infant_median"],
                     tmp2.loc["Total", "Tb_Prevention_Ipt_median"],
                     tmp2.loc["Total", "Hiv_Prevention_Prep_median"]]) * scaling_factor





