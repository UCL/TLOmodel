from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize, get_scenario_info,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
results_folder1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results_folder3 = get_scenario_outputs("scenario3.py", outputspath)[-1]

# look at one log (default draw=0 and run=0)
log = load_pickled_dataframes(results_folder1)

# ---------------------------------- Fraction HCW time-------------------------------------

# output fraction of HCW time by year
# summarise across all draws and runs
capacity1 = extract_results(
    results_folder1,
    module="tlo.methods.healthsystem.summary",
    key="health_system_annual_logs",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['capacity'].mean()
    ),
)
capacity1.columns = capacity1.columns.get_level_values(0)
hcw1 = pd.DataFrame(index=capacity1.index, columns=["median", "lower", "upper"])
hcw1["median"] = capacity1.median(axis=1)
hcw1["lower"] = capacity1.quantile(q=0.025, axis=1)
hcw1["upper"] = capacity1.quantile(q=0.975, axis=1)


capacity3 = extract_results(
    results_folder3,
    module="tlo.methods.healthsystem.summary",
    key="health_system_annual_logs",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['capacity'].mean()
    ),
)
capacity3.columns = capacity3.columns.get_level_values(0)
hcw3 = pd.DataFrame(index=capacity3.index, columns=["median", "lower", "upper"])
hcw3["median"] = capacity3.median(axis=1)
hcw3["lower"] = capacity3.quantile(q=0.025, axis=1)
hcw3["upper"] = capacity3.quantile(q=0.975, axis=1)

# plot
# Make plot
fig, ax = plt.subplots()
ax.plot(hcw1.index, hcw1["median"], "-", color="C3")
ax.fill_between(hcw1.index, hcw1["lower"], hcw1["upper"], color="C3", alpha=0.2)

ax.plot(hcw3.index, hcw3["median"], "-", color="C2")
ax.fill_between(hcw3.index, hcw3["lower"], hcw3["upper"], color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.title("Fraction of overall healthcare worker time")
plt.ylabel("Fraction of overall healthcare worker time")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()

# ---------------------------------- Total HSIs -------------------------------------


def summarise_treatment_counts(df_list, treatment_id):
    """ summarise the treatment counts across all draws/runs for one results folder
        requires a list of dataframes with all treatments listed with associated counts
    """
    number_runs = len(df_list)
    number_HSI_by_run = pd.DataFrame(index=np.arange(40), columns=np.arange(number_runs))
    column_names = [
        treatment_id + "_median",
        treatment_id + "_lower",
        treatment_id + "_upper"]
    out = pd.DataFrame(columns=column_names)

    for i in range(number_runs):
        number_HSI_by_run.iloc[:,i] = pd.Series(df_list[i].loc[:, treatment_id])

    out.iloc[:,0] = number_HSI_by_run.median(axis=1)
    out.iloc[:,1] = number_HSI_by_run.quantile(q=0.025, axis=1)
    out.iloc[:,2] = number_HSI_by_run.quantile(q=0.975, axis=1)

    return out


writer = pd.ExcelWriter(outputspath / ("treatment_counts" + ".xlsx"))


def write_to_excel(results_folder, module, key, column, sheet_name):
    info = get_scenario_info(results_folder)

    df_list = list()
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

            new = df[['date', column]].copy()
            df_list.append(pd.DataFrame(new[column].to_list()))

    # for column in each df, get median
    # list of treatment IDs
    list_tx_id = list(df_list[0].columns)
    results = pd.DataFrame(index=np.arange(40))

    for treatment_id in list_tx_id:
        tmp = summarise_treatment_counts(df_list, treatment_id)

        # append output to dataframe
        results = results.join(tmp)

    results.to_excel(writer, sheet_name=sheet_name)
    writer.save()


write_to_excel(results_folder=results_folder1,
               module="tlo.methods.healthsystem.summary",
               key="health_system_annual_logs",
               column="treatment_counts",
               sheet_name="tx_counts_scenario1")









# output total numbers of HSI for years 2022-2035 with uncertainty
total_hsi_sc1 = extract_results(
    results_folder1,
    module="tlo.methods.healthsystem.summary",
    key="health_system_annual_logs",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['hsi_treatment_id'].count()
    ),
    do_scaling=True,
)



total_hsi_sc3 = summarize(extract_results(
    results_folder3,
    module="tlo.methods.healthsystem",
    key="HSI_Event",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['TREATMENT_ID'].count()
    ),
    do_scaling=True,
),
    only_mean=False,
    collapse_columns=True
)

# plot
# Make plot
fig, ax = plt.subplots()
ax.plot(total_hsi_sc1.index, total_hsi_sc1["mean"], "-", color="C3")
ax.fill_between(total_hsi_sc1.index, total_hsi_sc1["lower"], total_hsi_sc1["upper"], color="C3", alpha=0.2)

ax.plot(total_hsi_sc3.index, total_hsi_sc3["mean"], "-", color="C2")
ax.fill_between(total_hsi_sc3.index, total_hsi_sc3["lower"], total_hsi_sc3["upper"], color="C2", alpha=0.2)
fig.subplots_adjust(left=0.15)
plt.title("Total numbers of appointments required")
plt.ylabel("Total numbers of appointments")
plt.legend(["Scenario 1", "Scenario 3"])

plt.show()

# ---------------------------------- HSI by type -------------------------------------
# output numbers of HSI by type for each year
hsi_by_type = summarize(extract_results(
    results_folder3,
    module="tlo.methods.healthsystem",
    key="HSI_Event",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'TREATMENT_ID'])['TREATMENT_ID'].count()),
    do_scaling=True
),
    only_mean=True,
    collapse_columns=True
)

# turn into matrix for heatmap
# turn multi-index into columns of df
tmp = hsi_by_type.reset_index()
# format long to wide
tmp2 = tmp.pivot_table(
    values='mean', index='TREATMENT_ID', columns='year',
    fill_value=0, aggfunc='mean')
# select generic, hiv and tb appts only - remove epi appts
tmp3 = tmp2[tmp2.index.str.startswith(('Generic', 'Hiv', 'Tb'))]
tmp4 = np.log10(tmp3)  # take logs of values as scale is huge
tmp4[tmp4 < 0] = 0  # fill -Inf with zeros
# rename treatment IDs
appt_types = ["General appt", "Circumcision", "HIV PrEP", "HIV test", "HIV treatment",
              "TB follow-up", "IPT", "TB test", "TB treatment", "Chest x-ray"]
tmp4.index = appt_types

cmap = sns.cm.rocket_r

# write to excel
outputpath = Path("./outputs")  # folder for convenience of storing outputs
writer = pd.ExcelWriter(outputpath / ("hsi_by_type3" + ".xlsx"))
tmp4.to_excel(writer, sheet_name="scenario3")
writer.save()

fig, ax = plt.subplots()
ax = sns.heatmap(tmp4,
                 xticklabels=2,
                 yticklabels=1,
                 vmax=7,
                 center=4,
                 linewidth=0.5,
                 cmap=cmap)
ax.set(xlabel=None, ylabel=None)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=10)
fig.subplots_adjust(left=0.2)
plt.title("Scenario 3")
plt.show()

# ---------------------------------- HIV incidence -------------------------------------

# adult hiv incidence
hiv_adult_inc1 = extract_results(
        results_folder1,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
)
# set first row to NA as it shows incidence=0
# hiv_adult_inc1.iloc[0, :] = np.nan
hiv_adult_inc1.columns = hiv_adult_inc1.columns.get_level_values(0)
hiv_inc1 = pd.DataFrame(index=hiv_adult_inc1.index, columns=["median", "lower", "upper"])
hiv_inc1["median"] = hiv_adult_inc1.median(axis=1)
hiv_inc1["lower"] = hiv_adult_inc1.quantile(q=0.025, axis=1)
hiv_inc1["upper"] = hiv_adult_inc1.quantile(q=0.975, axis=1)

hiv_adult_inc2 = extract_results(
        results_folder2,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
)
# set first row to NA as it shows incidence=0
# hiv_adult_inc2.iloc[0, :] = np.nan
hiv_adult_inc2.columns = hiv_adult_inc2.columns.get_level_values(0)
hiv_inc2 = pd.DataFrame(index=hiv_adult_inc2.index, columns=["median", "lower", "upper"])
hiv_inc2["median"] = hiv_adult_inc2.median(axis=1)
hiv_inc2["lower"] = hiv_adult_inc2.quantile(q=0.025, axis=1)
hiv_inc2["upper"] = hiv_adult_inc2.quantile(q=0.975, axis=1)


hiv_adult_inc3 = extract_results(
        results_folder3,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
)
# set first row to NA as it shows incidence=0
# hiv_adult_inc3.iloc[0, :] = np.nan
hiv_adult_inc3.columns = hiv_adult_inc3.columns.get_level_values(0)
hiv_inc3 = pd.DataFrame(index=hiv_adult_inc3.index, columns=["median", "lower", "upper"])
hiv_inc3["median"] = hiv_adult_inc3.median(axis=1)
hiv_inc3["lower"] = hiv_adult_inc3.quantile(q=0.025, axis=1)
hiv_inc3["upper"] = hiv_adult_inc3.quantile(q=0.975, axis=1)


# plot
resourcefilepath = Path("./resources")

xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

data_hiv_unaids = pd.read_excel(xls, sheet_name="unaids_infections_art2021")
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids["year"], format="%Y")
data_hiv_unaids = data_hiv_unaids.drop(columns=["year"])

# Make plot
fig, ax = plt.subplots()
ax.plot(hiv_inc1.index, hiv_inc1["median"] * 100, "-", color="C3")
ax.fill_between(hiv_inc1.index, hiv_inc1["lower"] * 100, hiv_inc1["upper"] * 100, color="C3",
                alpha=0.2)

# ax.plot(hiv_inc2.index, hiv_inc2["median"] * 100, "-", color="C0")
# ax.fill_between(hiv_inc2.index, hiv_inc2["lower"] * 100, hiv_inc2["upper"] * 100, color="C0",
#                 alpha=0.2)

ax.plot(hiv_inc3.index, hiv_inc3["median"] * 100, "-", color="C2")
ax.fill_between(hiv_inc3.index, hiv_inc3["lower"] * 100, hiv_inc3["upper"] * 100, color="C2",
                alpha=0.2)

ax.plot(data_hiv_unaids.index, data_hiv_unaids["incidence_per1000_age15_49"] / 10, "-", color="C4")
ax.fill_between(data_hiv_unaids.index, data_hiv_unaids["incidence_per1000_age15_49_lower"] / 10,
                data_hiv_unaids["incidence_per1000_age15_49_upper"] / 10, color="C4", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.axvline(x=hiv_adult_inc3.index[10])
plt.title("HIV incidence in adults aged 15-49 years")
plt.ylabel("HIV incidence per 100py")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3", "UNAIDS"])

plt.show()

# ---------------------------------- TB incidence -------------------------------------

# tb incidence
tb_adult_inc1 = extract_results(
        results_folder1,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)
tb_adult_inc1.columns = tb_adult_inc1.columns.get_level_values(0)
tb_inc1 = pd.DataFrame(index=tb_adult_inc1.index, columns=["median", "lower", "upper"])
tb_inc1["median"] = tb_adult_inc1.median(axis=1)
tb_inc1["lower"] = tb_adult_inc1.quantile(q=0.025, axis=1)
tb_inc1["upper"] = tb_adult_inc1.quantile(q=0.975, axis=1)

tb_adult_inc2 = extract_results(
        results_folder2,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)
tb_adult_inc2.columns = tb_adult_inc2.columns.get_level_values(0)
tb_inc2 = pd.DataFrame(index=tb_adult_inc2.index, columns=["median", "lower", "upper"])
tb_inc2["median"] = tb_adult_inc2.median(axis=1)
tb_inc2["lower"] = tb_adult_inc2.quantile(q=0.025, axis=1)
tb_inc2["upper"] = tb_adult_inc2.quantile(q=0.975, axis=1)

tb_adult_inc3 = extract_results(
        results_folder3,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)
tb_adult_inc3.columns = tb_adult_inc3.columns.get_level_values(0)
tb_inc3 = pd.DataFrame(index=tb_adult_inc3.index, columns=["median", "lower", "upper"])
tb_inc3["median"] = tb_adult_inc3.median(axis=1)
tb_inc3["lower"] = tb_adult_inc3.quantile(q=0.025, axis=1)
tb_inc3["upper"] = tb_adult_inc3.quantile(q=0.975, axis=1)


# get values per 100,000 py, pop runs 500,000
tb_inc1 = tb_inc1 / 5
tb_inc2 = tb_inc2 / 5
tb_inc3 = tb_inc3 / 5

# plot
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2020")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = pd.to_datetime(data_tb_who["year"], format="%Y")
data_tb_who = data_tb_who.drop(columns=["year"])

# Make plot
fig, ax = plt.subplots()
ax.plot(tb_inc1.index, tb_inc1["median"], "-", color="C3")
ax.fill_between(tb_inc1.index, tb_inc1["lower"], tb_inc1["upper"], color="C3", alpha=0.2)

# ax.plot(tb_inc2.index, tb_inc2["median"], "-", color="C0")
# ax.fill_between(tb_inc2.index, tb_inc2["lower"], tb_inc2["upper"], color="C0", alpha=0.2)

ax.plot(tb_inc3.index, tb_inc3["median"], "-", color="C2")
ax.fill_between(tb_inc3.index, tb_inc3["lower"], tb_inc3["upper"], color="C2", alpha=0.2)

ax.plot(data_tb_who.index, data_tb_who["incidence_per_100k"], "-", color="C4")
ax.fill_between(data_tb_who.index, data_tb_who["incidence_per_100k_low"],
                data_tb_who["incidence_per_100k_high"], color="C4", alpha=0.2)

fig.subplots_adjust(left=0.15)
plt.axvline(x=hiv_adult_inc3.index[10])

plt.ylabel("TB incidence per 100,000 population")
plt.title("Active TB incidence")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3", "WHO"])

plt.show()

# ---------------------------------- DEATHS ---------------------------------- #
# todo don't scale because converting to mortality per 100k person-years
results_deaths1 = extract_results(
    results_folder1,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
)

tmp = results_deaths1.reset_index()
tmp.columns = tmp.columns.get_level_values(0)

# AIDS deaths
# select cause of death
tmp = tmp.loc[
    (tmp.cause == "AIDS_TB") | (tmp.cause == "AIDS_non_TB")
    ]
# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS1 = pd.DataFrame(tmp.groupby(["year"]).sum())

# scenario 2
results_deaths2 = extract_results(
    results_folder2,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
)

tmp = results_deaths2.reset_index()
tmp.columns = tmp.columns.get_level_values(0)

# AIDS deaths
# select cause of death
tmp = tmp.loc[
    (tmp.cause == "AIDS_TB") | (tmp.cause == "AIDS_non_TB")
    ]
# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS2 = pd.DataFrame(tmp.groupby(["year"]).sum())

results_deaths3 = extract_results(
    results_folder3,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
)

tmp = results_deaths3.reset_index()
tmp.columns = tmp.columns.get_level_values(0)

# AIDS deaths
# select cause of death
tmp = tmp.loc[
    (tmp.cause == "AIDS_TB") | (tmp.cause == "AIDS_non_TB")
    ]
# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS3 = pd.DataFrame(tmp.groupby(["year"]).sum())

pop = 500000
# AIDS mortality rates per 100k person-years
total_aids_deaths_rate_100kpy1 = pd.Series(
    (model_deaths_AIDS1.median(axis=1).values / pop) * 100000
)
total_aids_deaths_rate_100kpy1.index = model_deaths_AIDS1.index

# scenario 2
# total_aids_deaths_rate_100kpy2 = pd.Series(
#     (model_deaths_AIDS2["mean"].values / pop) * 100000
# )
# total_aids_deaths_rate_100kpy_lower2 = pd.Series(
#     (model_deaths_AIDS2["lower"].values / pop) * 100000
# )
# total_aids_deaths_rate_100kpy_upper2 = pd.Series(
#     (model_deaths_AIDS2["upper"].values / [pop]) * 100000
# )
# total_aids_deaths_rate_100kpy2.index = model_deaths_AIDS2.index
# total_aids_deaths_rate_100kpy_lower2.index = model_deaths_AIDS2.index
# total_aids_deaths_rate_100kpy_upper2.index = model_deaths_AIDS2.index

# scenario 3
total_aids_deaths_rate_100kpy3 = pd.Series(
    (model_deaths_AIDS3.median(axis=1).values / pop) * 100000
)
total_aids_deaths_rate_100kpy3.index = model_deaths_AIDS3.index


# Make plot
fig, ax = plt.subplots()
ax.plot(total_aids_deaths_rate_100kpy1.index, total_aids_deaths_rate_100kpy1, "-", color="C3")
# ax.fill_between(total_aids_deaths_rate_100kpy_lower1.index,
#                 total_aids_deaths_rate_100kpy_lower1, total_aids_deaths_rate_100kpy_upper1, color="C3", alpha=0.2)

# ax.plot(total_aids_deaths_rate_100kpy2.index, total_aids_deaths_rate_100kpy2, "-", color="C0")
# ax.fill_between(total_aids_deaths_rate_100kpy_lower2.index,
#                 total_aids_deaths_rate_100kpy_lower2, total_aids_deaths_rate_100kpy_upper2, color="C0", alpha=0.2)

ax.plot(total_aids_deaths_rate_100kpy3.index, total_aids_deaths_rate_100kpy3, "-", color="C2")
# ax.fill_between(total_aids_deaths_rate_100kpy_lower3.index,
#                 total_aids_deaths_rate_100kpy_lower3, total_aids_deaths_rate_100kpy_upper3, color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)
# plt.axvline(x=hiv_adult_inc3.index[10])

plt.ylabel("AIDS mortality per 100,000 population")
plt.title("AIDS mortality per 100,000 population")
plt.legend(["Scenario 1", "Scenario 3"])
# plt.legend(["Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()

# -------------------------------- TB deaths -------------------------------------

# TB deaths
# select cause of death
tmp = results_deaths1.reset_index()
model_deaths_TB1 = tmp.loc[(tmp.cause == "TB")]

tmp = results_deaths2.reset_index()
model_deaths_TB2 = tmp.loc[(tmp.cause == "TB")]

tmp = results_deaths3.reset_index()
model_deaths_TB3 = tmp.loc[(tmp.cause == "TB")]

pop = 40000
# AIDS mortality rates per 100k person-years
total_tb_deaths_rate_100kpy1 = pd.Series(
    (model_deaths_TB1["mean"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_lower1 = pd.Series(
    (model_deaths_TB1["lower"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_upper1 = pd.Series(
    (model_deaths_TB1["upper"].values / [pop]) * 100000
)
total_tb_deaths_rate_100kpy1.index = model_deaths_TB1["year"]
total_tb_deaths_rate_100kpy_lower1.index = model_deaths_TB1["year"]
total_tb_deaths_rate_100kpy_upper1.index = model_deaths_TB1["year"]

# scenario 2
total_tb_deaths_rate_100kpy2 = pd.Series(
    (model_deaths_TB2["mean"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_lower2 = pd.Series(
    (model_deaths_TB2["lower"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_upper2 = pd.Series(
    (model_deaths_TB2["upper"].values / [pop]) * 100000
)
total_tb_deaths_rate_100kpy2.index = model_deaths_TB2["year"]
total_tb_deaths_rate_100kpy_lower2.index = model_deaths_TB2["year"]
total_tb_deaths_rate_100kpy_upper2.index = model_deaths_TB2["year"]

# scenario 3
total_tb_deaths_rate_100kpy3 = pd.Series(
    (model_deaths_TB3["mean"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_lower3 = pd.Series(
    (model_deaths_TB3["lower"].values / pop) * 100000
)
total_tb_deaths_rate_100kpy_upper3 = pd.Series(
    (model_deaths_TB3["upper"].values / [pop]) * 100000
)
total_tb_deaths_rate_100kpy3.index = model_deaths_TB3["year"]
total_tb_deaths_rate_100kpy_lower3.index = model_deaths_TB3["year"]
total_tb_deaths_rate_100kpy_upper3.index = model_deaths_TB3["year"]

# Make plot
fig, ax = plt.subplots()
ax.plot(total_tb_deaths_rate_100kpy1.index, total_tb_deaths_rate_100kpy1, "-", color="C3")
ax.fill_between(total_tb_deaths_rate_100kpy_lower1.index,
                total_tb_deaths_rate_100kpy_lower1, total_tb_deaths_rate_100kpy_upper1, color="C3", alpha=0.2)

ax.plot(total_tb_deaths_rate_100kpy2.index, total_tb_deaths_rate_100kpy2, "-", color="C0")
ax.fill_between(total_tb_deaths_rate_100kpy_lower2.index,
                total_tb_deaths_rate_100kpy_lower2, total_tb_deaths_rate_100kpy_upper2, color="C0", alpha=0.2)

ax.plot(total_tb_deaths_rate_100kpy3.index, total_tb_deaths_rate_100kpy3, "-", color="C2")
ax.fill_between(total_tb_deaths_rate_100kpy_lower3.index,
                total_tb_deaths_rate_100kpy_lower3, total_tb_deaths_rate_100kpy_upper3, color="C2", alpha=0.2)

fig.subplots_adjust(left=0.15)

plt.ylabel("TB mortality per 100,000 population")
plt.title("TB mortality per 100,000 population")
plt.legend(["Scenario 1", "Scenario 2", "Scenario 3"])

plt.show()

# ---------------------------------- PLOTS ---------------------------------------

# total numbers of HSI by year for each scenario (line plot)


# maybe leave this one
# numbers of HSI by type over 2022-2035 for each scenario (clustered bar plot)


# numbers of each appt type by year 2010-2035 (3 x heatmap)
# separate plot for each scenario


# fraction total HCW time used by year for each scenario (line plot)


# numbers of consumables used for each program by year for each scenario (2 x line plots)
