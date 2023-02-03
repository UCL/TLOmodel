""" load the outputs from a simulation and plot the results with comparison data """

import datetime
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import compare_number_of_deaths

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Function to make standard plot to compare model and data
def make_plot(model=None, data_mid=None, data_low=None, data_high=None, title_str=None):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, alpha=0.2)
    plt.title(title_str)
    plt.legend(["Model", "Data"])
    plt.gca().set_ylim(bottom=0)
    plt.savefig(
        outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
    )
    # plt.show()


# ---------------------------------------------------------------------- #
# %%: DATA
# ---------------------------------------------------------------------- #
start_date = 2010
end_date = 2020

# load all the data for calibration

# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2023")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = pd.to_datetime(data_tb_who["year"], format="%Y")
data_tb_who = data_tb_who.drop(columns=["year"])

# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name="latent_TB2014_summary")
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == "0_80"]
data_tb_latent_estimate = data_tb_latent_all_ages.proportion_latent_TB.values[0]
data_tb_latent_lower = abs(
    data_tb_latent_all_ages.proportion_latent_TB_lower.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_upper = abs(
    data_tb_latent_all_ages.proportion_latent_TB_upper.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_yerr = [data_tb_latent_lower, data_tb_latent_upper]

# TB deaths WHO
deaths_2010_2014 = data_tb_who.loc["2010-01-01":"2014-01-01"]
deaths_2015_2019 = data_tb_who.loc["2015-01-01":"2019-01-01"]

deaths_2010_2014_average = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv"].values.mean()
deaths_2010_2014_average_low = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv_low"].values.mean()
deaths_2010_2014_average_high = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv_high"].values.mean()

deaths_2015_2019_average = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv"].values.mean()
deaths_2015_2019_average_low = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv_low"].values.mean()
deaths_2015_2019_average_high = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv_high"].values.mean()


# TB treatment coverage
data_tb_ntp = pd.read_excel(xls_tb, sheet_name="NTP2019")
data_tb_ntp.index = pd.to_datetime(data_tb_ntp["year"], format="%Y")
data_tb_ntp = data_tb_ntp.drop(columns=["year"])

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name="unaids_infections_art2021")
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids["year"], format="%Y")
data_hiv_unaids = data_hiv_unaids.drop(columns=["year"])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
data_hiv_unaids_deaths.index = pd.to_datetime(
    data_hiv_unaids_deaths["year"], format="%Y"
)
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=["year"])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name="children0_14_prev_AIDSinfo")
data_hiv_aidsinfo.index = pd.to_datetime(data_hiv_aidsinfo["year"], format="%Y")
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=["year"])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name="unaids_program_perf")
data_hiv_program.index = pd.to_datetime(data_hiv_program["year"], format="%Y")
data_hiv_program = data_hiv_program.drop(columns=["year"])

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name="MPHIA_incidence2015")
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"
].values[0]
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_lower"
].values[0]
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_upper"
].values[0]
data_hiv_mphia_inc_yerr = [
    abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
    abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate),
]

data_hiv_mphia_prev = pd.read_excel(xls, sheet_name="MPHIA_prevalence_art2015")

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name="DHS_prevalence")

# MoH HIV testing data
data_hiv_moh_tests = pd.read_excel(xls, sheet_name="MoH_numbers_tests")
data_hiv_moh_tests.index = pd.to_datetime(data_hiv_moh_tests["year"], format="%Y")
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=["year"])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name="MoH_number_art")


# ---------------------------------------------------------------------- #
# %%: OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

# person-years all ages (irrespective of HIV status)
py_ = output["tlo.methods.demography"]["person_years"]
years = pd.to_datetime(py_["date"]).dt.year
py = pd.Series(dtype="int64", index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series)
        + (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]

py.index = pd.to_datetime(years, format="%Y")


# ---------------------------------------------------------------------- #
# %%: DISEASE BURDEN
# ---------------------------------------------------------------------- #


# ----------------------------- TB -------------------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs
TB_inc = output["tlo.methods.tb"]["tb_incidence"]
years = pd.to_datetime(TB_inc["date"]).dt.year
TB_inc.index = pd.to_datetime(years, format="%Y")
activeTB_inc_rate = (TB_inc["num_new_active_tb"] / py) * 100000

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate,
    data_mid=data_tb_who["incidence_per_100k"],
    data_low=data_tb_who["incidence_per_100k_low"],
    data_high=data_tb_who["incidence_per_100k_high"],
)
plt.show()

# # ---------------------------------------------------------------------- #
#
# # latent TB prevalence
# latentTB_prev = output["tlo.methods.tb"]["tb_prevalence"]
# latentTB_prev = latentTB_prev.set_index("date")
#
# title_str = "Latent TB prevalence"
# make_plot(
#     title_str=title_str,
#     model=latentTB_prev["tbPrevLatent"],
# )
# plt.ylim((0, 1.0))
# # add latent TB estimate from Houben & Dodd 2016 (value for year=2014)
# plt.errorbar(
#     latentTB_prev.index[4],
#     data_tb_latent_estimate,
#     yerr=[[data_tb_latent_yerr[0]], [data_tb_latent_yerr[1]]],
#     fmt="o",
# )
# # cohen, mathiasen 2019, 33.6% (22.4 - 42.9%)
# plt.errorbar(
#     latentTB_prev.index[9],
#     0.336,
#     yerr=[[0.092], [0.092]],
#     fmt="o",
# )
# plt.ylabel("Prevalence")
# plt.legend(["Model", "Estimate: Houben", "Estimate: Cohen"])
# # plt.savefig(
# #     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# # )
# plt.show()


# ---------------------------------------------------------------------- #

# proportion TB cases that are MDR
mdr = output["tlo.methods.tb"]["tb_mdr"]
mdr = mdr.set_index("date")

title_str = "Proportion of active cases that are MDR"
make_plot(
    title_str=title_str,
    model=mdr["tbPropActiveCasesMdr"],
)
# data from ResourceFile_TB sheet WHO_mdrTB2017
plt.errorbar(mdr.index[7], 0.0075, yerr=[[0.0059], [0.0105]], fmt="o")
plt.legend(["TLO", "WHO reported MDR cases"])
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()


# ---------------------------------------------------------------------- #

# proportion TB cases that are HIV+
# expect around 60% falling to 50% by 2017
tb_hiv = output["tlo.methods.tb"]["tb_incidence"]
tb_hiv = tb_hiv.set_index("date")

title_str = "Proportion of active cases that are HIV+"
make_plot(
    title_str=title_str,
    model=tb_hiv["prop_active_tb_in_plhiv"],
)
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()

# ----------------------------- HIV -------------------------------------- #

prev_and_inc_over_time = output["tlo.methods.hiv"][
    "summary_inc_and_prev_for_adults_and_children_and_fsw"
]
prev_and_inc_over_time = prev_and_inc_over_time.set_index("date")

# HIV - prevalence among in adults aged 15-49
title_str = "HIV Prevalence in Adults Aged 15-49 (%)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_prev_adult_1549"] * 100,
    data_mid=data_hiv_unaids["prevalence_age15_49"] * 100,
    data_low=data_hiv_unaids["prevalence_age15_49_lower"] * 100,
    data_high=data_hiv_unaids["prevalence_age15_49_upper"] * 100,
)

# MPHIA
plt.plot(
    prev_and_inc_over_time.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
    ].values[0],
    "gx",
)

# DHS
x_values = [prev_and_inc_over_time.index[0], prev_and_inc_over_time.index[5]]
y_values = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49"
]
y_lower = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 lower",
        ]
    )
)
y_upper = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 upper",
        ]
    )
)
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], fmt="o")
plt.ylim((0, 15))
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
orange_ci = mlines.Line2D([], [], color="C1", marker=".", markersize=15, label="DHS")
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])
plt.savefig(
    outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence 15-49
title_str = "HIV Incidence in Adults (15-49) (per 100 pyar)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_adult_inc_1549"] * 100,
    data_mid=data_hiv_unaids["incidence_per1000_age15_49"] / 10,
    data_low=data_hiv_unaids["incidence_per1000_age15_49_lower"] / 10,
    data_high=data_hiv_unaids["incidence_per1000_age15_49_upper"] / 10,
)

# MPHIA
plt.errorbar(
    prev_and_inc_over_time.index[6],
    data_hiv_mphia_inc_estimate,
    yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]],
    fmt="o",
)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
orange_ci = mlines.Line2D([], [], color="C1", marker=".", markersize=15, label="MPHIA")
plt.legend(handles=[red_line, blue_line, orange_ci])
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
title_str = "HIV Prevalence in Children (0-14) (%)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_prev_child"] * 100,
    data_mid=data_hiv_aidsinfo["prevalence_0_14"] * 100,
    data_low=data_hiv_aidsinfo["prevalence_0_14_lower"] * 100,
    data_high=data_hiv_aidsinfo["prevalence_0_14_upper"] * 100,
)
# MPHIA
plt.plot(
    prev_and_inc_over_time.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
    ].values[0],
    "gx",
)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, green_cross])
plt.savefig(
    outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence Children
title_str = "HIV Incidence in Children (0-14) per 100 py"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_child_inc"] * 100,
    data_mid=data_hiv_aidsinfo["incidence0_14_per100py"],
    data_low=data_hiv_aidsinfo["incidence0_14_per100py_lower"],
    data_high=data_hiv_aidsinfo["incidence0_14_per100py_upper"],
)
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()


# ---------------------------------------------------------------------- #

# HIV prevalence among female sex workers:

make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=prev_and_inc_over_time["hiv_prev_fsw"] * 100,
)
plt.show()


# ---------------------------------------------------------------------- #
# %%: DEATHS
# ---------------------------------------------------------------------- #


# deaths
deaths = output["tlo.methods.demography"]["death"].copy()  # outputs individual deaths
deaths = deaths.set_index("date")

# TB deaths will exclude TB/HIV
# keep if cause = TB
keep = (deaths.cause == "TB")
deaths_TB = deaths.loc[keep].copy()
deaths_TB["year"] = deaths_TB.index.year  # count by year
tot_tb_non_hiv_deaths = deaths_TB.groupby(by=["year"]).size()
tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format="%Y")

# TB/HIV deaths
keep = (deaths.cause == "AIDS_TB")
deaths_TB_HIV = deaths.loc[keep].copy()
deaths_TB_HIV["year"] = deaths_TB_HIV.index.year  # count by year
tot_tb_hiv_deaths = deaths_TB_HIV.groupby(by=["year"]).size()
tot_tb_hiv_deaths.index = pd.to_datetime(tot_tb_hiv_deaths.index, format="%Y")

# total TB deaths (including HIV+)
total_tb_deaths = tot_tb_non_hiv_deaths.add(tot_tb_hiv_deaths, fill_value=0)
total_tb_deaths.index = pd.to_datetime(total_tb_deaths.index, format="%Y")

# tb mortality rates per 100k person-years
total_tb_deaths_rate_100kpy = (total_tb_deaths / py) * 100000
tot_tb_hiv_deaths_rate_100kpy = (tot_tb_hiv_deaths / py) * 100000
tot_tb_non_hiv_deaths_rate_100kpy = (tot_tb_non_hiv_deaths / py) * 100000

# AIDS DEATHS
# limit to deaths among aged 15+, include HIV/TB deaths
keep = (deaths.age >= 15) & (
    (deaths.cause == "AIDS_TB") | (deaths.cause == "AIDS_non_TB")
)
deaths_AIDS = deaths.loc[keep].copy()
deaths_AIDS["year"] = deaths_AIDS.index.year
tot_aids_deaths = deaths_AIDS.groupby(by=["year"]).size()
tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format="%Y")

# aids mortality rates per 100k person-years
total_aids_deaths_rate_100kpy = (tot_aids_deaths / py) * 100000
#
# # ---------------------------------------------------------------------- #
#
# AIDS deaths (including HIV/TB deaths)
make_plot(
    title_str="Mortality to HIV-AIDS per 1000 capita, data=UNAIDS",
    model=total_aids_deaths_rate_100kpy,
    data_mid=data_hiv_unaids_deaths["AIDS_mortality_per_100k"],
    data_low=data_hiv_unaids_deaths["AIDS_mortality_per_100k_lower"],
    data_high=data_hiv_unaids_deaths["AIDS_mortality_per_100k_upper"],
)

plt.show()


# ---------------------------------------------------------------------- #

# TB deaths (excluding HIV/TB deaths)
make_plot(
    title_str="TB mortality rate (excl HIV) per 100,000 population, data=WHO",
    model=tot_tb_non_hiv_deaths_rate_100kpy,
    data_mid=data_tb_who["mortality_tb_excl_hiv_per_100k"],
    data_low=data_tb_who["mortality_tb_excl_hiv_per_100k_low"],
    data_high=data_tb_who["mortality_tb_excl_hiv_per_100k_high"],
)
plt.ylim((0, 80))
plt.show()


# ---------------------------------------------------------------------- #

# HIV/TB deaths only
make_plot(
    title_str="TB_HIV mortality rate per 100,000 population, data=WHO",
    model=tot_tb_hiv_deaths_rate_100kpy,
    data_mid=data_tb_who["mortality_tb_hiv_per_100k"],
    data_low=data_tb_who["mortality_tb_hiv_per_100k_low"],
    data_high=data_tb_who["mortality_tb_hiv_per_100k_high"],
)

plt.show()


# ---------------------------------------------------------------------- #
# %%: DEATHS - GBD COMPARISON
# ---------------------------------------------------------------------- #


outputpath = Path("./outputs")  # folder for convenience of storing outputs
list_of_paths = outputpath.glob('*.log')  # gets latest log file
latest_path = max(list_of_paths, key=lambda p: p.stat().st_ctime)

# latest_path = sim.log_filepath
# tlo.methods.deviance_measure.log written after log file below:
# outputs\deviance__2022-01-20T105927.log
# latest_path = "outputs\tb_transmission_runs__2023-01-23T132304.log"
death_compare = compare_number_of_deaths(latest_path, resourcefilepath)

# include all ages and both sexes
deaths2010 = death_compare.loc[("2010-2014", slice(None), slice(None), "AIDS")].sum()
deaths2015 = death_compare.loc[("2015-2019", slice(None), slice(None), "AIDS")].sum()

# include all ages and both sexes
deaths2010_TB = death_compare.loc[("2010-2014", slice(None), slice(None), "TB (non-AIDS)")].sum()
deaths2015_TB = death_compare.loc[("2015-2019", slice(None), slice(None), "TB (non-AIDS)")].sum()

x_vals = [1, 2, 3, 4]
labels = ["2010-2014", "2010-2014", "2015-2019", "2015-2019"]
col = ["mediumblue", "red", "mediumblue", "red"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
red_patch = mpatches.Patch(color="red", label="TLO")

# plot AIDS deaths
y_vals = [
    deaths2010["GBD_mean"],
    deaths2010["model"],
    deaths2015["GBD_mean"],
    deaths2015["model"],
]
y_lower = [
    abs(deaths2010["GBD_lower"] - deaths2010["GBD_mean"]),
    np.NAN,
    abs(deaths2015["GBD_lower"] - deaths2015["GBD_mean"]),
    np.NAN,
]
y_upper = [
    abs(deaths2010["GBD_upper"] - deaths2010["GBD_mean"]),
    np.NAN,
    abs(deaths2015["GBD_upper"] - deaths2015["GBD_mean"]),
    np.NAN,
]
plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals,
    [y_vals[0], np.NAN, y_vals[2], np.NAN],
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="lightskyblue",
    markerfacecolor="lightskyblue",
    ecolor="lightskyblue",
)
plt.xticks(ticks=x_vals, labels=labels)
plt.title("Deaths per year due to AIDS")
plt.legend(handles=[blue_patch, red_patch])
plt.tight_layout()
# plt.savefig(outputpath / ("HIV_TB_deaths_with_GBD" + datestamp + ".png"), format='png')
plt.show()


# plot TB deaths
x_vals = [1, 2, 3, 4, 5, 6]
labels = ["2010-2014", "2010-2014", "2010-2014", "2015-2019", "2015-2019", "2015-2019"]
col = ["mediumblue", "mediumseagreen", "red", "mediumblue", "mediumseagreen", "red"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
green_patch = mpatches.Patch(color="mediumseagreen", label="WHO")
red_patch = mpatches.Patch(color="red", label="TLO")

y_vals = [
    deaths2010_TB["GBD_mean"],
    deaths_2010_2014_average,
    deaths2010_TB["model"],
    deaths2015_TB["GBD_mean"],
    deaths_2015_2019_average,
    deaths2015_TB["model"],
]
y_lower = [
    abs(deaths2010_TB["GBD_lower"] - deaths2010_TB["GBD_mean"]),
    deaths_2010_2014_average_low,
    np.NAN,
    abs(deaths2015_TB["GBD_lower"] - deaths2015_TB["GBD_mean"]),
    deaths_2015_2019_average_low,
    np.NAN,
]
y_upper = [
    abs(deaths2010_TB["GBD_upper"] - deaths2010_TB["GBD_mean"]),
    deaths_2010_2014_average_high,
    np.NAN,
    abs(deaths2015_TB["GBD_upper"] - deaths2015_TB["GBD_mean"]),
    deaths_2015_2019_average_high,
    np.NAN,
]
plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals,
    [y_vals[0], y_vals[1], np.NAN, y_vals[3], y_vals[4], np.NAN],
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="lightskyblue",
    markerfacecolor="lightskyblue",
    ecolor="lightskyblue",
)
plt.xticks(ticks=x_vals, labels=labels)

plt.title("Deaths per year due to non-AIDS TB")
plt.legend(handles=[blue_patch, green_patch, red_patch])
plt.tight_layout()
# plt.savefig(outputpath / ("TB_deaths_with_GBD" + datestamp + ".png"), format='png')
plt.show()

# ---------------------------------------------------------------------- #
# %%: PROGRAM OUTPUTS
# ---------------------------------------------------------------------- #

# treatment coverage
Tb_tx_coverage = output["tlo.methods.tb"]["tb_treatment"]
Tb_tx_coverage = Tb_tx_coverage.set_index("date")
Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)

cov_over_time = output["tlo.methods.hiv"]["hiv_program_coverage"]
cov_over_time = cov_over_time.set_index("date")

# ---------------------------------------------------------------------- #

# HIV Treatment Cascade ("90-90-90") Plot for Adults
dx = cov_over_time["dx_adult"] * 100
art_among_dx = (cov_over_time["art_coverage_adult"] / cov_over_time["dx_adult"]) * 100
vs_among_art = (cov_over_time["art_coverage_adult_VL_suppression"]) * 100

pd.concat(
    {
        "diagnosed": dx,
        "art_among_diagnosed": art_among_dx,
        "vs_among_those_on_art": vs_among_art,
    },
    axis=1,
).plot()

plt.gca().spines["right"].set_color("none")
plt.gca().spines["top"].set_color("none")
plt.title("ART Cascade for Adults (15+)")

# data from UNAIDS 2021
# todo scatter the error bars
# unaids: diagnosed
x_values = data_hiv_program.index
y_values = data_hiv_program["percent_know_status"]
y_lower = abs(y_values - data_hiv_program["percent_know_status_lower"])
y_upper = abs(y_values - data_hiv_program["percent_know_status_upper"])
plt.errorbar(
    x_values,
    y_values,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="C0",
    markerfacecolor="C0",
    ecolor="C0",
)

# unaids: diagnosed and on art
x_values = data_hiv_program.index + pd.DateOffset(months=3)
y_values = data_hiv_program["percent_know_status_on_art"]
y_lower = abs(y_values - data_hiv_program["percent_know_status_on_art_lower"])
y_upper = abs(y_values - data_hiv_program["percent_know_status_on_art_upper"])
plt.errorbar(
    x_values,
    y_values,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="C1",
    markerfacecolor="C1",
    ecolor="C1",
)

# unaids: virally suppressed
x_values = data_hiv_program.index + pd.DateOffset(months=6)
y_values = data_hiv_program["percent_on_art_viral_suppr"]
y_lower = abs(y_values - data_hiv_program["percent_on_art_viral_suppr_lower"])
y_upper = abs(y_values - data_hiv_program["percent_on_art_viral_suppr_upper"])
# y_values.index = x_values
# y_lower.index = x_values
# y_lower.index = x_values
plt.errorbar(
    x_values,
    y_values,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="g",
    markerfacecolor="g",
    ecolor="g",
)
plt.ylim((20, 100))
plt.savefig(outputpath / ("HIV_art_cascade_adults" + datestamp + ".pdf"), format='pdf')

plt.show()

# ---------------------------------------------------------------------- #

# HIV Per capita testing rates - data from MoH quarterly reports
make_plot(
    title_str="Per capita testing rates for all ages",
    model=cov_over_time["per_capita_testing_rate"],
    data_mid=data_hiv_moh_tests["annual_testing_rate_all_ages"],
)
plt.legend(["TLO", "MoH"])
plt.ylim((0, 0.6))

plt.show()

# HIV testing yield
# data from MoH quarterly reports, reported yields are modified using SHINY90
testing_yield = cov_over_time["testing_yield"].copy()
testing_yield[0] = 0

make_plot(
    title_str="HIV testing yield - all ages",
    model=testing_yield,
    data_mid=data_hiv_moh_tests["testing_yield"],
)
plt.ylim((0, 0.1))
plt.legend(["TLO", "MoH"])

plt.show()


# ---------------------------------------------------------------------- #

# Percent of all HIV+ on ART
make_plot(
    title_str="Percent of all Adults (15+) HIV+ on ART",
    model=cov_over_time["art_coverage_adult"] * 100,
    data_mid=data_hiv_unaids["ART_coverage_all_HIV_adults"],
    data_low=data_hiv_unaids["ART_coverage_all_HIV_adults_lower"],
    data_high=data_hiv_unaids["ART_coverage_all_HIV_adults_upper"],
)
plt.legend(["TLO", "UNAIDS"])
# plt.savefig(outputpath / ("HIV_Proportion_on_ART" + datestamp + ".png"), format='png')

plt.show()

# ---------------------------------------------------------------------- #

# Circumcision
make_plot(
    title_str="Proportion of Men (15+) That Are Circumcised",
    model=cov_over_time["prop_men_circ"],
)
plt.plot(
    cov_over_time["prop_men_circ"].index[3], 0.23,
    "gx",
)
plt.plot(
    cov_over_time["prop_men_circ"].index[5], 0.279,
    "bx",
)
plt.ylim((0, 0.4))

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="KABP"
)
blue_cross = mlines.Line2D(
    [], [], linewidth=0, color="b", marker="x", markersize=7, label="MDHS"
)
# orange_ci = mlines.Line2D([], [], color="C1", marker=".", markersize=15, label="DHS")
plt.legend(handles=[red_line, green_cross, blue_cross])
plt.savefig(outputpath / ("Proportion_men_circumcised" + datestamp + ".png"), format='png')
plt.show()


# ---------------------------------------------------------------------- #

# PrEP among FSW
make_plot(
    title_str="Proportion of FSW That Are On PrEP",
    model=cov_over_time["prop_fsw_on_prep"],
)
plt.show()


# ---------------------------------------------------------------------- #

# Behaviour Change
make_plot(
    title_str="Proportion of Adults (15+) Exposed to Behaviour Change Intervention",
    model=cov_over_time["prop_adults_exposed_to_behav_intv"],
)
plt.show()

# ---------------------------------------------------------------------- #

# TB treatment coverage
make_plot(
    title_str="Percent of TB cases treated",
    model=Tb_tx_coverage["tbTreatmentCoverage"] * 100,
    data_mid=data_tb_ntp["treatment_coverage"],
)
plt.ylim((0, 100))

plt.legend(["TLO", "NTP"])
# plt.savefig(outputpath / ("Percent_tb_cases_treated" + datestamp + ".png"), format='png')

plt.show()

# # ---------------------------------------------------------------------- #
# distribution of survival times
# df = sim.population.props
#
# # HIV
# # select adults not on treatment (never treated)
# df_hiv = df.loc[
#     df.hv_inf &
#     (df.hv_art == "not") &
#     (df.age_years >= 15) &
#     ((df.cause_of_death == "AIDS_TB") | (df.cause_of_death == "AIDS_non_TB"))]
#
# len(df_hiv)  # 3020 from pop 100k
# len(df_hiv.date_of_death)
#
# # get times from infection to death in years
# survival_times = df_hiv.date_of_death - df_hiv.hv_date_inf
# survival_times = survival_times[survival_times.notnull()]
# survival_times = pd.Series(survival_times.dt.days / 365.25)
#
# # plot histogram
# survival_times.plot.hist(grid=True, bins=20, rwidth=0.9,
#                    color='#607c8e')
# plt.title('Distribution of survival times for PLHIV')
# plt.xlabel('Survival time, years')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.savefig(outputpath / ("Distribution of survival times for PLHIV (untreated)" + datestamp + ".png"), format='png')
# plt.show()
#
#
# # children with HIV
# # select adults not on treatment (never treated)
# df_hiv_child = df.loc[
#     df.hv_inf &
#     (df.hv_art == "not") &
#     (df.age_years < 15) &
#     ((df.cause_of_death == "AIDS_TB") | (df.cause_of_death == "AIDS_non_TB"))]
#
# len(df_hiv_child)  # 3020 from pop 100k
# len(df_hiv_child.date_of_death)
#
# # get times from infection to death in years
# survival_times = df_hiv_child.date_of_death - df_hiv_child.hv_date_inf
# survival_times = survival_times[survival_times.notnull()]
# survival_times = pd.Series(survival_times.dt.days / 365.25)
#
# # plot histogram
# survival_times.plot.hist(grid=True, bins=20, rwidth=0.9,
#                    color='#607c8e')
# plt.title('Distribution of survival times for PLHIV: children')
# plt.xlabel('Survival time, years')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.savefig(outputpath / (
#   "Distribution of survival times for HIV+ children (untreated)" + datestamp + ".png"),
#   format='png')
# plt.show()
#
#
# # TB
# # select adults never treated
# df_tb = df.loc[
#     ~df.tb_date_active.isnull() &
#     df.tb_ever_treated &
#     (df.age_years >= 15) &
#     ((df.cause_of_death == "AIDS_TB") | (df.cause_of_death == "TB"))]
#
# len(df_tb)  # 12 from pop 100k
# len(df_tb.date_of_death)
#
# # get times from infection to death in years
# survival_times = df_tb.date_of_death - df_tb.tb_date_active
# survival_times = survival_times[survival_times.notnull()]
# survival_times = pd.Series(survival_times.dt.days / 365.25)
#
# # plot histogram
# survival_times.plot.hist(grid=True, bins=20, rwidth=0.9,
#                    color='#607c8e')
# plt.title('Distribution of survival times for TB')
# plt.xlabel('Survival time, years')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.savefig(outputpath / ("Distribution of survival times for adults with TB" + datestamp + ".png"), format='png')
# plt.show()
#
# # # ---------------------------------------------------------------------- #
# # HIV test logger
#
# test = output["tlo.methods.hiv"]["hiv_test"].copy()
# test = test.set_index("date")
# test["year"] = test.index.year
# agg_tests = test.groupby(by=["year", "referred_from"]).size()
# total_tests_per_year = test.groupby(by=["year"]).size()
# referral_type_proportion = agg_tests.div(total_tests_per_year, level="year") * 100
#
# writer = pd.ExcelWriter(outputpath / ("hiv_tests" + datestamp + ".xlsx"))
# agg_tests.to_excel(writer, sheet_name="numbers_tests")
# referral_type_proportion.to_excel(writer, sheet_name="proportion_tests")
# writer.save()
