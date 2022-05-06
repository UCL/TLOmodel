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


start_date = 2010
end_date = 2020

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
)
plt.show()

# # ---------------------------------------------------------------------- #

# latent TB prevalence
latentTB_prev = output["tlo.methods.tb"]["tb_prevalence"]
latentTB_prev = latentTB_prev.set_index("date")

title_str = "Latent TB prevalence"
make_plot(
    title_str=title_str,
    model=latentTB_prev["tbPrevLatent"],
)
plt.ylim((0, 1.0))
plt.ylabel("Prevalence")

plt.show()


# # ---------------------------------------------------------------------- #

# latent TB incidence

# remove baseline prevalence
latentTB_inc = TB_inc["num_new_latent_tb"]
latentTB_inc = latentTB_inc.iloc[1:]

title_str = "Number new latent cases"
make_plot(
    title_str=title_str,
    model=latentTB_inc,
)
plt.ylabel("Number new latent cases")
plt.show()

# ---------------------------------------------------------------------- #

# latent TB incidence per 100,000 person-years - annual outputs
latentTB_inc = TB_inc["num_new_latent_tb"]
latentTB_inc = latentTB_inc.iloc[1:]
latentTB_inc_rate = (latentTB_inc / py) * 100000

make_plot(
    title_str="Latent TB Incidence (per 100k person-years)",
    model=latentTB_inc_rate,
)
plt.show()

# ---------------------------------------------------------------------- #

# TB treatment coverage

# treatment coverage
Tb_tx_coverage = output["tlo.methods.tb"]["tb_treatment"]
Tb_tx_coverage = Tb_tx_coverage.set_index("date")
Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)


make_plot(
    title_str="Percent of TB cases treated",
    model=Tb_tx_coverage["tbTreatmentCoverage"] * 100,
)
plt.ylim((0, 100))

plt.legend(["TLO", "NTP"])
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
plt.show()


# ---------------------------------------------------------------------- #

# proportion active TB cases that are smear-positive
tb_smear = output["tlo.methods.tb"]["tb_prevalence"]
tb_smear = tb_smear.set_index("date")

title_str = "Proportion of active cases that are smear-positive"
make_plot(
    title_str=title_str,
    model=tb_smear["tbPropSmearPositive"],
)
plt.show()


# ----------------------------- BCG -------------------------------------- #

model_vax_coverage = output["tlo.methods.epi"]["ep_vaccine_coverage"]
model_date = pd.to_datetime(model_vax_coverage.date)
model_date = model_date.apply(lambda x: x.year)

# BCG coverage
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(model_date, model_vax_coverage.epBcgCoverage)
plt.title("BCG vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_ylim(0, 110)




###

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
)
plt.ylim((0, 15))
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence 15-49
title_str = "HIV Incidence in Adults (15-49) (per 100 pyar)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_adult_inc_1549"] * 100,
)
plt.show()

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
title_str = "HIV Prevalence in Children (0-14) (%)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_prev_child"] * 100,
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence Children
title_str = "HIV Incidence in Children (0-14) (per 100 pyar)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_child_inc"] * 100,
)
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
    title_str="Mortality to HIV-AIDS per 1000 capita",
    model=total_aids_deaths_rate_100kpy,
)

plt.show()


# ---------------------------------------------------------------------- #

# TB deaths (excluding HIV/TB deaths)
make_plot(
    title_str="TB mortality rate (excl HIV) per 100,000 population",
    model=tot_tb_non_hiv_deaths_rate_100kpy,
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV/TB deaths only
make_plot(
    title_str="TB_HIV mortality rate per 100,000 population",
    model=tot_tb_hiv_deaths_rate_100kpy,
)
plt.show()



# ---------------------------------------------------------------------- #
# %%: PROGRAM OUTPUTS
# ---------------------------------------------------------------------- #

cov_over_time = output["tlo.methods.hiv"]["hiv_program_coverage"]
cov_over_time = cov_over_time.set_index("date")

# ---------------------------------------------------------------------- #

# HIV Treatment Cascade ("90-90-90") Plot for Adults
dx = cov_over_time["dx_adult"] * 100
art_among_dx = (cov_over_time["art_coverage_adult"] / cov_over_time["dx_adult"]) * 100
vs_among_art = (cov_over_time["art_coverage_adult_VL_suppression"]) * 100

# ---------------------------------------------------------------------- #


# ---------------------------------------------------------------------- #

# Percent of all HIV+ on ART
make_plot(
    title_str="Percent of all Adults (15+) HIV+ on ART",
    model=cov_over_time["art_coverage_adult"] * 100,
)
plt.show()

# ---------------------------------------------------------------------- #

# Circumcision
make_plot(
    title_str="Proportion of Men (15+) That Are Circumcised",
    model=cov_over_time["prop_men_circ"],
)
plt.ylim((0, 0.4))
plt.show()


# ---------------------------------------------------------------------- #

# PrEP among FSW
make_plot(
    title_str="Proportion of FSW That Are On PrEP",
    model=cov_over_time["prop_fsw_on_prep"],
)
plt.show()


