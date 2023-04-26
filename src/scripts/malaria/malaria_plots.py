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
# MAP
inc_MAP = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="MAP_InfectionData2023",
)
commodities_MAP = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="MAP_CommoditiesData2023",
)
treatment_MAP = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="txCov_MAPdata",
)

inc_WHO = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="WHO_CaseData2023",
)
test_WHO = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="WHO_TestData2023",
)

# WHO
WHO_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="WHO_MalReport",
)

# ------------------------------------- MULTIPLE RUN FIGURES -----------------------------------------#
# FIGURES

plt.style.use("ggplot")
plt.figure(1, figsize=(10, 10))

# Malaria incidence per 1000py - all ages with MAP model estimates
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(incMAP_data.Year, incMAP_data.inc_1000pyMean)  # MAP data
plt.fill_between(
    incMAP_data.Year,
    incMAP_data.inc_1000py_Lower,
    incMAP_data.inc_1000pyUpper,
    alpha=0.5,
)
plt.plot(WHO_data.Year, WHO_data.cases1000pyPoint)  # WHO data
plt.fill_between(
    WHO_data.Year, WHO_data.cases1000pyLower, WHO_data.cases1000pyUpper, alpha=0.5
)
plt.plot(
    model_years, inc_av, color="mediumseagreen"
)  # model - using the clinical counter for multiple episodes per person
# plt.plot(model_years, inc1.inc_clin_counter)
# plt.plot(model_years, inc2.inc_clin_counter)
# plt.plot(model_years, inc3.inc_clin_counter)
plt.title("Malaria Inc / 1000py")
plt.xlabel("Year")
plt.ylabel("Incidence (/1000py)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

# Malaria parasite prevalence rate - 2-10 year olds with MAP model estimates
# expect model estimates to be slightly higher as some will have
# undetectable parasitaemia
ax2 = plt.subplot(222)  # numrows, numcols, fignum
plt.plot(PfPRMAP_data.Year, PfPRMAP_data.PfPR_median)  # MAP data
plt.fill_between(
    PfPRMAP_data.Year, PfPRMAP_data.PfPR_LCI, PfPRMAP_data.PfPR_UCI, alpha=0.5
)
plt.plot(model_years, pfpr_av, color="mediumseagreen")  # model
plt.title("Malaria PfPR 2-10 yrs")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("PfPR (%)")
plt.gca().set_xlim(start_date, end_date)
plt.legend(["MAP", "Model"])
plt.tight_layout()

# Malaria treatment coverage - all ages with MAP model estimates
ax3 = plt.subplot(223)  # numrows, numcols, fignum
plt.plot(txMAP_data.Year, txMAP_data.ACT_coverage)  # MAP data
plt.plot(model_years, tx_av, color="mediumseagreen")  # model
plt.title("Malaria Treatment Coverage")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Treatment coverage (%)")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["MAP", "Model"])
plt.tight_layout()

# Malaria mortality rate - all ages with MAP model estimates
ax4 = plt.subplot(224)  # numrows, numcols, fignum
plt.plot(mortMAP_data.Year, mortMAP_data.mortality_rate_median)  # MAP data
plt.fill_between(
    mortMAP_data.Year,
    mortMAP_data.mortality_rate_LCI,
    mortMAP_data.mortality_rate_UCI,
    alpha=0.5,
)
plt.plot(WHO_data.Year, WHO_data.MortRatePerPersonPoint)  # WHO data
plt.fill_between(
    WHO_data.Year,
    WHO_data.MortRatePerPersonLower,
    WHO_data.MortRatePerPersonUpper,
    alpha=0.5,
)
plt.plot(model_years, mort_av, color="mediumseagreen")  # model
plt.title("Malaria Mortality Rate")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Mortality rate")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 0.0015)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

out_path = "//fi--san02/homes/tmangal/Thanzi la Onse/Malaria/model_outputs/ITN_projections_28Jan2010/"
figpath = out_path + "Baseline_averages29Jan2010" + datestamp + ".png"
plt.savefig(figpath, bbox_inches="tight")

plt.show()

plt.close()

# ########### district plots ###########################
# # get model output dates in correct format
# model_years = pd.to_datetime(prev_district.date)
# model_years = model_years.dt.year
# start_date = 2010
# end_date = 2025
#
# plt.figure(1, figsize=(30, 20))
#
# # Malaria parasite prevalence
# ax = plt.subplot(111)  # numrows, numcols, fignum
# plt.plot(model_years, prev_district.Balaka)  # model - using the clinical counter for multiple episodes per person
# plt.title("Parasite prevalence in Balaka")
# plt.xlabel("Year")
# plt.ylabel("Parasite prevalence in Balaka")
# plt.xticks(rotation=90)
# plt.gca().set_xlim(start_date, end_date)
# plt.legend(["Model"])
# plt.tight_layout()
# figpath = out_path + "district_output_seasonal" + datestamp + ".png"
#
# plt.savefig(figpath, bbox_inches='tight')
# plt.show()
# plt.close()


# ---------------------------- symptom plots ------------------------------#
# get model output dates in correct format
# model_years = pd.to_datetime(symp.date)
# model_years = model_years.dt.year
# start_date = 2010
# end_date = 2025

# plt.figure(1, figsize=(30, 20))

# # Malaria symptom prevalence
# ax = plt.subplot(221)  # numrows, numcols, fignum
# plt.bar(symp.date, symp.fever_prev, align='center', alpha=0.5)
# # plt.xticks(symp.date, symp.fever_prev)
# # plt.xticks(rotation=45)
# plt.title("Fever prevalence")
# plt.xlabel("Year")
# plt.ylabel("Fever prevalence")
# plt.tight_layout()
#
# ax = plt.subplot(222)  # numrows, numcols, fignum
# plt.bar(symp.date, symp.headache_prev, align='center', alpha=0.5)
# # plt.xticks(rotation=45)
# plt.title("Headache prevalence")
# plt.xlabel("Year")
# plt.ylabel("Headache prevalence")
# plt.tight_layout()
#
# ax = plt.subplot(223)  # numrows, numcols, fignum
# plt.bar(symp.date, symp.vomiting_prev, align='center', alpha=0.5)
# # plt.xticks(rotation=45)
# plt.title("Vomiting prevalence")
# plt.xlabel("Year")
# plt.ylabel("Vomiting prevalence")
# plt.tight_layout()
#
# ax = plt.subplot(224)  # numrows, numcols, fignum
# plt.bar(symp.date, symp.stomachache_prev, align='center', alpha=0.5)
# # plt.xticks(rotation=45)
# plt.title("Stomach ache prevalence")
# plt.xlabel("Year")
# plt.ylabel("Stomach ache prevalence")
# plt.tight_layout()
#
# plt.show()
# plt.close()
