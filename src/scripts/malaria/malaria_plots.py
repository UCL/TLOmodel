from tlo.analysis.utils import parse_log_file
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# model outputs
outputpath = "./outputs/malaria/"
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

resourcefilepath = Path("./resources")

# ----------------------------------- AVERAGE OUTPUTS -----------------------------------
logfile1 = outputpath + "Malaria_Baseline1__2020_01_28.log"
output1 = parse_log_file(logfile1)

logfile2 = outputpath + "Malaria_Baseline2__2020_01_28.log"
output2 = parse_log_file(logfile2)

logfile3 = outputpath + "Malaria_Baseline3__2020_01_28.log"
output3 = parse_log_file(logfile3)

inc1 = output1["tlo.methods.malaria"]["incidence"]
pfpr1 = output1["tlo.methods.malaria"]["prevalence"]
tx1 = output1["tlo.methods.malaria"]["tx_coverage"]
mort1 = output1["tlo.methods.malaria"]["ma_mortality"]

inc2 = output2["tlo.methods.malaria"]["incidence"]
pfpr2 = output2["tlo.methods.malaria"]["prevalence"]
tx2 = output2["tlo.methods.malaria"]["tx_coverage"]
mort2 = output2["tlo.methods.malaria"]["ma_mortality"]

inc3 = output3["tlo.methods.malaria"]["incidence"]
pfpr3 = output3["tlo.methods.malaria"]["prevalence"]
tx3 = output3["tlo.methods.malaria"]["tx_coverage"]
mort3 = output3["tlo.methods.malaria"]["ma_mortality"]

# take average of incidence clinical counter
inc_av = np.mean(
    [inc1.inc_clin_counter, inc2.inc_clin_counter, inc3.inc_clin_counter], axis=0
)
pfpr_av = np.mean(
    [pfpr1.child2_10_prev, pfpr2.child2_10_prev, pfpr3.child2_10_prev], axis=0
)
tx_av = np.mean(
    [tx1.treatment_coverage, tx2.treatment_coverage, tx3.treatment_coverage], axis=0
)
mort_av = np.mean([mort2.mort_rate, mort2.mort_rate, mort2.mort_rate], axis=0)

# ----------------------------------- SAVE OUTPUTS -----------------------------------
# out_path = '//fi--san02/homes/tmangal/Thanzi la Onse/Malaria/model_outputs/'
#
# if malaria_strat == 0:
#     savepath = out_path + "national_output_" + datestamp + ".xlsx"
# else:
#     savepath = out_path + "district_output_" + datestamp + ".xlsx"
#
# writer = pd.ExcelWriter(savepath, engine='xlsxwriter')
#
# inc_df = pd.DataFrame(inc)
# inc_df.to_excel(writer, sheet_name='inc')
#
# pfpr_df = pd.DataFrame(pfpr)
# pfpr_df.to_excel(writer, sheet_name='pfpr')
#
# tx_df = pd.DataFrame(tx)
# tx_df.to_excel(writer, sheet_name='tx')
#
# mort_df = pd.DataFrame(mort)
# mort_df.to_excel(writer, sheet_name='mort')
#
# # symp_df = pd.DataFrame(symp)
# # symp_df.to_excel(writer, sheet_name='symp')
#
# prev_district_df = pd.DataFrame(prev_district)
# prev_district_df.to_excel(writer, sheet_name='prev_district')
#
# writer.save()

# ----------------------------------- CREATE PLOTS-----------------------------------

# get model output dates in correct format
model_years = pd.to_datetime(inc1.date)
model_years = model_years.dt.year
start_date = 2010
end_date = 2025

# import malaria data
# MAP
incMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="inc1000py_MAPdata",
)
PfPRMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="PfPR_MAPdata",
)
mortMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="mortalityRate_MAPdata",
)
txMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="txCov_MAPdata",
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
