""" load the outputs from a simulation and plot the results with comparison data """

import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# ----------------------------------- CREATE PLOTS-----------------------------------

# import malaria data
# MAP
incMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx",
    sheet_name="MAP_InfectionData2023",
)
txMAP_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="txCov_MAPdata",
)

# WHO
WHO_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="WHO_CaseData2023",
)

# MAP mortality
MAP_mort = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="mortalityRate_MAPdata",
)

# MAP commodities
MAP_comm = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="MAP_CommoditiesData2023",
)

# WHO commodities
WHO_comm = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="WHO_TestData2023",
)

# NMCP rdt data
NMCP_comm = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_malaria.xlsx", sheet_name="NMCP",
)

# ---------------------------------------------------------------------- #
# %%: MODEL OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / "malaria_run.pickle", "rb") as f:
    output = pickle.load(f)

inc = output["tlo.methods.malaria"]["incidence"]
pfpr = output["tlo.methods.malaria"]["prevalence"]
tx = output["tlo.methods.malaria"]["tx_coverage"]
coinfection = output["tlo.methods.malaria"]["coinfection_prevalence"]

scaling_factor = output["tlo.methods.population"]["scaling_factor"].values[0][1]

# numbers rdt requested
cons = output["tlo.methods.healthsystem.summary"]["Consumables"]
rdt_item_code = '163'

rdt_usage_model = []
# extract item rdt values for each year
for row in range(cons.shape[0]):
    cons_dict = cons.at[row, 'Item_Available']
    rdt_usage_model.append(cons_dict.get(rdt_item_code))

scaled_rdt_usage_model = [i * scaling_factor for i in rdt_usage_model]

# Malaria RDT yield
# both datasets from 2010
MAP_rdt_yield = (MAP_comm.Tested_Positive_Public / MAP_comm.RDT_Consumption_Public) * 100
WHO_rdt_yield = (WHO_comm.NumPositiveCasesTestedByRDT / WHO_comm.NumSuspectedCasesTestedByRDT) * 100
# model rdt yield
model_yield = (rdt_usage_model[:-1] / inc.number_new_cases) * 100

# get model output dates in correct format
model_years = pd.to_datetime(inc.date)
model_years = model_years.dt.year

years_of_simulation = len(model_years)
# ------------------------------------- FIGURES -----------------------------------------#
start_date = 2010
end_date = model_years.values[-1]

# FIGURES
plt.style.use("ggplot")
plt.figure(1, figsize=(10, 10))

# Malaria incidence per 1000py - all ages with MAP model estimates
ax = plt.subplot(221)  # numrows, numcols, fignum
plt.plot(incMAP_data.Year, incMAP_data.IncidenceRatePer1000, color="crimson")  # MAP data
plt.plot(WHO_data.Year, WHO_data.IncidencePer1000, color="darkorchid")  # WHO data
plt.fill_between(
    WHO_data.Year, WHO_data.IncidencePer1000Low, WHO_data.IncidencePer1000High,
    color="darkorchid", alpha=0.1
)
plt.plot(
    model_years, inc.inc_1000py, color="mediumseagreen"
)  # model - using the clinical counter for multiple episodes per person
plt.title("Malaria Inc / 1000py")
plt.xlabel("Year")
plt.ylabel("Incidence (/1000py)")
plt.gca().set_ylim(0.0, 500)
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

# Malaria treatment coverage - all ages with MAP model estimates
ax2 = plt.subplot(222)  # numrows, numcols, fignum
plt.plot(txMAP_data.Year, txMAP_data.ACT_coverage, color="crimson")  # MAP data
plt.plot(model_years, tx.treatment_coverage, color="mediumseagreen")  # model
plt.title("Malaria Treatment Coverage")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Treatment coverage (%)")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 1.0)
plt.legend(["MAP", "Model"])
plt.tight_layout()

# Malaria rdt usage
ax3 = plt.subplot(223)  # numrows, numcols, fignum
plt.plot(MAP_comm.Year, MAP_comm.RDT_Consumption_Public, color="crimson")  # MAP data
plt.plot(WHO_comm.Year, WHO_comm.NumSuspectedCasesTestedByRDT, color="darkorchid")  # WHO data
plt.plot(NMCP_comm.Year, NMCP_comm.NMCP_RDTs_Qty_Dispersed, color="blue")  # WHO data
plt.plot(model_years, scaled_rdt_usage_model[:-1], color="mediumseagreen")  # model
plt.title("RDT usage")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Numbers of RDTs used")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 4.0e7)
plt.legend(labels=["MAP", "WHO", "NMCP", "Model"], )
plt.tight_layout()

# malaria rdt yield
# ax4 = plt.subplot(224)  # numrows, numcols, fignum
# plt.plot(MAP_comm.Year, MAP_rdt_yield, color="crimson")  # MAP data
# plt.plot(WHO_comm.Year, WHO_rdt_yield, color="darkorchid")  # WHO data
# plt.plot(model_years, model_yield, color="mediumseagreen")  # model
# plt.title("RDT yield (positive / suspected)")
# plt.xlabel("Year")
# plt.xticks(rotation=90)
# plt.ylabel("RDT yield (%)")
# plt.gca().set_xlim(start_date, end_date)
# plt.gca().set_ylim(0.0, 100)
# plt.legend(["MAP (Public)", "WHO", "Model"])
# plt.tight_layout()

plt.show()

plt.close()

# ------------------------------------- FIGURES - co-infection -----------------------------------------#

plt.style.use("ggplot")
plt.figure(1, figsize=(10, 5))

# malaria prevalence in HIV population
# model value is whole population, Obebe is pooled prev, Mirz is adults across all years
ax = plt.subplot(121)  # numrows, numcols, fignum
plt.plot(model_years, coinfection.prev_malaria_in_hiv_population * 100, color="blue")
plt.errorbar(
    model_years[1],
    16.5,
    yerr=[[10.6], [21.9]],
    fmt="o",
)
plt.axhline(y=27.34, color='g', linestyle='-')
plt.title("malaria prevalence in HIV population")
plt.xlabel("Year")
plt.ylabel("Prevalence")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["Model", "Obebe, 2021", "Mirzohreh, 2022"])
plt.tight_layout()

# Proportion of malaria cases with HIV
ax2 = plt.subplot(122)  # numrows, numcols, fignum
plt.plot(model_years, coinfection.prop_malaria_cases_with_hiv * 100, color="mediumseagreen")  # model
plt.title("Prop malaria cases with HIV")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Percent")
plt.gca().set_xlim(start_date, end_date)
plt.gca().set_ylim(0.0, 10.0)
plt.legend(["Model"])
plt.tight_layout()

plt.show()

plt.close()

# ------------------------------------- plot rdt delivery -----------------------------------------#
rdt_facilities = output["tlo.methods.malaria"]["rdt_log"]

# limit to first tests only?
# rdt_all = rdt_facilities
rdt_all = rdt_facilities.loc[~(rdt_facilities.called_by == 'Malaria_Treatment')]

# todo exclude people having repeat diagnostic tests (through malaria module and hsi_generic)
# just exclude for this plot - they can occur in reality
# could exclude tests in same person occurring within 1-4 days

# limit to children <5 yrs with fever
rdt_child = rdt_facilities.loc[(rdt_facilities.age <= 5) & rdt_facilities.fever_present]
# remove tests given for confirmation with treatment
rdt_child = rdt_child.loc[~(rdt_child.called_by == 'Malaria_Treatment')]

colours = ['#B7C3F3', '#DD7596', '#FFF68F']

plt.rcParams["axes.titlesize"] = 9

ax = plt.subplot(121)  # numrows, numcols, fignum
# calculate proportion of rdt delivered by facility level
level0 = rdt_all['facility_level'].value_counts()['0'] / len(rdt_all)
level1a = rdt_all['facility_level'].value_counts()['1a'] / len(rdt_all)
# level1b = rdt_all['facility_level'].value_counts()['1b'] / len(rdt_all)
level2 = rdt_all['facility_level'].value_counts()['2'] / len(rdt_all)

plt.pie([level0, level1a, level2], labels=['level 0', 'level 1a', 'level 2'],
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        autopct='%.1f%%',
        colors=colours)
plt.title("Facility level giving first rdt \n all ages")

ax2 = plt.subplot(122)  # numrows, numcols, fignum
# calculate proportion of rdt delivered by facility level - children with fever
level0 = rdt_child['facility_level'].value_counts()['0'] / len(rdt_child)
level1a = rdt_child['facility_level'].value_counts()['1a'] / len(rdt_child)
# level1b = rdt_child['facility_level'].value_counts()['1b'] / len(rdt_child)
level2 = rdt_child['facility_level'].value_counts()['2'] / len(rdt_child)

plt.pie([level0, level1a, level2], labels=['level 0', 'level 1a', 'level 2'],
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        autopct='%.1f%%',
        colors=colours)
plt.title("Facility level giving first rdt  \n children with fever")
plt.tight_layout()

plt.show()

# ------------------------------------- plot malaria deaths -----------------------------------------#

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

# deaths
deaths = output["tlo.methods.demography"]["death"].copy()  # outputs individual deaths
deaths = deaths.set_index("date")

# Malaria deaths
# keep if cause = malaria
keep = (deaths.cause == "Malaria")
deaths = deaths.loc[keep].copy()
deaths["year"] = deaths.index.year  # count by year
deaths_yr = deaths.groupby(by=["year"]).size()
deaths_yr.index = pd.to_datetime(deaths_yr.index, format="%Y")

# malaria mortality rates per 100k person-years
deaths_rate_100kpy = (deaths_yr / py) * 100_000

# # ---------------------------------------------------------------------- #
# Malaria deaths

x_labels = pd.to_datetime(deaths_rate_100kpy.index)
x_labels = x_labels.year

plt.style.use("ggplot")
plt.figure(1, figsize=(10, 10))

# Malaria deaths per 100,000py - all ages
ax = plt.subplot(111)  # numrows, numcols, fignum
plt.plot(MAP_mort.Year, MAP_mort.mortality_rate_median * 100_000, color="crimson")  # MAP data
plt.fill_between(
    MAP_mort.Year, MAP_mort.mortality_rate_LCI * 100_000, MAP_mort.mortality_rate_UCI * 100_000,
 color="crimson", alpha=0.5
)
plt.plot(WHO_data.Year, WHO_data.MortalityRatePer100_000, color="seagreen")  # WHO data
plt.fill_between(
    WHO_data.Year, WHO_data.MortalityRatePer100_000Low, WHO_data.MortalityRatePer100_000_High,
 color="seagreen", alpha=0.5
)
plt.plot(
    x_labels, deaths_rate_100kpy, color="blue")  # model
plt.title("Malaria deaths /100_000py")
plt.xlabel("Year")
plt.ylabel("Deaths (/100_000py)")
plt.gca().set_ylim(0.0, 200)
plt.xticks(rotation=90)
# plt.gca().set_xlim(start_date, end_date)
plt.legend(["MAP", "WHO", "Model"])
plt.tight_layout()

plt.show()
