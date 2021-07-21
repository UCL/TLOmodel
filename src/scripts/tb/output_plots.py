""" load the outputs from a simulation and plot the results with comparison data """

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
from pathlib import Path

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Function to make standard plot to compare model and data
def make_plot(
    model=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    title_str=None
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, '-', color='r')

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, '-')
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index,
                        data_low,
                        data_high,
                        alpha=0.2)
    plt.title(title_str)
    plt.legend(['Model', 'Data'])
    plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')
    plt.show()


# ---------------------------------------------------------------------- #
# %%: DATA
# ---------------------------------------------------------------------- #
start_date = 2010
end_date = 2020

# load all the data for calibration

# TB WHO data
data_tb_who = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_tb_who.index = pd.to_datetime(data_tb_who['year'], format='%Y')
data_tb_who = data_tb_who.drop(columns=['year'])

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='unaids_infections_art2021')
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids['year'], format='%Y')
data_hiv_unaids = data_hiv_unaids.drop(columns=['year'])

# MPHIA HIV data
data_hiv_mphia_inc = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='MPHIA_incidence2015')

data_hiv_mphia_prev = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='MPHIA_prevalence_art2015')

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='DHS_prevalence')


# ---------------------------------------------------------------------- #
# %%: OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)

# person-years all ages (irrespective of HIV status)
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.Series(dtype='int64', index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]

py.index = pd.to_datetime(years, format='%Y')

# ------------------------- DISEASE BURDEN ------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs
TB_inc = output['tlo.methods.tb']['tb_incidence']
TB_inc = TB_inc.set_index('date')
TB_inc.index = pd.to_datetime(TB_inc.index)
activeTB_inc_rate = (TB_inc['num_new_active_tb'] / py) * 100000

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate,
    data_mid=data_tb_who['incidence_per_100k'],
    data_low=data_tb_who['incidence_per_100k_low'],
    data_high=data_tb_who['incidence_per_100k_high']
)

# latent TB prevalence
latentTB_prev = output['tlo.methods.tb']['tb_prevalence']
latentTB_prev = latentTB_prev.set_index('date')

make_plot(
    title_str="Latent TB prevalence",
    model=latentTB_prev['tbPrevLatent'],
)

# proportion TB cases that are MDR
mdr = output['tlo.methods.tb']['tb_mdr']
mdr = mdr.set_index('date')

# HIV - prevalence among 15-49 year-olds
prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# todo add mphia and dhs
plt.style.use("ggplot")
plt.plot(prev_and_inc_over_time['hiv_prev_adult_1549'] * 100)  # model outputs
plt.fill_between(
    data_hiv_unaids.year,
    data_hiv_unaids['prevalence_age15plus_lower'],
    data_hiv_unaids['prevalence_age15plus_upper'],
    alpha=0.5, color="mediumseagreen",
)
plt.plot(data_hiv_unaids.year, data_hiv_unaids['prevalence_age15plus'], color="mediumseagreen")  # UNAIDS
plt.plot(2015, data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"], "x")  # MPHIA


plt.scatter(data_hiv_dhs_prev["Year"],
            data_hiv_dhs_prev["HIV prevalence among general population 15-49"], c='red')  # DHS

plt.errorbar(data_hiv_dhs_prev["Year"], data_hiv_dhs_prev["HIV prevalence among general population 15-49"],
             yerr=[data_hiv_dhs_prev["HIV prevalence among general population 15-49 lower"],
                   data_hiv_dhs_prev["HIV prevalence among general population 15-49 upper"]], linestyle='')


plt.title("HIV Prevalence in Adults (15-49) (%)")
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")
plt.xticks(rotation=90)
plt.gca().set_xlim(start_date, end_date)
plt.legend(["TLO", "UNAIDS", "MPHIA", "DHS"])
plt.show()



# HIV Incidence 15-49
make_plot(
    title_str="HIV Incidence in Adults (15-49) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_adult_inc_1549'] * 100,
    data_mid=data_hiv_unaids['inc_15_49_per1000'] / 10,
    data_low=data_hiv_unaids['inc_15_49_per1000lower'] / 10,
    data_high=data_hiv_unaids['inc_15_49_per1000upper'] / 10
)

# HIV Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (%)",
    model=prev_and_inc_over_time['hiv_prev_child'] * 100,
)

# HIV Incidence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_child_inc'] * 100,
)

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=prev_and_inc_over_time['hiv_prev_fsw'] * 100,
)


# ------------------------- DEATHS ------------------------- #

# deaths
deaths = output['tlo.methods.demography']['death'].copy()  # outputs individual deaths
deaths = deaths.set_index('date')

# TB deaths will exclude TB/HIV
to_drop = (deaths.cause != 'TB')
deaths_TB = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB['year'] = deaths_TB.index.year  # count by year
tot_tb_non_hiv_deaths = deaths_TB.groupby(by=['year']).size()
tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format='%Y')

# TB/HIV deaths
to_drop = (deaths.cause != 'AIDS_non_TB')
deaths_TB_HIV = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB_HIV['year'] = deaths_TB_HIV.index.year  # count by year
tot_tb_hiv_deaths = deaths_TB_HIV.groupby(by=['year']).size()
tot_tb_hiv_deaths.index = pd.to_datetime(tot_tb_hiv_deaths.index, format='%Y')

# total TB deaths (including HIV+)
total_tb_deaths = tot_tb_non_hiv_deaths.add(tot_tb_hiv_deaths, fill_value=0)
total_tb_deaths.index = pd.to_datetime(total_tb_deaths.index, format='%Y')

# mortality rates per 100k person-years
total_tb_deaths_rate = (total_tb_deaths / py) * 100000

tot_tb_hiv_deaths_rate = (tot_tb_hiv_deaths / py) * 100000

tot_tb_non_hiv_deaths_rate = (tot_tb_non_hiv_deaths / py) * 100000

# AIDS DEATHS
deaths = output['tlo.methods.demography']['death'].copy()
deaths = deaths.set_index('date')
# limit to deaths among aged 15+
to_drop = ((deaths.age < 15) | (deaths.cause != 'AIDS'))
deaths = deaths.drop(index=to_drop[to_drop].index)

# count by year:
deaths['year'] = deaths.index.year
tot_aids_deaths = deaths.groupby(by=['year']).size()

# plots


# ------------------------- PROGRAM OUTPUTS ------------------------- #

# treatment coverage
Tb_tx_coverage = output['tlo.methods.tb']['tb_treatment']
Tb_tx_coverage = Tb_tx_coverage.set_index('date')
Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)

cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
cov_over_time = cov_over_time.set_index('date')

# Treatment Cascade ("90-90-90") Plot for Adults
dx = cov_over_time['dx_adult']
art_among_dx = cov_over_time['art_coverage_adult'] / dx
vs_among_art = cov_over_time['art_coverage_adult_VL_suppression']
pd.concat({'diagnosed': dx,
           'art_among_diagnosed': art_among_dx,
           'vs_among_those_on_art': vs_among_art
           }, axis=1).plot()
plt.title('ART Cascade for Adults (15+)')
plt.savefig(outputpath / ("HIV_art_cascade_adults" + datestamp + ".pdf"), format='pdf')
plt.show()

# Per capita testing rates - data from MoH quarterly reports
make_plot(
    title_str="Per capita testing rates for adults (15+)",
    model=cov_over_time["per_capita_testing_rate"],
    data_mid=data["adult_tests_per_capita"]
)

# Percent on ART
make_plot(
    title_str="Percent of Adults (15+) on ART",
    model=cov_over_time["art_coverage_adult"] * 100,
    data_mid=data["percent15plus_on_art"],
    data_low=data["percent15plus_on_art_lower"],
    data_high=data["percent15plus_on_art_upper"]
)

# Circumcision
make_plot(
    title_str="Proportion of Men (15+) That Are Circumcised",
    model=cov_over_time["prop_men_circ"]
)

# PrEP among FSW
make_plot(
    title_str="Proportion of FSW That Are On PrEP",
    model=cov_over_time["prop_fsw_on_prep"]
)

# Behaviour Change
make_plot(
    title_str="Proportion of Adults (15+) Exposed to Behaviour Change Intervention",
    model=cov_over_time["prop_adults_exposed_to_behav_intv"]
)


