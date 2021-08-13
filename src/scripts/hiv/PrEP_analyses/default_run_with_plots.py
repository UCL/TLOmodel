"""Run a simulation with no HSI constraints and plot the prevalence and incidence and program coverage trajectories"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import datetime
import pickle
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
)


resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2031, 1, 1)
popsize = 10000

# set up the logging file
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
        'tlo.methods.demography': logging.INFO
    }
}

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=100, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(
                 resourcefilepath=resourcefilepath,
                 service_availability=["*"],
                 mode_appt_constraints=0,
                 ignore_cons_constraints=True,
                 ignore_priority=True,
                 capabilities_coefficient=1.0,
                 disable=False,
             ),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results
with open(outputpath / 'default_run.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)

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
    # plt.show()

# ---------------------------------------------------------------------- #
# %%: DATA
# ---------------------------------------------------------------------- #

# load all the data for calibration

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / 'ResourceFile_HIV.xlsx')

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name='unaids_infections_art2021')
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids['year'], format='%Y')
data_hiv_unaids = data_hiv_unaids.drop(columns=['year'])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name='unaids_mortality_dalys2021')
data_hiv_unaids_deaths.index = pd.to_datetime(data_hiv_unaids_deaths['year'], format='%Y')
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=['year'])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name='children0_14_prev_AIDSinfo')
data_hiv_aidsinfo.index = pd.to_datetime(data_hiv_aidsinfo['year'], format='%Y')
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=['year'])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name='unaids_program_perf')
data_hiv_program.index = pd.to_datetime(data_hiv_program['year'], format='%Y')
data_hiv_program = data_hiv_program.drop(columns=['year'])

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name='MPHIA_incidence2015')
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"].values[0]
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_lower"].values[0]
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_upper"].values[0]
data_hiv_mphia_inc_yerr = [abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
                           abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate)]

data_hiv_mphia_prev = pd.read_excel(xls, sheet_name='MPHIA_prevalence_art2015')

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name='DHS_prevalence')

# MoH HIV testing data
data_hiv_moh_tests = pd.read_excel(xls, sheet_name='MoH_numbers_tests')
data_hiv_moh_tests.index = pd.to_datetime(data_hiv_moh_tests['year'], format='%Y')
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=['year'])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name='MoH_number_art')



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



# ---------------------------------------------------------------------- #
# %%: DISEASE BURDEN
# ---------------------------------------------------------------------- #

# ----------------------------- HIV -------------------------------------- #

prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# HIV - prevalence among in adults aged 15+
make_plot(
    title_str="HIV Prevalence in Adults Aged 15+ (%)",
    model=prev_and_inc_over_time['hiv_prev_adult_15plus'] * 100,
    data_mid=data_hiv_unaids['prevalence_age15plus'],
    data_low=data_hiv_unaids['prevalence_age15plus_lower'],
    data_high=data_hiv_unaids['prevalence_age15plus_upper']
)


# MPHIA
plt.plot(prev_and_inc_over_time.index[6], data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"].values[0], 'gx')

# DHS
x_values = [prev_and_inc_over_time.index[0], prev_and_inc_over_time.index[5]]
y_values = data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49"]
y_lower = abs(y_values - (data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49 lower"]))
y_upper = abs(y_values - (data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49 upper"]))
plt.errorbar(x_values, y_values,
             yerr=[y_lower, y_upper], fmt='o')
plt.ylim((0, 15))
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
green_cross = mlines.Line2D([], [], linewidth=0, color='g', marker='x',
                          markersize=7, label='MPHIA')
orange_ci = mlines.Line2D([], [], color='C1', marker='.',
                          markersize=15, label='DHS')
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])

plt.show()
plt.savefig(outputpath / ("HIV_Prevalence_Adults" + datestamp + ".pdf"), format='pdf')


# ---------------------------------------------------------------------- #

# HIV Incidence 15-49
make_plot(
    title_str="HIV Incidence in Adults (15-49) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_adult_inc_1549'] * 100,
    data_mid=data_hiv_unaids['incidence_per_1000'] / 10,
    data_low=data_hiv_unaids['incidence_per_1000_lower'] / 10,
    data_high=data_hiv_unaids['incidence_per_1000_upper'] / 10
)

# MPHIA
plt.errorbar(prev_and_inc_over_time.index[6], data_hiv_mphia_inc_estimate,
             yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]], fmt='o')

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
orange_ci = mlines.Line2D([], [], color='C1', marker='.',
                          markersize=15, label='MPHIA')
plt.legend(handles=[red_line, blue_line, orange_ci])

plt.show()
plt.savefig(outputpath / ("HIV_Incidence_Adults" + datestamp + ".pdf"), format='pdf')

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (%)",
    model=prev_and_inc_over_time['hiv_prev_child'] * 100,
    data_mid=data_hiv_aidsinfo['prevalence_0_14'] * 100,
    data_low=data_hiv_aidsinfo['prevalence_0_14_lower'] * 100,
    data_high=data_hiv_aidsinfo['prevalence_0_14_upper'] * 100
)
# MPHIA
plt.plot(prev_and_inc_over_time.index[6], data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"].values[0], 'gx')

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
green_cross = mlines.Line2D([], [], linewidth=0, color='g', marker='x',
                          markersize=7, label='MPHIA')
plt.legend(handles=[red_line, blue_line, green_cross])

plt.show()
plt.savefig(outputpath / ("HIV_Prevalence_Children" + datestamp + ".pdf"), format='pdf')


# ---------------------------------------------------------------------- #

# HIV Incidence Children
#make_plot(
#    title_str="HIV Incidence in Children (0-14) (per 100 pyar)",
#    model=prev_and_inc_over_time['hiv_child_inc'] * 100,
#    data_mid=data['inc_0_14_per1000'] / 10,
#)
#plt.show()


# ---------------------------------------------------------------------- #

# HIV prevalence among pregnant women:
make_plot(
    title_str="HIV Prevalence among Pregnant Women (%)",
    model=prev_and_inc_over_time['hiv_prev_preg'] * 100,
)
plt.show()
plt.savefig(outputpath / ("HIV_Prevalence_Pregnant_Women" + datestamp + ".pdf"), format='pdf')

# HIV prevalence among pregnant and breastfeeding women:
make_plot(
    title_str="HIV Prevalence among Pregnant and Breastfeeding Women (%)",
    model=prev_and_inc_over_time['hiv_prev_preg_and_bf'] *100,
)
plt.show()
plt.savefig(outputpath / ("HIV_Prevalence_Pregnant_and_Breastfeeding_Women" + datestamp + ".pdf"), format='pdf')


# ---------------------------------------------------------------------- #

# Incidence Pregnant Women:
make_plot(
    title_str="HIV Incidence in Pregnant Women (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_preg_inc']*100,
)
plt.show()
plt.savefig(outputpath / ("HIV_Incidence_Pregnant_Women" + datestamp + ".pdf"), format='pdf')

# Incidence Pregnant and Breastfeeding Women:
make_plot(
    title_str="HIV Incidence in Pregnant and Breastfeeding Women (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_preg_and_bf_inc'] * 100,
)
plt.show()
plt.savefig(outputpath / ("HIV_Incidence_Pregnant_and_Breastfeeding_Women" + datestamp + ".pdf"), format='pdf')
# ---------------------------------------------------------------------- #
# %%: DEATHS
# ---------------------------------------------------------------------- #

# deaths
deaths = output['tlo.methods.demography']['death'].copy()  # outputs individual deaths
deaths = deaths.set_index('date')

# AIDS DEATHS
# limit to deaths among aged 15+, include HIV/TB deaths
keep = ((deaths.age >= 15) & ((deaths.cause == 'AIDS')))
deaths_AIDS = deaths.loc[keep].copy()
deaths_AIDS['year'] = deaths_AIDS.index.year
tot_aids_deaths = deaths_AIDS.groupby(by=['year']).size()
tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format='%Y')


# aids mortality rates per 1000 person-years
total_aids_deaths_rate_1000py = (tot_aids_deaths / py) * 1000

# ---------------------------------------------------------------------- #

# AIDS deaths (including HIV/TB deaths)
make_plot(
    title_str='Mortality to HIV-AIDS per 1000 capita',
    model=total_aids_deaths_rate_1000py,
    data_mid=data_hiv_unaids_deaths['AIDS_mortality_per_1000'],
    data_low=data_hiv_unaids_deaths['AIDS_mortality_per_1000_lower'],
    data_high=data_hiv_unaids_deaths['AIDS_mortality_per_1000_upper']
)

plt.legend(['TLO', 'UNAIDS'])
plt.show()


# ---------------------------------------------------------------------- #
# %%: PROGRAM OUTPUTS
# ---------------------------------------------------------------------- #

# treatment coverage
cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
cov_over_time = cov_over_time.set_index('date')


# ---------------------------------------------------------------------- #

# Treatment Cascade ("90-90-90") Plot for Adults
dx = cov_over_time['dx_adult'] * 100
art_among_dx = (cov_over_time['art_coverage_adult'] / cov_over_time['dx_adult']) * 100
vs_among_art = (cov_over_time['art_coverage_adult_VL_suppression']) * 100

pd.concat({'diagnosed': dx,
           'art_among_diagnosed': art_among_dx,
           'vs_among_those_on_art': vs_among_art
           }, axis=1).plot()

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.title('ART Cascade for Adults (15+)')
plt.savefig(outputpath / ("HIV_art_cascade_adults" + datestamp + ".pdf"), format='pdf')

# data from UNAIDS 2021
# todo scatter the error bars
# unaids: diagnosed
x_values = data_hiv_program.index
y_values = data_hiv_program["percent_know_status"]
y_lower = abs(y_values - data_hiv_program["percent_know_status_lower"])
y_upper = abs(y_values - data_hiv_program["percent_know_status_upper"])
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], ls='none',
             marker='o', markeredgecolor='C0', markerfacecolor='C0', ecolor="C0")

# unaids: diagnosed and on art
x_values = data_hiv_program.index + pd.DateOffset(months=3)
y_values = data_hiv_program["percent_know_status_on_art"]
y_lower = abs(y_values - data_hiv_program["percent_know_status_on_art_lower"])
y_upper = abs(y_values - data_hiv_program["percent_know_status_on_art_upper"])
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], ls='none',
             marker='o', markeredgecolor='C1', markerfacecolor='C1', ecolor="C1")

# unaids: virally suppressed
x_values = data_hiv_program.index + pd.DateOffset(months=6)
y_values = data_hiv_program["percent_on_art_viral_suppr"]
y_lower = abs(y_values - data_hiv_program["percent_on_art_viral_suppr_lower"])
y_upper = abs(y_values - data_hiv_program["percent_on_art_viral_suppr_upper"])
y_values.index=x_values
y_lower.index=x_values
y_lower.index=x_values
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], ls='none',
             marker='o', markeredgecolor='g', markerfacecolor='g', ecolor="g")

plt.show()
plt.savefig(outputpath / ("HIV_Treatment_Cascade_Adults" + datestamp + ".pdf"), format='pdf')

# ---------------------------------------------------------------------- #

# Per capita testing rates - data from MoH quarterly reports
make_plot(
    title_str="Per capita testing rates for adults (15+)",
    model=cov_over_time["per_capita_testing_rate"],
    data_mid=data_hiv_moh_tests["annual_testing_rate_adults"]
)
plt.legend(['TLO', 'MoH'])

plt.show()
plt.savefig(outputpath / ("HIV_Testing_Rate" + datestamp + ".pdf"), format='pdf')


# ---------------------------------------------------------------------- #

# Percent on ART
make_plot(
    title_str="Percent of Adults (15+) on ART",
    model=cov_over_time["art_coverage_adult"] * 100,
    data_mid=data_hiv_unaids["ART_coverage_all_HIV_adults"],
    data_low=data_hiv_unaids["ART_coverage_all_HIV_adults_lower"],
    data_high=data_hiv_unaids["ART_coverage_all_HIV_adults_upper"]
)
plt.legend(['TLO', 'UNAIDS'])

plt.show()
plt.savefig(outputpath / ("HIV_Percent_on_ART" + datestamp + ".pdf"), format='pdf')

# ---------------------------------------------------------------------- #

# PrEP among Pregnant and Breastfeeding Women
make_plot(
    title_str="Proportion of Pregnant and Breastfeeding Women That Are On PrEP",
    model=cov_over_time["prop_preg_and_bf_on_prep"]
)
plt.show()
plt.savefig(outputpath / ("HIV_PrEP_Among_Pregnant_Women" + datestamp + ".pdf"), format='pdf')


