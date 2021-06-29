"""Run a simulation with no HSI constraints and plot the prevalence and incidence and program coverage trajectories"""
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    simplified_births,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    epi,
    tb
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 100

# set up the log config
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.tb': logging.INFO,
        'tlo.methods.demography': logging.INFO
    }
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
sim = Simulation(start_date=start_date, seed=100, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(
                 resourcefilepath=resourcefilepath,
                 disable=True,
             ignore_cons_constraints=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             epi.Epi(resourcefilepath=resourcefilepath),
             tb.Tb(resourcefilepath=resourcefilepath),
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


# ---------------------------------------------------------------------- #
# PLOTS
# ---------------------------------------------------------------------- #

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


# ------------------------- OUTPUTS ------------------------- #
#
# # load the calibration data
# data = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_estimates')
# data.index = pd.to_datetime(data['year'], format='%Y')
# data = data.drop(columns=['year'])
#
# # person-years all ages (irrespective of HIV status)
# py_ = output['tlo.methods.demography']['person_years']
# years = pd.to_datetime(py_['date']).dt.year
# py = pd.Series(dtype='int64', index=years)
# for year in years:
#     tot_py = (
#         (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
#         (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
#     ).transpose()
#     py[year] = tot_py.sum().values[0]
#
# py.index = pd.to_datetime(years, format='%Y')
#
# # Active TB incidence - annual outputs
# TB_inc = output['tlo.methods.tb']['tb_incidence']
# TB_inc = TB_inc.set_index('date')
# TB_inc.index = pd.to_datetime(TB_inc.index)
# activeTB_inc_rate = (TB_inc['num_new_active_tb'] / py) * 100000
#
#
# # latent TB prevalence
# latentTB_prev = output['tlo.methods.tb']['tb_prevalence']
# latentTB_prev = latentTB_prev.set_index('date')
#
# # proportion TB cases that are MDR
# mdr = output['tlo.methods.tb']['tb_mdr']
# mdr = mdr.set_index('date')
#
# # deaths
# deaths = output['tlo.methods.demography']['death'].copy()  # outputs individual deaths
# deaths = deaths.set_index('date')
#
# # TB deaths will exclude TB/HIV
# to_drop = (deaths.cause != 'TB')
# deaths_TB = deaths.drop(index=to_drop[to_drop].index).copy()
# deaths_TB['year'] = deaths_TB.index.year  # count by year
# tot_tb_non_hiv_deaths = deaths_TB.groupby(by=['year']).size()
# tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format='%Y')
#
# # TB/HIV deaths
# to_drop = (deaths.cause != 'AIDS_non_TB')
# deaths_TB_HIV = deaths.drop(index=to_drop[to_drop].index).copy()
# deaths_TB_HIV['year'] = deaths_TB_HIV.index.year  # count by year
# tot_tb_hiv_deaths = deaths_TB_HIV.groupby(by=['year']).size()
# tot_tb_hiv_deaths.index = pd.to_datetime(tot_tb_hiv_deaths.index, format='%Y')
#
# # total TB deaths (including HIV+)
# total_tb_deaths = tot_tb_non_hiv_deaths.add(tot_tb_hiv_deaths, fill_value=0)
# total_tb_deaths.index = pd.to_datetime(total_tb_deaths.index, format='%Y')
#
# # mortality rates per 100k person-years
# total_tb_deaths_rate = (total_tb_deaths / py) * 100000
#
# tot_tb_hiv_deaths_rate = (tot_tb_hiv_deaths / py) * 100000
#
# tot_tb_non_hiv_deaths_rate = (tot_tb_non_hiv_deaths / py) * 100000
#
# # treatment coverage
# Tb_tx_coverage = output['tlo.methods.tb']['tb_treatment']
# Tb_tx_coverage = Tb_tx_coverage.set_index('date')
# Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)
#
#
# # ------------------------- PLOTS ------------------------- #
#
# # plot active tb incidence per 100k person-years
# make_plot(
#     title_str="Active TB Incidence (per 100k person-years)",
#     model=activeTB_inc_rate,
#     data_mid=data['incidence_per_100k'],
#     data_low=data['incidence_per_100k_low'],
#     data_high=data['incidence_per_100k_high']
# )
#
# # plot latent prevalence
# make_plot(
#     title_str="Latent TB prevalence",
#     model=latentTB_prev['tbPrevLatent'],
# )
#
# # plot tb (non-hiv) deaths per 100k person-years
# make_plot(
#     title_str="Mortality TB (excl HIV) per 100k py",
#     model=tot_tb_non_hiv_deaths_rate,
#     data_mid=data['mortality_tb_excl_hiv_per_100k'],
#     data_low=data['mortality_tb_excl_hiv_per_100k_low'],
#     data_high=data['mortality_tb_excl_hiv_per_100k_high']
# )
#
# # plot tb deaths per 100k person-years in PLHIV
# make_plot(
#     title_str="Mortality TB_HIV per 100k py",
#     model=tot_tb_hiv_deaths_rate,
#     data_mid=data['mortality_tb_hiv_per_100k'],
#     data_low=data['mortality_tb_hiv_per_100k_low'],
#     data_high=data['mortality_tb_hiv_per_100k_high']
# )
#
# # plot total tb deaths
# make_plot(
#     title_str="Mortality TB (all incl HIV) per 100k",
#     model=total_tb_deaths_rate,
#     data_mid=data['total_mortality_tb_per_100k'],
#     data_low=data['total_mortality_tb_per_100k_low'],
#     data_high=data['total_mortality_tb_per_100k_high']
# )
#
# # plot proportion of active tb cases on treatment
# make_plot(
#     title_str="TB treatment coverage",
#     model=Tb_tx_coverage['tbTreatmentCoverage'],
#     data_mid=data['TB_program_tx_coverage'],
# )
#
# # plot proportion of active tb cases that are tb-mdr
# # expect <1%
# make_plot(
#     title_str="Proportion TB cases that are MDR",
#     model=mdr['tbPropActiveCasesMdr'],
# )

# plot numbers of sputum tests / xpert tests per month

# plot ipt for HIV+ and contacts of TB cases

# plot by district
