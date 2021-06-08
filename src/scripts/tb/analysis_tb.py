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
    hiv,
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
end_date = Date(2015, 1, 1)
popsize = 15000

# Establish the simulation object
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
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             epi.Epi(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath),
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
    plt.savefig(outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')
    plt.show()


# ------------------------- OUTPUTS ------------------------- #
# Active TB incidence
activeTB_inc = output['tlo.methods.tb']['tb_incidence']
activeTB_inc = activeTB_inc.set_index('date')

# latent TB prevalence
latentTB_prev = output['tlo.methods.tb']['tb_prevalence']
latentTB_prev = latentTB_prev.set_index('date')

# deaths
deaths = output['tlo.methods.demography']['death'].copy()  # outputs individual deaths
deaths = deaths.set_index('date')

# TB deaths will exclude TB/HIV
to_drop = (deaths.cause != 'TB')
deaths_TB = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB['year'] = deaths_TB.index.year  # count by year
tot_tb_non_hiv_deaths = deaths_TB.groupby(by=['year']).size()

# TB/HIV deaths
to_drop = (deaths.cause != 'AIDS_non_TB')
deaths_TB_HIV = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB_HIV['year'] = deaths_TB_HIV.index.year  # count by year
tot_tb_hiv_deaths = deaths_TB_HIV.groupby(by=['year']).size()

# total TB deaths (including HIV+)
total_tb_deaths = tot_tb_non_hiv_deaths.add(tot_tb_hiv_deaths, fill_value=0)

# ------------------------- PLOTS ------------------------- #

# plot active tb incidence per 1000 population
make_plot(
    title_str="Active TB Incidence (per 1000 person-years)",
    model=activeTB_inc['tbIncActive100k'],
)
# plot latent prevalence
make_plot(
    title_str="Latent TB prevalence",
    model=latentTB_prev['tbPrevLatent'],
)


# plot proportion of active tb cases on treatment

# plot number tb-mdr

# plot numbers of sputum tests / xpert tests per month

# plot ipt for HIV+ and contacts of TB cases

# plot by district

