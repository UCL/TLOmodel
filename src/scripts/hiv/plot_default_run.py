"""Run a simulation with no HSI and plot the prevalence and incidence and program coverage trajectories"""
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

import pickle

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 10000

# Establish the simulation object
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
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
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



# %% Get ready to create plots:

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)


# load the calibration data
data = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='calibration_from_aids_info')
data.index = pd.to_datetime(data['year'], format='%Y')
data = data.drop(columns=['year'])

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



# %% : PREVALENCE AND INCIDENCE PLOTS

# adults - prevalence among 15-49 year-olds
prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# Prevalence 15-49
make_plot(
    title_str="HIV Prevalence in Adults (15-49) (%)",
    model=prev_and_inc_over_time['hiv_prev_adult_1549'] * 100,
    data_mid=data['prev_15_49'],
    data_low=data['prev_15_49_lower'],
    data_high=data['prev_15_49_upper']
)

# Incidence 15-49
make_plot(
    title_str="HIV Incidence in Adults (15-49) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_adult_inc_1549'] * 100,
    data_mid=data['inc_15_49_per1000'] / 10,
    data_low=data['inc_15_49_per1000lower'] / 10,
    data_high=data['inc_15_49_per1000upper'] / 10
)

# Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (%)",
    model=prev_and_inc_over_time['hiv_prev_child'] * 100,
)

# Incidence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_child_inc'] * 100,
)

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=prev_and_inc_over_time['hiv_prev_fsw'] * 100,
)


# %% PROGRAM COVERAGE PLOTS
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
