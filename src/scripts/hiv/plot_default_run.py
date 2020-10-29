"""Run a simulation with no HSI and plot the prevalence and incidence and program coverage trajectories"""
import datetime
from pathlib import Path

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

import matplotlib.pyplot as plt
from tlo.methods.hiv import unpack_raw_output_dict, map_to_age_group

from tlo.util import create_age_range_lookup

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

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
    }
}

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
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


# %% Create plots:

# read the results
output = parse_log_file(sim.log_filepath)

#%% : BASELINE HIV PREVALENCE
# adults:
adult_prev_and_inc_over_time = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw'][['date', 'hiv_prev_adult', 'hiv_adult_inc']]
adult_prev_and_inc_over_time = adult_prev_and_inc_over_time.set_index('date')
adult_prev_and_inc_over_time.plot()
plt.title('HIV Prevalence and Incidence in Adults (15+)')
plt.savefig(outputpath / ("HIV_adult_prev_and_inc_over_time" + datestamp + ".pdf"), format='pdf')
plt.show()

# children:
child_prev_and_inc_over_time = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw'][['date', 'hiv_prev_child', 'hiv_child_inc']]
child_prev_and_inc_over_time = child_prev_and_inc_over_time.set_index('date')
child_prev_and_inc_over_time.plot()
plt.title('HIV Prevalence and Incidence in Children (0-14)')
plt.savefig(outputpath / ("HIV_children_prev_and_inc_over_time" + datestamp + ".pdf"), format='pdf')
plt.show()

# female sex workers:
fsw_prev_over_time = output['tlo.methods.hiv']['summary_inc_and_prev_for_adults_and_children_and_fsw'][['date', 'hiv_prev_fsw']]
fsw_prev_over_time = fsw_prev_over_time.set_index('date')
fsw_prev_over_time.plot()
plt.title('HIV Prevalence and Incidence in Sex Workers (15-49)')
plt.savefig(outputpath / ("HIV_fsw_prev_over_time" + datestamp + ".pdf"), format='pdf')
plt.show()

#%% PROGRAM COVERAGE
cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
cov_over_time = cov_over_time.set_index('date')

# ART ("90-90-90")

# Adults:
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

# Circumcision
cov_over_time.plot(y='prop_men_circ')
plt.title('Proportion of Men (15+) That Are Circumcised')
plt.savefig(outputpath / ("HIV_porp_men_circ" + datestamp + ".pdf"), format='pdf')
plt.show()

# PrEP
cov_over_time.plot(y='prop_fsw_on_prep')
plt.title('Proportion of FSW That Are On PrEP')
plt.savefig(outputpath / ("HIV_prop_fsw_prep" + datestamp + ".pdf"), format='pdf')
plt.show()

# Behaviour Change
cov_over_time.plot(y='prop_adults_exposed_to_behav_intv')
plt.title('Proportion of Adults (15+) Exposed to Behaviour Change Intervention')
plt.savefig(outputpath / ("HIV_prop_behav_chg" + datestamp + ".pdf"), format='pdf')
plt.show()


