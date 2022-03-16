"""Run simulations to demonstrate the impact of HIV interventions in combination.
This can be run remotely on Azure.
It creates the file:
"""

import pickle
from pathlib import Path

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

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'combination_intervention_results.pickle'


# %% Define the simulation run:
def run_sim(scenario):

    # The resource files
    resourcefilepath = Path("./resources")

    start_date = Date(2010, 1, 1)
    end_date = Date(2030, 1, 1)
    popsize = 50000

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
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath)
                 )

    # Update the parameters are given in the scenario dict
    for p in scenario:
        assert p in sim.modules['Hiv'].parameters
        sim.modules['Hiv'].parameters[p] = scenario[p]

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Return the parsed_log-file
    return parse_log_file(sim.log_filepath)


# %% Define the scenarios:
ScenarioSet = {
    "no_intv": {
        "prob_spontaneous_test_12m": 0,
        "prob_start_art_after_hiv_test": 0,
        "prob_behav_chg_after_hiv_test": 0,
        "prob_prep_for_fsw_after_hiv_test": 0,
        "prob_circ_after_hiv_test": 0},

    "behav_chg_only": {
        "prob_spontaneous_test_12m": 0.2,
        "prob_start_art_after_hiv_test": 0,
        "prob_behav_chg_after_hiv_test": 0.5,
        "prob_prep_for_fsw_after_hiv_test": 0,
        "prob_circ_after_hiv_test": 0},

    "behav_chg_and_circ": {
        "prob_spontaneous_test_12m": 0.2,
        "prob_start_art_after_hiv_test": 0,
        "prob_behav_chg_after_hiv_test": 0.5,
        "prob_prep_for_fsw_after_hiv_test": 0,
        "prob_circ_after_hiv_test": 0.8},

    "behav_chg_and_circ_and_art": {
        "prob_spontaneous_test_12m": 0.2,
        "prob_start_art_after_hiv_test": 0.8,
        "prob_behav_chg_after_hiv_test": 0.5,
        "prob_prep_for_fsw_after_hiv_test": 0,
        "prob_circ_after_hiv_test": 0.8},

    "behav_chg_and_circ_and_art_and_prep": {
        "prob_spontaneous_test_12m": 0.2,
        "prob_start_art_after_hiv_test": 0.8,
        "prob_behav_chg_after_hiv_test": 0.5,
        "prob_prep_for_fsw_after_hiv_test": 0.6,
        "prob_circ_after_hiv_test": 0.8}
}


# %% Run the scenarios:
outputs = dict()
for scenario in ScenarioSet:
    outputs[scenario] = run_sim(ScenarioSet[scenario])

# %% Save the results
with open(results_filename, 'wb') as f:
    pickle.dump({
        'ScenarioSet': ScenarioSet,
        'outputs': outputs},
        f, pickle.HIGHEST_PROTOCOL
    )
