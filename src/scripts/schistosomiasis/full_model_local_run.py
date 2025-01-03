"""This file is an edited and updated version of the file `schisto_analysis.py` and has been created to allow a check
that the model is working as originally intended."""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.methods.fullmodel import fullmodel

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return outputpath / f"Schisto_{name}.png"


# Districts that are high prevalence and for which this model has been calibrated:
fitted_districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

# Name of species that being considered:
species = ('mansoni', 'haematobium')


# %% Run the simulation
def run_simulation(popsize,
                   hs_disable_and_reject_all,
                   equal_allocation_by_district,
                   mda_execute,
                   single_district):
    start_date = Date(2010, 1, 1)
    end_date = Date(2013, 12, 31)
    # For logging
    custom_levels = {
        '*': logging.WARNING,
        "tlo.methods.schisto": logging.INFO,
        # "tlo.methods.healthsystem.summary": logging.INFO,
        # "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.alri": logging.INFO,
        "tlo.methods.diarrhoea": logging.INFO,
        "tlo.methods.bladder_cancer": logging.INFO,
    }

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=100, log_config={"filename": "schisto_test_runs",
                                                                "custom_levels": custom_levels, })
    sim.register(*fullmodel(resourcefilepath=resourcefilepath,
                           use_simplified_births=True,
                           module_kwargs={
                               "HealthSystem": {"disable_and_reject_all": hs_disable_and_reject_all},
                               "Schisto": {"single_district": single_district,
                                           "mda_execute": mda_execute,},
                               "Demography": {"equal_allocation_by_district": equal_allocation_by_district}}
                           ))

    # sim.modules["Schisto"].parameters["calibration_scenario"] = 0
    sim.modules["Schisto"].parameters["scaleup_WASH"] = False
    sim.modules["Schisto"].parameters["scaleup_WASH_start_year"] = 2025
    sim.modules["Schisto"].parameters['mda_coverage'] = 0
    sim.modules["Schisto"].parameters['mda_target_group'] = 'SAC'
    sim.modules["Schisto"].parameters['mda_frequency_months'] = 12

    # initialise the population
    sim.make_initial_population(n=popsize)

    # start the simulation
    sim.simulate(end_date=end_date)

    output = parse_log_file(sim.log_filepath)

    return sim, output


# todo update these parameters
sim, output = run_simulation(popsize=500,
                             equal_allocation_by_district=True,
                             hs_disable_and_reject_all=False,  # if True, no HSIs run
                             mda_execute=True,
                             single_district=False)

# %% Extract and process the `pd.DataFrame`s needed

with open(outputpath / "test_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "test_run.pickle", "rb") as f:
    output = pickle.load(f)


