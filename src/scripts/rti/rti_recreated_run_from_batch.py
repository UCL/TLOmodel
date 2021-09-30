from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, get_failed_batch_run_information, get_scenario_outputs
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

resourcefilepath = Path('./resources')

results_folder = get_scenario_outputs('rti_incidence_parameterisation.py', outputspath)[-1]

seed, params, popsize, start_date = get_failed_batch_run_information(results_folder,
                                                                     'rti_incidence_parameterisation.py', 2, 0)
sim = Simulation(start_date=start_date, seed=seed)
# Register the modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True, service_availability=['*']),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
)
# Get the log file
logfile = sim.configure_logging(filename="LogFile")
# create and run the simulation
sim.make_initial_population(n=popsize)
for module in params.keys():
    for param in params[module].keys():
        sim.modules[module].parameters[param] = params[module][param]

end_date = Date(year=start_date.year + 20, month=start_date.month, day=start_date.day)
sim.simulate(end_date=end_date)

