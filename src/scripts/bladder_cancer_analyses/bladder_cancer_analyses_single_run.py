"""
File to run the model once for simple checks and analysis
"""

import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    bladder_cancer,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to cause_of_death log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2020,  1, 1)
popsize = 19000

# Establish the simulation object
log_config = {
    'filename': 'LogFile',
    'directory': outputpath,
    'custom_levels': {
        'tlo.methods.demography': logging.CRITICAL,
        'tlo.methods.contraception': logging.CRITICAL,
        'tlo.methods.healthsystem': logging.CRITICAL,
        'tlo.methods.labour': logging.CRITICAL,
        'tlo.methods.healthburden': logging.CRITICAL,
        'tlo.methods.symptommanager': logging.CRITICAL,
        'tlo.methods.healthseekingbehaviour': logging.CRITICAL,
        'tlo.methods.pregnancy_supervisor': logging.CRITICAL
    }
}
sim = Simulation(start_date=start_date, seed=4, log_config=log_config)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results
output = parse_log_file(sim.log_filepath)
