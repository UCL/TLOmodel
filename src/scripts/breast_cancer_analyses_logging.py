import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import make_age_grp_types, parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    breast_cancer,
    pregnancy_supervisor,
    symptommanager,
)


# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2040,  1, 1)
popsize = 100

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
sim = Simulation(start_date=start_date, seed=  1  ,  log_config=log_config)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow

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
             breast_cancer.BreastCancer(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# %% read the results
output = parse_log_file(sim.log_filepath)

