import datetime
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophagealcancer,
    pregnancy_supervisor, labour, healthseekingbehaviour, symptommanager)

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 500

# Establish the simulation object and set the seed
sim = Simulation(start_date=start_date)
sim.seed_rngs(0)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
             )

# # Manipulate parameters in order that there is a high burden of oes_cancer in order to do the checking:
# # ** DEBUG **
# # For debugging purposes, make the initial level of oes cancer very high
# self.parameters['init_prop_oes_cancer_stage'] = [val * 500 for val in
#                                                  self.parameters['init_prop_oes_cancer_stage']]
#
# # *** FOR DEBUG PURPOSES: USE A HIGH RATE OF ACQUISITION: ***
# self.parameters['r_low_grade_dysplasia_none'] = 0.5


# Establish the logger
logfile = sim.configure_logging(filename="LogFile")
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# %% TODO : Demonstrate the burden and the interventions
output = parse_log_file(logfile)



PREVALENCE
DALYS
DEATHS




# %% TODO: Demonstrate the impact of the interventions

