"""
This script is used in development. It will become the test script for pneumonia module.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path
from tlo import logging

from tlo import Date, Simulation
from tlo.methods import contraception, demography, pneumonia, enhanced_lifestyle, labour, healthsystem, \
    symptommanager, healthseekingbehaviour, pregnancy_supervisor, dx_algorithm_child

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
popsize = 500

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)
# sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))

# sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(pneumonia.ALRI(resourcefilepath=resourcefilepath))
# sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
