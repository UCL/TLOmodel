"""
This script is used in development. It will become the test script for diraahoea module.
"""

# %% Import Statements and initial declarations
import datetime
import os
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import contraception, demography, diarrhoea, healthsystem, enhanced_lifestyle, \
    symptommanager, healthburden, healthseekingbehaviour, dx_algorithm_child

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 2)
popsize = 200

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the outputs to file

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)
logging.getLogger("tlo.methods.demography").setLevel(logging.INFO)
logging.getLogger("tlo.methods.contraception").setLevel(logging.INFO)
logging.getLogger("tlo.methods.diarrhoea").setLevel(logging.INFO)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()
output = parse_log_file(logfile)
