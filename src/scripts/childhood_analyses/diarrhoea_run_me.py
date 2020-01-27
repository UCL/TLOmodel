"""
@ Ines - whilst we work on this, I think running this file will make it easier to do the debugging.
So let this me the file that we both use to run the model for now.

"""


import logging
import os
from pathlib import Path

import pytest
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, enhanced_lifestyle, diarrhoea, healthsystem, symptommanager, healthburden, \
    childhood_management, healthseekingbehaviour, contraception


resourcefilepath = Path('./resources')
outputpath = Path('./outputs/')


start_date = Date(2010, 1, 1)
end_date = Date(2019, 1, 1)
popsize = 1000


# Set up the logger:
logfile = outputpath / ("LogFile.log")

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)


sim = Simulation(start_date=start_date)
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
# sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath)) ## removing this so remove any health care seeking so Ines can focus on the 'natural history' and 'epidemiology'
sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(childhood_management.ChildhoodManagement(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

fh.flush()
output = parse_log_file(logfile)
