import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    malecircumcision,
    symptommanager,
    tb,
)

outputpath = Path("./outputs/hiv_tb")
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2018, 12, 31)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)


sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(
    resourcefilepath=resourcefilepath,
    disable=True)
)
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
sim.register(tb.Tb(resourcefilepath=resourcefilepath))
sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

# Sets all modules to WARNING threshold, then alters hiv and tb to INFO
custom_levels = {
    "*": logging.WARNING,
    "tlo.methods.hiv": logging.INFO,
    "tlo.methods.tb": logging.INFO,
    "tlo.method.malecircumcision": logging.INFO,
    "tlo.methods.demography": logging.INFO,
}
logfile = sim.configure_logging(filename="LogFile", custom_levels=custom_levels)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


