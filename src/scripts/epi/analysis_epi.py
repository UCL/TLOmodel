from pathlib import Path
import pandas as pd
import time
import datetime
import os
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    epi,
)

start_time = time.time()

# Where will output go
outputpath = Path("./outputs/epi")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2013, 12, 31)
popsize = 500

# Establish the simulation object
sim = Simulation(start_date=start_date)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=service_availability,
        mode_appt_constraints=0,
        ignore_cons_constraints=True,
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=True,
    )
)  # disables the health system constraints so all HSI events run
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(epi.Epi(resourcefilepath=resourcefilepath))

# Sets all modules to WARNING threshold, then alter epi to INFO
custom_levels = {"*": logging.WARNING, "tlo.methods.epi": logging.INFO}

# configure_logging automatically appends datetime
logfile = sim.configure_logging(filename="Epi_LogFile", custom_levels=custom_levels)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

print("--- %s seconds ---" % (time.time() - start_time))
