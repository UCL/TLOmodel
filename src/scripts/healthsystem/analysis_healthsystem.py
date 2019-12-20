import datetime
import logging
import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    mockitis,
)
# [NB. Working directory must be set to the root of TLO: TLOmodel/]

# Where will output go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")


start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=2010, month=12, day=31)
popsize = 50

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath / ('LogFile' + datestamp + '.log')

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)


# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ['*']

# -----------------------------------------------------------------

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=2,
                                       capabilities_coefficient=0.0))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# TODO: demonstrate the role of the squeeze factors-- force overloading etc
