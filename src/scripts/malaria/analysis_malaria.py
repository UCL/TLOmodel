import datetime
import logging
import os

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    lifestyle,
    malaria,
    hiv,
    malecircumcision
)

# Where will output go
outputpath = './src/scripts/malaria/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = "./resources/"

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 500

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(malaria.Malaria(resourcefilepath=resourcefilepath))
# sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
# sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.malaria').setLevel(logging.INFO)


# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()


# %% read the results
from tlo.analysis.utils import parse_log_file
import datetime

outputpath = './src/scripts/malaria/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath + 'LogFile' + datestamp + '.log'
output = parse_log_file(logfile)

inc = output['tlo.methods.malaria']['incidence']
pfpr = output['tlo.methods.malaria']['prevalence']
tx = output['tlo.methods.malaria']['tx_coverage']
mort = output['tlo.methods.malaria']['ma_mortality']


inc.to_csv(r'Z:\Thanzi la Onse\Malaria\inc.csv', header=True)
pfpr.to_csv(r'Z:\Thanzi la Onse\Malaria\pfpr.csv', header=True)
tx.to_csv(r'Z:\Thanzi la Onse\Malaria\tx.csv', header=True)
mort.to_csv(r'Z:\Thanzi la Onse\Malaria\mort.csv', header=True)
