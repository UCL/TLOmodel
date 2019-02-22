import datetime
import os
import logging

from tlo import Date, Simulation
from tlo.methods import demography
from tlo.methods import healthsystem


# Where will output go
outputpath = '/Users/tbh03/Dropbox (SPH Imperial College)/TLO Model Output/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
resourcefile_demography = '/Users/tbh03/PycharmProjects/TLOmodel/resources/Demography_WorkingFile_Complete.xlsx'


start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50


# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp  +'.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)


# Register the appropriate modules
demography_module = demography.Demography(workbook_path=resourcefile_demography)
healthsystem_module= healthsystem.HealthSystem()

sim.register(demography_module,healthsystem_module)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()





