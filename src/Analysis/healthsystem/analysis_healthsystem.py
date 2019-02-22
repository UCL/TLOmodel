import os
import time
import logging

from tlo import Date, Simulation
from tlo.methods import demography
from tlo.methods import healthsystem

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50


demography_workbook = os.path.join(os.path.dirname(__file__),
                                   'resources',
                                   workbook_name)

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
demography_module = demography.Demography(workbook_path=demography_workbook)
healthsystem_module= healthsystem.HealthSystem()

sim.register(core_module,healthsystem_module)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()





