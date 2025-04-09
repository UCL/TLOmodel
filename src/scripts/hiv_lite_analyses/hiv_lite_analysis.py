from tlo import Simulation
from pathlib import Path
from tlo import Date

from tlo.methods import demography, hivlite, simplified_births

outputpath = Path('./outputs')
resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
log_config = {}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             hivlite.HivLite(resourcefilepath=resourcefilepath)
             )
sim.make_initial_population(n=1000)

sim.simulate(end_date=end_date)
