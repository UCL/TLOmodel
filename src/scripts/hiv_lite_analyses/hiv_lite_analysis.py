from matplotlib import pyplot as plt

from tlo import Simulation, logging
from pathlib import Path
from tlo import Date
from tlo.analysis.utils import parse_log_file

from tlo.methods import demography, hivlite, simplified_births

outputpath = Path('./outputs')
resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)

log_config = {
    "filename": "hiv_lite_logs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hivlite": logging.INFO,
    },
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             hivlite.HivLite(resourcefilepath=resourcefilepath)
             )
sim.make_initial_population(n=20000)

sim.modules['HivLite'].parameters['art_coverage'] = 0.8

sim.simulate(end_date=end_date)
logs = parse_log_file(sim.log_filepath)
df = logs['tlo.methods.hivlite']['aids_cases']

df['year'] = df.date.dt.year
df = df.set_index('year')
df.drop('date', axis=1, inplace=True)

df.plot.bar(stacked=True)
plt.title('HIV cases by age group')
plt.show()
