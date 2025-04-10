from matplotlib import pyplot as plt

from tlo import Simulation, logging
from pathlib import Path
from tlo import Date
from tlo.analysis.utils import parse_log_file

from tlo.methods import demography, hivlite

outputpath = Path('./outputs')
resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)

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
             hivlite.HivLite(resourcefilepath=resourcefilepath)
             )
sim.make_initial_population(n=20000)

# sim.simulate(end_date=end_date)
logs = parse_log_file('outputs/hiv_lite_logs__2025-04-09T162555.log')
df = logs['tlo.methods.hivlite']['hiv_infection_cases']

df['year'] = df.date.dt.year
df = df.set_index('year')
df.drop('date', axis=1, inplace=True)
df.plot.bar(stacked=True)
plt.title('HIV cases by age group')
plt.show()
