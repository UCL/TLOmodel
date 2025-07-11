from pathlib import Path

from tlo import Simulation, Date, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, simplified_births

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)

resourcefilepath = Path("./resources")  # Path to resource files

outputpath = Path('./outputs')  # path to outputs folder

sim = Simulation(
    start_date=start_date,
    seed=0,
    log_config={
        'filename': 'demography_analysis',
        'directory': outputpath,
        'custom_levels': {
            '*': logging.WARNING,
            'tlo.methods.demography': logging.INFO
        }
    },
resourcefilepath=resourcefilepath
)

sim.register(demography.Demography()
             )
sim.make_initial_population(n=5000)
sim.simulate(end_date=end_date)

# run simulation and store logfile path
path_to_logfile = sim.log_filepath
# create copd logs dictionary
logs_dict = parse_log_file(path_to_logfile)['tlo.methods.demography']

# create a DataFrame that contains copd prevalence data
pop_df = logs_dict['population']
pop_df.to_excel(outputpath / 'demography_analysis.xlsx')
print(pop_df)
