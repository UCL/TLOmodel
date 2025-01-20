"""
    a contraception analyses script for kenya
"""
import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography_nuhdss, contraception_nuhdss, demography
)

# The outputs' path. This is where will output be stored
outputpath = Path("./outputs/")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Path to all resource files
resourcefilepath = Path("./resources")

# start and end dates of simulation.
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
# population size
popsize = 2000

# a file that will store all logs. Here if you want you include logs to specific files you should declare their
# logging level as logging.INFO. see below otherwise the logs for that particular module won't show in log file
log_config = {
    'filename': 'kenya_contraception_analyses',
    'directory': outputpath/'kenya_contraception_analyses',
    'custom_levels': {"*": logging.WARNING,
                      "tlo.methods.demography_nuhdss": logging.INFO
                      }
}

# Establish the simulation object. assign to simulation log configuration above if you want log file created
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    demography_nuhdss.Demography(resourcefilepath=resourcefilepath)
)

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# read results
output = parse_log_file(sim.log_filepath)
print(f"the outputs are {output['tlo.methods.demography_nuhdss']}")

# following the parsed outputs above you can proceed to extracting data and do your analyses
