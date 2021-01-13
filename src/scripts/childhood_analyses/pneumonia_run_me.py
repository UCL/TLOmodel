"""
This script is used in development. It will become the test script for pneumonia module.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path
from tlo import logging
from tlo.analysis.utils import parse_log_file

from tlo import Date, Simulation
from tlo.methods import contraception, demography, pneumonia, enhanced_lifestyle, labour, healthsystem, \
    symptommanager, healthseekingbehaviour, pregnancy_supervisor, dx_algorithm_child, healthburden

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
pop_size = 100
seed = 123

log_config = {
    "filename": "imci_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.CRITICAL,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.pneumonia": logging.INFO,
    }
}

# Used to configure health system behaviour
# service_availability = ["*"]

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(pneumonia.ALRI(resourcefilepath=resourcefilepath))

sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
# output = parse_log_file(sim.log_filepath)
