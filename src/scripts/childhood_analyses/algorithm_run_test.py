"""
This script is used in development. It will become the test script for diraahoea module.
"""

# %% Import Statements and initial declarations
import datetime

from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.methods import contraception, demography, ALRI, healthsystem, enhanced_lifestyle, \
    symptommanager, healthburden, healthseekingbehaviour, dx_algorithm_child, labour, pregnancy_supervisor

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

log_config = {
    "filename": "imci_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.CRITICAL,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.ALRI": logging.INFO,
        "tlo.methods.dx_algorithm_child": logging.INFO
    }
}

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 2)
pop_size = 1000
seed = 124

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# %% Run the Simulation


# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
# sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
sim.register(ALRI.ALRI(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)
