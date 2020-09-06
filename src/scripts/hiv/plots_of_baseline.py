"""Run a simulation with no HSI and plot prevalence at baseline and the prevalence and incidence trajectory"""
import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 5000

# Establish the simulation object
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
    }
}



# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# %% read the results

output = parse_log_file(sim.log_filepath)

# ANALYSES: BASELINE HIV PREVALENCE, BASELINE ART COVERAGE,
# ANALYSES: INCIDENCE RATE AND


