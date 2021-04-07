"""
This script is used in development only. It produces a basic run of the ALRI module
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path
from tlo import logging

from tlo import Date, Simulation
from tlo.methods import (
    ALRI,
    antenatal_care,
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath / ('LogFile' + datestamp + '.log')

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
pop_size = 100

log_config = {
    "filename": "alri_basic_run",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.INFO,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.ALRI": logging.INFO,
    }
}


# add file handler for the purpose of logging
sim = Simulation(start_date=start_date, log_config=log_config)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
             ALRI.ALRI(resourcefilepath=resourcefilepath)
             )


# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# # parse the simulation logfile to get the output dataframes
# output = parse_log_file(sim.log_filepath)
# one_person = output['tlo.methods.ALRI']['person_one']
#
#
# # save into an cvs file
# one_person.to_csv(r'./outputs/one_person2.csv', index=False)
