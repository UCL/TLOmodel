import datetime
from pathlib import Path

import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, \
    healthburden, symptommanager, healthseekingbehaviour

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

stats = output['tlo.methods.pregnancy_supervisor']['summary_stats']
stats['date'] = pd.to_datetime(stats['date'])
stats['year'] = stats['date'].dt.year

