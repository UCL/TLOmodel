import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2015,  1, 1)
popsize = 190000

# Establish the simulation object
log_config = {
    'filename': 'LogFile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.CRITICAL,
        'tlo.methods.epilepsy': logging.INFO
    }
}

sim = Simulation(start_date=start_date, seed=20, log_config=log_config)

# make a dataframe that contains the switches for which interventions are allowed or not allowed
# during this run. NB. These must use the exact 'registered strings' that the disease modules allow


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath)
             )

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


# %% read the results
output = parse_log_file(sim.log_filepath)

prop_seiz_stat_0 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_0'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_0.plot()
plt.title('Proportion of people with epilepsy seizure status 0')
plt.ylim(0.95, 1)
plt.show()

prop_seiz_stat_1 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_1'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_1.plot()
plt.title('Proportion of people with epilepsy seizure status 1')
plt.ylim(0, 0.02)
plt.show()

prop_seiz_stat_2 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_2'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_2.plot()
plt.title('Proportion of people with epilepsy seizure status 2')
plt.ylim(0, 0.02)
plt.show()

prop_seiz_stat_3 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_3.plot()
plt.title('Proportion of people with epilepsy seizure status 3')
plt.ylim(0, 0.005)
plt.show()

