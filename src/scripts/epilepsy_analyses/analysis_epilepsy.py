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
end_date = Date(2012,  1, 1)
popsize = 19

# Establish the simulation object
log_config = {
    'filename': 'LogFile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.CRITICAL,
        'tlo.methods.epilepsy': logging.INFO
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

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

prop_seiz_stat_1 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_1'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_1.plot()
plt.title('Proportion of people with epilepsy but no current seizures')
plt.ylim(0, 0.05)
plt.show()

prop_seiz_stat_2 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_2'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_2.plot()
plt.title('Proportion of people with infrequent epilepsy seizures')
plt.ylim(0, 0.02)
plt.show()

prop_seiz_stat_3 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_seiz_stat_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_seiz_stat_3.plot()
plt.title('Proportion of people with frequent epilepsy seizures')
plt.ylim(0, 0.005)
plt.show()

n_seiz_stat_1_3 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['n_seiz_stat_1_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
n_seiz_stat_1_3.plot()
plt.title('Number with epilepsy (past or current)')
plt.ylim(0, 100000)
plt.show()

n_seiz_stat_2_3 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['n_seiz_stat_2_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
n_seiz_stat_2_3.plot()
plt.title('Number with epilepsy (infrequent or frequent seizures)')
plt.ylim(0, 30000)
plt.show()

prop_antiepilep_seiz_stat_1 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_1'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_antiepilep_seiz_stat_1.plot()
plt.title('Proportion on antiepileptics amongst people with epilepsy but no current seizures')
plt.ylim(0, 1)
plt.show()

prop_antiepilep_seiz_stat_2 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_2'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_antiepilep_seiz_stat_2.plot()
plt.title('Proportion on antiepileptics amongst people with infrequent epilepsy seizures')
plt.ylim(0, 1)
plt.show()

prop_antiepilep_seiz_stat_3 = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['prop_antiepilep_seiz_stat_3'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
prop_antiepilep_seiz_stat_3.plot()
plt.title('Proportion on antiepileptics amongst people with frequent epilepsy seizures')
plt.ylim(0, 1)
plt.show()

n_epi_death = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['n_epi_death'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
n_epi_death.plot()
plt.title('Number of deaths from epilepsy')
plt.ylim(0, 100)
plt.show()

n_antiep = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['n_antiep'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
n_antiep.plot()
plt.title('Number of people on antiepileptics')
plt.ylim(0, 50000)
plt.show()

epi_death_rate = pd.Series(
 output['tlo.methods.epilepsy']['epilepsy_logging']['epi_death_rate'].values,
    index=output['tlo.methods.epilepsy']['epilepsy_logging']['date'])
epi_death_rate.plot()
plt.title('Rate of epilepsy death in people with seizures')
plt.ylim(0, 20 )
plt.show()

incidence_epilepsy = pd.Series(
 output['tlo.methods.epilepsy']['inc_epilepsy']['incidence_epilepsy'].values,
    index=output['tlo.methods.epilepsy']['inc_epilepsy']['date'])
incidence_epilepsy.plot()
plt.title('Incidence of epilepsy')
plt.ylim(0, 200 )
plt.show()


