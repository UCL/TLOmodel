import datetime
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    contraception,
    schisto
)

# Where will output go
outputpath = ""
# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
# resourcefilepath = Path(os.path.dirname(__file__)) / '../../../resources'
start_date = Date(2015, 1, 1)
end_date = Date(2022, 1, 1)
popsize = 100000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + "LogFile" + datestamp + ".log"

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

logging.getLogger("tlo.methods.demography").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.contraception").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.healthburden").setLevel(logging.INFO)
logging.getLogger("tlo.methods.healthsystem").setLevel(logging.WARNING)
logging.getLogger("tlo.methods.schisto").setLevel(logging.INFO)


# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(schisto.Schisto(resourcefilepath=resourcefilepath))

# Run the simulation and flush the logger
sim.seed_rngs(0)
# initialise the population
sim.make_initial_population(n=popsize)
# start the simulation
sim.simulate(end_date=end_date)
fh.flush()
output = parse_log_file(logfile)

# ---------------------------------------------------------------------------------------------------------
#   Check the prevalence of the symptoms
df = sim.population.props
params = sim.modules['Schisto'].parameters
symptoms_params = params['symptoms_haematobium'].set_index('symptoms').to_dict()['prevalence']

tot_pop_alive = len(df.index[df.is_alive])
huge_list = df['ss_haematobium_specific_symptoms'].dropna().tolist()
huge_list = [item for sublist in huge_list for item in sublist]
symptoms_prevalence = [[x, huge_list.count(x)] for x in set(huge_list)]
symptoms_prevalence = dict(symptoms_prevalence)
for s in symptoms_prevalence.keys():
    print(s + ", prevalence = " + str(round(symptoms_prevalence[s] / tot_pop_alive, 3)), ", expected = " + str(symptoms_params[s]))

# get prevalence per age_years
df_pa = df[['age_years', 'ss_is_infected']][df.is_alive]
age_groups_count = df_pa['age_years'].value_counts().to_dict()
infected_age_count = df_pa[df['ss_is_infected'] == 'Haematobium']['age_years'].value_counts().to_dict()
age_prev = {}
for k in age_groups_count.keys():
    if k in infected_age_count.keys():
        prev = infected_age_count[k] / age_groups_count[k]
    else:
        prev = 0
    age_prev.update({k: prev})

import matplotlib.pylab as plt
lists = sorted(age_prev.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.xlabel('Age')
plt.ylabel('Prevalence')
plt.title('Final prevalence per age')
plt.show()



# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#   Saving the results
# ---------------------------------------------------------------------------------------------------------

# save the log outputs


# ---------------------------------------------------------------------------------------------------------
#   INSPECTING & PLOTTING
# ---------------------------------------------------------------------------------------------------------
df = sim.population.props
params = sim.modules['Schisto'].parameters
loger_PSAC = output['tlo.methods.schisto']['PSAC']
loger_SAC = output['tlo.methods.schisto']['SAC']
loger_Adults = output['tlo.methods.schisto']['Adults']
loger_All = output['tlo.methods.schisto']['All']

# Prevalence
plt.plot(loger_Adults.date, loger_Adults.Prevalence, label='Adults')
plt.plot(loger_PSAC.date, loger_PSAC.Prevalence, label='PSAC')
plt.plot(loger_SAC.date, loger_SAC.Prevalence, label='SAC')
plt.plot(loger_All.date, loger_All.Prevalence, label='All')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('Prevalence of S.Haematobium')
plt.ylabel('% of infected sub-population')
plt.xlabel('logging date')
plt.show()

# DALYS
loger_daly = output['tlo.methods.healthburden']["DALYS"]
loger_daly.drop(columns=['sex', 'YLL_Demography_Other'], inplace=True)
loger_daly = loger_daly.groupby(['date', 'age_range'], as_index=False)['YLD_Schisto_Schisto_Symptoms'].sum() # this add M and F
age_map = {'0-4': 'PSAC', '5-9': 'SAC', '10-14': 'SAC'}
loger_daly['age_group'] = loger_daly['age_range'].map(age_map)
loger_daly.fillna('Adults', inplace=True)  # the reminder will be Adults
loger_daly.drop(columns=['age_range'], inplace=True)
loger_daly = loger_daly.groupby(['date', 'age_group'], as_index=False)['YLD_Schisto_Schisto_Symptoms'].sum() # this add M and F

plt.scatter(loger_daly.date[loger_daly.age_group == 'Adults'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'Adults'], label='Adults')
plt.scatter(loger_daly.date[loger_daly.age_group == 'PSAC'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'PSAC'], label='PSAC')
plt.scatter(loger_daly.date[loger_daly.age_group == 'SAC'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'SAC'], label='SAC')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('DALYs due to schistosomiasis')
plt.ylabel('DALYs')
plt.xlabel('logging date')
plt.show()


