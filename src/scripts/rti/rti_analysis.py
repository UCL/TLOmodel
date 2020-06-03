from pathlib import Path
import numpy as np
from matplotlib.sankey import Sankey

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti
)

# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')

# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
popsize = 10000

sim = Simulation(start_date=start_date)
logfile = sim.configure_logging(filename="LogFile")
# if os.path.exists(logfile):
#     os.remove(logfile)
# Make all services available:
service_availability = ['*']
logging.getLogger('tlo.methods.RTI').setLevel(logging.DEBUG)

# Register the appropriate 'core' modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
# sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=2,
                                       capabilities_coefficient=1.0,
                                       ignore_cons_constraints=False,
                                       disable=False))
# (NB. will run much faster with disable=True in the declaration of the HealthSystem)
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

# Register disease modules of interest:
sim.register(rti.RTI(resourcefilepath=resourcefilepath))

# custom_levels = {
#     # '*': logging.CRITICAL,  # disable logging for all modules
#     'tlo.methods.RTI': logging.INFO,  # enable logging at INFO level
#     'tlo.methods.RTI': logging.DEBUG
#                   }

# Run the simulation
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the results
output = parse_log_file(logfile)
rt_df = output['tlo.methods.rti']['summary_1m']
deaths_df = output['tlo.methods.demography']['death']
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
newdf = deaths_df[['person_id', 'date', 'cause']]
deaths_df['year'] = deaths_df['date'].dt.year
death_by_cause = deaths_df.groupby(['year', 'cause'])['person_id'].size()
rt_df.to_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/resultdf.csv')

healthappointment = output['tlo.methods.healthsystem']['HSI_Event']
soughthealthcare = len(healthappointment)
# death_with_medical = death_by_cause.get_group('death_with_med')
# imm_death = death_by_cause.get_group('RTI_imm_death')
# print(len(death_with_medical), len(imm_death))

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injcategories.txt')
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize}: injury characteristics")
# injcategories = [self.totfracnumber, self.totdisnumber, self.tottbi, self.totsoft, self.totintorg,
#                          self.totintbled, self.totsci, self.totamp, self.toteye, self.totextlac]
sankey = Sankey(ax=ax,
                scale=data[0]/(data[0] * data[0]),
                offset=0.2,
                format='%d')

sankey.add(flows=[sum(data), -data[0], -data[1], -data[2], -data[3], -data[4],
                  - data[5], -data[6], -data[7], -data[8], -data[9], - data[10]],
           labels=['Number of injuries', 'Fractures', 'Dislocations', 'TBI', 'Soft tiss', 'Int. org', 'Int. bleed',
                   'SCI', 'Amputation', 'Eye injury', 'Skin wounds', 'Burns'],
           orientations=[0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],  # arrow directions
           pathlengths=[0.1, 0.5, 0.8, 0.5, 0.5, 0.8, 0.8, 0.5, 0.4, 0.8, 0.5, 0.8],
           trunklength=2,
           edgecolor='#027368',
           facecolor='#027368')
sankey.finish()
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjuryCharacteristics.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize} RTI summary")

fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injlocs.txt')
labels = ['head', 'face', 'neck', 'thorax', 'abdomen', 'spine', 'upper x', 'lower x']
explode = [0, 0, 0, 0, 0, 0.2, 0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f"{yearsrun} year model run, N={popsize} AIS body regions")
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/AISregions.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injnumber.txt')
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8'], data / sum(data))
plt.xlabel('Number of injured body regions')
plt.ylabel('Frequency')
plt.title(f'{yearsrun} year model run, N={popsize}:'
          '\n'
          r'Distribution of number of injured body regions')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/NumberofInjuries.png')

plt.clf()
fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injsev.txt')
labels = ['mild', 'severe']
explode = [0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f'{yearsrun} year model run, N={popsize} Distribution of injury severity')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjurySeverity.png')
plt.clf()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/ISSscores.txt')
scores, counts = np.unique(data, return_counts=True)
fig, ax = plt.subplots()

ax.bar(scores, counts / sum(counts))
plt.xlabel('ISS scores')
plt.ylabel('Frequency')
plt.title(f'{yearsrun} year model run, N={popsize}: Distribution of ISS scores')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/ISSscoreDistribution.png')

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
diedaftermed = len(newdf.loc[newdf['cause'] == 'death_with_med'])
diedimm = len(newdf.loc[newdf['cause'] == 'RTI_imm_death'])
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize}: model flow")
sankey = Sankey(ax=ax,
                scale=data[0]/(data[0] * data[0]),
                offset=0.2,
                format='%d')

sankey.add(flows=[data[0], -soughthealthcare, -diedimm],
           labels=['Number of injured persons', 'sought health care', 'Died on scene'],
           orientations=[0, 0, 1],  # arrow directions
           pathlengths=[0.4, 0.1, 0.1],
           trunklength=0.5,
           edgecolor='#027368',
           facecolor='#027368')
sankey.add(flows=[soughthealthcare, -diedaftermed, -data[4], -(soughthealthcare-diedaftermed-data[4])],
           labels=['', 'Died after treatment', 'Treated but still disabled', 'Recovered'],
           prior=0,
           connect=(1, 0),
           orientations=[0, 1, -1, 0],
           pathlengths=[0.4, 0.2, 0.2, 0.1],
           trunklength=0.5,
           edgecolor='#58A4B0',
           facecolor='#58A4B0')

sankey.finish()
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIModelFlow.png')
plt.clf()
