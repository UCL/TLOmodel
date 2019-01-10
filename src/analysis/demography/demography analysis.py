#%% Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging

from tlo import Simulation, Date
from tlo.methods import demography
import os

#%% Run the Simulation
dirpath='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/'
workbookpath = dirpath + 'Demography_WorkingFile_Complete.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2050, 1, 1)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

logfile = dirpath + 'LogFile.log'
if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

sim.register(demography.Demography(workbook_path=workbookpath))
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

fh.flush()

#%% read the results

from tlo.analysis.utils import parse_log_file

output=parse_log_file(logfile)

#%% Plot Population Size Over time:

figurepath=dirpath
datestamp=datetime.date.today().strftime("__%Y_%m_%d")

pop_df = output['tlo.methods.demography']['population']

Data= pd.read_excel(workbookpath, sheet_name='Interpolated Pop Structure')
Data_Pop=Data.groupby(by='year')['value'].sum()
Data_Years=Data.groupby(by='year')['year'].mean()
Data_Years = pd.to_datetime(Data_Years, format='%Y')
Data_Pop_Normalised=100*Data_Pop/np.asarray(Data_Pop[(Data_Years==Date(2010,1,1))])


Model_Years=pd.to_datetime(pop_df.date)
Model_Pop=pop_df.total
Model_Pop_Normalised=100*np.asarray(Model_Pop)/np.asarray(Model_Pop[Model_Years=='2010-01-01'])


plt.plot(np.asarray(Model_Years),Model_Pop_Normalised)
plt.plot(Data_Years,Data_Pop_Normalised)
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size (Normalised to 2010)")
plt.gca().set_xlim(Date(2010,1,1), Date(2050,1,1))
plt.legend(['Model','Data'])
plt.savefig(figurepath+'PopSize'+ datestamp +'.pdf')

plt.show()



#%% Population Pyramid in 2015

# Make Dateframe of the relevant output:

df = pd.DataFrame(sim.modules['Demography'].store_PopulationStats)

# create arrays of the values to plot: POP 2015
m2015= ( df['Time'] >= Date(2015, 1, 1) ) & (df['Time'] < Date(2016, 1, 1)) # create mask for the 2015 results
X=df.loc[m2015] # subset to the year of interest
ModelOutput_Women_2015=np.array([ X['Population_Women_0-4'].values,
                      X['Population_Women_5-9'].values,
                      X['Population_Women_10-14'].values,
                      X['Population_Women_15-19'].values,
                      X['Population_Women_20-24'].values,
                      X['Population_Women_25-29'].values,
                      X['Population_Women_30-34'].values,
                      X['Population_Women_35-39'].values,
                      X['Population_Women_40-44'].values,
                      X['Population_Women_45-49'].values,
                      X['Population_Women_50-54'].values,
                      X['Population_Women_55-59'].values,
                      X['Population_Women_60-64'].values,
                      X['Population_Women_65-69'].values,
                      X['Population_Women_70-74'].values,
                      X['Population_Women_75-79'].values,
                      X['Population_Women_80-84'].values,
                      X['Population_Women_85-'].values
                      ])

ModelOutput_Men_2015=np.array([ X['Population_Men_0-4'].values,
                      X['Population_Men_5-9'].values,
                      X['Population_Men_10-14'].values,
                      X['Population_Men_15-19'].values,
                      X['Population_Men_20-24'].values,
                      X['Population_Men_25-29'].values,
                      X['Population_Men_30-34'].values,
                      X['Population_Men_35-39'].values,
                      X['Population_Men_40-44'].values,
                      X['Population_Men_45-49'].values,
                      X['Population_Men_50-54'].values,
                      X['Population_Men_55-59'].values,
                      X['Population_Men_60-64'].values,
                      X['Population_Men_65-69'].values,
                      X['Population_Men_70-74'].values,
                      X['Population_Men_75-79'].values,
                      X['Population_Men_80-84'].values,
                      X['Population_Men_85-'].values
                      ])

m2045= ( df['Time'] >= Date(2045, 1, 1) ) & (df['Time'] < Date(2046, 1, 1)) # create mask for the 2015 results
X=df.loc[m2045] # subset to the year of interest
ModelOutput_Women_2045=np.array([ X['Population_Women_0-4'].values,
                      X['Population_Women_5-9'].values,
                      X['Population_Women_10-14'].values,
                      X['Population_Women_15-19'].values,
                      X['Population_Women_20-24'].values,
                      X['Population_Women_25-29'].values,
                      X['Population_Women_30-34'].values,
                      X['Population_Women_35-39'].values,
                      X['Population_Women_40-44'].values,
                      X['Population_Women_45-49'].values,
                      X['Population_Women_50-54'].values,
                      X['Population_Women_55-59'].values,
                      X['Population_Women_60-64'].values,
                      X['Population_Women_65-69'].values,
                      X['Population_Women_70-74'].values,
                      X['Population_Women_75-79'].values,
                      X['Population_Women_80-84'].values,
                      X['Population_Women_85-'].values
                      ])
ModelOutput_Men_2045=np.array([ X['Population_Men_0-4'].values,
                      X['Population_Men_5-9'].values,
                      X['Population_Men_10-14'].values,
                      X['Population_Men_15-19'].values,
                      X['Population_Men_20-24'].values,
                      X['Population_Men_25-29'].values,
                      X['Population_Men_30-34'].values,
                      X['Population_Men_35-39'].values,
                      X['Population_Men_40-44'].values,
                      X['Population_Men_45-49'].values,
                      X['Population_Men_50-54'].values,
                      X['Population_Men_55-59'].values,
                      X['Population_Men_60-64'].values,
                      X['Population_Men_65-69'].values,
                      X['Population_Men_70-74'].values,
                      X['Population_Men_75-79'].values,
                      X['Population_Men_80-84'].values,
                      X['Population_Men_85-'].values
                      ])

Age_Group_Labels=["0-4",
                   "5-9",
                   "10-14",
                   "15-19",
                   "20-24",
                   "25-29",
                   "30-34",
                   "35-39",
                   "40-44",
                   "45-49",
                   "50-54",
                   "55-59",
                   "60-64",
                   "65-69",
                   "70-74",
                   "75-79",
                   "80-84",
                   "85+"]


# Load population data to compare population pyramid
Data= pd.read_excel(path,sheet_name='Interpolated Pop Structure')
df=Data[Data['year']==2015].copy()
df['agegrp'] = np.floor(df.age_to / 5)
df.loc[df.agegrp > 17, 'agegrp'] = 17
tab= df.pivot_table(values="value", index=["gender","agegrp"], aggfunc=np.sum)
Data_Women_2015=np.asarray(tab[ tab.index.get_level_values('gender') == 'female' ]).flatten()
Data_Men_2015=np.asarray(tab[ tab.index.get_level_values('gender') == 'male' ]).flatten()

df=Data[Data['year']==2045].copy()
df['agegrp'] = np.floor(df.age_to / 5)
df.loc[df.agegrp > 17, 'agegrp'] = 17
tab= df.pivot_table(values="value", index=["gender","agegrp"], aggfunc=np.sum)
Data_Women_2045=np.asarray(tab[ tab.index.get_level_values('gender') == 'female' ]).flatten()
Data_Men_2045=np.asarray(tab[ tab.index.get_level_values('gender') == 'male' ]).flatten()



# Traditional Populaiton Pyramid
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
y=np.arange(ModelOutput_Men_2015.size)
axes[0][0].barh(y,ModelOutput_Women_2015.flatten()/ModelOutput_Women_2015.flatten().sum(), align='center', color='red', zorder=10)
axes[0][0].set(title='Women, 2015')
axes[0][1].barh(y,ModelOutput_Men_2015.flatten()/ModelOutput_Men_2015.flatten().sum(), align='center', color='blue', zorder=10)
axes[0][1].set(title='Men, 2015')
axes[1][0].barh(y,ModelOutput_Women_2045.flatten()/ModelOutput_Women_2045.flatten().sum(), align='center', color='red', zorder=10)
axes[1][0].set(title='Women, 2045')
axes[1][1].barh(y,ModelOutput_Men_2045.flatten()/ModelOutput_Men_2045.flatten().sum(), align='center', color='blue', zorder=10)
axes[1][1].set(title='Men, 2045')
axes[0][0].invert_xaxis()
axes[0][0].set(yticks=y, yticklabels=Age_Group_Labels)
axes[0][0].yaxis.tick_right()
axes[1][0].invert_xaxis()
axes[1][0].set(yticks=y, yticklabels=Age_Group_Labels)
axes[1][0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.09)
plt.savefig(figurepath+'PopPyramidDataOnly'+ datestamp +'.pdf')
plt.show()


fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
y=np.arange(ModelOutput_Men_2015.size)
axes[0][0].plot(y,ModelOutput_Women_2015.flatten()/ModelOutput_Women_2015.flatten().sum())
axes[0][0].plot(y,Data_Women_2015.flatten()/Data_Women_2015.flatten().sum())
axes[0][0].set(title='Women, 2015')
axes[0][0].legend(['Model','Data'])
axes[0][1].plot(y,ModelOutput_Men_2015.flatten()/ModelOutput_Men_2015.flatten().sum())
axes[0][1].plot(y,Data_Men_2015.flatten()/Data_Men_2015.flatten().sum())
axes[0][1].set(title='Men, 2015')
axes[1][0].plot(y, ModelOutput_Women_2045.flatten() / ModelOutput_Women_2045.flatten().sum())
axes[1][0].plot(y, Data_Women_2045.flatten() / Data_Women_2045.flatten().sum())
axes[1][0].set(title='Women, 2045')
axes[1][1].plot(y,ModelOutput_Men_2045.flatten()/ModelOutput_Men_2045.flatten().sum())
axes[1][1].plot(y,Data_Men_2045.flatten()/Data_Men_2045.flatten().sum())
axes[1][1].set(title='Men, 2045')
axes[1][0].set(xticks=y, xticklabels=Age_Group_Labels)
axes[1][1].set(xticks=y, xticklabels=Age_Group_Labels)


for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.09)
plt.savefig(figurepath+'PopPyramid_ModelVsData'+ datestamp +'.pdf')
plt.show()





# #%% Plot deaths (by cause) --- won't work until multiple causes are being generated
#
# # % sum-up-deaths in bins of a year and plot
#
# df=pd.DataFrame([sim.modules['Demography'].store_EventsLog['DeathEvent_Time'],
#              sim.modules['Demography'].store_EventsLog['DeathEvent_Age'],
#              sim.modules['Demography'].store_EventsLog['DeathEvent_Cause']]).transpose()
# df['count']=1
# df.columns=['DeathEvent_Time','DeathEvent_Age','DeathEvent_Cause','Count']
#
#
# df['DeathEvent_Time'].hist()
#
# plt.xlabel('Date')
# plt.ylabel('Number of deaths')
# plt.title('Deaths Over Time')
# plt.savefig(figurepath+'DeathsOverTime'+ datestamp +'.pdf')
# plt.show()
#
#
# # Make code ready to do histogram (stacked) when multiple causes
# CauseTemp=pd.DataFrame(np.where(np.random.random(len(df))<0.5,'Other','HIV')) # create some dummy data to demonstarte
# df['DeathEvent_Cause']=CauseTemp
# pd.DataFrame({'Cause = Other': df.groupby('DeathEvent_Cause').get_group('Other').DeathEvent_Age,
#               'Cause = HIV':   df.groupby('DeathEvent_Cause').get_group('HIV').DeathEvent_Age}).plot.hist(stacked=True)
# plt.show()


#%% Plots births ....

df=pd.DataFrame([sim.modules['Demography'].store_EventsLog['BirthEvent_Time'],
             sim.modules['Demography'].store_EventsLog['BirthEvent_AgeOfMother'],
             sim.modules['Demography'].store_EventsLog['BirthEvent_Outcome']]).transpose()
df.columns=['BirthEvent_Time','BirthEvent_AgeOfMother','BirthEvent_Outcome']

plt.plot_date(df['BirthEvent_Time'],df['BirthEvent_AgeOfMother'])
plt.xlabel('Year')
plt.ylabel('Age of Mother')
plt.show()


#%% Plots deaths ...

df=pd.DataFrame([sim.modules['Demography'].store_EventsLog['DeathEvent_Time'],
             sim.modules['Demography'].store_EventsLog['DeathEvent_Age'],
             sim.modules['Demography'].store_EventsLog['DeathEvent_Cause']]).transpose()

df.columns=['DeathEvent_Time','DeathEvent_Age','DeathEvent_Cause']

plt.plot_date(df['DeathEvent_Time'],df['DeathEvent_Age'])
plt.show()


df['DeathEvent_Time'].hist()
plt.show()
