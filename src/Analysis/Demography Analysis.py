
#%% Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tlo import Simulation, Date
from tlo.methods import demography

import dill


#%% Run the Simulation
path = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/Old Versions/Demography_WorkingFile.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 100000

sim = Simulation(start_date=start_date)

sim.verboseoutput=False

core_module = demography.Demography(workbook_path=path)
sim.register(core_module)

sim.seed_rngs(0)

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)



#%% Make a nice plot;

figurepath='/Users/tbh03/PycharmProjects/TLOmodel/src/Analysis/Figures/'


plt.plot(sim.modules['Demography'].store_PopulationStats['Time'], sim.modules['Demography'].store_PopulationStats['Population_Total'])
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Populaton Size")

plt.savefig(figurepath+'PopSize.png')

plt.show()



#%% Population Pyramid in 2010

# Make Dateframe of the relevant output:

df = pd.DataFrame(sim.modules['Demography'].store_PopulationStats)
m2015= ( df['Time'] >= Date(2015, 1, 1) ) & (df['Time'] < Date(2016, 1, 1)) # create mask for the 2015 results
X=df.loc[m2015] # subset to the year of interest

# create arrays of the values to plot:

ModelOutput_Women=np.array([ X['Population_Women_0-4'].values,
                      X['Population_Women_5-9'].values,
                      X['Population_Women_0-4'].values,
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

ModelOutput_Men=np.array([ X['Population_Men_0-4'].values,
                      X['Population_Men_5-9'].values,
                      X['Population_Men_0-4'].values,
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

fig, axes = plt.subplots(ncols=2, sharey=True)

y=np.arange(ModelOutput_Men.size)

axes[0].barh(y,ModelOutput_Women.flatten(), align='center', color='red', zorder=10)
axes[0].set(title='Women')
axes[1].barh(y,ModelOutput_Men.flatten(), align='center', color='blue', zorder=10)
axes[1].set(title='Men')
axes[0].invert_xaxis()
axes[0].set(yticks=y, yticklabels=Age_Group_Labels)
axes[0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.09)

plt.savefig(figurepath+'PopulationPyramid2015.pdf')
plt.show()


#%% This a second cell

fig, ax = plt.subplots()
ax.barh(y, y)
plt.show()

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))


ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()
