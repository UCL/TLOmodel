
#%% Run the Simulation....
from tlo import Simulation, Date
from tlo.methods import demography

import dill

path = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2060, 1, 1)
popsize = 100000

sim = Simulation(start_date=start_date)

sim.verboseoutput=False

core_module = demography.Demography(workbook_path=path)
sim.register(core_module)

sim.seed_rngs(0)

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

dill.dump_session('LocalSessionDump')

#%% Make a nice plot;

import matplotlib.pyplot as plt

# dill.load_session('LocalSessionDump')

stats = sim.modules['Demography'].store['Population_Total']
time=sim.modules['Demography'].store['Time']

plt.plot(time, stats)

plt.show()


#%% This a second cell

print('Hello')
