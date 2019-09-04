import datetime
import logging
import os

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    hiv,
    lifestyle,
    malecircumcision,
    tb,
)

# Where will output go
outputpath = './src/scripts/hiv/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = "./resources/"

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Establish the logger
logfile = outputpath + 'LogFile' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(lifestyle.Lifestyle())
sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
sim.register(tb.Tb(resourcefilepath=resourcefilepath))
sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.hiv').setLevel(logging.INFO)
logging.getLogger('tlo.methods.malecircumcision').setLevel(logging.INFO)
logging.getLogger("tlo.methods.tb").setLevel(logging.INFO)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

# %% read the results
# out = sim.population.props
# out.to_csv(r'C:\Users\Tara\Documents\TLO\outputs.csv', header=True)
# import pandas as pd
# import numpy as np
import datetime
# import matplotlib.pyplot as plt
# from matplotlib import cm
from tlo.analysis.utils import parse_log_file
#
outputpath = './src/scripts/hiv/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
logfile = outputpath + 'LogFile' + datestamp + '.log'
output = parse_log_file(logfile)


## HIV
inc = output['tlo.methods.hiv']['hiv_infected']
prev_m = output['tlo.methods.hiv']['hiv_adult_prev_m']
prev_f = output['tlo.methods.hiv']['hiv_adult_prev_f']
prev_child = output['tlo.methods.hiv']['hiv_child_prev_m']
tx = output['tlo.methods.hiv']['hiv_treatment']
fsw = output['tlo.methods.hiv']['hiv_fsw']
mort = output['tlo.methods.hiv']['hiv_mortality']

inc.to_csv(r'Z:Thanzi la Onse\HIV\inc.csv', header=True)
prev_m.to_csv(r'Z:Thanzi la Onse\HIV\prev_m.csv', header=True)
prev_f.to_csv(r'Z:Thanzi la Onse\HIV\prev_f.csv', header=True)
prev_child.to_csv(r'Z:Thanzi la Onse\HIV\prev_child.csv', header=True)
tx.to_csv(r'Z:Thanzi la Onse\HIV\tx.csv', header=True)
fsw.to_csv(r'Z:Thanzi la Onse\HIV\fsw.csv', header=True)
mort.to_csv(r'Z:Thanzi la Onse\HIV\mort.csv', header=True)

# C:\Users\tdm522

## TB
tb_inc = output['tlo.methods.tb']['tb_incidence']
tb_prev_m = output['tlo.methods.tb']['tb_propActiveTbMale']
tb_prev_f = output['tlo.methods.tb']['tb_propActiveTbFemale']
tb_prev = output['tlo.methods.tb']['tb_prevalence']
tb_mort = output['tlo.methods.tb']['tb_mortality']


tb_inc.to_csv(r'Z:Thanzi la Onse\TB\inc.csv', header=True)
tb_prev_m.to_csv(r'Z:Thanzi la Onse\TB\prev_m.csv', header=True)
tb_prev_f.to_csv(r'Z:Thanzi la Onse\TB\prev_f.csv', header=True)
tb_prev.to_csv(r'Z:Thanzi la Onse\TB\prev_child.csv', header=True)
tb_mort.to_csv(r'Z:Thanzi la Onse\TB\tx.csv', header=True)


# # ------------------------------ PLOT DEATHS ----------------------------------------------------
# deaths_df = output['tlo.methods.demography']['death']
# deaths_df['date'] = pd.to_datetime(deaths_df['date'])
# deaths_df['year'] = deaths_df['date'].dt.year
# d_gp=deaths_df.groupby(['year', 'cause']).size().unstack().fillna(0)
#
# barWidth = 0.25
# # Set position of bar on X axis
# r1 = np.arange(len(d_gp.loc[:, 'Other']))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # extract data for bars
# bars1 = d_gp.loc[:, 'Other']
# bars2 = d_gp.loc[:, 'hiv']
# bars3 = d_gp.loc[:, 'tb']
#
# colours = cm.inferno_r(np.linspace(.2,.8, 3))
#
# # Make the plot
# plt.bar(r1, bars1, color=colours[0], width=barWidth, edgecolor='white', label='Other')
# plt.bar(r2, bars2, color=colours[1], width=barWidth, edgecolor='white', label='hiv')
# plt.bar(r3, bars3, color=colours[2], width=barWidth, edgecolor='white', label='tb')
#
#
# # Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['2010', '2011', '2012', '2013', '2014', '2015', '2016',
#                                                        '2017'])
# plt.xlabel('Year')
# plt.ylabel('Number of deaths')
# plt.title('Number of deaths')
#
# # Create legend & Show graphic
# plt.legend(['Other','HIV', 'TB'], loc='upper left')
# plt.show()
#
#
# # ------------------------------ PLOT HIV BY AGE ----------------------------------------------------
# hiv_df = output['tlo.methods.hiv']['adult_prev_m']
# hiv_df['date'] = pd.to_datetime(hiv_df['date'])
# hiv_df['year'] = hiv_df['date'].dt.year
#
# # select ages 15-55
# hiv_df2 = hiv_df.iloc[:,4:12]
# prev_df=pd.concat([hiv_df.loc[:, 'year'], hiv_df2], axis=1)
#
# plt.plot(prev_df.year, prev_df.iloc[:,1])
# plt.plot(prev_df.year, prev_df.iloc[:,2])
# plt.plot(prev_df.year, prev_df.iloc[:,3])
# plt.plot(prev_df.year, prev_df.iloc[:,4])
#
# plt.xlabel('Year')
# plt.ylabel('Prevalence')
# plt.title('HIV prevalence by age')
#
# # Create legend & Show graphic
# plt.legend(['15-19','20-24', '25-29'], loc='upper left')
# plt.show()
