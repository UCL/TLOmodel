import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, \
    healthburden

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))

logfile = sim.configure_logging(filename="LogFile")
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
output = parse_log_file(logfile)

# All births (not clear if this should be live births)
all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
all_births_by_year = all_births_df.groupby(['year'])['child'].size()

# Facility Deliveries
facility_deliveries = output['tlo.methods.labour']['facility_delivery']
facility_deliveries['date'] = pd.to_datetime(facility_deliveries['date'])
facility_deliveries['year'] = facility_deliveries['date'].dt.year
facility_by_year = facility_deliveries.groupby(['year'])['person_id'].size()

facility_delivery_births = pd.concat((facility_by_year, all_births_by_year), axis=1)
facility_delivery_births.columns = ['facility_deliveries', 'all_births']
facility_delivery_births['FBR'] = facility_delivery_births['facility_deliveries'] /\
                                  facility_delivery_births['all_births'] * 100

facility_delivery_births.plot.bar(y='FBR', stacked=True)
plt.title("Yearly Facility Delivery Rate")
plt.show()


# % of women who sought care following comps (not just labour onset)
# Met need? (by signal function)
# Crude number of each signal function/intervention?
# Number of women who are sent home to deliver because higher squeeze threshold is applied

