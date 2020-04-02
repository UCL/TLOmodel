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
end_date = Date(2015, 1, 1)
popsize = 5000

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

# def organise_logging_dataframe(module, logging_df):
#    df = output[f'tlo.methods.{module}'][f'{logging_df}']
#   df['date'] = pd.to_datetime(all_births_df['date'])
#   df['year'] = df['date'].dt.year
#   df_by_year = df.groupby(['year'])['child'].size()
#   return df_by_year

# organise_logging_dataframe('demography','on_birth')

# All births (not clear if this should be live births)
all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
all_births_by_year = all_births_df.groupby(['year'])['child'].size()

# Hospital Deliveries
hospital_deliveries = output['tlo.methods.labour']['hospital_delivery']
hospital_deliveries['date'] = pd.to_datetime(hospital_deliveries['date'])
hospital_deliveries['year'] = hospital_deliveries['date'].dt.year
hospital_deliveries_by_year = hospital_deliveries.groupby(['year'])['person_id'].size()

hospital_deliveries_births = pd.concat((hospital_deliveries_by_year, all_births_by_year), axis=1)
hospital_deliveries_births.columns = ['hospital_deliveries', 'all_births']
hospital_deliveries_births['HDR'] = hospital_deliveries_births['hospital_deliveries'] / \
                                    hospital_deliveries_births['all_births'] * 100

hospital_deliveries_births.plot.bar(y='HDR', stacked=True)
plt.title("Yearly HD Rate")
plt.show()

# Health Centre Deliveries
health_centre_deliveries = output['tlo.methods.labour']['health_centre_delivery']
health_centre_deliveries['date'] = pd.to_datetime(health_centre_deliveries['date'])
health_centre_deliveries['year'] = health_centre_deliveries['date'].dt.year
health_centre_deliveries_by_year = health_centre_deliveries.groupby(['year'])['person_id'].size()

health_centre_births = pd.concat((health_centre_deliveries_by_year, all_births_by_year), axis=1)
health_centre_births.columns = ['health_centre_deliveries', 'all_births']
health_centre_births['HCR'] = health_centre_births['health_centre_deliveries'] / \
                              health_centre_births['all_births'] * 100

health_centre_births.plot.bar(y='HCR', stacked=True)
plt.title("Yearly HC Rate")
plt.show()

# Home Births
home_births = output['tlo.methods.labour']['home_birth']
home_births['date'] = pd.to_datetime(home_births['date'])
home_births['year'] = home_births['date'].dt.year
home_births_by_year = home_births.groupby(['year'])['mother_id'].size()

home_births_births = pd.concat((home_births_by_year, all_births_by_year), axis=1)
home_births_births.columns = ['home_births', 'all_births']
home_births_births['HBR'] = home_births_births['home_births'] / \
                            home_births_births['all_births'] * 100

# delivery_setting_rates = pd.concat((hospital_deliveries_births, health_centre_births, home_births_births), axis=1)
# delivery_setting_rates.drop(('hospital_deliveries', 'all_births', ' health_centre_deliveries', 'all_births',
#                             'home_births', 'all_births'), axis=1)
# x= 'y'
home_births_births.plot.bar(y='HBR', stacked=True)
plt.title("Yearly HB Rate")
plt.show()


# Caesarean Section Rates
caesareans_df = output['tlo.methods.labour']['caesarean_section']
caesareans_df['date'] = pd.to_datetime(caesareans_df['date'])
caesareans_df['year'] = caesareans_df['date'].dt.year
caesareans_by_year = caesareans_df.groupby(['year'])['person_id'].size()

caesarean_births = pd.concat((caesareans_by_year, all_births_by_year), axis=1)
caesarean_births.columns = ['caesareans_by_year', 'all_births']
caesarean_births['CBR'] = caesarean_births['caesareans_by_year'] / \
                          caesarean_births['all_births'] * 100

caesarean_births.plot.bar(y='CBR', stacked=True)
plt.title("Yearly Total Caesarean Delivery Rate")
plt.show()

# % of women who sought care following comps (not just labour onset)
# Met need? (by signal function)
# Crude number of each signal function/intervention?
# Number of women who are sent home to deliver because higher squeeze threshold is applied

