"""This analysis file produces all mortality outputs"""

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
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)
service_availability = ['*']

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
#sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
#                                           service_availability=service_availability,
#                                           capabilities_coefficient=0.0,
#                                           mode_appt_constraints=2))

sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
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

# =====================================================================================================================
# Live births
live_births = output['tlo.methods.labour']['live_births']
live_births['date'] = pd.to_datetime(live_births['date'])
live_births['year'] = live_births['date'].dt.year
live_births_by_year = live_births.groupby(['year'])['child'].size()

# All births
all_births_df = output['tlo.methods.demography']['on_birth']
all_births_df['date'] = pd.to_datetime(all_births_df['date'])
all_births_df['year'] = all_births_df['date'].dt.year
all_births_by_year = all_births_df.groupby(['year'])['child'].size()

# ================================ All cause Early Neonatal Deaths ==============================================
all_cause_deaths = output['tlo.methods.demography']['death']
all_cause_deaths['date'] = pd.to_datetime(all_cause_deaths['date'])
all_cause_deaths['year'] = all_cause_deaths['date'].dt.year
neonatal_deaths_df = all_cause_deaths.loc[all_cause_deaths.cause == 'neonatal complications']
# Above will change we when append causes onto instantaneous death logging
neonatal_deaths_by_year = neonatal_deaths_df.groupby(['year'])['person_id'].size()

death_by_cause = neonatal_deaths_by_year.reset_index()
death_by_cause.index = death_by_cause['year']
death_by_cause.drop(columns='year', inplace=True)
death_by_cause = death_by_cause.rename(columns={'person_id': 'num_deaths'})

death_by_cause.plot.bar(stacked=True)
plt.title("Total Early Neonatal Deaths per Year")
plt.show()

# Early (first 7 days) NMR:
nmr_df = pd.concat((death_by_cause, live_births_by_year), axis=1)
nmr_df.columns = ['neonatal_deaths', 'live_births']
nmr_df['NMR'] = nmr_df['neonatal_deaths']/nmr_df['live_births'] * 1000

nmr_df.plot.bar(y='NMR', stacked=True)
plt.title("Yearly Early Neonatal Mortality Rate")
plt.show()

# ======================================= COMPLICATION INCIDENCE ======================================================

# todo: do we want all births or live births? or at this stage just crude numbers

def incidence_analysis(complication, birth_denominator):
    dataframe = output['tlo.methods.newborn_outcomes'][f'{complication}']
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['year'] = dataframe['date'].dt.year
    complication_per_year = dataframe.groupby(['year'])['person_id'].size()
    if ~dataframe.empty:

        complication_df = pd.concat((complication_per_year, all_births_by_year), axis=1)
        complication_df.columns = ['complication_cases', 'all_births']
        complication_df[f'{complication}_incidence'] = complication_df['complication_cases'] / complication_df[
            'all_births'] * birth_denominator

        complication_df.plot.bar(y=f'{complication}_incidence', stacked=True)
        plt.title(f"Yearly {complication} Incidence per {birth_denominator} births")
        plt.show()

    else:
        print(f'no cases of {complication} in this simulation run')

incidence_analysis('early_preterm', 100)

incidence_analysis('late_preterm', 100)

incidence_analysis('low_birth_weight_newborn', 100)

incidence_analysis('small_for_gestational_age_newborn', 100)

# Incidence of Neonatal Sepsis
incidence_analysis('early_onset_nb_sep', 1000)

# Incidence of Failure to transition
incidence_analysis('nb_failed_to_transition', 1000)

# Incidence of encephalopathy
incidence_analysis('neonatal_enceph', 1000)


