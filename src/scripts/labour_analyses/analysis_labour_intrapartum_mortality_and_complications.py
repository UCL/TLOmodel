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
end_date = Date(2013, 1, 1)
popsize = 10

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
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

# All cause Intrapartum Maternal Deaths

all_cause_deaths = output['tlo.methods.demography']['death']
all_cause_deaths['date'] = pd.to_datetime(all_cause_deaths['date'])
all_cause_deaths['year'] = all_cause_deaths['date'].dt.year
intrapartum_deaths_df = all_cause_deaths.loc[all_cause_deaths.cause == 'labour']
# Above will change we when append causes onto instantaneous death logging
intrapartum_deaths_by_year = intrapartum_deaths_df.groupby(['year'])['person_id'].size()

death_by_cause = intrapartum_deaths_by_year.reset_index()
death_by_cause.index = death_by_cause['year']
death_by_cause.drop(columns='year', inplace=True)
death_by_cause = death_by_cause.rename(columns={'person_id': 'num_deaths'})

death_by_cause.plot.bar(stacked=True)
plt.title("Total Intrapartum Deaths per Year")
plt.show()

# Intrapartum MMR
mmr_df = pd.concat((death_by_cause, live_births_by_year), axis=1)
mmr_df.columns = ['maternal_deaths', 'live_births']
mmr_df['MMR'] = mmr_df['maternal_deaths']/mmr_df['live_births'] * 100000

mmr_df.plot.bar(y='MMR', stacked=True)
plt.title("Yearly Intrapartum Maternal Mortality Rate")
plt.show()

# ======================================= COMPLICATION INCIDENCE ======================================================


def incidence_analysis(complication, birth_denominator):
    dataframe = output['tlo.methods.labour'][f'{complication}']
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['year'] = dataframe['date'].dt.year
    complication_per_year = dataframe.groupby(['year'])['person_id'].size()
    if ~complication_per_year.empty:

        complication_df = pd.concat((complication_per_year, all_births_by_year), axis=1)
        complication_df.columns = ['maternal_sepsis_cases', 'all_births']
        complication_df[f'{complication}'] = complication_df['complication_cases']/complication_df['all_births'] * \
                                             birth_denominator

        complication_df.plot.bar(y='sepsis_incidence', stacked=True)
        plt.title(f"Yearly {complication} Incidence")
        plt.show()

    else:
        print(f'no cases of {complication} in this simulation run')


   # TODO: maternal deaths by each contributing cause



# TODO: determine correct denominators
# Incidence of Obstructed Labour


incidence_analysis('obstructed_labour', 100)

# Incidence of Uterine Rupture
incidence_analysis('uterine_rupture', 100)

# Incidence of Antepartum Haemorrhage
incidence_analysis('antepartum_haem', 100)

# Incidence of Intrapartum Eclampsia
incidence_analysis('eclampsia', 1000)

# Incidence of Intrapartum direct maternal sepsis
incidence_analysis('sepsis', 100)

# Intrapartum Stillbirths
# Intrapartum SBR
# TODO: disable health system
