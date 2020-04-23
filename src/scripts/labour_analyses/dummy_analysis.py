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
end_date = Date(2012, 1, 1)
popsize = 1000


def sim_without_health_system():
    sim = Simulation(start_date=start_date)

    # Register the core modules

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2))

    sim.seed_rngs(0)

    # Run the simulation
    logfile = sim.configure_logging(filename="LogFile")
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)


    output = parse_log_file(logfile)

    stats = output['tlo.methods.labour']['summary_stats']
    stats['date'] = pd.to_datetime(stats['date'])
    stats['year'] = stats['date'].dt.year
    return stats

def sim_with_health_system():
    sim = Simulation(start_date=start_date)

    # Register the core modules

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']))

    sim.seed_rngs(0)

    # Run the simulation
    logfile = sim.configure_logging(filename="LogFile")
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)


    output = parse_log_file(logfile)

    stats_health_system = output['tlo.methods.labour']['summary_stats']
    stats_health_system['date'] = pd.to_datetime(stats_health_system['date'])
    stats_health_system['year'] = stats_health_system['date'].dt.year
    return stats_health_system



