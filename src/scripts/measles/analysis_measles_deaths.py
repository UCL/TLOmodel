from pathlib import Path
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import os

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    dx_algorithm_child,
    healthseekingbehaviour,
    symptommanager,
    antenatal_care,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    epi,
    measles
)

# The resource files
resourcefilepath = Path("./resources")

# store output files
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

start_date = Date(2010, 1, 1)
end_date = Date(2015, 12, 31)
popsize = 100


def run_sim(service_availability=[]):
    # Establish the simulation object and set the seed
    # seed is not set - each simulation run gets a random seed
    sim = Simulation(start_date=start_date, seed=32,
                     log_config={"filename": "LogFile",
                                 'custom_levels': {"*": logging.WARNING, "tlo.methods.measles": logging.INFO,
                                                   "tlo.methods.demography": logging.INFO}
                                 }
                     )

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=True,
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        contraception.Contraception(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        measles.Measles(resourcefilepath=resourcefilepath),
    )

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)

    # change seeding event prob to ensure outbreak can spread
    sim.modules['Covid'].parameters['prob_initial_infection'] = 0.001
    sim.modules['Covid'].parameters['beta'] = 0.008

    sim.simulate(end_date=end_date)

    return sim.log_filepath




def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    # incidence
    prev_and_inc_over_time = output['tlo.methods.covid']['summary']
    prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')
    incidence = prev_and_inc_over_time['NumberInfectedMonth']
    inc_age = output['tlo.methods.covid']['inc_by_age']

    # deaths
    # deaths = output['tlo.methods.demography']['death'].copy()
    # deaths = deaths.set_index('date')
    # # limit to deaths due to covid
    # to_drop = (deaths.cause != 'Covid')
    # deaths = deaths.drop(index=to_drop[to_drop].index)
    # # count by year:
    # deaths['year'] = deaths.index.year
    # tot_covid_deaths = deaths.groupby(by=['year']).size()

    # count by week:
    # deaths['week'] = deaths.index.week
    # tot_covid_deaths = deaths.groupby(by=['week']).size()

    deaths = output['tlo.methods.demography']['death'].copy()
    to_drop = (deaths.cause != 'Covid')
    deaths = deaths.drop(index=to_drop[to_drop].index)
    # work out time since outbreak began
    deaths['days'] = (deaths['date'] - Date(2010, 1, 1)).dt.days
    # count by days since outbreak start
    tot_covid_deaths = deaths.groupby(by=['days']).size()

    return {
        'incidence': incidence,
        'deaths': tot_covid_deaths,
    }



# run 1
# run baseline with no vaccination to allow cases to occur
# disable all hsi to prevent any measles treatment

# run 2
# no vaccination
# allow hsi for measles treatment

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

model_measles = log_df["tlo.methods.measles"]["incidence"]["inc_1000py"]
model_date = log_df["tlo.methods.measles"]["incidence"]["date"]
# ------------------------------------- PLOTS  ------------------------------------- #

