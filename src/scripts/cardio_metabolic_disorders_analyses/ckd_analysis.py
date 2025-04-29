"""
This file runs the CardioMetabolicDisorders module and outputs graphs with data for comparison for incidence,
prevalence, and deaths. It also produces a csv file of prevalence of different co-morbidities.
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# ------------------------------------------------- RUN THE SIMULATION -------------------------------------------------

def runsim(seed=0):
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging

    start_date = Date(2010, 1, 1)
    end_date = Date(2012, 12, 31)
    popsize = 5000

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config, show_progress_bar=True)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath,
                                                                     do_condition_combos=True),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim


sim = runsim()

output = parse_log_file(sim.log_filepath)

