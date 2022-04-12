"""This file is an edited and updated version of the file `schisto_analysis.py` and has been created to allow a check
that the model is working as originally intended."""

import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    schisto,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def get_simulation(seed, start_date, mda_execute=True):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 schisto.Schisto(resourcefilepath=resourcefilepath, mda_execute=mda_execute),
                 )
    return sim


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_run_without_mda(seed):
    """Run the Schisto module with default parameters for one year on a population of 10_000, with no MDA"""

    start_date = Date(2010, 1, 1)
    end_date = start_date + pd.DateOffset(years=1)
    popsize = 10_000

    sim = get_simulation(seed=seed, start_date=start_date, mda_execute=False)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


@pytest.mark.slow
def test_run_with_mda(seed):
    """Run the Schisto module with default parameters for 20 years on a population of 1_000, with MDA"""

    start_date = Date(2010, 1, 1)
    end_date = start_date + pd.DateOffset(years=20)
    popsize = 5_000

    sim = get_simulation(seed=seed, start_date=start_date, mda_execute=True)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
