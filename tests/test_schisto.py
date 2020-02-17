import os
import time
import logging
import pandas as pd
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    healthburden,
    healthsystem,
    schisto
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 100


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation_haem():

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto_Haematobium(resourcefilepath=resourcefilepath, symptoms_and_HSI=False))

    sim.seed_rngs(1)
    return sim


@pytest.fixture(scope='module')
def simulation_both():

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto_Haematobium(resourcefilepath=resourcefilepath, symptoms_and_HSI=False))
    sim.register(schisto.Schisto_Mansoni(resourcefilepath=resourcefilepath, symptoms_and_HSI=False))

    sim.seed_rngs(1)
    return sim


def test_run(simulation_both):
    simulation_both.make_initial_population(n=popsize)
    simulation_both.simulate(end_date=end_date)


def test_dtypes(simulation_both):
    # check types of columns
    df = simulation_both.population.props
    orig = simulation_both.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_one_schisto_type(simulation_haem):
    # check that there is no columns starting with 'sm' when only haematobium is registered
    df = simulation_haem.population.props
    for col in df.columns:
        assert col[:2] != 'sm'


def test_no_symptoms_or_HSI_or_MDA(simulation_haem):
    # check that with the symptoms turned off there will be no PZQ ever administered
    # and that the symptoms will all be nans
    df = simulation_haem.population.props
    assert(len(df.ss_last_PZQ_date.unique()) == 1)
    assert(df.ss_last_PZQ_date.unique()[0] == pd.Timestamp(year=1900, month=1, day=1))
    assert df.ss_symptoms.null().all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation_both()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
    test_dtypes(simulation)
