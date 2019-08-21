import logging
import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, healthsystem, hiv, lifestyle, male_circumcision, tb

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 50


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    service_availability = list(['hiv*', 'tb*', 'male_circumcision*'])

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=0.0))
    sim.register(tb.tb(resourcefilepath=resourcefilepath))
    sim.register(hiv.hiv(resourcefilepath=resourcefilepath))
    sim.register(male_circumcision.male_circumcision(resourcefilepath=resourcefilepath))
    sim.seed_rngs(0)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_params(simulation):
    # check types of columns
    df = simulation.population.props
    test = simulation.modules['tb'].parameters
    assert (test.isnull().values.any().any())

    df.isnull().values.any()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
