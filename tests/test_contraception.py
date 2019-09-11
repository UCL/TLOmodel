import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import contraception, demography

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 200


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    demography_module = demography.Demography(resourcefilepath=resourcefilepath)
    sim.register(demography_module)
    contraception_module = contraception.Contraception(resourcefilepath=resourcefilepath)
    sim.register(contraception_module)
    sim.seed_rngs(0)
    return sim


def __check_properties(df):
    assert not ((df.sex == 'M') & (df.co_contraception != 'not_using')).any()
    assert not ((df.age_years < 15) & (df.co_contraception != 'not_using')).any()


def test_make_initial_population(simulation):
    simulation.make_initial_population(n=popsize)


def test_initial_population(simulation):
    __check_properties(simulation.population.props)


def test_run(simulation):
    simulation.simulate(end_date=end_date)


def test_final_population(simulation):
    __check_properties(simulation.population.props)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
