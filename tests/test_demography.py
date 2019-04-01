import logging
import os
import time

import pytest

from tlo import Date, Simulation
from tlo.methods import demography

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = os.path.join(os.path.dirname(__file__), 'resources')

    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(resourcefilepath=resourcefilepath)
    sim.register(core_module)
    sim.seed_rngs(0)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_mothers_female(simulation):
    # check all mothers are female
    df = simulation.population.props
    mothers = df.loc[df.mother_id >= 0, 'mother_id']
    is_female = mothers.apply(lambda mother_id: df.at[mother_id, 'sex'] == 'F')
    assert is_female.all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
