import logging
import os
import time
import pytest
from pathlib import Path

from tlo import Simulation, Date, Property
from tlo.methods import demography, hypertension#, t2dm#, CVD

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)

@pytest.fixture(scope='module')
def simulation():
    demography_resource_path = Path(os.path.dirname(__file__)) / '../resources'
    hypertension_resource_path = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=demography_resource_path))
    sim.register(hypertension.HT(resourcefilepath=hypertension_resource_path))
    sim.seed_rngs(1)
    return sim


def test_NCD_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    simulation = simulation()
    test_NCD_simulation(simulation)



def test_hypertension_adults(simulation):
    # check all hypertensive individuals are adults
    df = simulation.population.props
    HTN = df.loc[df.current_status]
    is_adult = HTN.apply(lambda age_years: df.at.age_years > 17)
    assert is_adult.all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_NCD_simulation(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)

