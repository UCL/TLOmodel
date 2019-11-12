import logging
import os
import time
import pytest
from pathlib import Path

from tlo import Simulation, Date
from tlo.methods import demography, enhanced_lifestyle, healthsystem, hypertension

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 100


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)

@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(hypertension.Hypertension(resourcefilepath=resourcefilepath))
    sim.seed_rngs(0)
    return sim


def test_NCD_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_hypertension_adults(simulation):
    #TODO: need to check with Asif that this test age across time (not just during seeding)
    # check all hypertensive individuals are 18 years or over
    df = simulation.population.props
    HTN = df.loc[df.ht_current_status]
    assert (HTN.age_years >= 18).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_NCD_simulation(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)

