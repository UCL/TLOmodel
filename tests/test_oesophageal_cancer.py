import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophageal_cancer,
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(oesophageal_cancer.Oesophageal_Cancer(resourcefilepath=resourcefilepath))

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


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
