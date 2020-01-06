import logging
import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthsystem, labour
)

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(depression.Depression(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    test_dtypes(simulation)

def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row

    dfcol = df.columns
    orcol = orig.columns

    # for col in orcol:
    #     if col not in dfcol:
    #         print('this column not defined :' + col)
    #
    # for col in dfcol:
    #     if col not in orcol:
    #         print('this column not defined :' + col)


    assert (df.dtypes == orig.dtypes).all(): FAILS





if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    simulation.make_initial_population(n=popsize)
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
