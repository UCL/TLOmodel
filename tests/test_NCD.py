import logging
import os
import time
from pathlib import Path


import pytest

from tlo import Simulation, Date, Property
from tlo.methods import demography, hypertension, t2dm#, CVD

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 100


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)

@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)

    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       'demography.xlsx')
    core_module = demography.Demography(workbook_path=demography_workbook)
    logging.getLogger('tlo.methods.demography').setLevel(logging.FATAL)

    #diabetes_module = t2dm.T2DM()                         # This will load method for diabetes
    hypertension_module = hypertension.HT()               # This will load method for hypertension

    sim.register(core_module)
    #sim.register(diabetes_module)                          # This will register method for diabetes
    #sim.register(highcholesterol_module)                   # This will register method for high cholesterol
    sim.register(hypertension_module)                       # This will register method for hypertension

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


if __name__ == '__main__':
    simulation = simulation()
    test_NCD_simulation(simulation)



def test_hypertension_adults(simulation):
    # check all mothers are female
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

