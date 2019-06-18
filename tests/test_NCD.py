import logging
import os


import pytest

from tlo import Simulation, Date, Property
from tlo.methods import demography, hypertension, t2dm#, CVD

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 100


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)

    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       'demography.xlsx')
    core_module = demography.Demography(workbook_path=demography_workbook)
    logging.getLogger('tlo.methods.demography').setLevel(logging.FATAL)

    diabetes_module = t2dm.T2DM()                         # This will load method for diabetes
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


if __name__ == '__main__':
    simulation = simulation()
    test_NCD_simulation(simulation)
