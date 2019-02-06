import logging
import os

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, lifestyle

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 4, 1)
popsize = 1000


@pytest.fixture(scope='module')
def simulation():
    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       workbook_name)
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(1)
    sim.register(demography.Demography(workbook_path=demography_workbook))
    sim.register(lifestyle.Lifestyle())

    # turn off demography module logging
    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)

    return sim


def test_lifestyle_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    simulation = simulation()
    test_lifestyle_simulation(simulation)
