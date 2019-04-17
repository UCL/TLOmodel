import os
import time

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, contraception

workbook_name = 'contraception.xlsx'
workbook2_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 200


@pytest.fixture(scope='module')
def simulation():
    contraception_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       workbook_name)
    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       workbook2_name)

    sim = Simulation(start_date=start_date)
    demography_module = demography.Demography(workbook_path=demography_workbook)
    sim.register(demography_module)
    core_module = contraception.Contraception(workbook_path=contraception_workbook)
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


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
