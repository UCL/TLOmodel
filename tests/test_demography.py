import os
import time

import pytest

from tlo import Date, Simulation
from tlo.methods import demography

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 50


@pytest.fixture(scope='module')
def simulation():
    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       workbook_name)

    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=demography_workbook)
    sim.register(core_module)
    sim.seed_rngs(0)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    assert df.age_range.dtype == 'category'
    assert df.contraception.dtype == 'category'
    assert df.date_of_birth.dtype == 'datetime64[ns]'
    assert df.date_of_last_pregnancy.dtype == 'datetime64[ns]'
    assert df.is_alive.dtype == 'bool'
    assert df.mother_id.dtype == 'int64'
    assert df.sex.dtype == 'category'


def test_mothers_female(simulation):
    print(simulation.date)
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
