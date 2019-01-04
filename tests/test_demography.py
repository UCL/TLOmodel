import time

import pytest

from tlo import Simulation, Date
from tlo.methods import demography

path = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
path = '/Users/tamuri/Documents/2018/thanzi/Demography_WorkingFile_Complete.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 50


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    sim.register(core_module)
    sim.seed_rngs(0)
    return sim


def test_demography_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)

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
    # check all mothers are female
    df = simulation.population.props
    mothers = df.loc[df.mother_id >= 0, 'mother_id']
    is_female = mothers.apply(lambda mother_id: df.at[mother_id, 'sex'] == 'F')
    assert is_female.all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_demography_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)

