import pytest

from tlo import Simulation, Date
from tlo.methods import demography

path = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/Demography_WorkingFile_Complete.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 50


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    sim.register(core_module)
    sim.seed_rngs(0)
    simulation.verboseoutput = False
    return sim


def test_demography(simulation):
    simulation.verboseoutput = True
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)

    # check types of columns
    df = simulation.population.props
    assert df.date_of_birth.dtype == 'datetime64[ns]'
    assert df.sex.dtype == 'category'
    assert df.mother_id.dtype == 'int64'
    assert df.is_alive.dtype == 'bool'
    assert df.date_of_last_pregnancy.dtype == 'datetime64[ns]'
    assert df.age_range.dtype == 'category'
    assert df.contraception == 'category'


if __name__ == '__main__':
    simulation = simulation()
    test_demography(simulation)

