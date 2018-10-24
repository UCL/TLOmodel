import pytest

from tlo import Simulation, Date
from tlo.methods import demography, mockitis

path = '/Users/tamuri/Documents/2018/thanzi/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    mockitis_module = mockitis.Mockitis()
    sim.register(core_module)
    sim.register(mockitis_module)
    return sim


def test_mockitis_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_mockitis_simulation(simulation)

    stats = simulation.modules['Mockitis'].store['test_stat']
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.arange(0, len(stats)), stats)
    plt.show()
