import pytest

from tlo import Simulation, Date
from tlo.methods import demography, lifestyle

path = 'C:/Users/Andrew Phillips/Documents/thanzi la onse/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2060, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    lifestyle_module = lifestyle.Lifestyle()
    sim.register(core_module)
    sim.register(lifestyle_module)
    return sim


def test_lifestyle_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_lifestyle_simulation(simulation)

    # plot the urban total history
    stats = simulation.modules['Lifestyle'].store['urban_total']

    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.arange(0, len(stats)), stats)
    plt.show()
