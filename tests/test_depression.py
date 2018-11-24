import pytest

from tlo import Simulation, Date
from tlo.methods import demography, depression
from tlo.test import random_birth

path = 'C:/Users/Andrew Phillips/Documents/thanzi la onse/Demography.xlsx'
# Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    random_birth_module = random_birth.RandomBirth()
    random_birth_module.pregnancy_probability = 0.005
    depression_module = depression.Depression()
    sim.register(core_module)
    sim.register(depression_module)
    sim.register(random_birth_module)
    return sim


def test_depression_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_depression_simulation(simulation)

    # plot the total history

    stats = simulation.modules['Depression'].o_prop_depr['prop_depr']

    import matplotlib.pyplot as plt
    import numpy as np

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 0.4)
    plt.plot(xvals, yvals)
    plt.show()
