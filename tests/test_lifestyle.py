import pytest

from tlo import Simulation, Date
from tlo.methods import demography, lifestyle
from tlo.test import random_birth

path = 'C:/Users/Andrew Phillips/Documents/thanzi la onse/Demography.xlsx'
# Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
popsize = 1000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    random_birth_module = random_birth.RandomBirth()
    random_birth_module.pregnancy_probability = 0.005
    lifestyle_module = lifestyle.Lifestyle()
    sim.register(core_module)
    sim.register(lifestyle_module)
    sim.register(random_birth_module)
    return sim


def test_lifestyle_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_lifestyle_simulation(simulation)

    # plot the urban total history
  #  stats = simulation.modules['Lifestyle'].store['urban_total']
    stats2 = simulation.modules['Lifestyle'].store['alive']
    stats3 = simulation.modules['Lifestyle'].o_prop_m_urban_overwt['prop_m_urban_overwt']
    stats4 = simulation.modules['Lifestyle'].o_prop_f_urban_overwt['prop_f_urban_overwt']
    stats5 = simulation.modules['Lifestyle'].o_prop_m_rural_overwt['prop_m_rural_overwt']
    stats6 = simulation.modules['Lifestyle'].o_prop_f_rural_overwt['prop_f_rural_overwt']
    stats7 = simulation.modules['Lifestyle'].o_prop_urban['prop_urban']

    import matplotlib.pyplot as plt
    import numpy as np
   # plt.plot(np.arange(0, len(stats)), stats)
   # plt.show()

   # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats2))

    yvals = stats6
    plt.ylim(0, 1)
    plt.plot(xvals, yvals)
    plt.show()

