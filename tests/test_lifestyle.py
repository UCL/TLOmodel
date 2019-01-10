import logging

import pytest

from tlo import Simulation, Date
from tlo.methods import demography, lifestyle
from tlo.test import random_birth

path = 'C:/Users/Andrew Phillips/Dropbox/Thanzi la Onse/05 - Resources\Demographic data\Demography_WorkingFile_Complete.xlsx'
# Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 3000


@pytest.fixture
def simulation():
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(workbook_path=path)
    lifestyle_module = lifestyle.Lifestyle()
    sim.register(core_module)
    sim.register(lifestyle_module)

    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)

    return sim


def test_lifestyle_simulation(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


if __name__ == '__main__':
    simulation = simulation()
    test_lifestyle_simulation(simulation)

    # plot the urban total history
  # stats = simulation.modules['Lifestyle'].store['urban_total']
  # stats = simulation.modules['Lifestyle'].store['alive']
  # stats = simulation.modules['Lifestyle'].o_prop_m_urban_overwt['prop_m_urban_overwt']
  # stats = simulation.modules['Lifestyle'].o_prop_f_urban_overwt['prop_f_urban_overwt']
  # stats = simulation.modules['Lifestyle'].o_prop_m_rural_overwt['prop_m_rural_overwt']
  # stats = simulation.modules['Lifestyle'].o_prop_f_rural_overwt['prop_f_rural_overwt']
  # stats = simulation.modules['Lifestyle'].o_prop_m_urban_low_ex['prop_m_urban_low_ex']
  # stats = simulation.modules['Lifestyle'].o_prop_f_urban_low_ex['prop_f_urban_low_ex']
  # stats = simulation.modules['Lifestyle'].o_prop_m_rural_low_ex['prop_m_rural_low_ex']
  # stats = simulation.modules['Lifestyle'].o_prop_f_rural_low_ex['prop_f_rural_low_ex']
  # stats = simulation.modules['Lifestyle'].o_prop_urban['prop_urban']
  # stats = simulation.modules['Lifestyle'].o_prop_wealth1['prop_wealth1']
  # stats = simulation.modules['Lifestyle'].o_prop_tob['prop_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age1519_w1_tob['prop_m_age1519_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age2039_w1_tob['prop_m_age2039_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_agege40_w1_tob['prop_m_agege40_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age1519_w1_tob['prop_f_age1519_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age2039_w1_tob['prop_f_age2039_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_agege40_w1_tob['prop_f_agege40_w1_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age1519_w2_tob['prop_m_age1519_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age2039_w2_tob['prop_m_age2039_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_agege40_w2_tob['prop_m_agege40_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age1519_w2_tob['prop_f_age1519_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age2039_w2_tob['prop_f_age2039_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_agege40_w2_tob['prop_f_agege40_w2_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age1519_w3_tob['prop_m_age1519_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age2039_w3_tob['prop_m_age2039_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_agege40_w3_tob['prop_m_agege40_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age1519_w3_tob['prop_f_age1519_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age2039_w3_tob['prop_f_age2039_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_agege40_w3_tob['prop_f_agege40_w3_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age1519_w4_tob['prop_m_age1519_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age2039_w4_tob['prop_m_age2039_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_agege40_w4_tob['prop_m_agege40_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age1519_w4_tob['prop_f_age1519_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age2039_w4_tob['prop_f_age2039_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_agege40_w4_tob['prop_f_agege40_w4_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age1519_w5_tob['prop_m_age1519_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_age2039_w5_tob['prop_m_age2039_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_agege40_w5_tob['prop_m_agege40_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age1519_w5_tob['prop_f_age1519_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_age2039_w5_tob['prop_f_age2039_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_f_agege40_w5_tob['prop_f_agege40_w5_tob']
  # stats = simulation.modules['Lifestyle'].o_prop_m_ex_alc['prop_m_ex_alc']
  # stats = simulation.modules['Lifestyle'].o_prop_f_ex_alc['prop_f_ex_alc']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_1['prop_mar_stat_1']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_2['prop_mar_stat_2']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_3['prop_mar_stat_3']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_1_agege60['prop_mar_stat_1_agege60']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_2_agege60['prop_mar_stat_2_agege60']
  # stats = simulation.modules['Lifestyle'].o_prop_mar_stat_3_agege60['prop_mar_stat_3_agege60']
  # stats = simulation.modules['Lifestyle'].o_prop_f_1550_on_con['prop_f_1550_on_con']

    stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
  # stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']

    import matplotlib.pyplot as plt
    import numpy as np

    # plt.plot(np.arange(0, len(stats)), stats)
    # plt.show()

    # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()

#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
    stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']


    import matplotlib.pyplot as plt
    import numpy as np
   # plt.plot(np.arange(0, len(stats)), stats)
   # plt.show()

   # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()


  # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
    stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']

    import matplotlib.pyplot as plt
    import numpy as np

    # plt.plot(np.arange(0, len(stats)), stats)
    # plt.show()

    # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()

#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
    stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']


    import matplotlib.pyplot as plt
    import numpy as np
   # plt.plot(np.arange(0, len(stats)), stats)
   # plt.show()

   # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()


 # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
    stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']

    import matplotlib.pyplot as plt
    import numpy as np

    # plt.plot(np.arange(0, len(stats)), stats)
    # plt.show()

    # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()

#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
    stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']


    import matplotlib.pyplot as plt
    import numpy as np
   # plt.plot(np.arange(0, len(stats)), stats)
   # plt.show()

   # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()


 # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
    stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
  # stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']

    import matplotlib.pyplot as plt
    import numpy as np

    # plt.plot(np.arange(0, len(stats)), stats)
    # plt.show()

    # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()

#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w1['prop_age6_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age6_in_ed_w5['prop_age6_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w1['prop_age14_in_ed_w1']
#   stats = simulation.modules['Lifestyle'].o_prop_age14_in_ed_w5['prop_age14_in_ed_w5']
#   stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w1['prop_age19_in_ed_w1']
    stats = simulation.modules['Lifestyle'].o_prop_age19_in_ed_w5['prop_age19_in_ed_w5']


    import matplotlib.pyplot as plt
    import numpy as np
   # plt.plot(np.arange(0, len(stats)), stats)
   # plt.show()

   # plt.plot(np.arange(0, len(stats2)), stats2)

    xvals = np.arange(0, len(stats))

    yvals = stats
    plt.ylim(0, 1.0)
    plt.plot(xvals, yvals)
    plt.show()

