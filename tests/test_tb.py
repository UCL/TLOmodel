from tlo import Simulation, Date
from tlo.test import TB


def test_tb():
    sim = Simulation(start_date=Date(2018, 1, 1))
    tb = TB.TB()
    sim.register(tb)
    sim.make_initial_population(n=10)
    sim.simulate(end_date=Date(2020, 1, 1))
    df = sim.population.props



