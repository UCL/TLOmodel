import pandas as pd

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import TB


def test_TB():
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(2018, 1, 1))
    tb = TB.TB_baseline(name='tb')
    sim.register(tb)
    assert sim.modules['tb'] is tb  # checks - this will cause error if false

    # Seed the random number generators
    sim.seed_rngs(1)

    # Create a population of 2 individuals
    sim.make_initial_population(n=1000)
    assert len(sim.population) == 1000

    # Simulate for 24 months, but initialise_simulation has offset 12 months
    assert sim.date == Date(2018, 1, 1)
    sim.simulate(end_date=Date(2020, 1, 1))

    df = pd.DataFrame(sim.population.props)
    # df.to_csv('Q:/Thanzi la Onse/TB/test_dataframe.csv')
    df.to_csv('/Users/Tara/Documents/test_dataframe.csv')

    print(df['has_TB'].value_counts())


# check outputs in console

outputs = pd.read_csv('/Users/Tara/Documents/test_dataframe.csv', header=0, sep=',')

# outputs = pd.read_csv('Q:/Thanzi la Onse/TB/test_dataframe.csv', header=0, sep=',')
#outputs.head(20)
#outputs.describe(include='all')
#outputs['has_TB'].value_counts()
