import os
from pathlib import Path

import pytest
import pandas as pd

from tlo import Simulation, Date
from tlo.methods import demography, simplified_births

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1000


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim():
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.Simplifiedbirths(resourcefilepath=resourcefilepath)
                 )

    # Make the population
    sim.make_initial_population(n=popsize)
    return sim


def test_simplified_births_simulation():
    end_date = Date(2015, 12, 31)
    sim = get_sim()
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_breastfeeding_simplified_birth_logic():
    """This is a simple test to ensure that breastfeeding status is applied to all newly generated individuals on
     birth"""
    sim = get_sim()
    initial_pop_size = 100
    sim.make_initial_population(n=initial_pop_size)

    # Set the probability of birth at 1 to ensure births when we call SimplifiedBirthsEvent
    sim.modules['Simplifiedbirths'].parameters['birth_prob'] = 1

    # Force the probability of exclusive breastfeeding to 1, no other 'types' should occur from births in this sim run
    sim.modules['Simplifiedbirths'].parameters['prob_breastfeeding_type'] = [0, 0, 1]

    # Run the sim for 0 days, clear event queue
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.event_queue.queue.clear()

    # Run the SimplifiedBirthsEvent on the population
    birth_event = simplified_births.SimplifiedBirthsEvent(module=sim.modules['Simplifiedbirths'])
    birth_event.apply(sim.population.props)

    # define the dataframe
    df = sim.population.props

    # Ensure births are happening and the dataframe has grown
    assert len(df) > initial_pop_size

    # As we've forced all eligible women to give birth, then the the number of women who could be pregnant
    # should equal the number of newborns who have been born
    selected_women = df.loc[(df.sex == 'F') & df.is_alive & df.age_years.between(15, 49)]
    new_borns = df.loc[df.mother_id >= 0]
    assert len(selected_women) == (len(new_borns) - 1)
    # TODO: ask asif about generation of blank  line at the end of the dataframe

    # Finally we check to make sure all newborns have their breastfeeding status set to exclusive
    assert (df.loc[new_borns.index, 'nb_breastfeeding_status'] == 'exclusive').all().all()


#if __name__ == '__main__':
#    simulation = get_sim()
#    test_simplified_births_simulation(simulation)
