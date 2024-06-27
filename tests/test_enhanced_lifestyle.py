""""
This is a test file for Lifestyle Module. It contains a number of checks to ensure everything is running as expected
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, DateOffset, Module, Simulation
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, enhanced_lifestyle

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)
end_date = Date(2012, 4, 1)
popsize = 10000


@pytest.fixture
def simulation(seed):
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath)
                 )
    return sim


def check_properties(df):
    # no one under 15 can be overweight, low exercise, tobacco, excessive alcohol, married
    under15 = df.age_years < 15
    assert not (under15 & pd.notna(df.li_bmi)).any()
    assert not (under15 & df.li_low_ex).any()
    assert not (under15 & df.li_tob).any()
    assert not (under15 & df.li_ex_alc).any()
    assert not (under15 & (df.li_mar_stat != 1)).any()

    # education: no one 0-5 should be in education
    assert not ((df.age_years < 5) & (df.li_in_ed | (df.li_ed_lev != 1))).any()

    # education: no one under 13 can be in secondary education
    assert not ((df.age_years < 13) & (df.li_ed_lev == 3)).any()

    # education: no one over age 20 in education
    assert not ((df.age_years > 20) & df.li_in_ed).any()

    # Check sex workers, only women and non-zero:
    assert df.loc[df.sex == 'F'].li_is_sexworker.any()
    assert not df.loc[df.sex == 'M'].li_is_sexworker.any()

    # Check circumcision (no women circumcised, some men circumcised)
    assert not df.loc[df.sex == 'F'].li_is_circ.any()
    assert df.loc[df.sex == 'M'].li_is_circ.any()


def test_properties_and_dtypes(simulation):
    simulation.make_initial_population(n=popsize)
    check_properties(simulation.population.props)
    simulation.simulate(end_date=end_date)
    check_properties(simulation.population.props)
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()
    print(f'Function Output {simulation.resource_file_path()}')

def test_assign_rural_urban_by_district(simulation):
    """ test linear model integrity in assigning individual rural urban status based on their districts """
    # make an initial population
    simulation.make_initial_population(n=1)

    # Make this individual rural
    df = simulation.population.props
    df.loc[0, 'is_alive'] = True
    df.loc[0, 'is_urban'] = False
    df.loc[0, 'district_of_residence'] = 'Lilongwe'

    # confirm an individual is rural
    assert not df.loc[0, 'li_urban']

    # reset district of residence to an urban district(Here we choose a district of residence with 1.0 urban
    # probability i.e Lilongwe City), run the rural urban linear model and check the individual is now urban.
    df.loc[0, 'district_of_residence'] = 'Lilongwe City'
    rural_urban_lm = enhanced_lifestyle.LifestyleModels(simulation.modules['Lifestyle']).rural_urban_linear_model()
    df.loc[df.is_alive, 'li_urban'] = rural_urban_lm.predict(df.loc[df.is_alive], rng=np.random)

    # check an individual is now urban
    assert df.loc[0, 'li_urban']


def test_check_properties_daily_event():
    """ A test that seeks to test the integrity of lifestyle properties. It contains a dummy module with an event that
    runs daily to ensure properties what they are expected """
    class DummyModule(Module):
        """ a dummy module for testing lifestyle properties """
        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            event = DummyLifestyleEvent(self)
            sim.schedule_event(event, sim.date)

        def on_birth(self, mother, child):
            pass

    class DummyLifestyleEvent(RegularEvent, PopulationScopeEventMixin):
        """ An event that runs daily to check the integrity of lifestyle properties """

        def __init__(self, module):
            """schedule to run everyday
            """
            self.repeat_months = 1
            self.module = module
            super().__init__(module, frequency=DateOffset(days=1))

        def apply(self, population):
            """ Apply this event to the population.
            :param population: the current population
            """
            # check lifestyle properties
            check_properties(population.props)

    # Create simulation:
    sim = Simulation(start_date=start_date)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=2000)
    sim.simulate(end_date=end_date)
