import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._libs.tslibs.offsets import DateOffset

from tlo import Date, Simulation, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, simplified_births
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim(seed, popsize=1000):
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
                 )

    # Make the population
    sim.make_initial_population(n=popsize)
    check_property_integrity(sim)
    return sim


def set_prob_of_pregnancy_to_one(sim):
    """Set the probability of pregnancy to one (for all ages and for all years)"""
    sim.modules['SimplifiedBirths'].parameters['age_specific_fertility_rates']['asfr'] = 1.0
    return sim


def check_property_integrity(sim):
    """check that the properties are as expected"""

    # define dataframe
    alive = sim.population.props.loc[sim.population.props.is_alive]

    # For persons not currently pregnant:
    #   - check that no date of delivery is assigned to individuals or that the date is in the past
    np_si_date_of_last_delivery = alive.loc[~alive.is_pregnant, 'si_date_of_last_delivery']
    assert (pd.isnull(np_si_date_of_last_delivery) | (np_si_date_of_last_delivery <= sim.date)).all()

    #   - check that a date of last pregnancy is not assigned to individuals or that the date is prior to last delivery
    np_date_of_last_pregnancy = alive.loc[~alive.is_pregnant, 'date_of_last_pregnancy']
    assert (pd.isnull(np_date_of_last_pregnancy) | (np_date_of_last_pregnancy < np_si_date_of_last_delivery)).all()

    # For persons who are currently pregnant:
    #    - check that all pregnant women have been assigned a date of last pregnancy and that date is in the past
    p_date_of_last_pregnancy = alive.loc[alive.is_pregnant, 'date_of_last_pregnancy']
    assert not pd.isnull(p_date_of_last_pregnancy).any()
    assert (p_date_of_last_pregnancy <= sim.date).all()

    #   - check that all pregnant women have been assigned a date of delivery and that the date is in the future
    p_si_date_of_last_delivery = alive.loc[alive.is_pregnant, 'si_date_of_last_delivery']
    assert not pd.isnull(p_si_date_of_last_delivery).any()
    assert (p_si_date_of_last_delivery > sim.date).all()

    # Check that all women identified as a mother by a newborn have been pregnant and delivered and were alive and >15yo
    # at the time of delivery/birth of that child.
    df = sim.population.props
    mothers = df.loc[~df.date_of_birth.isna() & (df.mother_id >= 0)].mother_id.unique()

    if len(mothers) > 0:
        assert not df.loc[
            mothers, ['date_of_last_pregnancy', 'si_date_of_last_delivery']
        ].isna().any().any()

        newborns = df.loc[~df.date_of_birth.isna() & (df.mother_id >= 0), ['date_of_birth', 'mother_id']]
        newborns = newborns.merge(
            df[['date_of_birth', 'date_of_death']],
            left_on='mother_id',
            right_index=True,
            suffixes=("_nb", "_mother")
        )
        assert (
            ~(newborns['date_of_birth_nb'] < (newborns['date_of_birth_mother'] + pd.DateOffset(years=15))) &
            ~(newborns['date_of_birth_nb'] > newborns['date_of_death'])
        ).all()

    # Check that breastfeeding properties are correct
    #  - those aged 6-23mo are either not breastfed or breastfed non-exclusively (limit is >0.6y here to allow for
    #   interval between 6 months being reached and the occurrence of the poll event).
    assert (df.loc[
                df.is_alive &
                (df.age_exact_years > 0.6) &
                (df.age_exact_years < 2.0),
                'nb_breastfeeding_status'].isin(['none', 'non_exclusive'])).all()

    #  - those aged >=24mo are either not breastfed (limit is >2.1y here to allow for interval between 24 months being
    #   reached and the occurrence of the poll event).
    assert (df.loc[
                df.is_alive &
                (df.age_exact_years > 2.1),
                'nb_breastfeeding_status'] == 'none').all()


def test_pregnancy_and_birth_for_one_woman(seed):
    """Test to check that properties and sequence of events work as expected, when considering a single woman."""

    sim = get_sim(seed=seed, popsize=1)
    df = sim.population.props

    # confirm that the person is alive and eligible to become pregnant
    df.loc[0, 'sex'] = 'F'
    df.loc[0, 'is_alive'] = True
    df.loc[0, 'age_exact_years'] = 17.0
    df.loc[0, 'date_of_birth'] = sim.date - pd.DateOffset(years=17.0)
    df.loc[0, 'age_years'] = np.floor(df.loc[0].age_exact_years)
    df.loc[0, 'age_range'] = sim.modules['Demography'].AGE_RANGE_LOOKUP[df.loc[0].age_years]

    assert not df.loc[0, 'is_pregnant']
    assert pd.isnull(df.loc[0, 'date_of_last_pregnancy'])
    assert pd.isnull(df.loc[0, 'si_date_of_last_delivery'])

    # Set the probability of becoming pregnancy to 1
    sim = set_prob_of_pregnancy_to_one(sim)

    # Run the 'set_new_pregnancies' function on the selected population
    sb_event = simplified_births.SimplifiedBirthsPoll(module=sim.modules['SimplifiedBirths'])
    sb_event.set_new_pregnancies()

    # Check that woman is now pregnant, has a date of pregnancy of today, and a delivery date in the future
    assert df.loc[0, 'is_pregnant']
    assert sim.date == df.loc[0, 'date_of_last_pregnancy']
    assert (sim.date + pd.DateOffset(months=9)) == df.loc[0, 'si_date_of_last_delivery']

    # Update time to after the date of delivery and run the 'do_deliveries' function
    sim.date = df.loc[0, 'si_date_of_last_delivery'] + pd.DateOffset(days=0)
    sb_event.do_deliveries()

    # Check that woman's properties are updated accordingly
    assert not df.loc[0, 'is_pregnant']

    # Check that a baby is born and has an appropriate value for 'nb_breastfeeding_status'
    df = sim.population.props
    assert (2 == len(df.loc[~pd.isnull(df.date_of_birth)]))
    assert (df.loc[1, 'mother_id'] >= 0) & (df.loc[1, 'date_of_birth'] == sim.date)
    assert df.loc[0, 'nb_breastfeeding_status'] in ('none', 'non_exclusive', 'exclusive')

    check_property_integrity(sim)


def test_no_pregnancy_among_in_ineligible_populations(seed):
    """If no one in the population is pregnant, the SimplifiedBirthsPoll should not result in any pregnancies:"""

    # running pregnancy event with zero pregnancy probability
    sim = get_sim(seed=seed, popsize=400)

    # Set the probability of becoming pregnancy to 1
    sim = set_prob_of_pregnancy_to_one(sim)

    # define dataframe to make all women eligible
    df = sim.population.props
    df.loc[range(0, 100), 'sex'] = 'M'
    df.loc[range(100, 200), 'is_alive'] = False
    df.loc[range(200, 300), ['age_years', 'age_range']] = (14, '10-14')
    df.loc[range(300, 400), ['age_years', 'age_range']] = (50, '50-54')

    # make no one pregnant before the SimplifiedBirthsPoll is run:
    df.loc[:, 'is_pregnant'] = False

    # Run the 'set_new_pregnancies' function
    sb_event = simplified_births.SimplifiedBirthsPoll(module=sim.modules['SimplifiedBirths'])
    sb_event.set_new_pregnancies()

    # Check that no one became pregnant
    assert not df.is_pregnant.any()


def test_no_births_if_no_one_is_pregnant(seed):
    """If no one is pregnant, the SimplifiedBirthEvent should not result in any births:"""
    sim = get_sim(seed=seed, popsize=10_000)
    df = sim.population.props

    # confirm that the woman is alive and eligible to become pregnant
    df.loc[:, 'is_alive'] = True
    df.loc[:, 'is_pregnant'] = False

    # Run the do_deliveries function - and check that there are no births:
    sb_event = simplified_births.SimplifiedBirthsPoll(module=sim.modules['SimplifiedBirths'])
    sb_event.do_deliveries()

    # get population dataframe
    df = sim.population.props
    assert 0 == len(df.loc[df.date_of_birth.notna() & (df.mother_id >= 0)])


@pytest.mark.slow
def test_standard_run_using_simplified_birth_module(seed):
    """Run the model using the SimplifiedBirthsPoll and SimplifiedBirthEvent and check that properties are
    maintained correctly and that some number of births result."""

    # Get simulation object
    sim = get_sim(seed=seed, popsize=10_000)

    # Force all new borns to be given a breastfeeding status of 'exclusive'
    sim.modules['SimplifiedBirths'].parameters['prob_breastfeeding_type'] = [0, 0, 1]

    # Cause the 'check on configuration' of properties to run daily during the simulation.
    class CheckProperties(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(days=1))

        def apply(self, population):
            check_property_integrity(self.module.sim)

    sim.schedule_event(CheckProperties(sim.modules['SimplifiedBirths']), sim.date)

    # Run the sim
    sim.simulate(end_date=Date(2015, 1, 1))

    # check that there are births happening
    df = sim.population.props
    born_in_sim = df.loc[~pd.isnull(df.date_of_birth) & (df.mother_id >= 0)]
    assert len(born_in_sim) > 0

    # check that all newborns have their breastfeeding status set to exclusive if aged under six months
    assert (df.loc[
                df.index.isin(born_in_sim.index) &
                (df.age_exact_years < 0.5),
                'nb_breastfeeding_status'] == 'exclusive'
            ).all()
    check_dtypes(sim)


@pytest.mark.slow
def test_other_modules_running_with_simplified_births_module():
    """Run a "full simulation" using the simplified_births module and other disease modules"""
    sim = Simulation(
        start_date=Date(2010, 1, 1),
        log_config={
            'custom_levels': {
                '*': logging.WARNING,
            }
        }
    )
    sim.register(
        *fullmodel(
            resourcefilepath=resourcefilepath,
            use_simplified_births=True,
            module_kwargs={"HealthSystem": {"disable": True}},
        )
    )
    sim.make_initial_population(n=1_000)
    sim.simulate(end_date=Date(2011, 12, 31))
    check_property_integrity(sim)
    check_dtypes(sim)
