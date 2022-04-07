import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, enhanced_lifestyle

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


def __check_properties(df):
    # no one under 15 can be overweight, low exercise, tobacco, excessive alcohol, married
    under15 = df.age_years < 15
    assert not (under15 & df.li_bmi > 0).any()
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
    __check_properties(simulation.population.props)
    simulation.simulate(end_date=end_date)
    __check_properties(simulation.population.props)
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def run_lifestyle_event(simulation):
    """A method to run lifestyle event defined in module Lifestyle"""
    lifestyle_event = enhanced_lifestyle.LifestyleEvent(module=simulation.modules['Lifestyle'])
    lifestyle_event.handle_all_transitions()


def rural_urban_probabilities(simulation, rural_prob: float = 0.0, urban_prob: float = 0.0):
    """set probabilities for transition from rural to urban or urban to rural. for testing purpose we
    assume a probability of 1"""
    # adjust probability per 3 months of change from urban to rural to 1
    simulation.modules['Lifestyle'].parameters['r_rural'] = rural_prob

    # adjust probability per 3 months of change from rural to urban to 1
    simulation.modules['Lifestyle'].parameters['r_urban'] = urban_prob


def low_not_low_exercise_probabilities(simulation, low_prob: float = 0.0, not_low_prob: float = 0.0):
    """Set probabilities for transition from low exercise to not low exercise and from not low exercise to
    low exercise. for testing purpose we assume a probability of 1"""
    # adjust probability per 3 months of change from not low exercise to low exercise to 1
    simulation.modules['Lifestyle'].parameters['r_low_ex'] = low_prob

    # adjust probability per 3 months of change from low exercise to not low exercise to 1
    simulation.modules['Lifestyle'].parameters['r_not_low_ex'] = not_low_prob


def start_stop_smoking_tobacco_probabilities(simulation, start_tob_prob: float = 0.0, stop_tob_prob: float = 0.0):
    """set probabilities of both start and stop smoking tobacco. for testing purposes we assume a
    probability of 1"""
    # adjust probability per 3 months of change from not using tobacco to using tobacco if male
    # age 15-19 wealth level_1 to 1
    simulation.modules['Lifestyle'].parameters['r_tob'] = start_tob_prob

    # adjust probability per 3 months of change from tobacco using to not tobacco using to 1
    simulation.modules['Lifestyle'].parameters['r_not_tob'] = stop_tob_prob


def test_rural_urban_or_urban_rural_transition(simulation):
    """Test the individual ability to move from rural to urban or from urban to rural.
    STEPS   1.  create a population containing one individual
            2.  reset individual properties to match the scenario i.e rural to urban or urban to rural
            3.  adjust rural to urban/urban to rural transition probability to 1
            4.  run LifeStyle Event
            5.  check properties are as expected. if an individual was rural they have to be urban likewise if urban"""

    # create a population dataframe containing one individual
    simulation.make_initial_population(n=1)
    df = simulation.population.props

    # reset properties
    df.loc[0, 'is_alive'] = True
    df.loc[0, 'li_urban'] = False

    # 1. --------------- TEST TRANSITION FROM RURAL TO URBAN -------------------------

    # check the person now rural
    assert not df.loc[0, 'li_urban']

    # adjust rural to urban probability to 1
    rural_urban_probabilities(simulation, 0.0, 1.0)

    # run LifeStyle Event
    run_lifestyle_event(simulation)

    # check transition from rural to urban has happened
    assert df.loc[0, 'li_urban']

    # 2. --------------- TEST TRANSITION FROM URBAN TO RURAL -------------------------

    # adjust urban to rural probability to 1
    rural_urban_probabilities(simulation, 1.0, 0.0)

    # run LifeStyle Event
    run_lifestyle_event(simulation)

    # check transition from rural to urban has happened
    assert not df.loc[0, 'li_urban']

    # 3. ---------------- KILL THE INDIVIDUAL AND TEST TRANSITION FROM RURAL TO URBAN ------------------------
    # reset alive property to False
    df.loc[0, 'is_alive'] = False

    # now that the individual is rural adjust rural to urban probability to 1
    rural_urban_probabilities(simulation, 0.0, 1.0)

    # run LifeStyle Event
    run_lifestyle_event(simulation)

    # check nothing happens
    assert not df.loc[0, 'li_urban'], 'a dead person cannot move from rural to urban'


def test_low_exercise_or_not_low_exercise_transition(simulation):
    """Test individual transition from low to not low exercise and from not low to low exercise
    STEPS   1.  create a population containing one individual
            2.  reset individual properties to match the scenario i.e low exercise to not low, or vice versa
            3.  adjust low exercise / not low exercise transition probability to 1
            4.  run LifeStyle Event
            5.  check properties are as expected. if an individual was low exercise they have to be not low exercise,
                likewise if not low exercise the have to be low exercise"""

    # make initial population
    simulation.make_initial_population(n=1)
    # create a population dataframe
    df = simulation.population.props

    # modify some properties to match the scenario
    df.loc[0, 'is_alive'] = True
    df.loc[0, 'li_low_ex'] = False
    df.loc[0, 'age_years'] = 17

    # 1. --------- TEST TRANSITION FROM NOT LOW EXERCISE TO LOW EXERCISE ---------------
    # check the individual is not low exercise
    assert not df.loc[0, 'li_low_ex']

    # adjust low exercise/not low exercise probabilities
    low_not_low_exercise_probabilities(simulation, 1.0, 0.0)

    # schedule life style event
    run_lifestyle_event(simulation)

    # check the individual is now low exercise
    assert df.loc[0, 'li_low_ex']

    # 2. --------- TEST TRANSITION FROM LOW EXERCISE TO NOT LOW EXERCISE ---------------

    # adjust low exercise/not low exercise probabilities
    low_not_low_exercise_probabilities(simulation, 0.0, 1.0)

    # schedule life style event
    run_lifestyle_event(simulation)

    # check the individual now not low exercise
    assert not df.loc[0, 'li_low_ex']

    # check status of exposed to exercise increase campaign. it should be true in this case
    # todo after a meeting with Andrew or Tim

    # 3. --------- ADJUST AGE AND TEST TRANSITION FROM NOT LOW EXERCISE TO LOW EXERCISE ---------
    # adjust individual age
    df.loc[0, 'age_years'] = 9

    # adjust low exercise/not low exercise probabilities
    low_not_low_exercise_probabilities(simulation, 1.0, 0)

    # confirm the individual is not low exercise
    assert not df.loc[0, 'li_low_ex']

    # schedule life style event
    run_lifestyle_event(simulation)

    # check nothing happens. the individual should still be low exercise
    assert not df.loc[0, 'li_low_ex']

    # 4. --------- KILL THE INDIVIDUAL AND TEST TRANSITION FROM NOT LOW EXERCISE TO LOW EXERCISE --------
    # kill the individual
    df.loc[0, 'is_alive'] = False

    # adjust low exercise probability to 1
    low_not_low_exercise_probabilities(simulation, 1.0, 0.0)

    # confirm the individual is not low exercise
    assert not df.loc[0, 'li_low_ex']

    # schedule life style event
    run_lifestyle_event(simulation)

    # check nothing happens. the individual should still be low exercise
    assert not df.loc[0, 'li_low_ex']


def test_tobacco_use(simulation):
    """Test individual ability to start smoking tobacco and stop smoking if exposed to stop
    tobacco smoking campaign
    STEPS:  1.  create a population dataframe of one individual
            2.  reset the properties to make him eligible to start smoking tobacco
            3.  increase start tobacco smoking probability to 1
            4.  run Lifestyle Event
            5.  check all properties are as expected"""
    simulation.make_initial_population(n=1)
    df = simulation.population.props

    # reset properties
    df.loc[0, 'is_alive'] = True
    df.loc[0, 'sex'] = 'M'
    df.loc[0, 'age_years'] = 17
    df.loc[0, 'li_tob'] = False

    # 1. ------- TEST START SMOKING TOBACCO TO AN ELIGIBLE INDIVIDUAL -------
    # adjust start tobacco smoking probability to 1
    start_stop_smoking_tobacco_probabilities(simulation, 1.0, 0.0)

    # confirm the individual is not already smoking tobacco
    assert not df.loc[0, 'li_tob']

    # run lifestyle event
    run_lifestyle_event(simulation)

    # check the individual has now started smoking tobacco
    assert df.loc[0, 'li_tob']

    # 2. --------TEST STOP SMOKING TOBACCO -----------
    # adjust stop tobacco smoking probability to 1
    start_stop_smoking_tobacco_probabilities(simulation, 0.0, 1.0)

    # confirm that the person is smoking tobacco
    assert df.loc[0, 'li_tob']

    # run lifestyle event
    run_lifestyle_event(simulation)

    # confirm the individual has now stopped smoking tobacco
    assert not df.loc[0, 'li_tob']

    # the check below awaits confirmation from Andrew or Tim on the added properties and logic
    # check status of exposed to campaign stop tobacco smoking. it should be true in this case
    # todo: uncomment the below check if the modifications in Lifestyle module are adopted
    # assert df.loc[0, 'li_exposed_to_campaign_quit_smoking']

    # 3. ------- TEST START SMOKING TOBACCO TO AN INDIVIDUAL LESS THAN 15 YEARS-------
    # reset properties to be less than 15 years old
    df.loc[0, 'age_years'] = 14

    # adjust start tobacco smoking probability to 1
    start_stop_smoking_tobacco_probabilities(simulation, 1.0, 0.0)

    # confirm the individual is not already smoking tobacco
    assert not df.loc[0, 'li_tob']

    # run lifestyle event
    run_lifestyle_event(simulation)

    # check nothing happens
    assert not df.loc[0, 'li_tob']

    # check status of exposed to campaign stop tobacco smoking. it should be true in this case
    # todo after a meeting with andrew


# todo: create remaining tests for excess alcohol, marital status, education,
#  sanitation, access to hand washing, clean drinking water, wood burn stove, high sugar, high salt, BMI, sex workers

if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    t1 = time.time()
    print('Time taken', t1 - t0)
