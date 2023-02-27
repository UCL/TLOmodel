import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx
from scipy.stats import norm

from tlo import Date, Simulation
from tlo.lm import LinearModel, Predictor
from tlo.methods import demography, enhanced_lifestyle, healthsystem, simplified_births, stunting
from tlo.methods.demography import AgeUpdateEvent
from tlo.methods.stunting import HSI_Stunting_ComplementaryFeeding
from tlo.util import random_date


def get_sim(seed):
    """Return simulation objection with Stunting and other necessary modules registered."""

    start_date = Date(2010, 1, 1)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=start_date, seed=seed)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='all'),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 stunting.Stunting(resourcefilepath=resourcefilepath),
                 stunting.StuntingPropertiesOfOtherModules(),
                 )
    return sim


def check_dtypes(sim):
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def check_params_read(sim):
    """Check that every value has been read-in successfully"""
    p = sim.modules['Stunting'].parameters
    for param_name, param_type in sim.modules['Stunting'].PARAMETERS.items():
        assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
        assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
        assert isinstance(p[param_name],
                          param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                   f'resourcefile.'


def test_models(seed):
    """Check that all the models defined work"""
    popsize = 1000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    models = stunting.Models(sim.modules['Stunting'])
    df = sim.population.props

    models.lm_prob_becomes_stunted.predict(df.loc[df.is_alive])
    models.lm_prob_progression_to_severe_stunting.predict(df.loc[df.is_alive])
    models.lm_prob_natural_recovery.predict(df.loc[df.is_alive])


@pytest.mark.slow
def test_basic_run(seed):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(years=5)
    popsize = 1000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    check_dtypes(sim)
    check_params_read(sim)

    sim.simulate(end_date=sim.start_date + dur)
    check_dtypes(sim)


@pytest.mark.slow
def test_initial_prevalence_of_stunting(seed):
    """Check that initial prevalence of stunting is as expected"""
    sim = get_sim(seed)
    sim.make_initial_population(n=50_000)

    # Make all the population under five years old and re-run `initialise_population` for `Stunting`
    sim.population.props.date_of_birth = sim.population.props['is_alive'].apply(
        lambda _: random_date(sim.date - pd.DateOffset(years=5), sim.date - pd.DateOffset(days=1), sim.rng)
    )
    age_update_event = AgeUpdateEvent(sim.modules['Demography'], sim.modules['Demography'].AGE_RANGE_LOOKUP)
    age_update_event.apply(sim.population)
    sim.modules['Stunting'].initialise_population(sim.population)

    df = sim.population.props

    def get_agegrp(_exact):
        if _exact < 0.5:
            return '0_5mo'
        elif _exact < 1.0:
            return '6_11mo'
        elif _exact < 2.0:
            return '12_23mo'
        elif _exact < 3.0:
            return '24_35mo'
        elif _exact < 4.0:
            return '36_47mo'
        elif _exact < 5.0:
            return '48_59mo'
        else:
            return np.nan

    df['agegp'] = df.age_exact_years.apply(get_agegrp)
    df['any_stunted'] = (df['un_HAZ_category'] != 'HAZ>=-2')
    df['severely_stunted'] = (df['un_HAZ_category'] == 'HAZ<-3')

    prevalence_of_stunting_by_age = df.groupby(by=['agegp'])['any_stunted'].mean()
    prevalence_of_severe_stunting_given_any_stunting_by_age = df.loc[df['any_stunted']].groupby(by=['agegp'])[
        'severely_stunted'].mean()

    # Compare with targets
    for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
        mean, stdev = sim.modules['Stunting'].parameters[f'prev_HAZ_distribution_age_{agegp}']
        haz_distribution = norm(loc=mean, scale=stdev)

        assert haz_distribution.cdf(-2.0) == approx(prevalence_of_stunting_by_age[agegp], abs=0.02)
        assert (haz_distribution.cdf(-3.0) / haz_distribution.cdf(-2.0)) == approx(
            prevalence_of_severe_stunting_given_any_stunting_by_age[agegp], abs=0.05)


def test_polling_event_onset(seed):
    """Test that polling event causes onset of stunting correctly"""
    popsize = 1000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    # Make the risk of becoming stunted very high
    params = sim.modules['Stunting'].parameters
    params['base_inc_rate_stunting_by_agegp'] = [1.0] * len(params['base_inc_rate_stunting_by_agegp'])

    # Initialise Simulation
    sim.simulate(end_date=sim.start_date)

    # Get polling event
    poll = stunting.StuntingPollingEvent(sim.modules['Stunting'])

    # Make all the population not stunted and low wealth quantile (so at high risk)
    df = sim.population.props
    df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'
    df.loc[df.is_alive, 'li_wealth'] = 1

    # Run the poll
    poll.apply(sim.population)

    # Check that everyone under age of 5 becomes stunted and no-one over 5 is stunted
    assert (df.loc[df.is_alive & (df.age_years < 5), 'un_HAZ_category'] == '-3<=HAZ<-2').all()
    assert (df.loc[df.is_alive & (df.age_years >= 5), 'un_HAZ_category'] == 'HAZ>=-2').all()


def test_polling_event_recovery(seed):
    """Test that the polling event causes recovery correctly"""
    popsize = 1000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    # Make the risk of recovering very high
    params = sim.modules['Stunting'].parameters
    params['mean_years_to_1stdev_natural_improvement_in_stunting'] = 1e-3

    # Initialise Simulation
    sim.simulate(end_date=sim.start_date)

    # Get polling event
    poll = stunting.StuntingPollingEvent(sim.modules['Stunting'])

    # Make all the population stunted / severely stunted
    df = sim.population.props
    orig = pd.Series(index=df.index, data=np.random.choice(['HAZ<-3', '-3<=HAZ<-2'], len(df), p=[0.5, 0.5]))
    df.loc[df.is_alive & (df.age_years < 5), 'un_HAZ_category'] = orig.loc[df.is_alive & (df.age_years < 5)]
    assert df.loc[df.is_alive & (df.age_years < 5), 'un_HAZ_category'].isin(['HAZ<-3', '-3<=HAZ<-2']).all()

    # Run the poll
    poll.apply(sim.population)

    # Check that everyone under 5 has moved "up" one level (and that those over 5 are still not stunted)
    assert (df.loc[df.is_alive & (df.age_years < 5) & (orig == '-3<=HAZ<-2'), 'un_HAZ_category'] == 'HAZ>=-2').all()
    assert (df.loc[df.is_alive & (df.age_years < 5) & (orig == 'HAZ<-3'), 'un_HAZ_category'] == '-3<=HAZ<-2').all()
    assert (df.loc[df.is_alive & (df.age_years >= 5), 'un_HAZ_category'] == 'HAZ>=-2').all()


def test_polling_event_progression(seed):
    """Test that the polling event causes progression correctly"""
    popsize = 1000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    # Make the risk of progression be very high and no recovery
    params = sim.modules['Stunting'].parameters
    params['r_progression_severe_stunting_by_agegp'] = [1.0] * len(params['r_progression_severe_stunting_by_agegp'])
    params['mean_years_to_1stdev_natural_improvement_in_stunting'] = float('inf')

    # Initialise Simulation
    sim.simulate(end_date=sim.start_date)

    # Get polling event
    poll = stunting.StuntingPollingEvent(sim.modules['Stunting'])

    # Make all the population stunted / severely stunted
    df = sim.population.props
    orig = pd.Series(index=df.index, data=np.random.choice(['HAZ<-3', '-3<=HAZ<-2'], len(df), p=[0.5, 0.5]))
    df.loc[df.is_alive & (df.age_years < 5), 'un_HAZ_category'] = orig.loc[df.is_alive & (df.age_years < 5)]
    assert df.loc[df.is_alive & (df.age_years < 5), 'un_HAZ_category'].isin(['HAZ<-3', '-3<=HAZ<-2']).all()

    # Run the poll
    poll.apply(sim.population)

    # Check that those eligible have moved "down" one level (and those over 5 are still not stunted)
    assert (df.loc[df.is_alive & (df.age_years < 5) & (orig == '-3<=HAZ<-2'), 'un_HAZ_category'] == 'HAZ<-3').all()
    assert (df.loc[df.is_alive & (df.age_years < 5) & (orig == 'HAZ<-3'), 'un_HAZ_category'] == 'HAZ<-3').all()
    assert (df.loc[df.is_alive & (df.age_years >= 5), 'un_HAZ_category'] == 'HAZ>=-2').all()


def test_routine_assessment_for_chronic_undernutrition_if_stunted_and_correctly_diagnosed(seed):
    """Check that a call to `do_routine_assessment_for_chronic_undernutrition` can lead to immediate recovery for a
    stunted child (via an HSI), if there is checking and correct diagnosis."""
    popsize = 100
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    sim.modules['HealthSystem'].reset_queue()

    # Make one person have non-severe stunting
    df = sim.population.props
    person_id = 0
    df.loc[person_id, 'age_years'] = 2
    df.loc[person_id, 'age_exact_year'] = 2.0
    df.loc[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'

    # Make the probability of stunting checking/diagnosis as 1.0
    sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 1.0

    # Subject the person to `do_routine_assessment_for_chronic_undernutrition`
    sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)

    # Check that there is an HSI scheduled for this person
    hsi_event_scheduled = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)
    ]
    assert 1 == len(hsi_event_scheduled)
    assert sim.date == hsi_event_scheduled[0][0]
    the_hsi_event = hsi_event_scheduled[0][1]
    assert person_id == the_hsi_event.target

    # Make probability of treatment success is 1.0 (consumables are available through use of `ignore_cons_constraints`)
    sim.modules['Stunting'].parameters[
        'effectiveness_of_complementary_feeding_education_in_stunting_reduction'] = 1.0
    sim.modules['Stunting'].parameters[
        'effectiveness_of_food_supplementation_in_stunting_reduction'] = 1.0

    # Run the HSI event
    the_hsi_event.run(squeeze_factor=0.0)

    # Check that the person is not longer stunted
    assert df.at[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Check that there is a follow-up appointment scheduled
    hsi_event_scheduled_after_first_appt = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)
    ]
    assert 2 == len(hsi_event_scheduled_after_first_appt)
    assert (sim.date + pd.DateOffset(months=6)) == hsi_event_scheduled_after_first_appt[1][0]
    the_follow_up_hsi_event = hsi_event_scheduled_after_first_appt[1][1]

    # Run the Follow-up HSI event
    the_follow_up_hsi_event.run(squeeze_factor=0.0)

    # Check that after running the following appointments there are no further appointments scheduled
    assert hsi_event_scheduled_after_first_appt == [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)
    ]


def test_routine_assessment_for_chronic_undernutrition_if_stunted_but_no_checking(seed):
    """Check that a call to `do_routine_assessment_for_chronic_undernutrition` does not lead to an HSI for a stunted
    child, if there is no checking/diagnosis."""
    popsize = 100
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    sim.modules['HealthSystem'].reset_queue()

    # Make one person have severe stunting
    df = sim.population.props
    person_id = 0
    df.loc[person_id, 'age_years'] = 2
    df.loc[person_id, 'age_exact_year'] = 2.0
    df.loc[person_id, 'un_HAZ_category'] = 'HAZ<-3'

    # Make the probability of stunting checking/diagnosis as 0.0
    sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 0.0

    # Subject the person to `do_routine_assessment_for_chronic_undernutrition`
    sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)

    # Check that there is no HSI scheduled for this person
    hsi_event_scheduled = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
                           isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)]
    assert 0 == len(hsi_event_scheduled)

    # Then make the probability of stunting checking/diagnosis as 1.0 and check the HSI is scheduled for this person
    sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] = 1.0
    sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)
    hsi_event_scheduled = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
                           isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)]
    assert 1 == len(hsi_event_scheduled)


def test_routine_assessment_for_chronic_undernutrition_if_not_stunted(seed):
    """Check that a call to `do_routine_assessment_for_chronic_undernutrition` does not lead to an HSI if there is no
    stunting."""
    popsize = 100
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    sim.modules['HealthSystem'].reset_queue()

    # Make one person have no stunting
    df = sim.population.props
    person_id = 0
    df.loc[person_id, 'age_years'] = 2
    df.loc[person_id, 'age_exact_year'] = 2.0
    df.loc[person_id, 'un_HAZ_category'] = 'HAZ>=-2'

    # Subject the person to `do_routine_assessment_for_chronic_undernutrition`
    sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)

    # Check that there is no HSI scheduled for this person
    hsi_event_scheduled = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
                           isinstance(ev[1], HSI_Stunting_ComplementaryFeeding)]
    assert 0 == len(hsi_event_scheduled)


def test_math_of_incidence_calcs(seed):
    """Check that incidence of new stunting happens at the rate that is intended"""
    popsize = 50_000
    sim = get_sim(seed)
    sim.make_initial_population(n=popsize)

    # Define the annual probability of becoming stunted
    annual_prob = 0.25

    # Let there no recovery and no progression
    params = sim.modules['Stunting'].parameters
    params['mean_years_to_1stdev_natural_improvement_in_stunting'] = 1e6
    params['r_progression_severe_stunting_by_agegp'] = [0.0] * len(params['r_progression_severe_stunting_by_agegp'])

    # The population will be entirely composed of children under five not stunted
    df = sim.population.props
    df.loc[df.is_alive, 'age_years'] = 2
    df.loc[df.is_alive, 'age_exact_years'] = 2.0
    df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'

    # Initialise Simulation
    sim.simulate(end_date=sim.start_date)

    # Over-write model for annual prob of onset (a fixed value and no risk factors):
    sim.modules['Stunting'].models.lm_prob_becomes_stunted = LinearModel.multiplicative(
        Predictor('age_exact_years').when('< 5', annual_prob).otherwise(0.0)
    )

    # Create a comparison simple calculation:
    monthly_risk = 1.0 - np.exp(np.log(1.0 - annual_prob) * (1.0 / 12.0))  # 0.023688424222606752
    x0 = 1.0  # Initial proportion of people not stunted in simple calculation
    x = x0

    # Run polling event once per month for a year
    poll = stunting.StuntingPollingEvent(sim.modules['Stunting'])
    for date in pd.date_range(Date(2010, 1, 1), sim.date + pd.DateOffset(years=1), freq='MS', closed='left'):
        # Do incidence of stunting through the model's polling event
        sim.date = date
        poll.apply(sim.population)

        # Do incidence of stunting through the 'simple calculation'
        x = x * (1 - monthly_risk)

    # Compute proportion of person stunted i the model and in the simple calculation
    prop_model = (df.loc[df.is_alive, 'un_HAZ_category'] == '-3<=HAZ<-2').mean()
    prop_simple = 1.0 - x

    # Check that the proportions of persons stunted after one year matches the target of `annual_prob`
    assert annual_prob == \
           approx(prop_simple) == \
           approx(1.0 - (1.0 - monthly_risk) ** 12) == \
           approx(prop_model, abs=0.008)
