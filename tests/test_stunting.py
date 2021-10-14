import os
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx
from scipy.stats import norm

from tlo import Date, Simulation, logging
from tlo.methods import (
    demography,
    stunting,
    enhanced_lifestyle,
    healthsystem,
    simplified_births,
)
from tlo.methods.demography import AgeUpdateEvent
from tlo.util import random_date


def get_sim(tmpdir):
    """Return simulation objection with Stunting and other necessary modules registered."""

    start_date = Date(2010, 1, 1)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=start_date, seed=0, show_progress_bar=False, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            "tlo.methods.stunting": logging.INFO}
    })

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
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


def test_models(tmpdir):
    """Check that all the models defined work"""
    popsize = 1000
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)

    models = stunting.Models(sim.modules['Stunting'])
    df = sim.population.props

    models.lm_prob_becomes_stunted.predict(df.loc[df.is_alive])
    models.lm_prob_progression_to_severe_stunting.predict(df.loc[df.is_alive])
    models.lm_prob_natural_recovery.predict(df.loc[df.is_alive])
    models.lm_prob_improvement_with_interventions.predict(df.loc[df.is_alive])


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(years=5)
    popsize = 1000
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)

    check_dtypes(sim)
    check_params_read(sim)

    sim.simulate(end_date=sim.start_date + dur)
    check_dtypes(sim)


def test_initial_prevalence_of_stunting(tmpdir):
    """Check that initial prevalence of stunting is as expected"""
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=50_000)

    # Make all the population under five and re-run `initialise_population` for `Stunting`
    sim.population.props.date_of_birth = sim.population.props['is_alive'].apply(lambda _:
                                                                                random_date(
                                                                                    sim.date - pd.DateOffset(years=5),
                                                                                    sim.date - pd.DateOffset(days=1),
                                                                                    sim.rng)
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

        assert haz_distribution.cdf(-2.0) == approx(prevalence_of_stunting_by_age[agegp], abs=0.011)
        assert (haz_distribution.cdf(-3.0) / haz_distribution.cdf(-2.0)) == approx(
            prevalence_of_severe_stunting_given_any_stunting_by_age[agegp], abs=0.02)

#
# def test_stunting_polling(tmpdir):
#     """Check polling events leads to incident cases"""
#     # get simulation object:
#     dur = pd.DateOffset(months=3)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     # Make incidence of stunting very high :
#     params = sim.modules['Stunting'].parameters
#     for p in params:
#         if p.startswith('base_inc_rate_stunting_by_agegp'):
#             params[p] = [3 * v for v in params[p]]
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=sim.start_date + dur)
#     sim.event_queue.queue = []
#
#     # Run polling event: check that a stunting incident case is produced:
#     polling = StuntingPollingEvent(sim.modules['Stunting'])
#     polling.apply(sim.population)
#     assert len([q for q in sim.event_queue.queue if isinstance(q[2], StuntingOnsetEvent)]) > 0
#
#
# def test_recovery_moderate_stunting(tmpdir):
#     """Check natural recovery of moderate stunting, by reducing the probability of remained stunted and
#     reducing progression to severe stunting"""
#     dur = pd.DateOffset(days=0)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=sim.start_date + dur)
#     sim.event_queue.queue = []  # clear the queue
#
#     # increase incidence of stunting
#     sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # remove progression to severe stunting
#     sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 0.0)
#
#     # increase probability of natural recovery (without interventions)
#     params = sim.modules['Stunting'].parameters
#     params['prob_remained_stunted_in_the_next_3months'] = 0.0
#
#     # Get person to use:
#     df = sim.population.props
#     under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#     person_id = under5s.index[0]
#     assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'
#
#     # Run Stunting Polling event to get new incident cases:
#     polling = StuntingPollingEvent(module=sim.modules['Stunting'])
#     polling.apply(sim.population)
#
#     # Check that there is a StuntingOnsetEvent scheduled for this person:
#     onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingOnsetEvent)
#                          ][0]
#     date_of_scheduled_onset = onset_event_tuple[0]
#     onset_event = onset_event_tuple[1]
#     assert date_of_scheduled_onset > sim.date
#
#     # Run the onset event:
#     sim.date = date_of_scheduled_onset
#     onset_event.apply(person_id=person_id)
#
#     # Check properties of this individual: should now be moderately stunted
#     person = df.loc[person_id]
#     assert person['un_ever_stunted']
#     assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#     assert person['un_last_stunting_date_of_onset'] == sim.date
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#     assert pd.isnull(person['un_stunting_recovery_date'])
#
#     # Check that there is a StuntingRecoveryEvent scheduled for this person:
#     recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingRecoveryEvent)
#                          ][0]
#     date_of_scheduled_recov = recov_event_tuple[0]
#     recov_event = recov_event_tuple[1]
#     assert date_of_scheduled_recov > sim.date
#
#     # Run the recovery event:
#     sim.date = date_of_scheduled_recov
#     recov_event.apply(person_id=person_id)
#
#     # Check properties of this individual
#     person = df.loc[person_id]
#     assert person['un_HAZ_category'] == 'HAZ>=-2'
#     assert person['un_stunting_recovery_date'] == sim.date
#
#
# def test_recovery_severe_stunting(tmpdir):
#     """Check natural recovery of severe stunting, by reducing the probability of remained stunted"""
#     dur = pd.DateOffset(days=0)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=start_date + dur)
#     sim.event_queue.queue = []  # clear the queue
#
#     # increase incidence of stunting and progression to severe
#     sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#     sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # increase probability of natural recovery (without interventions)
#     params = sim.modules['Stunting'].parameters
#     params['prob_remained_stunted_in_the_next_3months'] = 0.0
#
#     # Get person to use:
#     df = sim.population.props
#     under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#     person_id = under5s.index[0]
#     assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'
#
#     # Run Stunting Polling event to get new incident cases:
#     polling = StuntingPollingEvent(module=sim.modules['Stunting'])
#     polling.apply(sim.population)
#
#     # Check that there is a StuntingOnsetEvent scheduled for this person:
#     onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingOnsetEvent)
#                          ][0]
#     date_of_scheduled_onset = onset_event_tuple[0]
#     onset_event = onset_event_tuple[1]
#     assert date_of_scheduled_onset > sim.date
#
#     # Run the onset event:
#     sim.date = date_of_scheduled_onset
#     onset_event.apply(person_id=person_id)
#
#     # Check properties of this individual: should now be moderately stunted
#     person = df.loc[person_id]
#     assert person['un_ever_stunted']
#     assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#     assert person['un_last_stunting_date_of_onset'] == sim.date
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#     assert pd.isnull(person['un_stunting_recovery_date'])
#
#     # Check that there is a ProgressionSevereStuntingEvent scheduled for this person:
#     progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                                isinstance(event_tuple[1], ProgressionSevereStuntingEvent)
#                                ][0]
#     date_of_scheduled_progression = progression_event_tuple[0]
#     progression_event = progression_event_tuple[1]
#     assert date_of_scheduled_progression > sim.date
#
#     # Run the progression to severe stunting event:
#     sim.date = date_of_scheduled_progression
#     progression_event.apply(person_id=person_id)
#
#     # Check properties of this individual: (should now be severely stunted and without a scheduled death date)
#     person = df.loc[person_id]
#     assert person['un_ever_stunted']
#     assert person['un_HAZ_category'] == 'HAZ<-3'
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#     assert pd.isnull(person['un_stunting_recovery_date'])
#
#     # Check that there is a StuntingRecoveryEvent scheduled for this person:
#     recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingRecoveryEvent)
#                          ][0]
#     date_of_scheduled_recov = recov_event_tuple[0]
#     recov_event = recov_event_tuple[1]
#     assert date_of_scheduled_recov > sim.date
#
#     # Run the recovery event:
#     sim.date = date_of_scheduled_recov
#     recov_event.apply(person_id=person_id)
#
#     # Check properties of this individual (can only improve by 1 sd in HAZ)
#     person = df.loc[person_id]
#     assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#     assert pd.isnull(person['un_stunting_recovery_date'])
#
#     # check they have no symptoms:
#     assert 0 == len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Stunting']))
#
#
#
#
# def test_treatment(tmpdir):
#     """ Test that providing a treatment prevent further stunting """
#
#     dur = pd.DateOffset(days=0)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=start_date + dur)
#     sim.event_queue.queue = []  # clear the queue
#
#     # increase incidence of stunting
#     sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # remove progression to severe
#     sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 0.0)
#
#     # increase intervention effectiveness
#     sim.modules['Stunting'].stunting_improvement_based_on_interventions = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # Get person to use:
#     df = sim.population.props
#     under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#     person_id = under5s.index[0]
#     assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'
#
#     # Run Stunting Polling event to get new incident cases:
#     polling = StuntingPollingEvent(module=sim.modules['Stunting'])
#     polling.apply(sim.population)
#
#     # Check that there is a StuntingOnsetEvent scheduled for this person:
#     onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingOnsetEvent)
#                          ][0]
#     date_of_scheduled_onset = onset_event_tuple[0]
#     onset_event = onset_event_tuple[1]
#     assert date_of_scheduled_onset > sim.date
#
#     # Run the onset event:
#     sim.date = date_of_scheduled_onset
#     onset_event.apply(person_id=person_id)
#
#     # Check properties of this individual: should now be moderately stunted
#     person = df.loc[person_id]
#     assert person['un_ever_stunted']
#     assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#     assert person['un_last_stunting_date_of_onset'] == sim.date
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#     assert pd.isnull(person['un_stunting_recovery_date'])
#
#     # Run Recovery Polling event to apply intervention effectiveness:
#     recovery_polling = StuntingRecoveryPollingEvent(module=sim.modules['Stunting'])
#     recovery_polling.apply(sim.population)
#
#     # Check that there is a StuntingRecoveryEvent scheduled for this person:
#     recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                             isinstance(event_tuple[1], StuntingRecoveryEvent)
#                             ][0]
#     date_of_scheduled_recovery = recovery_event_tuple[0]
#     recovery_event = recovery_event_tuple[1]
#     assert date_of_scheduled_recovery > sim.date
#
#     # Run the recovery event
#     sim.date = date_of_scheduled_recovery
#     recovery_event.apply(person_id=person_id)
#
#     # Check properties of this individual
#     person = df.loc[person_id]
#     assert person['un_HAZ_category'] == 'HAZ>=-2'
#     assert person['un_stunting_recovery_date'] == sim.date
#
#
# def test_use_of_HSI_complementary_feeding_education_only(tmpdir):
#     """ Check that the HSIs works"""
#     dur = pd.DateOffset(days=0)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=start_date + dur)
#     sim.event_queue.queue = []  # clear the queue
#
#     # increase incidence of stunting
#     sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # reduce progression to severe stunting
#     sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 0.0)
#
#     # increase intervention effectiveness
#     sim.modules['Stunting'].stunting_improvement_based_on_interventions = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # Get person to use:
#     df = sim.population.props
#     under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#     person_id = under5s.index[0]
#     assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'
#
#     # Run Stunting Polling event to get new incident cases:
#     polling = StuntingPollingEvent(module=sim.modules['Stunting'])
#     polling.apply(sim.population)
#
#     # Check that there is a StuntingOnsetEvent scheduled for this person:
#     onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], StuntingOnsetEvent)
#                          ][0]
#     date_of_scheduled_onset = onset_event_tuple[0]
#     onset_event = onset_event_tuple[1]
#     assert date_of_scheduled_onset > sim.date
#
#     # Run the onset event:
#     sim.date = date_of_scheduled_onset
#     onset_event.apply(person_id=person_id)
#
#     # Check properties of this individual: should now be moderately stunted
#     person = df.loc[person_id]
#     assert person['un_ever_stunted']
#     assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#     assert person['un_last_stunting_date_of_onset'] == sim.date
#     assert pd.isnull(person['un_stunting_recovery_date'])
#     assert pd.isnull(person['un_stunting_tx_start_date'])
#
#     # Run the HSI event
#     hsi = HSI_complementary_feeding_education_only(person_id=person_id, module=sim.modules['Stunting'])
#     hsi.run(squeeze_factor=0.0)
#
#     # Check that person is now on treatment:
#     assert sim.date == df.at[person_id, 'un_stunting_tx_start_date']
#
#     # Check that a CureEvent has been scheduled
#     recovery_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
#                       isinstance(event_tuple[1], StuntingRecoveryEvent)][0]
#
#     # Run the CureEvent
#     recovery_event.apply(person_id=person_id)
#
#     # Check that the person is cured and is alive still:
#     person = df.loc[person_id]
#     assert person['is_alive']
#     assert person['un_HAZ_category'] == 'HAZ>=-2'
#     assert not pd.isnull(person['un_stunting_recovery_date'])
#
#
# def test_use_of_HSI(tmpdir):
#     """ Check that the HSIs works"""
#
#     def test_use_HSI_by_intervention(intervention):
#         dur = pd.DateOffset(days=0)
#         popsize = 1000
#         sim = get_sim(tmpdir)
#
#         sim.make_initial_population(n=popsize)
#         sim.simulate(end_date=start_date + dur)
#         sim.event_queue.queue = []  # clear the queue
#
#         # increase incidence of stunting
#         sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
#             LinearModelType.MULTIPLICATIVE, 1.0)
#
#         # reduce progression to severe stunting
#         sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
#             LinearModelType.MULTIPLICATIVE, 1.0)
#
#         # increase intervention effectiveness
#         sim.modules['Stunting'].stunting_improvement_based_on_interventions = LinearModel(
#             LinearModelType.MULTIPLICATIVE, 1.0)
#
#         # Get person to use:
#         df = sim.population.props
#         under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#         person_id = under5s.index[0]
#         assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'
#
#         # Run Stunting Polling event to get new incident cases:
#         polling = StuntingPollingEvent(module=sim.modules['Stunting'])
#         polling.apply(sim.population)
#
#         # Check that there is a StuntingOnsetEvent scheduled for this person:
#         onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                              isinstance(event_tuple[1], StuntingOnsetEvent)
#                              ][0]
#         date_of_scheduled_onset = onset_event_tuple[0]
#         onset_event = onset_event_tuple[1]
#         assert date_of_scheduled_onset > sim.date
#
#         # Run the onset event:
#         sim.date = date_of_scheduled_onset
#         onset_event.apply(person_id=person_id)
#
#         # Check properties of this individual: should now be moderately stunted
#         person = df.loc[person_id]
#         assert person['un_ever_stunted']
#         assert person['un_HAZ_category'] == '-3<=HAZ<-2'
#         assert person['un_last_stunting_date_of_onset'] == sim.date
#         assert pd.isnull(person['un_stunting_recovery_date'])
#         assert pd.isnull(person['un_stunting_tx_start_date'])
#
#         # Run the HSI event
#         if intervention == 'education_only':
#             hsi = HSI_complementary_feeding_education_only(person_id=person_id, module=sim.modules['Stunting'])
#             hsi.run(squeeze_factor=0.0)
#         if intervention == 'supplementary_foods':
#             hsi = HSI_Stunting_ComplementaryFeeding(
#                 person_id=person_id, module=sim.modules['Stunting'])
#             hsi.run(squeeze_factor=0.0)
#
#         # Check that person is now on treatment:
#         assert sim.date == df.at[person_id, 'un_stunting_tx_start_date']
#
#         # Check that there is a StuntingRecoveryEvent scheduled for this person:
#         recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                                 isinstance(event_tuple[1], StuntingRecoveryEvent)
#                                 ][0]
#         date_of_scheduled_recovery = recovery_event_tuple[0]
#         recovery_event = recovery_event_tuple[1]
#         assert date_of_scheduled_recovery > sim.date
#
#         # Run the recovery event
#         sim.date = date_of_scheduled_recovery
#         recovery_event.apply(person_id=person_id)
#
#         # Run the Recovery/ CureEvent
#         recovery_event.apply(person_id=person_id)
#
#         # Check that the person is cured and is alive still:
#         person = df.loc[person_id]
#         assert person['is_alive']
#         assert person['un_HAZ_category'] == 'HAZ>=-2'
#
#     test_use_HSI_by_intervention(intervention='education_only')
#     test_use_HSI_by_intervention(intervention='supplementary_foods')
