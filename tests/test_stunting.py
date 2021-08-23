"""
Basic tests for the Stunting Module
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    demography,
    stunting,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    diarrhoea, wasting
)

from tlo.methods.healthsystem import HSI_Event

from tlo.methods.stunting import (
    StuntingPollingEvent,
    StuntingRecoveryPollingEvent,
    StuntingOnsetEvent,
    ProgressionSevereStuntingEvent,
    StuntingRecoveryEvent,
    HSI_complementary_feeding_education_only,
    HSI_complementary_feeding_with_supplementary_foods,
    StuntingLoggingEvent,
    PropertiesOfOtherModules
)

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)


def get_sim(tmpdir):
    """Return simulation objection with Stunting and other necessary modules registered."""
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
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 stunting.Stunting(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 stunting.PropertiesOfOtherModules(),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )
    return sim


def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# def check_configuration_of_properties(sim):
    # # check that the properties are ok:
    #
    # df = sim.population.props
    #
    # # Those that were never stunted, should have normal HAZ score:
    # assert (df.loc[~df.un_ever_stunted & ~df.date_of_birth.isna(), 'un_HAZ_category'] == 'HAZ>=-2').all().all()
    #
    # # Those that were never stunted and not clinically malnourished,
    # # should have not_applicable/null values for all the other properties:
    # # assert pd.isnull(df.loc[(~df.un_ever_stunted & ~df.date_of_birth.isna() &
    # #                         (df.un_clinical_acute_malnutrition == 'well')),
    # #                         ['un_last_stunting_date_of_onset',
    # #                          'un_sam_death_date'
    # #                          ]
    # #                         ]).all().all()
    #
    # # # Those that were ever stunted, should have a HAZ score below <-2
    # # assert (df.loc[df.un_ever_stunted, 'un_HAZ_category'] != 'HAZ>=-2').all().all()
    # #
    # # # Those that had stunting and no treatment, should have either a recovery date or a death_date
    # # # (but not both)
    # # has_recovery_date = ~pd.isnull(df.loc[df.un_ever_stunted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
    # #                                       'un_am_recovery_date'])
    # # has_death_date = ~pd.isnull(df.loc[df.un_ever_stunted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
    # #                                    'un_sam_death_date'])
    #
    # # has_recovery_date_or_death_date = has_recovery_date | has_death_date
    # # has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    # # # assert has_recovery_date_or_death_date.all()
    # # assert not has_both_recovery_date_and_death_date.any()
    #
    # # Those for whom the death date has past should be dead
    # assert not df.loc[df.un_ever_stunted & (df['un_sam_death_date'] < sim.date), 'is_alive'].any()
    # assert not df.loc[(df.un_clinical_acute_malnutrition == 'SAM') & (
    #     df['un_sam_death_date'] < sim.date), 'is_alive'].any()
    #
    # # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
    # #  but not others (among those who are alive)
    # has_symptoms_of_stunting = set(sim.modules['SymptomManager'].who_has('weight_loss'))
    # has_symptoms = set([p for p in has_symptoms_of_stunting if
    #                     'Stunting' in sim.modules['SymptomManager'].causes_of(p, 'weight_loss')
    #                     ])
    #
    # in_current_episode_before_recovery = \
    #     df.is_alive & \
    #     df.un_ever_stunted & \
    #     (df.un_last_stunting_date_of_onset <= sim.date) & \
    #     (sim.date <= df.un_am_recovery_date)
    # set_of_person_id_in_current_episode_before_recovery = set(
    #     in_current_episode_before_recovery[in_current_episode_before_recovery].index
    # )
    #
    # in_current_episode_before_death = \
    #     df.is_alive & \
    #     df.un_ever_stunted & \
    #     (df.un_last_stunting_date_of_onset <= sim.date) & \
    #     (sim.date <= df.un_sam_death_date)
    # set_of_person_id_in_current_episode_before_death = set(
    #     in_current_episode_before_death[in_current_episode_before_death].index
    # )
    #
    # in_current_episode_before_cure = \
    #     df.is_alive & \
    #     df.un_ever_stunted & \
    #     (df.un_last_stunting_date_of_onset <= sim.date) & \
    #     (df.un_acute_malnutrition_tx_start_date <= sim.date) & \
    #     pd.isnull(df.un_am_recovery_date) & \
    #     pd.isnull(df.un_sam_death_date)
    # set_of_person_id_in_current_episode_before_cure = set(
    #     in_current_episode_before_cure[in_current_episode_before_cure].index
    # )
    #
    # assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
    #     set_of_person_id_in_current_episode_before_death
    # )
    #
    # set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
    #     set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
    # )
    # assert set_of_person_id_in_current_episode == has_symptoms


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(months=3)
    popsize = 1000
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    check_dtypes(sim)
    # check_configuration_of_properties(sim)


def test_stunting_polling(tmpdir):
    """Check polling events leads to incident cases"""
    # get simulation object:
    dur = pd.DateOffset(months=3)
    popsize = 1000
    sim = get_sim(tmpdir)

    # Make incidence of stunting very high :
    params = sim.modules['Stunting'].parameters
    for p in params:
        if p.startswith('base_inc_rate_stunting_by_agegp'):
            params[p] = [3 * v for v in params[p]]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Run polling event: check that a stunting incident case is produced:
    polling = StuntingPollingEvent(sim.modules['Stunting'])
    polling.apply(sim.population)
    print(sim.event_queue.queue)
    assert len([q for q in sim.event_queue.queue if isinstance(q[2], StuntingOnsetEvent)]) > 0


def test_recovery_moderate_stunting(tmpdir):
    """Check natural recovery of moderate stunting, by reducing the probability of remained stunted and
    reducing progression to severe stunting"""
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # increase incidence of stunting
    sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # remove progression to severe stunting
    sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # increase probability of natural recovery (without interventions)
    params = sim.modules['Stunting'].parameters
    params['prob_remained_stunted_in_the_next_3months'] = 0.0

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Run Stunting Polling event to get new incident cases:
    polling = StuntingPollingEvent(module=sim.modules['Stunting'])
    polling.apply(sim.population)

    # Check that there is a StuntingOnsetEvent scheduled for this person:
    onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], StuntingOnsetEvent)
                         ][0]
    date_of_scheduled_onset = onset_event_tuple[0]
    onset_event = onset_event_tuple[1]
    assert date_of_scheduled_onset > sim.date

    # Run the onset event:
    sim.date = date_of_scheduled_onset
    onset_event.apply(person_id=person_id)

    # Check properties of this individual: should now be moderately stunted
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert person['un_last_stunting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_stunting_tx_start_date'])
    assert pd.isnull(person['un_stunting_recovery_date'])

    # Check that there is a StuntingRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], StuntingRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run the recovery event:
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_HAZ_category'] == 'HAZ>=-2'
    assert person['un_stunting_recovery_date'] == sim.date


def test_recovery_severe_stunting(tmpdir):
    """Check natural recovery of severe stunting, by reducing the probability of remained stunted"""
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # increase incidence of stunting and progression to severe
    sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase probability of natural recovery (without interventions)
    params = sim.modules['Stunting'].parameters
    params['prob_remained_stunted_in_the_next_3months'] = 0.0

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Run Stunting Polling event to get new incident cases:
    polling = StuntingPollingEvent(module=sim.modules['Stunting'])
    polling.apply(sim.population)

    # Check that there is a StuntingOnsetEvent scheduled for this person:
    onset_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], StuntingOnsetEvent)
                         ][0]
    date_of_scheduled_onset = onset_event_tuple[0]
    onset_event = onset_event_tuple[1]
    assert date_of_scheduled_onset > sim.date

    # Run the onset event:
    sim.date = date_of_scheduled_onset
    onset_event.apply(person_id=person_id)

    # Check properties of this individual: should now be moderately stunted
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert person['un_last_stunting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_stunting_tx_start_date'])
    assert pd.isnull(person['un_stunting_recovery_date'])

    # Check that there is a ProgressionSevereStuntingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereStuntingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe stunting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely stunted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == 'HAZ<-3'
    assert pd.isnull(person['un_stunting_tx_start_date'])
    assert pd.isnull(person['un_stunting_recovery_date'])

    # Check that there is a StuntingRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], StuntingRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run the recovery event:
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)

    # Check properties of this individual (can only improve by 1 sd in HAZ)
    person = df.loc[person_id]
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert pd.isnull(person['un_stunting_tx_start_date'])
    assert pd.isnull(person['un_stunting_recovery_date'])

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Stunting']))


def test_nat_hist_cure_if_death_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to die, it cause the episode to end without
    the person dying."""

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Stunting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of stunting and progression to severe
    sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    # increase parameters in moderate stunting for clinical SAM (MUAC and oedema) to be polled for death
    params = sim.modules['Stunting'].parameters
    params['proportion_-3<=HAZ<-2_with_MUAC<115mm'] = [5 * params['proportion_-3<=HAZ<-2_with_MUAC<115mm']]
    params['proportion_-3<=HAZ<-2_with_MUAC_115-<125mm'] = [params['proportion_-3<=HAZ<-2_with_MUAC_115-<125mm'] / 5]
    params['proportion_oedema_with_HAZ<-2'] = 0.9

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Run Stunting Polling event to get new incident cases:
    polling = StuntingPollingEvent(module=sim.modules['Stunting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately stunted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert person['un_last_stunting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereStuntingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereStuntingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe stunting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely stunted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == 'HAZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run Death Polling Polling event to apply death:
    death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Stunting'])
    death_polling.apply(sim.population)

    # Check that there is a SevereAcuteMalnutritionDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event now
    cure_event = ClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id, module=sim.modules['Stunting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not stunted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_HAZ_category'] == 'HAZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the death event that was originally scheduled - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']


def test_treatment(tmpdir):
    """ Test that providing a treatment prevent death and causes there to be a
    CureEvent (ClinicalAcuteMalnutritionRecoveryEvent) Scheduled """
    """
    This test sets the linear model of acute_malnutrition_recovery_based_on_interventions to be 1.0 (100% cure rate),
    when this lm is called in the do_when_am_treatment function (usually called in HSIs), 100% cure rate schedules the
    ClinicalAcuteMalnutritionRecoveryEvent (CureEvent).
    Death is prevented.  ---> check
    """
    # TODO: CHECK THIS - MAYBE NOT NEEDED

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Stunting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of stunting and progression to severe
    sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase parameters in moderate stunting for clinical SAM (MUAC and oedema) to be polled for death
    params = sim.modules['Stunting'].parameters
    params['proportion_-3<=HAZ<-2_with_MUAC<115mm'] = [5 * params['proportion_-3<=HAZ<-2_with_MUAC<115mm']]
    params['proportion_-3<=HAZ<-2_with_MUAC_115-<125mm'] = [params['proportion_-3<=HAZ<-2_with_MUAC_115-<125mm'] / 5]
    params['proportion_oedema_with_HAZ<-2'] = 0.9

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Run Stunting Polling event to get new incident cases:
    polling = StuntingPollingEvent(module=sim.modules['Stunting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately stunted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert person['un_last_stunting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereStuntingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereStuntingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe stunting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely stunted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == 'HAZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run Death Polling event to apply death:
    death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Stunting'])
    death_polling.apply(sim.population)

    # Check that there is a SevereAcuteMalnutritionDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # # change coverage and treatment effectiveness set to be 100%
    # params['coverage_supplementary_feeding_program'] = 1.0
    # params['coverage_outpatient_therapeutic_care'] = 1.0
    # params['coverage_inpatient_care'] = 1.0
    # params['recovery_rate_with_soy_RUSF'] = 1.0
    # params['recovery_rate_with_CSB++'] = 1.0
    # params['recovery_rate_with_standard_RUTF'] = 1.0
    # params['recovery_rate_with_inpatient_care'] = 1.0

    # Make 100% death rate by replacing with empty linear model 1.0
    for am in ['MAM', 'SAM']:
        sim.modules['Stunting'].acute_malnutrition_recovery_based_on_interventions[am] = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

    # Run the 'do_when_am_treatment' function
    interventions = ['SFP', 'OTC', 'ITC']
    for int in interventions:
        sim.modules['Stunting'].do_when_am_treatment(person_id=person_id, intervention=int)

    # Run the death event that was originally scheduled - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']

    # print(sim.find_events_for_person(person_id))

    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                  isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)][0]

    # Run the CureEvent
    cure_event.apply(person_id=person_id)

    # Check that the person is not infected and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_HAZ_category'] == 'HAZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_use_of_HSI_for_MAM(tmpdir):
    """ Check that the HSIs works"""
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Stunting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of stunting
    sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # reduce progression to severe stunting
    sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # decrease MUAC and oedema parameters in moderate stunting for clinical SAM
    params = sim.modules['Stunting'].parameters
    params['proportion_-3<=HAZ<-2_with_MUAC<115mm'] = 0.0  # no SAM with moderate stunting
    params['proportion_oedema_with_HAZ<-2'] = 0.0  # no SAM based on oedema
    params['prevalence_nutritional_oedema'] = 0.0  # no SAM based on oedema

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

    # Run Stunting Polling event to get new incident cases:
    polling = StuntingPollingEvent(module=sim.modules['Stunting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately stunted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_stunted']
    assert person['un_HAZ_category'] == '-3<=HAZ<-2'
    assert person['un_clinical_acute_malnutrition'] == 'MAM'
    assert person['un_last_stunting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])
    # Check not on treatment:
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])

    # Run the HSI event
    hsi = HSI_supplementary_feeding_programme_for_MAM(person_id=person_id, module=sim.modules['Stunting'])
    hsi.run(squeeze_factor=0.0)

    # Check that person is now on treatment:
    assert sim.date == df.at[person_id, 'un_acute_malnutrition_tx_start_date']

    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                  isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)][0]

    # Run the CureEvent
    cure_event.apply(person_id=person_id)

    # Check that the person is cured and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_HAZ_category'] == 'HAZ>=-2'
    assert person['un_clinical_acute_malnutrition'] == 'well'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_use_of_HSI_for_SAM(tmpdir):
    """ Check that the HSI_outpatient_therapeutic_programme_for_SAM and HSI_inpatient_care_for_complicated_SAM work"""

    def test_use_of_HSI_by_complication(complications):
        dur = pd.DateOffset(days=0)
        popsize = 1000
        sim = get_sim(tmpdir)

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date + dur)
        sim.event_queue.queue = []  # clear the queue

        # Make 100% death rate by replacing with empty linear model 1.0
        sim.modules['Stunting'].sam_death_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # Make 100% treatment effectiveness by replacing with empty linear model 1.0
        for am in ['MAM', 'SAM']:
            sim.modules['Stunting'].acute_malnutrition_recovery_based_on_interventions[am] = LinearModel(
                LinearModelType.MULTIPLICATIVE, 1.0)

        # increase incidence of stunting and progression to severe
        sim.modules['Stunting'].stunting_incidence_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)
        sim.modules['Stunting'].severe_stunting_progression_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # remove the probability of complications in SAM
        params = sim.modules['Stunting'].parameters
        if complications:
            params['prob_complications_in_SAM'] = 1.0  # only SAM with complications
        else:
            params['prob_complications_in_SAM'] = 0.0  # no SAM with complications

        # # change coverage and treatment effectiveness set to be 100%
        # params['coverage_supplementary_feeding_program'] = 1.0
        # params['coverage_outpatient_therapeutic_care'] = 1.0
        # params['coverage_inpatient_care'] = 1.0
        # params['recovery_rate_with_soy_RUSF'] = 1.0
        # params['recovery_rate_with_CSB++'] = 1.0
        # params['recovery_rate_with_standard_RUTF'] = 1.0
        # params['recovery_rate_with_inpatient_care'] = 1.0

        # Make 100% death rate by replacing with empty linear model 1.0
        sim.modules['Stunting'].acute_malnutrition_recovery_based_on_interventions['SAM'] = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # Get person to use:
        df = sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        assert df.loc[person_id, 'un_HAZ_category'] == 'HAZ>=-2'

        # Run Stunting Polling event to get new incident cases:
        polling = StuntingPollingEvent(module=sim.modules['Stunting'])
        polling.apply(sim.population)

        # Check properties of this individual:
        person = df.loc[person_id]
        assert person['un_ever_stunted']
        assert person['un_HAZ_category'] == '-3<=HAZ<-2'
        assert person['un_last_stunting_date_of_onset'] == sim.date
        assert pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])
        # Check not on treatment:
        assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])

        # Check that there is a ProgressionSevereStuntingEvent scheduled for this person:
        progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                   isinstance(event_tuple[1], ProgressionSevereStuntingEvent)
                                   ][0]
        date_of_scheduled_progression = progression_event_tuple[0]
        progression_event = progression_event_tuple[1]
        assert date_of_scheduled_progression > sim.date

        # Run the progression to severe stunting event:
        sim.date = date_of_scheduled_progression
        progression_event.apply(person_id=person_id)

        # Check properties of this individual: (should now be severely stunted and without a scheduled death date)
        person = df.loc[person_id]
        assert person['un_ever_stunted']
        assert person['un_HAZ_category'] == 'HAZ<-3'
        assert person['un_clinical_acute_malnutrition'] == 'SAM'
        assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
        assert pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])

        # Run the HSI event
        if complications:
            hsi = HSI_inpatient_care_for_complicated_SAM(person_id=person_id, module=sim.modules['Stunting'])
            hsi.run(squeeze_factor=0.0)
        else:
            hsi = HSI_outpatient_therapeutic_programme_for_SAM(person_id=person_id, module=sim.modules['Stunting'])
            hsi.run(squeeze_factor=0.0)

        # Check that person is now on treatment:
        assert sim.date == df.at[person_id, 'un_acute_malnutrition_tx_start_date']

        print(sim.find_events_for_person(person_id))

        # Check that a CureEvent has been scheduled
        cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                      isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)][0]

        # Run the CureEvent
        cure_event.apply(person_id=person_id)

        # Check that the person is cured and is alive still:
        person = df.loc[person_id]
        assert person['is_alive']
        assert person['un_HAZ_category'] == 'HAZ>=-2'
        assert person['un_clinical_acute_malnutrition'] == 'well'
        assert not pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])

    test_use_of_HSI_by_complication(complications=True)
    test_use_of_HSI_by_complication(complications=False)

