"""
 Basic tests for the Wasting Module
 """
import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    wasting,
)
from tlo.methods.wasting import (
    AcuteMalnutritionDeathPollingEvent,
    ClinicalAcuteMalnutritionRecoveryEvent,
    HSI_Wasting_InpatientCareForComplicated_SAM,
    HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
    HSI_Wasting_SupplementaryFeedingProgramme_MAM,
    ProgressionSevereWastingEvent,
    SevereAcuteMalnutritionDeathEvent,
    UpdateToMAM,
    WastingNaturalRecoveryEvent,
    WastingPollingEvent,
)

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 2)


def get_sim(tmpdir):
    """Return simulation objection with Wasting and other necessary modules registered."""
    sim = Simulation(start_date=start_date, seed=0, show_progress_bar=False, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            "tlo.methods.wasting": logging.INFO}
    })

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False, cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath)
                 )
    return sim


def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def check_configuration_of_properties(sim):
    # check that the properties are ok:

    df = sim.population.props

    # Those that were never wasted, should have normal WHZ score:
    assert (df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna(), 'un_WHZ_category'] == 'WHZ>=-2').all().all()

    # Those that were never wasted and not clinically malnourished,
    # should have not_applicable/null values for all the other properties:
    # assert pd.isnull(df.loc[(~df.un_ever_wasted & ~df.date_of_birth.isna() &
    #                         (df.un_clinical_acute_malnutrition == 'well')),
    #                         ['un_last_wasting_date_of_onset',
    #                          'un_sam_death_date'
    #                          ]
    #                         ]).all().all()

    # # Those that were ever wasted, should have a WHZ score below <-2
    # assert (df.loc[df.un_ever_wasted, 'un_WHZ_category'] != 'WHZ>=-2').all().all()
    #
    # # Those that had wasting and no treatment, should have either a recovery date or a death_date
    # # (but not both)
    # has_recovery_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
    #                                       'un_am_recovery_date'])
    # has_death_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
    #                                    'un_sam_death_date'])

    # has_recovery_date_or_death_date = has_recovery_date | has_death_date
    # has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    # # assert has_recovery_date_or_death_date.all()
    # assert not has_both_recovery_date_and_death_date.any()

    # Those for whom the death date has past should be dead
    assert not df.loc[df.un_ever_wasted & (df['un_sam_death_date'] < sim.date), 'is_alive'].any()
    assert not df.loc[(df.un_clinical_acute_malnutrition == 'SAM') & (
        df['un_sam_death_date'] < sim.date), 'is_alive'].any()

    # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
    #  but not others (among those who are alive)
    has_symptoms_of_wasting = set(sim.modules['SymptomManager'].who_has('weight_loss'))
    has_symptoms = set([p for p in has_symptoms_of_wasting if
                        'Wasting' in sim.modules['SymptomManager'].causes_of(p, 'weight_loss')
                        ])

    in_current_episode_before_recovery = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_am_recovery_date)
    set_of_person_id_in_current_episode_before_recovery = set(
        in_current_episode_before_recovery[in_current_episode_before_recovery].index
    )

    in_current_episode_before_death = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_sam_death_date)
    set_of_person_id_in_current_episode_before_death = set(
        in_current_episode_before_death[in_current_episode_before_death].index
    )

    in_current_episode_before_cure = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (df.un_acute_malnutrition_tx_start_date <= sim.date) & \
        pd.isnull(df.un_am_recovery_date) & \
        pd.isnull(df.un_sam_death_date)
    set_of_person_id_in_current_episode_before_cure = set(
        in_current_episode_before_cure[in_current_episode_before_cure].index
    )

    assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
        set_of_person_id_in_current_episode_before_death
    )

    set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
        set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
    )
    assert set_of_person_id_in_current_episode == has_symptoms


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(months=3)
    popsize = 1000
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    check_dtypes(sim)
    check_configuration_of_properties(sim)


def test_wasting_polling(tmpdir):
    """Check polling events leads to incident cases"""
    # get simulation object:
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    # Make incidence of wasting very high :
    params = sim.modules['Wasting'].parameters
    for p in params:
        if p.startswith('base_inc_rate_wasting_by_agegp'):
            params[p] = [3 * v for v in params[p]]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Run polling event: check that a severe incident case is produced:
    polling = WastingPollingEvent(sim.modules['Wasting'])
    polling.apply(sim.population)

    for q in sim.event_queue.queue:
        print(f'Qui {q[3]}')

    assert len([q for q in sim.event_queue.queue if isinstance(q[3], ProgressionSevereWastingEvent)]) > 0

    # Check properties of this individual: should now be moderately wasted
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date


def test_recovery_moderate_wasting(tmpdir):
    """Check natual recovery of moderate wasting """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 0% death rate by replacing with empty linear model 0.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # increase incidence of wasting
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # remove progression to severe wasting
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a WastingNaturalRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], WastingNaturalRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run the recovery event:
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert person['un_am_recovery_date'] == sim.date
    assert pd.isnull(person['un_sam_death_date'])


def test_recovery_severe_wasting(tmpdir):
    """ Check natural recovery to MAM by removing death rate for those with severe wasting,
     and check the onset of symptoms with SAM and revolving of symptoms when recovered to MAM """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 0% death rate by replacing with empty linear model 0.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # increase incidence of wasting and progression to severe
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereWastingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereWastingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check that they have some symptoms caused by Wasting (SAM only)
    assert 0 < len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Wasting']))

    # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run Death Polling event to apply death:
    death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Wasting'])
    death_polling.apply(sim.population)

    # Check that there is a UpdateToMAM scheduled for this person:
    recovery_to_mam_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                   isinstance(event_tuple[1], UpdateToMAM)
                                   ][0]
    print(sim.find_events_for_person(person_id))
    date_of_scheduled_recovery_to_mam = recovery_to_mam_event_tuple[0]
    recovery_to_mam_event = recovery_to_mam_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

    # Run a Recovery SAM to MAM Event
    sim.date = date_of_scheduled_recovery_to_mam
    recovery_to_mam_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_WHZ_category'] != 'WHZ<-3'
    assert (person['un_am_MUAC_category'] == '115-<125mm' or person['un_WHZ_category'] == '-3<=WHZ<-2')
    assert pd.isnull(person['un_sam_death_date'])

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Wasting']))


def test_nat_hist_death(tmpdir):
    """Check: Wasting onset --> death"""
    """ Check if the risk of death is 100% does everyone with SAM die? """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of wasting and progression to severe
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # Get the children to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereWastingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereWastingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run Death Polling Polling event to apply death:
    death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Wasting'])
    death_polling.apply(sim.population)

    # Check that there is a SevereAcuteMalnutritionDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run the death event:
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not pd.isnull(person['un_sam_death_date'])
    assert person['un_sam_death_date'] == sim.date
    assert not person['is_alive']


def test_nat_hist_cure_if_recovery_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to recover naturally, it cause the episode to
    end earlier."""

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 0% death rate by replacing with empty linear model 0.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # increase incidence of wasting
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # decrease progression to severe of wasting - so all children stay in moderate wasting state only
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted without progression to severe)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    print(sim.find_events_for_person(person_id))

    # Check that there is a WastingNaturalRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], WastingNaturalRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run a Cure Event
    cure_event = ClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id, module=sim.modules['Wasting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the recovery event that was originally scheduled - this should have no effect
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


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
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of wasting and progression to severe
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)
    # increase parameters in moderate wasting for clinical SAM (MUAC and oedema) to be polled for death
    params = sim.modules['Wasting'].parameters
    params['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = [5 * params['proportion_-3<=WHZ<-2_with_MUAC<115mm']]
    params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] = [params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] / 5]
    params['proportion_oedema_with_WHZ<-2'] = 0.9

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereWastingEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                               isinstance(event_tuple[1], ProgressionSevereWastingEvent)
                               ][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run Death Polling Polling event to apply death:
    death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Wasting'])
    death_polling.apply(sim.population)

    # Check that there is a SevereAcuteMalnutritionDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event now
    cure_event = ClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id, module=sim.modules['Wasting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the death event that was originally scheduled - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']


# def test_treatment(tmpdir):
#     """ Test that providing a treatment prevent death and causes there to be a
#     CureEvent (ClinicalAcuteMalnutritionRecoveryEvent) Scheduled """
#     """
#     This test sets the linear model of acute_malnutrition_recovery_based_on_interventions to be 1.0 (100% cure rate),
#     when this lm is called in the do_when_am_treatment function (usually called in HSIs), 100% cure rate schedules the
#     ClinicalAcuteMalnutritionRecoveryEvent (CureEvent).
#     Death is prevented.  ---> check
#     """
#     # TODO: CHECK THIS - MAYBE NOT NEEDED
#
#     dur = pd.DateOffset(days=0)
#     popsize = 1000
#     sim = get_sim(tmpdir)
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=start_date + dur)
#     sim.event_queue.queue = []  # clear the queue
#
#     # Make 100% death rate by replacing with empty linear model 1.0
#     sim.modules['Wasting'].sam_death_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # increase incidence of wasting and progression to severe
#     sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#     sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
#         LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # increase parameters in moderate wasting for clinical SAM (MUAC and oedema) to be polled for death
#     params = sim.modules['Wasting'].parameters
#     params['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = [5 * params['proportion_-3<=WHZ<-2_with_MUAC<115mm']]
#     params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] = [params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] / 5]
#     params['proportion_oedema_with_WHZ<-2'] = 0.9
#
#     # Get person to use:
#     df = sim.population.props
#     under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
#     person_id = under5s.index[0]
#     assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'
#
#     # Run Wasting Polling event to get new incident cases:
#     polling = WastingPollingEvent(module=sim.modules['Wasting'])
#     polling.apply(sim.population)
#
#     # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
#     person = df.loc[person_id]
#     assert person['un_ever_wasted']
#     assert person['un_WHZ_category'] == '-3<=WHZ<-2'
#     assert person['un_last_wasting_date_of_onset'] == sim.date
#     assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
#     assert pd.isnull(person['un_am_recovery_date'])
#     assert pd.isnull(person['un_sam_death_date'])
#
#     # Check that there is a ProgressionSevereWastingEvent scheduled for this person:
#     progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                                isinstance(event_tuple[1], ProgressionSevereWastingEvent)
#                                ][0]
#     date_of_scheduled_progression = progression_event_tuple[0]
#     progression_event = progression_event_tuple[1]
#     assert date_of_scheduled_progression > sim.date
#
#     # Run the progression to severe wasting event:
#     sim.date = date_of_scheduled_progression
#     progression_event.apply(person_id=person_id)
#
#     # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
#     person = df.loc[person_id]
#     assert person['un_ever_wasted']
#     assert person['un_WHZ_category'] == 'WHZ<-3'
#     assert person['un_clinical_acute_malnutrition'] == 'SAM'
#     assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
#     assert pd.isnull(person['un_am_recovery_date'])
#     assert pd.isnull(person['un_sam_death_date'])
#
#     # Run Death Polling event to apply death:
#     death_polling = AcuteMalnutritionDeathPollingEvent(module=sim.modules['Wasting'])
#     death_polling.apply(sim.population)
#
#     # Check that there is a SevereAcuteMalnutritionDeathEvent scheduled for this person:
#     death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
#                          isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)
#                          ][0]
#     date_of_scheduled_death = death_event_tuple[0]
#     death_event = death_event_tuple[1]
#     assert date_of_scheduled_death > sim.date
#
#     # # change coverage and treatment effectiveness set to be 100%
#     # params['coverage_supplementary_feeding_program'] = 1.0
#     # params['coverage_outpatient_therapeutic_care'] = 1.0
#     # params['coverage_inpatient_care'] = 1.0
#     # params['recovery_rate_with_soy_RUSF'] = 1.0
#     # params['recovery_rate_with_CSB++'] = 1.0
#     # params['recovery_rate_with_standard_RUTF'] = 1.0
#     # params['recovery_rate_with_inpatient_care'] = 1.0
#
#     # Make 100% death rate by replacing with empty linear model 1.0
#     for am in ['MAM', 'SAM']:
#         sim.modules['Wasting'].acute_malnutrition_recovery_based_on_interventions[am] = LinearModel(
#             LinearModelType.MULTIPLICATIVE, 1.0)
#
#     # Run the 'do_when_am_treatment' function
#     interventions = ['SFP', 'OTC', 'ITC']
#     for intv in interventions:
#         sim.modules['Wasting'].do_when_am_treatment(person_id=person_id, intervention=intv)
#
#     # Run the death event that was originally scheduled - this should have no effect and the person should not die
#     sim.date = date_of_scheduled_death
#     death_event.apply(person_id=person_id)
#     person = df.loc[person_id]
#     assert person['is_alive']
#
#     # print(sim.find_events_for_person(person_id))
#
#     # Check that a CureEvent has been scheduled
#     cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
#                   isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)][0]
#
#     # Run the CureEvent
#     cure_event.apply(person_id=person_id)
#
#     # Check that the person is not infected and is alive still:
#     person = df.loc[person_id]
#     assert person['is_alive']
#     assert person['un_WHZ_category'] == 'WHZ>=-2'
#     assert not pd.isnull(person['un_am_recovery_date'])
#     assert pd.isnull(person['un_sam_death_date'])


def test_use_of_HSI_for_MAM(tmpdir):
    """ Check that the HSIs works"""
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of wasting
    sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # reduce progression to severe wasting
    sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # decrease MUAC and oedema parameters in moderate wasting for clinical SAM
    params = sim.modules['Wasting'].parameters
    params['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = 0.0  # no SAM with moderate wasting
    params['proportion_oedema_with_WHZ<-2'] = 0.0  # no SAM based on oedema
    params['prevalence_nutritional_oedema'] = 0.0  # no SAM based on oedema

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_clinical_acute_malnutrition'] == 'MAM'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])
    # Check not on treatment:
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])

    # Run the HSI event
    hsi = HSI_Wasting_SupplementaryFeedingProgramme_MAM(person_id=person_id, module=sim.modules['Wasting'])
    print(f'------')
    print(f'personia {person}')
    print(f'------')
    hsi.run(squeeze_factor=0.0)

    # Check that person is now on treatment:
    assert sim.date == df.at[person_id, 'un_acute_malnutrition_tx_start_date']
    print(f'person trt start date {df.at[person_id, "un_acute_malnutrition_tx_start_date"]}')
    print(f'trt type {df.at[person_id, "un_am_treatment_type"]}')

    showme = sim.find_events_for_person(person_id)
    print(f'Show Me {showme}')

    for event_tuple in sim.find_events_for_person(person_id):
        print(f'Ev {event_tuple[1]}')

    print(f'events {sim.find_events_for_person(person_id)}')
    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                  isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)][0]  #add wastingnaturalrecoveryevent

    # Run the CureEvent
    cure_event.apply(person_id=person_id)

    # Check that the person is cured and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert person['un_clinical_acute_malnutrition'] == 'well'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_use_of_HSI_for_SAM(tmpdir):
    """
    Check that the HSI_Wasting_OutpatientTherapeuticProgramme_SAM and HSI_Wasting_InpatientCareForComplicated_SAM work
    """

    def test_use_of_HSI_by_complication(complications):
        dur = pd.DateOffset(days=0)
        popsize = 1000
        sim = get_sim(tmpdir)

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date + dur)
        sim.event_queue.queue = []  # clear the queue

        # Make 100% death rate by replacing with empty linear model 1.0
        sim.modules['Wasting'].sam_death_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # Make 100% treatment effectiveness by replacing with empty linear model 1.0
        for am in ['MAM', 'SAM']:
            sim.modules['Wasting'].acute_malnutrition_recovery_based_on_interventions[am] = LinearModel(
                LinearModelType.MULTIPLICATIVE, 1.0)

        # increase incidence of wasting and progression to severe
        sim.modules['Wasting'].wasting_incidence_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)
        sim.modules['Wasting'].severe_wasting_progression_equation = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # remove the probability of complications in SAM
        params = sim.modules['Wasting'].parameters
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
        sim.modules['Wasting'].acute_malnutrition_recovery_based_on_interventions['SAM'] = LinearModel(
            LinearModelType.MULTIPLICATIVE, 1.0)

        # Get person to use:
        df = sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

        # Run Wasting Polling event to get new incident cases:
        polling = WastingPollingEvent(module=sim.modules['Wasting'])
        polling.apply(sim.population)

        # Check properties of this individual:
        person = df.loc[person_id]
        assert person['un_ever_wasted']
        assert person['un_WHZ_category'] == '-3<=WHZ<-2'
        assert person['un_last_wasting_date_of_onset'] == sim.date
        assert pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])
        # Check not on treatment:
        assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])

        # Check that there is a ProgressionSevereWastingEvent scheduled for this person:
        progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                   isinstance(event_tuple[1], ProgressionSevereWastingEvent)
                                   ][0]
        date_of_scheduled_progression = progression_event_tuple[0]
        progression_event = progression_event_tuple[1]
        assert date_of_scheduled_progression > sim.date

        # Run the progression to severe wasting event:
        sim.date = date_of_scheduled_progression
        progression_event.apply(person_id=person_id)

        # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
        person = df.loc[person_id]
        assert person['un_ever_wasted']
        assert person['un_WHZ_category'] == 'WHZ<-3'
        assert person['un_clinical_acute_malnutrition'] == 'SAM'
        assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
        assert pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])

        # Run the HSI event
        if complications:
            hsi = HSI_Wasting_InpatientCareForComplicated_SAM(person_id=person_id, module=sim.modules['Wasting'])
            hsi.run(squeeze_factor=0.0)
        else:
            hsi = HSI_Wasting_OutpatientTherapeuticProgramme_SAM(person_id=person_id, module=sim.modules['Wasting'])
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
        assert person['un_WHZ_category'] == 'WHZ>=-2'
        assert person['un_clinical_acute_malnutrition'] == 'well'
        assert not pd.isnull(person['un_am_recovery_date'])
        assert pd.isnull(person['un_sam_death_date'])

    test_use_of_HSI_by_complication(complications=True)
    test_use_of_HSI_by_complication(complications=False)
