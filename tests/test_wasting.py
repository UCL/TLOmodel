"""
 Basic tests for the Wasting Module
 """
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hsi_generic_first_appts,
    simplified_births,
    symptommanager,
    wasting,
)
from tlo.methods.healthseekingbehaviour import HealthSeekingBehaviourPoll
from tlo.methods.wasting import (
    ClinicalAcuteMalnutritionRecoveryEvent,
    HSI_Wasting_InpatientCareForComplicated_SAM,
    HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
    ProgressionSevereWastingEvent,
    SevereAcuteMalnutritionDeathEvent,
    WastingNaturalRecoveryEvent,
    WastingPollingEvent,
)

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)


def get_sim(tmpdir):
    """
    Return simulation objection with Wasting and other necessary
    modules registered.
    """
    sim = Simulation(start_date=start_date, seed=0,
                     show_progress_bar=False,
                     log_config={
                         'filename': 'tmp',
                         'directory': tmpdir,
                         'custom_levels': {
                             "*": logging.WARNING,
                             "tlo.methods.wasting": logging.INFO}
                     })

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(
                     resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(
                     resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(
                     resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(
                     resourcefilepath=resourcefilepath),
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
    assert (
            df.loc[~df.un_ever_wasted &
                   ~df.date_of_birth.isna(), 'un_WHZ_category'] == 'WHZ>=-2'
    ).all()

    # Those for whom the death date has past should be dead
    assert not df.loc[df.un_ever_wasted &
                      (df['un_sam_death_date'] < sim.date), 'is_alive'].any()
    assert not df.loc[(df.un_clinical_acute_malnutrition == 'SAM') & (
            df['un_sam_death_date'] < sim.date), 'is_alive'].any()

    # Check that those in a current episode have symptoms of wasting
    # [caused by the wasting module] but not others (among those alive)
    has_symptoms_of_wasting = \
        set(sim.modules['SymptomManager'].who_has('weight_loss'))
    has_symptoms = set([p for p in has_symptoms_of_wasting if
                        'Wasting' in
                        sim.modules['SymptomManager'].causes_of(p,
                                                                'weight_loss')
                        ])

    in_current_episode_before_recovery = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_am_recovery_date)
    set_of_person_id_in_current_episode_before_recovery = set(
        in_current_episode_before_recovery[
            in_current_episode_before_recovery].index
    )

    in_current_episode_before_death = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_sam_death_date)
    set_of_person_id_in_current_episode_before_death = set(
        in_current_episode_before_death[in_current_episode_before_death].index
    )

    assert set() == \
           set_of_person_id_in_current_episode_before_recovery.intersection(
        set_of_person_id_in_current_episode_before_death
    )

    # WHZ standard deviation of -3, oedema, and MUAC <115mm should cause
    # severe acute malnutrition
    whz_index = df.index[df['un_WHZ_category'] == 'WHZ<-3']
    oedema_index = df.index[df['un_am_bilateral_oedema']]
    muac_index = df.index[df['un_am_MUAC_category'] == '<115mm']
    assert (df.loc[whz_index, 'un_clinical_acute_malnutrition'] == "SAM").all()
    assert (df.loc[
                oedema_index, 'un_clinical_acute_malnutrition'] == "SAM").all()
    assert (df.loc[
                muac_index, 'un_clinical_acute_malnutrition'] == "SAM").all()

    # all SAM individuals should have symptoms of wasting
    assert set(df.index[df.is_alive & (df.age_exact_years < 5) &
                        (df.un_clinical_acute_malnutrition == 'SAM')]
               ).issubset(has_symptoms)

    # All MAM individuals should have no symptoms of wasting
    assert set(df.index[df.is_alive & (df.age_exact_years < 5) &
                        (df.un_clinical_acute_malnutrition == 'MAM')]) \
           not in has_symptoms


@pytest.mark.slow
def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    popsize = 10_000
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
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
    params['base_inc_rate_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    params['progression_severe_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0]

    # re-initialise wasting linear models to use the updated parameter
    sim.modules['Wasting'].pre_initialise_population()

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Run polling event: check that a severe incident case is produced:
    polling = WastingPollingEvent(sim.modules['Wasting'])
    polling.apply(sim.population)

    assert len([q for q in sim.event_queue.queue if
                isinstance(q[3], ProgressionSevereWastingEvent)]) > 0

    # Check properties of this individual: should now be moderately wasted
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date


def test_recovery_moderate_wasting(tmpdir):
    """Check natural recovery of moderate wasting """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # get wasting module
    wmodule = sim.modules['Wasting']
    # increase wasting incidence rate to 100% for all
    # age groups(less than 5 years)
    wmodule.parameters['base_inc_rate_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0,
                                                            1.0, 1.0]
    wmodule.parameters['progression_severe_wasting_by_agegp'] = [0.0, 0.0, 0.0,
                                                                 0.0, 0.0, 0.0]

    # re-initialise wasting linear models to use the updated parameter
    wmodule.pre_initialise_population()

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

    # Check that there is a WastingNaturalRecoveryEvent scheduled
    # for this person
    recov_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], WastingNaturalRecoveryEvent)][0]
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


def test_recovery_severe_wasting_without_complications(tmpdir):
    """ Check natural recovery to MAM by removing death rate for those with
    severe wasting, and check the onset of symptoms with SAM and revolving
    of symptoms when recovered to MAM """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # get wasting module
    wmodule = sim.modules['Wasting']
    # increase wasting incidence rate to 100% for all
    # age groups(less than 5 years)
    wmodule.parameters['base_inc_rate_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0,
                                                            1.0, 1.0]
    wmodule.parameters['progression_severe_wasting_by_agegp'] = [1.0, 1.0, 1.0,
                                                                 1.0, 1.0, 1.0]

    # re-initialise wasting linear models to use the updated parameter
    wmodule.pre_initialise_population()

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

    # Check that there is a ProgressionSevereWastingEvent scheduled
    # for this person:
    progression_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], ProgressionSevereWastingEvent)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check individuals have symptoms caused by Wasting (SAM only)
    assert 0 < len(sim.modules['SymptomManager'].has_what(person_id,
                                                          sim.modules[
                                                              'Wasting']))

    # Check properties of this individual: (should now be severely wasted and
    # without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and
    # check care was sought
    ge = [ev[1] for ev in
          sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1],
                     hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)
          ][0]
    ge.run(squeeze_factor=0.0)

    # check HSI event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        HSI_Wasting_OutpatientTherapeuticProgramme_SAM)

    # Run the created instance of
    # HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in
              sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1],
                         HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # check recovery event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1],
                      ClinicalAcuteMalnutritionRecoveryEvent)

    # Run the recovery event and check the individual has recovered from SAM:
    sam_recovery_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], ClinicalAcuteMalnutritionRecoveryEvent)
         ][0]

    date_of_scheduled_recovery_to_mam = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

    # Run SAM recovery
    sim.date = date_of_scheduled_recovery_to_mam
    sam_recovery_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert (person['un_am_MUAC_category'] == '>=125mm')
    assert pd.isnull(person['un_sam_death_date'])

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id,
                                                           sim.modules[
                                                               'Wasting']))


def test_recovery_severe_wasting_with_complications(tmpdir):
    """ test individual's recovery from wasting with complications """
    dur = pd.DateOffset(days=3)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # get wasting module
    wmodule = sim.modules['Wasting']

    # Manually set this individual properties to have
    # severe acute malnutrition with complications
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'

    # make the individual have wasting complications
    wmodule.parameters['prob_complications_in_SAM'] = 1.0

    # assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # apply symptoms
    wmodule.wasting_clinical_symptoms(person_id)

    # by having severe wasting, this individual should be diagnosed as SAM
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # symptoms should be applied
    assert person_id in set(
        sim.modules['SymptomManager'].who_has('weight_loss'))

    # should have complications
    assert df.at[person_id, 'un_sam_with_complications']

    # make recovery rate to 100% and death rate to zero so that
    # this individual should recover
    wasting_module = sim.modules['Wasting']
    wasting_module.parameters['recovery_rate_with_inpatient_care'] = 1.0
    wasting_module.parameters['base_death_rate_untreated_SAM'] = 0.0

    # re-initialise wasting models
    wasting_module.pre_initialise_population()

    #   run care seeking event and ensure HSI for complicated SAM is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and
    # check care was sought
    ge = [ev[1] for ev in
          sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1],
                     hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)
          ][0]
    ge.run(squeeze_factor=0.0)

    # check HSI event for complicated SAM is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        HSI_Wasting_InpatientCareForComplicated_SAM)

    # Run the created instance of
    # HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in
              sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_InpatientCareForComplicated_SAM)][
        0]
    sam_ev.run(squeeze_factor=0.0)

    # check recovery event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1],
                      ClinicalAcuteMalnutritionRecoveryEvent)

    # Run the recovery event and check the individual has recovered from SAM:
    sam_recovery_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1],
                       ClinicalAcuteMalnutritionRecoveryEvent)][0]

    date_of_scheduled_recovery_to_mam = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

    # Run SAM recovery
    sim.date = date_of_scheduled_recovery_to_mam
    sam_recovery_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert (person['un_am_MUAC_category'] == '>=125mm')
    assert pd.isnull(person['un_sam_death_date'])

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id,
                                                           sim.modules[
                                                               'Wasting']))


def test_nat_hist_death(tmpdir):
    """Check: Wasting onset --> death"""
    """ Check if the risk of death is 100% does everyone with SAM die? """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # get wasting module
    wasting_module = sim.modules['Wasting']
    # Make 100% death rate by replacing with empty linear model 1.0
    wasting_module.sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # Get the children to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # re-set parameter values to make zero recovery rate and 100% death rate
    wasting_module.parameters['recovery_rate_with_standard_RUTF'] = 0.0
    wasting_module.parameters['recovery_rate_with_inpatient_care'] = 0.0
    wasting_module.parameters['base_death_rate_untreated_SAM'] = 1.0

    # re-initialise wasting models
    wasting_module.pre_initialise_population()

    # make an individual diagnosed as SAM by WHZ category.
    # We want to make this individual qualify for death
    df.loc[person_id, 'un_ever_wasted'] = True
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_clinical_acute_malnutrition'] = 'SAM'

    # apply wasting symptoms to this individual
    wasting_module.wasting_clinical_symptoms(person_id)

    # check symptoms are applied
    assert person_id in set(
        sim.modules['SymptomManager'].who_has('weight_loss'))

    # run health seeking behavior and ensure non-emergency event is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0
    # and check care was sought
    ge = [ev[1] for ev in
          sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1],
                     hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)
          ][0]
    ge.run(squeeze_factor=0.0)

    # check inpatient care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        HSI_Wasting_OutpatientTherapeuticProgramme_SAM)

    # Run the created instance of
    # HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in
              sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1],
                         HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # since there is zero recovery rate, check death event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1],
                      SevereAcuteMalnutritionDeathEvent)

    # # Run the acute death event and ensure the person is now dead:
    death_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not pd.isnull(person['un_sam_death_date'])
    assert person['un_sam_death_date'] == sim.date
    assert not person['is_alive']


def test_nat_hist_cure_if_recovery_scheduled(tmpdir):
    """ Show that if a cure event is run before when a person was going to
    recover naturally, it causes the episode to end earlier. """

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 0% death rate by replacing with empty linear model 0.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)
    wasting_module = sim.modules['Wasting']

    # increase wasting incidence rate to 100% and reduce rate of progress to
    # severe wasting to zero.We don't want individuals to progress to SAM as
    # we are testing for MAM natural recovery
    wasting_module.parameters['base_inc_rate_wasting_by_agegp'] = [1.0, 1.0,
                                                                   1.0, 1.0,
                                                                   1.0, 1.0]
    wasting_module.parameters['progression_severe_wasting_by_agegp'] = [0.0,
                                                                        0.0,
                                                                        0.0,
                                                                        0.0,
                                                                        0.0,
                                                                        0.0]
    wasting_module.pre_initialise_population()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted
    # without progression to severe)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a WastingNaturalRecoveryEvent scheduled for
    # this person:
    recov_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], WastingNaturalRecoveryEvent)][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run a Cure Event
    cure_event = ClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id,
                                                        module=sim.modules[
                                                            'Wasting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the recovery event that was originally scheduled -
    # this should have no effect
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_nat_hist_cure_if_death_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to die,
     it causes the episode to end without the person dying."""

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # get wasting module parameters
    params = sim.modules['Wasting'].parameters
    # increase to 100% death rate, incidence and progress to severe wasting
    params['base_death_rate_untreated_SAM'] = 1.0
    params['base_inc_rate_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    params['progression_severe_wasting_by_agegp'] = [1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0]
    params['prob_mam_death_after_care'] = [0.0, 1.0]

    # reduce to 100% recovery rate. This is to ensure death event is scheduled
    # for the individual each time we run this
    # test
    params['recovery_rate_with_standard_RUTF'] = 0.0
    params['recovery_rate_with_inpatient_care'] = 0.0

    # increase parameters in moderate wasting for clinical SAM
    # (MUAC and oedema) to be polled for death
    params['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = [
        5 * params['proportion_-3<=WHZ<-2_with_MUAC<115mm']]
    params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] = [
        params['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm'] / 5]
    params['proportion_oedema_with_WHZ<-2'] = 0.9

    # re-initialise wasting models
    sim.modules['Wasting'].pre_initialise_population()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingPollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted
    # with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a ProgressionSevereWastingEvent scheduled for this
    # person:
    progression_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], ProgressionSevereWastingEvent)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely wasted and
    # without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_acute_malnutrition_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # run health seeking behavior and ensure non-emergency event is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and
    # check care was sought
    ge = [ev[1] for ev in
          sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1],
                     hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)
          ][0]
    ge.run(squeeze_factor=0.0)

    # check inpatient care event is scheduled
    assert isinstance(
        sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
        HSI_Wasting_OutpatientTherapeuticProgramme_SAM)

    # Run the created instance of
    # HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in
              sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1],
                         HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # since there is zero recovery rate, check death event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1],
                      SevereAcuteMalnutritionDeathEvent)

    # Run Severe Acute Malnutrition Death Event scheduled for this person:
    death_event_tuple = \
        [event_tuple for event_tuple in sim.find_events_for_person(person_id)
         if isinstance(event_tuple[1], SevereAcuteMalnutritionDeathEvent)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event now
    cure_event = ClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id,
                                                        module=sim.modules[
                                                            'Wasting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the death event that was originally scheduled - this should have no
    # effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
