"""Test file for the COPD module."""
import os
from pathlib import Path

import numpy as np
import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import (
    copd,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hsi_generic_first_appts,
    simplified_births,
    symptommanager,
)
from tlo.methods.copd import CopdExacerbationEvent, HSI_Copd_TreatmentOnSevereExacerbation
from tlo.methods.healthseekingbehaviour import HealthSeekingBehaviourPoll

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2010, 1, 2)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_basic_run(tmpdir, seed):
    """Run the simulation with the Copd module and read the log from the Copd module."""

    popsize = 1000
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'bed_days',
            'directory': tmpdir,
        },
    )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True
                                                               ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 copd.Copd(resourcefilepath=resourcefilepath),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=Date(2030, 1, 1))  # Long run
    check_dtypes(sim)
    log = parse_log_file(sim.log_filepath)['tlo.methods.copd']

    # Unpack logged prevalence
    log_prev_copd = log['copd_prevalence']

    def unflatten(date):
        select_record = lambda df, _date: df.loc[df['date'] == _date].drop(columns=['date'])  # noqa: E731
        return unflatten_flattened_multi_index_in_logging(select_record(log_prev_copd, date)).iloc[0].T.unstack()


def get_simulation(pop_size):
    """ Return a simulation object

    :param pop_size: total number of individuals at the start of simulation """
    sim = Simulation(
        start_date=start_date
    )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'
                                           ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True
                                                               ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 copd.Copd(resourcefilepath=resourcefilepath),
                 )
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    return sim


def test_ch_lungfunction():
    """ test everyone ends up in lung function category 6 if high progression rate is set """
    sim = get_simulation(10)
    df = sim.population.props

    copd_module = sim.modules['Copd']

    # make all individuals qualify for progressing to the next lung function
    df.loc[df.index, 'is_alive'] = True
    df.loc[df.index, 'ch_lungfunction'] = 0
    df.loc[df.index, 'age_years'] = np.random.choice(range(20, 50), len(df))

    # check they're all eligible to progress to the next lung function
    assert all(copd.eligible_to_progress_to_next_lung_function(df)), 'some are still not eligible to progress to ' \
                                                                     'next lung function'

    # set probability of progressing to next lung function to 1. This will ensure everyone progresses
    # to the next lung function
    copd_module.parameters['prob_progress_to_next_cat'] = 1.0
    # re-initialise models to use updated parameters
    copd_module.pre_initialise_population()

    # Run a function to progress to next lung function six times and ensure all individuals have progressed to a higher
    # lung function(6)
    for _ in range(6):
        copd.CopdPollEvent(module=copd_module).progress_to_next_lung_function(df)
    # all individuals should progress to the highest lung function which in this case is 6
    assert all(df['ch_lungfunction'] == 6)


def test_exacerbations():
    """ test copd exacerbations. Zero risk of exacerbation should lead to no exacerbation event scheduled and higher
    risk of exacerbation should lead to many exacerbation events scheduled"""
    sim = get_simulation(1)  # get simulation object
    copd_module = sim.modules['Copd']  # get copd module

    # 1)--------------- NO RISK OF EXACERBATION DUE TO NON-ELIGIBILITY
    # reset individual properties to zero risk exacerbations.
    # reset age to <15 and lung function to 0
    df = sim.population.props
    df.loc[df.index, 'is_alive'] = True
    df.loc[df.index, 'age_years'] = 10
    df.loc[df.index, 'ch_lungfunction'] = np.NAN

    # clear the event queue
    sim.event_queue.queue = []

    # schedule copd poll event
    _event = copd.CopdPollEvent(copd_module)
    _event.run()

    # confirm no event on an individual has been scheduled
    _individual_events = sim.find_events_for_person(df.index[0])
    assert 0 == len(_individual_events), f'one or more events was scheduled for this ' \
                                         f'person {_individual_events}'

    # 2)----------  HIGH RISK EXACERBATION
    # reset individual properties to higher risk exacerbations.
    # reset age to >15 and lung function to 6
    df = sim.population.props
    df.loc[df.index, 'is_alive'] = True
    df.loc[df.index, 'age_years'] = 20
    df.loc[df.index, 'ch_lungfunction'] = 6

    # set severe and moderate exacerbation probability to maximum(1). This ensures all exacerbation events are schedules
    # on all eligible individuals
    copd_module.parameters['prob_mod_exacerb'][6] = 1.0
    copd_module.parameters['prob_sev_exacerb'][6] = 1.0

    # re-initialise models to use updated parameters
    copd_module.pre_initialise_population()

    # clear the event queue
    sim.event_queue.queue = []

    # run copd poll event
    _event = copd.CopdPollEvent(copd_module)
    _event.run()

    # confirm more than one event has been scheduled
    _exacerbation_events = [ev[1] for ev in sim.find_events_for_person(df.index[0]) if
                            isinstance(ev[1], CopdExacerbationEvent)]

    assert 1 < len(_exacerbation_events), f'not all events have been scheduled {_exacerbation_events}'


def test_moderate_exacerbation():
    """ test moderate exacerbation leads to;
          i) moderate symptoms
         ii) non-emergency care seeking
        iii) getting inhaler """

    sim = get_simulation(1)  # get simulation object
    copd_module = sim.modules['Copd']  # the copd module

    # sim.event_queue.queue = []  # clear the event queues

    df = sim.population.props  # population dataframe
    person_id = df.index[0]  # get person id

    # reset individual properties. An individual should be alive and without an inhaler
    df.at[person_id, 'is_alive'] = True
    df.loc[person_id, 'age_years'] = 20
    df.at[person_id, 'ch_has_inhaler'] = False

    # check individuals do not have symptoms before an event is run
    assert 'breathless_moderate' not in sim.modules['SymptomManager'].has_what(person_id)

    # run Copd Exacerbation event on an individual and confirm they now have a
    # non-emergency symptom(breathless moderate)
    copd.CopdExacerbationEvent(copd_module, person_id, severe=False).run()
    assert 'breathless_moderate' in sim.modules['SymptomManager'].has_what(person_id)

    # Run health seeking behavior event and check non-emergency care is sought
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericFirstApptAtFacilityLevel0)

    # check an individual has no inhaler before  scheduling facility care event
    assert not df.loc[person_id, "ch_has_inhaler"]

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check no emergency care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericFirstApptAtFacilityLevel0)][0]
    ge.run(squeeze_factor=0.0)

    # check that no HSI_CopdTreatmentOnSevereExacerbation event is scheduled. Only inhaler should be given
    for _event in sim.modules['HealthSystem'].find_events_for_person(person_id):
        assert not isinstance(_event[1], HSI_Copd_TreatmentOnSevereExacerbation)

    # check inhaler is given
    assert df.loc[person_id, "ch_has_inhaler"]


def test_severe_exacerbation():
    """ test severe exacerbation leads to;
          i) emergency symptoms
         ii) emergency care seeking
        iii) gets treatment """

    sim = get_simulation(1)  # get simulation object
    copd_module = sim.modules['Copd']  # the copd module

    # reset value to make an individual eligible for moderate exacerbations
    df = sim.population.props
    person_id = df.index[0]  # get person id
    df.at[person_id, 'is_alive'] = True
    df.at[person_id, 'age_years'] = 20
    df.at[person_id, 'ch_has_inhaler'] = False

    # check an individual do not have emergency symptoms before an event is run
    assert 'breathless_severe' not in sim.modules['SymptomManager'].has_what(person_id)

    # schedule exacerbations event setting severe to True. This will ensure the individual has severe exacerbation
    copd.CopdExacerbationEvent(copd_module, person_id, severe=True).run()

    # severe exacerbation should lead to severe symptom(breathless severe in this case). check this is true
    assert 'breathless_severe' in sim.modules['SymptomManager'].has_what(person_id, copd_module)

    # # Run health seeking behavior event and check emergency care is sought
    hsp = HealthSeekingBehaviourPoll(module=sim.modules['HealthSeekingBehaviour'])
    hsp.run()
    # check that an instance of HSI_GenericFirstApptAtFacilityLevel1 is created
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericEmergencyFirstApptAtFacilityLevel1)

    # check an individual has no inhaler before  scheduling facility care event
    assert not df.loc[person_id, "ch_has_inhaler"]

    # Run the created instance of HSI_GenericEmergencyFirstApptAtFacilityLevel1 and check emergency care was sort
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericEmergencyFirstApptAtFacilityLevel1)][0]
    ge.run(squeeze_factor=0.0)

    # check that HSI_CopdTreatmentOnSevereExacerbation event is scheduled. Inhaler should also be given
    assert isinstance([ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
                       isinstance(ev[1], HSI_Copd_TreatmentOnSevereExacerbation)][0],
                      HSI_Copd_TreatmentOnSevereExacerbation)

    # check inhaler is given
    assert df.loc[person_id, "ch_has_inhaler"]


def test_death_rate():
    """A function that is testing death rate logic. What we want to test;
            i)   Zero death rate should lead to No death
            ii)  High death rate but perfect treatment should lead to No deaths
            iii) High death rate should lead to Many deaths
    """
    # create population dataframe from simulation
    sim = get_simulation(100)   # get simulation object
    copd_module = sim.modules['Copd']  # the copd module

    df = sim.population.props
    # reset some properties
    df.loc[df.index, 'is_alive'] = True
    df.loc[df.index, 'age_years'] = 30
    df.loc[df.index, 'ch_function'] = 6
    df.loc[df.index, 'ch_will_die_this_episode'] = False

    # 1) -------------TEST ZERO DEATH RATE LEAD TO ZERO DEATH------------------------
    # make death rate due to severe exacerbation to zero. This will ensure no copd death rate is scheduled
    copd_module.parameters['prob_will_die_sev_exacerbation'] = 0

    # call copd exacerbation event and check no one is scheduled to die
    for idx in range(len(df.index)):
        _event = copd.CopdExacerbationEvent(copd_module, idx, severe=True)
        _event.run()
    # no one should be scheduled to die
    assert not df.ch_will_die_this_episode.all(), 'no one should be scheduled to die when death rate is 0'

    # schedule death event and confirm no one has died
    for idx in range(len(df.index)):
        _event = copd.CopdDeath(copd_module, idx)
        _event.run()

    # no one should die
    assert df.is_alive.all(), 'no one should die when death rate is 0'

    # 2) -------- TEST HIGH DEATH RATE BUT PERFECT TREATMENT SHOULD LEAD TO NO DEATHS ------------
    # reset death rate due to severe exacerbation to 1. This will ensure many deaths are scheduled
    copd_module.parameters['prob_will_die_sev_exacerbation'] = 1.0
    # reset will survive given oxygen probability to 1.0 to ensure all survive when care is given
    copd_module.parameters['prob_will_survive_given_oxygen'] = 1.0
    # call copd exacerbation event and confirm all have been scheduled to die
    for idx in range(len(df.index)):
        _event = copd.CopdExacerbationEvent(copd_module, idx, severe=True)
        _event.run()

    # all individuals should be scheduled to die
    assert df.ch_will_die_this_episode.all(), 'not all individuals are scheduled to die'

    # call treatment on severe exacerbation event and check that deaths has been canceled. we assume perfect treatment
    # is given
    for idx in range(len(df.index)):
        _event = copd.HSI_Copd_TreatmentOnSevereExacerbation(copd_module, idx)
        _event.run(0.0)

    # all individuals should now not be scheduled to die
    assert not df.ch_will_die_this_episode.all(), 'now that individuals have received perfect treatment,' \
                                                  'death should be canceled'

    # schedule death event and confirm no one is dead
    for idx in range(len(df.index)):
        _event = copd.CopdDeath(copd_module, idx)
        _event.run()

    # no one should die
    assert df.is_alive.all(), 'individuals who have received perfect copd treatment should not die'

    # 3) -------------TEST HIGH DEATH RATE LEAD TO HIGH DEATHS------------------------
    # reset death rate due to severe exacerbation to 1. This will ensure many deaths are scheduled
    copd_module.parameters['prob_will_die_sev_exacerbation'] = 1
    # call copd exacerbation event and confirm all have been scheduled to die
    for idx in range(len(df.index)):
        _event = copd.CopdExacerbationEvent(copd_module, idx, severe=True)
        _event.run()

    # all individuals should be scheduled to die
    assert len(df) == df.ch_will_die_this_episode.sum(), 'not all individuals are scheduled to die'

    # schedule death event and confirm all are dead
    for idx in range(len(df.index)):
        _event = copd.CopdDeath(copd_module, idx)
        _event.run()

    # all individuals should die
    assert not df.is_alive.all(), 'all individuals should die when death rate is set to max rate'
