"""Test file for the Alri module"""

import os
from pathlib import Path

import pandas as pd
from tlo import Date, Simulation, logging, Module, Property, Types
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    alri,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.alri import AlriPollingEvent, AlriIncidentCase, AlriNaturalRecoveryEvent, AlriDeathEvent, \
    AlriCureEvent, HSI_Alri_GenericTreatment

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)

class PropertiesOfOtherModules(Module):
    """For the purpose of the testing, create a module to generate the properties upon which this module relies"""
    # todo - update these name to reflect the properties that are already defined:

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'tmp_haemophilus_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, self.PROPERTIES.keys()] = False

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        self.sim.population.props.at[child, self.PROPERTIES.keys()] = False


def get_sim(tmpdir):
    """Return simulation objection with Alri and other necessary modules registered."""
    sim = Simulation(start_date=start_date, seed=0, show_progress_bar=False, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "Alri": logging.INFO}
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=True, do_checks=True),
        PropertiesOfOtherModules()
    )

    return sim


def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(months=1)
    popsize = 100
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    check_dtypes(sim)


def test_basic_run_lasting_two_years(tmpdir):
    """Check logging results in a run of the model for two years, with daily property config checking"""
    dur = pd.DateOffset(years=2)
    popsize = 500
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)

    # Read the log for the population counts of incidence:
    log_counts = parse_log_file(sim.log_filepath)['tlo.methods.alri']['event_counts']
    assert 0 < log_counts['incident_cases'].sum()
    assert 0 < log_counts['recovered_cases'].sum()
    assert 0 < log_counts['deaths'].sum()
    assert 0 == log_counts['cured_cases'].sum()

    # Read the log for the one individual being tracked:
    log_one_person = parse_log_file(sim.log_filepath)['tlo.methods.alri']['log_individual']
    log_one_person['date'] = pd.to_datetime(log_one_person['date'])
    log_one_person = log_one_person.set_index('date')
    assert log_one_person.index.equals(pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1)))
    assert set(log_one_person.columns) == set(sim.modules['Alri'].PROPERTIES.keys())


def test_alri_polling(tmpdir):
    """Cause infection --> ALRI onset --> complication --> death and check it is logged correctly"""
    # get simulation object:
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # Make incidence of alri very high :
    params = sim.modules['Alri'].parameters
    for p in params:
        if p.startswith('base_inc_rate_ALRI_by_'):
            params[p] = [3*v for v in params[p]]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Run polling event: check that an incident case is produced:
    polling = AlriPollingEvent(sim.modules['Alri'])
    polling.apply(sim.population)
    assert len([q for q in sim.event_queue.queue if isinstance(q[2], AlriIncidentCase)]) > 0


def test_nat_hist_recovery(tmpdir):
    """Check: Infection onset --> recovery"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 0%
    sim.modules['Alri'].risk_of_death = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check properties of this individual: (should now be infected and with scheduled_recovery_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert not pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Check events scheduled for this person:
    next_event_tuple = sim.find_events_for_person(person_id)[0]
    date = next_event_tuple[0]
    event = next_event_tuple[1]
    assert date > sim.date
    assert isinstance(event, AlriNaturalRecoveryEvent)

    # Run the recovery event:
    sim.date = date
    event.apply(person_id=person_id)

    # Check properties of this individual: (should now not be infected)
    person = df.loc[person_id]
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # check it's logged (one infection + one recovery)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_death(tmpdir):
    """Check: Infection onset --> death"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100%
    sim.modules['Alri'].risk_of_death = LinearModel.multiplicative()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check properties of this individual: (should now be infected and with a scheduled_death_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check events schedule for this person:
    next_event_tuple = sim.find_events_for_person(person_id)[0]
    date = next_event_tuple[0]
    event = next_event_tuple[1]
    assert date > sim.date
    assert isinstance(event, AlriDeathEvent)

    # Run the death event:
    sim.date = date
    event.apply(person_id=person_id)

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not person['is_alive']

    # check it's logged (one infection + one death)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_cure_if_recovery_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to recover naturally, it cause the episode to
    end earlier."""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 0%
    sim.modules['Alri'].risk_of_death = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check properties of this individual: (should now be infected and with a scheduled_recovery_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert not pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Check events schedule for this person:
    next_event_tuple = sim.find_events_for_person(person_id)[0]
    date_of_scheduled_recovery = next_event_tuple[0]
    recovery_event = next_event_tuple[1]
    assert date_of_scheduled_recovery > sim.date
    assert isinstance(recovery_event, AlriNaturalRecoveryEvent)

    # Run a Cure Event
    cure_event = AlriCureEvent(person_id=person_id, module=sim.modules['Alri'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not infected and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Run the recovery event that was originally scheduled) - this should have no effect
    sim.date = recovery_event
    recovery_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])


    # check it's logged (one infection + one cure)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_cure_if_death_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to die, it cause the episode to end without
    the person dying.
    """

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100%
    sim.modules['Alri'].risk_of_death = LinearModel.multiplicative()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check properties of this individual:
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check events schedule for this person:
    next_event_tuple = sim.find_events_for_person(person_id)[0]
    date_of_scheduled_death = next_event_tuple[0]
    death_event = next_event_tuple[1]
    assert date_of_scheduled_death > sim.date
    assert isinstance(death_event, AlriDeathEvent)

    # Run a Cure Event:
    cure_event = AlriCureEvent(person_id=person_id, module=sim.modules['Alri'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not infected and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Run the death event that was originally scheduled) - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']

    # check it's logged (one infection + one cure)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_treatment(tmpdir):
    """Test that providing a treatment prevent death and causes there to be a CureEvent Scheduled"""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100%
    sim.modules['Alri'].risk_of_death = LinearModel.multiplicative()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check properties of this individual:
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check events schedule for this person:
    next_event_tuple = sim.find_events_for_person(person_id)[0]
    date_of_scheduled_death = next_event_tuple[0]
    death_event = next_event_tuple[1]
    assert date_of_scheduled_death > sim.date
    assert isinstance(death_event, AlriDeathEvent)

    # Run the 'do_treatment' function
    sim.modules['Alri'].do_treatment(person_id=person_id, prob_of_cure=1.0)

    # Run the death event that was originally scheduled) - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['ri_current_infection_status']

    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if isinstance(event_tuple[1], AlriCureEvent)][0]

    # Run the CureEvent
    cure_event.apply(person_id=person_id)

    # Check that the person is not infected and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # check it's logged (one infection + one cure)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_complication_and_severe_complications():
    """todo TBD"""
    pass


def test_use_of_HSI(tmpdir):
    """Check that the HSI template works"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100%
    sim.modules['Alri'].risk_of_death = LinearModel.multiplicative()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

    # Check not on treatment:
    assert not df.at[person_id, 'ri_on_treatment']
    assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    # Run the HSI event
    hsi = HSI_Alri_GenericTreatment(person_id=person_id, module=sim.modules['Alri'])
    hsi.run(squeeze_factor=0.0)

    # Check that person is now on treatment:
    assert df.at[person_id, 'ri_on_treatment']
    assert sim.date == df.at[person_id, 'ri_ALRI_tx_start_date']



# todo - Additionally need some kind of test bed so that Ines can see the effects of the linear models she is programming.






