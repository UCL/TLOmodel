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
from tlo.methods.alri import AlriPollingEvent, AlriIncidentCase, AlriNaturalRecoveryEvent, AlriDeathEvent

# Path to the resource files used by the disease and intervention methods

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

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

    # Run incidence event: check that a death or recovery event is scheduled accordingly
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

    # Check events schedule for this person:
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

    # Run incidence event: check that a death or recovery event is scheduled accordingly
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


def test_cure_event():
    """Show that if a cure event is scheduled the person does not die (if they were going to die) or causes an early
    end to the episode (if they were going to recover)"""
    pass

def test_if_infection_leathal_all_die():
    """Show that if CFR is 100%, every infection results in a death"""
    pass

def test_treatment_prevents_deaths():
    """Show that treatment, when provided and has 100% effectiveness, prevents all deaths"""
    pass


# todo - Additionally need some kind of test bed so that Ines can see the effects of the linear models she is programming.






