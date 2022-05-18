"""Test file for the Alri module"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DateOffset

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.alri import (
    AlriCureEvent,
    AlriDeathEvent,
    AlriIncidentCase,
    AlriIncidentCase_Lethal_Severe_Pneumonia,
    AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia,
    AlriLoggingEvent,
    AlriNaturalRecoveryEvent,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    Models,
)
from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)


@pytest.fixture
def sim_hs_all_consumables(tmpdir, seed):
    """Return simulation objection with Alri and other necessary modules registered.
    All consumables available"""
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'tmp',
            'directory': tmpdir,
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.alri": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO
            }
        }
    )
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='all'),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=0, do_checks=True),
        AlriPropertiesOfOtherModules(),
    )
    return sim


@pytest.fixture
def sim_hs_no_consumables(tmpdir, seed):
    """Return simulation objection with Alri and other necessary modules registered.
    No consumable available"""
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'tmp',
            'directory': tmpdir,
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.alri": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO
            }
        }
    )
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='none'),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=0, do_checks=True),
        AlriPropertiesOfOtherModules(),
    )
    return sim


@pytest.fixture
def sim_hs_default_consumables(tmpdir, seed):
    """Return simulation objection with Alri and other necessary modules registered.
    Default consumables availability"""
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'tmp',
            'directory': tmpdir,
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.alri": logging.INFO}
        }
    )
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='default'),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=0, do_checks=True),
        AlriPropertiesOfOtherModules(),
    )
    return sim


def check_dtypes(sim_hs_all_consumables):
    sim = sim_hs_all_consumables
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_integrity_of_linear_models(sim_hs_all_consumables):
    """Run the models to make sure that is specified correctly and can run."""
    sim = sim_hs_all_consumables
    sim.make_initial_population(n=5000)
    alri = sim.modules['Alri']
    df = sim.population.props
    person_id = 0

    # make the models
    models = Models(alri)

    # --- compute_risk_of_acquisition & incidence_equations_by_pathogen:
    # 1) if no vaccine:
    df['va_pneumo_all_doses'] = False
    df['va_hib_all_doses'] = False
    models.make_model_for_acquisition_risk()

    for pathogen in alri.all_pathogens:
        res = models.compute_risk_of_acquisition(
            pathogen=pathogen,
            df=df.loc[df.is_alive & (df.age_years < 5)]
        )
        assert (res > 0).all() and (res <= 1.0).all()
        assert not res.isna().any()
        assert 'float64' == res.dtype.name

    # 2) If there are vaccines:
    # 2a) pneumococcal vaccine: if efficacy of vaccine is perfect, whoever has vaccine should have no risk of infection
    # from Strep_pneumoniae_PCV13
    models.p['rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y'] = 0.0
    models.p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y'] = 0.0
    df['va_pneumo_all_doses'] = True
    assert (0.0 == models.compute_risk_of_acquisition(
        pathogen='Strep_pneumoniae_PCV13',
        df=df.loc[df.is_alive & (df.age_years < 5)])
            ).all()

    # hib vaccine: if efficacy of vaccine is perfect, whoever has vaccine should have no risk of infection
    # from Hib (H.influenzae type-b)
    models.p['rr_Hib_ALRI_with_Hib_vaccine'] = 0.0
    df['va_hib_all_doses'] = True
    assert (0.0 == models.compute_risk_of_acquisition(
        pathogen='Hib',
        df=df.loc[df.is_alive & (df.age_years < 5)])
            ).all()

    # --- determine_disease_type
    # set efficacy of pneumococcal vaccine to be 100% (i.e. 0 relative risk of infection)
    models.p['rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y'] = 0.0
    models.p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y'] = 0.0
    for patho in alri.all_pathogens:
        for age in range(0, 100):
            for va_pneumo_all_doses in [True, False]:
                disease_type, bacterial_coinfection = \
                    models.determine_disease_type_and_secondary_bacterial_coinfection(
                        age=age,
                        pathogen=patho,
                        va_hib_all_doses=True,

                        va_pneumo_all_doses=va_pneumo_all_doses
                    )

                assert disease_type in alri.disease_types

                if patho in alri.pathogens['bacterial']:
                    assert pd.isnull(bacterial_coinfection)
                elif patho in alri.pathogens['fungal/other']:
                    assert pd.isnull(bacterial_coinfection)
                else:
                    # viral primary infection- may have a bacterial coinfection or may not:
                    assert pd.isnull(bacterial_coinfection) or \
                           bacterial_coinfection in alri.pathogens['bacterial']
                    # check that if has had pneumococcal vaccine they are not coinfected with `Strep_pneumoniae_PCV13`
                    if va_pneumo_all_doses:
                        assert bacterial_coinfection != 'Strep_pneumoniae_PCV13'

    # --- complications
    for patho in alri.all_pathogens:
        for coinf in (alri.pathogens['bacterial'] + [np.nan]):
            for disease_type in alri.disease_types:
                res = models.get_complications_that_onset(disease_type=disease_type,
                                                          primary_path_is_bacterial=(
                                                              patho in sim.modules['Alri'].pathogens['bacterial']
                                                          ),
                                                          has_secondary_bacterial_inf=pd.notnull(coinf)
                                                          )
                assert isinstance(res, set)
                assert all([c in alri.complications for c in res])

    # --- symptoms_for_disease
    for disease_type in alri.disease_types:
        res = models.symptoms_for_disease(disease_type)
        assert isinstance(res, set)
        assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])

    # --- symptoms_for_complication
    for complication in alri.complications:
        res = models.symptoms_for_complication(complication, oxygen_saturation='<90%')
        assert isinstance(res, set)
        assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])

    # --- death
    for disease_type in alri.disease_types:
        df.loc[person_id, [
            'ri_disease_type',
            'age_years',
            'ri_complication_sepsis',
            'hv_inf',
            'un_clinical_acute_malnutrition',
            'nb_low_birth_weight_status']
        ] = (
            disease_type,
            0,
            False,
            True,
            'SAM',
            'low_birth_weight'
        )
        res = models.will_die_of_alri(person_id)
        assert isinstance(res, (bool, np.bool_))


def test_basic_run(sim_hs_all_consumables):
    """Short run of the module using default parameters with check on dtypes"""
    sim = sim_hs_all_consumables

    dur = pd.DateOffset(months=1)
    popsize = 100
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    check_dtypes(sim_hs_all_consumables)


@pytest.mark.slow
def test_basic_run_lasting_two_years(sim_hs_all_consumables):
    """Check logging results in a run of the model for two years, including HSI, with daily property config checking"""
    sim = sim_hs_all_consumables

    dur = pd.DateOffset(years=2)
    popsize = 5000
    sim.make_initial_population(n=popsize)

    # increase death risk
    params = sim.modules['Alri'].parameters
    params['baseline_odds_alri_death'] *= 5.0

    sim.simulate(end_date=start_date + dur)

    # Read the log for the population counts of incidence:
    log_counts = parse_log_file(sim.log_filepath)['tlo.methods.alri']['event_counts']
    assert 0 < log_counts['incident_cases'].sum()
    assert 0 < log_counts['recovered_cases'].sum()
    assert 0 < log_counts['deaths'].sum()
    assert 0 < log_counts['cured_cases'].sum()

    # Read the log for the one individual being tracked:
    log_one_person = parse_log_file(sim.log_filepath)['tlo.methods.alri']['log_individual']
    log_one_person['date'] = pd.to_datetime(log_one_person['date'])
    log_one_person = log_one_person.set_index('date')
    assert log_one_person.index.equals(pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1)))
    assert set(log_one_person.columns) == set(sim.modules['Alri'].PROPERTIES.keys())


def test_alri_polling(sim_hs_all_consumables):
    """Check polling events leads to incident cases"""
    sim = sim_hs_all_consumables

    # get simulation object:
    popsize = 100
    sim.make_initial_population(n=popsize)

    # Make incidence of alri very high :
    params = sim.modules['Alri'].parameters
    for p in params:
        if p.startswith('base_inc_rate_ALRI_by_'):
            params[p] = [10 * v for v in params[p]]

    # start simulation
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # Run polling event: check that an incident case is produced:
    polling = AlriPollingEvent(sim.modules['Alri'])
    polling.run()
    assert len([q for q in sim.event_queue.queue if isinstance(q[2], AlriIncidentCase)]) > 0


def test_nat_hist_recovery(sim_hs_all_consumables):
    """Check: Infection onset --> recovery"""

    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 0% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return False

    sim.modules['Alri'].models.will_die_of_alri = death

    # make probability of symptoms very high
    params = sim.modules['Alri'].parameters
    all_symptoms = {
        'cough', 'difficult_breathing', 'tachypoea', 'chest_indrawing', 'danger_signs'
    }
    for p in params:
        if any([p.startswith(f"prob_{symptom}") for symptom in all_symptoms]):
            if isinstance(params[p], float):
                params[p] = 1.0
            else:
                params[p] = [1.0] * len(params[p])

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check properties of this individual: (should now be infected and with scheduled_recovery_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert not pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Check that they have some symptoms caused by ALRI
    assert 0 < len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Alri']))

    # Check that there is a AlriNaturalRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], AlriNaturalRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run the recovery event:
    sim.date = date_of_scheduled_recov
    recov_event.run()

    # Check properties of this individual: (should now not be infected)
    person = df.loc[person_id]
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # check they they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Alri']))

    # check it's logged (one infection + one recovery)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_death(sim_hs_all_consumables):
    """Check: Infection onset --> death"""
    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100% (not using a lambda function because code uses the keyword argument for clarity)
    def __will_die_of_alri(person_id):
        return True

    sim.modules['Alri'].models.will_die_of_alri = __will_die_of_alri

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check properties of this individual: (should now be infected and with a scheduled_death_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check that there is a AlriDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], AlriDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run the death event:
    sim.date = date_of_scheduled_death
    death_event.run()

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not person['is_alive']

    # check it's logged (one infection + one death)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_cure_if_recovery_scheduled(sim_hs_all_consumables):
    """Show that if a cure event is run before when a person was going to recover naturally, it cause the episode to
    end earlier."""
    sim = sim_hs_all_consumables

    popsize = 100

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 0% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return False

    sim.modules['Alri'].models.will_die_of_alri = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check properties of this individual: (should now be infected and with a scheduled_recovery_date)
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert not pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Check that there is a AlriNaturalRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], AlriNaturalRecoveryEvent)
                         ][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run a Cure Event
    cure_event = AlriCureEvent(person_id=person_id, module=sim.modules['Alri'])
    cure_event.run()

    # Check that the person is not infected and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert not person['ri_current_infection_status']
    assert pd.isnull(person['ri_primary_pathogen'])
    assert pd.isnull(person['ri_start_of_current_episode'])
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert pd.isnull(person['ri_scheduled_death_date'])

    # Run the recovery event that was originally scheduled) - this should have no effect
    sim.date = date_of_scheduled_recov
    recov_event.run()
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


def test_nat_hist_cure_if_death_scheduled(sim_hs_all_consumables):
    """Show that if a cure event is run before when a person was going to die, it cause the episode to end without
    the person dying."""
    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return True

    sim.modules['Alri'].models.will_die_of_alri = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check properties of this individual:
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check that there is a AlriDeathEvent scheduled for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], AlriDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event:
    cure_event = AlriCureEvent(person_id=person_id, module=sim.modules['Alri'])
    cure_event.run()

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
    death_event.run()
    person = df.loc[person_id]
    assert person['is_alive']

    # check it's logged (one infection + one cure)
    assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_immediate_onset_complications(sim_hs_all_consumables):
    """Check that if probability of immediately onsetting complications is 100%, then a person has all those
    complications immediately onset"""
    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)

    # make risk of immediate onset complications be 100% (so that person has all the complications)
    params = sim.modules['Alri'].parameters
    params['prob_pulmonary_complications_in_pneumonia'] = 1.0
    params['prob_bacteraemia_in_pneumonia'] = 1.0
    params['prob_progression_to_sepsis_with_bacteraemia'] = 1.0
    for p in params:
        if any([p.startswith(f'prob_{c}') for c in sim.modules['Alri'].complications]):
            params[p] = 1.0

    # start simulation
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case for a viral pathogen:
    pathogen = 'other_viral_pathogens'
    incidentcase = AlriIncidentCase(module=sim.modules['Alri'], person_id=person_id, pathogen=pathogen)
    incidentcase.run()

    # Check has some complications ['pneumothorax', 'pleural_effusion', 'hypoxaemia'] are present for pneumonia disease
    # caused by viruses
    if df.at[person_id, 'ri_disease_type'] == 'pneumonia':
        complications_cols = [
            f"ri_complication_{complication}" for complication in
            ['pneumothorax', 'pleural_effusion', 'hypoxaemia']]
        assert df.loc[person_id, complications_cols].all()

    # Check SpO2<93% if hypoxaemia is present
    if df.at[person_id, 'ri_complication_hypoxaemia']:
        assert df.at[person_id, 'ri_SpO2_level'] != '>=93%'


def test_no_immediate_onset_complications(sim_hs_all_consumables):
    """Check that if probability of immediately onsetting complications is 0%, then a person has none of those
    complications immediately onset
    """
    sim = sim_hs_all_consumables

    popsize = 100

    # make risk of immediate-onset complications be 0%
    params = sim.modules['Alri'].parameters
    params['prob_pulmonary_complications_in_pneumonia'] = 0.0
    params['prob_bacteraemia_in_pneumonia'] = 0.0
    for p in params:
        if any([p.startswith(f'prob_{c}') for c in sim.modules['Alri'].complications]):
            params[p] = 0.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(module=sim.modules['Alri'], person_id=person_id, pathogen=pathogen)
    incidentcase.run()

    # Check has no complications following onset (check #1)
    complications_cols = [f"ri_complication_{complication}" for complication in sim.modules['Alri'].complications]
    assert not df.loc[person_id, complications_cols].any()


def test_classification_based_on_symptoms_and_imci(sim_hs_all_consumables):
    """Check that `_get_disease_classification` gives the expected classification."""

    def make_hw_assesement_perfect(sim):
        p = sim.modules['Alri'].parameters
        p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
        p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
        p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
        p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0

    sim = sim_hs_all_consumables
    make_hw_assesement_perfect(sim)
    sim.make_initial_population(n=1000)
    hsi_alri_treatment = HSI_Alri_Treatment(sim.modules['Alri'], 0)

    # Construct examples of the correct classification of disease, given a range of symptoms, age and facility level
    imci_classification_on_symptoms = (
        # -- Children older than 2 months
        ('chest_indrawing_pneumonia',
         {'symptoms': ['chest_indrawing'],
          'child_is_younger_than_2_months': False}),
        ('chest_indrawing_pneumonia',
         {'symptoms': ['chest_indrawing', 'tachypnoea'],
          'child_is_younger_than_2_months': False}),
        ('fast_breathing_pneumonia',
         {'symptoms': ['tachypnoea'],
          'child_is_younger_than_2_months': False}),
        ('danger_signs_pneumonia',
         {'symptoms': ['danger_signs'],
          'child_is_younger_than_2_months': False}),
        ('danger_signs_pneumonia',
         {'symptoms': ['danger_signs', 'chest_indrawing'],
          'child_is_younger_than_2_months': False}),
        ('chest_indrawing_pneumonia',
         {'symptoms': ['chest_indrawing'],
          'child_is_younger_than_2_months': False}),

        # -- Children younger than 2 months
        ('serious_bacterial_infection',
         {'symptoms': ['danger_signs', 'chest_indrawing'],
          'child_is_younger_than_2_months': True}),
        ('fast_breathing_pneumonia',
         {'symptoms': ['tachypnoea'],
          'child_is_younger_than_2_months': True}),
        ('cough_or_cold',
         {'symptoms': ['cough'],
          'child_is_younger_than_2_months': True}),
        ('serious_bacterial_infection',
         {'symptoms': ['cough', 'danger_signs', 'difficult_breathing', 'fever', 'chest_indrawing'],
          'child_is_younger_than_2_months': True}),
        ('serious_bacterial_infection',
         {'symptoms': ['cough', 'danger_signs', 'difficult_breathing', 'fever', 'chest_indrawing'],
          'child_is_younger_than_2_months': True}),
    )

    recognised_classifications = {
        'fast_breathing_pneumonia',
        'chest_indrawing_pneumonia',
        'danger_signs_pneumonia',
        'cough_or_cold',
        'serious_bacterial_infection',
    }
    assert set([x[0] for x in imci_classification_on_symptoms]).issubset(recognised_classifications)

    _get_disease_classification = hsi_alri_treatment._get_disease_classification

    for _correct_imci_classification_on_symptoms, chars in imci_classification_on_symptoms:
        # If no oximeter available and does not need oxygen, and perfect HW assessment: classification should be the
        # IMCI classification
        assert _correct_imci_classification_on_symptoms == _get_disease_classification(
            age_exact_years=0.05 if chars['child_is_younger_than_2_months'] else 1.0,
            symptoms=chars['symptoms'],
            oxygen_saturation='>=93%',
            facility_level='1b',
            use_oximeter=False
        )

        # If no oximeter available and does need oxygen, and perfect HW assessment: classification should be the
        # IMCI classification
        assert _correct_imci_classification_on_symptoms == _get_disease_classification(
            age_exact_years=0.05 if chars['child_is_younger_than_2_months'] else 1.0,
            symptoms=chars['symptoms'],
            oxygen_saturation='<90%',
            facility_level='1b',
            use_oximeter=False
        )

        # If oximeter available and does oxygen, then classification should be the 'danger_signs_pneumonia'
        _classificiation = 'serious_bacterial_infection' if chars[
            'child_is_younger_than_2_months'] else 'danger_signs_pneumonia'
        assert _classificiation == _get_disease_classification(
            age_exact_years=0.05 if chars['child_is_younger_than_2_months'] else 1.0,
            symptoms=chars['symptoms'],
            oxygen_saturation='<90%',
            facility_level='1b',
            use_oximeter=True
        ), f"{_correct_imci_classification_on_symptoms=}"


def test_do_effects_of_alri_treatment(sim_hs_all_consumables):
    """Check that running `do_alri_treatment` can prevent a death from occurring."""
    sim = sim_hs_all_consumables
    popsize = 100
    sim.make_initial_population(n=popsize)

    # Set the treatment failure parameters to null
    p = sim.modules['Alri'].parameters
    p['5day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] = 0.0
    p['3day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] = 0.0
    p['1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2'] = 0.0
    p['rr_1st_line_treatment_failure_low_oxygen_saturation'] = 0.0

    # start simulation
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use (not currently infected):
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Make the incident case one that should cause 'severe_pneumonia' to need to be provided
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase_Lethal_Severe_Pneumonia(person_id=person_id, pathogen=pathogen,
                                                            module=sim.modules['Alri'])
    incidentcase.run()

    # Check properties of this individual:
    person = df.loc[person_id]
    assert person['ri_current_infection_status']
    assert person['ri_primary_pathogen'] == pathogen
    assert person['ri_start_of_current_episode'] == sim.date
    assert pd.isnull(person['ri_scheduled_recovery_date'])
    assert not pd.isnull(person['ri_scheduled_death_date'])

    # Check that there is a AlriDeathEvent schedule for this person:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], AlriDeathEvent)
                         ][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run the 'do_alri_treatment' function (as if from the HSI)
    sim.modules['Alri'].do_effects_of_treatment(person_id=person_id,
                                                antibiotic_provided='1st_line_IV_antibiotics',
                                                oxygen_provided=True)

    # Run the death event that was originally scheduled) - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.run()
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['ri_current_infection_status']

    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                  isinstance(event_tuple[1], AlriCureEvent)][0]

    # Run the CureEvent
    cure_event.run()

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


def test_severe_pneumonia_referral_from_hsi_first_appts(sim_hs_all_consumables):
    """Check that a person is scheduled a treatment HSI following a presentation at
    HSI_GenericFirstApptAtFacilityLevel0 with severe pneumonia."""
    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue
    sim.modules['HealthSystem'].reset_queue()

    # Get person to use (under 5 years old and not infected:
    df = sim.population.props
    under5s = df.loc[df.is_alive
                     & ~df['ri_current_infection_status']
                     & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # Give this person severe pneumonia:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase_Lethal_Severe_Pneumonia(person_id=int(person_id),
                                                            pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check infected and not on treatment:
    assert not df.at[person_id, 'ri_on_treatment']

    # Run the healthcare seeking behaviour poll
    sim.modules['HealthSeekingBehaviour'].theHealthSeekingBehaviourPoll.run()

    # Check that person 0 has an Emergency Generic HSI scheduled
    generic_appt = [event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_GenericEmergencyFirstApptAtFacilityLevel1)][0]

    # Run generic appt and check that there is an Outpatient `HSI_Alri_Treatment` scheduled
    generic_appt.run(squeeze_factor=0.0)
    hsi1 = [event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
            if isinstance(event_tuple[1], HSI_Alri_Treatment)
            ][0]
    assert hsi1.TREATMENT_ID == 'Alri_Pneumonia_Treatment_Outpatient'

    # Check not on treatment before referral:
    assert not df.at[person_id, 'ri_on_treatment']

    # run the first outpatient HSI ... which will lead to an in-patient HSI being scheduled
    hsi1.run(squeeze_factor=0.0)
    hsi2 = [event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
            if isinstance(event_tuple[1], HSI_Alri_Treatment)
            ][1]
    assert hsi2.TREATMENT_ID == 'Alri_Pneumonia_Treatment_Inpatient'

    # Run the second HSI (an in-patient)
    hsi2.run(squeeze_factor=0.0)

    # Check that the person is now on treatment
    assert df.at[person_id, 'ri_on_treatment']


def generate_hsi_sequence(sim, incident_case_event):
    """For a given simulation, let one person be affected by Alri, and record all the HSI that they have."""

    def make_hw_assesement_perfect(sim):
        """Make healthcare worker assesment perfect"""
        params = sim.modules['Alri'].parameters
        params['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

    def make_non_emergency_hsi_happen_immediately(sim):
        """Set the delay between symptoms onset and the generic HSI occurring to the least possible number of days."""
        sim.modules['HealthSeekingBehaviour'].parameters['max_days_delay_to_generic_HSI_after_symptoms'] = 0

    sim.make_initial_population(n=100)
    make_hw_assesement_perfect(sim)
    make_non_emergency_hsi_happen_immediately(sim)

    def _initialise_simulation_other_jobs(sim, **kwargs):
        """The other jobs that the usual `initialise_simulation` in `Alri` has to do in order for the module to work."""
        module = sim.modules['Alri']
        p = module.parameters
        module.max_duration_of_episode = DateOffset(
            days=p['max_alri_duration_in_days_without_treatment'] + p['days_between_treatment_and_cure']
        )
        module.models = Models(module)
        module.look_up_consumables()
        module.logging_event = AlriLoggingEvent(module)
        module.daly_wts['daly_non_severe_ALRI'] = sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
        module.daly_wts['daly_severe_ALRI'] = sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

    def one_person_to_have_disease(self, **kwargs):
        """Over-ride the `initialise_simulation` method of `Alri` to let one person have disease with particular
        characteristic."""
        _initialise_simulation_other_jobs(sim)

        # Get person to use (under 5 years old and not infected):
        df = sim.population.props
        under5s = df.loc[df.is_alive
                         & ~df['ri_current_infection_status']
                         & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        sim.schedule_event(
            incident_case_event(
                sim.modules['Alri'], person_id=person_id, pathogen=list(sim.modules['Alri'].all_pathogens)[0]),
            self.date
        )

    sim.modules['Alri'].initialise_simulation = one_person_to_have_disease
    sim.simulate(end_date=Date(2010, 3, 1))

    df = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']['HSI_Event'].set_index('date')

    # Return list of tuples of TREATMENT_ID and Facility_Level
    mask = df.TREATMENT_ID.str.startswith('Alri_') | df.TREATMENT_ID.str.startswith('FirstAttendance_')
    return [r for r in df.loc[mask, ['TREATMENT_ID', 'Facility_Level']].itertuples(index=False, name=None)]


# todo @Ines - In the below, we check that the sequence of HSI's is correct under some different circumstances.
#   We may wish to add more, if there are specific things you want to check. Maybe for `serious_bacterial_infection`
#   for the "direct referral" thing.


def test_treatment_pathway_if_all_consumables_mild_case(sim_hs_all_consumables):
    """Examine the treatment pathway for a person with a particular category of disease if consumables are available."""
    # Mild case (fast_breathing_pneumonia) and available consumables --> treatment at level 0, following non-emergency
    # appointment.
    assert [
               ('FirstAttendance_NonEmergency', '0'),
               ('Alri_Pneumonia_Treatment_Outpatient', '0'),
               ('Alri_Pneumonia_Treatment_Outpatient_Followup', '0'),
           ] == generate_hsi_sequence(sim=sim_hs_all_consumables,
                                      incident_case_event=AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia)


def test_treatment_pathway_if_all_consumables_severe_case(sim_hs_all_consumables):
    """Examine the treatment pathway for a person with a particular category of disease if consumables are available."""
    # Severe case and available consumables --> treatment as in-patient at level 1b, following emergency appointment
    # and referral.
    assert [
               ('FirstAttendance_Emergency', '1b'),
               ('Alri_Pneumonia_Treatment_Outpatient', '1b'),
               ('Alri_Pneumonia_Treatment_Inpatient', '1b'),
               ('Alri_Pneumonia_Treatment_Outpatient_Followup', '1b')
           ] == generate_hsi_sequence(sim=sim_hs_all_consumables,
                                      incident_case_event=AlriIncidentCase_Lethal_Severe_Pneumonia)


def test_treatment_pathway_if_no_consumables_mild_case(sim_hs_no_consumables):
    """Examine the treatment pathway for a person with a particular category of disease if consumables are available."""
    # Mild case (fast_breathing_pneumonia) and not available consumables --> successive referrals up to level 2,
    # following non-emergency appointment.
    assert [
               ('FirstAttendance_NonEmergency', '0'),
               ('Alri_Pneumonia_Treatment_Outpatient', '0'),
               ('Alri_Pneumonia_Treatment_Outpatient', '1a'),  # <-- referral due to lack of consumables
               ('Alri_Pneumonia_Treatment_Outpatient', '1b'),  # <-- referral due to lack of consumables
               ('Alri_Pneumonia_Treatment_Outpatient', '2'),   # <-- referral due to lack of consumables
               # (No follow-up appointment because treatment is never provided, due to lack of consumables).
           ] == generate_hsi_sequence(sim=sim_hs_no_consumables,
                                      incident_case_event=AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia)


def test_treatment_pathway_if_no_consumables_severe_case(sim_hs_no_consumables):
    """Examine the treatment pathway for a person with a particular category of disease if consumables are available."""
    # Severe case and not available consumables --> successive referrals up to level 2, following emergency
    # appointment.
    assert [
               ('FirstAttendance_Emergency', '1b'),
               ('Alri_Pneumonia_Treatment_Outpatient', '1b'),
               ('Alri_Pneumonia_Treatment_Inpatient', '1b'),
               ('Alri_Pneumonia_Treatment_Inpatient', '2'),  # <-- referral due to lack of consumables
               # (No follow-up appointment because treatment is never provided, due to lack of consumables).
           ] == generate_hsi_sequence(sim=sim_hs_no_consumables,
                                      incident_case_event=AlriIncidentCase_Lethal_Severe_Pneumonia)
