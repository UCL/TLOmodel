"""Test file for the Alri module"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    AlriNaturalRecoveryEvent,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_IMCI_Pneumonia_Treatment_Inpatient_level_1b,
    HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2,
    HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a,
    HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b,
    HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2,
    Models,
)
from tlo.methods.healthseekingbehaviour import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
    HSI_GenericFirstApptAtFacilityLevel0,
)

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
    """Check that `symptom_based_classification` gives the expected classification."""
    sim = sim_hs_all_consumables

    # Construct scenario and the expected classification if using only symptoms:
    classification_on_symptoms = (
        (
            'chest_indrawing_pneumonia',
            {'symptoms': ['chest_indrawing'], 'facility_level': '1a', 'age_exact_years': 0.5}),
        ('chest_indrawing_pneumonia',
         {'symptoms': ['chest_indrawing', 'tachypnoea'], 'facility_level': '1a', 'age_exact_years': 0.5}),
        ('fast_breathing_pneumonia', {'symptoms': ['tachypnoea'], 'facility_level': '1b', 'age_exact_years': 0.5}),
        ('danger_signs_pneumonia', {'symptoms': ['danger_signs'], 'facility_level': '1b', 'age_exact_years': 0.5}),
        ('danger_signs_pneumonia',
         {'symptoms': ['danger_signs', 'chest_indrawing'], 'facility_level': '1b', 'age_exact_years': 0.5}),
        ('serious_bacterial_infection',
         {'symptoms': ['chest_indrawing'], 'facility_level': '1b', 'age_exact_years': 0.1}),
        ('serious_bacterial_infection', {'symptoms': ['danger_signs'], 'facility_level': '2', 'age_exact_years': 0.1}),
        ('fast_breathing_pneumonia', {'symptoms': ['tachypnoea'], 'facility_level': '1b', 'age_exact_years': 0.1}),
        ('not_handled_at_facility_0', {'symptoms': ['tachypnoea'], 'facility_level': '0', 'age_exact_years': 0.1}),
        ('cough_or_cold', {'symptoms': ['cough'], 'facility_level': '1a', 'age_exact_years': 0.5}),
        ('serious_bacterial_infection',
         {'symptoms': ['cough', 'danger_signs', 'difficult_breathing', 'fever', 'chest_indrawing'],
          'facility_level': '2', 'age_exact_years': 0.1}),
        ('danger_signs_pneumonia',
         {'symptoms': ['cough', 'danger_signs', 'difficult_breathing', 'fever', 'chest_indrawing'],
          'facility_level': '2', 'age_exact_years': 1})
    )

    recognised_classifications = {
        'fast_breathing_pneumonia',
        'chest_indrawing_pneumonia',
        'danger_signs_pneumonia',
        'cough_or_cold',
        'serious_bacterial_infection',
        'not_handled_at_facility_0'
    }

    final_classification_hw_and_oximeter = sim.modules['Alri'].final_classification
    symptom_based_classification = sim.modules['Alri'].imci_classification_based_on_symptoms

    for correct_classification_on_symptoms, chars in classification_on_symptoms:
        # Check classification using only symptoms:
        assert symptom_based_classification(**chars) in recognised_classifications
        assert correct_classification_on_symptoms == symptom_based_classification(**chars)

        # Check IMCI classification if oximeter not available (should be same as symptoms)
        assert correct_classification_on_symptoms == final_classification_hw_and_oximeter(
            hw_assigned_classification=correct_classification_on_symptoms,
            # Assumes perfect quality of care by the health workers
            age_exact_years=0.5,
            oximeter_available=False,
            oxygen_saturation='<90%')

        # Check that IMCI classification if oximter available but no low oxygen saturation (should be same as symptoms)
        assert correct_classification_on_symptoms == final_classification_hw_and_oximeter(
            hw_assigned_classification=correct_classification_on_symptoms,
            age_exact_years=0.5,
            oximeter_available=True,
            oxygen_saturation='>=93%')

        # Check that IMCI classification if oximter available and low oxygen saturation (should be'danger_signs_
        # pneumonia' irrespective of symptoms)
        assert 'danger_signs_pneumonia' == final_classification_hw_and_oximeter(
            hw_assigned_classification=correct_classification_on_symptoms,
            age_exact_years=0.5,
            oximeter_available=True,
            oxygen_saturation='<90%')


def test_do_effects_of_alri_treatment(sim_hs_all_consumables):
    """Check that running `do_alri_treatment` with the appropriate treatment code causes there to be a CureEvent
    scheduled and prevents deaths."""
    sim = sim_hs_all_consumables

    popsize = 100
    sim.make_initial_population(n=popsize)

    # Set the treatment failure parameters to null
    params = sim.modules['Alri'].parameters
    params['5day_amoxicillin_treatment_failure_by_day6'] = 0.0
    params['5day_amoxicillin_relapse_by_day14'] = 0.0
    params['1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2'] = 0.0

    # start simulation
    sim.simulate(end_date=start_date)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Make the incident case one that should cause treatment 'IMCI_Treatment_severe_pneumonia' to need to be provided
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

    # give a classification
    df.at[person_id, 'ri_symptom_based_pneumonia_classification'] = 'danger_signs_pneumonia'

    # Run the 'do_alri_treatment' function (as if from the HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2)
    sim.modules['Alri'].do_effects_of_alri_treatment(person_id=person_id,
                                                     hsi_event=HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(
                                                         person_id=person_id, module=sim.modules['Alri']),
                                                     antibiotic='1st_line_IV_antibiotics',
                                                     oxygen=True)

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


def test_severe_pneumonia_referral_from_HSI_GenericFirstApptAtFacilityLevel0(sim_hs_all_consumables):
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
    assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    # Run the HSI event
    hsi = HSI_GenericFirstApptAtFacilityLevel0(person_id=int(person_id), module=sim.modules['HealthSeekingBehaviour'])
    hsi.run(squeeze_factor=0.0)

    # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Inpatient_level_1b` scheduled
    event_in_que = [event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Inpatient_level_1b)]
    assert len(event_in_que)

    # Check not on treatment before referral:
    assert not df.at[person_id, 'ri_on_treatment']
    assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    # run the HSI ...
    referral_event = event_in_que[0]
    referral_event.run(squeeze_factor=0.0)

    # Check that the person is now on treatment
    assert df.at[person_id, 'ri_on_treatment']
    assert pd.notnull(df.at[person_id, 'ri_ALRI_tx_start_date'])


def test_HSI_GenericFirstApptAtFacilityLevel0_and_referral_to_level1a(sim_hs_all_consumables, sim_hs_no_consumables):
    """ Check that someone with uncomplicated pneumonia can be treated at level 0 if all consumables available,
    or can be referred to a level 1a outpatient appointment.
    """

    for simulation in [sim_hs_all_consumables, sim_hs_no_consumables]:
        sim = simulation

        popsize = 100
        sim.make_initial_population(n=popsize)

        # make probability of mild symptoms very high, and severe symptoms low
        params = sim.modules['Alri'].parameters
        mild_symptoms = {
            'cough', 'difficult_breathing', 'tachypoea'
        }
        severe_symptoms = {
            'chest_indrawing', 'danger_signs'
        }
        for p in params:
            if any([p.startswith(f"prob_{symptom}") for symptom in mild_symptoms]):
                if isinstance(params[p], float):
                    params[p] = 1.0
                else:
                    params[p] = [1.0] * len(params[p])
            if any([p.startswith(f"prob_{symptom}") for symptom in severe_symptoms]):
                params[p] = 0.0

        # Assume perfect diagnosis by the health worker
        params = sim.modules['Alri'].parameters
        params['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

        # start simulation
        sim.simulate(end_date=start_date)
        sim.event_queue.queue = []  # clear the queue
        sim.modules['HealthSystem'].reset_queue()

        # Get person to use (under 5 years old and not infected:
        df = sim.population.props
        under5s = df.loc[df.is_alive
                         & ~df['ri_current_infection_status']
                         & (df['age_years'] < 5)]
        person_id = under5s.index[0]

        # Give this person uncomplicated pneumonia:
        pathogen = list(sim.modules['Alri'].all_pathogens)[0]
        incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
        incidentcase.run()

        # Check infected and not on treatment:
        assert not df.at[person_id, 'ri_on_treatment']
        assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

        # Run the HSI event
        hsi = HSI_GenericFirstApptAtFacilityLevel0(person_id=int(person_id),
                                                   module=sim.modules['HealthSeekingBehaviour'])
        hsi.run(squeeze_factor=0.0)

    if sim == sim_hs_all_consumables:
        # Check that the person is now on treatment (the HSI at level 0 gives the treatment directly)
        assert df.at[person_id, 'ri_on_treatment']
        assert pd.notnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    if sim == sim_hs_no_consumables:
        # No consumables, keep referring up
        # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a` scheduled
        assert len([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a)])

        # Run the HSI event
        hsi = HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a(person_id=int(person_id), module=sim.modules['Alri'])
        hsi.run(squeeze_factor=0.0)

        # Check not on treatment before referral:
        assert not df.at[person_id, 'ri_on_treatment']
        assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

        # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b` scheduled
        assert len([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b)])

        # Run the HSI event
        hsi = HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b(person_id=int(person_id), module=sim.modules['Alri'])
        hsi.run(squeeze_factor=0.0)

        # Check not on treatment before referral:
        assert not df.at[person_id, 'ri_on_treatment']
        assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

        print(sim.modules['HealthSystem'].find_events_for_person(person_id))

        # No consumables, keep referring up
        # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2` scheduled
        assert len([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2)])


def test_HSI_GenericEmergencyFirstApptAtFacilityLevel1_and_referral_to_level2(sim_hs_all_consumables,
                                                                              sim_hs_no_consumables):
    """ Check that someone with severe pneumonia can be treated at level 1b if all consumables available,
    or can be referred to a level 2 in-patient appointment.
    An emergency appointment for someone with `severe_pneumonia` when there are no constrains on consumables,
    should get treatment on the present appointment.
    An emergency appointment for someone with `severe_pneumonia` when there are no consumables, should lead to an
    in-patient appointment."""

    for simulation in [sim_hs_all_consumables, sim_hs_no_consumables]:
        sim = simulation

        popsize = 100
        sim.make_initial_population(n=popsize)

        # Assume perfect diagnosis by the health worker
        params = sim.modules['Alri'].parameters
        params['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
        params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
        params['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

        # start simulation
        sim.simulate(end_date=start_date)
        sim.event_queue.queue = []  # clear the queue
        sim.modules['HealthSystem'].reset_queue()

        # Get person to use (under 5 years old and not infected):
        df = sim.population.props
        under5s = df.loc[df.is_alive
                         & ~df['ri_current_infection_status']
                         & (df['age_years'] < 5)]
        person_id = under5s.index[0]

        # Give this person severe pneumonia:
        pathogen = list(sim.modules['Alri'].all_pathogens)[0]
        incidentcase = AlriIncidentCase_Lethal_Severe_Pneumonia(person_id=person_id, pathogen=pathogen,
                                                                module=sim.modules['Alri'])
        incidentcase.run()

        # Check infected and not on treatment:
        assert not df.at[person_id, 'ri_on_treatment']
        assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

        # Run the HSI event
        hsi = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=int(person_id),
                                                            module=sim.modules['HealthSeekingBehaviour'])
        hsi.run(squeeze_factor=0.0)

    if sim == sim_hs_all_consumables:
        # Check that the person is now on treatment (the HSI at level 1 gives the treatment directly)
        assert df.at[person_id, 'ri_on_treatment']
        assert pd.notnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    if sim == sim_hs_no_consumables:
        # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2` scheduled
        assert len([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2)])


def test_default(sim_hs_default_consumables):
    """ Check that someone with uncomplicated pneumonia can be treated at level 0 if all consumables available,
    or can be referred to a level 1a outpatient appointment.
    """

    sim = sim_hs_default_consumables

    popsize = 5000
    sim.make_initial_population(n=popsize)

    # Assume perfect diagnosis by the health worker
    params = sim.modules['Alri'].parameters
    params['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
    params['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
    params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
    params['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
    params['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
    params['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

    # start simulation
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
    incidentcase = AlriIncidentCase_Lethal_Severe_Pneumonia(person_id=person_id,
                                                            pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.run()

    # Check infected and not on treatment:
    assert not df.at[person_id, 'ri_on_treatment']
    assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    # Run the HSI event
    hsi = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=int(person_id),
                                                        module=sim.modules['HealthSeekingBehaviour'])
    hsi.run(squeeze_factor=0.0)

    # if treatment not available, referral to level 2
    if not df.at[person_id, 'ri_on_treatment'] and pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date']):

        # Check that there is an `HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2` scheduled
        assert len([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)
                    if isinstance(event_tuple[1], HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2)])

        # Check not on treatment:
        assert not df.at[person_id, 'ri_on_treatment']
        assert pd.isnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

        # Run the HSI event
        hsi = HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=int(person_id), module=sim.modules['Alri'])
        hsi.run(squeeze_factor=0.0)

        # Check on treatment:
        assert df.at[person_id, 'ri_on_treatment']
        assert pd.notnull(df.at[person_id, 'ri_ALRI_tx_start_date'])

    # if the person is now on treatment then no referrals is needed
    elif df.at[person_id, 'ri_on_treatment'] and pd.notnull(df.at[person_id, 'ri_ALRI_tx_start_date']):
        print([event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)])
        # Check that there are no referrals if treated
        assert len(
            [event_tuple[1] for event_tuple in sim.modules['HealthSystem'].find_events_for_person(person_id)]) == 0
