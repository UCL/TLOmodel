"""Test file for the Alri module"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel
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
    AlriDelayedOnsetComplication,
    AlriIncidentCase,
    AlriNaturalRecoveryEvent,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_GenericTreatment,
    Models,
)

# Path to the resource files used by the disease and intervention methods

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)


def get_sim(tmpdir):
    """Return simulation objection with Alri and other necessary modules registered."""
    sim = Simulation(start_date=start_date, seed=0, show_progress_bar=False, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.INFO}
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=0, do_checks=True),
        AlriPropertiesOfOtherModules()
    )
    return sim


def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_integrity_of_linear_models(tmpdir):
    """Run the models to make sure that is specified correctly and can run."""
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=5000)
    alri = sim.modules['Alri']
    df = sim.population.props
    person_id = 0

    # make the models
    models = Models(alri)

    # --- compute_risk_of_acquisition & incidence_equations_by_pathogen:
    # if no vaccine and very high risk:
    # make risk of vaccine high:
    for patho in alri.all_pathogens:
        models.p[f'base_inc_rate_ALRI_by_{patho}'] = [0.5] * len(models.p[f'base_inc_rate_ALRI_by_{patho}'])
    models.make_model_for_acquisition_risk()

    # ensure no one has the relevant vaccines:
    df['va_pneumo_all_doses'] = False
    df['va_hib_all_doses'] = False

    for pathogen in alri.all_pathogens:
        res = models.compute_risk_of_aquisition(
            pathogen=pathogen,
            df=df.loc[df.is_alive & (df.age_years < 5)]
        )
        assert (res > 0).all() and (res <= 1.0).all()
        assert not res.isna().any()
        assert 'float64' == res.dtype.name

    # pneumococcal vaccine: if efficacy of vaccine is perfect, whomever has vaccine should have no risk of infection
    # from Strep_pneumoniae_PCV13
    models.p['rr_infection_strep_with_pneumococcal_vaccine'] = 0.0
    df['va_pneumo_all_doses'] = True
    assert (0.0 == models.compute_risk_of_aquisition(
        pathogen='Strep_pneumoniae_PCV13',
        df=df.loc[df.is_alive & (df.age_years < 5)])
            ).all()

    # pneumococcal vaccine: if efficacy of vaccine is perfect, whomever has vaccine should have no risk of infection
    # from Strep_pneumoniae_PCV13
    models.p['rr_infection_hib_haemophilus_vaccine'] = 0.0
    df['va_hib_all_doses'] = True
    assert (0.0 == models.compute_risk_of_aquisition(
        pathogen='Hib',
        df=df.loc[df.is_alive & (df.age_years < 5)])
            ).all()

    # --- determine_disease_type
    # set efficacy of pneumococcal vaccine to be 100% (i.e. 0 relative risk of infection)
    models.p['rr_infection_strep_with_pneumococcal_vaccine'] = 0.0
    for patho in alri.all_pathogens:
        for age in range(0, 100):
            for va_pneumo_all_doses in [True, False]:
                disease_type, bacterial_coinfection = \
                    models.determine_disease_type_and_secondary_bacterial_coinfection(
                        age=age,
                        pathogen=patho,
                        va_pneumo_all_doses=va_pneumo_all_doses
                    )

                assert disease_type in alri.disease_types

                if patho in alri.pathogens['bacterial']:
                    assert pd.isnull(bacterial_coinfection)
                elif patho in alri.pathogens['fungal']:
                    assert pd.isnull(bacterial_coinfection)
                else:
                    # viral primary infection- may have a bacterial coinfection or may not:
                    assert pd.isnull(bacterial_coinfection) or bacterial_coinfection in alri.pathogens['bacterial']
                    # check that if has had pneumococcal vaccine they are not coinfected with `Strep_pneumoniae_PCV13`
                    if va_pneumo_all_doses:
                        assert bacterial_coinfection != 'Strep_pneumoniae_PCV13'

    # --- complications
    for patho in alri.all_pathogens:
        for coinf in (alri.pathogens['bacterial'] + [np.nan]):
            for disease_type in alri.disease_types:
                df.loc[person_id, [
                    'ri_primary_pathogen',
                    'ri_secondary_bacterial_pathogen',
                    'ri_disease_type']
                ] = (
                    patho,
                    coinf,
                    disease_type
                )
                res = models.complications(person_id)

                assert isinstance(res, set)
                assert all([c in alri.complications for c in res])

    # --- delayed_complications
    for ri_complication_sepsis in [True, False]:
        for ri_complication_pneumothorax in [True, False]:
            for ri_complication_respiratory_failure in [True, False]:
                for ri_complication_lung_abscess in [True, False]:
                    for ri_complication_empyema in [True, False]:
                        df.loc[person_id, [
                            'ri_complication_sepsis',
                            'ri_complication_pneumothorax',
                            'ri_complication_respiratory_failure',
                            'ri_complication_lung_abscess',
                            'ri_complication_empyema']
                        ] = (
                            ri_complication_sepsis,
                            ri_complication_pneumothorax,
                            ri_complication_respiratory_failure,
                            ri_complication_lung_abscess,
                            ri_complication_empyema
                        )
                        res = models.delayed_complications(person_id=person_id)
                        assert isinstance(res, set)
                        assert all([c in ['sepsis', 'respiratory_failure'] for c in res])

    # --- symptoms_for_disease
    for disease_type in alri.disease_types:
        res = models.symptoms_for_disease(disease_type)
        assert isinstance(res, set)
        assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])

    # --- symptoms_for_complication
    for complication in alri.complications:
        res = models.symptoms_for_complication(complication)
        assert isinstance(res, set)
        assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])

    # --- death
    for disease_type in alri.disease_types:
        df.loc[person_id, [
            'ri_disease_type',
            'age_years',
            'ri_complication_sepsis',
            'ri_complication_respiratory_failure',
            'ri_complication_meningitis',
            'hv_inf',
            'un_clinical_acute_malnutrition',
            'nb_low_birth_weight_status']
        ] = (
            disease_type,
            0,
            False,
            False,
            False,
            True,
            'SAM',
            'low_birth_weight'
        )
        res = models.death(person_id)
        assert isinstance(res, bool)


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
    """Check polling events leads to incident cases"""
    # get simulation object:
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # Make incidence of alri very high :
    params = sim.modules['Alri'].parameters
    for p in params:
        if p.startswith('base_inc_rate_ALRI_by_'):
            params[p] = [3 * v for v in params[p]]

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

    # make probability of death 0% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return False
    sim.modules['Alri'].models.death = death

    # make probability of symptoms very high
    params = sim.modules['Alri'].parameters
    all_symptoms = {
        'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'chest_pain', 'cyanosis',
        'respiratory_distress', 'danger_signs'
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
    incidentcase.apply(person_id=person_id)

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
    recov_event.apply(person_id=person_id)

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


def test_nat_hist_death(tmpdir):
    """Check: Infection onset --> death"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return True
    sim.modules['Alri'].models.death = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

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
    death_event.apply(person_id=person_id)

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

    # make probability of death 0% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return False
    sim.modules['Alri'].models.death = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

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
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)
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
    the person dying."""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return True
    sim.modules['Alri'].models.death = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

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


def test_immediate_onset_complications(tmpdir):
    """Check that if probability of immediately onsetting complications is 100%, then a person has all those
    complications immediately onset"""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # make risk of immediate onset complications be 100% (so that person has all the complications)
    params = sim.modules['Alri'].parameters
    for p in params:
        if any([p.startswith(f'prob_{c}') for c in sim.modules['Alri'].complications]):
            params[p] = 1.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case for a viral pathogen:
    pathogen = 'other_viral_pathogens'
    incidentcase = AlriIncidentCase(module=sim.modules['Alri'], person_id=person_id, pathogen=pathogen)
    incidentcase.apply(person_id=person_id)

    # Check has some complications ['pneumothorax', 'pleural_effusion', 'sepsis', 'hypoxia'] are present for all
    #  disease causes by viruses
    complications_cols = [
        f"ri_complication_{complication}" for complication in ['pneumothorax', 'pleural_effusion', 'sepsis', 'hypoxia']
    ]
    assert df.loc[person_id, complications_cols].all()


def test_no_immediate_onset_complications(tmpdir):
    """Check that if probability of immediately onsetting complications is 0%, then a person has none of those
    complications immediately onset
    """
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # make risk of immediate-onset complications be 0%
    params = sim.modules['Alri'].parameters
    for p in params:
        if any([p.startswith(f'prob_{c}') for c in sim.modules['Alri'].complications]):
            params[p] = 0.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(module=sim.modules['Alri'], person_id=person_id, pathogen=pathogen)
    incidentcase.apply(person_id=person_id)

    # Check has no complications following onset (check #1)
    complications_cols = [f"ri_complication_{complication}" for complication in sim.modules['Alri'].complications]
    assert not df.loc[person_id, complications_cols].any()


def test_delayed_onset_complications(tmpdir):
    """Check that if the probability of each delayed onset complications is 100%, then a person will have all of those
    complications onset with a delay.
    """
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # make risk of immediate-onset complications be 100% for those complications that are required for onset of
    # other delayed complications (and 0% for immediate onset of the complications that will be delayed onset)
    params = sim.modules['Alri'].parameters
    for p in params:
        if any([p.startswith(f'prob_{c}') for c in {'pneumothorax', 'lung_abscess', 'empyema'}]):
            params[p] = 1.0
        elif any([p.startswith(f'prob_{c}') for c in {'respiratory_failure', 'sepsis'}]):
            params[p] = 0.0

    # make risk of delayed-onset complications be 100%
    params['prob_pneumothorax_to_respiratory_failure'] = 1.0
    params['prob_lung_abscess_to_sepsis'] = 1.0
    params['prob_empyema_to_sepsis'] = 1.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = 'other_Strepto_Enterococci'
    # <-- a bacterial infection can lead to delayed symptoms of sepsis and respiratory_failure
    incidentcase = AlriIncidentCase(module=sim.modules['Alri'], person_id=person_id, pathogen=pathogen)
    incidentcase.apply(person_id=person_id)

    # Check has does not have the complications that will be delayed onset
    person = df.loc[person_id]
    assert not person['ri_complication_respiratory_failure']
    assert not person['ri_complication_sepsis']

    # Check has delayed onset complication events scheduled (total of two) and at an appropriate time
    delayed_onset_event_tuples = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                  isinstance(event_tuple[1], AlriDelayedOnsetComplication)]
    assert 2 == len(delayed_onset_event_tuples)
    dates_of_delayedonset = [event_tuple[0] for event_tuple in delayed_onset_event_tuples]
    assert all([d == dates_of_delayedonset[0] for d in dates_of_delayedonset])

    date_of_outcome = person[['ri_scheduled_recovery_date', 'ri_scheduled_death_date']][
        ~person[['ri_scheduled_recovery_date', 'ri_scheduled_death_date']].isna()].values[0]
    assert sim.date <= dates_of_delayedonset[0] <= date_of_outcome

    # run the delayed onset event
    sim.date = dates_of_delayedonset[0]
    for event_tuple in delayed_onset_event_tuples:
        event_tuple[1].apply(person_id=person_id)

    # check that they now have delayed onset complications (respiratory_failure and sepsis)
    person = df.loc[person_id]
    assert person['ri_complication_respiratory_failure']
    assert person['ri_complication_sepsis']


def test_treatment(tmpdir):
    """Test that providing a treatment prevent death and causes there to be a CureEvent Scheduled"""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100% (not using a lambda function because code uses the keyword argument for clarity)
    def death(person_id):
        return True
    sim.modules['Alri'].models.death = death

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
    incidentcase = AlriIncidentCase(person_id=person_id, pathogen=pathogen, module=sim.modules['Alri'])
    incidentcase.apply(person_id=person_id)

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

    # Run the 'do_treatment' function
    sim.modules['Alri'].do_treatment(person_id=person_id, prob_of_cure=1.0)

    # Run the death event that was originally scheduled) - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['ri_current_infection_status']

    # Check that a CureEvent has been scheduled
    cure_event = [event_tuple[1] for event_tuple in sim.find_events_for_person(person_id) if
                  isinstance(event_tuple[1], AlriCureEvent)][0]

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


def test_use_of_HSI(tmpdir):
    """Check that the HSI template works"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # make probability of death 100%
    sim.modules['Alri'].p_death = LinearModel.multiplicative()

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert not df.loc[person_id, 'ri_current_infection_status']

    # Run incident case:
    pathogen = list(sim.modules['Alri'].all_pathogens)[0]
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
