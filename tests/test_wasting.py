"""
Basic tests for the Wasting Module
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
    wasting,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    contraception,
    labour,
    pregnancy_supervisor,
    newborn_outcomes,
    care_of_women_during_pregnancy,
    postnatal_supervisor


)

from tlo.methods.healthsystem import HSI_Event

from tlo.methods.wasting import (
    AcuteMalnutritionDeathPollingEvent,
    SevereAcuteMalnutritionDeathEvent,
    ProgressionSevereWastingEvent,
    WastingNaturalRecoveryEvent,
    ClinicalAcuteMalnutritionRecoveryEvent,
    WastingPollingEvent,
    UpdateToMAM,
    HSI_supplementary_feeding_programme_for_MAM,
    HSI_outpatient_therapeutic_programme_for_SAM,
    HSI_inpatient_care_for_complicated_SAM
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
            "tlo.methods.wasting": logging.INFO}
    })

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )
    return sim


def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# def test_integrity_of_linear_models(tmpdir):
#     """Run the models to make sure that is specified correctly and can run."""
#     sim = get_sim(tmpdir)
#     sim.make_initial_population(n=5000)
#     alri = sim.modules['Alri']
#     df = sim.population.props
#     person_id = 0
#
#     # make the models
#     models = Models(alri)
#
#     # --- compute_risk_of_acquisition & incidence_equations_by_pathogen:
#     # if no vaccine and very high risk:
#     # make risk of vaccine high:
#     for patho in alri.all_pathogens:
#         models.p[f'base_inc_rate_ALRI_by_{patho}'] = [0.5] * len(models.p[f'base_inc_rate_ALRI_by_{patho}'])
#     models.make_model_for_acquisition_risk()
#
#     # ensure no one has the relevant vaccines:
#     df['va_pneumo_all_doses'] = False
#     df['va_hib_all_doses'] = False
#
#     for pathogen in alri.all_pathogens:
#         res = models.compute_risk_of_aquisition(
#             pathogen=pathogen,
#             df=df.loc[df.is_alive & (df.age_years < 5)]
#         )
#         assert (res > 0).all() and (res <= 1.0).all()
#         assert not res.isna().any()
#         assert 'float64' == res.dtype.name
#
#     # pneumococcal vaccine: if efficacy of vaccine is perfect, whomever has vaccine should have no risk of infection
#     # from Strep_pneumoniae_PCV13
#     models.p['rr_infection_strep_with_pneumococcal_vaccine'] = 0.0
#     df['va_pneumo_all_doses'] = True
#     assert (0.0 == models.compute_risk_of_aquisition(
#         pathogen='Strep_pneumoniae_PCV13',
#         df=df.loc[df.is_alive & (df.age_years < 5)])
#             ).all()
#
#     # pneumococcal vaccine: if efficacy of vaccine is perfect, whomever has vaccine should have no risk of infection
#     # from Strep_pneumoniae_PCV13
#     models.p['rr_infection_hib_haemophilus_vaccine'] = 0.0
#     df['va_hib_all_doses'] = True
#     assert (0.0 == models.compute_risk_of_aquisition(
#         pathogen='Hib',
#         df=df.loc[df.is_alive & (df.age_years < 5)])
#             ).all()
#
#     # --- determine_disease_type
#     # set efficacy of pneumococcal vaccine to be 100% (i.e. 0 relative risk of infection)
#     models.p['rr_infection_strep_with_pneumococcal_vaccine'] = 0.0
#     for patho in alri.all_pathogens:
#         for age in range(0, 100):
#             for va_pneumo_all_doses in [True, False]:
#                 disease_type, bacterial_coinfection = \
#                     models.determine_disease_type_and_secondary_bacterial_coinfection(
#                         age=age,
#                         pathogen=patho,
#                         va_pneumo_all_doses=va_pneumo_all_doses
#                     )
#
#                 assert disease_type in alri.disease_types
#
#                 if patho in alri.pathogens['bacterial']:
#                     assert pd.isnull(bacterial_coinfection)
#                 elif patho in alri.pathogens['fungal']:
#                     assert pd.isnull(bacterial_coinfection)
#                 else:
#                     # viral primary infection- may have a bacterial coinfection or may not:
#                     assert pd.isnull(bacterial_coinfection) or bacterial_coinfection in alri.pathogens['bacterial']
#                     # check that if has had pneumococcal vaccine they are not coinfected with `Strep_pneumoniae_PCV13`
#                     if va_pneumo_all_doses:
#                         assert bacterial_coinfection != 'Strep_pneumoniae_PCV13'
#
#     # --- complications
#     for patho in alri.all_pathogens:
#         for coinf in (alri.pathogens['bacterial'] + [np.nan]):
#             for disease_type in alri.disease_types:
#                 df.loc[person_id, [
#                     'ri_primary_pathogen',
#                     'ri_secondary_bacterial_pathogen',
#                     'ri_disease_type']
#                 ] = (
#                     patho,
#                     coinf,
#                     disease_type
#                 )
#                 res = models.complications(person_id)
#
#                 assert isinstance(res, set)
#                 assert all([c in alri.complications for c in res])
#
#     # --- delayed_complications
#     for ri_complication_sepsis in [True, False]:
#         for ri_complication_pneumothorax in [True, False]:
#             for ri_complication_respiratory_failure in [True, False]:
#                 for ri_complication_lung_abscess in [True, False]:
#                     for ri_complication_empyema in [True, False]:
#                         df.loc[person_id, [
#                             'ri_complication_sepsis',
#                             'ri_complication_pneumothorax',
#                             'ri_complication_respiratory_failure',
#                             'ri_complication_lung_abscess',
#                             'ri_complication_empyema']
#                         ] = (
#                             ri_complication_sepsis,
#                             ri_complication_pneumothorax,
#                             ri_complication_respiratory_failure,
#                             ri_complication_lung_abscess,
#                             ri_complication_empyema
#                         )
#                         res = models.delayed_complications(person_id=person_id)
#                         assert isinstance(res, set)
#                         assert all([c in ['sepsis', 'respiratory_failure'] for c in res])
#
#     # --- symptoms_for_disease
#     for disease_type in alri.disease_types:
#         res = models.symptoms_for_disease(disease_type)
#         assert isinstance(res, set)
#         assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])
#
#     # --- symptoms_for_complication
#     for complication in alri.complications:
#         res = models.symptoms_for_complication(complication)
#         assert isinstance(res, set)
#         assert all([s in sim.modules['SymptomManager'].symptom_names for s in res])
#
#     # --- death
#     for disease_type in alri.disease_types:
#         df.loc[person_id, [
#             'ri_disease_type',
#             'age_years',
#             'ri_complication_sepsis',
#             'ri_complication_respiratory_failure',
#             'ri_complication_meningitis',
#             'hv_inf',
#             'un_clinical_acute_malnutrition',
#             'nb_low_birth_weight_status']
#         ] = (
#             disease_type,
#             0,
#             False,
#             False,
#             False,
#             True,
#             'SAM',
#             'low_birth_weight'
#         )
#         res = models.death(person_id)
#         assert isinstance(res, bool)


# def check_configuration_of_properties(sim):
#     # check that the properties are ok:
#
#     df = sim.population.props
#
#     # Those that were never wasted, should have normal WHZ score:
#     assert (df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna(), 'un_WHZ_category'] == 'WHZ>=-2').all().all()
#
#     # Those that were never wasted and not clinically malnourished,
#     # should have not_applicable/null values for all the other properties:
#     # assert pd.isnull(df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna() &
#     #                         (df.un_clinical_acute_malnutrition == 'well'),
#     #                         ['un_last_wasting_date_of_onset',
#     #                          'un_sam_death_date',
#     #                          'un_am_recovery_date',
#     #                          'un_am_discharge_date',
#     #                          'un_acute_malnutrition_tx_start_date']
#     #                         ]).all().all()
#
#     # Those that were ever wasted, should have a WHZ socre below <-2
#     # assert (df.loc[df.un_ever_wasted, 'un_WHZ_category'] != 'WHZ>=-2').all().all()
#
#     # Those that had wasting and no treatment, should have either a recovery date or a death_date
#     # (but not both)
#     has_recovery_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
#                                           'un_am_recovery_date'])
#     has_death_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
#                                        'un_sam_death_date'])
#
#     has_recovery_date_or_death_date = has_recovery_date | has_death_date
#     has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
#     # assert has_recovery_date_or_death_date.all()
#     assert not has_both_recovery_date_and_death_date.any()
#
#     # Those for whom the death date has past should be dead
#     assert not df.loc[df.un_ever_wasted & (df['un_sam_death_date'] < sim.date), 'is_alive'].any()
#     assert not df.loc[(df.un_clinical_acute_malnutrition == 'SAM') & (
#         df['un_sam_death_date'] < sim.date), 'is_alive'].any()
#
#     # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
#     #  but not others (among those who are alive)
#     has_symptoms_of_wasting = set(sim.modules['SymptomManager'].who_has('weight_loss'))
#     has_symptoms = set([p for p in has_symptoms_of_wasting if
#                         'Wasting' in sim.modules['SymptomManager'].causes_of(p, 'weight_loss')
#                         ])
#
#     in_current_episode_before_recovery = \
#         df.is_alive & \
#         df.un_ever_wasted & \
#         (df.un_last_wasting_date_of_onset <= sim.date) & \
#         (sim.date <= df.un_am_recovery_date)
#     set_of_person_id_in_current_episode_before_recovery = set(
#         in_current_episode_before_recovery[in_current_episode_before_recovery].index
#     )
#
#     in_current_episode_before_death = \
#         df.is_alive & \
#         df.un_ever_wasted & \
#         (df.un_last_wasting_date_of_onset <= sim.date) & \
#         (sim.date <= df.un_sam_death_date)
#     set_of_person_id_in_current_episode_before_death = set(
#         in_current_episode_before_death[in_current_episode_before_death].index
#     )
#
#     in_current_episode_before_cure = \
#         df.is_alive & \
#         df.un_ever_wasted & \
#         (df.un_last_wasting_date_of_onset <= sim.date) & \
#         (df.un_acute_malnutrition_tx_start_date <= sim.date) & \
#         pd.isnull(df.un_am_recovery_date) & \
#         pd.isnull(df.un_sam_death_date)
#     set_of_person_id_in_current_episode_before_cure = set(
#         in_current_episode_before_cure[in_current_episode_before_cure].index
#     )
#
#     assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
#         set_of_person_id_in_current_episode_before_death
#     )
#
#     set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
#         set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
#     )
#     assert set_of_person_id_in_current_episode == has_symptoms


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(months=3)
    popsize = 100
    sim = get_sim(tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    check_dtypes(sim)


# def test_basic_run_lasting_two_years(tmpdir):
#     """Check logging results in a run of the model for two years, with daily property config checking"""
#     dur = pd.DateOffset(years=2)
#     popsize = 500
#     sim = get_sim(tmpdir)
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=start_date + dur)
#
#     # Read the log for the population counts of incidence:
#     log_counts = parse_log_file(sim.log_filepath)['tlo.methods.wasting']['event_counts']
#     assert 0 < log_counts['incident_cases'].sum()
#     assert 0 < log_counts['recovered_cases'].sum()
#     assert 0 < log_counts['deaths'].sum()
#     assert 0 == log_counts['cured_cases'].sum()
#
#     # Read the log for the one individual being tracked:
#     log_one_person = parse_log_file(sim.log_filepath)['tlo.methods.alri']['log_individual']
#     log_one_person['date'] = pd.to_datetime(log_one_person['date'])
#     log_one_person = log_one_person.set_index('date')
#     assert log_one_person.index.equals(pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1)))
#     assert set(log_one_person.columns) == set(sim.modules['Alri'].PROPERTIES.keys())


def test_wasting_polling(tmpdir):
    """Check polling events leads to incident cases"""
    # get simulation object:
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    # Make incidence of alri very high :
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
    assert len([q for q in sim.event_queue.queue if isinstance(q[2], ProgressionSevereWastingEvent)]) > 0


def test_nat_hist_recovery(tmpdir):
    """Check: Infection onset --> recovery"""
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0)
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    params = sim.modules['Wasting'].parameters

    # increase incidence of wasting
    params['base_inc_rate_wasting_by_agegp'] = [1.0 == v for v in params['base_inc_rate_wasting_by_agegp']]

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
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that they have some symptoms caused by Wasting
    assert 0 < len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Wasting']))

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

    # Check properties of this individual: (should now not be infected)
    person = df.loc[person_id]
    assert not person['un_WHZ_category'] == 'WHZ>=-2'
    assert pd.isnull(person['un_last_wasting_date_of_onset'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # check they they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id, sim.modules['Wasting']))

    # # check it's logged (one infection + one recovery)
    # assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    # assert 1 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_death(tmpdir):
    """Check: Wasting onset --> death"""
    """ Check if the risk of death is 100% does everyone with SAM die? """
    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    params = sim.modules['Wasting'].parameters

    # increase incidence of wasting
    params['base_inc_rate_wasting_by_agegp'] = [1.0 == v for v in params['base_inc_rate_wasting_by_agegp']]
    # increase progression to severe of wasting
    params['progression_severe_wasting_by_agegp'] = [1.0 == v for v in params['progression_severe_wasting_by_agegp']]

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

    # # check it's logged (one infection + one death)
    # assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    # assert 1 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_cure_if_recovery_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to recover naturally, it cause the episode to
    end earlier."""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 0% death rate by replacing with empty linear model 0.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 0.0)

    # decrease progression to severe of wasting - so all children stay in moderate wasting state only
    params = sim.modules['Wasting'].parameters
    params['progression_severe_wasting_by_agegp'] = [0.0 for v in params['progression_severe_wasting_by_agegp']]

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

    # # check it's logged (one infection + one cure)
    # assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    # assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()


def test_nat_hist_cure_if_death_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to die, it cause the episode to end without
    the person dying."""

    dur = pd.DateOffset(days=0)
    popsize = 100
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Make 100% death rate by replacing with empty linear model 1.0
    sim.modules['Wasting'].sam_death_equation = LinearModel(
        LinearModelType.MULTIPLICATIVE, 1.0)

    # increase incidence of wasting and progression to severe
    params = sim.modules['Wasting'].parameters
    params['base_inc_rate_wasting_by_agegp'] = [5 * v for v in params['base_inc_rate_wasting_by_agegp']]
    params['progression_severe_wasting_by_agegp'] = [5 * v for v in params['progression_severe_wasting_by_agegp']]
    # increase parameters in moderate wasting for clinical SAM (MUAC and oedema) to be polled for death
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

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert person['is_alive']

    # check it's logged (one infection + one cure)
    # assert 1 == sim.modules['Alri'].logging_event.trackers['incident_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['recovered_cases'].report_current_total()
    # assert 0 == sim.modules['Alri'].logging_event.trackers['deaths'].report_current_total()
    # assert 1 == sim.modules['Alri'].logging_event.trackers['cured_cases'].report_current_total()
