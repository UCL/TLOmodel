"""
Basic tests for the Diarrhoea Module
"""
import os
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from pandas import DateOffset

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    Metadata,
    demography,
    diarrhoea,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.diarrhoea import (
    DiarrhoeaCureEvent,
    DiarrhoeaIncidentCase,
    DiarrhoeaNaturalRecoveryEvent,
    HSI_Diarrhoea_Treatment_Inpatient,
    HSI_Diarrhoea_Treatment_Outpatient,
    increase_incidence_of_pathogens,
    increase_risk_of_death,
    make_treatment_perfect,
)
from tlo.methods.hsi_generic_first_appts import HSI_GenericNonEmergencyFirstAppt

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_combined_log(log_filepath):
    """Merge the logs for incident_case and end_of_episode to give a record of each incident case that has ended"""
    log = parse_log_file(log_filepath)['tlo.methods.diarrhoea']
    m = log['incident_case'][[
        'person_id',
        'age_years',
        'date',
        'date_of_outcome',
        'will_die'
    ]].merge(log['end_of_case'], left_on=['person_id', 'date'], right_on=['person_id', 'date_of_onset'], how='inner',
             suffixes=['_i', '_o'])
    # <-- merging is on person_id and date_of_onset of episode
    return m


@pytest.mark.slow
def test_basic_run_of_diarrhoea_module_with_default_params(tmpdir, seed):
    """Check that the module run and that properties are maintained correctly, using health system and default
    parameters"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    log_config = {'filename': 'tmpfile',
                  'directory': tmpdir,
                  'custom_levels': {
                      "tlo.methods.diarrhoea": logging.INFO}
                  }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # Get combined log and check its content as are expected
    m = get_combined_log(sim.log_filepath)
    assert (m.loc[m.will_die].outcome == 'death').all()
    assert (m.loc[~m.will_die].outcome == 'recovery').all()
    assert not (m.outcome == 'cure').any()
    assert (m['date_of_outcome'] == m['date_o']).all()


@pytest.mark.slow
def test_basic_run_of_diarrhoea_module_with_zero_incidence(seed):
    """Run with zero incidence and check for no cases or deaths"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # **Zero-out incidence**:
    for param_name in sim.modules['Diarrhoea'].parameters.keys():
        if param_name.startswith('base_inc_rate_diarrhoea_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = \
                [0.0 * v for v in sim.modules['Diarrhoea'].parameters[param_name]]

        # Increase symptoms (to be consistent with other checks):
        if param_name.startswith('proportion_AWD_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('fever_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('vomiting_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0

    # Increase death (to be consistent with other checks):
    increase_risk_of_death(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # Check for zero-level of diarrhoea
    df = sim.population.props
    assert 0 == df.loc[df.is_alive].gi_has_diarrhoea.sum()
    assert pd.isnull(df.loc[df.is_alive, 'gi_date_end_of_last_episode']).all()

    # Check for zero level of death
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith('Diarrhoea').any()


@pytest.mark.slow
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_zero_death(tmpdir, seed):
    """Check that there are incident cases, and that everyone recovers naturally"""

    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 2000

    log_config = {'filename': 'tmpfile',
                  'directory': tmpdir,
                  'custom_levels': {
                      "tlo.methods.diarrhoea": logging.INFO}
                  }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Increase incidence of pathogens:
    increase_incidence_of_pathogens(sim.modules['Diarrhoea'])

    # Make risk of death 0%:
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # Check for non-zero-level of diarrhoea
    df = sim.population.props
    assert pd.notnull(df.loc[df.is_alive, 'gi_date_end_of_last_episode']).any()

    # Check for zero level of death
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

    # Examine the log to check that logged outcomes are consistent with the expectations when case is onset and that
    # everyone recovered.
    m = get_combined_log(sim.log_filepath)
    assert not m.will_die.any()
    assert (m.outcome == 'recovery').all()
    assert (m['date_of_outcome'] == m['date_o']).all()


@pytest.mark.slow
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_no_treatment(tmpdir, seed):
    """Check that there are incident cases, treatments and deaths occurring correctly"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 2000

    log_config = {'filename': 'tmpfile',
                  'directory': tmpdir,
                  'custom_levels': {
                      "tlo.methods.diarrhoea": logging.INFO}
                  }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Increase incidence of pathogens:
    increase_incidence_of_pathogens(sim.modules['Diarrhoea'])

    # Increase death:
    increase_risk_of_death(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # Check for non-zero-level of diarrhoea
    df = sim.population.props
    assert pd.notnull(df.loc[df.is_alive, 'gi_date_end_of_last_episode']).any()

    # Check for non-zero level of death
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

    # Check that those with a gi_last_diarrhoea_death_date in the past, are now dead
    # NB. Cannot guarantee that all will have a cause of death that is Diarrhoea, because OtherDeathPoll can also
    #  cause deaths.
    gi_death_date_in_past = ~pd.isnull(df.gi_scheduled_date_death) & (df.gi_scheduled_date_death < sim.date)
    assert 0 < gi_death_date_in_past.sum()
    assert not df.loc[gi_death_date_in_past, 'is_alive'].any()

    # Examine the log to check that logged outcomes are consistent with the expectations when case is onset
    m = get_combined_log(sim.log_filepath)
    assert (m.loc[m.will_die].outcome == 'death').all()
    assert (m.loc[~m.will_die].outcome == 'recovery').all()
    assert not (m.outcome == 'cure').any()
    assert (m['date_of_outcome'] == m['date_o']).all()


@pytest.mark.slow
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_with_perfect_treatment(tmpdir, seed):
    """Run with high incidence and perfect treatment, with and without spurious symptoms of diarrhoea being generated"""

    def run(spurious_symptoms):
        # Run with everyone getting symptoms and seeking care and perfect treatment efficacy:
        # Check that everyone is cured and no deaths;
        start_date = Date(2010, 1, 1)
        end_date = Date(2010, 12, 31)  # reduce run time because with spurious_symptoms=True, things get slow
        popsize = 1000

        log_config = {'filename': 'tmpfile',
                      'directory': tmpdir,
                      'custom_levels': {
                          "tlo.methods.diarrhoea": logging.INFO}
                      }

        sim = Simulation(start_date=start_date, seed=seed, show_progress_bar=True, log_config=log_config)

        # Register the appropriate modules
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True,
                         cons_availability='all',
                     ),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath,
                                                   spurious_symptoms=spurious_symptoms
                                                   ),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         resourcefilepath=resourcefilepath,
                         force_any_symptom_to_lead_to_healthcareseeking=True
                         # every symptom leads to healthcare seeking
                     ),
                     diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                     diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                     )
        # Edit rate of spurious symptoms to be limited to additional cases of diarrhoea:
        sp_symps = sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence']
        for symp in sp_symps['name']:
            sp_symps.loc[
                sp_symps['name'] == symp,
                ['prob_spurious_occurrence_in_adults_per_day', 'prob_spurious_occurrence_in_children_per_day']
            ] = 5.0 / 1000 if symp == 'diarrhoea' else 0.0

        # Increase incidence of pathogens:
        increase_incidence_of_pathogens(sim.modules['Diarrhoea'])

        # Increase risk of death (and make it depend only on blood-in-diarrhoea and dehydration)
        increase_risk_of_death(sim.modules['Diarrhoea'])

        # Make treatment perfect
        make_treatment_perfect(sim.modules['Diarrhoea'])

        # Make long duration so as to allow time for healthcare seeking
        for pathogen in sim.modules['Diarrhoea'].pathogens:
            sim.modules['Diarrhoea'].parameters[f"prob_prolonged_diarr_{pathogen}"] = 1.0

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=end_date)
        check_dtypes(sim)

        # Check for non-zero-level of diarrhoea
        df = sim.population.props
        assert pd.notnull(df.loc[df.is_alive, 'gi_date_end_of_last_episode']).any()

        # check that there have not been any deaths caused by Diarrhoea
        assert not df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

        # open the logs to check that no one died and that there are many cures
        # (there is not a cure in the instance that the natural recovery happens first).
        m = get_combined_log(sim.log_filepath)
        assert m.loc[~m.will_die].outcome.isin(['recovery', 'cure']).all()
        assert (m.loc[m.will_die].outcome == 'cure').all()
        assert not (m.outcome == 'death').any()

    # # run without spurious symptoms
    run(spurious_symptoms=False)

    # run with spurious symptoms
    run(spurious_symptoms=True)


def test_do_when_presentation_with_diarrhoea_severe_dehydration(seed):
    """Check that when someone presents with diarrhoea and severe dehydration, the correct HSI is created"""

    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Make DxTest for danger signs perfect:
    sim.modules['Diarrhoea'].parameters['sensitivity_severe_dehydration_visual_inspection'] = 1.0
    sim.modules['Diarrhoea'].parameters['specificity_severe_dehydration_visual_inspection'] = 1.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with bloody diarrhoea and severe dehydration
    person_id = 0
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'bloody',
        'gi_dehydration': 'severe',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': pd.NaT,
        'gi_scheduled_date_death': sim.date + DateOffset(days=2),
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    generic_hsi = HSI_GenericNonEmergencyFirstAppt(
        module=sim.modules['HealthSeekingBehaviour'], person_id=person_id)

    # 1) If DxTest of danger signs perfect and 100% chance of referral --> Inpatient HSI should be created
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['Diarrhoea'].parameters['prob_hospitalization_on_danger_signs'] = 1.0
    sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
        person_id=person_id, hsi_event=generic_hsi)
    evs = sim.modules['HealthSystem'].find_events_for_person(person_id)

    assert 1 == len(evs)
    assert isinstance(evs[0][1], HSI_Diarrhoea_Treatment_Inpatient)

    # 2) If DxTest of danger signs perfect but 0% chance of referral --> Inpatient HSI should not be created
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['Diarrhoea'].parameters['prob_hospitalization_on_danger_signs'] = 0.0
    sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
        person_id=person_id, hsi_event=generic_hsi)
    evs = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert 1 == len(evs)
    assert isinstance(evs[0][1], HSI_Diarrhoea_Treatment_Outpatient)


def test_do_when_presentation_with_diarrhoea_severe_dehydration_dxtest_notfunctional(seed):
    """Check that when someone presents with diarrhoea and severe dehydration but the DxTest for danger signs
    is not functional (0% sensitivity, 0% specificity) that an Outpatient appointment is created"""

    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Make DxTest for danger signs not functional:
    sim.modules['Diarrhoea'].parameters['sensitivity_severe_dehydration_visual_inspection'] = 0.0
    sim.modules['Diarrhoea'].parameters['specificity_severe_dehydration_visual_inspection'] = 0.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with bloody diarrhoea and severe dehydration
    person_id = 0
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'bloody',
        'gi_dehydration': 'severe',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': pd.NaT,
        'gi_scheduled_date_death': sim.date + DateOffset(days=2),
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    generic_hsi = HSI_GenericNonEmergencyFirstAppt(
        module=sim.modules['HealthSeekingBehaviour'], person_id=person_id)

    # Only an out-patient appointment should be created as the DxTest for danger signs is not functional.
    sim.modules['Diarrhoea'].parameters['prob_hospitalization_on_danger_signs'] = 0.0
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
        person_id=person_id, hsi_event=generic_hsi)
    evs = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert 1 == len(evs)
    assert isinstance(evs[0][1], HSI_Diarrhoea_Treatment_Outpatient)


def test_do_when_presentation_with_diarrhoea_non_severe_dehydration(seed):
    """Check that when someone presents with diarrhoea and non-severe dehydration, the out-patient HSI is created"""

    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Make DxTest for danger signs perfect:
    sim.modules['Diarrhoea'].parameters['sensitivity_severe_dehydration_visual_inspection'] = 1.0
    sim.modules['Diarrhoea'].parameters['specificity_severe_dehydration_visual_inspection'] = 1.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with bloody diarrhoea and 'some' dehydration
    person_id = 0
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'bloody',
        'gi_dehydration': 'some',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': pd.NaT,
        'gi_scheduled_date_death': sim.date + DateOffset(days=2),
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    generic_hsi = HSI_GenericNonEmergencyFirstAppt(
        module=sim.modules['HealthSeekingBehaviour'], person_id=person_id)

    # 1) Outpatient HSI should be created
    sim.modules['HealthSystem'].reset_queue()
    sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
        person_id=person_id, hsi_event=generic_hsi)
    evs = sim.modules['HealthSystem'].find_events_for_person(person_id)

    assert 1 == len(evs)
    assert isinstance(evs[0][1], HSI_Diarrhoea_Treatment_Outpatient)


def test_run_each_of_the_HSI(seed):
    """Check that HSI specified can be run correctly"""
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # The Out-patient HSI
    hsi_outpatient = HSI_Diarrhoea_Treatment_Outpatient(person_id=0, module=sim.modules['Diarrhoea'])
    hsi_outpatient.run(squeeze_factor=0)

    # The In-patient HSI
    hsi_outpatient = HSI_Diarrhoea_Treatment_Inpatient(person_id=0, module=sim.modules['Diarrhoea'])
    hsi_outpatient.run(squeeze_factor=0)


def test_does_treatment_prevent_death(seed):
    """Check that the helper function 'does_treatment_prevent_death' works as expected."""

    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )

    # Increase incidence and risk of death in Diarrhoea episodes
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    does_treatment_prevent_death = sim.modules['Diarrhoea'].models.does_treatment_prevent_death

    # False if there is no change in characteristics
    assert False is does_treatment_prevent_death(
        pathogen='shigella',
        type=('bloody', 'bloody'),
        duration_longer_than_13days=False,
        dehydration=('severe', 'severe'),
        age_exact_years=2,
        ri_current_infection_status=False,
        untreated_hiv=False,
        un_clinical_acute_malnutrition='SAM'
    )

    # True some of the time if there a improvement in dehydration (severe --> none)
    assert any([does_treatment_prevent_death(
        pathogen='shigella',
        type='watery',
        duration_longer_than_13days=False,
        dehydration=('severe', 'none'),
        age_exact_years=2,
        ri_current_infection_status=False,
        untreated_hiv=False,
        un_clinical_acute_malnutrition='SAM'
    ) for _ in range(1000)])

    # True some of the time if there a improvement in type (watery --> bloody)
    assert any([does_treatment_prevent_death(
        pathogen='shigella',
        type=('bloody', 'watery'),
        duration_longer_than_13days=False,
        dehydration='none',
        age_exact_years=2,
        ri_current_infection_status=False,
        untreated_hiv=False,
        un_clinical_acute_malnutrition='SAM'
    ) for _ in range(1000)])

    # True all of the time if there a improvement in type (watery --> bloody) and dehydration
    assert all([does_treatment_prevent_death(
        pathogen='shigella',
        type=('bloody', 'watery'),
        duration_longer_than_13days=False,
        dehydration=('severe', 'none'),
        age_exact_years=2,
        ri_current_infection_status=False,
        untreated_hiv=False,
        un_clinical_acute_malnutrition='SAM'
    ) for _ in range(1000)])


def test_do_treatment_for_those_that_will_die_if_consumables_available(seed):
    """Check that when someone who will die and is provided with treatment, that the death is prevented."""

    # ** If consumables are available **:
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works
    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with bloody diarrhoea and severe dehydration:
    person_id = 0
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'bloody',
        'gi_dehydration': 'severe',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': pd.NaT,
        'gi_scheduled_date_death': sim.date + DateOffset(days=2),
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        add_or_remove='+',
        disease_module=sim.modules['Diarrhoea']
    )

    # Run 'do_treatment' from an in-patient HSI.
    in_patient_hsi = HSI_Diarrhoea_Treatment_Inpatient(
        module=sim.modules['Diarrhoea'], person_id=person_id)
    sim.modules['Diarrhoea'].do_treatment(person_id=person_id, hsi_event=in_patient_hsi)

    # Check that death is cancelled
    assert pd.isnull(df.at[person_id, 'gi_scheduled_date_death'])

    # Check that cure event is scheduled
    assert any([isinstance(ev[1], DiarrhoeaCureEvent) for ev in sim.find_events_for_person(person_id=person_id)])

    # Check that treatment is recorded to have occurred
    assert pd.notnull(df.at[person_id, 'gi_treatment_date'])


def test_do_treatment_for_those_that_will_die_if_consumables_not_available(seed):
    """Check that when someone who will die and is provided with treatment, but that consumables are not available,
    that the death is not prevented"""

    # ** If consumables are available **:
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works
    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='none'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with bloody diarrhoea and severe dehydration:
    person_id = 0
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'bloody',
        'gi_dehydration': 'severe',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': pd.NaT,
        'gi_scheduled_date_death': sim.date + DateOffset(days=2),
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        add_or_remove='+',
        disease_module=sim.modules['Diarrhoea']
    )

    # Run 'do_treatment' from an in-patient HSI.
    sim.event_queue.queue = []
    in_patient_hsi = HSI_Diarrhoea_Treatment_Inpatient(
        module=sim.modules['Diarrhoea'], person_id=person_id)
    sim.modules['Diarrhoea'].do_treatment(person_id=person_id, hsi_event=in_patient_hsi)

    # Check that death is not cancelled
    assert pd.notnull(df.at[person_id, 'gi_scheduled_date_death'])

    # Check that no cure event is scheduled
    assert not any([isinstance(ev[1], DiarrhoeaCureEvent) for ev in sim.find_events_for_person(person_id=person_id)])

    # Check that treatment is recorded to have occurred
    assert pd.notnull(df.at[person_id, 'gi_treatment_date'])


def test_do_treatment_for_those_that_will_not_die(seed):
    """Check that when someone who will not die and is provided with treatment and gets zinc, that the date of cure is
    brought forward"""

    # ** If consumables are available **:
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works
    sim = Simulation(start_date=start_date, seed=seed)
    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all'
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)
    df = sim.population.props

    # Set that person_id=0 is a child with watery diarrhoea and 'some' dehydration:
    person_id = 0
    scheduled_date_recovery = sim.date + DateOffset(days=10)
    props_new = {
        'age_years': 2,
        'age_exact_years': 2.0,
        'gi_has_diarrhoea': True,
        'gi_pathogen': 'shigella',
        'gi_type': 'watery',
        'gi_dehydration': 'some',
        'gi_duration_longer_than_13days': True,
        'gi_date_of_onset': sim.date,
        'gi_date_end_of_last_episode': sim.date + DateOffset(days=20),
        'gi_scheduled_date_recovery': scheduled_date_recovery,
        'gi_scheduled_date_death': pd.NaT,
        'gi_treatment_date': pd.NaT,
    }
    df.loc[person_id, props_new.keys()] = props_new.values()
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        add_or_remove='+',
        disease_module=sim.modules['Diarrhoea']
    )
    # Run 'do_treatment' from an out-patient HSI.
    in_patient_hsi = HSI_Diarrhoea_Treatment_Outpatient(
        module=sim.modules['Diarrhoea'], person_id=person_id)
    sim.modules['Diarrhoea'].do_treatment(person_id=person_id, hsi_event=in_patient_hsi)

    # check that a Cure Event is scheduled for earlier
    evs = sim.find_events_for_person(person_id)
    assert 1 == len(evs)
    assert isinstance(evs[0][1], DiarrhoeaCureEvent)
    assert evs[0][0] == scheduled_date_recovery - \
           pd.DateOffset(days=sim.modules['Diarrhoea'].parameters['number_of_days_reduced_duration_with_zinc'])

    #  Run the Cure Event and check episode is ended.
    sim.date = evs[0][0]
    evs[0][1].apply(person_id=person_id)
    assert not df.at[person_id, 'gi_has_diarrhoea']

    # Check that a recovery event occurring later has no effect and does not error.
    sim.date = scheduled_date_recovery
    recovery_event = DiarrhoeaNaturalRecoveryEvent(module=sim.modules['Diarrhoea'], person_id=person_id)
    recovery_event.apply(person_id=person_id)
    assert not df.at[person_id, 'gi_has_diarrhoea']


def test_effect_of_vaccine(seed):
    """Check that if the vaccine is perfect, no one infected with rotavirus and who has the vaccine gets severe
     dehydration."""

    # Create dummy simulation
    start_date = Date(2010, 1, 1)
    popsize = 200

    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # Get the method that determines dehydration
    get_dehydration = sim.modules['Diarrhoea'].models.get_dehydration

    # increase probability to ensure at least one case of severe dehydration when vaccine is imperfect
    sim.modules['Diarrhoea'].parameters['prob_dehydration_by_rotavirus'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_dehydration_by_shigella'] = 1.0

    # 1) Make effect of vaccine perfect
    sim.modules['Diarrhoea'].parameters['rr_severe_dehydration_due_to_rotavirus_with_R1_under1yo'] = 0.0
    sim.modules['Diarrhoea'].parameters['rr_severe_dehydration_due_to_rotavirus_with_R1_over1yo'] = 0.0

    # Check that if person has vaccine and is infected with rotavirus, there is never severe dehydration...
    assert 'severe' not in [get_dehydration(pathogen='rotavirus', va_rota_all_doses=True, age_years=1)
                            for _ in range(100)]
    assert 'severe' not in [get_dehydration(pathogen='rotavirus', va_rota_all_doses=True, age_years=4)
                            for _ in range(100)]

    # ... but if no vaccine or infected with another pathogen, then sometimes it is severe dehydration.
    assert 'severe' in [get_dehydration(pathogen='rotavirus', va_rota_all_doses=False, age_years=1)
                        for _ in range(100)]
    assert 'severe' in [get_dehydration(pathogen='shigella', va_rota_all_doses=True, age_years=1)
                        for _ in range(100)]

    # 2) Make effect of vaccine imperfect
    sim.modules['Diarrhoea'].parameters['rr_severe_dehydration_due_to_rotavirus_with_R1_under1yo'] = 0.5
    sim.modules['Diarrhoea'].parameters['rr_severe_dehydration_due_to_rotavirus_with_R1_over1yo'] = 0.5

    # Check that if the vaccine is imperfect and the person is infected with rotavirus, then there sometimes is severe
    # dehydration.
    assert 'severe' in [get_dehydration(pathogen='rotavirus', va_rota_all_doses=True, age_years=1)
                        for _ in range(100)]
    assert 'severe' in [get_dehydration(pathogen='rotavirus', va_rota_all_doses=True, age_years=2)
                        for _ in range(100)]


def test_check_perfect_treatment_leads_to_zero_risk_of_death(seed):
    """Check that for any permutation of condition, if the treatment is successful, then it prevents death"""

    start_date = Date(2010, 1, 1)
    popsize = 200

    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)

    diarrhoea_module = sim.modules['Diarrhoea']

    for (
        _pathogen,
        _type,
        _duration_longer_than_13days,
        _dehydration,
        _age_exact_years,
        _ri_current_infection_status,
        _untreated_hiv,
        _un_clinical_acute_malnutrition
    ) in product(
        diarrhoea_module.pathogens,
        ['watery', 'bloody'],
        [False, True],
        ['none', 'some', 'severe'],
        range(0, 4, 1),
        [False, True],
        [False, True],
        ['MAM', 'SAM', 'well']
    ):
        # Define the argument to `_get_probability_that_treatment_blocks_death` to represent successful treatment
        _args = {
            'pathogen': _pathogen,
            'type': (_type, 'watery'),  # <-- successful treatment removes blood in diarrhoea
            'duration_longer_than_13days': _duration_longer_than_13days,
            'dehydration': (_dehydration, 'none'),  # <-- successful treatment removes dehydration
            'age_exact_years': _age_exact_years,
            'ri_current_infection_status': _ri_current_infection_status,
            'untreated_hiv': _untreated_hiv,
            'un_clinical_acute_malnutrition': _un_clinical_acute_malnutrition
        }

        assert 1.0 == diarrhoea_module.models._get_probability_that_treatment_blocks_death(**_args), \
            f"Perfect treatment does not prevent death: {_args=}"


@pytest.mark.slow
def test_zero_deaths_when_perfect_treatment(seed):
    """Check that there are no deaths when treatment is perfect and there is perfect healthcare seeking, and no
    healthcare constraints."""

    def get_number_of_deaths_from_cohort_of_children_with_diarrhoea(
        force_any_symptom_to_lead_to_healthcareseeking,
        do_make_treatment_perfect,
        do_increase_risk_of_death=True,
    ) -> int:
        """Run a cohort of children all with newly onset diarrhoea and return number of them that die from diarrhoea."""

        class DummyModule(Module):
            """Dummy module that will cause everyone to have diarrhoea from the first day of the simulation"""
            METADATA = {Metadata.DISEASE_MODULE}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                diarrhoea_module = sim.modules['Diarrhoea']
                pathogens = diarrhoea_module.pathogens

                df = sim.population.props
                for idx in df[df.is_alive].index:
                    sim.schedule_event(
                        event=DiarrhoeaIncidentCase(module=diarrhoea_module,
                                                    person_id=idx,
                                                    pathogen=self.rng.choice(pathogens)),
                        date=sim.date
                    )

        start_date = Date(2010, 1, 1)
        popsize = 10_000
        sim = Simulation(start_date=start_date, seed=seed)
        # Register the appropriate modules
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True,
                         cons_availability='all',
                     ),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         resourcefilepath=resourcefilepath,
                         force_any_symptom_to_lead_to_healthcareseeking=force_any_symptom_to_lead_to_healthcareseeking
                         # every symptom leads to health-care seeking
                     ),
                     diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                     diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                     DummyModule(),
                     )
        # Make entire population under five years old
        sim.modules['Demography'].parameters['max_age_initial'] = 5

        # Set high risk of death and perfect treatment
        if do_increase_risk_of_death:
            increase_risk_of_death(sim.modules['Diarrhoea'])

        if do_make_treatment_perfect:
            make_treatment_perfect(sim.modules['Diarrhoea'])

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date + pd.DateOffset(months=3))

        df = sim.population.props

        # Return number of children who have died with a cause of Diarrhoea
        died_of_diarrhoea = df.loc[~df.is_alive & df['cause_of_death'].str.startswith('Diarrhoea')].index
        return len(died_of_diarrhoea)

    # Some deaths with imperfect treatment and default healthcare seeking
    assert 0 < get_number_of_deaths_from_cohort_of_children_with_diarrhoea(
        force_any_symptom_to_lead_to_healthcareseeking=False,
        do_make_treatment_perfect=False,
    )

    # Some deaths with imperfect treatment and perfect healthcare seeking
    assert 0 < get_number_of_deaths_from_cohort_of_children_with_diarrhoea(
        force_any_symptom_to_lead_to_healthcareseeking=True,
        do_make_treatment_perfect=False,
    )

    # No deaths with perfect healthcare seeking and perfect treatment
    assert 0 == get_number_of_deaths_from_cohort_of_children_with_diarrhoea(
        force_any_symptom_to_lead_to_healthcareseeking=True,
        do_make_treatment_perfect=True,
    )
