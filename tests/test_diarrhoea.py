"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    diarrhoea,
    healthsystem,
    enhanced_lifestyle,
    symptommanager,
    healthburden,
    healthseekingbehaviour,
    dx_algorithm_child,
    labour,
    pregnancy_supervisor)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def check_configuration_of_properties(sim):
    # check that the properties are ok:

    df = sim.population.props

    # Those that have never had diarrhoea, should have null values:
    assert pd.isnull(df.loc[df['gi_last_diarrhoea_pathogen'] == 'none', [
        'gi_last_diarrhoea_date_of_onset',
        'gi_last_diarrhoea_duration',
        'gi_last_diarrhoea_recovered_date',
        'gi_last_diarrhoea_death_date',
        'gi_last_diarrhoea_treatment_date']
    ]).all().all()

    # those that have have never had diarrhoea should have 'none' for type and pathogen of last episode
    assert (df.loc[df['gi_last_diarrhoea_pathogen'] == 'none'].index == df.loc[
        df['gi_last_diarrhoea_type'] == 'none'].index).all()
    assert (df.loc[df['gi_last_diarrhoea_pathogen'] == 'none'].index == df.loc[
        df['gi_last_diarrhoea_dehydration'] == 'none'].index).all()


    # Those that have had diarrhoea, should have a pathogen and a number of days duration
    assert (df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_pathogen'] != 'none').all()

    assert not pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_duration']).any()


    # Those that have had diarrhoea, should have either a recovery date or a death_date (but not bith)
    has_recovery_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_recovered_date'])

    has_death_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_death_date'])

    has_recovery_date_or_death_date = has_recovery_date | has_death_date
    has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    assert has_recovery_date_or_death_date.all()
    assert not has_both_recovery_date_and_death_date.any()

    # Those for whom the death date has past should be dead
    assert not df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_death_date']) &
        (df['gi_last_diarrhoea_death_date'] < sim.date),
        'is_alive'].any()

    # TODO Those currently in an episode of diarrhoea should have symptoms and no-one else should.
    date_of_outcome = df['gi_last_diarrhoea_recovered_date'].fillna(df['gi_last_diarrhoea_death_date'])

    in_current_episode = ~pd.isnull(df.gi_last_diarrhoea_date_of_onset) & \
                         (df.gi_last_diarrhoea_date_of_onset <= sim.date) & \
                         ~pd.isnull(date_of_outcome) & \
                         (date_of_outcome > sim.date)

    assert list(in_current_episode[in_current_episode].index.values) ==\
           sim.modules['SymptomManager'].who_has('diarrhoea')

def test_basic_run_of_diarrhoea_module_with_default_params():
    # Check that the module run and that properties are maintained correctly, using health system and default parameters
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

def test_basic_run_of_diarrhoea_module_with_zero_incidence():
    # Run with zero incidence and check for no cases or deaths
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    for param_name in sim.modules['Diarrhoea'].parameters.keys():
        # **Zero-out incidence**:
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
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for zero-level of diarrhoea
    assert (df['gi_last_diarrhoea_pathogen'] == 'none').all()
    assert (df['gi_last_diarrhoea_type'] == 'none').all()
    assert (df['gi_last_diarrhoea_dehydration'] == 'none').all()

    # Check for zero level of recovery
    assert pd.isnull(df['gi_last_diarrhoea_recovered_date']).all()

    # Check for zero level of death
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_no_treatment():
    # Check that there are incident cases, treatments and deaths occurring correctly
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    for param_name in sim.modules['Diarrhoea'].parameters.keys():
        # Increase incidence:
        if param_name.startswith('base_inc_rate_diarrhoea_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = \
                [4.0 * v for v in sim.modules['Diarrhoea'].parameters[param_name]]

        # Increase symptoms:
        if param_name.startswith('proportion_AWD_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('fever_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('vomiting_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0

    # Increase death:
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for non-zero-level of diarrhoea
    assert (df['gi_last_diarrhoea_pathogen'] != 'none').any()
    assert (df['gi_last_diarrhoea_type'] != 'none').any()
    assert (df['gi_last_diarrhoea_dehydration'] != 'none').any()

    # Check for non-zero level of recovery
    assert not pd.isnull(df['gi_last_diarrhoea_recovered_date']).all()

    # Check for non-zero level of death
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

    # Check that those with a gi_last_diarrhoea_death_date in the past, are now dead with a cause of death of Diarrhoea
    dead_due_to_diarrhoea = ~df.is_alive & df.cause_of_death.str.startswith('Diarrhoea')
    gi_death_date_in_past = ~pd.isnull(df.gi_last_diarrhoea_death_date) & (df.gi_last_diarrhoea_death_date <= sim.date)
    assert (dead_due_to_diarrhoea == gi_death_date_in_past).all()

def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_with_perfect_treatment():
    # Run with everyone gets symptoms and seeks care and perfect treatment efficacy: check that everyone is cured and no deaths;
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=True
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # ensures that every symptom lead to healthcare seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    for param_name in sim.modules['Diarrhoea'].parameters.keys():
        # Increase incidence:
        if param_name.startswith('base_inc_rate_diarrhoea_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = \
                [4.0 * v for v in sim.modules['Diarrhoea'].parameters[param_name]]

        # Increase symptoms so that everyone gets symptoms:
        if param_name.startswith('proportion_AWD_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('fever_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('vomiting_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0

    # Increase death:
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

    # Apply perfect efficacy for treatments:
    sim.modules['Diarrhoea'].parameters['days_onset_severe_dehydration_before_death'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_Treatment_PlanA'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_Treatment_PlanB'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_Treatment_PlanC'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_HSI_Diarrhoea_Severe_Persistent_Diarrhoea'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea'] = 1.0
    sim.modules['Diarrhoea'].parameters['prob_of_cure_given_HSI_Diarrhoea_Dysentery'] = 1.0

    # Make long duration so as to allow time for healthcare seeking
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_rotavirus'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_shigella'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_adenovirus'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_cryptosporidium'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_campylobacter'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_ST-ETEC'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_sapovirus'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_norovirus'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_astrovirus'] = 12
    sim.modules['Diarrhoea'].parameters['mean_days_duration_with_tEPEC'] = 12

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for non-zero-level of diarrhoea
    assert (df['gi_last_diarrhoea_pathogen'] != 'none').any()
    assert (df['gi_last_diarrhoea_type'] != 'none').any()
    assert (df['gi_last_diarrhoea_dehydration'] != 'none').any()

    # Check for non-zero level of recovery
    assert not pd.isnull(df['gi_last_diarrhoea_recovered_date']).all()

    # Check that all of those who got diarrhoea got treatment
    got_diarrhoea = ~pd.isnull(df.gi_last_diarrhoea_date_of_onset)
    assert not pd.isnull(df.loc[got_diarrhoea, 'gi_last_diarrhoea_treatment_date']).any()
    # ******* todo: this is failing because one person (67) is not getting treatment!
    # **** could be to do with short duration episode!?!!?!?



    # Check for zero level of death
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()


# TODO Run with intervention but no health-care-seeking: check that no cures happen etc, some deaths hapen;








