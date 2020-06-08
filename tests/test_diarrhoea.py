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
    assert pd.isnull(df.loc[df['gi_last_diarrhoea_dehydration'] == 'none',[
        'gi_last_diarrhoea_date_of_onset',
        'gi_last_diarrhoea_duration',
        'gi_last_diarrhoea_recovered_date',
        'gi_last_diarrhoea_death_date']
    ]).any().any()

    # Those that have had diarrhoea, should have a pathogen and a number of days duration
    assert (df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_pathogen'] != 'none').all()

    assert not pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_duration']).any()


    # Those that have had diarrhoea, should have either a recovery date or a death_date
    has_recovery_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_recovered_date'])

    has_death_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_death_date'])

    has_recovery_date_or_death_date = has_recovery_date | has_death_date

    assert has_recovery_date_or_death_date.all()

    # Those for whom the death date has past should be dead
    assert not df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_death_date']) &
        (df['gi_last_diarrhoea_death_date'] < sim.date),
        'is_alive'].any()

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
        if param_name.startswith('vomiting_by_rotavirus'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_rotavirus'):
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
    # Check that there are incidences, treatments and deaths occuring correctly
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
        if param_name.startswith('vomiting_by_rotavirus'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_rotavirus'):
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


# TODO Run with intervention but not symptoms: check that no cures happen etc, some deaths hapen;


# TODO Run with high level of intervention and perfect efficacy: check that some all cures happen and no deaths;





