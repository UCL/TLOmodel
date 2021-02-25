"""
Basic tests for the ALRI Module
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    contraception,
    demography,
    ALRI,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.ALRI import (
    HSI_iCCM_Pneumonia_Treatment_level_0,
    HSI_iCCM_Severe_Pneumonia_Treatment_level_0,
    HSI_IMCI_No_Pneumonia_Treatment_level_1,
    HSI_IMCI_Pneumonia_Treatment_level_1,
    HSI_IMCI_Severe_Pneumonia_Treatment_level_1,
    HSI_IMCI_Pneumonia_Treatment_level_2,
    HSI_IMCI_Severe_Pneumonia_Treatment_level_2,
)
from tlo.methods.healthsystem import HSI_Event

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

    # Those that do not have an ALRI, should have not_applicable/null values for all the other properties:
    assert (df.loc[~df.ri_current_ALRI_status & ~df.date_of_birth.isna(), [
        'ri_primary_ALRI_pathogen',
        'ri_secondary_bacterial_pathogen',
        'ri_ALRI_disease_type']
            ] == 'not_applicable').all().all()

    assert pd.isnull(df.loc[~df.date_of_birth.isna() & ~df['ri_current_ALRI_status'], [
        'ri_ALRI_event_date_of_onset',
        'ri_ALRI_event_recovered_date',
        'ri_ALRI_tx_start_date',
        'ri_ALRI_event_death_date']
                     ]).all().all()

    # Those that have had ALRI, should have a pathogen and a number of days duration
    assert (df.loc[df.df.ri_current_ALRI_status, 'ri_primary_ALRI_pathogen'] != 'none').all()
    assert not pd.isnull(df.loc[df.ri_current_ALRI_status, 'ri_ALRI_event_date_of_onset']).any()

    # Those that have had diarrhoea and no treatment, should have either a recovery date or a death_date (but not both)
    has_recovery_date = ~pd.isnull(df.loc[df.ri_current_ALRI_status & pd.isnull(df.ri_ALRI_tx_start_date),
                                          'ri_ALRI_event_recovered_date'])
    has_death_date = ~pd.isnull(df.loc[df.ri_current_ALRI_status & pd.isnull(df.ri_ALRI_tx_start_date),
                                       'ri_ALRI_event_death_date'])
    has_recovery_date_or_death_date = has_recovery_date | has_death_date
    has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    assert has_recovery_date_or_death_date.all()
    assert not has_both_recovery_date_and_death_date.any()

    # Those for whom the death date has past should be dead
    assert not df.loc[df.ri_current_ALRI_status & (df['ri_ALRI_event_death_date'] < sim.date), 'is_alive'].any()

    # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
    #  but not others (among those who are alive)
    has_symptoms_of_diar = set(sim.modules['SymptomManager'].who_has('difficult_breathing'))
    has_symptoms = set([p for p in has_symptoms_of_diar if
                        'Diarrhoea' in sim.modules['SymptomManager'].causes_of(p, 'difficult_breathing')
                        ])

    in_current_episode_before_recovery = \
        df.is_alive & \
        df.ri_current_ALRI_status & \
        (df.ri_ALRI_event_date_of_onset <= sim.date) & \
        (sim.date <= df.ri_ALRI_event_recovered_date)
    set_of_person_id_in_current_episode_before_recovery = set(
        in_current_episode_before_recovery[in_current_episode_before_recovery].index
    )

    in_current_episode_before_death = \
        df.is_alive & \
        df.ri_current_ALRI_status & \
        (df.ri_ALRI_event_date_of_onset <= sim.date) & \
        (sim.date <= df.ri_ALRI_event_death_date)
    set_of_person_id_in_current_episode_before_death = set(
        in_current_episode_before_death[in_current_episode_before_death].index
    )

    in_current_episode_before_cure = \
        df.is_alive & \
        df.ri_current_ALRI_status & \
        (df.ri_ALRI_event_date_of_onset <= sim.date) & \
        (df.ri_ALRI_tx_start_date <= sim.date) & \
        pd.isnull(df.ri_ALRI_event_recovered_date) & \
        pd.isnull(df.ri_ALRI_event_death_date)
    set_of_person_id_in_current_episode_before_cure = set(
        in_current_episode_before_cure[in_current_episode_before_cure].index
    )

    assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
        set_of_person_id_in_current_episode_before_death
    )

    set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
        set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
    )
    # assert set_of_person_id_in_current_episode == has_symptoms


def test_basic_run_of_diarrhoea_module_with_default_params():
    # Check that the module run and that properties are maintained correctly, using health system and default parameters
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0)

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
                 ALRI.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

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

    sim = Simulation(start_date=start_date, seed=0)

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
                 ALRI.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    for param_name in sim.modules['ALRI'].parameters.keys():
        # **Zero-out incidence**:
        if param_name.startswith('base_inc_rate_ALRI_by_'):
            sim.modules['ALRI'].parameters[param_name] = \
                [0.0 * v for v in sim.modules['ALRI'].parameters[param_name]]

        # # Increase symptoms (to be consistent with other checks):
        # if param_name.startswith('proportion_AWD_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('fever_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('vomiting_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('dehydration_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0

    # Increase death (to be consistent with other checks):
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bronchiolitis'] = 0.5
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_viral_pneumonia'] = 0.5
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bacterial_pneumonia'] = 0.5

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for zero-level of diarrhoea
    assert 0 == df.loc[df.is_alive].gi_ever_had_diarrhoea.sum()
    assert (df.loc[df.is_alive, 'ri_primary_ALRI_pathogen'] == 'not_applicable').all()
    assert (df.loc[df.is_alive, 'ri_ALRI_disease_type'] == 'not_applicable').all()

    # Check for zero level of recovery
    assert pd.isnull(df.loc[df.is_alive, 'ri_ALRI_event_recovered_date']).all()

    # Check for zero level of death
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith('ALRI').any()


@pytest.mark.group2
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_no_treatment():
    # Check that there are incident cases, treatments and deaths occurring correctly
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 2000

    sim = Simulation(start_date=start_date, seed=0)

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
                 ALRI.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    for param_name in sim.modules['ALRI'].parameters.keys():
        # Increase incidence:
        if param_name.startswith('base_inc_rate_ALRI_by_'):
            sim.modules['ALRI'].parameters[param_name] = \
                [4.0 * v for v in sim.modules['ALRI'].parameters[param_name]]

        # # Increase symptoms:
        # if param_name.startswith('proportion_AWD_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('fever_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('vomiting_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        # if param_name.startswith('dehydration_by_'):
        #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0

    # Increase death (to be consistent with other checks):
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bronchiolitis'] = 0.5
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_viral_pneumonia'] = 0.5
    sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bacterial_pneumonia'] = 0.5

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for non-zero-level of diarrhoea
    assert 0 < df.ri_current_ALRI_status.sum()
    assert (df['ri_primary_ALRI_pathogen'] != 'none').any()
    assert (df['ri_ALRI_disease_type'] != 'none').any()

    # Check for non-zero level of recovery
    assert not pd.isnull(df['ri_ALRI_event_recovered_date']).all()

    # Check for non-zero level of death
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('ALRI').any()

    # Check that those with a gi_last_diarrhoea_death_date in the past, are now dead
    # NB. Cannot guarantee that all will have a cause of death that is Diarrhoea, because OtherDeathPoll can also
    #  cause deaths.
    ri_death_date_in_past = ~pd.isnull(df.ri_ALRI_event_death_date) & (df.ri_ALRI_event_death_date <= sim.date)
    assert (
        ~(df.loc[ri_death_date_in_past, 'is_alive']) & ~pd.isnull(df.loc[ri_death_date_in_past, 'date_of_birth'])
    ).all()


@pytest.mark.group2
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_with_perfect_treatment():
    """
    Run with high incidence and perfect treatment, with and without spurious symptoms of cough/difficult breathing
     being generated
    """

    def run(spurious_symptoms):
        # Run with everyone getting symptoms and seeking care and perfect treatment efficacy:
        # Check that everyone is cured and no deaths;
        start_date = Date(2010, 1, 1)
        end_date = Date(2010, 12, 31)  # reduce run time because with spurious_symptoms=True, things get slow
        popsize = 4000

        sim = Simulation(start_date=start_date, seed=0)

        # Register the appropriate modules
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     contraception.Contraception(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True
                     ),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath,
                                                   spurious_symptoms=spurious_symptoms),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         resourcefilepath=resourcefilepath,
                         force_any_symptom_to_lead_to_healthcareseeking=True
                         # every symptom leads to healthcare seeking
                     ),
                     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                     labour.Labour(resourcefilepath=resourcefilepath),
                     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                     ALRI.ALRI(resourcefilepath=resourcefilepath),
                     dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                     )

        for param_name in sim.modules['ALRI'].parameters.keys():
            # Increase incidence:
            if param_name.startswith('base_inc_rate_ALRI_by_'):
                sim.modules['ALRI'].parameters[param_name] = \
                    [4.0 * v for v in sim.modules['ALRI'].parameters[param_name]]

            # # Increase symptoms so that everyone gets symptoms:
            # if param_name.startswith('proportion_AWD_by_'):
            #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
            # if param_name.startswith('fever_by_'):
            #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
            # if param_name.startswith('vomiting_by_'):
            #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0
            # if param_name.startswith('dehydration_by_'):
            #     sim.modules['Diarrhoea'].parameters[param_name] = 1.0

        # Increase death (to be consistent with other checks):
        sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bronchiolitis'] = 0.5
        sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_viral_pneumonia'] = 0.5
        sim.modules['ALRI'].parameters['base_death_rate_ALRI_by_bacterial_pneumonia'] = 0.5

        # Apply perfect efficacy for treatments:
        sim.modules['ALRI'].parameters['prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment'] = 1.0
        sim.modules['ALRI'].parameters['prob_of_cure_for_pneumonia_with_severe_complication_given_IMCI_severe_pneumonia_treatment'] = 1.0

        # Make long duration so as to allow time for healthcare seeking
        # not added for ALRI, no property on duration of episode

        sim.make_initial_population(n=popsize)
        check_configuration_of_properties(sim)

        sim.simulate(end_date=end_date)

        check_dtypes(sim)
        check_configuration_of_properties(sim)

        df = sim.population.props

        # Check for non-zero-level of diarrhoea
        assert 0 < df.ri_current_ALRI_status.sum()
        assert (df['ri_primary_ALRI_pathogen'] != 'none').any()
        assert (df['ri_ALRI_disease_type'] != 'none').any()

        # Check for non-zero level of recovery
        assert not pd.isnull(df['ri_ALRI_event_recovered_date']).all()

        # Check that all of those who got diarrhoea got treatment or recovered naturally before treatment was provided
        # and no one died of the Diarrhoea. (Limited to those whose last onset diarrhoea was one month ago to give time
        # for outcomes to have occurred).
        had_ALRI_a_month_ago = df.ri_current_ALRI_status & (
            df.ri_ALRI_event_date_of_onset < (sim.date - pd.DateOffset(months=1))
        )
        got_treatment = ~pd.isnull(
            df.loc[had_ALRI_a_month_ago, 'ri_ALRI_tx_start_date']
        )
        recovered_naturally = ~pd.isnull(
            df.loc[had_ALRI_a_month_ago & pd.isnull(df['ri_ALRI_tx_start_date']),
                   'ri_ALRI_event_recovered_date']
        )
        assert (got_treatment | recovered_naturally).all()

        # check that there have not been any deaths caused by Diarrhoea
        assert not df.cause_of_death.loc[~df.is_alive].str.startswith('ALRI').any()

    # run without spurious symptoms
    run(spurious_symptoms=False)

    # run with spurious symptoms
    run(spurious_symptoms=True)



def test_run_each_of_the_HSI():
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ALRI.ALRI(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # update the availability of consumables, such that all are available:
    sim.modules['HealthSystem'].cons_item_code_availability_today = \
        sim.modules['HealthSystem'].prob_item_codes_available > 0.0

    list_of_hsi = [
        'HSI_iCCM_Pneumonia_Treatment_level_0',
        'HSI_iCCM_Severe_Pneumonia_Treatment_level_0',
        'HSI_IMCI_No_Pneumonia_Treatment_level_1',
        'HSI_IMCI_Pneumonia_Treatment_level_1',
        'HSI_IMCI_Severe_Pneumonia_Treatment_level_1',
        'HSI_IMCI_Pneumonia_Treatment_level_2',
        'HSI_IMCI_Severe_Pneumonia_Treatment_level_2'
    ]

    for name_of_hsi in list_of_hsi:
        hsi_event = eval(name_of_hsi +
                         "(person_id=0, "
                         "module=sim.modules['ALRI'],"
                         ""
                         ")"
                         )
        hsi_event.run(squeeze_factor=0)
