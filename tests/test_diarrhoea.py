"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    diarrhoea,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.diarrhoea import (
    HSI_Diarrhoea_Dysentery,
    HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea,
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea,
    HSI_Diarrhoea_Treatment_PlanA,
    HSI_Diarrhoea_Treatment_PlanB,
    HSI_Diarrhoea_Treatment_PlanC,
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

    # Those that have never had diarrhoea, should have not_applicable/null values for all the other properties:
    assert (df.loc[~df.gi_ever_had_diarrhoea & ~df.date_of_birth.isna(), [
        'gi_last_diarrhoea_pathogen',
        'gi_last_diarrhoea_type',
        'gi_last_diarrhoea_dehydration']
    ] == 'not_applicable').all().all()

    assert pd.isnull(df.loc[~df.date_of_birth.isna() & ~df['gi_ever_had_diarrhoea'], [
        'gi_last_diarrhoea_date_of_onset',
        'gi_last_diarrhoea_duration',
        'gi_last_diarrhoea_recovered_date',
        'gi_last_diarrhoea_death_date',
        'gi_last_diarrhoea_treatment_date']
    ]).all().all()

    # Those that have had diarrhoea, should have a pathogen and a number of days duration
    assert (df.loc[df.gi_ever_had_diarrhoea, 'gi_last_diarrhoea_pathogen'] != 'none').all()
    assert not pd.isnull(df.loc[df.gi_ever_had_diarrhoea, 'gi_last_diarrhoea_duration']).any()

    # Those that have had diarrhoea and no treatment, should have either a recovery date or a death_date (but not both)
    has_recovery_date = ~pd.isnull(df.loc[df.gi_ever_had_diarrhoea & pd.isnull(df.gi_last_diarrhoea_treatment_date),
                                          'gi_last_diarrhoea_recovered_date'])
    has_death_date = ~pd.isnull(df.loc[df.gi_ever_had_diarrhoea & pd.isnull(df.gi_last_diarrhoea_treatment_date),
                                       'gi_last_diarrhoea_death_date'])
    has_recovery_date_or_death_date = has_recovery_date | has_death_date
    has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    assert has_recovery_date_or_death_date.all()
    assert not has_both_recovery_date_and_death_date.any()

    # Those for whom the death date has past should be dead
    assert not df.loc[df.gi_ever_had_diarrhoea & (df['gi_last_diarrhoea_death_date'] < sim.date), 'is_alive'].any()

    # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
    #  but not others (among those who are alive)
    has_symptoms_of_diar = set(sim.modules['SymptomManager'].who_has('diarrhoea'))
    has_symptoms = set([p for p in has_symptoms_of_diar if
                        'Diarrhoea' in sim.modules['SymptomManager'].causes_of(p, 'diarrhoea')
                        ])

    in_current_episode_before_recovery = \
        df.is_alive & \
        df.gi_ever_had_diarrhoea & \
        (df.gi_last_diarrhoea_date_of_onset <= sim.date) & \
        (sim.date <= df.gi_last_diarrhoea_recovered_date)
    set_of_person_id_in_current_episode_before_recovery = set(
        in_current_episode_before_recovery[in_current_episode_before_recovery].index
    )

    in_current_episode_before_death = \
        df.is_alive & \
        df.gi_ever_had_diarrhoea & \
        (df.gi_last_diarrhoea_date_of_onset <= sim.date) & \
        (sim.date <= df.gi_last_diarrhoea_death_date)
    set_of_person_id_in_current_episode_before_death = set(
        in_current_episode_before_death[in_current_episode_before_death].index
    )

    in_current_episode_before_cure = \
        df.is_alive & \
        df.gi_ever_had_diarrhoea & \
        (df.gi_last_diarrhoea_date_of_onset <= sim.date) & \
        (df.gi_last_diarrhoea_treatment_date <= sim.date) & \
        pd.isnull(df.gi_last_diarrhoea_recovered_date) & \
        pd.isnull(df.gi_last_diarrhoea_death_date)
    set_of_person_id_in_current_episode_before_cure = set(
        in_current_episode_before_cure[in_current_episode_before_cure].index
    )

    assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
        set_of_person_id_in_current_episode_before_death
    )

    set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
        set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
    )
    assert set_of_person_id_in_current_episode == has_symptoms


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
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
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
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
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

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for zero-level of diarrhoea
    assert 0 == df.loc[df.is_alive].gi_ever_had_diarrhoea.sum()
    assert (df.loc[df.is_alive, 'gi_last_diarrhoea_pathogen'] == 'not_applicable').all()
    assert (df.loc[df.is_alive, 'gi_last_diarrhoea_type'] == 'not_applicable').all()
    assert (df.loc[df.is_alive, 'gi_last_diarrhoea_dehydration'] == 'not_applicable').all()

    # Check for zero level of recovery
    assert pd.isnull(df.loc[df.is_alive, 'gi_last_diarrhoea_recovered_date']).all()

    # Check for zero level of death
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith('Diarrhoea').any()


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
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
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

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    df = sim.population.props

    # Check for non-zero-level of diarrhoea
    assert 0 < df.gi_ever_had_diarrhoea.sum()
    assert (df['gi_last_diarrhoea_pathogen'] != 'none').any()
    assert (df['gi_last_diarrhoea_type'] != 'none').any()
    assert (df['gi_last_diarrhoea_dehydration'] != 'none').any()

    # Check for non-zero level of recovery
    assert not pd.isnull(df['gi_last_diarrhoea_recovered_date']).all()

    # Check for non-zero level of death
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

    # Check that those with a gi_last_diarrhoea_death_date in the past, are now dead
    # NB. Cannot guarantee that all will have a cause of death that is Diarrhoea, because OtherDeathPoll can also
    #  cause deaths.
    gi_death_date_in_past = ~pd.isnull(df.gi_last_diarrhoea_death_date) & (df.gi_last_diarrhoea_death_date <= sim.date)
    assert (
        ~(df.loc[gi_death_date_in_past, 'is_alive']) & ~pd.isnull(df.loc[gi_death_date_in_past, 'date_of_birth'])
    ).all()


@pytest.mark.group2
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_with_perfect_treatment():
    """Run with high incidence and perfect treatment, with and without spurious symptoms of diarrhoea being generated"""

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
                     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                     antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                     labour.Labour(resourcefilepath=resourcefilepath),
                     newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                     postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
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
        sim.modules['Diarrhoea'].parameters['min_days_duration_of_episode'] = 3
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

        sim.make_initial_population(n=popsize)
        check_configuration_of_properties(sim)

        sim.simulate(end_date=end_date)

        check_dtypes(sim)
        check_configuration_of_properties(sim)

        df = sim.population.props

        # Check for non-zero-level of diarrhoea
        assert 0 < df.gi_ever_had_diarrhoea.sum()
        assert (df['gi_last_diarrhoea_pathogen'] != 'none').any()
        assert (df['gi_last_diarrhoea_type'] != 'none').any()
        assert (df['gi_last_diarrhoea_dehydration'] != 'none').any()

        # Check for non-zero level of recovery
        assert not pd.isnull(df['gi_last_diarrhoea_recovered_date']).all()

        # Check that all of those who got diarrhoea got treatment or recovered naturally before treatment was provided
        # (limited to those who did not did not die before that episode ended) and that no one died of the Diarrhoea.
        had_diarrhoea = \
            ~pd.isnull(df.date_of_birth) & \
            df.gi_ever_had_diarrhoea & \
            (df.gi_end_of_last_episode < sim.date) & \
            (
                pd.isnull(df.date_of_death) | (df.date_of_death > df.gi_end_of_last_episode)
            )

        got_treatment = ~pd.isnull(
            df.loc[had_diarrhoea, 'gi_last_diarrhoea_treatment_date']
        )
        recovered_naturally = ~pd.isnull(
            df.loc[had_diarrhoea & pd.isnull(df['gi_last_diarrhoea_treatment_date']),
                   'gi_last_diarrhoea_recovered_date']
        )
        assert (got_treatment | recovered_naturally).all()

        # check that there have not been any deaths caused by Diarrhoea
        assert not df.cause_of_death.loc[~df.is_alive].str.startswith('Diarrhoea').any()

    # run without spurious symptoms
    run(spurious_symptoms=False)

    # run with spurious symptoms
    run(spurious_symptoms=True)


def test_dx_algorithm_for_diarrhoea_outcomes():
    """Create a person and check if the functions in dx_algorithm_child create the correct HSI"""

    def make_blank_simulation():
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
                         force_any_symptom_to_lead_to_healthcareseeking=True
                         # every symptom leads to health-care seeking
                     ),
                     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                     antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                     labour.Labour(resourcefilepath=resourcefilepath),
                     newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                     postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                     diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                     dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                     )

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)

        # Create the HSI event that is notionally doing the call on diagnostic algorithm
        class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = 'DummyHSIEvent'

            def apply(self, person_id, squeeze_factor):
                pass

        hsi_event = DummyHSIEvent(module=sim.modules['Diarrhoea'], person_id=0)

        # check that the queue of events is empty
        assert 0 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)

        return sim, hsi_event

    # ---- PERSON WITH NO DEHYDRATION AND NON-BLOODY DIARRHOEA: ---> PLAN A ----
    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - severe dehydration:
    df = sim.population.props
    duration = 5
    df.at[0, 'gi_ever_had_diarrhoea'] = True
    df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
    df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
    df.at[0, 'gi_last_diarrhoea_dehydration'] = 'none'
    df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
    df.at[0, 'gi_last_diarrhoea_duration'] = duration

    df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
    df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
    df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
    df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)

    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    # Run the diagnostic algorithm:
    sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
        person_id=0,
        hsi_event=hsi_event
    )

    assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanA)

    # ---- PERSON WITH NON-SEVERE DEHYRATION AND NON-BLOODY DIARRHOEA: ---> PLAN B ----
    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - severe dehydration:
    df = sim.population.props
    duration = 5
    df.at[0, 'gi_ever_had_diarrhoea'] = True
    df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
    df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
    df.at[0, 'gi_last_diarrhoea_dehydration'] = 'some'
    df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
    df.at[0, 'gi_last_diarrhoea_duration'] = duration

    df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
    df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
    df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
    df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)

    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='dehydration',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    # Run the diagnostic algorithm:
    sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
        person_id=0,
        hsi_event=hsi_event
    )

    assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanB)

    # %% ---- PERSON WITH SEVERE DEHYRATION and BLOODY DIARRHOEA: --> PLAN C PLUS DYSSENTRY HSI ----

    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - severe dehydration:
    df = sim.population.props
    duration = 5
    df.at[0, 'gi_ever_had_diarrhoea'] = True
    df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
    df.at[0, 'gi_last_diarrhoea_type'] = 'bloody'
    df.at[0, 'gi_last_diarrhoea_dehydration'] = 'severe'
    df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
    df.at[0, 'gi_last_diarrhoea_duration'] = duration

    df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
    df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
    df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
    df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)

    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='dehydration',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='bloody_stool',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    # Run the diagnostic algorithm:
    sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
        person_id=0,
        hsi_event=hsi_event
    )

    assert 2 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanC)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[1][4], HSI_Diarrhoea_Dysentery)

    # %% ---- PERSON WITH NO DEHYDRATION and NON-BLOODY DIARRHOEA BUT LONG-LASTING : --> PLAN A PLUS NON-SEVERE ----

    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - severe dehydration:
    df = sim.population.props
    duration = 20
    df.at[0, 'gi_ever_had_diarrhoea'] = True
    df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
    df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
    df.at[0, 'gi_last_diarrhoea_dehydration'] = 'none'
    df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
    df.at[0, 'gi_last_diarrhoea_duration'] = duration

    df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
    df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
    df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
    df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)

    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='bloody_stool',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    # Run the diagnostic algorithm:
    sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
        person_id=0,
        hsi_event=hsi_event
    )

    assert 2 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanA)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[1][4], HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea)

    # %% ---- PERSON WITH SOME DEHYDRATION and NON-BLOODY DIARRHOEA BUT LONG-LASTING : --> PLAN B PLUS SEVERE ----

    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - severe dehydration:
    df = sim.population.props
    duration = 20
    df.at[0, 'gi_ever_had_diarrhoea'] = True
    df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
    df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
    df.at[0, 'gi_last_diarrhoea_dehydration'] = 'some'
    df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
    df.at[0, 'gi_last_diarrhoea_duration'] = duration

    df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
    df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
    df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
    df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)

    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='diarrhoea',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    sim.modules['SymptomManager'].change_symptom(
        person_id=0,
        symptom_string='dehydration',
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )
    # Run the diagnostic algorithm:
    sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
        person_id=0,
        hsi_event=hsi_event
    )

    assert 2 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Severe_Persistent_Diarrhoea)
    assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[1][4], HSI_Diarrhoea_Treatment_PlanB)


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
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # update the availability of consumables, such that all are available:
    sim.modules['HealthSystem'].cons_item_code_availability_today = \
        sim.modules['HealthSystem'].prob_item_codes_available > 0.0

    list_of_hsi = [
        'HSI_Diarrhoea_Treatment_PlanA',
        'HSI_Diarrhoea_Treatment_PlanB',
        'HSI_Diarrhoea_Treatment_PlanC',
        'HSI_Diarrhoea_Severe_Persistent_Diarrhoea',
        'HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea',
        'HSI_Diarrhoea_Dysentery'
    ]

    for name_of_hsi in list_of_hsi:
        hsi_event = eval(name_of_hsi +
                         "(person_id=0, "
                         "module=sim.modules['Diarrhoea'],"
                         ""
                         ")"
                         )
        hsi_event.run(squeeze_factor=0)
