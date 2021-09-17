"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    diarrhoea,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
)
from tlo.methods.diarrhoea import (
    HSI_Diarrhoea_Treatment_Inpatient,
    HSI_Diarrhoea_Treatment_Outpatient,
    increase_incidence_of_pathogens,
    increase_risk_of_death,
    make_treatment_perfect,
)

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


def test_basic_run_of_diarrhoea_module_with_default_params():
    """Check that the module run and that properties are maintained correctly, using health system and default
    parameters"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 dx_algorithm_child.DxAlgorithmChild()
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_basic_run_of_diarrhoea_module_with_zero_incidence():
    """Run with zero incidence and check for no cases or deaths"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0)

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
                 hiv.DummyHivModule(),
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


def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_no_treatment(tmpdir):
    """Check that there are incident cases, treatments and deaths occurring correctly"""
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    popsize = 2000

    log_config = {'filename': 'tmpfile',
                  'directory': tmpdir,
                  'custom_levels': {
                      "Diarrhoea": logging.INFO}
                  }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

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


@pytest.mark.group2
def test_basic_run_of_diarrhoea_module_with_high_incidence_and_high_death_and_with_perfect_treatment(tmpdir):
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
                          "Diarrhoea": logging.INFO}
                      }

        sim = Simulation(start_date=start_date, seed=0, show_progress_bar=True, log_config=log_config)

        # Register the appropriate modules
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True,
                         ignore_cons_constraints=True,
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
                     dx_algorithm_child.DxAlgorithmChild()
                     )
        # Edit rate of spurious symptoms to be limited to additional cases of diarrhoea:
        sp_symps = sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence']
        for symp in sp_symps['generic_symptom_name']:
            sp_symps.loc[
                sp_symps['generic_symptom_name'] == symp,
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


def test_run_each_of_the_HSI():
    """Check that HSI specified can be run correctly"""
    start_date = Date(2010, 1, 1)
    popsize = 200  # smallest population size that works

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     ignore_cons_constraints=True
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


def test_does_treatment_prevent_death():
    """Check that the helper function 'does_treatment_prevent_death' works as expected."""

    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 dx_algorithm_child.DxAlgorithmChild(),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                 hiv.DummyHivModule(),
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


def test_do_treatment():
    """Check that the function `do_treatment` work as expected"""
    # todo!
    pass
