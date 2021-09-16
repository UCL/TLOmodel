"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

import numpy as np
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
    HSI_Diarrhoea_Treatment_Outpatient,
    HSI_Diarrhoea_Treatment_Inpatient,
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


def increase_incidence_of_pathogens(sim):
    """Helper function to increase the incidence of pathogens and symptoms onset with Diarrhoea."""

    # Increase incidence of pathogens (such that almost certain to get at least one pathogen each year)
    pathogens = sim.modules['Diarrhoea'].pathogens
    for pathogen in pathogens:
        sim.modules['Diarrhoea'].parameters[f"base_inc_rate_diarrhoea_by_{pathogen}"] = \
            [0.95 / len(sim.modules['Diarrhoea'].pathogens)] * 3

    probs = pd.DataFrame(
        [sim.modules['Diarrhoea'].parameters[f"base_inc_rate_diarrhoea_by_{pathogen}"] for pathogen in pathogens]
    )
    assert np.isclose(probs.sum(0), 0.95).all()

    # Increase symptoms so that everyone gets symptoms:
    for param_name in sim.modules['Diarrhoea'].parameters.keys():
        if param_name.startswith('proportion_AWD_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('fever_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('vomiting_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_'):
            sim.modules['Diarrhoea'].parameters[param_name] = 1.0
    return sim


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
                 diarrhoea.PropertiesOfOtherModules(),
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
                 diarrhoea.PropertiesOfOtherModules(),
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
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

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
                 diarrhoea.PropertiesOfOtherModules(),
                 )

    # Increase incidence of pathogens:
    sim = increase_incidence_of_pathogens(sim)

    # Increase death:
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
    sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

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
                     diarrhoea.PropertiesOfOtherModules(),
                     dx_algorithm_child.DxAlgorithmChild()
                     )
        # Edit rate of spurious symptoms to be limited to additional cases of diarrhoea:
        sp_symps = sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence']
        for symp in sp_symps['generic_symptom_name']:
            sp_symps.loc[
                sp_symps['generic_symptom_name'] == symp,
                ['prob_spurious_occurrence_in_adults_per_day', 'prob_spurious_occurrence_in_children_per_day']
            ] = 5.0/1000 if symp == 'diarrhoea' else 0.0

        # Increase incidence of pathogens:
        sim = increase_incidence_of_pathogens(sim)

        # Increase death:
        sim.modules['Diarrhoea'].parameters['case_fatality_rate_AWD'] = 0.5
        sim.modules['Diarrhoea'].parameters['case_fatality_rate_dysentery'] = 0.5

        # Apply perfect efficacy for treatments:
        sim.modules['Diarrhoea'].parameters['ors_effectiveness_on_diarrhoea_mortality'] = 1.0
        sim.modules['Diarrhoea'].parameters['prob_of_cure_given_WHO_PlanC'] = 1.0
        sim.modules['Diarrhoea'].parameters['ors_effectiveness_against_severe_dehydration'] = 1.0
        sim.modules['Diarrhoea'].parameters['antibiotic_effectiveness_for_dysentery'] = 1.0

        # Apply perfect assessment and classification by the health care worker
        sim.modules['Diarrhoea'].parameters['prob_at_least_ors_given_by_hw'] = 1.0
        sim.modules['Diarrhoea'].parameters['prob_recommended_treatment_given_by_hw'] = 1.0
        sim.modules['Diarrhoea'].parameters['prob_antibiotic_given_for_dysentery_by_hw'] = 1.0
        sim.modules['Diarrhoea'].parameters['prob_multivitamins_given_for_persistent_diarrhoea_by_hw'] = 1.0
        sim.modules['Diarrhoea'].parameters['prob_hospitalization_referral_for_severe_diarrhoea'] = 1.0
        sim.modules['Diarrhoea'].parameters['sensitivity_danger_signs_visual_inspection'] = 1.0
        sim.modules['Diarrhoea'].parameters['specificity_danger_signs_visual_inspection'] = 1.0

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
                     disable=False
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(
                     resourcefilepath=resourcefilepath,
                     force_any_symptom_to_lead_to_healthcareseeking=True  # every symptom leads to health-care seeking
                 ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
                 diarrhoea.PropertiesOfOtherModules(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)

    # update the availability of consumables, such that all are available:
    sim.modules['HealthSystem'].cons_item_code_availability_today = \
        sim.modules['HealthSystem'].prob_item_codes_available > 0.0

    list_of_hsi = [
        'HSI_Diarrhoea_Treatment_Outpatient',
        'HSI_Diarrhoea_Treatment_Inpatient',
    ]

    for name_of_hsi in list_of_hsi:
        hsi_event = eval(name_of_hsi +
                         "(person_id=0, "
                         "module=sim.modules['Diarrhoea'])")
        hsi_event.run(squeeze_factor=0)


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
                 diarrhoea.PropertiesOfOtherModules(),
                 hiv.DummyHivModule(),
                 )
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






# todo - align this with the logic desired for the algorithm
# def test_dx_algorithm_for_diarrhoea_outcomes():
#     """Create a person and check if the functions in dx_algorithm_child create the correct HSI"""
#
#     def make_blank_simulation():
#         start_date = Date(2010, 1, 1)
#         popsize = 200  # smallest population size that works
#
#         sim = Simulation(start_date=start_date, seed=0)
#
#         # Register the appropriate modules
#         sim.register(demography.Demography(resourcefilepath=resourcefilepath),
#                      simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#                      enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#                      healthsystem.HealthSystem(
#                          resourcefilepath=resourcefilepath,
#                          disable=False
#                      ),
#                      symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#                      healthseekingbehaviour.HealthSeekingBehaviour(
#                          resourcefilepath=resourcefilepath,
#                          force_any_symptom_to_lead_to_healthcareseeking=True
#                          # every symptom leads to health-care seeking
#                      ),
#                      healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#                      diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath, do_checks=True),
#                      diarrhoea.PropertiesOfOtherModules(),
#                      dx_algorithm_child.DxAlgorithmChild()
#                      )
#
#         sim.make_initial_population(n=popsize)
#         sim.simulate(end_date=start_date)
#
#         # Create the HSI event that is notionally doing the call on diagnostic algorithm
#         class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
#             def __init__(self, module, person_id):
#                 super().__init__(module, person_id=person_id)
#                 self.TREATMENT_ID = 'DummyHSIEvent'
#
#             def apply(self, person_id, squeeze_factor):
#                 pass
#
#         hsi_event = DummyHSIEvent(module=sim.modules['Diarrhoea'], person_id=0)
#
#         # check that the queue of events is empty
#         assert 0 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#
#         return sim, hsi_event
#
#     # ---- PERSON WITH NO DEHYDRATION AND NON-BLOODY DIARRHOEA: ---> PLAN A ----
#     # Set up the simulation:
#     sim, hsi_event = make_blank_simulation()
#
#     # Set up the person - severe dehydration:
#     df = sim.population.props
#     duration = 5
#     df.at[0, 'gi_ever_had_diarrhoea'] = True
#     df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
#     df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
#     df.at[0, 'gi_last_diarrhoea_dehydration'] = 'none'
#     df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
#     df.at[0, 'gi_last_diarrhoea_duration'] = duration
#
#     df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
#     df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
#     df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
#     df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)
#
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='diarrhoea',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     # Run the diagnostic algorithm:
#     sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
#         person_id=0,
#         hsi_event=hsi_event
#     )
#
#     assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#     assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_Outpatient)
#
#     # ---- PERSON WITH NON-SEVERE DEHYDRATION AND NON-BLOODY DIARRHOEA: ---> PLAN B ----
#     # Set up the simulation:
#     sim, hsi_event = make_blank_simulation()
#
#     # Set up the person - some dehydration:
#     df = sim.population.props
#     duration = 5
#     df.at[0, 'gi_ever_had_diarrhoea'] = True
#     df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
#     df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
#     df.at[0, 'gi_last_diarrhoea_dehydration'] = 'some'
#     df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
#     df.at[0, 'gi_last_diarrhoea_duration'] = duration
#
#     df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
#     df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
#     df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
#     df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)
#
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='diarrhoea',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='dehydration',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     # Run the diagnostic algorithm:
#     sim.modules['HealthSystem'].dx_manager.dx_tests['danger_signs_visual_inspection'][0].specificity = 1.0
#     sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
#         person_id=0,
#         hsi_event=hsi_event
#     )
#
#     assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#     assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanB)
#
#     # %% ---- PERSON WITH SEVERE DEHYRATION and NON-BLOODY DIARRHOEA: --> PLAN C ----
#
#     # Set up the simulation:
#     sim, hsi_event = make_blank_simulation()
#
#     # Set up the person - severe dehydration:
#     df = sim.population.props
#     duration = 5
#     df.at[0, 'gi_ever_had_diarrhoea'] = True
#     df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
#     df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
#     df.at[0, 'gi_last_diarrhoea_dehydration'] = 'severe'
#     df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
#     df.at[0, 'gi_last_diarrhoea_duration'] = duration
#
#     df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
#     df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
#     df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
#     df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)
#
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='diarrhoea',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='dehydration',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='bloody_stool',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     # Run the diagnostic algorithm:
#     sim.modules['HealthSystem'].dx_manager.dx_tests['danger_signs_visual_inspection'][0].sensitivity = 1.0
#     sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
#         person_id=0,
#         hsi_event=hsi_event
#     )
#
#     assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#     assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_Inpatient)
#
#     # %% ---- PERSON WITH NO DEHYDRATION and NON-BLOODY DIARRHOEA BUT LONG-LASTING : --> PLAN A ----
#
#     # Set up the simulation:
#     sim, hsi_event = make_blank_simulation()
#
#     # Set up the person - severe dehydration:
#     df = sim.population.props
#     duration = 20
#     df.at[0, 'gi_ever_had_diarrhoea'] = True
#     df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
#     df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
#     df.at[0, 'gi_last_diarrhoea_dehydration'] = 'none'
#     df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
#     df.at[0, 'gi_last_diarrhoea_duration'] = duration
#
#     df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
#     df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
#     df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
#     df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)
#
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='diarrhoea',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='bloody_stool',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     # Run the diagnostic algorithm:
#     sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
#         person_id=0,
#         hsi_event=hsi_event
#     )
#
#     assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#     assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_Outpatient)
#
#     # %% ---- PERSON WITH SOME DEHYDRATION and NON-BLOODY DIARRHOEA BUT LONG-LASTING : --> PLAN B ----
#
#     # Set up the simulation:
#     sim, hsi_event = make_blank_simulation()
#
#     # Set up the person - severe dehydration:
#     df = sim.population.props
#     duration = 20
#     df.at[0, 'gi_ever_had_diarrhoea'] = True
#     df.at[0, 'gi_last_diarrhoea_pathogen'] = 'shigella'
#     df.at[0, 'gi_last_diarrhoea_type'] = 'watery'
#     df.at[0, 'gi_last_diarrhoea_dehydration'] = 'some'
#     df.at[0, 'gi_last_diarrhoea_date_of_onset'] = sim.date - pd.DateOffset(days=duration)
#     df.at[0, 'gi_last_diarrhoea_duration'] = duration
#
#     df.at[0, 'gi_last_diarrhoea_recovered_date'] = sim.date + pd.DateOffset(days=1)
#     df.at[0, 'gi_last_diarrhoea_death_date'] = pd.NaT
#     df.at[0, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
#     df.at[0, 'gi_end_of_last_episode'] = sim.date + pd.DateOffset(days=1)
#
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='diarrhoea',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     sim.modules['SymptomManager'].change_symptom(
#         person_id=0,
#         symptom_string='dehydration',
#         disease_module=sim.modules['Diarrhoea'],
#         add_or_remove='+'
#     )
#     # Run the diagnostic algorithm:
#     sim.modules['HealthSystem'].dx_manager.dx_tests['danger_signs_visual_inspection'][0].specificity = 1.0
#     sim.modules['DxAlgorithmChild'].do_when_diarrhoea(
#         person_id=0,
#         hsi_event=hsi_event
#     )
#
#     assert 1 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
#     assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4], HSI_Diarrhoea_Treatment_PlanB)
