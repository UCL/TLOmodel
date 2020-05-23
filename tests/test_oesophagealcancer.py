import os
from pathlib import Path

import pytest

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophagealcancer,
    pregnancy_supervisor,
    labour,
    healthseekingbehaviour,
    symptommanager
)

# %% Setup:
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

# parameters for whole suite of tests:
start_date = Date(2010, 1, 1)
popsize = 1000

# %% Construction of simulation objects:
def make_simulation_healthsystemdisabled():
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

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
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )
    return sim

def make_simulation_nohsi():
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )
    return sim

# %% Manipulation of parameters:
def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.0] * 6
    return sim

def make_high_init_prev(sim):
    # Set initial prevalence to a high value:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.1] * 6
    return sim

def incr_rates_of_progression_and_effect_of_treatment(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.05

    # Rates of cancer progression per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5

    # Effect of treatment in reducing progression: set so that treatment prevent progression
    sim.modules['OesophagealCancer'].parameters['rr_high_grade_dysp_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage1_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage2_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage3_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage4_undergone_curative_treatment'] = 0.0

    return sim


# %% Checks:
def check_dtypes(sim):
    # check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def check_configuration_of_population(sim):

    # get df for alive persons:
    df = sim.population.props.loc[sim.population.props.is_alive]

    # check that the oc_status and oc_status_any_dysplasia_or_cancer properties always correspond correctly:
    assert set(df.loc[df.oc_status == 'none'].index) == set(df.loc[~df.oc_status_any_dysplasia_or_cancer].index)

    x = df.loc[(df.oc_status == 'none') & (df.oc_status_any_dysplasia_or_cancer), ['oc_status', 'oc_status_any_dysplasia_or_cancer']]

    # check that no one under twenty has cancer
    assert not df.loc[df.age_years < 20].oc_status_any_dysplasia_or_cancer.any()

    # check that diagnosis and treatment is never applied to someone who has never had cancer:
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_treatment']).all()
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_palliative_care']).all()
    assert (df.loc[df.oc_status == 'none', 'oc_stage_at_which_treatment_applied'] == 'none').all()

    # check that treatment is never done for those with oc_status 'stage4'
    assert 0 == (df.oc_stage_at_which_treatment_applied == 'level4').sum()
    assert 0 == (df.loc[~pd.isnull(df.oc_date_treatment)].oc_stage_at_which_treatment_applied == 'none').sum()

    # check that those with symptom are a subset of those with cancer:
    assert set(sim.modules['SymptomManager'].who_has('dysphagia')).issubset(df.index[df.oc_status!='none'])

    # check that those diagnosed are a subset of those with the symptom (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_diagnosis)]).issubset(df.index[df.oc_status_any_dysplasia_or_cancer])
    assert set(df.index[~pd.isnull(df.oc_date_diagnosis)]).issubset(sim.modules['SymptomManager'].who_has('dysphagia'))
    assert (df.loc[~pd.isnull(df.oc_date_diagnosis)].oc_date_diagnosis <= sim.date).all()

    # check that date diagnosed is consistent with the age of the person (ie. not before they were 20.0
    age_at_dx = (df.loc[~pd.isnull(df.oc_date_diagnosis)].oc_date_diagnosis - df.loc[~pd.isnull(df.oc_date_diagnosis)].date_of_birth)
    assert all([int(x.days / 365.25) >= 20 for x in age_at_dx])

    # check that those treated are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_treatment)]).issubset(df.index[~pd.isnull(df.oc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oc_date_treatment)].oc_date_diagnosis <= df.loc[~pd.isnull(df.oc_date_treatment)].oc_date_treatment).all()

    # check that those on palliative care are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_palliative_care)]).issubset(df.index[~pd.isnull(df.oc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oc_date_palliative_care)].oc_date_diagnosis <= df.loc[~pd.isnull(df.oc_date_palliative_care)].oc_date_diagnosis).all()

# %% Tests:
def test_initial_config_of_pop_high_prevalence():
    """Tests of the the way the population is configured: with high initial prevalence values """
    sim = make_simulation_healthsystemdisabled()
    sim = make_high_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)

def test_initial_config_of_pop_zero_prevalence():
    """Tests of the the way the population is configured: with zero initial prevalence values """
    sim = make_simulation_healthsystemdisabled()
    sim = zero_out_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)

def test_initial_config_of_pop_usual_prevalence():
    """Tests of the the way the population is configured: with usual initial prevalence values"""
    sim = make_simulation_healthsystemdisabled()
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)

def test_run_sim_from_high_prevalence():
    """Run the simulation from the usual prevalence values and high rates of incidence and check configuration of
    properties at the end"""
    sim = make_simulation_healthsystemdisabled()
    sim = make_high_init_prev(sim)
    sim = incr_rates_of_progression_and_effect_of_treatment(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)
    sim.simulate(end_date=Date(2012, 1, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)





# def test_check_progression_through_stages_is_happeneing():
#     sim = make_simulation_healthsystemdisabled()
#
#     # set initial prevalence to only be in the initial stage (low_grade_dysplasia)
#     sim = zero_out_init_prev(sim)
#     # todoset box 1 to have something in
#
#
#     # remove effect of treatment:
#
#
#     sim = incr_rates_of_progression_and_effect_of_treatment(sim)
#
#     sim.make_initial_population(n=5000)
#     check_dtypes(sim)
#     check_configuration_of_population(sim)
#     sim.simulate(end_date=Date(2015, 1, 1))
#     check_dtypes(sim)
#     check_configuration_of_population(sim)
#
#     # check that there are not some people in all the later stages:


# def test_check_progression_through_stages_is_blocked_by_treatment():
#     sim = make_simulation_healthsystemdisabled()
#
#     # set initial prevalence to only be in the initial stage (low_grade_dysplasia)
#     sim = zero_out_init_prev(sim)
#     # todoset box 1 to have something in
#
#     # let the effect of treatment be perfect; and let everyone get on treatment (how?)
#
#     sim = incr_rates_of_progression_and_effect_of_treatment(sim)
#
#     sim.make_initial_population(n=5000)
#     check_dtypes(sim)
#     check_configuration_of_population(sim)
#     sim.simulate(end_date=Date(2015, 1, 1))
#     check_dtypes(sim)
#     check_configuration_of_population(sim)

    # check that there are not some people in all the later stages:


# def test_run_from_zero_init():
#     """Tests on the population following simulation"""
#     sim = make_simulation_healthsystemdisabled()
#     sim = zero_out_init_prev(sim)
#     end_date = Date(2020, 1, 1)
#     sim.simulate(end_date=end_date)
#     test_dtypes(sim)
#
#     # Further tests:


# def test_run_from_nonzero_init():
#     """Tests on the population following simulation"""
#     sim = make_simulation_healthsystemdisabled()
#     end_date = Date(2020, 1, 1)
#     sim.simulate(end_date=end_date)
#     test_dtypes(sim)
#
#     # Further tests:


# def test_run_from_zero_init_nohsi():
#     """Tests on the population following simulation"""
#     sim = make_simulation_nohsi()
#     sim = zero_out_init_prev(sim)
#     end_date = Date(2020, 1, 1)
#     sim.simulate(end_date=end_date)
#     test_dtypes(sim)
#
#     # Further test


# def test_run_from_nonzero_init_nohsi():
#     """Tests on the population following simulation"""
#     sim = make_simulation_nohsi()
#     end_date = Date(2020, 1, 1)
#     sim.simulate(end_date=end_date)
#     test_dtypes(sim)
#
#     # Further tests:




"""Other tests:


# DYNAMIC CHECKS:
* that treatment will lead to a piling up of people in a particular stage

* To check the working of the HSI, good shortcut would be to increase the baseline risk of cancer:

* check that treatment reduced risk of progression

* check that progression works:

* no dx w/o health system

* lots of dx, treatment etc w/ health system

*

"""
