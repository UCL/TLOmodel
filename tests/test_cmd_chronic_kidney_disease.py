import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    cardio_metabolic_disorders,
    cmd_chronic_kidney_disease,
    demography,
    depression,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)
end_date = Date(2010, 1, 2)
popsize = 1000


def get_simulation(seed):
    """Return simulation objection with cmd_chronic_kidney_disease and other necessary modules registered."""
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        resourcefilepath=resourcefilepath
    )

    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(disable=False, cons_availability='all'),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(  # force symptoms to lead to health care seeking:
                     force_any_symptom_to_lead_to_healthcareseeking=True
                 ),
                 healthburden.HealthBurden(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 cmd_chronic_kidney_disease.CMDChronicKidneyDisease(),
                 depression.Depression(),
                 hiv.Hiv(),
                 epi.Epi(),
                 tb.Tb(),
                 )
    return sim


def get_simulation_healthsystemdisabled(seed):
    """Make the simulation with the health system disabled
    """
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        resourcefilepath=resourcefilepath
    )

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(disable=True),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(  # force symptoms to lead to health care seeking:
                     force_any_symptom_to_lead_to_healthcareseeking=False
                 ),
                 healthburden.HealthBurden(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 cmd_chronic_kidney_disease.CMDChronicKidneyDisease(),
                 depression.Depression(),
                 hiv.Hiv(),
                 epi.Epi(),
                 tb.Tb(),
                 )
    return sim


def get_simulation_nohsi(seed):
    """Make the simulation with:
    * the healthsystem enabled but with no service availability (so no HSI run)
    """
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(service_availability=[]),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 healthburden.HealthBurden(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 cmd_chronic_kidney_disease.CMDChronicKidneyDisease(),
                 depression.Depression(),
                 hiv.Hiv(),
                 epi.Epi(),
                 tb.Tb(),
                 )
    return sim


def check_dtypes(sim):
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['CMDChronicKidneyDisease'].parameters['init_prob_any_ckd'] = [0.0] * 2
    return sim


def make_high_init_prev(sim):
    # Set initial prevalence to a high value:
    sim.modules['CMDChronicKidneyDisease'].parameters['init_prob_any_ckd'] = [0.1] * 2
    return sim


def incr_rate_of_onset_stage1_4(sim):
    # Rate of CKD onset per # months:
    # sim.modules['CMDChronicKidneyDisease'].parameters['rate_onset_to_stage1_4'] = 0.05
    sim.modules['CMDChronicKidneyDisease'].parameters['rate_onset_to_stage1_4'] = 0.5
    return sim


def zero_rate_of_onset_stage1_4(sim):
    # Rate of CKD onset per # months:
    sim.modules['CMDChronicKidneyDisease'].parameters['rate_onset_to_stage1_4'] = 0.00
    return sim


def incr_rates_of_progression(sim):
    # Rates of CKD progression:
    sim.modules['CMDChronicKidneyDisease'].parameters['rate_onset_to_stage1_4'] *= 5
    sim.modules['CMDChronicKidneyDisease'].parameters['rate_stage1_4_to_stage5'] *= 5
    return sim


def check_configuration_of_population(sim):
    df = sim.population.props.copy()

    # Boolean for any stage of CKD
    df['ckd_status_any_stage'] = df.ckd_status != 'pre_diagnosis'

    # Get alive people:
    df = df.loc[df.is_alive]

    # check that diagnosis and treatment is never applied to someone who has never had CKD
    assert pd.isnull(df.loc[df.ckd_status == 'pre_diagnosis', 'ckd_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.ckd_status == 'pre_diagnosis', 'ckd_date_treatment']).all()

    # check that those diagnosed are a subset of those with any CKD (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.ckd_date_diagnosis)]).issubset(df.index[df.ckd_status_any_stage])


@pytest.mark.slow
def test_basic_run(seed):
    """Run the simulation with the cmd_chronic_kidney_disease module and read the log from the
    cmd_chronic_kidney_disease module."""
    sim = get_simulation(seed)
    sim.make_initial_population(n=popsize)

    check_dtypes(sim)
    sim.simulate(end_date=Date(2010, 5, 1))
    check_dtypes(sim)


def test_initial_config_of_pop_usual_prevalence(seed):
    """Tests the way the population is configured: with usual initial prevalence values"""
    sim = get_simulation_healthsystemdisabled(seed=seed)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)


def test_initial_config_of_pop_zero_prevalence(seed):
    """Tests the way the population is configured: with zero initial prevalence values """
    sim = get_simulation_healthsystemdisabled(seed=seed)
    sim = zero_out_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)
    df = sim.population.props
    assert (df.loc[df.is_alive & ~df.nc_chronic_kidney_disease].ckd_status == 'pre_diagnosis').all()


def test_initial_config_of_pop_high_prevalence(seed):
    """Tests the way the population is configured: with high initial prevalence values """
    sim = get_simulation_healthsystemdisabled(seed=seed)
    sim = make_high_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)


@pytest.mark.slow
def test_run_sim_from_high_prevalence(seed):
    """Run the simulation from the usual prevalence values and high rates of incidence and check configuration of
    properties at the end"""
    sim = get_simulation_healthsystemdisabled(seed=seed)
    sim = make_high_init_prev(sim)
    sim = incr_rates_of_progression(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)
    sim.simulate(end_date=Date(2010, 4, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)
