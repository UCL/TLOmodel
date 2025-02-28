import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    diabetic_retinopathy,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
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
    """Return simulation objection with Diabetic Retinopathy and other necessary modules registered."""
    sim = Simulation(
            start_date=start_date,
            seed=seed,
        )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True
                                                               ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 diabetic_retinopathy.DiabeticRetinopathy(),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 )
    return sim


def get_simulation_healthsystemdisabled(seed):
    """Make the simulation with the health system disabled
    """
    sim = Simulation(
        start_date=start_date,
        seed=seed,
    )

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=False
                                                               ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 diabetic_retinopathy.DiabeticRetinopathy(),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 )
    return sim


def check_dtypes(sim):
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['DiabeticRetinopathy'].parameters['init_prob_any_dr'] = 0.0
    sim.modules['DiabeticRetinopathy'].parameters['init_prob_late_dr'] = 0.0
    return sim


def check_configuration_of_population(sim):
    df = sim.population.props.copy()

    # Boolean for any stage of diabetic retinopathy
    df['dr_status_any_stage'] = df.dr_status != 'none'

    # Get alive people:
    df = df.loc[df.is_alive]

    # check that diagnosis and treatment is never applied to someone who has never had diabetic retinopathy:
    assert pd.isnull(df.loc[df.dr_status == 'none', 'dr_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.dr_status == 'none', 'dr_date_treatment']).all()

    # check that those with symptoms are a subset of those with diabetic retinopathy:
    assert set(sim.modules['SymptomManager'].who_has('blindness_full')).issubset(df.index[df.dr_status != 'none'])
    assert set(sim.modules['SymptomManager'].who_has('blindness_partial')).issubset(df.index[df.dr_status != 'none'])

    # check that those diagnosed are a subset of those with any dr (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.dr_date_diagnosis)]).issubset(df.index[df.dr_status_any_stage])


@pytest.mark.slow
def test_basic_run(seed):
    """Run the simulation with the diabetic_retinopathy module and read the log from the diabetic_retinopathy module."""
    sim = get_simulation(seed)
    sim.make_initial_population(n=popsize)

    check_dtypes(sim)
    #TODO Add check params when resourcefile has been created
    # check_params_read(sim)
    sim.simulate(end_date=Date(2010, 5, 1))
    check_dtypes(sim)


def test_initial_config_of_pop_zero_prevalence(seed):
    """Tests of the the way the population is configured: with zero initial prevalence values """
    sim = get_simulation_healthsystemdisabled(seed=seed)
    sim = zero_out_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)
    df = sim.population.props
    assert (df.loc[df.is_alive].dr_status == 'none').all()


