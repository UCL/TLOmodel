import os
import time
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    rti,
    symptommanager,
    healthseekingbehaviour,
    contraception,
    depression,
    epi,
    epilepsy,
    dx_algorithm_child,
    dx_algorithm_adult,
    hiv,
    tb,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    pregnancy_supervisor,
    male_circumcision,
    Metadata,

)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000

def check_dtypes(simulation):
    # check types of columns in dataframe, check they are the same, list those that aren't
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all(), ['where dtypes are not the same:', df.dtypes != orig.dtypes]


@pytest.fixture(scope='module')
def test_run():
    """
    This test runs a simulation with a functioning health system with full service availability and no set
    constraints
    """

    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'

    sim.simulate(end_date=end_date)

    check_dtypes(sim)

def test_module_properties():
    """ A test to see whether the logical flows through the module are followed"""
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'

    # Increase the incidence so that we get more people flowing through the RTI model
    params['base_rate_injrti'] = params['base_rate_injrti'] * 5

    sim.simulate(end_date=end_date)

    df = sim.population.props
    # Test whether a person involved in a road traffic accident in the last month either died on scene or was assigned
    # a non-fatal injury, but never both
    assert len(df.loc[df.is_alive & df.rt_road_traffic_inc & df.rt_imm_death & (df.rt_inj_severity != 'none')]) == 0
    # Test that those assigned non-fatal injuries have corresponding injury severity scores
    assert (df.loc[df.is_alive & (df.rt_inj_severity != 'none'), 'rt_ISS_score'] > 0).all()
    assert (df.loc[df.is_alive & (df.rt_inj_severity != 'none'), 'rt_MAIS_military_score'] > 0).all()
    # Test that those who died on scene do not have any injuries stored and no injury severity scores
    cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
            'rt_injury_7', 'rt_injury_8']
    for col in cols:
        assert (df.loc[df.is_alive & df.rt_imm_death, col] == 'none').all()
    assert (df.loc[df.is_alive & df.rt_imm_death, 'rt_ISS_score'] == 0).all()
    assert (df.loc[df.is_alive & df.rt_imm_death, 'rt_MAIS_military_score'] == 0).all()
    # Test that those who have received treatment have been diagnosed in a generic appointment first
    assert (df.loc[df.is_alive & df.rt_road_traffic_inc & df.rt_med_int, 'rt_diagnosed']).all()
    # Test that those who have attempted to get access to treatment, but died because it was unavailable, went through
    # the generic appointments
    assert (df.loc[df.is_alive & df.rt_road_traffic_inc & df.rt_unavailable_med_death, 'rt_diagnosed']).all()
    # Check that recovery dates are after the date of injury
    those_injured_index = df.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death].index
    the_result_of_test = []
    for person in those_injured_index:
        the_result_of_test.append([df.loc[person, 'rt_date_inj'] < date for date in
                                   df.loc[person, 'rt_date_to_remove_daly'] if date is not pd.NaT])
    did_all_pass_test = [True if all(list) else False for list in the_result_of_test]
    assert all(did_all_pass_test)

    assert (df.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_date_inj'] < date for date in
            df.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_date_to_remove_daly'])
    check_dtypes(sim)

def test_with_spurious_symptoms():
    # Run the model with spurious symptoms

    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

def test_with_more_modules():
    # Run the simulation with multiple models, see if any errors or unexpected changes to the datatypes occurs,
    # no particular reason for the choice of modules
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the RTI HSIs"""
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_health_system_events_wont_run():
    """
    Test the model with no service availability
    """
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_sim_high_incidence():
    """
    Run the model with a high incidence, where many people are involved in road traffic injuries
    :return:
    """
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    orig_inc = params['base_rate_injrti']
    params['base_rate_injrti'] = orig_inc * 100
    params['allowed_interventions'] = 'none'

    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_tiny_population():
    """
    Run the model with a small population size
    :return:
    """
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)
    # Note that when n=1 an error was thrown up by the enhanced_lifestyle module when calculating bmi
    sim.make_initial_population(n=2)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_no_capabilities():
    """
    Run the model with a capabilities coefficient of 0.0
    :return:
    """
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)
    # Note that when n=1 an error was thrown up by the enhanced_lifestyle module when calculating bmi
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_health_system_disabled():
    """
    Test the model with the health system disabled
    :return:
    """
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


if __name__ == '__main__':
    t0 = time.time()
    test_run()
    t1 = time.time()
    print('Time taken', t1 - t0)
