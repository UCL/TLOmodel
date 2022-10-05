import os
import time
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    depression,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)
from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1

# create simulation parameters
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000


def check_dtypes(simulation):
    # check types of columns in dataframe, check they are the same, list those that aren't
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all(), ['where dtypes are not the same:', df.dtypes != orig.dtypes]


def create_basic_rti_sim(population_size, seed):
    # create the basic outline of an rti simulation object
    sim = Simulation(start_date=start_date, seed=seed)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=population_size)
    return sim


@pytest.mark.slow
def test_run(seed):
    """
    This test runs a simulation with a functioning health system with full service availability and no set
    constraints
    """
    # create sim object
    sim = create_basic_rti_sim(popsize, seed)
    # run simulation
    sim.simulate(end_date=end_date)
    # check datatypes are same through sim
    check_dtypes(sim)


@pytest.mark.slow
def test_all_injuries_run(seed):
    """
    This test runs a simulation with a functioning health system with full service availability and no set
    constraints
    """
    # create sim object
    sim = create_basic_rti_sim(97, seed)
    # create a list of injuries to assign the individuals in the population
    injuries_to_assign = sim.modules['RTI'].INJURY_CODES
    # assign injuries to the population at random
    sim.population.props['rt_injury_1'] = injuries_to_assign
    # change the datatype back to a category
    sim.population.props['rt_injury_1'] = sim.population.props['rt_injury_1'].astype("category")
    # Check that each injury appears at least once in the population, ensuring that as the simulation runs no new
    # categorical variables will be assigned to rt_injury_1
    assert len(set(sim.population.props['rt_injury_1'].dtypes.categories)
               .intersection(set(sim.modules['RTI'].INJURY_CODES))) == len(set(sim.modules['RTI'].INJURY_CODES))
    # Change permanent injuries in the simulation as they have no associated DALY weight, choose fractured skull
    sim.population.props.loc[sim.population.props['rt_injury_1'].str.contains('P'), 'rt_injury_1'] = '112'
    # give those who have no injury an injury, choose fractured skull
    sim.population.props.loc[sim.population.props['rt_injury_1'] == 'none', 'rt_injury_1'] = '112'
    # final check to ensure everyone has a non permanent injury so a daly weight can be assigned
    assert "none" not in sim.population.props['rt_injury_1'].unique()
    assert not sim.population.props['rt_injury_1'].str.contains('P').any()
    # Assign people the emergency care triggering symptom so they enter the health system
    sim.population.props['sy_severe_trauma'] = 2
    # Assign an injury date
    sim.population.props['rt_date_inj'] = sim.start_date
    # Show that they have been injured
    sim.population.props['rt_road_traffic_inc'] = True
    # Assign them a random ISS score
    sim.population.props['rt_ISS_score'] = sim.rng.randint(1, 76, size=len(sim.population.props))
    # Assign them a random MAIS score
    sim.population.props['rt_MAIS_military_score'] = sim.rng.randint(1, 6, size=len(sim.population.props))
    # Assign them a date to check mortality without the health system
    sim.population.props['rt_date_death_no_med'] = sim.start_date + pd.DateOffset(weeks=2)
    # Assign an injury severity
    sim.population.props['rt_inj_severity'] = sim.rng.choice(['none', 'mild', 'severe'], len(sim.population.props))
    # Change the dtype back to category
    sim.population.props['rt_inj_severity'] = sim.population.props['rt_inj_severity'].astype("category")
    # replace those with an injury severity of 'none' with a mild injury severity
    sim.population.props.loc[sim.population.props['rt_inj_severity'] == 'none', 'rt_inj_severity'] = 'mild'
    # Assign daly weights to the population
    sim.modules['RTI'].rti_assign_daly_weights(sim.population.props.index)
    # Schedule the generic emergency appointment
    for person_id in sim.population.props.index:
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_GenericEmergencyFirstApptAtFacilityLevel1(module=sim.modules['RTI'], person_id=person_id),
            priority=0,
            topen=sim.date
        )
    # run simulation
    sim.simulate(end_date=end_date)
    # check datatypes are same through sim
    check_dtypes(sim)


@pytest.mark.slow
def test_all_injuries_run_no_healthsystem(seed):
    """
    This test runs a simulation with a functioning health system with full service availability and no set
    constraints
    """
    # create sim object
    sim = Simulation(start_date=start_date, seed=seed)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
                 )
    sim.make_initial_population(n=97)
    # create a list of injuries to assign the individuals in the population
    injuries_to_assign = sim.modules['RTI'].INJURY_CODES
    # assign injuries to the population at random
    sim.population.props['rt_injury_1'] = injuries_to_assign
    # change the datatype back to a category
    sim.population.props['rt_injury_1'] = sim.population.props['rt_injury_1'].astype("category")
    # Check that each injury appears at least once in the population, ensuring that as the simulation runs no new
    # categorical variables will be assigned to rt_injury_1
    assert len(set(sim.population.props['rt_injury_1'].dtypes.categories)
               .intersection(set(sim.modules['RTI'].INJURY_CODES))) == len(set(sim.modules['RTI'].INJURY_CODES))
    # Change permanent injuries in the simulation as they have no associated DALY weight, choose fractured skull
    sim.population.props.loc[sim.population.props['rt_injury_1'].str.contains('P'), 'rt_injury_1'] = '112'
    # give those who have no injury an injury, choose fractured skull
    sim.population.props.loc[sim.population.props['rt_injury_1'] == 'none', 'rt_injury_1'] = '112'
    # final check to ensure everyone has a non permanent injury so a daly weight can be assigned
    assert "none" not in sim.population.props['rt_injury_1'].unique()
    assert not sim.population.props['rt_injury_1'].str.contains('P').any()
    # Assign people the emergency care triggering symptom so they enter the health system
    sim.population.props['sy_severe_trauma'] = 2
    # Assign an injury date
    sim.population.props['rt_date_inj'] = sim.start_date
    # Show that they have been injured
    sim.population.props['rt_road_traffic_inc'] = True
    # Assign them a random ISS score
    sim.population.props['rt_ISS_score'] = sim.rng.randint(1, 76, size=len(sim.population.props))
    # Assign them a random MAIS score
    sim.population.props['rt_MAIS_military_score'] = sim.rng.randint(1, 6, size=len(sim.population.props))
    # Assign them a date to check mortality without the health system
    sim.population.props['rt_date_death_no_med'] = sim.start_date + pd.DateOffset(weeks=2)
    # Assign an injury severity
    sim.population.props['rt_inj_severity'] = sim.rng.choice(['none', 'mild', 'severe'], len(sim.population.props))
    # Change the dtype back to category
    sim.population.props['rt_inj_severity'] = sim.population.props['rt_inj_severity'].astype("category")
    # replace those with an injury severity of 'none' with a mild injury severity
    sim.population.props.loc[sim.population.props['rt_inj_severity'] == 'none', 'rt_inj_severity'] = 'mild'
    # Assign daly weights to the population
    sim.modules['RTI'].rti_assign_daly_weights(sim.population.props.index)
    # Schedule the generic emergency appointment
    for person_id in sim.population.props.index:
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_GenericEmergencyFirstApptAtFacilityLevel1(module=sim.modules['RTI'], person_id=person_id),
            priority=0,
            topen=sim.date
        )
    # run simulation
    sim.simulate(end_date=end_date)
    # check datatypes are same through sim
    check_dtypes(sim)


@pytest.mark.slow
def test_blocked_interventions(seed):
    sim = create_basic_rti_sim(popsize, seed)
    params = sim.modules['RTI'].parameters
    params['blocked_interventions'] = ['Minor Surgery']
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    sim = create_basic_rti_sim(popsize, seed)
    params = sim.modules['RTI'].parameters
    params['blocked_interventions'] = ['Fracture Casts']
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    sim = create_basic_rti_sim(popsize, seed)
    params = sim.modules['RTI'].parameters
    params['blocked_interventions'] = ['Minor Surgery', 'Fracture Casts']
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


@pytest.mark.slow
def test_module_properties(seed):
    """ A test to see whether the logical flows through the module are followed"""
    sim = create_basic_rti_sim(popsize, seed)
    params = sim.modules['RTI'].parameters
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
        the_result_of_test.append(
            [
                df.loc[person, 'rt_date_inj'] < date
                for date in df.loc[person, rti.RTI.DATE_TO_REMOVE_DALY_COLUMNS]
                if date is not pd.NaT
            ]
        )
    did_all_pass_test = [True if all(test_list) else False for test_list in the_result_of_test]
    assert all(did_all_pass_test)

    assert (
        df.loc[
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_date_inj'
        ] < date
        for date in df.loc[
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death,
            rti.RTI.DATE_TO_REMOVE_DALY_COLUMNS
        ]
    )
    check_dtypes(sim)


@pytest.mark.slow
def test_with_more_modules(seed):
    # Run the simulation with multiple modules, see if any errors or unexpected changes to the datatypes occurs
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    # Make the population
    sim.make_initial_population(n=popsize)
    # run the simulation
    sim.simulate(end_date=end_date)
    # check datatypes
    check_dtypes(sim)


@pytest.mark.slow
def test_run_health_system_high_squeeze(seed):
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the RTI HSIs"""
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the modules and change healthsystem parameters
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
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )
    # make the initial population
    sim.make_initial_population(n=popsize)
    # Run the simulation
    sim.simulate(end_date=end_date)
    # check the datatypes
    check_dtypes(sim)


@pytest.mark.slow
def test_run_health_system_events_wont_run(seed):
    """
    Test the model with no service availability
    """
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules, make service availability = []
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )
    # make initial population
    sim.make_initial_population(n=popsize)
    # Run the simulation
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


@pytest.mark.slow
def test_sim_high_incidence(seed):
    """
    Run the model with a high incidence, where many people are involved in road traffic injuries
    :return:
    """
    # create the simulation object
    sim = create_basic_rti_sim(popsize, seed)
    # get rti module parameters
    params = sim.modules['RTI'].parameters
    # get the original incidence
    orig_inc = params['base_rate_injrti']
    # incrase simulation incidence
    params['base_rate_injrti'] = orig_inc * 100
    # run simulation
    sim.simulate(end_date=end_date)
    # check datatypes
    check_dtypes(sim)


@pytest.mark.slow
def test_tiny_population(seed):
    """
    Run the model with a small population size
    :return:
    """
    # create simulation with a population size of 2
    sim = create_basic_rti_sim(2, seed)
    # run simulation
    sim.simulate(end_date=end_date)
    # check datatypes
    check_dtypes(sim)


@pytest.mark.slow
def test_no_capabilities(seed):
    """
    Run the model with a capabilities coefficient of 0.0
    :return:
    """
    # Register the core modules, make capabilities coefficient = 0.0
    sim = Simulation(start_date=start_date, seed=seed)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )
    # make initial population
    sim.make_initial_population(n=popsize)
    # Run the simulation
    sim.simulate(end_date=end_date)
    # check the datatypes
    check_dtypes(sim)


@pytest.mark.slow
def test_health_system_disabled(seed):
    """
    Test the model with the health system disabled
    :return:
    """
    # create simulation object
    sim = Simulation(start_date=start_date, seed=seed)
    # get resource file path
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    # register modules, health system is disabled
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )
    # make the initial population
    sim.make_initial_population(n=popsize)
    # run the simulation
    sim.simulate(end_date=end_date)
    # check the datatypes
    check_dtypes(sim)


if __name__ == '__main__':
    t0 = time.time()
    test_run()
    t1 = time.time()
    print('Time taken', t1 - t0)
