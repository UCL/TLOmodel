import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200


# Simply test whether the system runs under multiple configurations of the healthsystem
# NB. Running the dummy Mockitits and ChronicSyndrome modules test all aspects of the healthsystem module.

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_using_parameter_or_argument_to_set_service_availability():
    """
    Check that can set service_availability through argument or through parameter.
    Should be equal to what is specified by the parameter, but overwrite with what was provided in arguement if an
    argument was specified -- provided for backward compatibility.)
    """

    # No specification with argument --> everything is available
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath)
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert sim.modules['HealthSystem'].service_availability == ['*']

    # Editing parameters --> that is reflected in what is used
    sim = Simulation(start_date=start_date, seed=0)
    service_availability_params = ['HSI_that_begin_with_A*', 'HSI_that_begin_with_B*']
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath)
    )
    sim.modules['HealthSystem'].parameters['Service_Availability'] = service_availability_params
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert sim.modules['HealthSystem'].service_availability == service_availability_params

    # Editing parameters, but with an argument provided to module --> argument over-writes parameter edits
    sim = Simulation(start_date=start_date, seed=0)
    service_availability_arg = ['HSI_that_begin_with_C*']
    service_availability_params = ['HSI_that_begin_with_A*', 'HSI_that_begin_with_B*']
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability_arg)
    )
    sim.modules['HealthSystem'].parameters['Service_Availability'] = service_availability_params
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert sim.modules['HealthSystem'].service_availability == service_availability_arg


def test_run_with_healthsystem_no_disease_modules_defined():
    sim = Simulation(start_date=start_date, seed=0)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_no_interventions_allowed(tmpdir):
    # There should be no events run or scheduled

    # Establish the simulation object
    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {"*": logging.INFO}
    }
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # Get ready for temporary log-file
    # Define the service availability as null
    service_availability = []

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome(),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks for the healthsystem
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0.0).all()
    assert len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE) == 0

    # Do the checks for the symptom manager: some symptoms should be registered
    assert sim.population.props.loc[:, sim.population.props.columns.str.startswith('sy_')] \
        .apply(lambda x: x != set()).any().any()
    assert (sim.population.props.loc[:, sim.population.props.columns.str.startswith('sy_')].dtypes == 'int64').all()
    assert not pd.isnull(sim.population.props.loc[:, sim.population.props.columns.str.startswith('sy_')]).any().any()

    # Check that no one was cured of mockitis:
    assert not any(sim.population.props['mi_status'] == 'P')  # No cures


def test_run_in_mode_0_with_capacity(tmpdir):
    # Events should run and there be no squeeze factors
    # (Mode 0 -> No Constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=0),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome(),
                 )

    # Set the availability of consumables to 50% for everything
    sim.modules['HealthSystem'].prob_item_codes_available.loc[:, :] = 0.5

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks for health system appts
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that at least some consumables requests fail due to lack of availability
    items_not_available = [
        v for v in output['tlo.methods.healthsystem']['Consumables']['Item_NotAvailable'] if v != '{}'
    ]
    pkgs_not_available = [
        v for v in output['tlo.methods.healthsystem']['Consumables']['Package_NotAvailable'] if v != '{}'
    ]
    assert 0 < len(items_not_available + pkgs_not_available)

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


def test_run_in_mode_0_no_capacity(tmpdir):
    # Every events should run (no did_not_run) and no squeeze factors
    # (Mode 0 -> No Constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=0),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome(),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath)
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


def test_run_in_mode_1_with_capacity(tmpdir):
    # All events should run with some zero squeeze factors
    # (Mode 1 -> elastic constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=1),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


def test_run_in_mode_1_with_no_capacity(tmpdir):
    # Events should run but with high squeeze factors
    # (Mode 1 -> elastic constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=1),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert hsi_events['did_run'].all()
    assert (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'Squeeze_Factor'] == 100.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


def test_run_in_mode_2_with_capacity(tmpdir):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.group2
def test_run_in_mode_2_with_no_capacity(tmpdir):
    # No individual level events should run and the log should contain events with a flag showing that all individual
    # events did not run. Population level events should have run.
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation, manually setting smaller values to decrease runtime (logfile size)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert not (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'did_run'].astype(bool)).any()  # not any Individual level
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'did_run']).astype(bool).all()  # all Population level events
    assert pd.isnull(sim.population.props['mi_date_cure']).all()  # No cures of mockitis occurring

    # Check that no mockitis cured occured (though health system)
    assert not any(sim.population.props['mi_status'] == 'P')


def test_run_in_mode_0_with_capacity_ignoring_cons_constraints(tmpdir):
    # Events should run and there be no squeeze factors
    # (Mode 0 -> No Constraints)
    # Ignoring consumables constraints --> all requests for consumables granted

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=0,
                                           ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks for the consumables: all requests granted and nothing in NotAvailable
    assert 0 == len([v for v in output['tlo.methods.healthsystem']['Consumables']['Item_NotAvailable'] if v != '{}'])
    assert 0 == len([v for v in output['tlo.methods.healthsystem']['Consumables']['Package_NotAvailable'] if v != '{}'])
    assert 0 < len([v for v in output['tlo.methods.healthsystem']['Consumables']['Item_Available'] if v != '{}'])
    assert 0 < len([v for v in output['tlo.methods.healthsystem']['Consumables']['Package_Available'] if v != '{}'])

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.group2
def test_run_in_with_hs_disabled(tmpdir):
    # All events should run but no logging from healthsystem

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=2000)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    assert 'HSI_Event' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert 'Consumables' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert 'Capacity' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert not pd.isnull(sim.population.props['mi_date_cure']).all()  # At least some cures occurred (through HS)
    assert any(sim.population.props['mi_status'] == 'P')  # At least some mockitis cure have occurred (though HS)

    # Check for hsi_wrappers in the main event queue
    list_of_ev_name = [ev[2] for ev in sim.event_queue.queue]
    assert any(['HSIEventWrapper' in str(ev_name) for ev_name in list_of_ev_name])


def test_run_in_mode_2_with_capacity_with_health_seeking_behaviour(tmpdir):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "log", "directory": tmpdir})

    # Define the service availability
    service_availability = ['*']

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the check for the occurance of the GenericFirstAppt which is created by the HSB module
    assert 'GenericFirstApptAtFacilityLevel1' in output['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'].values

    # Check that some mockitis cured occured (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


def test_use_of_helper_function_get_all_consumables():
    """Test that the helper function 'get_all_consumables' in the base class of the HSI works as expected."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create simulation with the healthsystem and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
        # Disable sorting + checks to avoid error due to missing dependencies
        sort_modules=False,
        check_all_dependencies=False
    )

    # Define availability of items
    item_code_is_available = [0, 1, 2, 3]
    item_code_not_available = [4, 5, 6, 7]
    pkg_code_is_available = [1, 2]
    pkg_code_not_available = [3, 4, 5]

    # Edit the item-package lookup-table to create packages that will be available or not
    lookup = sim.modules['HealthSystem'].parameters['Consumables']
    lookup['Intervention_Pkg_Code'] = -99
    lookup.loc[item_code_is_available[0], 'Intervention_Pkg_Code'] = pkg_code_is_available[0]
    lookup.loc[item_code_is_available[1:3], 'Intervention_Pkg_Code'] = pkg_code_is_available[1]
    lookup.loc[item_code_not_available[0], 'Intervention_Pkg_Code'] = pkg_code_not_available[0]
    lookup.loc[item_code_not_available[1:3], 'Intervention_Pkg_Code'] = pkg_code_not_available[1]
    lookup.loc[[item_code_is_available[3], item_code_not_available[3]], 'Intervention_Pkg_Code'] = \
        pkg_code_not_available[2]

    # Process consumables file
    hs = sim.modules['HealthSystem']
    hs.process_consumables_file()

    # Manually edit availability probabilities to force some items to be (not) available
    hs.prob_item_codes_available.loc[item_code_is_available] = 1
    hs.prob_item_codes_available.loc[item_code_not_available] = 0

    # Update availability ready for calling
    hs.determine_availability_of_consumables_today()

    # Create a dummy HSI event:
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = module.sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = 0
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    hsi_event = HSI_Dummy(module=sim.modules['DummyModule'], person_id=0)

    # Test using item_codes:
    assert True is hsi_event.get_all_consumables(item_codes=item_code_is_available[0])
    assert True is hsi_event.get_all_consumables(item_codes=[item_code_is_available[0]])
    assert True is hsi_event.get_all_consumables(item_codes=item_code_is_available)
    assert False is hsi_event.get_all_consumables(item_codes=item_code_not_available)
    assert False is hsi_event.get_all_consumables(item_codes=[
        item_code_is_available[0], item_code_not_available[0]])
    assert False is hsi_event.get_all_consumables(item_codes=[
        item_code_not_available[0], item_code_is_available[0]])

    # Test using pkg_codes:
    assert True is hsi_event.get_all_consumables(pkg_codes=pkg_code_is_available[0])
    assert True is hsi_event.get_all_consumables(pkg_codes=[pkg_code_is_available[0]])
    assert True is hsi_event.get_all_consumables(pkg_codes=pkg_code_is_available)
    assert False is hsi_event.get_all_consumables(pkg_codes=pkg_code_not_available[2])
    assert False is hsi_event.get_all_consumables(pkg_codes=pkg_code_not_available)
    assert False is hsi_event.get_all_consumables(pkg_codes=[
        pkg_code_is_available[0], pkg_code_not_available[0]])
    assert False is hsi_event.get_all_consumables(pkg_codes=[
        pkg_code_is_available[0], pkg_code_not_available[2]])

    # Test using item_codes and pkg_codes:
    assert True is hsi_event.get_all_consumables(item_codes=item_code_is_available, pkg_codes=pkg_code_is_available)
    assert False is hsi_event.get_all_consumables(item_codes=item_code_not_available, pkg_codes=pkg_code_is_available)
    assert False is hsi_event.get_all_consumables(item_codes=item_code_is_available, pkg_codes=pkg_code_not_available)

    # Test using footprints:
    assert True is hsi_event.get_all_consumables(footprint={
        'Item_Code': {item_code_is_available[0]: 10, item_code_is_available[1]: 20},
        'Intervention_Package_Code': {pkg_code_is_available[0]: 1, pkg_code_is_available[1]: 10}
    })
    assert False is hsi_event.get_all_consumables(footprint={
        'Item_Code': {item_code_not_available[0]: 1, item_code_is_available[1]: 20},
        'Intervention_Package_Code': {pkg_code_is_available[0]: 1, pkg_code_is_available[1]: 10}
    })


def test_speeding_up_request_consumables():

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = 0
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the healthsystem and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date)

    hs = sim.modules['HealthSystem']
    hsi_event = HSI_Dummy(module=sim.modules['DummyModule'], person_id=0)

    # refresh availabilities of consumables to all available:
    hs.determine_availability_of_consumables_today()

    # create a very complicated footprint:
    consumables = hs.parameters['Consumables']
    pkg_code = \
        pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'ORS',
                            'Intervention_Pkg_Code'])[0]

    cons_req_as_footprint = {
        'Intervention_Package_Code': {pkg_code: 2, 3: 1, 5: 1},
        'Item_Code': {99: 2, 98: 4, 97: 6}
    }

    # test running request_consumables
    _ = hs.request_consumables(cons_req_as_footprint=cons_req_as_footprint, hsi_event=hsi_event)

    # Check it all works with footprints with different configurations
    _ = hs.request_consumables(cons_req_as_footprint={
        'Intervention_Package_Code': {},
        'Item_Code': {}},
        hsi_event=hsi_event
    )

    _ = hs.request_consumables(cons_req_as_footprint={
        'Intervention_Package_Code': {1: 10},
        'Item_Code': {}},
        hsi_event=hsi_event
    )

    _ = hs.request_consumables(cons_req_as_footprint={
        'Intervention_Package_Code': {},
        'Item_Code': {1: 10}},
        hsi_event=hsi_event
    )

    # Time it!
    import time
    start = time.time()
    # do a one-time update (as would happen each day in the simulation)
    hs.determine_availability_of_consumables_today()

    # run the whole "request_consumables" routine many times to capture a time that it takes:
    # - no logging
    for i in range(1000):
        _ = hs.request_consumables(cons_req_as_footprint=cons_req_as_footprint, hsi_event=hsi_event, to_log=False)
    end = time.time()
    print(f"Elapsed time for 1000 X request_consumables (no logging): {end - start}")
    # with original code: elapsed time = 13.770344972610474
    # first version edit: elapsed time = 5.978830099105835
    # second version edit: elapsed time = 5.255127191543579
    # third version edit: elapsed time = 20.793461084365845
    # fourth version (with pre-computing): 1.0886831283569336
    # fifth version (allowing asserts): 1.1119129657745361

    # - with logging
    for i in range(1000):
        _ = hs.request_consumables(cons_req_as_footprint=cons_req_as_footprint, hsi_event=hsi_event, to_log=True)
    end = time.time()
    print(f"Elapsed time for 1000 X request_consumables (with logging): {end - start}")
    # with original code: elapsed time = 13.770344972610474
    # fifth version (allowing asserts): 2.5592801570892334

    # check functionality of helper function getting consumables as individual items
    start = time.time()
    for i in range(1000):
        _ = hs.get_consumables_as_individual_items(cons_req_as_footprint=cons_req_as_footprint)
    end = time.time()
    print(f"Elapsed time for 1000 X get_consumables_as_individual_items: {end - start}")
    # with looping through dict: elapsed time = 2.2378311157226562
    # with pandas manipulations: 16.766106843948364


def check_bed_days_basics(hs_disable):
    """Check all the basic functionality about bed-days footprints and capacity management by the health-system"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI event with No Bed-days specified
    class HSI_Dummy_NoBedDaysSpec(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 1
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({
                'high_dependency_bed': 10,
                'general_bed': 5
            })

        def apply(self, person_id, squeeze_factor):
            print(f'squeeze-factor is {squeeze_factor}')
            print(f'Bed-days allocated to this event: {self.bed_days_allocated_to_this_event}')

    # Create simulation with the healthsystem and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.end_date = start_date + pd.DateOffset(days=100)
    hs = sim.modules['HealthSystem']

    # 1) Check that HSI_Event come with correctly formatted bed-days footprints, whether explicitly defined or not.
    #  No bed-days specified
    person_id = 0
    hsi_nobd = HSI_Dummy_NoBedDaysSpec(module=sim.modules['DummyModule'], person_id=person_id)
    hs.check_beddays_footrpint_format(hsi_nobd.BEDDAYS_FOOTPRINT)

    # Bed-days specified
    hsi_bd = HSI_Dummy(module=sim.modules['DummyModule'], person_id=person_id)
    hs.check_beddays_footrpint_format(hsi_bd.BEDDAYS_FOOTPRINT)

    # 2) Check that helper-function to make footprints works as expected:
    assert {'non_bed_space': 0, 'general_bed': 0, 'high_dependency_bed': 0} \
        == hsi_nobd.make_beddays_footprint({})
    assert {'non_bed_space': 0, 'general_bed': 4, 'high_dependency_bed': 1} \
        == hsi_nobd.make_beddays_footprint({'general_bed': 4, 'high_dependency_bed': 1})

    # 2) Check that can schedule an HSI with a bed-day footprint
    hs.schedule_hsi_event(hsi_event=hsi_nobd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)
    hs.schedule_hsi_event(hsi_event=hsi_bd, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1), priority=0)

    # 3) Check that HSI can be run by the health system and correctly report the number of bed-days provided:
    #  - if the health-system does update the '_received_info_about_bed_days' property:
    info_sent_to_hsi = {k: int(hsi_bd.BEDDAYS_FOOTPRINT[k] * 0.5) for k in hsi_bd.BEDDAYS_FOOTPRINT}
    hsi_bd._received_info_about_bed_days = info_sent_to_hsi
    hsi_bd.apply(person_id=0, squeeze_factor=0.0)
    assert info_sent_to_hsi == hsi_bd.bed_days_allocated_to_this_event

    #  - confirm that if the `_received_info_about_bed_days` is not written to, it defaults to the full footprint
    # (this it what happens when the event is from inside the HSIEventWrapper)
    hsi_bd_a = HSI_Dummy(module=sim.modules['DummyModule'], person_id=0)
    hsi_bd_a.apply(person_id=0, squeeze_factor=0.0)
    assert hsi_bd_a.bed_days_allocated_to_this_event == hsi_bd_a.BEDDAYS_FOOTPRINT
    assert hsi_bd_a.is_all_beddays_allocated()

    # 4) Check that footprint can be correctly recorded in the tracker:
    # - if the days fall safely inside the period of the simulation
    hs.initialise_beddays_tracker()

    # store copy of the original tracker
    import copy
    orig = copy.deepcopy(hs.bed_tracker)

    # impose the footprint:
    sim.date += pd.DateOffset(days=5)
    df = sim.population.props
    assert not df.at[person_id, 'hs_is_inpatient']
    hs.impose_beddays_footprint(hsi_bd)
    assert df.at[person_id, 'hs_is_inpatient']  # should be flagged as in-patient

    # check imposition works:
    footprint = hsi_bd.BEDDAYS_FOOTPRINT
    the_facility_name = 'National Hospital'

    diff = pd.DataFrame()
    for bed_type in hsi_bd.BEDDAYS_FOOTPRINT:
        diff[bed_type] = - (
            hs.bed_tracker[bed_type].loc[:, the_facility_name] - orig[bed_type].loc[:, the_facility_name]
        )

    first_day = diff[diff.sum(axis=1) > 0].index.min()
    last_day = diff[diff.sum(axis=1) > 0].index.max()

    assert diff.sum().sum() == sum(footprint.values())
    assert (diff.sum(axis=1) <= 1).all()
    assert first_day == sim.date
    assert last_day == sim.date + pd.DateOffset(days=sum(footprint.values()) - 1)
    assert (1 == diff.loc[
           (diff.index >= first_day) &
           (diff.index <= last_day)].sum(axis=1)).all()

    # check that beds timed to be used in the order specified (descending order of intensiveness):
    for i, bed_type in enumerate(sim.modules['HealthSystem'].bed_types):
        d = diff[diff.columns[i]]
        this_bed_type_starts_on = d.loc[d > 0].index.min()
        if i > 0:
            d_last_bed_type = diff[diff.columns[i-1]]
            last_bed_type_ends_on = d_last_bed_type.loc[d_last_bed_type > 0].index.max()
            if not (pd.isnull(last_bed_type_ends_on) or pd.isnull(this_bed_type_starts_on)):
                assert this_bed_type_starts_on > last_bed_type_ends_on

    # - if the days extend beyond the period of the simulation:
    hs.initialise_beddays_tracker()

    # store copy of the original tracker
    orig = copy.deepcopy(hs.bed_tracker)

    # impose the footprint (that will extend past end of the simulation): should not error and should not extend df
    sim.date = sim.end_date - pd.DateOffset(days=1)
    hs.impose_beddays_footprint(hsi_bd)

    # check that additional columns have not been added
    for bed_type in hs.bed_tracker:
        assert all(orig[bed_type].columns == hs.bed_tracker[bed_type].columns)

    # tracker should show only the 2 days in the high-dependency bed that occur before end of simulation
    assert orig['general_bed'].equals(hs.bed_tracker['general_bed'])
    assert orig['non_bed_space'].equals(hs.bed_tracker['non_bed_space'])
    assert all(
        [0] * 99 + [1] * 2 == (
            orig['high_dependency_bed'].loc[:, the_facility_name] -
            hs.bed_tracker['high_dependency_bed'].loc[:, the_facility_name]
        ).values
    )


def check_bed_days_property_is_inpatient(hs_disable):
    """Check that the is_inpatient property is controlled correctly and kept in sync with the bed-tracker"""

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date
            )
            self.in_patient_status = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.end_date),
                columns=[0, 1, 2],
                data=False
            )

            # Schedule person_id=0 to attend care on day 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)
            # Schedule person_id=1 to attend care on day 5
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=self.sim.date + pd.DateOffset(days=5),
                tclose=None,
                priority=0)

            # Schedule person_id=2 to attend care on day 12, and then again on day 14 [overlapping in-patient durations]
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=self.sim.date + pd.DateOffset(days=12),
                tclose=None,
                priority=0)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=2),
                topen=self.sim.date + pd.DateOffset(days=14),
                tclose=None,
                priority=0)

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            self.module.in_patient_status.loc[self.sim.date.normalize()] = \
                population.props.loc[[0, 1, 2], 'hs_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=20))

    # check that the daily checks on 'is_inpatient' are as expected:
    assert all([False] * 2 + [True] * 5 + [False] * 14 ==
               sim.modules['DummyModule'].in_patient_status[0].values
               )
    assert all([False] * 5 + [True] * 5 + [False] * 11 ==
               sim.modules['DummyModule'].in_patient_status[1].values
               )
    assert all([False] * 12 + [True] * 7 + [False] * 2 ==
               sim.modules['DummyModule'].in_patient_status[2].values
               )

    # check that in-patient status is consistent with recorded usage of beds
    tot_time_as_in_patient = sim.modules['DummyModule'].in_patient_status.sum(axis=1)
    tracker = sim.modules['HealthSystem'].bed_tracker['general_bed']
    beds_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)
    assert (beds_occupied == tot_time_as_in_patient).all()

    check_dtypes(sim)


def check_bed_days_released_on_death(hs_disable):
    """Check that bed-days scheduled to be occupied are released upon the death of the person"""

    class DummyModule(Module):
        METADATA = {
            Metadata.DISEASE_MODULE,
            Metadata.USES_HEALTHSYSTEM
        }

        CAUSES_OF_DEATH = {
            'death_from_dummy_module': Cause(label='DummyModule'),
        }

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule event that will query the status of the property 'is_inpatient' each day
            self.sim.schedule_event(
                QueryInPatientStatus(self),
                self.sim.date
            )
            self.in_patient_status = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.end_date),
                columns=[0, 1],
                data=False
            )

            # Schedule person_id=0 and person_id=1 to attend care on day 2 for 10 days
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Dummy(self, person_id=1),
                topen=self.sim.date + pd.DateOffset(days=2),
                tclose=None,
                priority=0)

            # Schedule person_id=0 to die on day 5
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.sim.modules['Demography'], 0, 'death_from_dummy_module'),
                self.sim.date + pd.DateOffset(days=5)
            )

    class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            self.module.in_patient_status.loc[self.sim.date.normalize()] = \
                population.props.loc[[0, 1], 'hs_is_inpatient'].values

    # Create a dummy HSI with both-types of Bed Day specified
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = 2
            self.ALERT_OTHER_DISEASES = []
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 10})

        def apply(self, person_id, squeeze_factor):
            pass

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=0)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=hs_disable),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=20))

    # Test that all bed-days released when person dies
    assert not sim.population.props.at[0, 'is_alive']   # person 0 has died
    assert sim.population.props.at[1, 'is_alive']   # person 1 is alive

    tracker = sim.modules['HealthSystem'].bed_tracker['general_bed']
    bed_occupied = tracker.sum(axis=1)[0] - tracker.sum(axis=1)
    assert all([0] * 2 + [2] * 3 + [1] * 7 + [0] * 9 == bed_occupied.values)


def test_bed_days_if_healthsystem_not_disabled():
    check_bed_days_basics(hs_disable=False)
    check_bed_days_property_is_inpatient(hs_disable=False)
    check_bed_days_released_on_death(hs_disable=False)


def test_bed_days_if_healthsystem_is_disabled():
    check_bed_days_basics(hs_disable=True)
    check_bed_days_property_is_inpatient(hs_disable=True)
    check_bed_days_released_on_death(hs_disable=True)
