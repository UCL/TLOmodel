import os
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.consumables import Consumables, create_dummy_data_for_cons_availability
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthsystem import HealthSystem, HealthSystemChangeParameters, HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

"""
Test whether the system runs under multiple configurations of the healthsystem. (Running the dummy Mockitits and
ChronicSyndrome modules is intended to test all aspects of the healthsystem module.)
"""


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_using_parameter_or_argument_to_set_service_availability(seed):
    """
    Check that can set service_availability through argument or through parameter.
    Should be equal to what is specified by the parameter, but overwrite with what was provided in argument if an
    argument was specified -- provided for backward compatibility.)
    """

    # No specification with argument --> everything is available
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath)
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=0))
    assert sim.modules['HealthSystem'].service_availability == ['*']

    # Editing parameters --> that is reflected in what is used
    sim = Simulation(start_date=start_date, seed=seed)
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
    sim = Simulation(start_date=start_date, seed=seed)
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


@pytest.mark.slow
def test_run_with_healthsystem_no_disease_modules_defined(seed):
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


@pytest.mark.slow
def test_run_no_interventions_allowed(tmpdir, seed):
    # There should be no events run or scheduled

    # Establish the simulation object
    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {"*": logging.INFO}
    }
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

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


@pytest.mark.slow
def test_run_in_mode_0_with_capacity(tmpdir, seed):
    # Events should run and there be no squeeze factors
    # (Mode 0 -> No Constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome(),
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks for health system appts
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some Mockitis cured occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
def test_run_in_mode_0_no_capacity(tmpdir, seed):
    # Every events should run (no did_not_run) and no squeeze factors
    # (Mode 0 -> No Constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome(),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath)
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
def test_run_in_mode_1_with_capacity(tmpdir, seed):
    # All events should run with some zero squeeze factors
    # (Mode 1 -> elastic constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
def test_run_in_mode_1_with_no_capacity(tmpdir, seed):
    # Events should run but with high squeeze factors
    # (Mode 1 -> elastic constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert hsi_events['did_run'].all()
    assert (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'Squeeze_Factor'] == 100.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'Squeeze_Factor'] == 0.0).all()

    # Check that some Mockitis cures occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
def test_run_in_mode_2_with_capacity(tmpdir, seed):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()

    # Check that some Mockitis cures occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
@pytest.mark.group2
def test_run_in_mode_2_with_no_capacity(tmpdir, seed):
    # No individual level events should run and the log should contain events with a flag showing that all individual
    # events did not run. Population level events should have run.
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation, manually setting smaller values to decrease runtime (logfile size)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert not (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'did_run'].astype(bool)).any()  # not any Individual level
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'did_run']).astype(bool).all()  # all Population level events
    assert pd.isnull(sim.population.props['mi_date_cure']).all()  # No cures of mockitis occurring

    # Check that no Mockitis cures occurred (though health system)
    assert not any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
@pytest.mark.group2
def test_run_in_with_hs_disabled(tmpdir, seed):
    # All events should run but no logging from healthsystem

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=2000)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the checks
    assert 'HSI_Event' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert 'Consumables' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert 'Capacity' not in output['tlo.methods.healthsystem']  # HealthSystem no logging
    assert not pd.isnull(sim.population.props['mi_date_cure']).all()  # At least some cures occurred (through HS)
    assert any(sim.population.props['mi_status'] == 'P')  # At least some mockitis cure have occurred (though HS)

    # Check for hsi_wrappers in the main event queue
    list_of_ev_name = [ev[3] for ev in sim.event_queue.queue]
    assert any(['HSIEventWrapper' in str(ev_name) for ev_name in list_of_ev_name])


@pytest.mark.slow
def test_run_in_mode_2_with_capacity_with_health_seeking_behaviour(tmpdir, seed):
    # All events should run
    # (Mode 2 -> hard constraints)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

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
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome()
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Do the check for the occurrence of the GenericFirstAppt which is created by the HSB module
    assert 'FirstAttendance_NonEmergency' in output['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'].values

    # Check that some mockitis cured occurred (though health system)
    assert any(sim.population.props['mi_status'] == 'P')


@pytest.mark.slow
def test_all_appt_types_can_run(seed):
    """Check that if an appointment type is declared as one that can run at a facility-type of level `x` that it can
    run at the level for persons in any district."""

    # Create Dummy Module to host the HSI
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI event class
    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, appt_type, level):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'DummyHSIEvent'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({appt_type: 1})
            self.ACCEPTED_FACILITY_LEVEL = level

            self.this_hsi_event_ran = False

        def apply(self, person_id, squeeze_factor):
            if (squeeze_factor != 99) and (squeeze_factor != np.inf):  # todo - this changed!
                # Check that this appointment is being run and run not with a squeeze_factor that signifies that a cadre
                # is not at all available.
                self.this_hsi_event_ran = True

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules and simulate for 0 days
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=1,
                                           use_funded_or_actual_staffing='funded_plus'),
                 # <-- hard constraint (only HSI events with no squeeze factor can run)
                 # <-- using the 'funded_plus' number/distribution of officers
                 DummyModule()
                 )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)

    # Get pointer to the HealthSystemScheduler event
    healthsystemscheduler = sim.modules['HealthSystem'].healthsystemscheduler

    # Get the table showing which types of appointment can occur at which level
    appt_types_offered = sim.modules['HealthSystem'].parameters['Appt_Offered_By_Facility_Level'].set_index(
        'Appt_Type_Code')

    # Get the all the districts in which a person could be resident, and allocate one person to each district
    person_for_district = {d: i for i, d in enumerate(sim.population.props['district_of_residence'].cat.categories)}
    sim.population.props.loc[person_for_district.values(), 'is_alive'] = True
    sim.population.props.loc[person_for_district.values(), 'district_of_residence'] = list(person_for_district.keys())

    # For each type of appointment, for a person in each district, create the HSI, schedule the HSI and check it runs
    error_msg = list()

    def check_appt_works(district, level, appt_type):
        sim.modules['HealthSystem'].reset_queue()

        hsi = DummyHSIEvent(module=sim.modules['DummyModule'],
                            person_id=person_for_district[district],
                            appt_type=appt_type,
                            level=level)

        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi,
            topen=sim.date,
            tclose=sim.date + pd.DateOffset(days=1),
            priority=1
        )

        healthsystemscheduler.apply(sim.population)

        if not hsi.this_hsi_event_ran:
            return False
        else:
            return True

    for _district in person_for_district:
        for _facility_level_col_name in appt_types_offered.columns:
            for _appt_type in appt_types_offered[_facility_level_col_name].loc[
                appt_types_offered[_facility_level_col_name]
            ].index:
                _level = _facility_level_col_name.split('_')[-1]
                if not check_appt_works(district=_district, level=_level, appt_type=_appt_type):
                    error_msg.append(f"The HSI did not run: "
                                     f"level={_level}, appt_type={_appt_type}, district={_district}")

    if len(error_msg):
        for _line in error_msg:
            print(_line)

    assert 0 == len(error_msg)


@pytest.mark.slow
def test_two_loggers_in_healthsystem(seed, tmpdir):
    """Check that two different loggers used by the HealthSystem for more/less detailed logged information are
    consistent with one another."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.modules['HealthSystem'].schedule_hsi_event(HSI_Dummy(self, person_id=0),
                                                           topen=self.sim.date,
                                                           tclose=None,
                                                           priority=0)

    # Create a dummy HSI event:
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'Under5OPD': 1})
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})
            self.ACCEPTED_FACILITY_LEVEL = '1a'

        def apply(self, person_id, squeeze_factor):
            # Request a consumable (either 0 or 1)
            self.get_consumables(item_codes=self.module.rng.choice((0, 1), p=(0.5, 0.5)))

            # Schedule another occurrence of itself in three days.
            sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                           topen=self.sim.date + pd.DateOffset(days=3),
                                                           tclose=None,
                                                           priority=0)

    # Set up simulation:
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'tmpfile',
        'directory': tmpdir,
        'custom_levels': {
            "tlo.methods.healthsystem": logging.INFO,
            "tlo.methods.healthsystem.summary": logging.INFO
        }
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False
    )
    sim.make_initial_population(n=1000)

    # Replace consumables class with version that declares only one consumable, available with probability 0.5
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    all_fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)

    sim.modules['HealthSystem'].consumables = Consumables(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability={0: 0.5, 1: 0.5},
            months=list(range(1, 13)),
            facility_ids=list(all_fac_ids)),
        rng=sim.modules['HealthSystem'].rng,
        availability='default'
    )

    sim.simulate(end_date=start_date + pd.DateOffset(years=2))
    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Standard log:
    detailed_hsi_event = log["tlo.methods.healthsystem"]['HSI_Event']
    detailed_capacity = log["tlo.methods.healthsystem"]['Capacity']
    detailed_consumables = log["tlo.methods.healthsystem"]['Consumables']

    assert {'date', 'TREATMENT_ID', 'did_run', 'Squeeze_Factor', 'Number_By_Appt_Type_Code', 'Person_ID',
            'Facility_Level', 'Facility_ID',
            } == set(detailed_hsi_event.columns)
    assert {'date', 'Frac_Time_Used_Overall', 'Frac_Time_Used_By_Facility_ID', 'Frac_Time_Used_By_OfficerType',
            } == set(detailed_capacity.columns)
    assert {'date', 'TREATMENT_ID', 'Item_Available', 'Item_NotAvailable'
            } == set(detailed_consumables.columns)

    bed_types = sim.modules['HealthSystem'].bed_days.bed_types
    detailed_beddays = {bed_type: log["tlo.methods.healthsystem"][f"bed_tracker_{bed_type}"] for bed_type in bed_types}

    # Summary log:
    summary_hsi_event = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
    summary_capacity = log["tlo.methods.healthsystem.summary"]["Capacity"]
    summary_consumables = log["tlo.methods.healthsystem.summary"]["Consumables"]
    summary_beddays = log["tlo.methods.healthsystem.summary"]["BedDays"]

    # Check correspondence between the two logs
    #  - Counts of TREATMENT_ID (total over entire period of log)
    assert summary_hsi_event['TREATMENT_ID'].apply(pd.Series).sum().to_dict() == \
           detailed_hsi_event.groupby('TREATMENT_ID').size().to_dict()

    #  - Appointments (total over entire period of the log)
    assert summary_hsi_event['Number_By_Appt_Type_Code'].apply(pd.Series).sum().to_dict() == \
        detailed_hsi_event['Number_By_Appt_Type_Code'].apply(pd.Series).sum().to_dict()

    #  - Average fraction of HCW time used (year by year)
    assert summary_capacity.set_index(pd.to_datetime(summary_capacity.date).dt.year
                                      )['average_Frac_Time_Used_Overall'].round(4).to_dict() == \
        detailed_capacity.set_index(pd.to_datetime(detailed_capacity.date).dt.year
                                    )['Frac_Time_Used_Overall'].groupby(level=0).mean().round(4).to_dict()

    #  - Consumables (total over entire period of log that are available / not available)  # add _Item_
    assert summary_consumables['Item_Available'].apply(pd.Series).sum().to_dict() == \
           detailed_consumables['Item_Available'].apply(
               lambda x: {f'{k}': v for k, v in eval(x).items()}).apply(pd.Series).sum().to_dict()
    assert summary_consumables['Item_NotAvailable'].apply(pd.Series).sum().to_dict() == \
           detailed_consumables['Item_NotAvailable'].apply(
               lambda x: {f'{k}': v for k, v in eval(x).items()}).apply(pd.Series).sum().to_dict()

    #  - Bed-Days (bed-type by bed-type and year by year)
    for _bed_type in bed_types:
        # Detailed:
        tracker = detailed_beddays[_bed_type]\
            .assign(year=pd.to_datetime(detailed_beddays[_bed_type].date).dt.year)\
            .set_index('year')\
            .drop(columns=['date'])\
            .T
        tracker.index = tracker.index.astype(int)
        capacity = sim.modules['HealthSystem'].bed_days._scaled_capacity[_bed_type]
        detail_beddays_used = tracker.sub(capacity, axis=0).mul(-1).sum().groupby(level=0).sum().to_dict()

        # Summary: total bed-days used by year
        summary_beddays_used = summary_beddays\
            .assign(year=pd.to_datetime(summary_beddays.date).dt.year)\
            .set_index('year')[_bed_type]\
            .to_dict()

        assert detail_beddays_used == summary_beddays_used

    # Check the count of appointment type (total) matches the count split by level
    counts_of_appts_by_level = pd.concat(
        {idx: pd.DataFrame.from_dict(mydict)
         for idx, mydict in summary_hsi_event['Number_By_Appt_Type_Code_And_Level'].iteritems()
         }).unstack().fillna(0.0).astype(int)

    assert summary_hsi_event['Number_By_Appt_Type_Code'].apply(pd.Series).sum().to_dict() == \
           counts_of_appts_by_level.sum(axis=1, level=1).sum(axis=0).to_dict()


@pytest.mark.slow
def test_summary_logger_generated_in_year_long_simulation(seed, tmpdir):
    """Check that the summary logger is created when the simulation lasts exactly one year."""

    def summary_logger_is_present(end_date_of_simulation):
        """Returns True if the summary logger is present when using the specified end_date for the simulation."""
        # Create a dummy disease module (to be the parent of the dummy HSI)
        class DummyModule(Module):
            METADATA = {Metadata.DISEASE_MODULE}

            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                sim.modules['HealthSystem'].schedule_hsi_event(HSI_Dummy(self, person_id=0),
                                                               topen=self.sim.date,
                                                               tclose=None,
                                                               priority=0)

        # Create a dummy HSI event:
        class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = 'Dummy'
                self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'Under5OPD': 1})
                self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})
                self.ACCEPTED_FACILITY_LEVEL = '1a'

            def apply(self, person_id, squeeze_factor):
                # Request a consumable (either 0 or 1)
                self.get_consumables(item_codes=self.module.rng.choice((0, 1), p=(0.5, 0.5)))

                # Schedule another occurrence of itself in three days.
                sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                               topen=self.sim.date + pd.DateOffset(days=3),
                                                               tclose=None,
                                                               priority=0)

        # Set up simulation:
        sim = Simulation(start_date=start_date, seed=seed, log_config={
            'filename': 'tmpfile',
            'directory': tmpdir,
            'custom_levels': {
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO
            }
        })

        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
            DummyModule(),
            sort_modules=False,
            check_all_dependencies=False
        )
        sim.make_initial_population(n=1000)

        sim.simulate(end_date=end_date_of_simulation)
        log = parse_log_file(sim.log_filepath)

        return ('tlo.methods.healthsystem.summary' in log) and len(log['tlo.methods.healthsystem.summary'])

    assert summary_logger_is_present(start_date + pd.DateOffset(years=1))


def test_HealthSystemChangeParameters(seed, tmpdir):
    """Check that the event `HealthSystemChangeParameters` can change the internal parameters of the HealthSystem. And
    check that this is effectual in the case of consumables."""

    initial_parameters = {
        'mode_appt_constraints': 0,
        'ignore_priority': False,
        'capabilities_coefficient': 0.5,
        'cons_availability': 'all',
        'beds_availability': 'default',
    }
    new_parameters = {
        'mode_appt_constraints': 2,
        'ignore_priority': True,
        'capabilities_coefficient': 1.0,
        'cons_availability': 'none',
        'beds_availability': 'none',
    }

    class CheckHealthSystemParameters(RegularEvent, PopulationScopeEventMixin):

        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))

        def apply(self, population):
            hs = self.sim.modules['HealthSystem']
            _params = dict()
            _params['mode_appt_constraints'] = hs.mode_appt_constraints
            _params['ignore_priority'] = hs.ignore_priority
            _params['capabilities_coefficient'] = hs.capabilities_coefficient
            _params['cons_availability'] = hs.consumables.cons_availability
            _params['beds_availability'] = hs.bed_days.availability

            logger = logging.getLogger('tlo.methods.healthsystem')
            logger.info(key='CheckHealthSystemParameters', data=_params)

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'Under5OPD': 1})
            self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})
            self.ACCEPTED_FACILITY_LEVEL = '1a'

        def apply(self, person_id, squeeze_factor):
            logger = logging.getLogger('tlo.methods.healthsystem')
            logger.info(key='HSI_Dummy_get_consumables',
                        data=self.get_consumables(item_codes=list(range(100)), return_individual_results=True)
                        )
            sim.modules['HealthSystem'].schedule_hsi_event(self,
                                                           topen=self.sim.date + pd.DateOffset(days=1),
                                                           tclose=None,
                                                           priority=0)

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            hs = sim.modules['HealthSystem']
            sim.schedule_event(CheckHealthSystemParameters(self), sim.date)
            sim.schedule_event(HealthSystemChangeParameters(hs, parameters=new_parameters),
                               sim.date + pd.DateOffset(days=2))
            sim.modules['HealthSystem'].schedule_hsi_event(HSI_Dummy(self, 0), topen=sim.date, tclose=None, priority=0)

    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'tmpfile',
        'directory': tmpdir,
        'custom_levels': {
            "tlo.methods.healthsystem": logging.INFO,
        }
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, **initial_parameters),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=7))

    # Check parameters are changed as expected:
    logged_params = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem'][
        'CheckHealthSystemParameters'].set_index('date')
    assert logged_params.loc[start_date].to_dict() == initial_parameters
    assert logged_params.loc[start_date + pd.DateOffset(days=4)].to_dict() == new_parameters

    logged_access_consumables = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem'][
        'HSI_Dummy_get_consumables'].set_index('date')
    assert logged_access_consumables.loc[start_date].all()  # All consumables available at start
    assert not logged_access_consumables.loc[start_date + pd.DateOffset(days=4)].any()  # No consumables available after
    #                                                                                 parameter change


def test_is_treatment_id_allowed():
    """Check the pattern matching in `is_treatment_id_allowed` works as expected."""
    hs = HealthSystem(resourcefilepath=resourcefilepath)

    # An empty list means nothing is allowed
    assert not hs.is_treatment_id_allowed('Hiv', [])

    # A list that contains only an asteriks ['*'] means run anything
    assert hs.is_treatment_id_allowed('Hiv', ['*'])

    # If the list is not empty, then a treatment_id with a first part "FirstAttendance_" is also allowed
    assert hs.is_treatment_id_allowed('FirstAttendance_Em', ["A_B_C_D_E"])
    assert not hs.is_treatment_id_allowed('FirstAttendance_Em', [])

    # An entry in the list of the form "A_B_C" means a treatment_id that matches exactly is allowed
    assert hs.is_treatment_id_allowed('A', ['A', 'B_C_D', 'E_F_G_H'])
    assert hs.is_treatment_id_allowed('B_C_D', ['A', 'B_C_D', 'E_F_G_H'])

    assert not hs.is_treatment_id_allowed('A_', ['A', 'B_C_D', 'E_F_G_H'])
    assert not hs.is_treatment_id_allowed('E_F_G', ['E', 'E_F', 'E_F_G_H'])

    # An entry in the list of the form "A_B_*" means that a treatment_id that begins "A_B_" or "A_B" is allowed
    assert hs.is_treatment_id_allowed('Hiv_X', ['Hiv_*'])
    assert hs.is_treatment_id_allowed('Hiv_Y', ['Hiv_*'])
    assert hs.is_treatment_id_allowed('Hiv_A_B_C', ['Hiv_A_B_*'])
    assert hs.is_treatment_id_allowed('Hiv_A_B', ['Hiv_A_B_*'])
    assert hs.is_treatment_id_allowed('Hiv_A_B_C_D', ['Hiv_A_B_C_*'])
    assert hs.is_treatment_id_allowed('Hiv_A_B_C', ['Hiv_A_B_C_*'])
    assert hs.is_treatment_id_allowed('Hiv_X_1_2_3_4', ['Hiv_X_*'])
    assert hs.is_treatment_id_allowed('Hiv_X_1_2_3_4', ['Hiv_*'])

    assert not hs.is_treatment_id_allowed('Hiv_X', ['Hiv_A_*'])
    assert not hs.is_treatment_id_allowed('Hiv_Y', ['Y_*'])
    assert not hs.is_treatment_id_allowed('Hiv_A_B_C', ['Hiv_X_B_C_*'])
    assert not hs.is_treatment_id_allowed('Hiv_A_B_C', ['Hiv1_A_B_C_*'])
    assert not hs.is_treatment_id_allowed('Hiv_X_1_2_3_4', ['Hiv_Y_*'])
    assert not hs.is_treatment_id_allowed('A', ['A_B_C_*'])

    # (An asteriks that is not preceded by an "_" has no effect is allowing treatment_ids).
    assert not hs.is_treatment_id_allowed('Hiv_A_B', ['Hiv*'])
    assert not hs.is_treatment_id_allowed('Hiv', ['Hiv*'])

    # (And no confusion about stubs that are similar...)
    assert hs.is_treatment_id_allowed('Epi', ['Epi_*'])
    assert not hs.is_treatment_id_allowed('Epilepsy', ['Epi_*'])
    assert not hs.is_treatment_id_allowed('Epi', ['Epilepsy_*'])
    assert hs.is_treatment_id_allowed('Epilepsy', ['Epilepsy_*'])
    assert hs.is_treatment_id_allowed('Epi', ['Epi', 'Epilepsy_*'])
    assert hs.is_treatment_id_allowed('Epilepsy', ['Epi', 'Epilepsy_*'])


def test_manipulation_of_service_availability(seed, tmpdir):
    """Check that the parameter `service_availability` can be used to allow/disallow certain `TREATMENT_ID`s.
    N.B. This is setting service_availability through a change in parameter, as would be done by BatchRunner."""

    generic_first_appts = {'FirstAttendance_NonEmergency', 'FirstAttendance_Emergency'}

    def get_set_of_treatment_ids_that_run(service_availability) -> Set[str]:
        """Return set of TREATMENT_IDs that occur when running the simulation with the `service_availability`."""
        sim = Simulation(start_date=start_date, seed=seed, log_config={
            'filename': 'tmpfile',
            'directory': tmpdir,
            'custom_levels': {
                "tlo.methods.healthsystem": logging.INFO,
            }
        })

        sim.register(*fullmodel(resourcefilepath=resourcefilepath))
        sim.modules['HealthSystem'].parameters['Service_Availability'] = service_availability  # Change parameter
        sim.make_initial_population(n=500)
        sim.simulate(end_date=start_date + pd.DateOffset(days=7))

        log = parse_log_file(
            sim.log_filepath, level=logging.DEBUG
        )['tlo.methods.healthsystem']
        if 'HSI_Event' in log:
            return set(log['HSI_Event']['TREATMENT_ID'].value_counts().to_dict().keys())
        else:
            return set()

    # Run model with everything available by default using "*"
    everything = get_set_of_treatment_ids_that_run(service_availability=["*"])

    # Run model with everything specified individually
    assert everything == get_set_of_treatment_ids_that_run(service_availability=list(everything))

    # Run model with nothing available
    assert set() == get_set_of_treatment_ids_that_run(service_availability=[])

    # Only allow 'Hiv_Test' (Not `Hiv_Treatment`)
    assert set({'Hiv_Test'}) == \
           get_set_of_treatment_ids_that_run(service_availability=["Hiv_Test_*"]) - generic_first_appts

    # Allow all `Hiv` things (but nothing else)
    assert set({'Hiv_Test', 'Hiv_Treatment'}) == \
           get_set_of_treatment_ids_that_run(service_availability=["Hiv_*"]) - generic_first_appts

    # Allow all except `Hiv_Test`
    everything_except_hiv_test = everything - set({'Hiv_Test'})
    run_everything_except_hiv_test = \
        get_set_of_treatment_ids_that_run(service_availability=list(everything_except_hiv_test))
    assert 'Hiv_Test' not in run_everything_except_hiv_test
    assert len(run_everything_except_hiv_test.union(everything))

    # Allow all except `Hiv_Treatment`
    everything_except_hiv_treatment = everything - set({'Hiv_Treatment'})
    run_everything_except_hiv_treatment = \
        get_set_of_treatment_ids_that_run(service_availability=list(everything_except_hiv_treatment))
    assert 'Hiv_Treatment' not in run_everything_except_hiv_treatment
    assert len(run_everything_except_hiv_treatment.union(everything))

    # Allow all except `HIV*`
    everything_except_hiv_anything = {x for x in everything if not x.startswith('Hiv')}
    run_everything_except_hiv_anything = \
        get_set_of_treatment_ids_that_run(service_availability=list(everything_except_hiv_anything))
    assert 'Hiv_Treatment' not in run_everything_except_hiv_anything
    assert 'Hiv_Test' not in run_everything_except_hiv_anything
    assert len(run_everything_except_hiv_anything.union(everything))


def test_hsi_run_on_same_day_if_scheduled_for_same_day(seed, tmpdir):
    """An HSI_Event which is scheduled for the current day should run on the current day. This should be the case
    whether the HSI_Event is scheduled from initialise_simulation, a normal event, or an HSI_Event."""

    class DummyHSI_To_Run_On_Same_Day(HSI_Event, IndividualScopeEventMixin):
        """HSI event that will demonstrate it has been run."""
        def __init__(self, module, person_id, source):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = f"{self.__class__.__name__ }_{source}"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = '1a'

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyHSI_To_Run_On_First_Day_Of_Simulation(HSI_Event, IndividualScopeEventMixin):
        """HSI event that schedules another HSI_Event for the same day"""
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = self.__class__.__name__
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = '1a'

        def apply(self, person_id, squeeze_factor):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                DummyHSI_To_Run_On_Same_Day(module=self.module, person_id=person_id, source='HSI'),
                topen=self.sim.date,
                tclose=None,
                priority=0)

    class Event_To_Run_On_First_Day_Of_Simulation(Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                DummyHSI_To_Run_On_Same_Day(module=self.module, person_id=person_id, source='Event'),
                topen=self.sim.date,
                tclose=None,
                priority=0)

    class DummyModule(Module):
        """Schedules an HSI to occur on the first day of the simulation from initialise_simulation, and an event that
         will schedule the event for the same day."""

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule the HSI to run on the same day
            sim.modules['HealthSystem'].schedule_hsi_event(
                DummyHSI_To_Run_On_Same_Day(self, person_id=0, source='initialise_simulation'),
                topen=self.sim.date,
                tclose=None,
                priority=0)

            # Schedule an HSI that will schedule a further HSI to run on the same day
            sim.modules['HealthSystem'].schedule_hsi_event(
                DummyHSI_To_Run_On_First_Day_Of_Simulation(module=self, person_id=0),
                topen=self.sim.date,
                tclose=None,
                priority=0)

            # Schedule an event that will schedule an HSI to run on the same day
            sim.schedule_event(Event_To_Run_On_First_Day_Of_Simulation(self, person_id=0), sim.date)

    log_config = {
        "filename": "log",
        "directory": tmpdir,
    }
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable=False,
                     cons_availability='all',
                 ),
                 DummyModule(),
                 check_all_dependencies=False,
                 )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=5))

    # Check that all events ran on the same day, the first day of the simulation.
    log = parse_log_file(
        sim.log_filepath, level=logging.DEBUG
    )['tlo.methods.healthsystem']['HSI_Event']
    assert 4 == len(log)  # 3 HSI events should have occurred
    assert (log['date'] == sim.start_date).all()
    assert log['TREATMENT_ID'].to_list() == [
        'DummyHSI_To_Run_On_Same_Day_initialise_simulation',
        'DummyHSI_To_Run_On_First_Day_Of_Simulation',
        'DummyHSI_To_Run_On_Same_Day_Event',
        'DummyHSI_To_Run_On_Same_Day_HSI',
    ]
