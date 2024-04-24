"""Collection of shared fixtures"""
from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, List

import pytest

from tlo import Date, Module, Simulation
from tlo.methods import (
    demography,
    diarrhoea,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    stunting,
    symptommanager,
)

DEFAULT_SEED = 83563095832589325021

def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help="Seed(s) for simulation-level random number generator in tests",
    )
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option is set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"), scope="session")

## """Fixtures and classes that are to be utilised for sharing simulations across the test framework."""

if TYPE_CHECKING:
    from pandas import DataFrame

@pytest.fixture(scope="session")
def jan_1st_2010() -> Date:
    return Date(2010, 1, 1)


@pytest.fixture(scope="session")
def resource_filepath() -> Path:
    return (Path(os.path.dirname(__file__)) / "../resources").resolve()


@pytest.fixture(scope="session")
def small_shared_sim(seed, jan_1st_2010, resource_filepath):
    """
    Shared simulation object that can be re-used across multiple tests.

    Note that to ensure the object can be shared between tests, it is
    necessary that:
    - All modules that are to make use of the shared simulation,
    and their dependencies, are registered with the simulation.
    - The initial population cannot be changed and must be sufficient
    for all tests in which the object is to be used.
    - The simulation cannot be further simulated into the future.
    """
    sim = Simulation(start_date=jan_1st_2010, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resource_filepath),
        diarrhoea.Diarrhoea(resourcefilepath=resource_filepath, do_checks=True),
        diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resource_filepath),
        healthburden.HealthBurden(resourcefilepath=resource_filepath),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resource_filepath,
            force_any_symptom_to_lead_to_healthcareseeking=True,
        ),
        healthsystem.HealthSystem(
            resourcefilepath=resource_filepath, cons_availability="all"
        ),
        simplified_births.SimplifiedBirths(resourcefilepath=resource_filepath),
        stunting.Stunting(resourcefilepath=resource_filepath),
        symptommanager.SymptomManager(resourcefilepath=resource_filepath),
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)
    return sim


class _BaseSharedSim:
    """
    Base class for creating tests that utilise a shared simulation.
    Module-level tests that want to utilise the shared simulation
    should inherit from this class, and set the "module" attribute
    appropriately.

    This base class also defines a number of safe "setup/teardown"
    fixtures to ensure that the state of the shared simulation is not
    inadvertently altered between tests, creating a knock-on effect.
    If a test needs to alter the state of the simulation; for example
    - Clearing the HSI event queue / event queue
    - Changing module parameters
    - Changing patient details
    then use the fixtures provided to ensure that the original state
    of these objects is restored after the test runs. Then during the
    test, you are free to edit these properties in the shared
    simulation.

    This class also provides several properties for quick access to
    properties of the shared simulation.
    """

    # Set this class-wide variable to be the name
    # of the module that the class will be testing.
    module: str
    # This is how to access the shared simulation resource
    # that tests will automatically hook into.
    sim: Simulation

    @property
    def this_module(self) -> Module:
        """
        Points to the disease module being tested by this class,
        within the shared simulation.
        """
        return self.sim.modules[self.module]

    @property
    def shared_sim_df(self) -> DataFrame:
        """
        Points to the population DataFrame used by the
        shared simulation.

        WARNING: Writes are persistent!
        Use the setup fixture if you intend to make changes to
        the shared DataFrame.
        """
        return self.sim.population.props

    @property
    def shared_sim_healthsystem(self) -> healthsystem.HealthSystem:
        """
        Points to the HealthSystem module if in use by the shared
        simulation.
        """
        return self.sim.modules["HealthSystem"]

    @pytest.fixture(autouse=True, scope="function")
    def _attach_to_shared_sim(self, small_shared_sim):
        """
        Before each test in this suite, provide access to the shared
        simulation fixture defined above. This ensures that every test
        is run with the persistent simulation object in its context.

        NOTE: this is not strictly necessary in the current implementation
        where we only have one simulation object to share; as we could
        just pass the shared_small_sim explicitly to every test in the
        (derived) class.
        However, it does make accessing the simulation much more similar
        to the main codebase (via self.sim.XXX rather than
        shared_small_sim.XXX) and means we save on explicitly passing the
        same fixture to a lot of tests since we do it automatically.
        If we later define another simulation object that we want to share
        between another set of tests, we can re-use this base class for
        that purpose too, further saving on code repetition.

        WARNING: Writes to the shared simulation object will thus be
        persistent between tests! If a test needs to modify module
        parameters, use a fresh HSI queue, or similar, use the setup
        fixtures also provided with the base class.
        """
        self.sim = small_shared_sim

    @pytest.fixture(scope="function")
    def clears_hsi_queue(self):
        """
        Flags this test as needing to clear the HSI event queue
        in the shared simulation.

        Using this fixture will cause pytest to:
        - Cache the current HSI queue of the shared simulation,
        - Clear the queue,
        - Run the test,
        - Restore the old queue during test teardown.
        The queue can safely be manually cleared again during the
        test if this is necessary (EG if testing two calls to the
        HSI scheduler).
        """
        cached_queue = list(self.shared_sim_healthsystem.HSI_EVENT_QUEUE)
        self.shared_sim_healthsystem.reset_queue()
        yield
        self.shared_sim_healthsystem.HSI_EVENT_QUEUE = list(cached_queue)

    @pytest.fixture(scope="function")
    def changes_event_queue(self):
        """
        Flags the test as needing to change the simulation
        event queue, normally to check that certain events
        have been scheduled based on treatment routines.

        Using this fixture will cause pytest to cache the
        event queue prior to running the test, then restore
        the event queue to this state at the end of the test.
        """
        old_event_queue = copy(self.sim.event_queue)
        yield
        self.sim.event_queue = old_event_queue

    @pytest.fixture(scope="function")
    def changes_module_properties(self):
        """
        Flags this test as needing to change the properties of the
        module being tested in the shared simulation.
        
        Using this fixture will cause pytest to cache the current
        parameters used for the module in question prior to the test
        commencing. Upon test teardown, module parameter values will
        be restored to their pre-test state.
        """
        old_param_values = dict(self.this_module.parameters)
        yield
        self.this_module.parameters = dict(old_param_values)

    @pytest.fixture(scope="function")
    def changes_patient_properties(self, patient_id: int | List[int]):
        """
        Flags this test as needing to manually change the properties
        of the patient at the given index(es) in the shared simulation
        DataFrame.

        Using this fixture will cause pytest to cache the corresponding
        rows in the population DataFrame prior to the test commencing.
        Upon test teardown, these DataFrame rows will be restored to
        their pre-test state.
        """
        cached_values = self.shared_sim_df.loc[patient_id].copy()
        yield
        self.shared_sim_df.loc[patient_id] = cached_values

    @pytest.fixture(scope="function")
    def changes_sim_date(self):
        """
        Flags this test as needing to manually change the date of the
        shared simulation; typically needed when testing cures or deaths
        that are scheduled then occur.

        Using this fixture will cause pytest to cache the date of the
        shared simulation prior to running the test, then restore the
        simulation to that date during teardown.
        """
        old_date = Date(self.sim.date)
        yield
        self.sim.date = Date(old_date)
