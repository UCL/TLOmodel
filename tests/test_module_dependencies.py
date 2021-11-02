"""Tests for automatic checking and ordering of method module dependencies."""

import os
from pathlib import Path

import pytest

from tlo import Date, Module, Simulation
from tlo.dependencies import (
    ModuleDependencyError,
    get_all_dependencies,
    get_all_required_dependencies,
    get_dependencies_and_initialise,
    get_module_class_map,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"

simulation_start_date = Date(2010, 1, 1)
simulation_end_date = Date(2010, 4, 1)
simulation_seed = 645407762
simulation_initial_population = 1000


module_class_map = get_module_class_map(excluded_modules={'Module', 'Skeleton', 'Tb'})


def parameterize_module_class(test_function):
    return pytest.mark.parametrize(
        "module_class", module_class_map.values(), ids=lambda cls: cls.__name__
    )(test_function)


@pytest.fixture
def sim():
    return Simulation(start_date=simulation_start_date, seed=simulation_seed)


@pytest.fixture
def dependent_module_pair():

    class Module1(Module):
        pass

    class Module2(Module):
        INIT_DEPENDENCIES = {'Module1'}

    return Module1, Module2


def register_modules_and_initialise(sim, modules, **register_kwargs):
    sim.register(*modules, **register_kwargs)
    sim.make_initial_population(n=simulation_initial_population)
    sim.end_date = simulation_end_date
    for module in sim.modules.values():
        module.initialise_simulation(sim)


def register_modules_and_simulate(sim, modules, **register_kwargs):
    sim.register(*modules, **register_kwargs)
    sim.make_initial_population(n=simulation_initial_population)
    sim.simulate(end_date=simulation_end_date)


@parameterize_module_class
def test_init_dependencies_all_exist(module_class):
    """Check declared INIT_DEPENDENCIES all correspond to actual classes."""
    assert module_class.INIT_DEPENDENCIES.issubset(module_class_map.keys())


@parameterize_module_class
def test_additional_dependencies_all_exist(module_class):
    """Check declared ADDITIONAL_DEPENDENCIES all correspond to actual classes."""
    assert module_class.ADDITIONAL_DEPENDENCIES.issubset(module_class_map.keys())


@parameterize_module_class
def test_no_repeated_dependencies(module_class):
    """Check that INIT_DEPENDENCIES and ADDITIONAL_DEPENDENCIES do not overlap."""
    assert (
        (module_class.INIT_DEPENDENCIES & module_class.ADDITIONAL_DEPENDENCIES) == set()
    )


def test_circular_dependency_raises_error_on_register(sim, dependent_module_pair):
    """Check that trying to register modules with circular dependencies raises error."""

    Module1, Module2 = dependent_module_pair

    Module1.INIT_DEPENDENCIES |= {'Module2'}

    with pytest.raises(ModuleDependencyError, match="circular"):
        sim.register(Module1(), Module2())


def test_missing_dependency_raises_error_on_register(sim, dependent_module_pair):
    """Check that trying to register modules with missing dependency raises error."""

    _, Module2 = dependent_module_pair

    with pytest.raises(ModuleDependencyError, match="missing"):
        sim.register(Module2())


@parameterize_module_class
def test_module_init_dependencies_complete(sim, module_class):
    """Check declared INIT_DEPENDENCIES are sufficient for successful initialisation"""
    try:
        register_modules_and_initialise(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=False
        )
    except Exception:
        pytest.fail(
            f"Module {module_class.__name__} appears to be missing dependencies "
            f"required to run initialise_population and initialise_simulation in the "
            f"INIT_DEPENDENCIES class attribute which is currently set to "
            f"{{{', '.join(module_class.INIT_DEPENDENCIES)}}}."
        )


@pytest.mark.parametrize(
    "module_and_init_dependency_pair",
    [
        (module, module_class_map[dependency_name])
        for module in module_class_map.values()
        for dependency_name in module.INIT_DEPENDENCIES
    ],
    ids=lambda pair: f"{pair[0].__name__}, {pair[1].__name__}"
)
def test_module_init_dependencies_all_required(sim, module_and_init_dependency_pair):
    """Check that all INIT_DEPENDENCIES are required for successful initialisation"""
    module_class, dependency_class = module_and_init_dependency_pair
    try:
        register_modules_and_initialise(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                excluded_module_classes={dependency_class},
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=False
        )
    except Exception:
        # This is the expected behaviour i.e. that trying to initialise with
        # a dependency removed results in an exception
        pass
    else:
        pytest.fail(
            f'The dependency {dependency_class.__name__} of {module_class.__name__} '
            'does not appear to be required to run initialise_population and '
            'initialise_simulation without errors and so should be removed '
            f'from {module_class.__name__}.INIT_DEPENDENCIES'
        )


@parameterize_module_class
def test_module_dependencies_complete(sim, module_class):
    """Check declared dependencies are sufficient for successful (short) simulation.

    Dependencies here refers to the union of INIT_DEPENDENCIES and
    ADDITIONAL_DEPENDENCIES.
    """
    try:
        register_modules_and_simulate(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                get_dependencies=get_all_dependencies,
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=True
        )
    except Exception:
        all_dependencies = get_all_required_dependencies(module_class)
        pytest.fail(
            f"Module {module_class.__name__} appears to be missing dependencies "
            f"required to run simulation in the union of the INIT_DEPENDENCIES and "
            f"ADDITIONAL_DEPENDENCIES class attributes which is currently "
            f"{{{', '.join(all_dependencies)}}}."
        )


@pytest.mark.parametrize(
    "module_and_dependency_pair",
    [
        (module, module_class_map[dependency_name])
        for module in module_class_map.values()
        # Skip test for NewbornOutcomes as long simulation needed for birth events to occur and dependencies to be used
        if module.__name__ not in {
            'NewbornOutcomes'
        }
        for dependency_name in get_all_required_dependencies(module)
    ],
    ids=lambda pair: f"{pair[0].__name__}, {pair[1].__name__}"
)
def test_module_dependencies_all_required(sim, module_and_dependency_pair):
    """Check that declared dependencies are required for successful simulation.

    Dependencies here refers to the union of INIT_DEPENDENCIES and
    ADDITIONAL_DEPENDENCIES.
    """
    module_class, dependency_class = module_and_dependency_pair
    try:
        register_modules_and_simulate(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                excluded_module_classes={dependency_class},
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=False
        )
    except Exception:
        # This is the expected behaviour i.e. that trying to initialise with
        # a dependency removed results in an exception
        pass
    else:
        pytest.fail(
            f'The dependency {dependency_class.__name__} of {module_class.__name__} '
            'does not appear to be required to run simulation without errors and so '
            f'should be removed from the dependencies of {module_class.__name__}.'
        )
