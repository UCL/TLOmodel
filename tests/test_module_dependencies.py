"""Tests for automatic checking and ordering of method module dependencies."""

import importlib
import inspect
import os
import pkgutil
from pathlib import Path

import pytest

import tlo.methods
from tlo import Date, Module, Simulation
from tlo.simulation import ModuleDependencyError

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"

simulation_start_date = Date(2010, 1, 1)
simulation_end_date = Date(2011, 1, 1)
simulation_seed = 645407762
simulation_initial_population = 10


def is_valid_tlo_module_class(obj, excluded_modules):
    return (
        inspect.isclass(obj)
        and issubclass(obj, Module)
        and obj.__name__ not in excluded_modules
    )


def get_module_class_map(excluded_modules):
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    module_classes = {}
    for _, methods_module_name, _ in pkgutil.iter_modules([methods_package_path]):
        methods_module = importlib.import_module(f'tlo.methods.{methods_module_name}')
        for _, obj in inspect.getmembers(methods_module):
            if is_valid_tlo_module_class(obj, excluded_modules):
                if module_classes.get(obj.__name__) not in {None, obj}:
                    raise RuntimeError(
                        f'Multiple modules with name {obj.__name__} are defined'
                    )
                else:
                    module_classes[obj.__name__] = obj
    return module_classes


module_class_map = get_module_class_map(
    excluded_modules={
        'Module',
        'Skeleton',
        'Tb',
        'DummyHivModule',
        'PropertiesOfOtherModules'
    }
)


def get_dependencies_and_initialise(module_class, excluded_module_classes=None):
    visited = set()
    if excluded_module_classes is None:
        excluded_module_classes = set()

    def depth_first_search(module_class):
        if module_class not in (visited | excluded_module_classes):
            for dependency_name in module_class.INIT_DEPENDENCIES:
                yield from depth_first_search(module_class_map[dependency_name])
            visited.add(module_class)
            yield module_class(resourcefilepath=resourcefilepath)

    yield from depth_first_search(module_class)


@pytest.fixture
def sim():
    return Simulation(start_date=simulation_start_date, seed=simulation_seed)


@pytest.fixture(params=module_class_map.values())
def module_class(request):
    return request.param


@pytest.fixture
def dependent_module_pair():

    class Module1(Module):
        pass

    class Module2(Module):
        INIT_DEPENDENCIES = {'Module1'}

    return Module1, Module2


def register_modules_and_initialise(sim, modules):
    sim.register(*modules)
    sim.make_initial_population(n=simulation_initial_population)
    sim.end_date = simulation_end_date
    for module in sim.modules.values():
        module.initialise_simulation(sim)


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


def test_module_init_dependencies_complete(sim, module_class):
    """Check declared INIT_DEPENDENCIES are sufficient for successful initialisation"""
    try:
        register_modules_and_initialise(
            sim, get_dependencies_and_initialise(module_class)
        )
    except Exception:
        pytest.fail(
            f"Module {module_class.__name__} appears to be missing dependencies "
            f"required to run initialise_population and initialise_simulation in the "
            f"INIT_DEPENDENCIES class attribute which is currently set to "
            f"{{{', '.join(module_class.INIT_DEPENDENCIES)}}}."
        )


def test_module_init_dependencies_all_required(sim, module_class):
    """Check that all INIT_DEPENDENCIES are required for successful initialisation"""
    for dependency_name in module_class.INIT_DEPENDENCIES:
        dependency_class = module_class_map[dependency_name]
        try:
            register_modules_and_initialise(
                sim, get_dependencies_and_initialise(module_class, {dependency_class})
            )
        except Exception:
            # This is the expected behaviour i.e. that trying to initialise with
            # a dependency removed results in an exception
            pass
        else:
            pytest.fail(
                f'The dependency {dependency_name} of {module_class.__name__} '
                'does not appear to be required to run initialise_population and '
                'initialise_simulation without errors and so should be removed '
                f'from {module_class.__name__}.INIT_DEPENDENCIES'
            )
