"""Tests for automatic checking and ordering of method module dependencies."""
import os
from pathlib import Path
from random import seed as set_seed
from random import shuffle
from types import GeneratorType

import pytest

from tlo import Date, Module, Simulation, logging
from tlo.dependencies import (
    ModuleDependencyError,
    get_all_dependencies,
    get_all_required_dependencies,
    get_dependencies_and_initialise,
    get_module_class_map,
    topologically_sort_modules,
)
from tlo.methods import hiv, simplified_births

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"

simulation_start_date = Date(2010, 1, 1)
simulation_end_date = Date(2010, 9, 1)
simulation_initial_population = 1000

module_class_map = get_module_class_map(
    excluded_modules={
        "Module",
        "Skeleton",
        "SimplifiedPregnancyAndLabour",
    }
)


def parameterize_module_class(test_function):
    return pytest.mark.parametrize(
        "module_class", module_class_map.values(), ids=lambda cls: cls.__name__
    )(test_function)


@pytest.fixture
def sim(seed):
    return Simulation(start_date=simulation_start_date, seed=seed)


@pytest.fixture
def dependent_module_pair():
    class Module1(Module):
        pass

    class Module2(Module):
        INIT_DEPENDENCIES = {'Module1'}

    return Module1, Module2


@pytest.fixture
def dependent_module_chain():
    return [
        type(
            f'Module{i}',
            (Module,),
            {'INIT_DEPENDENCIES': frozenset({f'Module{i - 1}'})} if i != 0 else {}
        )
        for i in range(10)
    ]


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
def test_optional_init_dependencies_all_exist(module_class):
    """Check declared INIT_DEPENDENCIES all correspond to actual classes."""
    assert module_class.OPTIONAL_INIT_DEPENDENCIES.issubset(module_class_map.keys())


@parameterize_module_class
def test_additional_dependencies_all_exist(module_class):
    """Check declared ADDITIONAL_DEPENDENCIES all correspond to actual classes."""
    assert module_class.ADDITIONAL_DEPENDENCIES.issubset(module_class_map.keys())


@parameterize_module_class
def test_alternative_to_all_exist(module_class):
    """Check declared ALTERNATIVE_TO entries all correspond to actual classes."""
    assert module_class.ALTERNATIVE_TO.issubset(module_class_map.keys())


@parameterize_module_class
def test_no_repeated_dependencies(module_class):
    """Check that declared depedency sets do not overlap."""
    assert (
        (module_class.INIT_DEPENDENCIES & module_class.ADDITIONAL_DEPENDENCIES) == set()
    )
    assert (
        (module_class.INIT_DEPENDENCIES & module_class.OPTIONAL_INIT_DEPENDENCIES) == set()
    )
    assert (
        (module_class.ADDITIONAL_DEPENDENCIES & module_class.OPTIONAL_INIT_DEPENDENCIES) == set()
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


def test_topological_sort_modules(seed, dependent_module_chain):
    set_seed(seed)
    modules = [module() for module in dependent_module_chain]
    shuffle(modules)
    sorted_modules = list(topologically_sort_modules(modules))
    # Module in dependency chain named such that (unique) topological sort should
    # correspond to sorting by module class name
    assert sorted(modules, key=lambda module: type(module).__name__) == sorted_modules


def test_get_dependencies_and_initialise(dependent_module_chain):
    module_class_map = {module.__name__: module for module in dependent_module_chain}
    for i, module_class in enumerate(dependent_module_chain):
        module_instances = get_dependencies_and_initialise(
            module_class, module_class_map=module_class_map
        )
        assert isinstance(module_instances, GeneratorType)
        module_instances = list(module_instances)
        assert all(isinstance(module, Module) for module in module_instances)
        module_names = set(type(module).__name__ for module in module_instances)
        assert module_names == set(f'Module{j}' for j in range(i + 1))


def test_get_dependencies_and_initialise_excluded_modules(dependent_module_chain):
    module_class_map = {module.__name__: module for module in dependent_module_chain}
    for i, module_class in enumerate(dependent_module_chain[1:]):
        module_instances = get_dependencies_and_initialise(
            module_class,
            module_class_map=module_class_map,
            excluded_module_classes={dependent_module_chain[i]}
        )
        module_names = set(type(module).__name__ for module in module_instances)
        # Excluded immediate dependency therefore should only be seed module returned
        assert module_names == {module_class.__name__}


@parameterize_module_class
def test_module_dependencies_allow_initialisation(sim, module_class):
    """Check declared dependencies are sufficient for successful initialisation"""
    try:
        register_modules_and_initialise(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                get_dependencies=get_all_dependencies,
                resourcefilepath=resourcefilepath
            ),
            check_all_dependencies=False
        )
    except Exception:
        pytest.fail(
            f'Module {module_class.__name__} appears to be missing dependency '
            f"declarations required to run initialise_population and "
            f"initialise_simulation. The INIT_DEPENDENCIES class attribute should "
            f"specify any modules needing to be initialised before this module "
            f"while ADDITIONAL_DEPENDENCIES should specify all other dependencies. The "
            f"INIT_DEPENDENCIES class attribute is currently set to "
            f"{{{', '.join(module_class.INIT_DEPENDENCIES)}}} and the "
            f"ADDITIONAL_DEPENDENCIES class attribute is currently set to "
            f"{{{', '.join(module_class.ADDITIONAL_DEPENDENCIES)}}}."
        )


@pytest.mark.slow
@parameterize_module_class
def test_module_dependencies_complete(sim, module_class):
    """Check declared dependencies are sufficient for successful (short) simulation.

    Dependencies here refers to the union of INIT_DEPENDENCIES and
    ADDITIONAL_DEPENDENCIES.
    """
    try:
        # If this module is an 'alternative' to one or more other modules, exclude these
        # modules from being selected to avoid clashes with this module
        excluded_module_classes = {
            module_class_map[module_name] for module_name in module_class.ALTERNATIVE_TO
        }
        register_modules_and_simulate(
            sim,
            get_dependencies_and_initialise(
                module_class,
                module_class_map=module_class_map,
                get_dependencies=get_all_dependencies,
                excluded_module_classes=excluded_module_classes,
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
        for dependency_name in sorted(get_all_required_dependencies(module))
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


def test_auto_register_module_dependencies(tmpdir):
    """ check if module dependencies are registered as expected when an argument to auto register modules in simulation
    is set to True """
    # configure logging
    log_config = {
        'filename': 'LogFile',
        'directory': tmpdir,
        'custom_levels': {
            '*': logging.CRITICAL,
            'tlo.method.demography': logging.INFO
        }
    }
    # set simulation start date
    start_date = Date(2010, 1, 1)

    # register required modules for a simple simulation. We have included copd for as it has some dependencies. We want
    # to test if the dependencies can be automatically registered when the auto register argument in simulation
    # is set to True
    def register_disease_modules_manually():
        """ Test manually registering disease modules without including all dependencies and leaving to false an
        option to auto register missing dependencies. This should fail with module dependency error """
        with pytest.raises(ModuleDependencyError, match='missing'):
            # configure simulation
            sim = Simulation(start_date=start_date, seed=0, log_config=log_config, resourcefilepath=resourcefilepath)
            # the lines below should fail with missing dependencies
            sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))

    def register_disease_modules_using_labour_modules_for_births():
        """ Test registering disease modules without including all dependencies and not using simplified births
        module BUT setting to true an option to auto register missing dependencies. This should register all necessary
        modules including all labour modules """
        # configure simulation
        sim = Simulation(start_date=start_date, seed=0, log_config=log_config, resourcefilepath=resourcefilepath)
        # re-register modules with auto-register-module argument set to True and using labour modules for births
        sim.register(hiv.Hiv(resourcefilepath=resourcefilepath),
                     auto_register_dependencies=True)
        # get module dependencies
        required_dependencies = get_all_required_dependencies(sim.modules["Hiv"])
        # check registered dependencies
        registered_module_names = set(sim.modules.keys())
        # all required dependencies should be available in registered dependencies
        assert required_dependencies <= registered_module_names

    def register_disease_modules_using_simplified_births_for_births():
        """ Test registering disease modules without including all dependencies BUT setting to true an option to auto
        register missing dependencies and using simplified births module.This should register all necessary modules
        except labour modules since we're using simplified births """
        # configure simulation
        sim = Simulation(start_date=start_date, seed=0, log_config=log_config, resourcefilepath=resourcefilepath)
        sim.register(hiv.Hiv(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     auto_register_dependencies=True
                     )
        # now that we're using simplified births we want to ensure that all alternative dependencies are not registered
        alternative_dependencies = simplified_births.SimplifiedBirths.ALTERNATIVE_TO
        # get registered modules
        registered_module_names = set(sim.modules.keys())
        # no alternative dependency(labour modules) should get registered when using simplified births
        for dependency in alternative_dependencies:
            assert dependency not in registered_module_names, (f'{dependency} should not be registered when simplified'
                                                               f' module has been registered')

    # test registering disease modules manually(when all dependencies are not included and auto register missing
    # dependencies option is set to false)
    register_disease_modules_manually()

    # test auto registering disease modules using labor modules for births
    register_disease_modules_using_labour_modules_for_births()

    # test auto registering disease modules using simplified module for births
    register_disease_modules_using_simplified_births_for_births()
