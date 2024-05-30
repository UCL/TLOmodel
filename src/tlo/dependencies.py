"""Functions for getting, checking and sorting dependencies of ``Module`` subclasses."""

import importlib
import inspect
import os
import pkgutil
from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Set, Type, Union

import tlo.methods
from tlo import Module
from tlo.methods.hsi_generic_first_appts import GenericFirstApptModule


class ModuleDependencyError(Exception):
    """Raised when a module dependency is missing or there are circular dependencies."""


class MultipleModuleInstanceError(Exception):
    """Raised when multiple instances of the same module are registered."""


DependencyGetter = Callable[[Union[Module, Type[Module]], Set[str]], Set[str]]


def get_init_dependencies(
    module: Union[Module, Type[Module]],
    module_names_present: Set[str]
) -> Set[str]:
    """Get the initialisation dependencies for a ``Module`` subclass.

    :param module: ``Module`` subclass to get dependencies for.
    :param module_names_present: Set of names of ``Module`` subclasses that will be
        present in simulation to use to select optional initialisation dependencies.
    :return: Set of ``Module`` subclass names corresponding to initialisation
        dependencies of ``module``, including any optional dependencies present.
    """
    return (
        module.INIT_DEPENDENCIES
        | (module.OPTIONAL_INIT_DEPENDENCIES & module_names_present)
    )


def get_all_dependencies(
    module: Union[Module, Type[Module]],
    module_names_present: Set[str]
) -> Set[str]:
    """Get all dependencies for a ``Module`` subclass.

    :param module: ``Module`` subclass to get dependencies for.
    :param module_names_present: Set of names of ``Module`` subclasses that will be
        present in simulation to use to select optional initialisation dependencies.
    :return: Set of ``Module`` subclass names corresponding to dependencies of
        ``module``, including any optional dependencies present.
    """
    return (
        get_init_dependencies(module, module_names_present)
        | module.ADDITIONAL_DEPENDENCIES
    )


def get_all_required_dependencies(
    module: Union[Module, Type[Module]],
    module_names_present: Optional[Set[str]] = None
) -> Set[str]:
    """Get all non-optional dependencies for a ``Module`` subclass.

    :param module: ``Module`` subclass to get dependencies for.
    :param module_names_present: Set of names of ``Module`` subclasses that will be
        present in simulation to use to select optional initialisation dependencies.
        Unused by this function, but kept as an argument to ensure a consistent
        interface with the other dependency-getter functions.
    :return: Set of ``Module`` subclass names corresponding to non-optional dependencies
        of ``module``.
    """
    return module.INIT_DEPENDENCIES | module.ADDITIONAL_DEPENDENCIES


def topologically_sort_modules(
    module_instances: Iterable[Module],
    get_dependencies: DependencyGetter = get_init_dependencies,
) -> Generator[Module, None, None]:
    """Generator which yields topological sort of modules based on their dependencies.

    A topological sort of a dependency graph is ordered such that any dependencies of a
    node in the graph are guaranteed to be yielded before the node itself. This
    implementation uses a depth-first search algorithm
    (https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search).

    :param module_instances: The set of module instances to topologically sort. The
        yielded module instances will consist of all nodes in this set which must
        include instances of their (recursive) dependencies.
    :param get_dependencies: Function which given a module gets the set of module
        dependencies. Defaults to returing the ``Module.INIT_DEPENDENCIES`` class
        attribute.

    :raises ModuleDependencyError: Raised when a module dependency is missing from
        ``module_instances`` or a module has circular dependencies.
    :raises MultipleModuleInstanceError: Raised when multiple instances of the same
        module are passed in ``module_instances``.

    :return: Generator which yields module instances in topologically sorted order.
    """
    module_instances = list(module_instances)
    module_instance_map = {type(module).__name__: module for module in module_instances}
    if len(module_instance_map) != len(module_instances):
        raise MultipleModuleInstanceError(
            'Multiple instances of one or more `Module` subclasses were passed to the '
            'Simulation.register method. If you are sure this is correct, you can '
            'disable this check (and the automatic dependency sorting) by setting '
            'sort_modules=False in Simulation.register.'
        )
    visited, currently_processing = set(), set()

    def depth_first_search(module):
        if module not in visited:
            if module in currently_processing:
                raise ModuleDependencyError(
                    f'Module {module} has circular dependencies.'
                )
            currently_processing.add(module)
            dependencies = get_dependencies(
                module_instance_map[module], module_instance_map.keys()
            )
            for dependency in sorted(dependencies):
                if dependency not in module_instance_map:
                    alternatives_with_instances = [
                        name for name, instance in module_instance_map.items()
                        if dependency in instance.ALTERNATIVE_TO
                    ]
                    if len(alternatives_with_instances) != 1:
                        message = (
                            f'Module {module} depends on {dependency} which is '
                            'missing from modules to register'
                        )
                        if len(alternatives_with_instances) == 0:
                            message += f' as are any alternatives to {dependency}.'
                        else:
                            message += (
                                ' and there are multiple alternatives '
                                f'({alternatives_with_instances}) so which '
                                'to use to resolve dependency is ambiguous.'
                            )
                        raise ModuleDependencyError(message)

                    else:
                        yield from depth_first_search(alternatives_with_instances[0])

                else:
                    yield from depth_first_search(dependency)
            currently_processing.remove(module)
            visited.add(module)
            yield module_instance_map[module]

    for module_instance in module_instances:
        yield from depth_first_search(type(module_instance).__name__)


def is_valid_tlo_module_subclass(obj: Any, excluded_modules: Set[str]) -> bool:
    """Determine whether object is a ``Module`` subclass and not in an excluded set.

    :param obj: Object to check if ``Module`` subclass.
    :param excluded_modules: Set of names of ``Module`` subclasses to force check to
        return ``False`` for.

    :return: ``True`` is ``obj`` is a _strict_ subclass of ``Module`` and not in the
        ``excluded_modules`` set.
    """
    return (
        inspect.isclass(obj)
        and issubclass(obj, Module)
        and obj is not Module
        and obj is not GenericFirstApptModule
        and obj.__name__ not in excluded_modules
    )


def get_module_class_map(excluded_modules: Set[str]) -> Mapping[str, Type[Module]]:
    """Constructs a map from ``Module`` subclass names to class objects.

    :param excluded_modules: Set of ``Module`` subclass names to exclude from map.

    :return: A mapping from unqualified ``Module`` subclass to names to the correponding
        class objects. This adds an implicit requirement that the names of all the
        ``Module`` subclasses are unique.

    :raises RuntimError: Raised if multiple ``Module`` subclasses with the same name are
        defined (and not included in the ``exclude_modules`` set).
    """
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    module_classes = {}
    for _, methods_module_name, _ in pkgutil.iter_modules([methods_package_path]):
        methods_module = importlib.import_module(f'tlo.methods.{methods_module_name}')
        for _, obj in inspect.getmembers(methods_module):
            if is_valid_tlo_module_subclass(obj, excluded_modules):
                if module_classes.get(obj.__name__) not in {None, obj}:
                    raise RuntimeError(
                        f'Multiple modules with name {obj.__name__} are defined'
                    )
                else:
                    module_classes[obj.__name__] = obj
    return module_classes


def get_dependencies_and_initialise(
    *module_classes: Type[Module],
    module_class_map: Mapping[str, Type[Module]],
    excluded_module_classes: Optional[Set[Module]] = None,
    get_dependencies: DependencyGetter = get_init_dependencies,
    **module_class_kwargs
) -> Generator[Module, None, None]:
    """Generate a sequence of ``Module`` instances including all dependencies.

    The generated sequence of initialised ``Module`` subclass instances will correspond
    to all the (recursive) dependencies of the seed ``Module`` subclasses in
    ``module_classes``.

    :param module_classes: ``Module`` subclass(es) to seed dependency search with.
    :param module_class_map: Mapping from ``Module`` subclass names to classes.
    :param excluded_module_classes: Any ``Module`` subclasses to not yield instances
        of in the returned generator.
    :param get_dependencies: Function which given a module gets the set of module
        dependencies. Defaults to returing the ``Module.INIT_DEPENDENCIES`` class
        attribute.
    :param module_class_kwargs: Any keyword arguments to pass to initialisers for
        ``Module`` subclasses if present in their ``__init__`` method signature.

    :return: Sequence of initialised ``Module`` subclass instances corresponding to all
        of the ``Module`` subclasses and their the (recursive) dependencies in the seed
        ``module_classes``.
    """
    if excluded_module_classes is None:
        excluded_module_classes = set()

    visited = set()

    def initialise_module(module_class):
        signature = inspect.signature(module_class)
        relevant_kwargs = {
            key: value for key, value in module_class_kwargs.items()
            if key in signature.parameters
        }
        bound_args = signature.bind(**relevant_kwargs)
        return module_class(*bound_args.args, **bound_args.kwargs)

    def depth_first_search(module_class):
        if module_class not in (visited | excluded_module_classes):
            visited.add(module_class)
            yield initialise_module(module_class)
            dependencies = get_dependencies(module_class, module_class_map.keys())
            for dependency_name in sorted(dependencies):
                yield from depth_first_search(module_class_map[dependency_name])

    for module_class in module_classes:
        yield from depth_first_search(module_class)


def check_dependencies_present(
    module_instances: Iterable[Module],
    get_dependencies: DependencyGetter = get_all_dependencies,
):
    """Check whether an iterable of modules contains the required dependencies.

    :param module_instances: Iterable of ``Module`` subclass instances to check.
    :param get_dependencies: Callable which extracts the set of dependencies to check
        for from a module instance. Defaults to extracting all dependencies.

    :raises ModuleDependencyError: Raised if any dependencies are missing.
    """
    module_instances = list(module_instances)
    modules_present = {type(module).__name__ for module in module_instances}
    modules_present_are_alternatives_to = set.union(
        # Force conversion to set to avoid errors when using set.union with frozenset
        *(set(module.ALTERNATIVE_TO) for module in module_instances)
    )
    modules_required = set.union(
        *(set(get_dependencies(module, modules_present)) for module in module_instances)
    )
    missing_dependencies = modules_required - modules_present
    missing_dependencies_without_alternatives_present = (
        missing_dependencies - modules_present_are_alternatives_to
    )
    if not missing_dependencies_without_alternatives_present == set():

        raise ModuleDependencyError(
            'One or more required dependency is missing from the module list and no '
            'alternative to this / these modules are available either: '
            f'{missing_dependencies_without_alternatives_present}'
        )
