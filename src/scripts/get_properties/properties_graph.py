"""Functions for getting, checking, and sorting properties of ``Module`` subclasses."""

from typing import Any, Dict, Mapping, Set, Type, Union
import inspect
import importlib
import os
import pkgutil

import tlo.methods
from tlo import Module
from tlo.methods.hiv import Hiv
from tlo.methods.tb import Tb


def get_properties(
    module: Union[Module, Type[Module]],
) -> Set[str]:
    """Get the properties for a ``Module`` subclass.

    :param module: ``Module`` subclass to get properties for.
    :return: Set of ``Module`` subclass names corresponding to properties of ``module``.
    """
    return module.PROPERTIES


def check_properties_in_module(module: Any, properties: Set[str]) -> Set[str]:
    """Check if any of the properties are used in the given module's script."""
    used_properties = set()

    # Get the source code of the module
    source_code = inspect.getsource(module)

    # Check each property for usage in the source code
    for prop in properties:
        if prop in source_code:
            used_properties.add(prop)

    return used_properties


def is_valid_tlo_module_subclass(obj: Any, excluded_modules: Set[str]) -> bool:
    """Check if the object is a valid TLO Module subclass."""
    return isinstance(obj, type) and issubclass(obj, Module) and obj.__name__ not in excluded_modules


def get_module_property_map(excluded_modules: Set[str]) -> Mapping[str, Set[Type[Module]]]:
    """Constructs a map from property names to sets of Module subclass objects.

    :param excluded_modules: Set of Module subclass names to exclude from the map.

    :return: A mapping from property names to sets of corresponding Module subclass objects.
    This adds an implicit requirement that the names of all the Module subclasses are unique.

    :raises RuntimeError: Raised if multiple Module subclasses with the same name are defined
        (and not included in the excluded_modules set).
    """
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    module_property_map: Dict[str, Set[Type[Module]]] = {}

    for _, methods_module_name, _ in pkgutil.iter_modules([methods_package_path]):
        methods_module = importlib.import_module(f'tlo.methods.{methods_module_name}')
        for _, obj in inspect.getmembers(methods_module):
            if is_valid_tlo_module_subclass(obj, excluded_modules):
                properties = get_properties(obj)

                for prop in properties:
                    if prop not in module_property_map:
                        module_property_map[prop] = set()
                    module_property_map[prop].add(obj)

    return module_property_map


# Get properties from your target Module subclass
module_properties = get_properties(Hiv)

# Use the function to find used properties
used_properties = check_properties_in_module(Tb, module_properties)

# Print the results
if used_properties:
    print("The following properties are used in the other module's script:")
    for prop in used_properties:
        print(f"- {prop}")
else:
    print("No properties are used in the other module's script.")

# Example usage of get_module_property_map
excluded = {'SomeExcludedModule'}  # Specify any excluded modules
property_map = get_module_property_map(excluded)


def get_module_property_map(excluded_modules: Set[str]) -> Mapping[str, Set[Type[Module]]]:
    """Constructs a map from property names to sets of Module subclass objects.

    :param excluded_modules: Set of Module subclass names to exclude from the map.

    :return: A mapping from property names to sets of corresponding Module subclass objects.
    This adds an implicit requirement that the names of all the Module subclasses are unique.

    :raises RuntimeError: Raised if multiple Module subclasses with the same name are defined
        (and not included in the excluded_modules set).
    """
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    module_property_map: Dict[str, Set[Type[Module]]] = {}

    for _, methods_module_name, _ in pkgutil.iter_modules([methods_package_path]):
        methods_module = importlib.import_module(f'tlo.methods.{methods_module_name}')
        for _, obj in inspect.getmembers(methods_module):
            if is_valid_tlo_module_subclass(obj, excluded_modules):
                properties = get_properties(obj)

                for prop in properties:
                    if prop not in module_property_map:
                        module_property_map[prop] = set()
                    module_property_map[prop].add(obj)

    return module_property_map


# Print the property map for verification
for prop, modules in property_map.items():
    print(f"Property '{prop}' is provided by the following modules: {', '.join(module.__name__ for module in modules)}")
