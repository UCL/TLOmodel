"""Construct a graph showing dependencies between modules."""

import argparse
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Set, Type, Union
import numpy as np
import importlib
import inspect
import os
import pkgutil
import pydot

import tlo.methods
from tlo import Module
from tlo.methods.hiv import Hiv
from tlo.methods.tb import Tb
from tlo.dependencies import DependencyGetter, get_all_dependencies, is_valid_tlo_module_subclass
from tlo.methods import Metadata

SHORT_TREATMENT_ID_TO_COLOR_MAP = MappingProxyType({
    # Define your color mappings here
    '*': 'black',
    'FirstAttendance*': 'darkgrey',
    # ... (other mappings)
})


def _standardize_short_treatment_id(short_treatment_id: str) -> str:
    return short_treatment_id.replace('_*', '*').rstrip('*') + '*'


def get_color_short_treatment_id(short_treatment_id: str) -> str:
    """Return the colour assigned to this shorted TREATMENT_ID."""
    return SHORT_TREATMENT_ID_TO_COLOR_MAP.get(
        _standardize_short_treatment_id(short_treatment_id), np.nan
    )


def get_properties(module: Union[Module, Type[Module]]) -> Set[str]:
    """Get the properties for a ``Module`` subclass."""
    return module.PROPERTIES


def check_properties_in_module(module: Any, properties: Set[str]) -> Set[str]:
    """Check if any of the properties are used in the given module's script."""
    used_properties = set()
    source_code = inspect.getsource(module)

    for prop in properties:
        if prop in source_code:
            used_properties.add(prop)

    return used_properties


def get_module_property_map(excluded_modules: Set[str]) -> Mapping[str, Set[Type[Module]]]:
    """Constructs a map from property names to sets of Module subclass objects."""
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


def construct_module_dependency_graph(
    excluded_modules: Set[str],
    get_dependencies: DependencyGetter = get_all_dependencies,
):
    """Construct a pydot object representing the module dependency graph."""
    if pydot is None:
        raise RuntimeError("pydot package must be installed")

    module_class_map = get_module_property_map(excluded_modules)
    module_graph = pydot.Dot("modules", graph_type="digraph", rankdir='LR')

    for key, module in module_class_map.items():
        for dependency in get_properties(module):
            if dependency not in excluded_modules:
                module_graph.add_edge(pydot.Edge(dependency, key))

    return module_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_file", type=Path, help=(
            "Path to output graph to. File extension will determine output format - for example: dot, dia, png, svg"
        )
    )
    args = parser.parse_args()

    excluded_modules = {
        "Mockitis",
        "ChronicSyndrome",
        "Skeleton",
    }

    module_graph = construct_module_dependency_graph(excluded_modules)

    format = (
        args.output_file.suffix[1:] if args.output_file.suffix else "raw"
    )
    module_graph.write(args.output_file, format=format)
