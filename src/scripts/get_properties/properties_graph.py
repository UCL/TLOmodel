"""Construct a graph showing dependencies between modules."""

import argparse
import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Set, Type, Union

import numpy as np
import pydot

import tlo.methods
from tlo import Module
from tlo.dependencies import DependencyGetter, get_all_dependencies, is_valid_tlo_module_subclass
from tlo.methods import Metadata
from tlo.methods.hiv import Hiv
from tlo.methods.tb import Tb

try:
    import pydot
except ImportError:
    pydot = None
SHORT_TREATMENT_ID_TO_COLOR_MAP = MappingProxyType({
    '*': 'black',
    'FirstAttendance*': 'darkgrey',
    'Inpatient*': 'silver',
    'Contraception*': 'darkseagreen',
    'AntenatalCare*': 'green',
    'DeliveryCare*': 'limegreen',
    'PostnatalCare*': 'springgreen',
    'CareOfWomenDuringPregnancy*': '#4D804D',
    'Labour*': '#19A719',
    'NewbornOutcomes*': '#19E659',
    'PostnatalSupervisor*': '#5D8C5D',
    'PregnancySupervisor*': '#27C066',
    'Alri*': 'darkorange',
    'Diarrhoea*': 'tan',
    'Undernutrition*': 'gold',
    'Epi*': 'darkgoldenrod',
    'Stunting*': '#D58936',
    'StuntingPropertiesOfOtherModules*': "#EAC143",
    'Wasting*': '#DE9F0E',
    'Hiv*': 'deepskyblue',
    'Malaria*': 'lightsteelblue',
    'Measles*': 'cornflowerblue',
    'Tb*': 'mediumslateblue',
    'Schisto*': 'skyblue',
    'CardioMetabolicDisorders*': 'brown',
    'BladderCancer*': 'orchid',
    'BreastCancer*': 'mediumvioletred',
    'OesophagealCancer*': 'deeppink',
    'ProstateCancer*': 'hotpink',
    'OtherAdultCancer*': 'palevioletred',
    'Depression*': 'indianred',
    'Epilepsy*': 'red',
    'Copd*': 'lightcoral',
    'RTI*': 'lightsalmon',
    'Lifestyle*': 'silver',
})


def _standardize_short_treatment_id(short_treatment_id):
    return short_treatment_id.replace('_*', '*').rstrip('*') + '*'


def get_color_short_treatment_id(short_treatment_id: str) -> str:
    """Return the colour assigned to this shorted TREATMENT_ID.

    Returns `np.nan` if treatment_id is not recognised.
    """
    return SHORT_TREATMENT_ID_TO_COLOR_MAP.get(
        _standardize_short_treatment_id(short_treatment_id), np.nan
    )


def get_properties(
    module: Union[Module, Type[Module]],
) -> Set[str]:
    """Get the properties for a ``Module`` subclass.

    :param module: ``Module`` subclass to get properties for.
    :return: Set of ``Module`` subclass names corresponding to properties of ``module``.
    """
    if hasattr(module, 'PROPERTIES'):
        return module.PROPERTIES
    return None


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


def get_module_property_map(excluded_modules: Set[str]) -> Mapping[str, Set[Type[Module]]]:
    """Constructs a map from property names to sets of Module subclass objects.

    :param excluded_modules: Set of Module subclass names to exclude from the map.

    :return: A mapping from property names to sets of corresponding Module subclass objects.
    This adds an implicit requirement that the names of all the Module subclasses are unique.

    :raises RuntimeError: Raised if multiple Module subclasses with the same name are defined
        (and not included in the excluded_modules set).
    """
    properties_dictionary = {}
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))

    for _, main_module_name, _ in pkgutil.iter_modules([methods_package_path]):
        methods_module = importlib.import_module(f'tlo.methods.{main_module_name}')
        for _, obj in inspect.getmembers(methods_module):
            if is_valid_tlo_module_subclass(obj, excluded_modules):
                properties_dictionary[obj.__name__] = obj
    return properties_dictionary


def construct_property_dependency_graph(
    excluded_modules: Set[str],
    disease_module_node_defaults: dict,
    other_module_node_defaults: dict,
    pregnancy_related_module_node_defaults: dict,
    cancer_related_module_node_defaults: dict,
    properies_node_defaults: dict,
    get_dependencies: DependencyGetter = get_properties,
):
    """Construct a pydot object representing module dependency graph.
    :param excluded_modules: Set of ``Module`` subclass names to not included in graph.
    :param get_dependencies:  Function which given a module gets the set of property
        dependencies. Defaults to extracting all dependencies.
    :return: Pydot directed graph representing module dependencies.
    """
    if pydot is None:
        raise RuntimeError("pydot package must be installed")

    property_class_map = get_module_property_map(excluded_modules)
    property_graph = pydot.Dot("properties", graph_type="digraph", rankdir='LR')

    cancer_module_names = [
        'BladderCancer', 'BreastCancer', 'OtherAdultCancer',
        'OesophagealCancer', 'ProstateCancer'
    ]

    pregnancy_module_names = [
        'Contraception', 'Labour', 'PregnancySupervisor',
        'PostnatalSupervisor', 'NewbornOutcomes', 'CareOfWomenDuringPregnancy'
    ]

    # Subgraphs for different groups of modules - attempt at aesthetics
    disease_module_subgraph = pydot.Subgraph("disease_modules")
    property_graph.add_subgraph(disease_module_subgraph)

    pregnancy_modules_subgraph = pydot.Subgraph("pregnancy_modules")
    property_graph.add_subgraph(pregnancy_modules_subgraph)

    other_module_subgraph = pydot.Subgraph("other_modules")
    property_graph.add_subgraph(other_module_subgraph)

    cancer_modules_subgraph = pydot.Subgraph("cancer_modules")
    cancer_modules_subgraph.set_rank('same')
    property_graph.add_subgraph(cancer_modules_subgraph)

    infectious_diseases_subgraph = pydot.Subgraph("infectious_diseases")
    property_graph.add_subgraph(infectious_diseases_subgraph)

    properties_diseases_subgraph = pydot.Subgraph("properties")
    property_graph.add_subgraph(properties_diseases_subgraph)

    # Set default styles for nodes
    disease_module_node_defaults["style"] = "filled"
    other_module_node_defaults["style"] = "filled"
    pregnancy_related_module_node_defaults["style"] = "filled"
    cancer_related_module_node_defaults["style"] = "filled"
    properies_node_defaults["style"] = "filled"

    for name, module_class in property_class_map.items():  # only works for disease modules, not properties
        colour = get_color_short_treatment_id(name)
        node_attributes = {
            "fillcolor": colour,
            "color": "black",  # Outline color
            "fontname": "Arial",
        }

        if name in pregnancy_module_names:
            node_attributes.update(pregnancy_related_module_node_defaults)
            node_attributes["shape"] = "diamond"  # Pregnancy modules
            pregnancy_modules_subgraph.add_node(pydot.Node(name, **node_attributes))

        elif name in cancer_module_names:
            node_attributes.update(cancer_related_module_node_defaults)
            node_attributes["shape"] = "invtrapezium"  # Cancer modules
            cancer_modules_subgraph.add_node(pydot.Node(name, **node_attributes))

        elif Metadata.DISEASE_MODULE not in module_class.METADATA:
            node_attributes.update(other_module_node_defaults)
            node_attributes["shape"] = "ellipse"  # Other modules
            other_module_subgraph.add_node(pydot.Node(name, **node_attributes))
        else:
            node_attributes.update(disease_module_node_defaults)
            node_attributes["shape"] = "box"  # Disease modules
            disease_module_subgraph.add_node(pydot.Node(name, **node_attributes))

    for key, property_module in property_class_map.items():
        if property_module not in excluded_modules:
            properties_of_module = get_dependencies(property_module)
            for key, dependent_module in property_class_map.items():
                if property_module != dependent_module:
                    used_properties = check_properties_in_module(dependent_module, properties_of_module)
                    for property in used_properties:
                        if property.startswith("ri"):
                            node_attributes = {
                                "fillcolor": "darkorange",
                                "color": "black",  # Outline color
                                "fontname": "Arial",
                            }
                        else:
                            node_attributes = {
                                "fillcolor": "white",
                                "color": "black",  # Outline color
                                "fontname": "Arial",
                            }
                        node_attributes.update(properies_node_defaults)
                        node_attributes["shape"] = "square"
                        properties_diseases_subgraph.add_node(pydot.Node(property, **node_attributes))
                        properties_diseases_subgraph.set_rank('same')
                        property_graph.add_edge(pydot.Edge(property, key))

    return property_graph


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
        "AlriPropertiesOfOtherModules",
        "DiarrhoeaPropertiesOfOtherModules",
        "DummyHivModule",
        "SimplifiedBirths",
        "Demography",
        "HealthBurden",
        "SymptomManager",
        "DummyTbModule",
        "ImprovedHealthSystemAndCareSeekingScenarioSwitcher",
        "HealthSeekingBehaviour",
        "HealthSystem",
        "Deviance",
        "SimplifiedPregnancyAndLabour",
        "DummyDisease",
        "Module"
    }

    module_graph = construct_property_dependency_graph(
        excluded_modules,
        disease_module_node_defaults={"shape": "box"},
        other_module_node_defaults={"shape": "ellipse"},
        pregnancy_related_module_node_defaults={"shape": "diamond"},
        cancer_related_module_node_defaults={"shape": "invtrapezium"},
        properies_node_defaults={"shape": "square"}
    )

    format = (
        args.output_file.suffix[1:] if args.output_file.suffix else "raw"
    )
    module_graph.write(args.output_file, format=format)
