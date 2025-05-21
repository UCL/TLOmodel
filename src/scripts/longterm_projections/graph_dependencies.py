"""Construct a graph showing dependencies between modules."""

import argparse
import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Set, Type

import numpy as np

import tlo.methods
from tlo import Module
from tlo.analysis.utils import _standardize_short_treatment_id
from tlo.dependencies import DependencyGetter, get_all_dependencies, is_valid_tlo_module_subclass
from tlo.methods import Metadata

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


def get_module_class_map_set_sequence(excluded_modules: Set[str]) -> dict[str, Type[Module]]:
    """Constructs a map from ``Module`` subclass names to class objects.

    :param excluded_modules: Set of ``Module`` subclass names to exclude from map.

    :return: A mapping from unqualified ``Module`` subclass to names to the corresponding
        class objects. This adds an implicit requirement that the names of all the
        ``Module`` subclasses are unique.

    :raises RuntimeError: Raised if multiple ``Module`` subclasses with the same name are
        defined (and not included in the ``exclude_modules`` set).
    """
    methods_package_path = os.path.dirname(inspect.getfile(tlo.methods))
    module_classes = {}
    sequential_order = []

    # Define the desired order for specific modules
    desired_sequence = [
        "Lifestyle",
        "Contraception",
        "Labour",
        "PregnancySupervisor",
        "PostnatalSupervisor",
        "NewbornOutcomes"
    ]

    # Collect Cancer modules separately
    cancer_modules = []

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

                    # Collect modules in the desired sequence or as cancer modules
                    if obj.__name__ in desired_sequence:
                        sequential_order.append(obj.__name__)
                    elif obj.__name__.endswith('Cancer'):
                        cancer_modules.append(obj.__name__)

    # Create the ordered dictionary
    ordered_module_classes = {}

    # Insert modules from the desired sequence first
    for name in desired_sequence:
        if name in module_classes:
            ordered_module_classes[name] = module_classes[name]

    # Add cancer modules after the desired sequence
    for name in cancer_modules:
        if name in module_classes:
            ordered_module_classes[name] = module_classes[name]

    # Add any remaining modules that weren't in the specified lists
    remaining_modules = {name: module for name, module in module_classes.items() if name not in ordered_module_classes}
    ordered_module_classes.update(remaining_modules)
    return ordered_module_classes


def construct_module_dependency_graph(
    excluded_modules: Set[str],
    disease_module_node_defaults: dict,
    other_module_node_defaults: dict,
    pregnancy_related_module_node_defaults: dict,
    cancer_related_module_node_defaults: dict,
    #infection_related_module_node_defaults: dict,
    get_dependencies: DependencyGetter = get_all_dependencies,
):
    """Construct a pydot object representing module dependency graph.
    :param excluded_modules: Set of ``Module`` subclass names to not included in graph.
    :param disease_module_node_defaults: Any dot node attributes to apply to by default
        to disease module nodes.
    :param pregnancy_related_module_node_defaults: Any dot node attributes to apply to by default
        to pregnancy/birth related module nodes.
    :param cancer_related_module_node_defaults: Any dot node attributes to apply to by default
        to cancer related module nodes.
    :param other_module_node_defaults: Any dot node attributes to apply to by default
        to non-disease module nodes.
    :param infection_module_node_defaults: Any dot node attributes to apply to by default
        to  infectious disease module nodes.
    :param get_dependencies:  Function which given a module gets the set of module
        dependencies. Defaults to extracting all dependencies.
    :return: Pydot directed graph representing module dependencies.
    """
    if pydot is None:
        raise RuntimeError("pydot package must be installed")

    cancer_module_names = [
        'BladderCancer', 'BreastCancer', 'OtherAdultCancer',
        'OesophagealCancer', 'ProstateCancer'
    ]

    pregnancy_module_names = [
        'Contraception', 'Labour', 'PregnancySupervisor',
        'PostnatalSupervisor', 'NewbornOutcomes', 'CareOfWomenDuringPregnancy'
    ]

    infectious_diseases_names = [
        'Hiv', 'Tb', 'Malaria', 'Measles',
    ]

    module_class_map = get_module_class_map_set_sequence(excluded_modules)
    module_graph = pydot.Dot("modules", graph_type="digraph", rankdir='LR')

    # Subgraphs for different groups of modules
    disease_module_subgraph = pydot.Subgraph("disease_modules")
    #disease_module_subgraph.set_rank('same')
    module_graph.add_subgraph(disease_module_subgraph)

    pregnancy_modules_subgraph = pydot.Subgraph("pregnancy_modules")
    pregnancy_modules_subgraph.set_rank('same')
    module_graph.add_subgraph(pregnancy_modules_subgraph)

    other_module_subgraph = pydot.Subgraph("other_modules")
    other_module_subgraph.set_rank('same')
    module_graph.add_subgraph(other_module_subgraph)

    cancer_modules_subgraph = pydot.Subgraph("cancer_modules")
    cancer_modules_subgraph.set_rank('same')
    module_graph.add_subgraph(cancer_modules_subgraph)

    infectious_diseases_subgraph = pydot.Subgraph("infectious_diseases")
    infectious_diseases_subgraph.set_rank('same')
    module_graph.add_subgraph(infectious_diseases_subgraph)

    # Set default styles for nodes
    disease_module_node_defaults["style"] = "filled"
    other_module_node_defaults["style"] = "filled"
    pregnancy_related_module_node_defaults["style"] = "filled"
    cancer_related_module_node_defaults["style"] = "filled"
    #infection_related_module_node_defaults["style"] = "filled"

    # List of modules to group together
    for name, module_class in module_class_map.items():
        # Determine the color based on the module name
        colour = get_color_short_treatment_id(name)

        # Create the node with determined attributes, including outlines
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

    for key, module in module_class_map.items():
        for dependency in get_dependencies(module, module_class_map.keys()):
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
        "DummyDisease"
    }

    module_graph = construct_module_dependency_graph(
        excluded_modules,
        disease_module_node_defaults={"shape": "box"},
        other_module_node_defaults={"shape": "ellipse"},
        pregnancy_related_module_node_defaults={"shape": "diamond"},
        cancer_related_module_node_defaults={"shape": "invtrapezium"}
    )

    format = (
        args.output_file.suffix[1:] if args.output_file.suffix else "raw"
    )
    module_graph.write(args.output_file, format=format)
