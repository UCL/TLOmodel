"""Construct a graph showing dependencies between modules."""

import argparse
from pathlib import Path
from typing import Dict, Set

from tlo.dependencies import DependencyGetter, get_all_dependencies, get_module_class_map
from tlo.methods import Metadata

try:
    import pydot
except ImportError:
    pydot = None


def construct_module_dependency_graph(
    excluded_modules: Set[str],
    disease_module_node_defaults: Dict,
    other_module_node_defaults: Dict,
    get_dependencies: DependencyGetter = get_all_dependencies,
):
    """Construct a pydot object representing module dependency graph.
    :param excluded_modules: Set of ``Module`` subclass names to not included in graph.
    :param disease_module_node_defaults: Any dot node attributes to apply to by default
        to disease module nodes.
    :param other_module_node_defaults: Any dot node attributes to apply to by default
        to non-disease module nodes.
    :param get_dependencies:  Function which given a module gets the set of module
        dependencies. Defaults to extracting all dependencies.
    :return: Pydot directed graph representing module dependencies.
    """
    if pydot is None:
        raise RuntimeError("pydot package must be installed")

    module_class_map = get_module_class_map(excluded_modules)
    module_graph = pydot.Dot("modules", graph_type="digraph")
    disease_module_subgraph = pydot.Subgraph("disease_modules")
    module_graph.add_subgraph(disease_module_subgraph)
    other_module_subgraph = pydot.Subgraph("other_modules")
    module_graph.add_subgraph(other_module_subgraph)

    # Set default styles for nodes
    disease_module_node_defaults["style"] = "filled"
    other_module_node_defaults["style"] = "filled"

    for name, module_class in module_class_map.items():
        # Determine attributes based on module type
        node_attributes = {}

        if Metadata.DISEASE_MODULE in module_class.METADATA and name.endswith("Cancer"):
            node_attributes.update(disease_module_node_defaults)
            node_attributes["color"] = "lightblue"  # Color for disease modules and Cancer
        else:
            node_attributes.update(other_module_node_defaults)
            node_attributes["color"] = "lightgreen"  # Default color for other modules

        # Create the node with determined attributes
        node = pydot.Node(name, **node_attributes)

        # Add the node to the appropriate subgraph
        if Metadata.DISEASE_MODULE in module_class.METADATA or name.endswith("Cancer"):
            disease_module_subgraph.add_node(node)
        else:
            other_module_subgraph.add_node(node)

    for key, module in module_class_map.items():
        for dependency in get_dependencies(module, module_class_map.keys()):
            if dependency not in excluded_modules:
                module_graph.add_edge(pydot.Edge(key, dependency))

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
        "SimplifiedPregnancyAndLabour"
    }

    module_graph = construct_module_dependency_graph(
        excluded_modules,
        disease_module_node_defaults={"fontname": "Arial", "shape": "box"},
        other_module_node_defaults={"fontname": "Arial", "shape": "ellipse"},
    )

    format = (
        args.output_file.suffix[1:] if args.output_file.suffix else "raw"
    )
    module_graph.write(args.output_file, format=format)
