"""This needs `pydot` and `graphviz` package installed.
 * `pip install pydot`
 * `brew install graphviz`
"""

from pathlib import Path

import pandas as pd
import pydot

from tlo.analysis.utils import (
    get_color_short_treatment_id,
    get_filtered_treatment_ids,
    order_of_short_treatment_ids,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Plot a graph off the TREATMENT_IDs defined in the model."""

    all = pd.Series(get_filtered_treatment_ids())

    base = pd.Series(index=all.index, data="*")
    zero_level_up = all.apply(lambda s: "_".join(c for i, c in enumerate(s.split("_")) if i < 1))
    one_level_up = all.apply(lambda s: "_".join(c for i, c in enumerate(s.split("_")) if i < 2))
    two_level_up = all.apply(lambda s: "_".join(c for i, c in enumerate(s.split("_")) if i < 3))
    # todo generalise to any depth

    split = pd.concat([base, zero_level_up, one_level_up, two_level_up], axis=1)
    split = split.set_index(zero_level_up + '*')
    split = split.loc[order_of_short_treatment_ids(split.index)]  # ordering

    graph = pydot.Dot(graph_type='digraph',
                      rankdir='LR',
                      strict=True,
                      )
    for short_treatment_id, row in split.iterrows():

        graph.add_node(pydot.Node(name="*", label="*", style='filled', fillcolor="white", shape='box'))

        color = get_color_short_treatment_id(short_treatment_id)

        def plot_from_level(_l):
            origin_node = row[_l]
            destination = row[_l + 1]
            graph.add_node(
                pydot.Node(name=destination, label=destination, style='filled', fillcolor=color, shape='box')
            )

            if (origin_node != destination):
                graph.add_edge(pydot.Edge(origin_node, destination))

        plot_from_level(0)
        plot_from_level(1)
        plot_from_level(2)

    graph.write_png(output_folder / "treatment_ids.png")


if __name__ == "__main__":
    apply(
        results_folder=None,
        output_folder=Path('./outputs'),
        resourcefilepath=None
    )
