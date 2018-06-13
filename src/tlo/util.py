"""This file contains helpful utility functions."""

import pandas as pd


def show_changes(sim, initial_state, final_state):
    """Visually highlight changes in population properties over time.

    This is intended for interactive testing on small populations. It uses
    pandas' styling support to colour changed property values red. Within
    a Jupyter notebook, just calling this function at the end of a cell will
    display colourful output.

    :param Simulation sim: the simulation these states came from
    :param DataFrame initial_state: the initial population properties
    :param DataFrame final_state: the final population properties
    :return: a styled DataFrame
    """
    # Make both DataFrames the same size
    len1, len2 = len(initial_state), len(final_state)
    assert len1 <= len2
    if len1 < len2:
        initial_state = initial_state.append(
            sim.population._create_props(len2 - len1),
            ignore_index=True, sort=False)
    # Figure out which cells changed
    changed = ~(initial_state == final_state)
    changed[pd.isnull(initial_state) & pd.isnull(final_state)] = False
    # Apply styling
    style = changed.applymap(lambda v: 'color: red' if v else 'color: black')
    return final_state.style.apply(lambda df: style, axis=None).applymap(
        lambda cell: 'background-color: yellow',
        subset=pd.IndexSlice[len1:])
