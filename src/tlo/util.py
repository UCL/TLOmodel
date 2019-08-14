"""This file contains helpful utility functions."""
import timeit  # remove after development

import pandas as pd
import numpy as np


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


def transition_states(initial_series: pd.Series, prob_matrix: pd.DataFrame, seed: int = None) -> pd.Series:
    """Transitions states based on probability matrix

    This should carry out all state transitions for a specific state_column
    based on the probability of state-transition between each of the states.
    Timing values for 1M rows per state, 4 states, 100 times:
    - Looping through groups: [59.5, 58.7, 59.5]
    - Using apply: [84.2, 83.3, 84.4]
    Because of this, looping through the groups was chosen

    :param Series initial_series: the initial state series
    :param DataFrame prob_matrix: DataFrame of state-transition probabilities
        columns are the original state, rows are the new state. values are the probabilities
    :param integer seed: should only be set during testing
    :return: Series with states changed according to probabilities
    """
    # TODO: assert that columns add up to one during initialisation of disease?
    rng = np.random.RandomState(seed)
    final_states = pd.Series()

    # for each state, get the random choice states and add to the final_states Series
    state_indexes = initial_series.groupby(initial_series).groups
    all_states = prob_matrix.columns.tolist()
    for state in all_states:
        try:
            state_index = state_indexes[state]
        except KeyError:
            # no initial state, not going to be transitioning. skip over step in loop
            continue
        new_states = rng.choice(
            all_states, len(state_index), p=prob_matrix[state]
        )
        final_states = final_states.append(
            pd.Series(new_states, index=state_index)
        )
    return final_states
