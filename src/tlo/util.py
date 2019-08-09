"""This file contains helpful utility functions."""
from collections import defaultdict

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


def transition_states(initial_df: pd.DataFrame, state_column: str, prob_matrix: pd.DataFrame,
                      seed: int = None) -> pd.Series:
    """Carrys out transitions on all states based on probability matrix

    This should carry out all state transitions for a specific state_column
    based on the probability of state-transition between each of the states.
    Cumulative sum for these per original state, so the same random sample
    values can be used for all transitions from one state to every other

    :param DataFrame initial_df: the initial dataframe
    :param String state_column: the column to update for the probability matrix
    :param DataFrame prob_matrix: DataFrame of state-transition probabilities
        columns are the original state, rows are the new state. values are the probabilities
    :param integer seed: should only be set during testing
    :return: Series with states changed according to probabilities
    """
    # Create new object with just the state and is_alive column to avoid copying all of the data?
    # Or is pandas sufficiently clever that it just links it even if you're doing it through
    # function calls?
    df = initial_df.loc[:, [state_column, 'is_alive']]

    rng = np.random.RandomState(seed)

    # columns are original state, rows are the new state
    all_states = prob_matrix.columns.tolist()
    prob_matrix.rename(
        index={row_num: col_name for row_num, col_name in
               zip(range(len(prob_matrix)), all_states)}
    )
    summed_prob = prob_matrix.cumsum(axis=0)

    # Couldn't find a way to apply random_sample directly into df so...
    # I think because of bug in 0.23.1 which means you can't use kwags for apply? Upgrade?
    # Random draw per state: series > dataframe > melted by state > merge into df
    random_draw = pd.DataFrame(
        df.groupby(state_column).size().apply(lambda x: rng.random_sample(x)).to_frame('random_draw')
    )
    random_draw = (pd.melt(random_draw.random_draw.apply(pd.Series).reset_index(),
                           id_vars=['state'],
                           value_name='random_draw')
                   ).drop(['variable', 'state'], axis=1)
    df = pd.concat([df, random_draw], axis=1)

    # default state is that none of the states have been updated
    df['state_updated'] = False

    # Create dictionary of original states to new states
    state_transitions = defaultdict(list)
    for original_state in all_states:
        for new_state in all_states:
            if not np.isnan(summed_prob.loc[original_state, new_state]):
                state_transitions[original_state].append(new_state)

    # Update all transition states using the dictionary of productive transitions
    # I feel like I shouldn't be looping but haven't worked out a better way
    for original_state in state_transitions.keys():
        changeable_states = df.loc[
            df.is_alive & (df[state_column] == original_state) & ~df.state_updated
            ]
        random_draw = changeable_states.loc[
            changeable_states[state_column] == original_state, 'random_draw'
        ]
        # Go through states in reverse so that the smaller probabilities are not overwritten
        for new_state in reversed(state_transitions[original_state]):
            probability = summed_prob.loc[new_state, original_state]
            updating_state = changeable_states.index[probability > random_draw]
            df.loc[updating_state, state_column] = new_state
            df.loc[updating_state, 'state_updated'] = True

    return df[state_column]
