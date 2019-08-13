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


def transition_states(intial_series: pd.Series, prob_matrix: pd.DataFrame,
                      seed: int = None, method='assign') -> pd.Series:
    """Carrys out transitions on all states based on probability matrix

    This should carry out all state transitions for a specific state_column
    based on the probability of state-transition between each of the states.
    Cumulative sum for these per original state, so the same random sample
    values can be used for all transitions from one state to every other

    :param Series intial_series: the initial state series
    :param DataFrame prob_matrix: DataFrame of state-transition probabilities
        columns are the original state, rows are the new state. values are the probabilities
    :param integer seed: should only be set during testing
    :return: Series with states changed according to probabilities
    """

    rng = np.random.RandomState(seed)
    all_states = prob_matrix.columns.tolist()
    if method == 'pipe':
        # create df of current state, merge in probabilities
        df = pd.DataFrame(intial_series).merge(prob_matrix, left_on='state', right_index=True)
        # group by current state, carry out random sample on group with probabilities
        # Think this was too optimistic, doesn't work
        df.groupby('state').pipe(lambda group: rng.choice(all_states, group.size(), p=group.min().values))
    elif method == 'groups':
        final_states = pd.Series()
        df = pd.DataFrame(intial_series).groupby('state')
        for state in all_states:
            new_states = rng.choice(all_states, len(df.groups[state]), p=prob_matrix[state])
            final_states = final_states.append(pd.Series(new_states, index=df.groups[state]))
    else:
        # create dict of starting state ids. state: ids
        initial_states = {key: intial_series[intial_series == key] for key in all_states}
        final_states = pd.Series()
        # numpy random choice per current state
        for state in all_states:
            new_states = rng.choice(all_states, len(initial_states[state]), p=prob_matrix[state])
            if method == 'loop':
                # join using loop
                initial_states[state].loc[:] = new_states
                final_states = final_states.append(initial_states[state])
            elif method == 'assign':
                final_states = final_states.append(pd.Series(new_states, index=initial_states[state].index))
    return final_states


def time_functions(func="transition_states"):
    states = ['a', 'b', 'c', 'd']
    df = pd.DataFrame({'state': states * 10000,
                       'other_data_1': range(0, 40000),
                       'is_alive': True})

    prob_matrix = pd.DataFrame(columns=states, index=states)
    # key is original state, values are probability for new states
    #                   A    B    C    D
    prob_matrix['a'] = [0.9, 0.1, 0.0, 0.0]
    prob_matrix['b'] = [0.1, 0.3, 0.6, 0.0]
    prob_matrix['c'] = [0.0, 0.2, 0.6, 0.2]
    prob_matrix['d'] = [0.0, 0.0, 0.3, 0.7]

    all_states = prob_matrix.columns.tolist()
    # columns are original state, rows are the new state
    prob_matrix.rename(
        index={row_num: col_name for row_num, col_name in
               zip(range(len(prob_matrix)), all_states)}
    )
    if func == "transition_states_loop":
        states: pd.Series = transition_states(df["state"], prob_matrix, method='loop')
    elif func == "transition_states_assign":
        states: pd.Series = transition_states(df["state"], prob_matrix, method='assign')
    elif func == "transition_states_pipe":
        states: pd.Series = transition_states(df["state"], prob_matrix, method='pipe')
    elif func == "transition_states_groups":
        states: pd.Series = transition_states(df["state"], prob_matrix, method='groups')


if __name__ == '__main__':
    print("loop")
    timed = timeit.timeit('time_functions("transition_states_loop")',
                          setup='from __main__ import time_functions',
                          number=100)
    print(timed)
    print("assign")
    timed = timeit.timeit('time_functions("transition_states_assign")',
                          setup='from __main__ import time_functions',
                          number=100)
    print(timed)
    print("groups")
    timed = timeit.timeit('time_functions("transition_states_groups")',
                          setup='from __main__ import time_functions',
                          number=100)
    print(timed)
    print("pipe")
    timed = timeit.timeit('time_functions("transition_states_pipe")',
                          setup='from __main__ import time_functions',
                          number=100)
    print(timed)
