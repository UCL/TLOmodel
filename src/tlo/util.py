"""This file contains helpful utility functions."""
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas.io.json import normalize


def create_age_range_lookup(min_age: int, max_age: int, range_size: int = 5) -> (list, Dict[int, str]):
    """Create age-range categories and a dictionary that will map all whole years to age-range categories

    If the minimum age is not zero then a below minimum age category will be made,
    then age ranges until maximum age will be made by the range size,
    all other ages will map to the greater than maximum age category.

    :param min_age: Minimum age for categories,
    :param max_age: Maximum age for categories, a greater than maximum age category will be made
    :param range_size: Size of each category between minimum and maximum ages
    :returns:
        age_categories: ordered list of age categories available
        lookup: Default dict of integers to maximum age mapping to the age categories
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit
    parts = chunks(range(min_age, max_age), range_size)

    default_category = f'{max_age}+'
    lookup = defaultdict(lambda: default_category)
    age_categories = []

    # create category for minimum age
    if min_age > 0:
        under_min_age_category = f'0-{min_age}'
        age_categories.append(under_min_age_category)
        for i in range(0, min_age):
            lookup[i] = under_min_age_category

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = f'{start}-{end}'
        age_categories.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    age_categories.append(default_category)

    return age_categories, lookup


def nested_to_record(df: pd.DataFrame) -> Dict[str, Any]:
    """Transform dataframe into flattened dictionary,
    a convenience wrapper for nested_to_record (from pandas.io.json.normalize) on a df.to_dict() output

    Dictionary keys are in the form f'{column_name}_{index_name}'

    :param df: Input dataframe
    :return: flattened dictionary
    """
    df = df.copy()
    # convert index and column to string as keys are required to be strings for nested_to_record
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return normalize.nested_to_record(df.to_dict(), sep="_")


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


def transition_states(initial_series: pd.Series, prob_matrix: pd.DataFrame, rng: np.random.RandomState) -> pd.Series:
    """Transition a series of states based on probability matrix

    This should carry out all state transitions for a Series (i.e. column in DataFrame)
    based on the probability of state-transition matrix.

    Timing values for 1M rows per state, 4 states, 100 times:
    - Looping through groups: [59.5, 58.7, 59.5]
    - Using apply: [84.2, 83.3, 84.4]
    Because of this, looping through the groups was chosen

    :param Series initial_series: the initial state series
    :param DataFrame prob_matrix: DataFrame of state-transition probabilities
        columns are the original state, rows are the new state. values are the probabilities
    :param RandomState rng: RandomState from the disease module
    :return: Series with states changed according to probabilities
    """
    # Create final_series with index so that we are sure it's the same size as the original
    final_states = pd.Series(None, index=initial_series.index, dtype=initial_series.dtype)

    # for each state, get the random choice states and add to the final_states Series
    state_indexes = initial_series.groupby(initial_series).groups
    all_states = prob_matrix.index.tolist()
    for state, state_index in state_indexes.items():
        if not state_index.empty:
            new_states = rng.choice(all_states, len(state_index), p=prob_matrix[state])
            final_states[state_index] = new_states
    return final_states

def choose_outcome_from_a_dict(dict_of_probs: dict, rng: np.random.RandomState) -> str:
    """
    This is a helper function that will chose an outcome from a dict that is organsised as {outcome : p_outcome}

    :param dict_of_probs:
    :param rng:
    :return:
    """

    outcomes = np.array(list(dict_of_probs.keys()))
    probs = np.array(list(dict_of_probs.values()))

    return str(rng.choice(outcomes, p=probs))
