"""This file contains helpful utility functions."""
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pandas.io.json import normalize

from tlo import Population


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


class BitsetHandler():
    def __init__(self, population: Population, column: str, elements: List[str]):
        """Provides functions to operate on an int column in the population dataframe as a bitset

        :param population: The TLO Population object (not the props dataframe)
        :param column: The integer property column that will be used as a bitset
        :param elements: A list of strings specifying the elements of the bitset
        :returns: instance of BitsetHandler for supplied arguments
        """
        assert isinstance(population, Population), 'First argument is the population object (not the `props` dataframe)'
        assert column in population.props.columns, 'Column not found in population dataframe'
        self._population = population
        self._column: str = column
        self._elements: List[str] = elements
        self._lookup = {k: 2 ** i for i, k in enumerate(elements)}
        self._lookup.update(dict((v, k) for k, v in self._lookup.items()))

    @property
    def df(self) -> pd.DataFrame:
        return self._population.props

    def element_repr(self, element: str) -> str:
        """Returns integer representation of the specified element"""
        return self._lookup[element]

    def set(self, where, *elements: str) -> None:
        """For individuals matching `where` argument, set the bit (i.e. set to True) the given elements

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call. It can be index
        items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset

        :param where: condition to filter rows that will be set
        :param elements: one or more elements to set to True"""
        value = sum([self._lookup[x] for x in elements])
        self.df.loc[where, self._column] = np.bitwise_or(self.df.loc[where, self._column], value)

    def unset(self, where, *elements: str) -> None:
        """For individuals matching `where` argument, unset the bit (i.e. set to False) the given elements

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call. It can be index
        items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset

        :param where: condition to filter rows that will be unset
        :param elements: one or more elements to set to False"""
        value = sum([self._lookup[x] for x in elements])
        value = np.invert(np.array(value, np.int64))
        self.df.loc[where, self._column] = np.bitwise_and(self.df.loc[where, self._column], value)

    def has_all(self, where, *elements: str, first=False) -> Union[pd.Series, bool]:
        """Check whether individual(s) have all the elements given set to True

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call. It can be index
        items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset

        If a single individual is supplied as the where clause, returns a bool, otherwise a Series of bool

        :param where: condition to filter rows that will checked
        :param elements: one or more elements to set to True
        :param first: a keyword argument to return first item in Series instead of Series"""
        value = sum([self._lookup[x] for x in elements])
        matched = np.bitwise_and(self.df.loc[where, self._column], value) == value
        if first:
            return matched.iloc[0]
        return matched

    def has_any(self, where, *elements: str, first=False):
        """Sister method to `has_all` but instead checks whether matching rows have any of the elements
        set to True"""
        matched = pd.Series(False, index=self.df.index[where])
        for element in elements:
            matched = matched.where(matched, self.has_all(where, element))
        if first:
            return matched.iloc[0]
        return matched

    def get(self, where, first=False):
        """Returns a Series with set of string of elements where bit is True

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call. It can be index
        items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset

        :param where: condition to filter rows that will returned
        """
        def int_to_set(integer):
            bin_repr = format(integer, 'b')
            return {self._lookup[2 ** k] for k, v in enumerate(reversed(list(bin_repr))) if v == '1'}
        sets = self.df.loc[where, self._column].apply(int_to_set)
        if first:
            return sets.iloc[0]
        return sets

    def compress(self, uncompressed: pd.DataFrame) -> None:
        def convert(column):
            value_of_column = self._lookup[column.name]
            return column.replace({True: value_of_column, False: 0})
        collapsed = uncompressed.apply(convert).sum(axis=1).astype('int64')
        self.df.loc[uncompressed.index, self._column] = collapsed

    def uncompress(self, where=None) -> pd.DataFrame:
        """Returns an exploded representation of the bitset

        Each element bit becomes a column and each column is a bool indicating whether the bit is
        set for the element
        """
        if where is None:
            where = self.df.index
        collect = dict()
        for element in self._elements:
            collect[element] = self.has_all(where, element)
        return pd.DataFrame(collect)

    def clear(self, where) -> None:
        """Clears all the bits for the specified rows"""
        self.df.loc[where, self._column] = 0
