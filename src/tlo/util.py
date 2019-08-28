"""This file contains helpful utility functions."""

import json
from typing import Dict

import numpy as np
import pandas as pd

from tlo import Parameter


def load_parameters(resource: pd.DataFrame, parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
    """Automatically load parameters from resource dataframe, returning updated parameter dictionary

    Automatically updates the values of data types:
        - Integers
        - Real numbers
        - Lists
        - Categorical
        - Strings

    :param Dict parameters: Module's parameters
    :param DataFrame resource: DataFrame with index of the parameter_name and a column of `value`
    :return: parameters dictionary updated with values from the resource dataframe
    """
    # should parse BOOL and DATE, just need to write some tests if we want these?
    skipped_data_types = ('BOOL', 'DATA_FRAME', 'DATE', 'SERIES')

    # for each supported parameter, convert to the correct type
    for parameter_name, parameter_definition in parameters.items():
        if parameter_definition.type_.name in skipped_data_types:
            continue

        # For each parameter, raise error if the value can't be coerced
        # Could the Exception message for a parameter that isn't in the resource more explicit? currently:
        # 'the label [int_basic] is not in the [index]'
        parameter_value = resource.loc[parameter_name, 'value']
        error_message = (f"The value of '{parameter_value}' for parameter '{parameter_name}' "
                         f"could not be parsed as a {parameter_definition.type_.name} data type")
        if parameter_definition.python_type == list:
            try:
                # chose json.loads instead of save_eval
                # because it raises error instead of joining two strings without a comma
                parameter_value = json.loads(parameter_value)
                assert isinstance(parameter_value, list)
            except (json.decoder.JSONDecodeError, AssertionError) as e:
                raise ValueError(error_message) from e
        elif parameter_definition.python_type == pd.Categorical:
            categories = parameter_definition.categories
            assert parameter_value in categories, f"{error_message}\nvalid values: {categories}"
            parameter_value = pd.Categorical(parameter_value, categories=categories)
        elif parameter_definition.type_.name == 'STRING':
            parameter_value = parameter_value.strip()
        else:
            # All other data types
            try:
                parameter_value = parameter_definition.python_type(parameter_value)
            except Exception as e:
                raise ValueError(error_message) from e

        # Save the values to the parameters
        parameters[parameter_name] = parameter_value

    return parameters


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
    all_states = prob_matrix.columns.tolist()
    for state, state_index in state_indexes.items():
        new_states = rng.choice(
            all_states, len(state_index), p=prob_matrix[state]
        )
        final_states[state_index] = new_states
    return final_states
