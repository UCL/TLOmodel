"""Unit tests for utility functions."""

import pandas as pd
import pytest

import tlo.util
from tlo import Date


@pytest.fixture
def mock_pop():
    class Population:
        def _create_props(self, n):
            return pd.DataFrame(
                {'col1': [1.0] * n,
                 'col2': [''] * n,
                 'col3': [[]] * n,
                 'col4': [Date(0)] * n})
    return Population()


@pytest.fixture
def mock_sim(mock_pop):
    class Simulation:
        population = mock_pop
    return Simulation()


def test_show_changes(mock_sim):
    initial = pd.DataFrame(
        {'col1': [1.0, 2.0],
         'col2': ['a', 'b'],
         'col3': [[], [1]],
         'col4': [Date(2020, 1, 1), None]})
    initial.index.name = 'row'
    final = pd.DataFrame(
        {'col1': [3.0, 2.0, 1.0],
         'col2': ['a', 'c', ''],
         'col3': [[2], [1], []],
         'col4': [Date(2020, 1, 1), Date(2020, 1, 2), None]})
    final.index.name = 'row'

    changes = tlo.util.show_changes(mock_sim, initial, final)
    html = changes.render()

    assert len(changes.index) == len(final)
    assert html.count('color:  black;') == 7  # unchanged
    assert html.count('color:  red;') == 5    # changed
    assert html.count('background-color:  yellow;') == 4  # final row


def test_show_changes_same_size(mock_sim):
    initial = pd.DataFrame(
        {'col1': [1.0, 2.0],
         'col2': ['a', 'b'],
         'col3': [[], [1]],
         'col4': [Date(2020, 1, 1), None]})
    initial.index.name = 'row'
    final = pd.DataFrame(
        {'col1': [1.5, 2],
         'col2': ['aa', 'b'],
         'col3': [[], [1, 1]],
         'col4': [Date(2020, 1, 2), None]})
    final.index.name = 'row'

    changes = tlo.util.show_changes(mock_sim, initial, final)
    html = changes.render()

    assert len(changes.index) == len(final)
    assert html.count('color:  black;') == 4  # unchanged
    assert html.count('color:  red;') == 4    # changed
    assert html.count('background-color:  yellow;') == 0
