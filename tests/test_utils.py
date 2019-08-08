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


class TestTransitionsStates:
    def setup(self):

        self.states = ['a', 'b', 'c', 'd']

        self.df = pd.DataFrame({'state': self.states * 1000,
                                'other_data_1': range(0, 4000),
                                'is_alive': True})
        self.prob_matrix = pd.DataFrame(columns=self.states, index=self.states)
        # key is original state, values are probability for new states
        #                        A    B     C    D
        self.prob_matrix['a'] = [None, 0.1, None, None]
        self.prob_matrix['b'] = [0.1, None, 0.6, None]
        self.prob_matrix['c'] = [None, 0.2, None, 0.2]
        self.prob_matrix['d'] = [None, None, 0.3, None]
        # set seed for testing. Is there a better way do test this?
        self.seed = 1234

        # repeat each state by the list of integers
        nested_states = [
            list(state * repeat) for state, repeat in zip(self.states, [894, 404, 1606, 1096])
        ]
        self.expected = pd.DataFrame({'state': sum(nested_states, []),
                                 'other_data_1': pd.Series(range(0, 4000)),
                                 'is_alive': True})

    def test_all_alive(self):
        """Simple case, should change roughly to the probabilites given"""
        expected = self.expected.copy()
        output = tlo.util.transition_states(self.df, 'state', self.prob_matrix, seed=self.seed)
        output_merged = self.df.copy()
        output_merged['state'] = output
        expected_size = expected.groupby('state').size()
        output_size = output_merged.groupby('state').size()
        assert (expected_size + output_size).equals(expected_size * 2)

    def test_all_dead(self):
        """No changes should be made if all individuals are dead"""
        df = self.df.copy()
        df['is_alive'] = False
        output = tlo.util.transition_states(df, 'state', self.prob_matrix, seed=self.seed)
        df['state'] = output
        expected_size = self.df.groupby('state').size()
        output_size = df.groupby('state').size()
        assert (expected_size + output_size).equals(expected_size * 2)
