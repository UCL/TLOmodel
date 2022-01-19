"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

import tlo.util
from tlo import Date


@pytest.fixture
def mock_pop():
    class Population:
        def _create_props(self, n):
            return pd.DataFrame({'col1': [1.0] * n, 'col2': [''] * n, 'col3': [[]] * n, 'col4': [Date(0)] * n})

    return Population()


@pytest.fixture
def mock_sim(mock_pop):
    class Simulation:
        population = mock_pop

    return Simulation()


class TestTransitionsStates:
    def setup(self):
        # create rng for testing
        self.rng = np.random.RandomState(1234)

        self.states = list("abcd")
        prob_matrix = pd.DataFrame(columns=self.states, index=self.states)
        # key is original state, values are probability for new states
        #                   A    B    C    D
        prob_matrix["a"] = [0.9, 0.1, 0.0, 0.0]
        prob_matrix["b"] = [0.1, 0.3, 0.6, 0.0]
        prob_matrix["c"] = [0.0, 0.2, 0.6, 0.2]
        prob_matrix["d"] = [0.0, 0.0, 0.3, 0.7]
        # columns are original state, rows are the new state
        all_states = prob_matrix.columns.tolist()
        prob_matrix.rename(index={row_num: col_name for row_num, col_name in zip(range(len(prob_matrix)), all_states)})
        self.prob_matrix = prob_matrix

        # default input data
        self.input = pd.DataFrame({'state': self.states * 1_000, 'other_data_1': range(0, 4_000)})

        # default output data
        nested_states = [  # perfect ratio would be: [1000, 600, 1500, 900]
            list(state * repeat) for state, repeat in zip(self.states, [1018, 598, 1465, 919])
        ]
        self.expected = pd.DataFrame({'state': sum(nested_states, []), 'other_data_1': pd.Series(range(0, 4000))})

    def test_simple_case(self):
        """Simple case, should change to probabilities found per seed"""

        expected = self.expected.copy()

        # run the function
        output = tlo.util.transition_states(self.input.state, self.prob_matrix, rng=self.rng)
        output_merged = self.input.copy()
        output_merged['state'] = output
        expected_size = expected.groupby('state').size()
        output_size = output_merged.groupby('state').size()
        pd.testing.assert_series_equal(expected_size, output_size)

    def test_state_doesnt_transition(self):
        """State d shouldn't transition at all"""
        nested_states = [  # perfect ratio would be: [1000, 600, 1400, 1000]
            list(state * repeat) for state, repeat in zip(self.states, [1018, 598, 1384, 1000])
        ]
        expected = pd.DataFrame({'state': sum(nested_states, []), 'other_data_1': pd.Series(range(0, 4000))})
        prob_matrix = self.prob_matrix.copy()
        prob_matrix['c'] = [0.0, 0.2, 0.8, 0.0]
        prob_matrix['d'] = [0.0, 0.0, 0.0, 1.0]

        # run the function
        output = tlo.util.transition_states(self.input.state, prob_matrix, rng=self.rng)
        output_merged = self.input.copy()
        output_merged['state'] = output
        expected_size = expected.groupby('state').size()
        output_size = output_merged.groupby('state').size()
        pd.testing.assert_series_equal(expected_size, output_size)

    def test_none_in_initial_state(self):
        """States start without any in state c"""
        # input
        nested_states = [list(state * repeat) for state, repeat in zip(self.states, [2000, 2000, 0, 2000])]
        input = pd.DataFrame({'state': sum(nested_states, []), 'other_data_1': pd.Series(range(0, 6000))})

        # default output data
        nested_states = [  # perfect ratio would be: [2000, 800, 1800, 1400]
            list(state * repeat) for state, repeat in zip(self.states, [2006, 760, 1853, 1381])
        ]
        self.expected = pd.DataFrame({'state': sum(nested_states, []), 'other_data_1': pd.Series(range(0, 6000))})
        expected = self.expected.copy()

        # run the function
        output = tlo.util.transition_states(input.state, self.prob_matrix, rng=self.rng)
        output_merged = input.copy()
        output_merged['state'] = output
        expected_size = expected.groupby('state').size()
        output_size = output_merged.groupby('state').size()
        pd.testing.assert_series_equal(expected_size, output_size)

    def test_transition_removes_state(self):
        """Transition causes complete removal of a state a from the series"""
        new_states = list('bcd')
        nested_states = [  # perfect ratio would be: [1600, 1500, 900]
            list(state * repeat) for state, repeat in zip(new_states, [1616, 1465, 919])
        ]
        expected = pd.DataFrame({'state': sum(nested_states, []), 'other_data_1': pd.Series(range(0, 4000))})
        prob_matrix = self.prob_matrix.copy()
        prob_matrix["a"] = [0.0, 1.0, 0.0, 0.0]
        prob_matrix["b"] = [0.0, 0.4, 0.6, 0.0]

        # run the function
        output = tlo.util.transition_states(self.input.state, prob_matrix, rng=self.rng)
        output_merged = self.input.copy()
        output_merged['state'] = output
        expected_size = expected.groupby('state').size()
        output_size = output_merged.groupby('state').size()
        pd.testing.assert_series_equal(expected_size, output_size)

    def test_type_is_maintained(self):
        """Given a categorical input, this data type should be maintained"""
        categorical_input = pd.Series(pd.Categorical(self.input.state))

        # run the function
        output = tlo.util.transition_states(categorical_input, self.prob_matrix, rng=self.rng)
        print(output.dtype)
        assert output.dtype == categorical_input.dtype


class TestCreateAgeRangeLookup:
    def test_default_range(self):
        """
        Given just a minimum and maximum age,
        should get a below age category, ranges of 5 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(10, 20)
        assert ranges == ["0-10", "10-14", "15-19", "20+"]
        for i in range(0, 10):
            assert lookup[i] == ranges[0]
        for i in range(10, 15):
            assert lookup[i] == ranges[1]
        for i in range(15, 20):
            assert lookup[i] == ranges[2]
        for i in range(20, 25):
            assert lookup[i] == ranges[3]

    def test_user_range(self):
        """
        Given just a minimum age, maximum age, and a range size of 10
        should get a below age category, ranges of 10 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(10, 30, 10)
        assert ranges == ["0-10", "10-19", "20-29", "30+"]
        for i in range(0, 10):
            assert lookup[i] == ranges[0]
        for i in range(10, 20):
            assert lookup[i] == ranges[1]
        for i in range(20, 30):
            assert lookup[i] == ranges[2]
        for i in range(30, 40):
            assert lookup[i] == ranges[3]

    def test_no_under_min_age(self):
        """
        Given just a minimum age of zero, and a maximum age
        should get a ranges of 5 and then an above maximum age category
        """
        ranges, lookup = tlo.util.create_age_range_lookup(0, 10)
        assert ranges == ["0-4", "5-9", "10+"]
        for i in range(0, 5):
            assert lookup[i] == ranges[0]
        for i in range(5, 10):
            assert lookup[i] == ranges[1]
        for i in range(10, 15):
            assert lookup[i] == ranges[2]


@pytest.mark.slow
def test_sample_outcome(tmpdir):
    """Check that helper function `sample_outcome` works correctly."""

    # Create probability matrix for four individual (0-3) with four possible outcomes (A, B, C).
    probs = pd.DataFrame({
        'A': {0: 1.0, 1: 0.0, 2: 0.25, 3: 0.0},
        'B': {0: 0.0, 1: 1.0, 2: 0.25, 3: 0.0},
        'C': {0: 0.0, 1: 0.0, 2: 0.50, 3: 0.0},
    })
    rng = np.random.RandomState(seed=0)

    list_of_results = list()
    n = 5000
    for i in range(n):
        list_of_results.append(tlo.util.sample_outcome(probs, rng))
    res = pd.DataFrame(list_of_results)

    assert (res[0] == 'A').all()
    assert (res[1] == 'B').all()
    assert (res[2].isin(['A', 'B', 'C'])).all()
    assert 3 not in res.columns

    for op in ['A', 'B', 'C']:
        prob = probs.loc[2, op]
        assert res[2].value_counts()[op] == pytest.approx(probs.loc[2, op] * n, abs=2 * np.sqrt(n * prob * (1 - prob)))
