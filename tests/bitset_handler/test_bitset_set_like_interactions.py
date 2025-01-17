"""
Tests for set-like interactions with a pd.Series object of BitsetDtype.
"""
import operator
from typing import Any, Callable, Iterable, List, Set

import pandas as pd
import pytest

from tlo.bitset_handler.bitset_extension import BitsetDtype, CastableForPandasOps, ElementType


def seq_of_sets_to_series(sets: Iterable[Set[ElementType]], dtype: BitsetDtype) -> pd.Series:
    """
    Casts a sequence of sets representing a single BitsetDtype to a
    series with those entries of the appropriate dtype.
    """
    return pd.Series(data=sets, dtype=dtype, copy=True)


@pytest.fixture(scope="function")
def small_series(_1st_3_entries: List[Set[ElementType]], dtype: BitsetDtype):
    """
    Recall that the first 3 entries are always fixed in confest;
    repeating the values here just for ease of reference:

    {"1", "e"},
    {"a", "d"},
    {"2", "4", "5"},
    """
    return pd.Series(_1st_3_entries, dtype=dtype, copy=True)


@pytest.mark.parametrize(
    ["op", "r_value", "expected"],
    [
        pytest.param(
            [operator.or_, operator.add, operator.sub],
            set(),
            [{"1", "e"}, {"a", "d"}, {"2", "4", "5"}],
            id="ADD, OR, SUB w/ empty set",
        ),
        pytest.param(
            [operator.or_, operator.add],
            "a",
            [{"1", "a", "e"}, {"a", "d"}, {"2", "4", "5", "a"}],
            id="ADD, OR w/ scalar element",
        ),
        pytest.param(
            [operator.or_, operator.add],
            {"1", "2", "a", "d"},
            [
                {"1", "2", "a", "d", "e"},
                {"1", "2", "a", "d"},
                {"1", "2", "4", "5", "a", "d"},
            ],
            id="ADD, OR w/ multiple-entry set",
        ),
        pytest.param(
            operator.and_,
            set(),
            [set()] * 3,
            id="AND w/ empty set",
        ),
        pytest.param(
            operator.and_,
            "a",
            [set(), {"a"}, set()],
            id="AND w/ scalar element",
        ),
        pytest.param(
            operator.and_,
            {"1", "a"},
            [{"1"}, {"a"}, set()],
            id="AND w/ multiple-entry set",
        ),
        pytest.param(
            [operator.eq, operator.le, operator.lt],
            set(),
            pd.Series([False, False, False], dtype=bool),
            id="EQ, LE, LT w/ empty set",
        ),
        pytest.param(
            [operator.eq, operator.le, operator.lt],
            "a",
            pd.Series([False, False, False], dtype=bool),
            id="EQ, LE, LT w/ scalar element",
        ),
        pytest.param(
            [operator.eq, operator.ge, operator.le],
            {"1", "e"},
            pd.Series([True, False, False], dtype=bool),
            id="EQ, GE, LE w/ multiple-entry set",
        ),
        pytest.param(
            [operator.ge, operator.gt],
            set(),
            pd.Series([True, True, True], dtype=bool),
            id="GE, GT w/ empty set",
        ),
        pytest.param(
            [operator.ge, operator.gt],
            "a",
            pd.Series([False, True, False], dtype=bool),
            id="GE, GT w/ scalar element",
        ),
        pytest.param(
            [operator.gt, operator.lt],
            {"1, e"},
            pd.Series([False, False, False], dtype=bool),
            id="GT, LT w/ multiple-entry set",
        ),
        pytest.param(
            operator.sub,
            "a",
            [{"1", "e"}, {"d"}, {"2", "4", "5"}],
            id="SUB w/ scalar element",
        ),
        pytest.param(
            operator.sub,
            {"1", "2", "d", "e"},
            [set(), {"a"}, {"4", "5"}],
            id="SUB w/ multiple-entry set",
        ),
    ],
)
def test_series_operation_with_value(
    small_series: pd.Series,
    dtype: BitsetDtype,
    op: List[Callable[[Any, Any], Any]] | Callable[[Any, Any], Any],
    r_value: CastableForPandasOps,
    expected: List[Set[ElementType]] | pd.Series
) -> None:
    """
    The expected value can be passed in as either a list of sets that will be
    converted to the appropriate pd.Series of bitsets, or as an explicit pd.Series
    of booleans (which is used when testing the comparison operations ==, <=, etc).

    If r_value is a scalar, the test will run once using the scalar as the r_value,
    and then again using the cast of the scalar to a set of one element as the r_value.
    - In cases such as this, the two results are expected to be the same,
      which saves us verbiage in the list of test cases above.
    """
    expected = (
        seq_of_sets_to_series(expected, dtype)
        if isinstance(expected, list)
        else expected
    )

    if not isinstance(op, list):
        op = [op]
    if isinstance(r_value, ElementType):
        r_values = [r_value, {r_value}]
    else:
        r_values = [r_value]

    for operation in op:
        for r_v in r_values:
            result = operation(small_series, r_v)
            assert (
                expected == result
            ).all(), f"Series do not match after operation {operation.__name__} with {r_v} on the right."
