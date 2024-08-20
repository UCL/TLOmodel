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
    return pd.Series(_1st_3_entries, dtype=dtype, copy=True)


# METHODS:
# and
# eq
# ge, gt, le, lt
# sub(tract)

@pytest.mark.parametrize(
    ["op", "r_value", "where", "expected"],
    [
        pytest.param(
            [operator.or_, operator.add],
            set(),
            None,
            [{"1", "e"}, {"a", "d"}, {"2", "4", "5"}],
            id="ADD, OR : Series w/ empty set (adding nothing does nothing)",
        ),
        pytest.param(
            [operator.or_, operator.add],
            {"a"},
            None,
            [{"1", "a", "e"}, {"a", "d"}, {"2", "4", "5", "a"}],
            id="ADD, OR : Series w/ single element set",
        ),
        pytest.param(
            [operator.or_, operator.add],
            {"1", "2", "a", "d"},
            None,
            [
                {"1", "2", "a", "d", "e"},
                {"1", "2", "a", "d"},
                {"1", "2", "4", "5", "a", "d"},
            ],
            id="ADD, OR : Series w/ multiple-entry set",
        ),
        pytest.param(
            operator.or_,
            set(),
            0,
            [{"1", "e"}],
            id="OR : Single entry w/ empty set (adding nothing does nothing)",
        ),
        pytest.param(
            operator.or_,
            {"1", "2", "d"},
            1,
            [{"1", "2", "a", "d"}],
            id="OR : Single entry w/ multiple-entry set",
        ),
    ],
)
def test_operation(
    small_series: pd.Series,
    dtype: BitsetDtype,
    op: List[Callable[[Any, Any], Any]] | Callable[[Any, Any], Any],
    r_value: CastableForPandasOps,
    where: int | None,
    expected: List[Set[ElementType]]
) -> None:
    expected = seq_of_sets_to_series(expected, dtype)
    l_value = small_series[where] if where is not None else small_series

    if not isinstance(op, list):
        op = [op]
    for operation in op:
        # Assume index "None" to mean "whole series"
        result = operation(l_value, r_value)
        assert (
            expected == result
        ).all(), f"Series do not match after operation {operation.__name__} with {r_value} on the right."
