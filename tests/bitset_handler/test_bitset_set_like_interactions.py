"""
Tests for set-like interactions with a pd.Series object of BitsetDtype.
"""
from typing import Iterable, List, Set

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
# add
# and
# eq
# ge, gt, le, lt
# or
# sub(tract)


@pytest.mark.parametrize(
    ["to_add", "where", "expected"],
    [
        pytest.param(
            set(),
            None,
            [{"1", "e"}, {"a", "d"}, {"2", "4", "5"}],
            id="Adding nothing does nothing.",
        ),
        pytest.param(
            set(),
            0,
            [{"1", "e"}],
            id="Adding nothing does nothing, even when selecting a single element",
        ),
        pytest.param(
            {"a"},
            None,
            [{"1", "a", "e"}, {"a", "d"}, {"2", "4", "5", "a"}],
            id="Adding a single element to all sets.",
        ),
        pytest.param(
            {"a"},
            slice(2),
            [{"1", "a", "e"}, {"a", "d"}],
            id="Manipulate only the first 2 entries.",
        ),
        pytest.param(
            {"1", "2", "d"},
            1,
            [{"1", "2", "a", "d"}],
            id="Manipulate only a single entry.",
        ),
        pytest.param(
            {"1", "2", "a", "d"},
            slice(2),
            [
                {"1", "2", "a", "d", "e"},
                {"1", "2", "a", "d"},
            ],
            id="Add sets of multiple elements to a slice of the series.",
        ),
    ],
)
def test_or(
    small_series: pd.Series,
    dtype: BitsetDtype,
    to_add: CastableForPandasOps,
    where: int | slice | Iterable[int] | None,
    expected: List[Set[ElementType]]
):
    # Note that expected will be compared against
    # small_series[where] so should only contain the
    # entries it expects to see!
    expected = seq_of_sets_to_series(expected, dtype)

    # Assume index "None" to mean "whole series"
    result = (
        small_series[where] | to_add if where is not None else small_series | to_add
    )

    assert (expected == result).all(), f"Series do not match after adding {to_add}"

    # Using _add__ should give the same result, since
    # __or__ delegates to __add__.
    if not isinstance(where, int):
        # Sets do not have + defined on them, so we do not have a
        # delegation method for when we extract a single value from
        # the series.
        result_from_adding = (
            small_series[where] + to_add if where is not None else small_series + to_add
        )

        assert (result_from_adding == result).all(), "Using + in place of | returned different values, despite delegation!"
