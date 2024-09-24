"""
Implements the fixtures required in 
https://github.com/pandas-dev/pandas/blob/bdb509f95a8c0ff16530cedb01c2efc822c0d314/pandas/core/dtypes/dtypes.py,

which allows us to run the pandas-provided test suite for custom dtypes.
Additional tests and fixtures can be defined on top of those listed in the link above, if we want to
run our own tests.
"""

from typing import List, Set

import numpy as np
import pytest
from numpy.dtypes import BytesDType  # pylint: disable=E0611
from numpy.random import PCG64, Generator
from numpy.typing import NDArray

from tlo.bitset_handler.bitset_extension import BitsetArray, BitsetDtype, ElementType


@pytest.fixture(scope="session")
def _rng() -> Generator:
    return Generator(PCG64(seed=0))


@pytest.fixture(scope="session")
def _set_elements() -> Set[ElementType]:
    return {"1", "2", "3", "4", "5", "a", "b", "c", "d", "e"}


@pytest.fixture(scope="session")
def dtype(_set_elements: Set[ElementType]) -> BitsetDtype:
    return BitsetDtype(_set_elements)


@pytest.fixture(scope="session")
def _1st_3_entries() -> List[Set[ElementType]]:
    """
    We will fix the first 3 entries of the data fixture,
    which is helpful to ensure we have some explicit test
    values that we can directly change if needed.
    """
    return [
        {"1", "e"}, {"a", "d"}, {"2", "4", "5"},
    ]

@pytest.fixture(scope="session")
def _raw_sets(
    _1st_3_entries: List[Set[ElementType]], _rng: Generator, _set_elements: Set[ElementType]
) -> List[Set[ElementType]]:
    """
    Length 100 list of sets, the first 3 of which are those in
    the _1st_3_entries fixture. These sets will be used as the
    'raw_data' for the Bitset Extension test suite.
    """
    set_entries = list(_1st_3_entries)
    elements = list(_set_elements)
    for _ in range(100-len(_1st_3_entries)):
        set_entries.append(
            {
                elements[i]
                for i in _rng.integers(
                    0, len(elements), size=_rng.integers(0, len(elements))
                )
            }
        )
    return set_entries

@pytest.fixture(scope="session")
def _raw_data(
    _raw_sets: List[Set[ElementType]], dtype: BitsetDtype
) -> NDArray[np.bytes_]:
    data = np.zeros((100,), dtype=dtype.np_array_dtype)
    for i, set_value in enumerate(_raw_sets):
        data[i] = dtype.as_bytes(set_value)
    return data


@pytest.fixture(scope="session")
def data(
    _raw_data: NDArray[np.bytes_], dtype: BitsetDtype
) -> BitsetArray:
    return BitsetArray(data=_raw_data, dtype=dtype, copy=True)


@pytest.fixture
def data_for_twos(dtype: BitsetDtype) -> None:
    pytest.skip(f"{dtype} does not support divmod")


@pytest.fixture
def data_missing(dtype: BitsetDtype) -> np.ndarray[BytesDType]:
    data = np.zeros((2,), dtype=dtype.np_array_dtype)
    data[0] = dtype.na_value
    data[1] = dtype.as_bytes({"a"})
    return data
