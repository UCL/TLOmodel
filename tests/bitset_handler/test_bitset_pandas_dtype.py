from typing import Set

import numpy as np
from numpy.random import Generator, PCG64
import pytest
from pandas.tests.extension.base import BaseDtypeTests

from tlo.bitset_handler.bitset_extension import BitsetDtype, BitsetArray, NodeType

@pytest.fixture(scope="session")
def _rng() -> Generator:
    return Generator(PCG64(seed=0))


@pytest.fixture(scope="session")
def _set_elements() -> Set[NodeType]:
    return {"1", "2", "3", "4", "5", "a", "b", "c", "d", "e"}


@pytest.fixture(scope="session")
def dtype(_set_elements: Set[int | str]) -> BitsetDtype:
    return BitsetDtype(_set_elements)

@pytest.fixture(scope="session")
def data(_rng: Generator, _set_elements: Set[int | str], dtype: BitsetDtype) -> BitsetArray:
    elements = list(_set_elements)
    data = np.zeros((100,), dtype=dtype.np_array_dtype)
    data[0] = dtype.as_bytes({"1", "e"})
    data[1] = dtype.as_bytes({"a", "d"})
    data[2] = dtype.as_bytes({"2", "4", "5"})
    for i in range(3, data.shape[0]):
        data[i] = dtype.as_bytes(
            {
                elements[i]
                for i in _rng.integers(
                    0, len(elements), size=_rng.integers(0, len(elements))
                )
            }
        )
    return BitsetArray(data, dtype=dtype, copy=True)

@pytest.fixture
def data_for_twos(dtype):
    pytest.skip(f"{dtype} does not support divmod")


@pytest.fixture
def data_missing(dtype):
    data = np.zeros((2,), dtype=dtype.np_array_dtype)
    data[0] = dtype.na_value
    data[1] = dtype.as_bytes({"a"})
    return data


class TestMyDtype(BaseDtypeTests):
    """
    Setting the dtype fixture, above, to out BitsetDtype results in us inheriting
    all default pandas tests for extension Dtypes.

    Additional tests can be added to this class if we so desire.
    """
