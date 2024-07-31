import re
from typing import Set

import numpy as np
import pytest
from numpy.dtypes import BytesDType
from numpy.random import PCG64, Generator
from pandas.tests.extension.base import BaseDtypeTests

from tlo.bitset_handler.bitset_extension import BitsetArray, BitsetDtype, NodeType


@pytest.fixture(scope="session")
def _rng() -> Generator:
    return Generator(PCG64(seed=0))


@pytest.fixture(scope="session")
def _set_elements() -> Set[NodeType]:
    return {"1", "2", "3", "4", "5", "a", "b", "c", "d", "e"}


@pytest.fixture(scope="session")
def dtype(_set_elements: Set[NodeType]) -> BitsetDtype:
    return BitsetDtype(_set_elements)


@pytest.fixture(scope="session")
def data(
    _rng: Generator, _set_elements: Set[NodeType], dtype: BitsetDtype
) -> BitsetArray:
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
def data_for_twos(dtype: BitsetDtype) -> None:
    pytest.skip(f"{dtype} does not support divmod")


@pytest.fixture
def data_missing(dtype: BitsetDtype) -> np.ndarray[BytesDType]:
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

    def test_construct_from_string_another_type_raises(
        self, dtype: BitsetDtype
    ) -> None:
        """
        Reimplementation as the error message we expect is different from that provided
        by base ``pandas`` implementation.
        """
        msg = (
            "Need at least 2 (comma-separated) elements in string to construct bitset."
        )
        with pytest.raises(TypeError, match=re.escape(msg)):
            type(dtype).construct_from_string("another_type")
