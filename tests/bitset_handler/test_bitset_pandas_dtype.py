import re

import pytest
from pandas.tests.extension.base import BaseDtypeTests

from tlo.bitset_handler.bitset_extension import BitsetDtype


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
