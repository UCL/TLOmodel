import pytest
from pandas.tests.extension.base import BaseDtypeTests

from tlo.bitsethandler.bitsethandler import BitsetDtype

@pytest.fixture
def dtype():
    return BitsetDtype()


class TestMyDtype(BaseDtypeTests):
    pass