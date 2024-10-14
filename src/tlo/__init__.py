"""
The top-level TLO framework module.

We import our key classes so they're available in the main namespace.

Pandas' Timestamp is used as our main date type (even though it has nanosecond resolution...)
"""
from importlib.metadata import PackageNotFoundError, version

from pandas import Timestamp as Date  # noqa
from pandas.tseries.offsets import DateOffset  # noqa

from .core import Module, Parameter, Property, Types  # noqa
from .population import Population  # noqa
from .simulation import Simulation  # noqa

try:
    __version__ = version("tlo")
except PackageNotFoundError:
    # package is not installed
    pass


DAYS_IN_YEAR = 365.25
DAYS_IN_MONTH = DAYS_IN_YEAR / 12
