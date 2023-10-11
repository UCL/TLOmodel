"""
The top-level TLO framework module.

We import our key classes so they're available in the main namespace.

Pandas' Timestamp is used as our main date type (even though it has nanosecond resolution...)
"""
from pandas import Timestamp as Date  # noqa
from pandas.tseries.offsets import DateOffset  # noqa

from .core import Module, Parameter, Property, Types  # noqa
from .population import Population  # noqa
from .simulation import Simulation  # noqa

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


DAYS_IN_YEAR = 365.25
DAYS_IN_MONTH = DAYS_IN_YEAR / 12
