__version__ = '0.1.0'

# We import our key classes so they're available in the main namespace.
# Pandas' Timestamp is used as our main date type (even though it has nanosecond resolution...)

from pandas import Timestamp as Date  # noqa
from pandas.tseries.offsets import DateOffset  # noqa

from .core import Module, Parameter, Property, Types  # noqa
from .population import Person, Population  # noqa
from .simulation import Simulation  # noqa
