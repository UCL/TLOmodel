__version__ = '0.1.0'

# We use this as our main date type (even though it has nanosecond resolution...)
from pandas import Timestamp as Date  # noqa
from pandas.tseries.offsets import DateOffset  # noqa

# Import our key classes so they're available in the main namespace
from .core import (  # noqa
    Parameter,
    Property,
    Types,
)
from .population import Person, Population  # noqa
from .simulation import Simulation  # noqa
