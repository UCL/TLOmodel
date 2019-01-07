"""
The top-level TLO framework module.

We import our key classes so they're available in the main namespace.

Pandas' Timestamp is used as our main date type (even though it has nanosecond resolution...)
"""
import logging
import sys

from pandas import Timestamp as Date  # noqa
from pandas.tseries.offsets import DateOffset  # noqa

from .core import Module, Parameter, Property, Types  # noqa
from .population import Population  # noqa
from .simulation import Simulation  # noqa

__version__ = '0.1.0'

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s|%(name)s|%(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

logging.basicConfig(level=logging.DEBUG)
