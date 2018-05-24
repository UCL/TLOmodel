__version__ = '0.1.0'

# Import our key classes so they're available in the main namespace
from .core import (  # noqa
    Parameter,
    Property,
    Types,
)
from .population import Person, Population  # noqa
