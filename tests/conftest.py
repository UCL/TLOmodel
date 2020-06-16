"""Collection of shared fixtures"""
import pytest

from tlo.logging.core import _LOGGERS


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset all disease module loggers before each test functions/methods"""
    for logger_name, logger in _LOGGERS.items():
        if logger_name != 'tlo':
            logger.reset_attributes()
    yield
