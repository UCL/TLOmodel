"""Collection of shared fixtures"""
import pytest

from tlo import logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset all logging in test setup"""
    logger = logging.getLogger('tlo.test.logger')
    logger.reset_attributes()
    yield
