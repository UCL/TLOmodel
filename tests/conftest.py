"""Collection of shared fixtures"""
import pytest

from tlo import logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Remove all tlo handlers and filters during teardown of test"""
    yield
    logger = logging.getLogger("tlo")
    logger.handlers.clear()
    logger.filters.clear()
