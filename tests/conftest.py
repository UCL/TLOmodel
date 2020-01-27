"""Collection of shared fixtures"""

import pytest

from tlo import logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Remove all root handlers and filters during teardown of test"""
    yield
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.filters.clear()
    logger._std_logger.manager.disable = 0
