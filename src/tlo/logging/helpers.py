import logging as _logging
import sys
from pathlib import Path
from typing import Dict, Iterable

from .core import _FORMATTER, DEBUG, getLogger


def set_output_file(log_path: Path) -> _logging.FileHandler:
    """Add filehandler to logger

    :param log_path: path for file
    :return: filehandler object
    """
    file_handler = _logging.FileHandler(log_path)
    file_handler.setFormatter(_FORMATTER)
    getLogger('tlo').handlers = [h for h in getLogger('tlo').handlers
                                 if not isinstance(h, _logging.FileHandler)]
    getLogger('tlo').addHandler(file_handler)
    return file_handler


def set_logging_levels(custom_levels: Dict[str, int], modules: Iterable[str]):
    """Set custom logging levels for disease modules

    :param custom_levels: Dictionary of modules and their level, '*' can be used
    as a key for all modules
    :param modules: string values of all registered modules
    """
    for key, value in custom_levels.items():
        if key == '*':
            for module in modules:
                getLogger(module).setLevel(value)
        else:
            getLogger(key).setLevel(value)


def init_logging(simulation=None):
    """Initialise default logging with stdout stream"""
    handler = _logging.StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    handler.setFormatter(_FORMATTER)
    logger = getLogger('tlo')
    logger.handlers.clear()
    logger.filters.clear()
    logger.addHandler(handler)
    _logging.basicConfig(level=_logging.WARNING)
    if simulation:
        logger.simulation = simulation
