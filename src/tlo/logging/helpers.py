import gzip
import logging as _logging
import sys
from pathlib import Path
from typing import Dict, Iterable

from .core import _FORMATTER, _LOGGERS, DEBUG, getLogger


def set_output_file(log_path: Path) -> _logging.StreamHandler:
    """Add filehandler to logger

    :param log_path: path for file
    :return: filehandler object
    """
    # if we haven't been given a gzip file, make it so
    if not log_path.name.endswith('.gz'):
        log_path = log_path.parent / (log_path.name + '.gz')

    # log directly to this compressed file
    gzip_file = gzip.open(log_path, mode='wt', encoding='utf-8')

    stream_handler = _logging.StreamHandler(gzip_file)
    stream_handler.setFormatter(_FORMATTER)

    # should be the only stream handler for all tlo logging
    getLogger('tlo').handlers = [h for h in getLogger('tlo').handlers
                                 if not isinstance(h, (_logging.FileHandler, _logging.StreamHandler))]
    getLogger('tlo').addHandler(stream_handler)
    return stream_handler


def set_logging_levels(custom_levels: Dict[str, int], modules: Iterable[str]):
    """Set custom logging levels for disease modules

    :param custom_levels: Dictionary of modules and their level, '*' can be used as a key for all modules
    :param modules: string values of all registered modules
    """
    for key, value in custom_levels.items():
        if key == '*':
            for module in modules:
                getLogger(module).setLevel(value)
        else:
            getLogger(key).setLevel(value)


def init_logging(add_stdout_handler=True):
    """Initialise default logging with stdout stream"""
    for logger_name, logger in _LOGGERS.items():
        logger.reset_attributes()
    if add_stdout_handler:
        handler = _logging.StreamHandler(sys.stdout)
        handler.setLevel(DEBUG)
        handler.setFormatter(_FORMATTER)
        getLogger('tlo').addHandler(handler)
    _logging.basicConfig(level=_logging.WARNING)


def set_simulation(simulation):
    """
    Inject simulation into logger for structured logging, called by the simulation
    :param simulation:
    :return:
    """
    logger = getLogger('tlo')
    logger.simulation = simulation
