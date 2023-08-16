import logging as _logging
import sys
from pathlib import Path
from typing import Dict

from .core import _FORMATTER, _LOGGERS, DEBUG, getLogger


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


def set_logging_levels(custom_levels: Dict[str, int]):
    """Set custom logging levels for disease modules

    :param custom_levels: Dictionary of modules and their level, '*' can be used as a key for all modules
    """
    # get list of `tlo.` loggers to process (this assumes logger have been setup on module import)
    loggers = {_logging.getLogger(name) for name in _logging.root.manager.loggerDict if name.startswith('tlo.methods')}

    # set the baseline logging level from methods, if it's been set
    if '*' in custom_levels:
        getLogger('tlo.methods').setLevel(custom_levels['*'])

    # loop over each of the tlo loggers
    for logger in loggers:
        # get the full name
        logger_name = logger.name
        matched = False
        # look for name, or any parent name, in the custom levels
        while len(logger_name):
            if logger_name in custom_levels:
                getLogger(logger_name).setLevel(custom_levels[logger_name])
                matched = True
                break
            elif logger_name == 'tlo.methods':
                # we've reached the top-level of the `tlo.methods` logger
                break
            else:
                # get the parent logger name
                logger_name = '.'.join(logger_name.split(".")[:-1])
        # if we exited without finding a matching logger in custom levels
        if not matched:
            if '*' in custom_levels:
                getLogger(logger.name).setLevel(custom_levels['*'])

    # loggers named in custom_level but, for some reason, haven't been getLogger-ed yet
    loggers = {logger.name for logger in loggers}
    for logger_name, logger_level in custom_levels.items():
        if logger_name != "*" and logger_name not in loggers:
            getLogger(logger_name).setLevel(logger_level)


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
