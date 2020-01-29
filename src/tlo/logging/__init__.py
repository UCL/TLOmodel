import logging as _logging
import sys
from pathlib import Path
from typing import Dict, Iterable

# stdlib logging functions ---

def disable(level):
    _logging.disable(level=level)


def getLogger(name=None):
    if name:
        assert name.startswith("tlo"), "Only logging of tlo modules is allowed"
        # Singleton loggers
        if name not in _loggers.keys():
            _loggers[name] = Logger(name)
        return _loggers[name]
    else:
        return _loggers['tlo']


# stdlib logging classes ---

class Logger:
    """
    TLO logging facade so that logging can be intercepted and customised
    """

    def __init__(self, name: str, level=_logging.NOTSET):
        if name == 'tlo':
            self._std_logger = _logging.getLogger()
        else:
            self._std_logger = _logging.getLogger(name=name)
        self.name = name
        self.handlers = self._std_logger.handlers
        self.filters = self._std_logger.filters

    def __repr__(self):
        return f'<tlo Logger containing {self._std_logger}>'

    def addHandler(self, hdlr):
        self._std_logger.addHandler(hdlr=hdlr)

    def setLevel(self, level):
        self._std_logger.setLevel(level)

    def critical(self, msg, *args, **kwargs):
        self._std_logger.critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._std_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._std_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._std_logger.warning(msg, *args, **kwargs)

    def removeFilter(self, filter):
        self._std_logger.removeFilter(filter)

    def removeHandler(self, hdlr):
        self._std_logger.removeHandler(hdlr)


_loggers = {'tlo': Logger('tlo', _logging.WARNING)}
_FORMATTER = _logging.Formatter('%(levelname)s|%(name)s|%(message)s')

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING


# custom logging ---


def add_filehandler(log_path: Path) -> _logging.FileHandler:
    """Add filehandler to logger

    :param log_path: path for file
    :return: filehandler object
    """
    fh = _logging.FileHandler(log_path)
    fh.setFormatter(_FORMATTER)
    getLogger().addHandler(fh)
    return fh


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


def init_logging():
    """Initialise default logging with stdout stream"""
    handler = _logging.StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    handler.setFormatter(_FORMATTER)
    logger = getLogger()
    logger.handlers.clear()
    logger.filters.clear()
    logger.addHandler(handler)

    _logging.basicConfig(level=DEBUG)


init_logging()
