"""Custom logging for tlo"""

import logging as _logging

# allow access to helper functions ---


def basicConfig(**kwargs):
    _logging.basicConfig(**kwargs)


def getLogger(name=None):
    if name:
        return Logger._python_logger.manager.getLogger(name)
    else:
        return _logging.root

# allow access to logging classes ---


class FileHandler(_logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super(FileHandler, self).__init__(filename, mode=mode, encoding=encoding, delay=delay)


class Formatter(_logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(Formatter, self).__init__(fmt=fmt, datefmt=datefmt, style=style)


class Logger:
    _python_logger = _logging.getLoggerClass()

    def setLevel(self, level):
        self._python_logger.setLevel()

    def critical(self, msg, *args, **kwargs):
        self._python_logger.critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._python_logger.debug(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self._python_logger.fatal(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._python_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._python_logger.warning(msg, *args, **kwargs)


class StreamHandler(_logging.StreamHandler):
    def __init__(self, stream=None):
        super(StreamHandler, self).__init__(stream=stream)

# allow access to logging levels ---


CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
