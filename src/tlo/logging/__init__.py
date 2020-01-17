import logging as _logging


def basicConfig(**kwargs):
    _logging.basicConfig(**kwargs)


def getLogger(name=None):
    if name:
        return Logger.manager.getLogger(name)
    else:
        return root_logger


# allow access to logging classes ---


class FileHandler(_logging.FileHandler):
    pass


class Formatter(_logging.Formatter):
    pass


class StreamHandler(_logging.StreamHandler):
    pass


class Logger:
    def __init__(self, name, level=_logging.NOTSET):
        self._StdLogger = _logging.getLogger(name=name)
        self.name = name
        self.handlers = self._StdLogger.handlers

    def addHandler(self, hdlr):
        self._StdLogger.addHandler(hdlr=hdlr)

    def setLevel(self, level):
        self._StdLogger.setLevel(level)

    def critical(self, msg, *args, **kwargs):
        self._StdLogger.critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._StdLogger.debug(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self._StdLogger.fatal(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._StdLogger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._StdLogger.warning(msg, *args, **kwargs)


class Manager:

    def __init__(self):
        self.loggers = {"root": root_logger}

    def getLogger(self, name):
        """
        Ge
        """
        rv = None
        if not isinstance(name, str):
            raise TypeError('A logger name must be a string')
        _logging._acquireLock()
        try:
            if name in self.loggers:
                rv = self.loggers[name]
            else:
                rv = _loggerClass(name)
                rv.manager = self
                self.loggers[name] = rv
        finally:
            _logging._releaseLock()
        return rv


# set up singleton objects ---
_loggerClass = Logger
root_logger = Logger("root", _logging.WARNING)
Logger.manager = Manager()

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
