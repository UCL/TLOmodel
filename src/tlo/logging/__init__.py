import logging as _logging

# logging functions ---


def basicConfig(**kwargs):
    _logging.basicConfig(**kwargs)


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


# logging classes ---


class FileHandler(_logging.FileHandler):
    pass


class Formatter(_logging.Formatter):
    pass


class StreamHandler(_logging.StreamHandler):
    pass


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


# setup default logger

_loggers = {'tlo': Logger('tlo', _logging.WARNING)}

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
