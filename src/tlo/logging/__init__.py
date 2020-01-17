import logging as _logging


# logging functions ---


def basicConfig(**kwargs):
    _logging.basicConfig(**kwargs)


def getLogger(name=None):
    if name:
        # Singleton loggers
        if name not in _loggers.keys():
            _loggers[name] = Logger(name)
        return _loggers[name]
    else:
        return _loggers['root']


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

    def __init__(self, name, level=_logging.NOTSET):
        if name == 'root':
            self._StdLogger = _logging.getLogger()
        else:
            self._StdLogger = _logging.getLogger(name=name)
        self.name = name
        self.handlers = self._StdLogger.handlers

    def __repr__(self):
        return f'<tlo Logger containing {self._StdLogger}>'

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


# setup default logger

_loggers = {'root': Logger('root', _logging.WARNING)}

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
