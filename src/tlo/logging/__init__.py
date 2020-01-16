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
        self.propagate = self._StdLogger.propagate
        self.level = self._StdLogger.level
        self.parent = self._StdLogger.parent
        self.disabled = self._StdLogger.disabled

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


class RootLogger(Logger):
    """
    A root logger is not that different to any other logger, except that
    it must have a logging level and there is only one instance of it in
    the hierarchy.
    """
    def __init__(self, level):
        """
        Initialize the logger with the name "root".
        """
        super(RootLogger, self).__init__("root", level)


class Manager(_logging.Manager):

    def getLogger(self, name):
        """
        Get a logger with the specified name (channel name), creating it
        if it doesn't yet exist. This name is a dot-separated hierarchical
        name, such as "a", "a.b", "a.b.c" or similar.

        If a PlaceHolder existed for the specified name [i.e. the logger
        didn't exist but a child of it did], replace it with the created
        logger and fix up the parent/child references which pointed to the
        placeholder to now point to the logger.
        """
        rv = None
        if not isinstance(name, str):
            raise TypeError('A logger name must be a string')
        _logging._acquireLock()
        try:
            if name in self.loggerDict:
                rv = self.loggerDict[name]
                if isinstance(rv, _logging.PlaceHolder):
                    ph = rv
                    rv = (self.loggerClass or _loggerClass)(name)
                    rv.manager = self
                    self.loggerDict[name] = rv
                    self._fixupChildren(ph, rv)
                    self._fixupParents(rv)
            else:
                rv = (self.loggerClass or _loggerClass)(name)
                rv.manager = self
                self.loggerDict[name] = rv
                self._fixupParents(rv)
        finally:
            _logging._releaseLock()
        return rv

    def setLoggerClass(self, klass):
        """
        Set the class to be used when instantiating a logger with this Manager.
        """
        if klass != Logger:
            if not issubclass(klass, Logger):
                raise TypeError("logger not derived from logging.Logger: "
                                + klass.__name__)
        self.loggerClass = klass

    def _fixupParents(self, alogger):
        """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
        name = alogger.name
        i = name.rfind(".")
        rv = None
        while (i > 0) and not rv:
            substr = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = _logging.PlaceHolder(alogger)
            else:
                obj = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    assert isinstance(obj, _logging.PlaceHolder)
                    obj.append(alogger)
            i = name.rfind(".", 0, i - 1)
        if not rv:
            rv = self.root
        alogger.parent = rv


# set up singleton objects ---
_loggerClass = Logger
root_logger = RootLogger(_logging.WARNING)
Logger.root = root_logger
Logger.manager = Manager(Logger.root)

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
