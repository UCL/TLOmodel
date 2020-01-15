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
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super(FileHandler, self).__init__(filename, mode=mode, encoding=encoding, delay=delay)


class Formatter(_logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(Formatter, self).__init__(fmt=fmt, datefmt=datefmt, style=style)



class StreamHandler(_logging.StreamHandler):
    def __init__(self, stream=None):
        super(StreamHandler, self).__init__(stream=stream)


class Logger:
    _PythonLogger = _logging.getLoggerClass()

    def __init__(self, name, level=_logging.NOTSET):
        self._PythonLogger.__init__(self._PythonLogger, name, level=level)
        self.handlers = self._PythonLogger.handlers
        self.propagate = self._PythonLogger.propagate
        self.level = self._PythonLogger.level
        self.parent = self._PythonLogger.parent
        self.disabled = self._PythonLogger.disabled


    def addHandler(self, hdlr):
        self._PythonLogger.addHandler(self._PythonLogger, hdlr=hdlr)

    def setLevel(self, level):
        print(f"you have set the level {level}")
        self._PythonLogger.setLevel(level)

    def critical(self, msg, *args, **kwargs):
        self._PythonLogger.critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._PythonLogger.debug(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self._PythonLogger.fatal(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        print("----")
        print("I am an info")
        print("----")
        self._PythonLogger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._PythonLogger.warning(msg, *args, **kwargs)


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
        Logger.__init__(self, "root", level)

# set up single instance of root logger and logger

root_logger = RootLogger(_logging.WARNING)
Logger.root = root_logger
Logger.manager = _logging.Manager(Logger.root)

# allow access to logging levels ---

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING
