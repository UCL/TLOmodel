import json
import logging as _logging

from . import encoding


def disable(level):
    _logging.disable(level=level)


def getLogger(name='tlo'):
    """Returns a TLO logger of the specified name"""
    if name not in _LOGGERS.keys():
        _LOGGERS[name] = Logger(name)
    return _LOGGERS[name]


class FilterRecord:
    def __init__(self, name):
        self.name = name
        self.nlen = len(name)


class Logger:
    """
    TLO logging facade so that logging can be intercepted and customised
    """

    def __init__(self, name: str, level=_logging.NOTSET):
        assert name.startswith('tlo'), 'Only logging of tlo modules is allowed'
        self._std_logger = _logging.getLogger(name=name)
        self._std_logger.setLevel(level)
        if name == 'tlo':
            self._std_logger.propagate = False
        self.name = self._std_logger.name
        self.keys = set()
        # populated by init_logging(simulation)
        self.simulation = None

    def __repr__(self):
        return f'<tlo Logger containing {self._std_logger}>'

    @property
    def handlers(self):
        return self._std_logger.handlers

    @handlers.setter
    def handlers(self, handlers):
        self._std_logger.handlers.clear()
        for handler in handlers:
            self._std_logger.handlers.append(handler)

    @property
    def filters(self):
        return self._std_logger.filters

    @filters.setter
    def filters(self, filters):
        self._std_logger.filters.clear()
        for filter in filters:
            self._std_logger.filters.append(filter)

    def addHandler(self, hdlr):
        self._std_logger.addHandler(hdlr=hdlr)

    def setLevel(self, level):
        self._std_logger.setLevel(level)

    def _msg(self, level, key, data: dict = None, description=None):
        # TODO: will we always want a dict?
        tlo_logger = getLogger('tlo')
        record = FilterRecord(self.name)
        header = {}
        if key not in self.keys:
            self.keys.add(key)
            # create header json
            header = {"level": level,
                      "module": self.name,
                      "key": key,
                      "columns": {key: value.dtype.name for key, value in data.items()},
                      "description": description}

        # create row data json
        row = {"module": self.name, "key": key,
               "date": tlo_logger.simulation.date.isoformat(),
               "values": list(data.values())}

        # for each handler, write json data if allowed
        for handler in tlo_logger.handlers:
            if handler.filter(record) and self._std_logger.level >= eval(level):
                if header:
                    json.dump(header, handler.stream)
                    handler.stream.write(handler.terminator)
                json.dump(row, handler.stream, cls=encoding.PandasEncoder)
                handler.stream.write(handler.terminator)

    def critical(self, msg=None, *args, key=None, data: dict = None, description=None, **kwargs):
        if msg:
            self._std_logger.critical(msg, *args, **kwargs)
        elif key and data:
            self._msg(level="CRITICAL", key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised")

    def debug(self, msg=None, *args, key=None, data: dict = None, description=None, **kwargs):
        if msg:
            self._std_logger.debug(msg, *args, **kwargs)
        elif key and data:
            self._msg(level="DEBUG", key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised")

    def info(self, msg=None, *args, key=None, data: dict = None, description=None, **kwargs):
        if msg:
            self._std_logger.info(msg, *args, **kwargs)
        elif key and data:
            self._msg(level="INFO", key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised")

    def warning(self, msg=None, *args, key=None, data: dict = None, description=None, **kwargs):
        if msg:
            self._std_logger.warning(msg, *args, **kwargs)
        elif key and data:
            self._msg(level="WARNING", key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised")

    def removeFilter(self, fltr):
        self._std_logger.removeFilter(fltr)

    def removeHandler(self, hdlr):
        self._std_logger.removeHandler(hdlr)


CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING

_FORMATTER = _logging.Formatter('%(levelname)s|%(name)s|%(message)s')
_LOGGERS = {'tlo': Logger('tlo', WARNING)}
