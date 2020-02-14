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
        tlo_logger = getLogger('tlo')
        # TODO: filter messages
        if key not in self.keys:
            self.keys.add(key)
            # write header json
            columns = {"date": "pd.Timestamp"}
            columns.update({key: value.dtype.name for key, value in data.items()})
            header = {"level": level,
                      "module": self.name,
                      "key": key,
                      "columns": columns,
                      "description": description}
            for handler in tlo_logger.handlers:
                json.dump(header, handler.stream)
                handler.stream.write(handler.terminator)

        # write data json
        values = [tlo_logger.simulation.date.isoformat()]
        values.extend(data.values())
        row = {"module": self.name, "key": key,
               "values": values}
        for handler in tlo_logger.handlers:
            json.dump(row, handler.stream, cls=encoding.PandasEncoder)
            handler.stream.write(handler.terminator)

    def critical(self, msg, *args, **kwargs):
        self._std_logger.critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._std_logger.debug(msg, *args, **kwargs)

    def info(self, msg=None, *args, key=None, data: dict = None, description=None, **kwargs):
        if msg:
            self._std_logger.info(msg, *args, **kwargs)
        elif key and data:
            self._msg(level="INFO", key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised")

    def warning(self, msg, *args, **kwargs):
        self._std_logger.warning(msg, *args, **kwargs)

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

