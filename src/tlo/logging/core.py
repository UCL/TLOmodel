import json
import logging as _logging
from typing import Union

import pandas as pd

from tlo import util
from tlo.logging import encoding


def disable(level):
    _logging.disable(level=level)


def getLogger(name='tlo'):
    """Returns a TLO logger of the specified name"""
    if name not in _LOGGERS.keys():
        _LOGGERS[name] = Logger(name)
    return _LOGGERS[name]


class FilterRecord:
    """Class to allow stdlib handler filtering"""
    def __init__(self, name):
        self.name = name
        self.nlen = len(name)


class Logger:
    """
    TLO logging facade so that logging can be intercepted and customised
    """

    def __init__(self, name: str, level=_logging.NOTSET):
        assert name.startswith('tlo'), 'Only logging of tlo modules is allowed'
        # std library logger for oldstyle logging and tracking
        self._std_logger = _logging.getLogger(name=name)
        self._std_logger.setLevel(level)
        if name == 'tlo':
            self._std_logger.propagate = False
        self.name = self._std_logger.name
        # track keys given during structured logging
        self.keys = set()
        # populated by init_logging(simulation)
        self.simulation = None
        # to ensure only structured or oldstyle logging used for a single module
        # can be removed after transition
        self.logged_stdlib = False
        self.logged_structured = False

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

    def reset_attributes(self):
        """Reset logger attributes to an unset state"""
        # clear all logger settings
        self.handlers.clear()
        self.filters.clear()
        self.keys.clear()
        self.simulation = None
        # boolean attributes used for now, can be removed after transition to structured logging
        self.logged_stdlib = False
        self.logged_structured = False

    def setLevel(self, level):
        self._std_logger.setLevel(level)

    def _convert_log_data(self, data):
        """Convert log data to a dictionary if it isn't already"""
        if isinstance(data, dict):
            return data
        if isinstance(data, pd.DataFrame):
            if len(data.index) == 1:
                converted_data = data.to_dict('records')[0]
            else:
                converted_data = util.nested_to_record(data)
            return converted_data
        if isinstance(data, (list, set, tuple)):
            return {f'item_{index + 1}': value for index, value in enumerate(data)}
        if isinstance(data, str):
            return {'message': data}

        raise ValueError(f'Unexpected type given as data:\n{data}')

    def _log_message(self, level, key, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None):
        """Writes structured log message if handler allows this and logging level is allowed

        Will write a header line the first time a new logging key is encountered
        Then will only write data rows in later rows for this logging key

        :param level: Level the message is being logged as
        :param key: logging key
        :param data: data to be logged
        :param description: description of this log type
        """
        # message level less than than the logger level, early exit
        if eval(level) < self._std_logger.level:
            return

        data = self._convert_log_data(data)
        tlo_logger = getLogger('tlo')
        # create record to allow for handler filtering
        record = FilterRecord(self.name)

        header = {}
        if key not in self.keys:
            # new log key, so create header json row
            self.keys.add(key)
            header = {"type": "header",
                      "level": level,
                      "module": self.name,
                      "key": key,
                      # using type().__name__ so both pandas and stdlib types can be used
                      "columns": {key: type(value).__name__ for key, value in data.items()},
                      "description": description}

        # create data json row
        row = {"type": "data",
               "module": self.name, "key": key,
               "date": tlo_logger.simulation.date.isoformat(),
               "values": list(data.values())}

        # for each handler, write json data if allowed
        for handler in tlo_logger.handlers:
            if handler.filter(record):
                if header:
                    json.dump(header, handler.stream)
                    handler.stream.write(handler.terminator)
                json.dump(row, handler.stream, cls=encoding.PandasEncoder)
                handler.stream.write(handler.terminator)

    def _mixed_logging_check(self, is_structured: bool):
        """Set booleans for logging type and throw exception if both types of logging haven't been used"""
        if is_structured:
            self.logged_structured = True
        else:
            self.logged_stdlib = True

        if self.logged_structured and self.logged_stdlib:
            raise ValueError(f"Both oldstyle and structured logging has been used for {self.name}, "
                             "please update all logging to use structured logging")

    def _try_log_message(self, level, key, data, description):
        """Log strucured message, if key or data are None, then throw exception"""
        if key and data:
            self._mixed_logging_check(is_structured=True)
            self._log_message(level=level, key=key, data=data, description=description)
        else:
            raise ValueError("Logging information was not recognised. Structured logging requires both key and data")

    def critical(self, msg=None, *args, key: str = None, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None,
                 description=None, **kwargs):
        # std logger branch can be removed once transition is completed
        if msg or msg == []:
            self._mixed_logging_check(is_structured=False)
            self._std_logger.critical(msg, *args, **kwargs)
        else:
            self._try_log_message(level="CRITICAL", key=key, data=data, description=description)

    def debug(self, msg=None, *args, key: str = None, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None,
              description=None, **kwargs):
        # std logger branch can be removed once transition is completed
        if msg or msg == []:
            self._mixed_logging_check(is_structured=False)
            self._std_logger.debug(msg, *args, **kwargs)
        else:
            self._try_log_message(level="DEBUG", key=key, data=data, description=description)

    def info(self, msg=None, *args, key: str = None, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None,
             description=None, **kwargs):
        # std logger branch can be removed once transition is completed
        if msg or msg == []:
            self._mixed_logging_check(is_structured=False)
            self._std_logger.info(msg, *args, **kwargs)
        else:
            self._try_log_message(level="INFO", key=key, data=data, description=description)

    def warning(self, msg=None, *args, key: str = None, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None,
                description=None, **kwargs):
        # std logger branch can be removed once transition is completed
        if msg or msg == []:
            self._mixed_logging_check(is_structured=False)
            self._std_logger.warning(msg, *args, **kwargs)
        else:
            self._try_log_message(level="WARNING", key=key, data=data, description=description)

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
