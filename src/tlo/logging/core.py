import hashlib
import json
import logging as _logging
from typing import Union

import pandas as pd

from tlo.logging import encoding


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
    HASH_LEN = 10

    def __init__(self, name: str, level=_logging.NOTSET):
        assert name.startswith('tlo'), 'Only logging of tlo modules is allowed'
        # std library logger for oldstyle logging and tracking
        self._std_logger = _logging.getLogger(name=name)
        self._std_logger.setLevel(level)
        if name == 'tlo':
            self._std_logger.propagate = False
        self.name = self._std_logger.name
        # track keys given during structured logging
        self.keys = dict()
        # populated by init_logging(simulation)
        self.simulation = None
        # to ensure only structured or oldstyle logging used for a single module
        # can be removed after transition
        self.logged_stdlib = False
        self.logged_structured = False
        self._disable_dataframe_logging = True

    def __repr__(self):
        return f'<TLO Logger containing {self._std_logger}>'

    @property
    def handlers(self):
        return self._std_logger.handlers

    @property
    def level(self):
        return self._std_logger.level

    @handlers.setter
    def handlers(self, handlers):
        self._std_logger.handlers.clear()
        for handler in handlers:
            self._std_logger.handlers.append(handler)

    def addHandler(self, hdlr):
        self._std_logger.addHandler(hdlr=hdlr)

    def reset_attributes(self):
        """Reset logger attributes to an unset state"""
        # clear all logger settings
        self.handlers.clear()
        self.keys.clear()
        self.simulation = None
        # boolean attributes used for now, can be removed after transition to structured logging
        self.logged_stdlib = False
        self.logged_structured = False

    def setLevel(self, level):
        self._std_logger.setLevel(level)

    def _get_data_as_dict(self, data):
        """Convert log data to a dictionary if it isn't already"""
        if isinstance(data, dict):
            return data
        if isinstance(data, pd.DataFrame):
            if len(data.index) == 1:
                converted_data = data.to_dict('records')[0]
            elif self._disable_dataframe_logging:
                raise ValueError("Logging an entire dataframe is disabled, if you need this feature let us know")
            else:
                converted_data = {'dataframe': data.to_dict('index')}
            return converted_data
        if isinstance(data, (list, set, tuple, pd.Series)):
            return {f'item_{index + 1}': value for index, value in enumerate(data)}
        if isinstance(data, str):
            return {'message': data}

        raise ValueError(f'Unexpected type given as data:\n{data}')

    def _get_json(self, level, key, data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None):
        """Writes structured log message if handler allows this and logging level is allowed

        Will write a header line the first time a new logging key is encountered
        Then will only write data rows in later rows for this logging key

        :param level: Level the message is being logged as
        :param key: logging key
        :param data: data to be logged
        :param description: description of this log type
        """
        # message level less than than the logger level, early exit
        if level < self._std_logger.level:
            return

        data = self._get_data_as_dict(data)
        header_json = ""

        if key not in self.keys:
            # new log key, so create header json row
            uuid = hashlib.md5(f"{self.name}+{key}".encode()).hexdigest()[:Logger.HASH_LEN]
            self.keys[key] = uuid

            header = {
                "uuid": uuid,
                "type": "header",
                "module": self.name,
                "key": key,
                "level": _logging.getLevelName(level),
                # using type().__name__ so both pandas and stdlib types can be used
                "columns": {key: type(value).__name__ for key, value in data.items()},
                "description": description
            }
            header_json = json.dumps(header) + "\n"

        uuid = self.keys[key]

        # create data json row; in DEBUG mode we echo the module and key for easier eyeballing
        if self._std_logger.level == DEBUG:
            row = {"date": getLogger('tlo').simulation.date.isoformat(),
                   "module": self.name,
                   "key": key,
                   "uuid": uuid,
                   "values": list(data.values())}
        else:
            row = {"uuid": uuid,
                   "date": getLogger('tlo').simulation.date.isoformat(),
                   "values": list(data.values())}

        row_json = json.dumps(row, cls=encoding.PandasEncoder)

        return f"{header_json}{row_json}"

    def _make_old_style_msg(self, level, msg):
        return '%s|%s|%s' % (level, self.name, msg)

    def _check_logging_style(self, is_structured: bool):
        """Set booleans for logging type and throw exception if both types of logging haven't been used"""
        if is_structured:
            self.logged_structured = True
        else:
            self.logged_stdlib = True

        if self.logged_structured and self.logged_stdlib:
            raise ValueError(f"Both oldstyle and structured logging has been used for {self.name}, "
                             "please update all logging to use structured logging")

    def _handle_old_or_new(self, msg=None, *args, key=None, data=None, description=None, level, **kwargs):
        if self._std_logger.isEnabledFor(level):
            level_str = _logging.getLevelName(level)  # e.g. 'CRITICAL', 'INFO' etc.
            level_function = getattr(self._std_logger, level_str.lower())  # e.g. `critical` or `info` methods
            # if this is an old-style logging call
            if msg or msg == []:
                # NOTE: std logger branch can be removed once transition is completed
                self._check_logging_style(is_structured=False)
                level_function(self._make_old_style_msg(level_str, msg), *args, **kwargs)
            # otherwise, this is a new-style structure logging call
            else:
                if key is None or data is None:
                    raise ValueError("Structured logging requires `key` and `data` keyword arguments")
                self._check_logging_style(is_structured=True)
                level_function(self._get_json(level=level, key=key, data=data, description=description))

    def critical(self, msg=None, *args, key: str = None,
                 data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._handle_old_or_new(msg, *args, key=key, data=data, description=description, level=CRITICAL, **kwargs)

    def debug(self, msg=None, *args, key: str = None,
              data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._handle_old_or_new(msg, *args, key=key, data=data, description=description, level=DEBUG, **kwargs)

    def info(self, msg=None, *args, key: str = None,
             data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._handle_old_or_new(msg, *args, key=key, data=data, description=description, level=INFO, **kwargs)

    def warning(self, msg=None, *args, key: str = None,
                data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._handle_old_or_new(msg, *args, key=key, data=data, description=description, level=WARNING, **kwargs)


CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING

_FORMATTER = _logging.Formatter('%(message)s')
_LOGGERS = {'tlo': Logger('tlo', WARNING)}
