import hashlib
import json
import logging as _logging
from typing import Union

import pandas as pd

from tlo.logging import encoding


def disable(level):
    _logging.disable(level)


def getLogger(name='tlo'):
    """Returns a TLO logger of the specified name"""
    if name not in _LOGGERS:
        _LOGGERS[name] = Logger(name)
    return _LOGGERS[name]


class _MockSim:
    # used as place holder for any logging that happens before simulation is setup!
    class MockDate:
        @staticmethod
        def isoformat():
            return "0000-00-00T00:00:00"
    date = MockDate()


class InconsistentLoggedColumnsError(Exception):
    """Error raised when structured log entry has different columns from header."""


class Logger:
    """A Logger for TLO log messages, with simplified usage. Outputs structured log messages in JSON
    format and is connected to the Simulation instance."""
    HASH_LEN = 10

    def __init__(self, name: str, level=_logging.NOTSET):

        assert name.startswith('tlo'), f'Only logging of tlo modules is allowed; name is {name}'

        # we build our logger on top of the standard python logging
        self._std_logger = _logging.getLogger(name=name)
        self._std_logger.setLevel(level)
        self.name = self._std_logger.name

        # don't propograte messages up from "tlo" to root logger
        if name == 'tlo':
            self._std_logger.propagate = False

        # the key of the structured logging calls for this logger
        self.keys = dict()
        
        # the columns for the structured logging calls for this logger
        self.columns = dict()

        # populated by init_logging(simulation) for the top-level "tlo" logger
        self.simulation = _MockSim()

        # a logger should only be using old-style or new-style logging, not a mixture
        self.logged_stdlib = False
        self.logged_structured = False

        # disable logging multirow dataframes until we're confident it's robust
        self._disable_dataframe_logging = True

    def __repr__(self):
        return f'<TLOmodel Logger `{self.name}` ({_logging.getLevelName(self.level)})>'

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

    def isEnabledFor(self, level):
        return self._std_logger.isEnabledFor(level)

    def reset_attributes(self):
        """Reset logger attributes to an unset state"""
        # clear all logger settings
        self.handlers.clear()
        self.keys.clear()
        self.columns.clear()
        self.simulation = _MockSim()
        # boolean attributes used for now, can be removed after transition to structured logging
        self.logged_stdlib = False
        self.logged_structured = False
        self.setLevel(INFO)

    def setLevel(self, level):
        self._std_logger.setLevel(level)

    def _get_data_as_dict(self, data):
        """Convert log data to a dictionary if it isn't already"""
        
        def sort_by_numeric_or_str_key(dict_: dict) -> dict:
            return dict(
                sorted(dict_.items(), key=lambda i: (isinstance(i[0], str), i[0]))
            )
        
        if isinstance(data, dict):
            return sort_by_numeric_or_str_key(data)
        if isinstance(data, pd.DataFrame):
            if len(data.index) == 1:
                return data.to_dict('records')[0]
            elif self._disable_dataframe_logging:
                raise ValueError("Logging multirow dataframes is disabled - if you need this feature let us know")
            else:
                return {'dataframe': sort_by_numeric_or_str_key(data.to_dict('index'))}
        if isinstance(data, (list, set, tuple, pd.Series)):
            if isinstance(data, set):
                data = sorted(data)
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
        
        def get_columns_from_data_dict(data):
            # using type().__name__ so both pandas and stdlib types can be used
            return {k: type(v).__name__ for k, v, in data.items()}

        if key not in self.keys:
            # new log key, so create header json row
            uuid = hashlib.md5(f"{self.name}+{key}".encode()).hexdigest()[:Logger.HASH_LEN]
            self.keys[key] = uuid
            columns = get_columns_from_data_dict(data)
            self.columns[key] = columns

            header = {
                "uuid": uuid,
                "type": "header",
                "module": self.name,
                "key": key,
                "level": _logging.getLevelName(level),
                "columns": columns,
                "description": description
            }
            header_json = json.dumps(header) + "\n"
        else:
            columns = get_columns_from_data_dict(data)
            if columns != self.columns[key]:
                header_columns = set(self.columns[key].items())
                logged_columns = set(columns.items())
                msg = (
                    f"Inconsistent columns in logged values for {self.name} logger "
                    f"with key {key} compared to header generated from initial log "
                    f"entry:\n"
                    f"  Columns in header not in logged values are\n"
                    f"  {dict(header_columns - logged_columns)}\n"
                    f"  Columns in logged values not in header are\n"
                    f"  {dict(logged_columns - header_columns)}"
                )
                raise InconsistentLoggedColumnsError(msg)

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
        return f'{level}|{self.name}|{msg}'

    def _check_logging_style(self, is_structured: bool):
        """Set booleans for logging type and throw exception if both types of logging haven't been used"""
        if is_structured:
            self.logged_structured = True
        else:
            self.logged_stdlib = True

        if self.logged_structured and self.logged_stdlib:
            raise ValueError(f"Both oldstyle and structured logging has been used for {self.name}, "
                             "please update all logging to use structured logging")

    def _check_and_filter(self, msg=None, *args, key=None, data=None, description=None, level, **kwargs):
        if self._std_logger.isEnabledFor(level):
            level_str = _logging.getLevelName(level)  # e.g. 'CRITICAL', 'INFO' etc.
            level_function = getattr(self._std_logger, level_str.lower())  # e.g. `critical` or `info` methods
            if key is None or data is None:
                raise ValueError("Structured logging requires `key` and `data` keyword arguments")
            self._check_logging_style(is_structured=True)
            level_function(self._get_json(level=level, key=key, data=data, description=description))

    def critical(self, msg=None, *args, key: str = None,
                 data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._check_and_filter(msg, *args, key=key, data=data, description=description, level=CRITICAL, **kwargs)

    def debug(self, msg=None, *args, key: str = None,
              data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._check_and_filter(msg, *args, key=key, data=data, description=description, level=DEBUG, **kwargs)

    def info(self, msg=None, *args, key: str = None,
             data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._check_and_filter(msg, *args, key=key, data=data, description=description, level=INFO, **kwargs)

    def warning(self, msg=None, *args, key: str = None,
                data: Union[dict, pd.DataFrame, list, set, tuple, str] = None, description=None, **kwargs):
        self._check_and_filter(msg, *args, key=key, data=data, description=description, level=WARNING, **kwargs)


CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING

_FORMATTER = _logging.Formatter('%(message)s')
_LOGGERS = {'tlo': Logger('tlo', WARNING)}
