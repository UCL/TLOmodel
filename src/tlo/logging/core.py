from __future__ import annotations

import hashlib
import json
import logging as _logging
import sys
from functools import partialmethod
from pathlib import Path
from typing import Callable, List, Optional, TypeAlias, Union

import numpy as np
import pandas as pd

from tlo.logging import encoding


LogLevel: TypeAlias = int
LogData: TypeAlias = Union[str, dict, list, set, tuple, pd.DataFrame, pd.Series]
SimulationDateGetter: TypeAlias = Callable[[], str]

CRITICAL = _logging.CRITICAL
DEBUG = _logging.DEBUG
FATAL = _logging.FATAL
INFO = _logging.INFO
WARNING = _logging.WARNING

_DEFAULT_LEVEL = INFO

_DEFAULT_FORMATTER = _logging.Formatter('%(message)s')

class InconsistentLoggedColumnsError(Exception):
    """Error raised when structured log entry has different columns from header."""


def _mock_simulation_date_getter() -> str:
    return "0000-00-00T00:00:00"


_get_simulation_date: SimulationDateGetter = _mock_simulation_date_getter
_loggers: dict[str, Logger] = {}


def initialise(
    add_stdout_handler: bool = True,
    simulation_date_getter: SimulationDateGetter = _mock_simulation_date_getter,
    root_level: LogLevel = WARNING,
    stdout_handler_level: LogLevel = DEBUG,
    formatter: _logging.Formatter = _DEFAULT_FORMATTER,
) -> None:
    """Initialise logging system and set up root `tlo` logger.
    
    :param add_stdout_handler: Whether to add a handler to output log entries to stdout.
    :param simulation_date_getter: Zero-argument function returning simulation date as
        string in ISO format to use in log entries. Defaults to function returning a
        a fixed dummy date for use before a simulation has been initialised.
    :param root_level: Logging level for root `tlo` logger.
    :param formatter: Formatter to use for logging to stdout.
    """
    global _get_simulation_date, _loggers
    _get_simulation_date = simulation_date_getter
    for logger in _loggers.values():
        logger.reset_attributes()
    root_logger = Logger("tlo", root_level)
    _loggers["tlo"] = root_logger
    if add_stdout_handler:
        handler = _logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout_handler_level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def set_output_file(
    log_path: Path,
    formatter: _logging.Formatter = _DEFAULT_FORMATTER,
) -> _logging.FileHandler:
    """Add file handler to logger.

    :param log_path: Path for file.
    :return: File handler object.
    """
    file_handler = _logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger = getLogger('tlo')
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, _logging.FileHandler)
    ]
    logger.addHandler(file_handler)
    return file_handler


def disable(level: LogLevel) -> None:
    """Disable all logging calls of specified level and below."""
    _logging.disable(level)


def getLogger(name: str = "tlo") -> Logger:
    """Returns a TLO logger of the specified name"""
    if name not in _loggers:
        _loggers[name] = Logger(name)
    return _loggers[name]


def _sort_by_numeric_or_str_key(data: dict) -> dict:
    """Sort a data dictionary with keys that may be either strings or numeric types."""
    return dict(sorted(data.items(), key=lambda i: (isinstance(i[0], str), i[0])))


def _get_log_data_as_dict(data: LogData) -> dict:
    """Convert log data to a dictionary if it isn't already"""
    if isinstance(data, dict):
        return _sort_by_numeric_or_str_key(data)
    if isinstance(data, pd.DataFrame):
        if len(data) == 1:
            data_dict = data.iloc[0].to_dict()
            return _sort_by_numeric_or_str_key(data_dict)
        else:
            raise ValueError(
                "Logging multirow dataframes is not currently supported - "
                "if you need this feature let us know"
            )
    if isinstance(data, (list, set, tuple, pd.Series)):
        if isinstance(data, set):
            data = sorted(data)
        return {f"item_{index + 1}": value for index, value in enumerate(data)}
    if isinstance(data, str):
        return {"message": data}
    raise ValueError(f"Unexpected type given as data:\n{data}")


def _convert_numpy_scalars_to_python_types(data: dict) -> dict:
    """Convert NumPy scalar types to suitable standard Python types."""
    return {
        key: value.item() if isinstance(value, (np.number, np.bool_)) else value
        for key, value in data.items()
    }


def _get_columns_from_data_dict(data: dict) -> dict:
    """Get columns dictionary specifying types of data dictionary values."""
    # using type().__name__ so both pandas and stdlib types can be used
    return {k: type(v).__name__ for k, v, in data.items()}


class Logger:
    """Logger for structured log messages output by simulation.

    Outputs structured log messages in JSON format along with simulation date log entry
    was generated at. Log messages are associated with a string key and for each key
    the log message data is expected to have a fixed structure:
    
    - Collection like data (tuples, lists, sets) should be of fixed length.
    - Mapping like data (dictionaries, pandas series and dataframes) should have a fixed
      set of keys and the values should be of fixed data types.
    
    The first log message for a given key will generate a 'header' log entry which
    records the structure of the message with subsequent log messages only logging the
    values for efficiency, hence the requirement for the structure to remain fixed.
    """

    HASH_LEN = 10

    def __init__(self, name: str, level: LogLevel = _DEFAULT_LEVEL) -> None:
        assert name.startswith(
            "tlo"
        ), f"Only logging of tlo modules is allowed; name is {name}"
        # we build our logger on top of the standard python logging
        self._std_logger = _logging.getLogger(name=name)
        self._std_logger.setLevel(level)
        # don't propograte messages up from "tlo" to root logger
        if name == "tlo":
            self._std_logger.propagate = False
        # the unique identifiers of the structured logging calls for this logger
        self._uuids = dict()
        # the columns for the structured logging calls for this logger
        self._columns = dict()

    def __repr__(self) -> str:
        return f"<TLOmodel Logger `{self.name}` ({_logging.getLevelName(self.level)})>"

    @property
    def name(self) -> str:
        return self._std_logger.name

    @property
    def handlers(self) -> List[_logging.Handler]:
        return self._std_logger.handlers

    @property
    def level(self) -> LogLevel:
        return self._std_logger.level

    @handlers.setter
    def handlers(self, handlers: List[_logging.Handler]):
        self._std_logger.handlers.clear()
        for handler in handlers:
            self._std_logger.handlers.append(handler)

    def addHandler(self, hdlr: _logging.Handler):
        self._std_logger.addHandler(hdlr=hdlr)

    def isEnabledFor(self, level: LogLevel) -> bool:
        return self._std_logger.isEnabledFor(level)

    def reset_attributes(self) -> None:
        """Reset logger attributes to an unset state"""
        # clear all logger settings
        self.handlers.clear()
        self._uuids.clear()
        self._columns.clear()
        self.setLevel(_DEFAULT_LEVEL)

    def setLevel(self, level: LogLevel) -> None:
        self._std_logger.setLevel(level)

    def _get_uuid(self, key: str) -> str:
        hexdigest = hashlib.md5(f"{self.name}+{key}".encode()).hexdigest()
        return hexdigest[: Logger.HASH_LEN]

    def _get_json(
        self,
        level: int,
        key: str,
        data: Optional[LogData] = None,
        description: Optional[str] = None,
    ) -> str:
        """Writes structured log message if handler allows this and level is allowed.

        Will write a header line the first time a new logging key is encountered.
        Then will only write data rows in later rows for this logging key.

        :param level: Level the message is being logged as.
        :param key: Logging key.
        :param data: Data to be logged.
        :param description: Description of this log type.

        :returns: String with JSON-encoded data row and optionally header row.
        """
        # message level less than than the logger level, early exit
        if level < self._std_logger.level:
            return

        data = _get_log_data_as_dict(data)
        data = _convert_numpy_scalars_to_python_types(data)
        header_json = None

        if key not in self._uuids:
            # new log key, so create header json row
            uuid = self._get_uuid(key)
            columns = _get_columns_from_data_dict(data)
            self._uuids[key] = uuid
            self._columns[key] = columns
            header = {
                "uuid": uuid,
                "type": "header",
                "module": self.name,
                "key": key,
                "level": _logging.getLevelName(level),
                "columns": columns,
                "description": description,
            }
            header_json = json.dumps(header)
        else:
            uuid = self._uuids[key]
            columns = _get_columns_from_data_dict(data)
            if columns != self._columns[key]:
                header_columns = set(self._columns[key].items())
                logged_columns = set(columns.items())
                msg = (
                    f"Inconsistent columns in logged values for {self.name} logger "
                    f"with key {key} compared to header generated from initial log "
                    f"entry:\n"
                    f"  Columns in header not in logged values are\n"
                    f"  {dict(sorted(header_columns - logged_columns))}\n"
                    f"  Columns in logged values not in header are\n"
                    f"  {dict(sorted(logged_columns - header_columns))}"
                )
                raise InconsistentLoggedColumnsError(msg)

        # create data json row
        row = {
            "uuid": uuid,
            "date": _get_simulation_date(),
            "values": list(data.values()),
        }
        if self._std_logger.level == DEBUG:
            # in DEBUG mode we echo the module and key for easier eyeballing
            row["module"] = self.name
            row["key"] = self.name

        row_json = json.dumps(row, cls=encoding.PandasEncoder)

        return row_json if header_json is None else f"{header_json}\n{row_json}"

    def log(
        self,
        level: LogLevel,
        key: str,
        data: LogData,
        description: Optional[str] = None,
    ) -> None:
        """Log structured data for a key at specified level with optional description.
        
        :param level: Level the message is being logged as.
        :param key: Logging key.
        :param data: Data to be logged.
        :param description: Description of this log type.
        """
        if self._std_logger.isEnabledFor(level):
            msg = self._get_json(
                level=level, key=key, data=data, description=description
            )
            self._std_logger.log(level=level, msg=msg)
            
    critical = partialmethod(log, level=CRITICAL)
    debug = partialmethod(log, level=DEBUG)
    info = partialmethod(log, level=INFO)
    warning = partialmethod(log, level=WARNING)
