import contextlib
import json
import logging as _logging
import sys
from collections.abc import Generator, Iterable, Mapping
from itertools import chain, product, repeat
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

import tlo.logging as logging
import tlo.logging.core as core


def _single_row_dataframe(data: dict) -> pd.DataFrame:
    # Single row dataframe 'type' which allows construction by calling on a dictionary
    # of scalars by using an explicit length 1 index while also giving a readable
    # test parameter identifier
    return pd.DataFrame(data, index=[0])


LOGGING_LEVELS = [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL]
CATCH_ALL_LEVEL = -1
STRING_DATA_VALUES = ["foo", "bar", "spam"]
ITERABLE_DATA_VALUES = [(1, 2), (3, 1, 2), ("d", "e"), ("a", "c", 1)]
MAPPING_DATA_VALUES = [{"a": 1, "b": "spam", 2: None}, {"eggs": "foo", "bar": 1.25}]
SUPPORTED_SEQUENCE_TYPES = [list, tuple, pd.Series]
SUPPORTED_ITERABLE_TYPES = SUPPORTED_SEQUENCE_TYPES + [set]
SUPPORTED_MAPPING_TYPES = [dict, _single_row_dataframe]
LOGGER_NAMES = ["tlo", "tlo.methods"]
SIMULATION_DATE = "2010-01-01T00:00:00"


class UpdateableSimulateDateGetter:

    def __init__(self, start_date=pd.Timestamp(2010, 1, 1)):
        self._date = start_date

    def increment_date(self, days=1) -> None:
        self._date += pd.DateOffset(days=days)

    def __call__(self) -> str:
        return self._date.isoformat()


@pytest.fixture
def simulation_date_getter() -> core.SimulationDateGetter:
    return lambda: SIMULATION_DATE


@pytest.fixture
def root_level() -> core.LogLevel:
    return logging.WARNING


@pytest.fixture
def stdout_handler_level() -> core.LogLevel:
    return logging.DEBUG


@pytest.fixture
def add_stdout_handler() -> bool:
    return False


@pytest.fixture(autouse=True)
def initialise_logging(
    add_stdout_handler: bool,
    simulation_date_getter: core.SimulationDateGetter,
    root_level: core.LogLevel,
    stdout_handler_level: core.LogLevel,
) -> Generator[None, None, None]:
    logging.initialise(
        add_stdout_handler=add_stdout_handler,
        simulation_date_getter=simulation_date_getter,
        root_level=root_level,
        stdout_handler_level=stdout_handler_level,
    )
    yield
    logging.reset()


@pytest.mark.parametrize("add_stdout_handler", [True, False])
@pytest.mark.parametrize("root_level", LOGGING_LEVELS, ids=_logging.getLevelName)
@pytest.mark.parametrize(
    "stdout_handler_level", LOGGING_LEVELS, ids=_logging.getLevelName
)
def test_initialise_logging(
    add_stdout_handler: bool,
    simulation_date_getter: core.SimulationDateGetter,
    root_level: core.LogLevel,
    stdout_handler_level: core.LogLevel,
) -> None:
    logger = logging.getLogger("tlo")
    assert logger.level == root_level
    if add_stdout_handler:
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, _logging.StreamHandler)
        assert handler.stream is sys.stdout
        assert handler.level == stdout_handler_level
    else:
        assert len(logger.handlers) == 0
    assert core._get_simulation_date is simulation_date_getter


def _check_handlers(
    logger: core.Logger, expected_number_handlers: int, expected_log_path: Path
) -> None:
    assert len(logger.handlers) == expected_number_handlers
    file_handlers = [h for h in logger.handlers if isinstance(h, _logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename == str(expected_log_path)


@pytest.mark.parametrize("add_stdout_handler", [True, False])
def test_set_output_file(add_stdout_handler: bool, tmp_path: Path) -> None:
    log_path_1 = tmp_path / "test-1.log"
    log_path_2 = tmp_path / "test-2.log"
    logging.set_output_file(log_path_1)
    logger = logging.getLogger("tlo")
    expected_number_handlers = 2 if add_stdout_handler else 1
    _check_handlers(logger, expected_number_handlers, log_path_1)
    # Setting output file a second time should replace previous file handler rather
    # than add an additional handler and keep existing
    logging.set_output_file(log_path_2)
    _check_handlers(logger, expected_number_handlers, log_path_2)


@pytest.mark.parametrize("logger_name", ["tlo", "tlo.methods"])
def test_getLogger(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    assert logger.name == logger_name
    assert isinstance(logger.handlers, list)
    assert isinstance(logger.level, int)
    assert logger.isEnabledFor(logger.level)
    assert logging.getLogger(logger_name) is logger


@pytest.mark.parametrize("logger_name", ["foo", "spam.tlo"])
def test_getLogger_invalid_name_raises(logger_name: str) -> None:
    with pytest.raises(AssertionError, match=logger_name):
        logging.getLogger(logger_name)


@pytest.mark.parametrize("mapping_data", MAPPING_DATA_VALUES)
@pytest.mark.parametrize("mapping_type", SUPPORTED_MAPPING_TYPES)
def test_get_log_data_as_dict_with_mapping_types(
    mapping_data: Mapping, mapping_type: Callable
) -> None:
    log_data = mapping_type(mapping_data)
    data_dict = core._get_log_data_as_dict(log_data)
    assert len(data_dict) == len(mapping_data)
    assert set(data_dict.keys()) == set(map(str, mapping_data.keys()))
    assert set(data_dict.values()) == set(mapping_data.values())
    # Dictionary returned should be invariant to original ordering
    assert data_dict == core._get_log_data_as_dict(
        mapping_type(dict(reversed(mapping_data.items())))
    )


@pytest.mark.parametrize("mapping_data", MAPPING_DATA_VALUES)
def test_get_log_data_as_dict_with_multirow_dataframe_raises(
    mapping_data: Mapping,
) -> None:
    log_data = pd.DataFrame(mapping_data, index=[0, 1])
    with pytest.raises(ValueError, match="multirow"):
        core._get_log_data_as_dict(log_data)


@pytest.mark.parametrize("values", ITERABLE_DATA_VALUES)
@pytest.mark.parametrize("sequence_type", SUPPORTED_SEQUENCE_TYPES)
def test_get_log_data_as_dict_with_sequence_types(
    values: Iterable, sequence_type: Callable
) -> None:
    log_data = sequence_type(values)
    data_dict = core._get_log_data_as_dict(log_data)
    assert len(data_dict) == len(log_data)
    assert list(data_dict.keys()) == [f"item_{i+1}" for i in range(len(log_data))]
    assert list(data_dict.values()) == list(log_data)


@pytest.mark.parametrize("values", ITERABLE_DATA_VALUES)
def test_get_log_data_as_dict_with_set(values: Iterable) -> None:
    data = set(values)
    data_dict = core._get_log_data_as_dict(data)
    assert len(data_dict) == len(data)
    assert list(data_dict.keys()) == [f"item_{i+1}" for i in range(len(data))]
    assert set(data_dict.values()) == data
    # Dictionary returned should be invariant to original ordering
    assert data_dict == core._get_log_data_as_dict(set(reversed(values)))


def test_convert_numpy_scalars_to_python_types() -> None:
    data = {
        "a": np.int64(1),
        "b": np.int32(42),
        "c": np.float64(0.5),
        "d": np.bool_(True),
    }
    expected_converted_data = {"a": 1, "b": 42, "c": 0.5, "d": True}
    converted_data = core._convert_numpy_scalars_to_python_types(data)
    assert converted_data == expected_converted_data


def test_get_columns_from_data_dict() -> None:
    data = {
        "a": 1,
        "b": 0.5,
        "c": False,
        "d": "foo",
        "e": pd.Timestamp("2010-01-01"),
    }
    expected_columns = {
        "a": "int",
        "b": "float",
        "c": "bool",
        "d": "str",
        "e": "Timestamp",
    }
    columns = core._get_columns_from_data_dict(data)
    assert columns == expected_columns


@contextlib.contextmanager
def _propagate_to_root() -> Generator[None, None, None]:
    # Enable propagation to root logger to allow pytest capturing to work
    root_logger = logging.getLogger("tlo")
    root_logger._std_logger.propagate = True
    yield
    root_logger._std_logger.propagate = False


def _setup_caplog_and_get_logger(
    caplog: pytest.LogCaptureFixture, logger_name: str, logger_level: core.LogLevel
) -> core.Logger:
    caplog.set_level(CATCH_ALL_LEVEL, logger_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    return logger


@pytest.mark.parametrize("disable_level", LOGGING_LEVELS, ids=_logging.getLevelName)
@pytest.mark.parametrize("logger_level_offset", [-5, 0, 5])
@pytest.mark.parametrize("data", STRING_DATA_VALUES)
@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
def test_disable(
    disable_level: core.LogLevel,
    logger_level_offset: int,
    data: str,
    logger_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = _setup_caplog_and_get_logger(caplog, logger_name, CATCH_ALL_LEVEL)
    logging.disable(disable_level)
    assert not logger.isEnabledFor(disable_level)
    message_level = disable_level + logger_level_offset
    with _propagate_to_root():
        logger.log(message_level, key="message", data=data)
    if message_level > disable_level:
        # Message level is above disable level and so should have been captured
        assert len(caplog.records) == 1
        assert data in caplog.records[0].msg
    else:
        # Message level is below disable level and so should not have been captured
        assert len(caplog.records) == 0


def _check_captured_log_output_for_levels(
    caplog: pytest.LogCaptureFixture,
    message_level: core.LogLevel,
    logger_level: core.LogLevel,
    data: str,
) -> None:
    if message_level >= logger_level:
        # Message level is at or above logger's level and so should have been captured
        assert len(caplog.records) == 1
        assert data in caplog.records[0].msg
    else:
        # Message level is below logger's set level and so should not have been captured
        assert len(caplog.records) == 0


@pytest.mark.parametrize("message_level", LOGGING_LEVELS, ids=_logging.getLevelName)
@pytest.mark.parametrize("logger_level_offset", [-5, 0, 5])
@pytest.mark.parametrize("data", STRING_DATA_VALUES)
@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
def test_logging_with_log(
    message_level: core.LogLevel,
    logger_level_offset: int,
    data: str,
    logger_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger_level = message_level + logger_level_offset
    logger = _setup_caplog_and_get_logger(caplog, logger_name, logger_level)
    with _propagate_to_root():
        logger.log(level=message_level, key="message", data=data)
    _check_captured_log_output_for_levels(caplog, message_level, logger_level, data)


@pytest.mark.parametrize("message_level", LOGGING_LEVELS, ids=_logging.getLevelName)
@pytest.mark.parametrize("logger_level_offset", [-5, 0, 5])
@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
@pytest.mark.parametrize("data", STRING_DATA_VALUES)
def test_logging_with_convenience_methods(
    message_level: core.LogLevel,
    logger_level_offset: int,
    data: str,
    logger_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger_level = message_level + logger_level_offset
    logger = _setup_caplog_and_get_logger(caplog, logger_name, logger_level)
    convenience_method = getattr(logger, _logging.getLevelName(message_level).lower())
    with _propagate_to_root():
        convenience_method(key="message", data=data)
    _check_captured_log_output_for_levels(caplog, message_level, logger_level, data)


def _check_header(
    header: dict[str, str | dict[str, str]],
    expected_module: str,
    expected_key: str,
    expected_level: str,
    expected_description: str,
    expected_columns: dict[str, str],
) -> None:
    assert set(header.keys()) == {
        "uuid",
        "type",
        "module",
        "key",
        "level",
        "columns",
        "description",
    }
    assert isinstance(header["uuid"], str)
    assert set(header["uuid"]) <= set("abcdef0123456789")
    assert header["type"] == "header"
    assert header["module"] == expected_module
    assert header["key"] == expected_key
    assert header["level"] == expected_level
    assert header["description"] == expected_description
    assert isinstance(header["columns"], dict)
    assert header["columns"] == expected_columns


def _check_row(
    row: dict[str, str],
    logger_level: core.LogLevel,
    expected_uuid: str,
    expected_date: str,
    expected_values: list,
    expected_module: str,
    expected_key: str,
) -> None:
    assert row["uuid"] == expected_uuid
    assert row["date"] == expected_date
    assert row["values"] == expected_values
    if logger_level == logging.DEBUG:
        assert row["module"] == expected_module
        assert row["key"] == expected_key


def _parse_and_check_log_records(
    caplog: pytest.LogCaptureFixture,
    logger_name: str,
    logger_level: core.LogLevel,
    message_level: core.LogLevel,
    data_dicts: dict,
    dates: str,
    keys: str,
    description: str | None = None,
) -> None:
    headers = {}
    for record, data_dict, date, key in zip(caplog.records, data_dicts, dates, keys):
        message_lines = record.msg.split("\n")
        if key not in headers:
            # First record for key therefore expect both header and row lines
            assert len(message_lines) == 2
            header_line, row_line = message_lines
            headers[key] = json.loads(header_line)
            _check_header(
                header=headers[key],
                expected_module=logger_name,
                expected_key=key,
                expected_level=_logging.getLevelName(logger_level),
                expected_description=description,
                expected_columns=logging.core._get_columns_from_data_dict(data_dict),
            )
        else:
            # Subsequent records for key should only have row line
            assert len(message_lines) == 1
            row_line = message_lines[0]
        row = json.loads(row_line)
        _check_row(
            row=row,
            logger_level=message_level,
            expected_uuid=headers[key]["uuid"],
            expected_date=date,
            expected_values=list(data_dict.values()),
            expected_module=logger_name,
            expected_key=key,
        )


@pytest.mark.parametrize("level", LOGGING_LEVELS, ids=_logging.getLevelName)
@pytest.mark.parametrize(
    "data_type,data",
    list(
        chain(
            zip([str] * len(STRING_DATA_VALUES), STRING_DATA_VALUES),
            product(SUPPORTED_ITERABLE_TYPES, ITERABLE_DATA_VALUES),
            product(SUPPORTED_MAPPING_TYPES, MAPPING_DATA_VALUES),
        )
    ),
)
@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
@pytest.mark.parametrize("key", STRING_DATA_VALUES)
@pytest.mark.parametrize("description", [None, "test"])
@pytest.mark.parametrize("number_repeats", [1, 2, 3])
def test_logging_structured_data(
    level: core.LogLevel,
    data_type: Callable,
    data: Mapping | Iterable,
    logger_name: str,
    key: str,
    description: str,
    number_repeats: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = _setup_caplog_and_get_logger(caplog, logger_name, level)
    log_data = data_type(data)
    data_dict = logging.core._get_log_data_as_dict(log_data)
    with _propagate_to_root():
        for _ in range(number_repeats):
            logger.log(level=level, key=key, data=log_data, description=description)
    assert len(caplog.records) == number_repeats
    _parse_and_check_log_records(
        caplog=caplog,
        logger_name=logger_name,
        logger_level=level,
        message_level=level,
        data_dicts=repeat(data_dict),
        dates=repeat(SIMULATION_DATE),
        keys=repeat(key),
        description=description,
    )


@pytest.mark.parametrize("simulation_date_getter", [UpdateableSimulateDateGetter()])
@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
@pytest.mark.parametrize("number_dates", [2, 3])
def test_logging_updating_simulation_date(
    simulation_date_getter: core.SimulationDateGetter,
    logger_name: str,
    root_level: core.LogLevel,
    number_dates: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = _setup_caplog_and_get_logger(caplog, logger_name, root_level)
    key = "message"
    data = "spam"
    data_dict = logging.core._get_log_data_as_dict(data)
    dates = []
    with _propagate_to_root():
        for _ in range(number_dates):
            logger.log(level=root_level, key=key, data=data)
            dates.append(simulation_date_getter())
            simulation_date_getter.increment_date()
    # Dates should be unique
    assert len(set(dates)) == len(dates)
    assert len(caplog.records) == number_dates
    _parse_and_check_log_records(
        caplog=caplog,
        logger_name=logger_name,
        logger_level=root_level,
        message_level=root_level,
        data_dicts=repeat(data_dict),
        dates=dates,
        keys=repeat(key),
        description=None,
    )


@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
def test_logging_structured_data_multiple_keys(
    logger_name: str,
    root_level: core.LogLevel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = _setup_caplog_and_get_logger(caplog, logger_name, root_level)
    keys = ["foo", "bar", "foo", "foo", "bar"]
    data_values = ["a", "b", "c", "d", "e"]
    data_dicts = [logging.core._get_log_data_as_dict(data) for data in data_values]
    with _propagate_to_root():
        for key, data in zip(keys, data_values):
            logger.log(level=root_level, key=key, data=data)
    assert len(caplog.records) == len(keys)
    _parse_and_check_log_records(
        caplog=caplog,
        logger_name=logger_name,
        logger_level=root_level,
        message_level=root_level,
        data_dicts=data_dicts,
        dates=repeat(SIMULATION_DATE),
        keys=keys,
        description=None,
    )


@pytest.mark.parametrize("level", LOGGING_LEVELS)
def test_logging_to_file(level: core.LogLevel, tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    file_handler = logging.set_output_file(log_path)
    loggers = [logging.getLogger(name) for name in LOGGER_NAMES]
    key = "message"
    for logger, data in zip(loggers, STRING_DATA_VALUES):
        logger.setLevel(level)
        logger.log(level=level, key=key, data=data)
    _logging.shutdown([lambda: file_handler])
    with log_path.open("r") as log_file:
        log_lines = log_file.readlines()
    # Should have two lines (one header + one data row per logger)
    assert len(log_lines) == 2 * len(loggers)
    for name, data in zip(LOGGER_NAMES, STRING_DATA_VALUES):
        header = json.loads(log_lines.pop(0))
        row = json.loads(log_lines.pop(0))
        _check_header(
            header=header,
            expected_module=name,
            expected_key=key,
            expected_level=_logging.getLevelName(level),
            expected_description=None,
            expected_columns={key: "str"},
        )
        _check_row(
            row=row,
            logger_level=level,
            expected_uuid=header["uuid"],
            expected_date=SIMULATION_DATE,
            expected_values=[data],
            expected_module=name,
            expected_key=key,
        )


@pytest.mark.parametrize(
    "inconsistent_data_iterables",
    [
        ({"a": 1, "b": 2}, {"a": 3, "b": 4, "c": 5}),
        ({"a": 1}, {"b": 2}),
        ({"a": None, "b": 2}, {"a": 1, "b": 2}),
        ([1], [0.5]),
        (["a", "b"], ["a", "b", "c"]),
        ("foo", "bar", ["spam"]),
    ],
)
def test_logging_structured_data_inconsistent_columns_raises(
    inconsistent_data_iterables: Iterable[core.LogData], root_level: core.LogLevel
) -> None:
    logger = logging.getLogger("tlo")
    with pytest.raises(core.InconsistentLoggedColumnsError):
        for data in inconsistent_data_iterables:
            logger.log(level=root_level, key="message", data=data)


@pytest.mark.parametrize(
    "consistent_data_iterables",
    [
        ([np.int64(1)], [2], [np.int32(1)]),
        ([{"a": np.bool_(False)}, {"a": False}]),
        ((1.5, 2), (np.float64(0), np.int64(2))),
    ],
)
def test_logging_structured_data_mixed_numpy_python_scalars(
    consistent_data_iterables: Iterable[core.LogData], root_level: core.LogLevel
) -> None:
    logger = logging.getLogger("tlo")
    # Should run without any exceptions
    for data in consistent_data_iterables:
        logger.log(level=root_level, key="message", data=data)
