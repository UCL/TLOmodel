import sys

import pytest

from tlo import logging


@pytest.fixture(scope='function')
def basic_configuration(tmpdir):
    """Uses all helper functions and classes from logging"""
    # simulation config
    formatter = logging.Formatter('%(levelname)s|%(name)s|%(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(stream_handler)
    logging.basicConfig(level=logging.DEBUG)

    # tlo module config
    file_name = tmpdir.join('test.log')
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    yield file_handler, file_name

    file_handler.close()


def read_file(file_handler, file_name):
    file_handler.flush()
    with open(file_name) as handle:
        lines = handle.readlines()
    return lines


def log_message(output_level, logger_level, message):
    logger = logging.getLogger("tlo.test.logger")
    logger.setLevel(logger_level)

    if output_level == 'logging.DEBUG':
        logger.debug(message)
    elif output_level == 'logging.INFO':
        logger.info(message)
    elif output_level == 'logging.WARNING':
        logger.warning(message)
    elif output_level == 'logging.CRITICAL':
        logger.critical(message)


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_same_level(basic_configuration, logging_level):
    logging_int = eval(logging_level)
    log_message(logging_level, logging_int, "test message")
    lines = read_file(*basic_configuration)

    assert [f'{logging_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_lower_level(basic_configuration, logging_level):
    logging_int = eval(logging_level) - 1
    log_message(logging_level, logging_int, "test message")
    lines = read_file(*basic_configuration)

    assert [f'{logging_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_higher_level(basic_configuration, logging_level):
    logging_int = eval(logging_level) + 1
    log_message(logging_level, logging_int, "test message")
    lines = read_file(*basic_configuration)

    assert [] == lines
