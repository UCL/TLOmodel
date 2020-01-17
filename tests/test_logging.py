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
    """
    Reads file and returns the lines
    :param file_handler: filehandler (to flush) though might be a bit unnecessary
    :param file_name: path to file
    :return: list of lines
    """
    file_handler.flush()
    with open(file_name) as handle:
        lines = handle.readlines()
    return lines


def log_message(message_level, logger_level, message):
    """
    Sets up logger level, and writes message at the message level

    :param message_level: level that the message will be added as
    :param logger_level: level that the logger is set to
    :param message: message to be written to log
    """
    logger = logging.getLogger("tlo.test.logger")
    logger.setLevel(logger_level)

    if message_level == 'logging.DEBUG':
        logger.debug(message)
    elif message_level == 'logging.INFO':
        logger.info(message)
    elif message_level == 'logging.WARNING':
        logger.warning(message)
    elif message_level == 'logging.CRITICAL':
        logger.critical(message)


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_same_level(basic_configuration, message_level):
    # given that messages are at the same level as the logger
    logger_level = eval(message_level)
    log_message(message_level, logger_level, "test message")
    lines = read_file(*basic_configuration)

    # messages should be written to log
    assert [f'{message_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_higher_level(basic_configuration, message_level):
    # given that messages are at a higher level as the logger
    logging_level = eval(message_level) - 1
    log_message(message_level, logging_level, "test message")
    lines = read_file(*basic_configuration)

    # messages should be written to log
    assert [f'{message_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines


@pytest.mark.parametrize("logging_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
def test_messages_at_lower_level(basic_configuration, message_level):
    # given that messages are at a lower level as the logger
    logging_level = eval(message_level) + 1
    log_message(message_level, logging_level, "test message")
    lines = read_file(*basic_configuration)

    # messages should not be written to log
    assert [] == lines
