import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.methods import demography, enhanced_lifestyle

start_date = Date(2010, 1, 1)
popsize = 500


@pytest.fixture(scope='function')
def basic_configuration(tmpdir):
    """Setup basic file handler configuration"""
    # tlo module config
    file_name = tmpdir.join('test.log')
    file_handler = logging.set_output_file(file_name)

    yield file_handler, file_name

    file_handler.close()


@pytest.fixture(scope='function')
def simulation_configuration(tmpdir):
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    f = sim.configure_logging("log", directory=tmpdir)

    yield sim.output_file, f

    sim.output_file.close()
    logger = logging.getLogger('tlo.test.logger')
    logger.keys.clear()


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


def log_message(message_level, logger_level, message, logger_name='tlo.test.logger', structured_logging=False):
    """
    Sets up logger level, and writes message at the message level

    :param message_level: level that the message will be added as
    :param logger_level: level that the logger is set to
    :param message: message to be written to log
    :param logger_name: name of the logger
    :param structured_logging:

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    if structured_logging:
        if message_level == 'logging.DEBUG':
            logger.debug(key='tlo.test.logger', data=message)
        elif message_level == 'logging.INFO':
            logger.info(key='tlo.test.logger', data=message)
        elif message_level == 'logging.WARNING':
            logger.warning(key='tlo.test.logger', data=message)
        elif message_level == 'logging.CRITICAL':
            logger.critical(key='tlo.test.logger', data=message)
    else:
        if message_level == 'logging.DEBUG':
            logger.debug(message)
        elif message_level == 'logging.INFO':
            logger.info(message)
        elif message_level == 'logging.WARNING':
            logger.warning(message)
        elif message_level == 'logging.CRITICAL':
            logger.critical(message)


class TestStdLibLogging:
    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_at_same_level(self, basic_configuration, message_level):
        # given that messages are at the same level as the logger
        logger_level = eval(message_level)
        log_message(message_level, logger_level, "test message")
        lines = read_file(*basic_configuration)

        # messages should be written to log
        assert [f'{message_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines

    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_at_higher_level(self, basic_configuration, message_level):
        # given that messages are at a higher level as the logger
        logging_level = eval(message_level) - 1
        log_message(message_level, logging_level, "test message")
        lines = read_file(*basic_configuration)

        # messages should be written to log
        assert [f'{message_level.strip("logging.")}|tlo.test.logger|test message\n'] == lines

    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_at_lower_level(self, basic_configuration, message_level):
        # given that messages are at a lower level as the logger
        logging_level = eval(message_level) + 1
        log_message(message_level, logging_level, "test message")
        lines = read_file(*basic_configuration)

        # messages should not be written to log
        assert [] == lines

    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_disable(self, basic_configuration, message_level):
        # given that messages are at a higher level as the logger BUT the logger is disabled at critical
        logging_level = eval(message_level) - 1
        logging.disable(logging.CRITICAL)
        log_message(message_level, logging_level, "test message")
        lines = read_file(*basic_configuration)

        # messages should not be written to log
        assert [] == lines


class TestStructuredLogging:
    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_same_level(self, simulation_configuration, message_level):
        # given that messages are at the same level as the logger
        logger_level = eval(message_level)
        message = {"message": pd.Series([12.5])[0]}
        file_handler, file_path = simulation_configuration
        log_message(message_level, logger_level, message, structured_logging=True)

        lines = read_file(file_handler, file_path)
        header_json = json.loads(lines[0])
        data_json = json.loads(lines[1])

        # message should be written to log
        assert len(lines) == 2
        assert header_json['level'] == message_level.lstrip("logging.")
        assert 'message' in header_json['columns']
        assert header_json['columns']['message'] == 'float64'
        assert data_json['values'] == [12.5]

    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_higher_level(self, simulation_configuration, message_level):
        # given that messages are a higher level than the logger
        logger_level = eval(message_level) + 1
        message = {"message": pd.Series([12.5])[0]}
        file_handler, file_path = simulation_configuration
        log_message(message_level, logger_level, message, structured_logging=True)

        lines = read_file(file_handler, file_path)
        header_json = json.loads(lines[0])
        data_json = json.loads(lines[1])

        # message should be written to log
        assert len(lines) == 2
        assert header_json['level'] == message_level.lstrip("logging.")
        assert 'message' in header_json['columns']
        assert header_json['columns']['message'] == 'float64'
        assert data_json['values'] == [12.5]

    @pytest.mark.parametrize("message_level", ["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.CRITICAL"])
    def test_messages_lower_level(self, simulation_configuration, message_level):
        # given that messages are at a lower level than logger
        logger_level = eval(message_level) - 1
        message = {"message": pd.Series([12.5])[0]}
        file_handler, file_path = simulation_configuration
        log_message(message_level, logger_level, message, structured_logging=True)

        lines = read_file(file_handler, file_path)

        # message should be written to log
        assert [] == lines

