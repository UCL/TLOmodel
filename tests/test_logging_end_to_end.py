import pickle
from collections.abc import Mapping
from io import StringIO

import pandas as pd
from pytest import fixture

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file


@fixture(scope="session")
def log_input():
    """Creates dataframe for logging"""
    # a dataframe containing data we want to log (and get back out)
    log_string = "\n".join((
        "col1_str;hello;world;lorem;ipsum;dolor;sit",
        "col2_int;1;3;5;7;8;10",
        "col3_float;2;4;6;8;9;null",
        "col4_cat;cat1;cat1;cat2;cat2;cat1;cat2",
        "col5_set;set();{'one'};{None};{'three','four'};{'eight'};set()",
        "col6_list;[];['two'];[None];[5, 6, 7];[];[]",
        "col7_date;2020-06-19T00:22:58.586101;2020-06-20T00:23:58.586101;2020-06-21T00:24:58.586101;2020-06-22T00:25"
        ":58.586101;2020-06-23T00:25:58.586101;null",
        "col8_fixed_list;['one', 1];['two', 2];[None, None];['three', 3];['four', 4];['five', 5]"
    ))
    # read in, then transpose
    log_input = pd.read_csv(StringIO(log_string), sep=';').T
    log_input.reset_index(inplace=True)
    log_input.columns = log_input.iloc[0]
    log_input.drop(log_input.index[0], axis=0, inplace=True)
    log_input.reset_index(inplace=True, drop=True)

    # give columns the proper type
    log_input.col4_cat = log_input.col4_cat.astype('category')
    log_input.col5_set = log_input.col5_set.apply(lambda x: eval(x))  # use python eval to create a column of sets
    log_input.col6_list = log_input.col6_list.apply(lambda x: eval(x))  # use python eval to create a column of lists
    log_input.col7_date = log_input.col7_date.astype('datetime64')
    log_input.col8_fixed_list = log_input.col8_fixed_list.apply(
        lambda x: eval(x))  # use python eval to create a column of lists
    return log_input


@fixture(scope="class")
def class_scoped_seed(request):
    return request.config.getoption("--seed")


@fixture(scope="class")
def log_path(tmpdir_factory, log_input, class_scoped_seed):
    """
    Runs simulation of mock disease, returns the logfile path
    :param log_input: dataframe to log from
    :param tmpdir_factory: tmpdir_factory for logfile
    :return: logfile path
    """

    # imagine we have a simulation
    sim = Simulation(start_date=Date(2010, 1, 1),
                     seed=class_scoped_seed,
                     log_config={'filename': 'logfile', 'directory': tmpdir_factory.mktemp("logs")})

    # a logger connected to that simulation
    logger = logging.getLogger('tlo.test')
    logger.setLevel(logging.INFO)
    # Allowing logging of entire dataframe only for testing
    logger._disable_dataframe_logging = False

    # log data as dicts
    for index, row in log_input.iterrows():
        logger.info(key='rows_as_dicts', data=row.to_dict())
        sim.date = sim.date + pd.DateOffset(days=1)

    # log data as single-row dataframe
    for index in range(len(log_input)):
        logger.info(key='rows_as_individuals', data=log_input.loc[[index]])
        sim.date = sim.date + pd.DateOffset(days=1)

    # log data as multi-row dataframe
    for _ in range(2):
        logger.info(key='multi_row_df', data=log_input)
        sim.date = sim.date + pd.DateOffset(days=1)

    # log data as fixed length list
    for item in log_input.col8_fixed_list.values:
        logger.info(key='a_fixed_length_list',
                    data=item)
        sim.date = sim.date + pd.DateOffset(days=1)

    # log data as variable length list
    for item in log_input.col6_list.values:
        logger.info(key='a_variable_length_list',
                    data={'list_header': item})
        sim.date = sim.date + pd.DateOffset(days=1)

    # test logging of strings
    logger.info(key='string_value',
                data='I am a message.')
    sim.date = sim.date + pd.DateOffset(days=1)

    # test categorical
    for item in log_input.col4_cat:
        logger.info(key='categorical',
                    data={'cat': pd.Categorical(item, categories=['cat1', 'cat2'])})

    # end the simulation
    sim.simulate(end_date=sim.date)

    return sim.log_filepath


@fixture(scope="class")
def test_log_df(log_path):
    """
    Convenience fixture to run loggernaires simulation, parse the logfile and return the data for loggernaires module
    :param log_path: fixture to run the simulation, returining the log path
    :return: dictionary of 'tlo.test' dataframes
    """
    return parse_log_file(log_path)['tlo.test']


class TestWriteAndReadLog:
    def test_rows_as_dicts(self, test_log_df, log_input):
        # get table to compare
        log_output = test_log_df['rows_as_dicts'].drop(['date'], axis=1)

        # categories need to be set manually
        log_output.col4_cat = log_output.col4_cat.astype('category')

        assert log_input.equals(log_output)

    def test_rows_as_individuals(self, test_log_df, log_input):
        # get table to compare
        log_output = test_log_df['rows_as_individuals'].drop(['date'], axis=1)

        # categories need to be set manually
        log_output.col4_cat = log_output.col4_cat.astype('category')
        assert log_input.equals(log_output)

    def test_log_entire_df(self, test_log_df, log_input):
        # get table to compare
        log_output = test_log_df['multi_row_df'].drop(['date'], axis=1)

        # within nested dicts/entire df, need manual setting of special types
        log_output.col4_cat = log_output.col4_cat.astype('category')
        log_input.col5_set = log_input.col5_set.apply(list)
        log_output.col7_date = log_output.col7_date.astype('datetime64')
        # deal with index matching by resetting index
        log_output.reset_index(inplace=True, drop=True)
        expected_output = log_input.append(log_input, ignore_index=True)

        assert expected_output.equals(log_output)

    def test_fixed_length_list(self, test_log_df):
        log_df = test_log_df['a_fixed_length_list'].drop(['date'], axis=1)

        expected_output = pd.DataFrame(
            {'item_1': ['one', 'two', None, 'three', 'four', 'five'],
             'item_2': [1, 2, None, 3, 4, 5]}
        )

        assert expected_output.equals(log_df)

    def test_variable_length_list(self, test_log_df, log_input):
        log_df = test_log_df['a_variable_length_list']

        assert log_input.col6_list.equals(log_df.list_header)

    def test_string(self, test_log_df):
        log_df = test_log_df['string_value']

        assert pd.Series('I am a message.').equals(log_df.message)

    def test_categorical(self, test_log_df, log_input):
        log_df = test_log_df['categorical']
        assert log_input.col4_cat.equals(log_df.cat)


def test_parse_log_file(log_path):
    parsed_log = parse_log_file(log_path)
    assert isinstance(parsed_log, Mapping)
    assert 'tlo.test' in parsed_log


def test_parsed_log_object_pickleable(log_path, tmp_path):
    parsed_log = parse_log_file(log_path)
    with open(tmp_path / "parsed_log.pickle", "wb") as pickle_file:
        pickle.dump(parsed_log, pickle_file, pickle.HIGHEST_PROTOCOL)
