from io import StringIO

import numpy as np
import pandas as pd
from pytest import fixture

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file

# a dataframe containing data we want to log (and get back out)
LOG_INPUT = """
col1_str;hello;world;lorem;ipsum;dolor;sit
col2_int;1;3;5;7;8;10
col3_float;2;4;6;8;9;null
col4_cat;cat1;cat1;cat2;cat2;cat1;cat2
col5_set;set();{'one'};{None};{'three','four'};{'eight'};set()
col6_list;[];['two'];[None];[5, 6, 7];[];[]
col7_date;2020-06-19T00:22:58.586101;2020-06-20T00:23:58.586101;2020-06-21T00:24:58.586101;2020-06-22T00:25:58.586101;2020-06-23T00:25:58.586101;null
"""
# read in, then transpose
log_input = pd.read_csv(StringIO(LOG_INPUT), sep=';', skiprows=1).T
log_input.columns = log_input.iloc[0]
log_input.drop(log_input.index[0], axis=0, inplace=True)
log_input.reset_index(inplace=True)
# give columns the proper type
log_input.col4_cat = log_input.col4_cat.astype('category')
log_input.col5_set = log_input.col5_set.apply(lambda x: eval(x))  # use python eval to create a column of sets
log_input.col6_list = log_input.col6_list.apply(lambda x: eval(x))  # use python eval to create a column of lists
log_input.col7_date = log_input.col7_date.astype('datetime64')


@fixture(scope="class")
def log_path(tmpdir_factory):
    """
    Runs simulation of mock disease, returns the logfile path
    :param tmpdir_factory: tmpdir_factory for logfile
    :return: logfile path
    """
    # imagine we have a simulation
    sim = Simulation(start_date=Date(2010, 1, 1),
                     seed=0,
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
    logger.info(key='multi_row_df', data=log_input)
    sim.date = sim.date + pd.DateOffset(days=1)

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
    def setup(self):
        self.dates = [pd.Timestamp("2010-01-01 00:00:00"), pd.Timestamp("2010-01-29 00:00:00")]

    def test_rows_as_dicts(self, test_log_df):
        # get table to compare
        log_output = test_log_df['rows_as_dicts'].drop(['date'], axis=1)

        # categories need to be set manually
        log_output.col4_cat = log_output.col4_cat.astype('category')

        test = log_input
        assert log_input.equals(log_output)

    def test_rows_as_individuals(self, test_log_df):
        # get table to compare
        log_output = test_log_df['rows_as_individuals'].drop(['date'], axis=1)

        # categories need to be set manually
        log_output.col4_cat = log_output.col4_cat.astype('category')
        assert log_input.equals(log_output)

    def test_log_entire_df(self, test_log_df):
        # get table to compare
        log_output = test_log_df['rows_as_individuals'].drop(['date'], axis=1)

        # categories need to be set manually
        log_output.col4_cat = log_output.col4_cat.astype('category')

        assert log_input.equals(log_output)

    def test_fixed_length_list(self, test_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "item_1": [2.0, 2.0],
             "item_2": [2.0, 2.0],
             "item_3": [1.5, 1.5]
             }
        )
        log_df = test_log_df['a_fixed_length_list']

        assert expected_df.equals(log_df)

    def test_variable_length_list(self, test_log_df):
        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "list_head": [
                    [46, 33, 95, 9, 66, 67],
                    [46, 33, 95, 9, 66, 67, 12],
                ],
            }
        )
        log_df = test_log_df['a_variable_length_list']

        assert expected_df.equals(log_df)

    def test_string(self, test_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "message": ["we currently have 16 total count over 50",
                         "we currently have 16 total count over 50"],
             }
        )
        log_df = test_log_df['counting_but_string']

        assert expected_df.equals(log_df)

    def test_individual(self, test_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "ln_a": [46, 46],
             "ln_b": [58, 58],
             "ln_c": [22, 22],
             "ln_set": [{5, 94}, {5, 94}],
             "ln_date": [pd.Timestamp("2014-09-06 10:40:00") for x in range(2)],
             }
        )
        log_df = test_log_df['single_individual']

        assert expected_df.equals(log_df)

    def test_three_people(self, test_log_df):
        expected_df = pd.DataFrame.from_dict(
            data={(0, '0'): {'date': self.dates[0],
                             'ln_a': 46,
                             'ln_b': 58,
                             'ln_c': 22,
                             'ln_set': [5, 94],
                             'ln_date': '2014-09-06T10:40:00'},
                  (0, '1'): {'date': self.dates[0],
                             'ln_a': 33,
                             'ln_b': 52,
                             'ln_c': 91,
                             'ln_set': None,
                             'ln_date': '2015-02-27T01:20:00'},
                  (0, '2'): {'date': self.dates[0],
                             'ln_a': 95,
                             'ln_b': 93,
                             'ln_c': 47,
                             'ln_set': [],
                             'ln_date': '2020-06-12T22:13:20'},
                  (1, '0'): {'date': self.dates[1],
                             'ln_a': 46,
                             'ln_b': 58,
                             'ln_c': 22,
                             'ln_set': [5, 94],
                             'ln_date': '2014-09-06T10:40:00'},
                  (1, '1'): {'date': self.dates[1],
                             'ln_a': 33,
                             'ln_b': 52,
                             'ln_c': 91,
                             'ln_set': None,
                             'ln_date': '2015-02-27T01:20:00'},
                  (1, '2'): {'date': self.dates[1],
                             'ln_a': 95,
                             'ln_b': 93,
                             'ln_c': 47,
                             'ln_set': [],
                             'ln_date': '2020-06-12T22:13:20'}},
            orient='index'
        )
        log_df = test_log_df['three_people']

        assert expected_df.equals(log_df)

    def test_nested_dictionary(self, test_log_df):
        counts = {"a": 4, "b": 6, "c": 2.0}

        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "count_over_50": [counts, counts],

            }
        )
        log_df = test_log_df['nested_dictionary']

        assert expected_df.equals(log_df)

    def test_set_in_dict(self, test_log_df):
        counts = {4, 6, 2.0}

        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "count_over_50": [counts, counts],

            }
        )
        log_df = test_log_df['set_in_dict']

        assert expected_df.equals(log_df)


def test_missing_values(self, test_log_df):
    """Only NaT is converted to None, other missing values are only valid in python's parsing of json"""
    expected_df = pd.DataFrame(
        {"date": self.dates,
         "NaT": [None, None],
         "NaN": [np.nan, np.nan],
         "float_inf": [float("inf"), float("inf")],
         "float_-inf": [float("-inf"), float("-inf")],
         "float_nan": [float("nan"), float("nan")],
         }
    )
    log_df = test_log_df['missing_values']

    assert expected_df.equals(log_df)


def test_every_type(self, test_log_df):
    counts_over_50_dict = {"a": 4, "b": 6, "c": 2.0}
    counts_over_50_set = {4, 6, 2.0}

    expected_df = pd.DataFrame(
        {"date": self.dates,
         "a_over_50": [4, 4],
         "mostly_nan": [float("nan") for x in range(2)],
         "c_over_50_div_2": [3.0, 3.0],
         "b_over_50_as_list": [[6], [6]],
         "random_date": [pd.Timestamp("2020-06-12 22:13:20"), pd.Timestamp("2019-07-01 16:53:20")],
         "count_over_50_as_dict": [counts_over_50_dict, counts_over_50_dict],
         "count_over_50_as_set": [counts_over_50_set, counts_over_50_set]
         }
    )
    log_df = test_log_df['with_every_type']

    assert expected_df.equals(log_df)


class TestParseLogAtLoggingLevel:
    def setup(self):
        self.dates = [pd.Timestamp("2010-01-01 00:00:00"), pd.Timestamp("2010-01-29 00:00:00")]

    def test_same_level(self, log_path):
        parsed_log = parse_log_file(log_path, level=logging.INFO)

        assert 'tlo.test' in parsed_log.keys()

    def test_parse_log_at_higher_level(self, log_path):
        parsed_log = parse_log_file(log_path, level=logging.CRITICAL)

        assert parsed_log == {}

    def test_parse_log_at_lower_level(self, log_path):
        parsed_log = parse_log_file(log_path, level=logging.DEBUG)

        assert 'tlo.test' in parsed_log.keys()
