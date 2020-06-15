import pandas as pd
from pytest import fixture

from tests.logging.mock_disease import MockModule
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file


@fixture(scope="class")
def mock_disease_log_df(tmpdir_factory):
    """
    Runs simulation of mock disease, parses the logfile and returns the dictionary mock disease keys
    :param tmpdir_factory: tmpdir_factory for logfile
    :return: mock disease logging dictionary
    """
    # To reproduce the results, you need to set the seed for the Simulation instance. The Simulation
    # will seed the random number generators for each module when they are registered.
    # If a seed argument is not given, one is generated. It is output in the log and can be
    # used to reproduce results of a run
    seed = 567

    # By default, all output is recorded at the "INFO" level (and up) to standard out. You can
    # configure the behaviour by passing options to the `log_config` argument of
    # Simulation.
    tmpdir = tmpdir_factory.mktemp("logs")
    log_config = {
        "filename": "mock_sim",  # The prefix for the output file. A timestamp will be added to this.
        "directory": tmpdir,  # The default output path is `./output`. Change it here, if necessary
    }

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 2, 1)
    pop_size = 10

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        MockModule("unused_resources")
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    log_df = parse_log_file(sim.log_filepath)
    return log_df['tlo.testing.MockModule']


class TestWriteAndReadLogFile:
    def setup(self):
        self.dates = [pd.Timestamp("2010-01-01 00:00:00"), pd.Timestamp("2010-01-29 00:00:00")]

    def test_dictionary(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "count_a_over_50": [4, 4],
             "count_b_over_50": [6, 6],
             "count_c_over_50": [2.0, 2.0],
             }
        )
        log_df = mock_disease_log_df['each_group_over_50_unscaled']

        assert expected_df.equals(log_df)

    def test_fixed_length_list(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "item_1": [2.0, 2.0],
             "item_2": [2.0, 2.0],
             "item_3": [1.5, 1.5]
             }
        )
        log_df = mock_disease_log_df['a_fixed_length_list']

        assert expected_df.equals(log_df)

    def test_variable_length_list(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "list_head": [
                    [46, 33, 95, 9, 66, 67],
                    [46, 33, 95, 9, 66, 67, 12],
                ],
            }
        )
        log_df = mock_disease_log_df['a_variable_length_list']

        assert expected_df.equals(log_df)

    def test_string(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "message": ["we currently have 16 total count over 50",
                         "we currently have 16 total count over 50"],
             }
        )
        log_df = mock_disease_log_df['counting_but_string']

        assert expected_df.equals(log_df)

    def test_individual(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "mm_a": [46, 46],
             "mm_b": [58, 58],
             "mm_c": [22, 22],
             "mm_date": [pd.Timestamp("2016-08-12 11:06:40") for x in range(2)],

             }
        )
        log_df = mock_disease_log_df['single_individual']

        assert expected_df.equals(log_df)

    def test_three_people(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "mm_a_0": [46, 46],
             "mm_a_1": [33, 33],
             "mm_a_2": [95, 95],
             "mm_b_0": [58, 58],
             "mm_b_1": [52, 52],
             "mm_b_2": [93, 93],
             "mm_c_0": [22, 22],
             "mm_c_1": [91, 91],
             "mm_c_2": [47, 47],
             "mm_date_0": [pd.Timestamp("2016-08-12 11:06:40") for x in range(2)],
             "mm_date_1": [pd.Timestamp("2017-11-18 10:13:20") for x in range(2)],
             "mm_date_2": [pd.Timestamp("2017-08-29 09:46:40") for x in range(2)],

             }
        )
        log_df = mock_disease_log_df['three_people']

        assert expected_df.equals(log_df)

    def test_every_type(self, mock_disease_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "a_over_50": [4, 4],
             "mostly_nan": [float("nan") for x in range(2)],
             "c_over_50_div_2": [3.0, 3.0],
             "b_over_50_as_list": [[6], [6]],
             "random_date": [pd.Timestamp("2020-06-24 12:00:00"), pd.Timestamp("2015-05-07 12:00:00")]
             }
        )
        log_df = mock_disease_log_df['with_every_type']

        assert expected_df.equals(log_df)
