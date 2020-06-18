import numpy as np
import pandas as pd
from pytest import fixture

from tlo import Date, Module, Parameter, Property, Simulation, Types
from tlo.analysis.utils import parse_log_file
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.logging import INFO, getLogger

logger = getLogger("tlo.testing.loggernaires")
logger.setLevel(INFO)


class LoggerNaires(Module):
    PARAMETERS = {
        'test': Parameter(Types.REAL, 'this is a test')
    }

    PROPERTIES = {
        'ln_a': Property(Types.REAL, 'numeric a'),
        'ln_b': Property(Types.REAL, 'numeric b'),
        'ln_c': Property(Types.REAL, 'numeric c'),
        'ln_date': Property(Types.DATE, 'date'),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

    def on_birth(self, mother, child):
        pass

    def read_parameters(self, data_folder):
        pass

    def initialise_simulation(self, sim):
        sim.schedule_event(MockLogEvent(self), sim.date + pd.DateOffset(months=0))

    def initialise_population(self, population):
        df = population.props
        for name, _type in self.PROPERTIES.items():
            if name == "ln_date":
                df[name] = self.rng.randint(1400, 1600, population.initial_size) * 1_000_000_000_000_000
                df[name] = df[name].astype('datetime64[ns]')
            else:
                df[name] = self.rng.randint(0, 100, population.initial_size)


class MockLogEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(weeks=4))

    def apply(self, population):
        df = population.props
        a_over_50 = sum(df.ln_a > 50)
        b_over_50 = sum(df.ln_b > 50)
        c_over_50 = sum(df.ln_c > 50)
        # Allowing logging of entire dataframe only for testing
        logger._disable_dataframe_logging = False

        # the preferred way to log, because it maps naturally to a row in a dataframe
        logger.info(key="each_group_over_50_unscaled",
                    data={"count_a_over_50": a_over_50, "count_b_over_50": b_over_50,
                          "count_c_over_50": c_over_50 / 3},
                    description="count over 50 for each group")

        logger.info(key="each_group_over_50_scaled",
                    data={"count_a_over_50": a_over_50, "count_b_over_50": b_over_50,
                          "count_c_over_50": c_over_50 / 3},
                    description="count over 50 for each group; a and b are raw numbers, c is normalised",
                    scale_me=['count_a_over_50', 'count_c_over_50'])

        logger.info(key="a_fixed_length_list",
                    data=[a_over_50 / 2, b_over_50 / 3, c_over_50 / 4],
                    description="divide a, b, c by 2, 3, 4 respectively")

        logger.info(key="a_variable_length_list",
                    data={"list_head": list(df.loc[0:self.module.rng.randint(2, 8), "ln_a"])},
                    description="the first few interesting items from property a, random selection")

        logger.info(key="counting_but_string",
                    data="we currently have %d total count over 50" % (a_over_50 + b_over_50 + c_over_50),
                    description="total count of loggernaires over 50, but as a string")

        logger.info(key="single_individual",
                    data=df.loc[[0]],
                    description="entire record for person 0")

        logger.info(key="three_people",
                    data=df.loc[[0, 1, 2]],
                    description="three people (0-2, inclusive), output as a multi-indexed dataframe")

        logger.info(key="nested_dictionary",
                    data={
                        "count_over_50":
                            {"a": a_over_50,
                             "b": b_over_50,
                             "c": c_over_50 / 3
                             },
                    },
                    description="count over 50 for each group")

        logger.info(key="set_in_dict",
                    data={"count_over_50": set([a_over_50, b_over_50, c_over_50 / 3])},
                    description="count over 50 for each group")

        logger.info(key="with_every_type",
                    data={"a_over_50": a_over_50,
                          "mostly_nan": np.nan,
                          "c_over_50_div_2": c_over_50 / 2,
                          "b_over_50_as_list": [b_over_50],
                          "random_date": df.loc[self.module.rng.randint(0, len(df)), "ln_date"],
                          "count_over_50_as_dict": {"a": a_over_50, "b": b_over_50, "c": c_over_50 / 3},
                          "count_over_50_as_set": {a_over_50, b_over_50, c_over_50 / 3}
                          },
                    description="including a little bit of everything, columns have different types")


@fixture(scope="class")
def loggernaires_log_df(tmpdir_factory):
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
        "filename": "loggernaires_sim",  # The prefix for the output file. A timestamp will be added to this.
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
        LoggerNaires("unused_resources")
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    log_df = parse_log_file(sim.log_filepath)
    return log_df['tlo.testing.loggernaires']


class TestWriteAndReadLogFile:
    def setup(self):
        self.dates = [pd.Timestamp("2010-01-01 00:00:00"), pd.Timestamp("2010-01-29 00:00:00")]

    def test_dictionary(self, loggernaires_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "count_a_over_50": [4, 4],
             "count_b_over_50": [6, 6],
             "count_c_over_50": [2.0, 2.0],
             }
        )
        log_df = loggernaires_log_df['each_group_over_50_unscaled']

        assert expected_df.equals(log_df)

    def test_fixed_length_list(self, loggernaires_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "item_1": [2.0, 2.0],
             "item_2": [2.0, 2.0],
             "item_3": [1.5, 1.5]
             }
        )
        log_df = loggernaires_log_df['a_fixed_length_list']

        assert expected_df.equals(log_df)

    def test_variable_length_list(self, loggernaires_log_df):
        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "list_head": [
                    [46, 33, 95, 9, 66, 67],
                    [46, 33, 95, 9, 66, 67, 12],
                ],
            }
        )
        log_df = loggernaires_log_df['a_variable_length_list']

        assert expected_df.equals(log_df)

    def test_string(self, loggernaires_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "message": ["we currently have 16 total count over 50",
                         "we currently have 16 total count over 50"],
             }
        )
        log_df = loggernaires_log_df['counting_but_string']

        assert expected_df.equals(log_df)

    def test_individual(self, loggernaires_log_df):
        expected_df = pd.DataFrame(
            {"date": self.dates,
             "ln_a": [46, 46],
             "ln_b": [58, 58],
             "ln_c": [22, 22],
             "ln_date": [pd.Timestamp("2016-08-12 11:06:40") for x in range(2)],

             }
        )
        log_df = loggernaires_log_df['single_individual']

        assert expected_df.equals(log_df)

    def test_three_people(self, loggernaires_log_df):
        expected_df = pd.DataFrame.from_dict(
            data={(0, '0'): {'date': pd.Timestamp('2010-01-01 00:00:00'),
                             'ln_a': 46,
                             'ln_b': 58,
                             'ln_c': 22,
                             'ln_date': '2016-08-12T11:06:40'},
                  (0, '1'): {'date': pd.Timestamp('2010-01-01 00:00:00'),
                             'ln_a': 33,
                             'ln_b': 52,
                             'ln_c': 91,
                             'ln_date': '2017-11-18T10:13:20'},
                  (0, '2'): {'date': pd.Timestamp('2010-01-01 00:00:00'),
                             'ln_a': 95,
                             'ln_b': 93,
                             'ln_c': 47,
                             'ln_date': '2017-08-29T09:46:40'},
                  (1, '0'): {'date': pd.Timestamp('2010-01-29 00:00:00'),
                             'ln_a': 46,
                             'ln_b': 58,
                             'ln_c': 22,
                             'ln_date': '2016-08-12T11:06:40'},
                  (1, '1'): {'date': pd.Timestamp('2010-01-29 00:00:00'),
                             'ln_a': 33,
                             'ln_b': 52,
                             'ln_c': 91,
                             'ln_date': '2017-11-18T10:13:20'},
                  (1, '2'): {'date': pd.Timestamp('2010-01-29 00:00:00'),
                             'ln_a': 95,
                             'ln_b': 93,
                             'ln_c': 47,
                             'ln_date': '2017-08-29T09:46:40'}},
            orient='index'
        )
        log_df = loggernaires_log_df['three_people']

        assert expected_df.equals(log_df)

    def test_nested_dictionary(self, loggernaires_log_df):
        counts = {"a": 4, "b": 6, "c": 2.0}

        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "count_over_50": [counts, counts],

            }
        )
        log_df = loggernaires_log_df['nested_dictionary']

        assert expected_df.equals(log_df)

    def test_set_in_dict(self, loggernaires_log_df):
        # converting set to list so that ordering is correct
        counts = list({4, 6, 2.0})

        expected_df = pd.DataFrame(
            {
                "date": self.dates,
                "count_over_50": [counts, counts],

            }
        )
        log_df = loggernaires_log_df['set_in_dict']

        assert expected_df.equals(log_df)

    def test_every_type(self, loggernaires_log_df):
        counts_over_50_dict = {"a": 4, "b": 6, "c": 2.0}
        counts_over_50_list = list({4, 6, 2.0})

        expected_df = pd.DataFrame(
            {"date": self.dates,
             "a_over_50": [4, 4],
             "mostly_nan": [float("nan") for x in range(2)],
             "c_over_50_div_2": [3.0, 3.0],
             "b_over_50_as_list": [[6], [6]],
             "random_date": [pd.Timestamp("2020-06-24 12:00:00"), pd.Timestamp("2015-05-07 12:00:00")],
             "count_over_50_as_dict": [counts_over_50_dict, counts_over_50_dict],
             "count_over_50_as_set": [counts_over_50_list, counts_over_50_list]
             }
        )
        log_df = loggernaires_log_df['with_every_type']

        assert expected_df.equals(log_df)
