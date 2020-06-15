from pytest import fixture

from tests.logging.mock_disease import MockModule
from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file


@fixture
def mock_disease_log_df(tmpdir):
    """
    Runs simulation of mock disease, parses the logfile and returns the dictionary mock disease keys
    :param tmpdir: tmpdir for logfile
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
    def test_dictionary(self, mock_disease_log_df):
        assert mock_disease_log_df
