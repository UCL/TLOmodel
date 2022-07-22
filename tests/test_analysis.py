import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import (
    flatten_multi_index_series_into_dict_for_logging,
    get_root_path,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.methods import demography

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def test_parse_log():
    log_file = Path(__file__).parent / 'resources' / 'structured_log.txt'

    output = parse_log_file(log_file)

    assert 'tlo.methods.epilepsy' in output
    assert set(output['tlo.methods.epilepsy'].keys()) == {'incidence_epilepsy', 'epilepsy_logging', '_metadata'}


def test_flattening_and_unflattening_multiindex(tmpdir):
    """Check that a pd.Series with multi-index can be "flattened" (in order to log it) and "unflattened" (after reading
    the log) using `flatten_multi_index_series_into_dict_for_logging` and `unflatten_flattened_multi_index_in_logging`,
    respectively."""

    def run_simulation_and_parse_log(series_to_log: pd.Series) -> pd.DataFrame:
        """Run the simulation, convert a pd.Series to a dict using `flatten_multi_index_series_into_dict_for_logging`,
        log it to a particular key on the first date of the simulation, and return the pd.DataFrame created by
        `parse_log` for that key."""

        logger = logging.getLogger('tlo.methods.demography')
        logger.setLevel(logging.INFO)

        class DummyModule(Module):
            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                logger.info(
                    key='key',
                    data=flatten_multi_index_series_into_dict_for_logging(series_to_log)
                )

        sim = Simulation(start_date=sim_start_date, seed=0, log_config={
            'filename': 'temp',
            'directory': tmpdir,
        })
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            DummyModule()
        )
        sim.make_initial_population(n=100)
        sim.simulate(end_date=sim_start_date)

        return parse_log_file(sim.log_filepath)['tlo.methods.demography']['key']

    sim_start_date = Date(2010, 1, 1)

    for num_of_levels in range(1, 4):
        # Make original pd.Series with column-wise multi-index (with the specified number of levels).
        idx = pd.MultiIndex.from_product(
            [['1', '2', '3'] for _ in range(num_of_levels)],
            names=[f'col_level_{_x}' for _x in range(num_of_levels)]
        )
        original = pd.Series(index=idx, data=100*np.random.random([len(idx)]).round(4))

        # Let this original series be logged in the simulation and get the parsed log;
        df_rtn = run_simulation_and_parse_log(series_to_log=original)

        # Confirm that the original can retrieved from the log using `unflatten_flattened_multi_index_in_logging`
        series_unflattened = unflatten_flattened_multi_index_in_logging(
            df_rtn.loc[pd.to_datetime(df_rtn.date) == sim_start_date].drop(columns=['date'])
        ).iloc[0]

        # Check equal
        pd.testing.assert_series_equal(original, series_unflattened.rename(None))


def test_get_root_path():
    """Check that `get_root_path` works as expected."""

    ROOT_PATH = Path(os.path.abspath(
        Path(os.path.dirname(__file__)) / '../'
    ))

    def is_correct_absolute_path(_path):
        return (ROOT_PATH == _path) and _path.is_absolute() and isinstance(_path, Path)

    assert is_correct_absolute_path(get_root_path())

    for test_dir in [
        os.path.abspath(Path(os.path.dirname(__file__)) / '../src/'),
        os.path.abspath(Path(os.path.dirname(__file__)) / '../resources/'),
        os.path.abspath(Path(os.path.dirname(__file__)) / '../tests/'),
        os.path.abspath(Path(os.path.dirname(__file__)) / '../'),
        os.path.abspath(Path(os.path.dirname(__file__))),
    ]:
        assert is_correct_absolute_path(get_root_path(test_dir)), f"Failed on {test_dir=}"
