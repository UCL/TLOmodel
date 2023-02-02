import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Property, Simulation, Types, logging
from tlo.analysis.utils import (
    colors_in_matplotlib,
    flatten_multi_index_series_into_dict_for_logging,
    get_coarse_appt_type,
    get_color_cause_of_death_or_daly_label,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    get_filtered_treatment_ids,
    get_root_path,
    order_of_coarse_appt,
    order_of_short_treatment_ids,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def test_parse_log():
    log_file = Path(__file__).parent / 'resources' / 'structured_log.txt'

    output = parse_log_file(log_file)

    assert 'tlo.methods.epilepsy' in output
    assert set(output['tlo.methods.epilepsy'].keys()) == {'incidence_epilepsy', 'epilepsy_logging', '_metadata'}


def test_parse_log_levels(tmpdir):
    # setup a toy simulation to test logging
    logger = logging.getLogger('tlo.methods.dummy')
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 1)
    pop_size = 100

    class DummyEvent(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger.info(key="info_level1", data={'number': np.random.randint(0, 100)})
            logger.info(key="info_level2", data={'number': np.random.randint(0, 100)})
            logger.debug(key="debug_level", data={'number': np.random.randint(0, 100)})

    class Dummy(Module):
        PROPERTIES = {'dummy': Property(Types.INT, description='dummy')}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def on_birth(self, mother, child):
            pass

        def initialise_simulation(self, sim: Simulation):
            sim.schedule_event(DummyEvent(self, frequency=DateOffset(months=1)), start_date)

    # test parsing when log level is INFO
    sim = Simulation(start_date=start_date, log_config={'filename': 'temp', 'directory': tmpdir})
    sim.register(Dummy())
    logger.setLevel(logging.INFO)
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath)

    # At INFO level
    assert len(output['tlo.methods.dummy']['_metadata']['tlo.methods.dummy']) == 2  # should have two tables

    # tables should be at level INFO
    for k, v in output['tlo.methods.dummy']['_metadata']['tlo.methods.dummy'].items():
        assert v['level'] == 'INFO'

    # test parsing when log level is DEBUG
    sim = Simulation(start_date=start_date, log_config={'filename': 'temp2', 'directory': tmpdir})
    sim.register(Dummy())
    logger.setLevel(logging.DEBUG)
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath, level=logging.INFO)  # we're parsing everything above INFO level

    # logged DEBUG but parsed at INFO levels
    assert len(output['tlo.methods.dummy']['_metadata']['tlo.methods.dummy']) == 2
    assert 'debug_level' not in output['tlo.methods.dummy']

    # logged DEBUG and parsed at DEBUG level
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    assert len(output['tlo.methods.dummy']['_metadata']['tlo.methods.dummy']) == 3
    assert 'debug_level' in output['tlo.methods.dummy']


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


def test_coarse_appt_type():
    """Check the function that maps each appt_types to a coarser definition."""
    appt_types = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv'
    )['Appt_Type_Code'].values

    appts = pd.DataFrame({
            "original": pd.Series(appt_types),
            "coarse": pd.Series(appt_types).map(get_coarse_appt_type)
    })

    coarse_appts = appts['coarse'].drop_duplicates()

    assert not pd.isnull(appts).any().any()
    assert 13 == len(coarse_appts)  # 12 coarse categories

    # Check can run sorting on these
    assert 13 == len(sorted(coarse_appts, key=order_of_coarse_appt))


def test_colormap_coarse_appts():
    """Check the function that allocates a unique colour to each coarse appointment type."""
    coarse_appt_types = pd.read_csv(
            resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv'
        )['Appt_Type_Code'].map(get_coarse_appt_type).drop_duplicates().values

    coarse_appt_types = sorted(coarse_appt_types, key=order_of_coarse_appt)

    colors = [get_color_coarse_appt(x) for x in coarse_appt_types]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt('????')  # Return `np.nan` if appt_type not recognised.
    assert all(map(lambda x: x in colors_in_matplotlib(), colors))  # All colors recognised


def test_get_treatment_ids(tmpdir):
    """Check the function that generates the list of TREATMENT_IDs defined in the model."""

    x = get_filtered_treatment_ids()  # All TREATMENT_IDs
    y = get_filtered_treatment_ids(depth=1)  # TREATMENT_IDs to the first level of depth (i.e. module level)

    assert isinstance(x, list)
    assert all([isinstance(_x, str) for _x in x])

    assert isinstance(y, list)
    assert all([isinstance(_y, str) for _y in y])

    assert len(y) < len(x)


def test_colormap_short_treatment_id():
    """Check the function that allocates a unique colour to each shortened TREATMENT_ID (i.e. each module)"""

    short_treatment_ids = sorted(get_filtered_treatment_ids(depth=1), key=order_of_short_treatment_ids)
    colors = [get_color_short_treatment_id(x) for x in short_treatment_ids]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt('????')  # Return `np.nan` if appt_type not recognised.
    assert all(map(lambda x: x in colors_in_matplotlib(), colors))  # All colors recognised


def test_colormap_cause_of_death_label(seed):
    """Check that all the Cause-of-Deaths labels defined in the full model are assigned to a unique colour when
     plotting."""

    def get_all_cause_of_death_labels(seed=0) -> List[str]:
        """Return list of all the causes of death defined in the full model."""
        start_date = Date(2010, 1, 1)
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(*fullmodel(resourcefilepath=resourcefilepath, use_simplified_births=False))
        sim.make_initial_population(n=1_000)
        sim.simulate(end_date=start_date)
        mapper, _ = (sim.modules['Demography']).create_mappers_from_causes_of_death_to_label()
        return sorted(set(mapper.values()))

    all_labels = get_all_cause_of_death_labels(seed)

    colors = [get_color_cause_of_death_or_daly_label(_label) for _label in all_labels]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt('????')  # Return `np.nan` if label is not recognised.
    assert all(map(lambda x: x in colors_in_matplotlib(), colors))  # All colors recognised
