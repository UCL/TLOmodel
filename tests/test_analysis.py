import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import (
    flatten_multi_index_series_into_dict_for_logging,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    get_corase_appt_type,
    get_filtered_treatment_ids,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging, order_of_short_treatment_ids,
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


def test_corase_appt_type():
    """Check the function that maps each appt_types to a coarser definition."""
    appt_types = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv'
    )['Appt_Type'].values

    appts = pd.DataFrame({
            "original": pd.Series(appt_types),
            "coarse": pd.Series(appt_types).map(get_corase_appt_type)
    })

    assert not pd.isnull(appts).any().any()
    assert 17 == len(appts['coarse'].drop_duplicates())  # 17 coarse categories


def test_colormap_coarse_appts():
    """Check the function that allocates a unique colour to each coarse appointment type."""
    coarse_appt_types = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv'
    )['Appt_Type'].map(get_corase_appt_type).drop_duplicates().values

    colors = [get_color_coarse_appt(x) for x in coarse_appt_types]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt('????')  # Return `np.nan` if appt_type not recognised.

    # Check can produce plot:
    fig, ax = plt.subplots()
    for i, (_appt_type, _color) in enumerate(zip(coarse_appt_types, colors)):
        ax.bar(i, 10, color=_color, label=_appt_type)
    ax.legend(fontsize=10, ncol=2)
    ax.set_title('Colormap for Coarse Appointment Types')
    plt.close(fig)


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

    # Check can produce plot:
    fig, ax = plt.subplots()
    for i, (_short_treatment_id, _color) in enumerate(zip(short_treatment_ids, colors)):
        ax.bar(i, 10, color=_color, label=_short_treatment_id)
    ax.legend(fontsize=10, ncol=2)
    ax.set_title('Colormap for Short TREATMENT_IDs')
    plt.close(fig)
