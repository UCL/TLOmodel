from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from tlo import Date, DateOffset, Module, Population, Simulation, logging
from tlo.analysis.utils import merge_log_files, parse_log_file
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthsystem import HSI_Event, HSIEventQueueItem
from tlo.simulation import (
    EventQueue,
    SimulationNotInitialisedError,
    SimulationPreviouslyInitialisedError,
)


def _check_basic_simulation_attributes_equal(
    simulation_1: Simulation, simulation_2: Simulation
) -> None:
    for attribute in [
        "start_date",
        "end_date",
        "date",
        "show_progress_bar",
        "_custom_log_levels",
        "_seed",
        "_initialised",
    ]:
        assert getattr(simulation_1, attribute) == getattr(simulation_2, attribute)


def _nested_dict_are_equal(nested_dict_1: dict, nested_dict_2: dict) -> bool:
    for key, value in nested_dict_1.items():
        if key not in nested_dict_2:
            return False
        if isinstance(value, np.ndarray):
            if not np.all(value == nested_dict_2[key]):
                return False
        elif isinstance(value, dict):
            if not _nested_dict_are_equal(value, nested_dict_2[key]):
                return False
        elif value != nested_dict_2[key]:
            return False
    return True


def _check_random_state_equal(
    rng_1: np.random.RandomState, rng_2: np.random.RandomState
) -> None:
    rng_state_1 = rng_1.get_state(legacy=False)
    rng_state_2 = rng_2.get_state(legacy=False)
    assert _nested_dict_are_equal(rng_state_1, rng_state_2)


def _check_population_equal(population_1: Population, population_2: Population) -> None:
    assert population_1.initial_size == population_2.initial_size
    assert population_1.new_row.equals(population_2.new_row)
    assert population_1.new_rows.equals(population_2.new_rows)
    assert population_1.next_person_id == population_2.next_person_id
    assert population_1.props.equals(population_2.props)


def _check_modules_are_equal(
    modules_dict_1: Dict[str, Module], modules_dict_2: Dict[str, Module]
) -> None:
    for module_name, module_1 in modules_dict_1.items():
        assert module_name in modules_dict_2
        module_2 = modules_dict_2[module_name]
        assert module_2.PARAMETERS == module_1.PARAMETERS
        assert module_2.PROPERTIES == module_1.PROPERTIES
        _check_random_state_equal(module_1.rng, module_2.rng)


def _check_event_queues_are_equal(
    event_queue_1: EventQueue, event_queue_2: EventQueue
) -> None:
    assert len(event_queue_1) == len(event_queue_2)
    for (*date_priority_count_1, event_1), (*date_priority_count_2, event_2) in zip(
        event_queue_1.queue, event_queue_2.queue
    ):
        assert date_priority_count_1 == date_priority_count_2
        if isinstance(event_1.target, Population):
            # We don't check for equality of populations here as we do separately and
            # it would create a lot of redundancy to check for every event
            assert isinstance(event_2.target, Population)
        else:
            assert event_1.target == event_2.target
        assert event_1.priority == event_1.priority
        assert type(event_1.module) is type(event_2.module)  # noqa: E721


def _check_hsi_events_are_equal(hsi_event_1: HSI_Event, hsi_event_2: HSI_Event) -> None:
    if isinstance(hsi_event_1.target, Population):
        # We don't check for equality of populations here as we do separately and
        # it would create a lot of redundancy to check for every HSI event
        assert isinstance(hsi_event_2.target, Population)
    else:
        assert hsi_event_1.target == hsi_event_2.target
    assert hsi_event_1.module.name == hsi_event_2.module.name
    assert hsi_event_1.TREATMENT_ID == hsi_event_2.TREATMENT_ID
    assert hsi_event_1.ACCEPTED_FACILITY_LEVEL == hsi_event_2.ACCEPTED_FACILITY_LEVEL
    assert hsi_event_1.BEDDAYS_FOOTPRINT == hsi_event_2.BEDDAYS_FOOTPRINT
    assert (
        hsi_event_1._received_info_about_bed_days
        == hsi_event_2._received_info_about_bed_days
    )
    assert hsi_event_1.expected_time_requests == hsi_event_2.expected_time_requests
    assert hsi_event_1.facility_info == hsi_event_2.facility_info


def _check_hsi_event_queues_are_equal(
    hsi_event_queue_1: List[HSIEventQueueItem],
    hsi_event_queue_2: List[HSIEventQueueItem],
) -> None:
    assert len(hsi_event_queue_1) == len(hsi_event_queue_2)
    for hsi_event_queue_item_1, hsi_event_queue_item_2 in zip(
        hsi_event_queue_1, hsi_event_queue_2
    ):
        assert hsi_event_queue_item_1.priority == hsi_event_queue_item_2.priority
        assert hsi_event_queue_item_1.topen == hsi_event_queue_item_2.topen
        assert (
            hsi_event_queue_item_1.rand_queue_counter
            == hsi_event_queue_item_2.rand_queue_counter
        )
        assert hsi_event_queue_item_1.tclose == hsi_event_queue_item_2.tclose
        _check_hsi_events_are_equal(
            hsi_event_queue_item_1.hsi_event, hsi_event_queue_item_2.hsi_event
        )


def _check_simulations_are_equal(
    simulation_1: Simulation, simulation_2: Simulation
) -> None:
    _check_basic_simulation_attributes_equal(simulation_1, simulation_2)
    _check_modules_are_equal(simulation_1.modules, simulation_2.modules)
    _check_random_state_equal(simulation_1.rng, simulation_2.rng)
    _check_event_queues_are_equal(simulation_1.event_queue, simulation_2.event_queue)
    _check_hsi_event_queues_are_equal(
        simulation_1.modules["HealthSystem"].HSI_EVENT_QUEUE,
        simulation_2.modules["HealthSystem"].HSI_EVENT_QUEUE,
    )
    _check_population_equal(simulation_1.population, simulation_2.population)


@pytest.fixture(scope="module")
def resource_file_path():
    return Path(__file__).parents[1] / "resources"


@pytest.fixture(scope="module")
def initial_population_size():
    return 5000


@pytest.fixture(scope="module")
def start_date():
    return Date(2010, 1, 1)


@pytest.fixture(scope="module")
def end_date(start_date):
    return start_date + DateOffset(days=180)


@pytest.fixture(scope="module")
def intermediate_date(start_date, end_date):
    return start_date + (end_date - start_date) / 2


@pytest.fixture(scope="module")
def logging_custom_levels():
    return {"*": logging.INFO}


def _simulation_factory(
    output_directory, start_date, seed, resource_file_path, logging_custom_levels
):
    log_config = {
        "filename": "test",
        "directory": output_directory,
        "custom_levels": logging_custom_levels,
    }
    simulation = Simulation(
        start_date=start_date,
        seed=seed,
        log_config=log_config,
        resourcefilepath=resource_file_path
    )
    simulation.register(
        *fullmodel()
    )
    return simulation


@pytest.fixture
def simulation(tmp_path, start_date, seed, resource_file_path, logging_custom_levels):
    return _simulation_factory(
        tmp_path, start_date, seed, resource_file_path, logging_custom_levels
    )


@pytest.fixture(scope="module")
def simulated_simulation(
    tmp_path_factory,
    start_date,
    end_date,
    seed,
    resource_file_path,
    initial_population_size,
    logging_custom_levels,
):
    tmp_path = tmp_path_factory.mktemp("simulated_simulation")
    simulation = _simulation_factory(
        tmp_path, start_date, seed, resource_file_path, logging_custom_levels
    )
    simulation.make_initial_population(n=initial_population_size)
    simulation.simulate(end_date=end_date)
    return simulation


def test_save_to_pickle_creates_file(tmp_path, simulation):
    pickle_path = tmp_path / "simulation.pkl"
    simulation.save_to_pickle(pickle_path=pickle_path)
    assert pickle_path.exists()


def test_save_load_pickle_after_initialising(
    tmp_path, simulation, initial_population_size
):
    simulation.make_initial_population(n=initial_population_size)
    simulation.initialise(end_date=simulation.start_date)
    pickle_path = tmp_path / "simulation.pkl"
    simulation.save_to_pickle(pickle_path=pickle_path)
    loaded_simulation = Simulation.load_from_pickle(pickle_path)
    _check_simulations_are_equal(simulation, loaded_simulation)


def test_save_load_pickle_after_simulating(tmp_path, simulated_simulation):
    pickle_path = tmp_path / "simulation.pkl"
    simulated_simulation.save_to_pickle(pickle_path=pickle_path)
    loaded_simulation = Simulation.load_from_pickle(pickle_path)
    _check_simulations_are_equal(simulated_simulation, loaded_simulation)


def _check_parsed_logs_are_equal(
    log_path_1: Path,
    log_path_2: Path,
    module_name_key_pairs_to_skip: set[tuple[str, str]],
) -> None:
    logs_dict_1 = parse_log_file(log_path_1)
    logs_dict_2 = parse_log_file(log_path_2)
    assert logs_dict_1.keys() == logs_dict_2.keys()
    for module_name in logs_dict_1.keys():
        module_logs_1 = logs_dict_1[module_name]
        module_logs_2 = logs_dict_2[module_name]
        assert module_logs_1.keys() == module_logs_2.keys()
        for key in module_logs_1:
            if key == "_metadata":
                assert module_logs_1[key] == module_logs_2[key]
            elif (module_name, key) not in module_name_key_pairs_to_skip:
                assert module_logs_1[key].equals(module_logs_2[key])


@pytest.mark.slow
def test_continuous_and_interrupted_simulations_equal(
    tmp_path,
    simulation,
    simulated_simulation,
    initial_population_size,
    intermediate_date,
    end_date,
    logging_custom_levels,
):
    simulation.make_initial_population(n=initial_population_size)
    simulation.initialise(end_date=end_date)
    simulation.run_simulation_to(to_date=intermediate_date)
    pickle_path = tmp_path / "simulation.pkl"
    simulation.save_to_pickle(pickle_path=pickle_path)
    simulation.close_output_file()
    log_config = {
        "filename": "test_continued",
        "directory": tmp_path,
        "custom_levels": logging_custom_levels,
    }
    interrupted_simulation = Simulation.load_from_pickle(pickle_path, log_config)
    interrupted_simulation.run_simulation_to(to_date=end_date)
    interrupted_simulation.finalise()
    _check_simulations_are_equal(simulated_simulation, interrupted_simulation)
    merged_log_path = tmp_path / "concatenated.log"
    merge_log_files(
        simulation.log_filepath, interrupted_simulation.log_filepath, merged_log_path
    )
    _check_parsed_logs_are_equal(
        simulated_simulation.log_filepath, merged_log_path, {("tlo.simulation", "info")}
    )


def test_run_simulation_to_past_end_date_raises(
    simulation, initial_population_size, end_date
):
    simulation.make_initial_population(n=initial_population_size)
    simulation.initialise(end_date=end_date)
    with pytest.raises(ValueError, match="after simulation end date"):
        simulation.run_simulation_to(to_date=end_date + DateOffset(days=1))


def test_run_simulation_without_initialisation_raises(
    simulation, initial_population_size, end_date
):
    simulation.make_initial_population(n=initial_population_size)
    with pytest.raises(SimulationNotInitialisedError):
        simulation.run_simulation_to(to_date=end_date)


def test_initialise_simulation_twice_raises(
    simulation, initial_population_size, end_date
):
    simulation.make_initial_population(n=initial_population_size)
    simulation.initialise(end_date=end_date)
    with pytest.raises(SimulationPreviouslyInitialisedError):
        simulation.initialise(end_date=end_date)

def test_resourcefilepath_is_set_correctly(simulation, resource_file_path):
    assert simulation.resourcefilepath == resource_file_path
