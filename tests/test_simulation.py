from pathlib import Path
from typing import Dict, List

import numpy as np

from tlo import logging, Date, Module, Population, Simulation
from tlo.simulation import EventQueue
from tlo.methods.healthsystem import HSI_Event, HSIEventQueueItem
from tlo.methods.fullmodel import fullmodel

resource_file_path = Path(__file__).parents[1] / "resources"


def _check_basic_simulation_attributes_equal(
    simulation_1: Simulation, simulation_2: Simulation
) -> None:
    for attribute in [
        "start_date",
        "end_date",
        "date",
        "show_progress_bar",
        "_custom_log_levels",
        "_log_filepath",
        "_seed",
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
    assert next(event_queue_1.counter) == next(event_queue_2.counter)
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
        assert event_1.module.name == event_2.module.name


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


def test_save_pickle(tmp_path, seed):
    start_date = Date(2010, 1, 1)
    to_date = Date(2010, 2, 1)
    end_date = Date(2010, 3, 1)
    log_config = {
        "filename": "test.log",
        "directory": tmp_path,
        "custom_levels": {"*": logging.INFO},
    }
    simulation = Simulation(
        start_date=start_date,
        seed=seed,
        log_config=log_config,
    )
    simulation.register(
        *fullmodel(
            resourcefilepath=resource_file_path,
            use_simplified_births=True,
        )
    )
    simulation.make_initial_population(n=1000)
    simulation.initialise(end_date=end_date)
    simulation.run_simulation_to(to_date=to_date)
    pickle_path = tmp_path / "simulation.pkl"
    simulation.save_to_pickle(pickle_path=pickle_path)
    loaded_simulation = Simulation.load_from_pickle(pickle_path)
    _check_simulations_are_equal(simulation, loaded_simulation)
