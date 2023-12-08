from pathlib import Path

import numpy as np

from tlo import logging, Date, Population, Simulation
from tlo.methods.fullmodel import fullmodel

resource_file_path = Path(__file__).parents[1] / "resources"


def _check_nested_dict_equal(nested_dict_1: dict, nested_dict_2: dict) -> bool:
    for key, value in nested_dict_1.items():
        if key not in nested_dict_2:
            return False
        if isinstance(value, np.ndarray):
            if not np.all(value == nested_dict_2[key]):
                return False
        elif isinstance(value, dict):
            if not _check_nested_dict_equal(value, nested_dict_2[key]):
                return False
        elif value != nested_dict_2[key]:
            return False
    return True


def _check_random_state_equal(
    rng_1: np.random.RandomState, rng_2: np.random.RandomState
) -> bool:
    rng_state_1 = rng_1.get_state(legacy=False)
    rng_state_2 = rng_2.get_state(legacy=False)
    return _check_nested_dict_equal(rng_state_1, rng_state_2)


def _check_population_equal(population_1: Population, population_2: Population) -> bool:
    if population_1.initial_size != population_2.initial_size:
        return False
    if not population_1.new_row.equals(population_2.new_row):
        return False
    if not population_1.new_rows.equals(population_2.new_rows):
        return False
    if not population_1.next_person_id != population_2.next_person_id:
        return False
    if not population_1.props.equal(population_2.props):
        return False
    return True


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
    for attribute in ["start_date", "end_date", "_seed"]:
        assert getattr(simulation, attribute) == getattr(loaded_simulation, attribute)
    for module_name, module in simulation.modules.items():
        assert module_name in loaded_simulation.modules
        loaded_module = loaded_simulation.modules[module_name]
        assert loaded_module.PARAMETERS == module.PARAMETERS
        assert loaded_module.PROPERTIES == module.PROPERTIES
        assert _check_random_state_equal(module.rng, loaded_module.rng)
    assert _check_random_state_equal(simulation.rng, loaded_simulation.rng)
    assert len(simulation.event_queue) == len(loaded_simulation.event_queue)
    assert next(simulation.event_queue.counter) == next(
        loaded_simulation.event_queue.counter
    )
    for (*date_priority_count, event), (
        *loaded_date_priority_count,
        loaded_event,
    ) in zip(simulation.event_queue.queue, loaded_simulation.event_queue.queue):
        assert date_priority_count == loaded_date_priority_count
        if isinstance(event.target, Population):
            assert isinstance(loaded_event.target, Population)
            assert loaded_event.target is loaded_simulation.population
        else:
            assert event.target == loaded_event.target
        assert event.priority == loaded_event.priority
        assert event.module.name == loaded_event.module.name
    _check_population_equal(simulation.population, loaded_simulation.population)
