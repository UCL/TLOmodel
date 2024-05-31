import numpy as np
import pandas as pd
import pytest

from tlo.core import Property, Types
from tlo.population import Population


@pytest.fixture
def properties():
    return {
        f"{type_.name.lower()}_{i}": Property(type_, f"Column {i} of type {type_}")
        for type_ in [Types.INT, Types.BOOL, Types.REAL, Types.DATE, Types.BITSET]
        for i in range(5)
    }


@pytest.fixture(params=[1, 100, 1000])
def initial_size(request):
    return request.param


@pytest.fixture(params=[None, 0.02, 0.1])
def append_size(request, initial_size):
    return (
        request.param
        if request.param is None
        else max(int(initial_size * request.param), 1)
    )


@pytest.fixture
def population(properties, initial_size, append_size):
    return Population(properties, initial_size, append_size)


@pytest.fixture
def rng(seed):
    return np.random.RandomState(seed % 2**32)


def _generate_random_values(property, rng, size=None):
    if property.type_ == Types.DATE:
        return np.datetime64("2010-01-01") + rng.randint(0, 4000, size=size)
    elif property.type_ in (Types.INT, Types.BITSET):
        return rng.randint(low=0, high=100, size=size)
    elif property.type_ == Types.REAL:
        return rng.standard_normal(size=size)
    elif property.type_ == Types.BOOL:
        return rng.uniform(size=size) < 0.5
    else:
        msg = f"Unhandled type {property.type_}"
        raise ValueError(msg)


@pytest.fixture
def population_with_random_property_values(population, properties, initial_size, rng):

    for name, property in properties.items():
        population.props[name] = pd.Series(
            _generate_random_values(property, rng, initial_size),
            dtype=property.pandas_type,
        )

    return population


def test_population_invalid_append_size_raises(properties, initial_size):
    with pytest.raises(AssertionError, match="greater than 0"):
        Population(properties, initial_size, append_size=-1)


def test_population_attributes(population, properties, initial_size, append_size):
    assert population.initial_size == initial_size
    assert population.next_person_id == initial_size
    if append_size is not None:
        assert len(population.new_rows) == append_size
    else:
        assert 0 < len(population.new_rows) <= initial_size
    assert len(population.props.index) == initial_size
    assert len(population.props.columns) == len(properties)
    assert set(population.props.columns) == properties.keys()
    assert all(
        properties[name].pandas_type == col.dtype
        for name, col in population.props.items()
    )


def test_population_do_birth(population):
    initial_population_props_copy = population.props.copy()
    initial_size = population.initial_size
    append_size = len(population.new_rows)

    def check_population(population, birth_number):
        expected_next_person_id = initial_size + birth_number
        # population size should increase by append_size on first birth and after
        # every subsequent append_size births by a further append_size
        expected_size = (
            initial_size + ((birth_number - 1) // append_size + 1) * append_size
        )
        assert all(initial_population_props_copy.columns == population.props.columns)
        assert all(initial_population_props_copy.dtypes == population.props.dtypes)
        assert population.next_person_id == expected_next_person_id
        assert len(population.props.index) == expected_size

    for birth_number in range(1, append_size + 2):
        population.do_birth()
        check_population(population, birth_number)


def test_population_individual_properties_read_only_write_raises(
    population, properties
):
    individual_properties = population.individual_properties(
        person_id=0, read_only=True
    )
    for property_name in properties:
        with pytest.raises(ValueError, match="read-only"):
            individual_properties[property_name] = 0


@pytest.mark.parametrize("read_only", [True, False])
@pytest.mark.parametrize("person_id", [0, 1, -1])
def test_population_individual_properties_read(
    population_with_random_property_values, properties, rng, read_only, person_id
):
    person_id = person_id % population_with_random_property_values.initial_size
    population_dataframe = population_with_random_property_values.props
    individual_properties = (
        population_with_random_property_values.individual_properties(
            person_id=person_id, read_only=read_only
        )
    )
    for property_name in properties:
        assert (
            individual_properties[property_name]
            == population_dataframe.at[person_id, property_name]
        )
    # Try reading all properties (in a new random order) a second time to check any
    # caching mechanism is working as expected
    shuffled_property_names = list(list(properties.keys()))
    rng.shuffle(shuffled_property_names)
    for property_name in shuffled_property_names:
        assert (
            individual_properties[property_name]
            == population_dataframe.at[person_id, property_name]
        )


@pytest.mark.parametrize("person_id", [0, 1, -1])
def test_population_individual_properties_write_with_context_manager(
    population_with_random_property_values, properties, rng, person_id
):
    initial_population_dataframe = population_with_random_property_values.props.copy()
    person_id = person_id % population_with_random_property_values.initial_size
    updated_values = {}
    with population_with_random_property_values.individual_properties(
        person_id=person_id, read_only=False
    ) as individual_properties:
        for property_name, property in properties.items():
            updated_values[property_name] = _generate_random_values(property, rng)
            individual_properties[property_name] = updated_values[property_name]
    # Population dataframe should see updated properties for person_id row
    for property_name, property in properties.items():
        assert (
            population_with_random_property_values.props.at[person_id, property_name]
            == updated_values[property_name]
        )
    # All other rows in population dataframe should be unchanged
    all_rows_except_updated = ~initial_population_dataframe.index.isin([person_id])
    assert population_with_random_property_values.props[all_rows_except_updated].equals(
        initial_population_dataframe[all_rows_except_updated]
    )


@pytest.mark.parametrize("person_id", [0, 1, -1])
def test_population_individual_properties_write_with_sync(
    population_with_random_property_values, properties, rng, person_id
):
    initial_population_dataframe = population_with_random_property_values.props.copy()
    person_id = person_id % population_with_random_property_values.initial_size
    updated_values = {}
    individual_properties = (
        population_with_random_property_values.individual_properties(
            person_id=person_id, read_only=False
        )
    )
    for property_name, property in properties.items():
        updated_values[property_name] = _generate_random_values(property, rng)
        individual_properties[property_name] = updated_values[property_name]
    # Before syncrhonization all values in population dataframe should be unchanged
    assert initial_population_dataframe.equals(
        population_with_random_property_values.props
    )
    individual_properties.synchronize_updates_to_dataframe()
    # After synchronization all values in population dataframe should be updated values
    for property_name, property in properties.items():
        assert (
            population_with_random_property_values.props.at[person_id, property_name]
            == updated_values[property_name]
        )
