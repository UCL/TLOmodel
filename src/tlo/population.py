"""Types for representing a properties of a population of individuals."""

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, Optional, Set

import pandas as pd

from tlo import Property, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IndividualProperties:
    """Memoized view of population dataframe row that is optionally read-only.

    This class should not be instantiated directly but instead the
    :py:meth:`Population.individual_properties` context manager method used to create
    instances for a given population.
    """

    def __init__(
        self, population_dataframe: pd.DataFrame, person_id: int, read_only: bool = True
    ):
        self._finalized = False
        self._read_only = read_only
        self._property_cache: Dict[str, Any] = {}
        # Avoid storing a reference to population_dataframe internally by mediating
        # access via closures to guard against direct access
        self._get_value_at = lambda key: population_dataframe.at[person_id, key]
        if not read_only:
            self._properties_updated: Set[str] = set()

            def synchronize_updates_to_dataframe():
                row_index = population_dataframe.index.get_loc(person_id)
                for key in self._properties_updated:
                    # This chained indexing approach to setting dataframe values is
                    # significantly (~3 to 4 times) quicker than using at / iat
                    # indexers, but will fail when copy-on-write is enabled which will
                    # be default in Pandas 3
                    column = population_dataframe[key]
                    column.values[row_index] = self._property_cache[key]

            self._synchronize_updates_to_dataframe = synchronize_updates_to_dataframe

    def __getitem__(self, key: str) -> Any:
        if self._finalized:
            msg = f"Cannot read value for {key} as instance has been finalized"
            raise ValueError(msg)
        try:
            return self._property_cache[key]
        except KeyError:
            value = self._get_value_at(key)
            self._property_cache[key] = value
            return value

    def __setitem__(self, key: str, value: Any) -> None:
        if self._finalized:
            msg = f"Cannot set value for {key} as instance has been finalized"
            raise ValueError(msg)
        if self._read_only:
            msg = f"Cannot set value for {key} as destination is read-only"
            raise ValueError(msg)
        self._properties_updated.add(key)
        self._property_cache[key] = value

    def synchronize_updates_to_dataframe(self) -> None:
        """Synchronize values for any updated properties to population dataframe."""
        if not self._read_only:
            self._synchronize_updates_to_dataframe()
            self._properties_updated.clear()

    def finalize(self) -> None:
        """Synchronize updates to population dataframe and prevent further access."""
        self.synchronize_updates_to_dataframe()
        self._finalized = True


class Population:
    """A complete population of individuals.

    Useful properties of a population:

    `props`
        A Pandas DataFrame with the properties of all individuals as columns.
    """

    __slots__ = (
        "props",
        "initial_size",
        "new_row",
        "next_person_id",
        "new_rows",
    )

    def __init__(
        self,
        properties: Dict[str, Property],
        initial_size: int,
        append_size: Optional[int] = None,
    ):
        """Create a new population.

        This will create the required the population dataframe and initialise
        individual's properties as dataframe columns with 'empty' values. The simulation
        will then call disease modules to fill in suitable starting values.

        :param properties: Dictionary defining properties (columns) to initialise
            population dataframe with, keyed by property name and with values
            :py:class:`Property` instances defining the property type.
        :param initial_size: The initial population size.
        :param append_size: How many rows to append when growing the population
            dataframe (optional).
        """
        self.initial_size = initial_size

        # Create empty property arrays
        self.props = self._create_props(initial_size, properties)

        if append_size is None:
            # approximation based on runs to increase capacity of dataframe ~twice a year
            # TODO: profile adjustment of this and more clever calculation
            append_size = math.ceil(initial_size * 0.02)

        assert (
            append_size > 0
        ), "Number of rows to append when growing must be greater than 0"

        logger.info(key="info", data=f"Dataframe capacity append size: {append_size}")

        # keep a copy of a new, empty, row to quickly append as population grows
        self.new_row = self.props.loc[[0]].copy()
        self.new_rows = self.props.loc[[0] * append_size].copy()

        # use the person_id of the next person to be added to the dataframe to increase capacity
        self.next_person_id = initial_size

    def _create_props(self, size: int, properties: Dict[str, Property]) -> pd.DataFrame:
        """Internal helper function to create a properties dataframe.

        :param size: the number of rows to create
        """
        return pd.DataFrame(
            data={
                property_name: property.create_series(property_name, size)
                for property_name, property in properties.items()
            },
            index=pd.RangeIndex(stop=size, name="person"),
        )

    def do_birth(self):
        """Create a new person within the population.

        TODO: This will over-allocate capacity in the population dataframe for efficiency.

        :return: id of the new person
        """
        # get index of the last row
        index_of_last_row = self.props.index[-1]

        # the index of the next person
        if self.next_person_id > index_of_last_row:
            # we need to add some rows
            self.props = pd.concat(
                (self.props, self.new_rows), ignore_index=True, sort=False
            )
            self.props.index.name = "person"
            logger.info(
                key="info",
                data=f"Increased capacity of population dataframe to {len(self.props)}",
            )

        new_index = self.next_person_id
        self.next_person_id += 1

        return new_index

    def make_test_property(self, name, type_):
        """Create a new property for test purposes.

        When testing a particular method in isolation, it's helpful to be able to define
        the properties it reads that would normally be provided by other methods. That is
        what this is for. It adds an extra column into the property DataFrame for this
        population, set up with the appropriate type.

        This should only be used in tests, not in your actual module code!

        :param name: the name of the property to add
        :param type_: a member of the :py:class:`Types` enumeration giving the type of
            the property
        """
        from tlo import Property

        prop = Property(type_, "A test property")
        size = self.initial_size if self.props.empty else len(self.props)
        self.props[name] = prop.create_series(name, size)

    @contextmanager
    def individual_properties(
        self, person_id: int, read_only: bool = True
    ) -> Generator[IndividualProperties, None, None]:
        """
        Context manager for a memoized view of a row of the population dataframe.

        The view returned represents the properties of an individual with properties
        accessible by indexing using string column names, and lazily read-on demand
        from the population dataframe.

        Optionally the view returned may allow updating properties as well as reading.
        In this case on exit from the ``with`` block in which the context is entered,
        any updates to the individual properties will be written back to the population
        dataframe.
        
        Once the ``with`` block in which the context is entered has been exited the view
        returned will raise an error on any subsequent attempts at reading or writing
        properties.

        :param person_id: Row index of the dataframe row to extract.
        :param read_only: Whether view is read-only or allows updating properties. If
            ``True`` :py:meth:`IndividualProperties.synchronize_updates_to_dataframe`
            method needs to be called for any updates to be written back to population
            dataframe.
        :returns: Object allowing memoized access to an individual's properties.
        """
        properties = IndividualProperties(self.props, person_id, read_only=read_only)
        yield properties
        properties.finalize()
