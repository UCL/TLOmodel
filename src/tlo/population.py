"""The Person and Population classes."""
import math

import pandas as pd

from tlo import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Population:
    """A complete population of individuals.

    Useful properties of a population:

    `sim`
        The Simulation instance controlling this population.

    `props`
        A pandas DataFrame with the properties of all individuals as columns.
    """

    __slots__ = ('props', 'sim', 'initial_size', 'new_row', 'next_person_id', 'new_rows')

    def __init__(self, sim, initial_size: int, append_size: int = None):
        """Create a new population.

        This will create the required Person objects and initialise their
        properties with 'empty' values. The simulation will then ask disease
        modules to fill in suitable starting values.

        :param sim: the Simulation containing this population
        :param initial_size: the initial population size
        :param append_size: how many rows to append when growing the population dataframe (optional)
        """
        self.sim = sim
        self.initial_size = initial_size

        # Create empty property arrays
        self.props = self._create_props(initial_size)

        if append_size is None:
            # approximation based on runs to increase capacity of dataframe ~twice a year
            # TODO: profile adjustment of this and more clever calculation
            append_size = math.ceil(initial_size * 0.02)

        assert append_size > 0, "Number of rows to append when growing must be greater than 0"

        logger.info(key="info", data=f"Dataframe capacity append size: {append_size}")

        # keep a copy of a new rows to quickly append as population grows
        self.new_row = self.props.loc[[0]].copy()
        self.new_rows = self.props.loc[[0] * append_size].copy()

        # use the person_id of the next person to be added to the dataframe to increase capacity
        self.next_person_id = initial_size

    def _create_props(self, size):
        """Internal helper function to create a properties dataframe.

        :param size: the number of rows to create
        """
        props = pd.DataFrame(index=pd.RangeIndex(stop=size, name="person"))
        for module in self.sim.modules.values():
            for prop_name, prop in module.PROPERTIES.items():
                props[prop_name] = prop.create_series(prop_name, size)
        return props

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
            self.props = pd.concat((self.props, self.new_rows), ignore_index=True, sort=False)
            self.props.index.name = 'person'
            logger.info(key="info", data=f"Increased capacity of population dataframe to {len(self.props)}")

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
        prop = Property(type_, 'A test property')
        size = self.initial_size if self.props.empty else len(self.props)
        self.props[name] = prop.create_series(name, size)
