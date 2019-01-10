"""The Person and Population classes."""
import logging

import pandas as pd

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

    __slots__ = ('props', 'sim', 'initial_size', 'new_row')

    def __init__(self, sim, initial_size):
        """Create a new population.

        This will create the required Person objects and initialise their
        properties with 'empty' values. The simulation will then ask disease
        modules to fill in suitable starting values.

        :param sim: the Simulation containing this population
        :param initial_size: the initial population size
        """
        self.sim = sim
        self.initial_size = initial_size

        # Create empty property arrays
        self.props = self._create_props(initial_size)
        self.props.index.name = 'person'

        # keep a copy of a new row, so we can quickly append as population grows
        self.new_row = self.props[self.props.index == 0].copy()

    def _create_props(self, size):
        """Internal helper function to create a properties dataframe.

        :param size: the number of rows to create
        """
        props = pd.DataFrame()
        for module in self.sim.modules.values():
            for prop_name, prop in module.PROPERTIES.items():
                props[prop_name] = prop.create_series(prop_name, size)
        return props

    def do_birth(self):
        """Create a new person within the population.

        TODO: This will over-allocate capacity in the population dataframe for efficiency.

        :return: id of the new person
        """
        new_index = len(self.props)
        logger.debug('do_birth:%s', new_index)

        self.props = self.props.append(self.new_row.copy(), ignore_index=True, sort=False)
        self.props.index.name = 'person'

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
