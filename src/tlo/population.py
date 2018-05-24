"""The Person and Population classes."""

import pandas as pd


class Person:
    """An individual within the population.

    Useful attributes of a person:

    `population`
        The population this person is part of

    `sim`
        The Simulation instance controlling this population

    `props`
        A single row of the pandas DataFrame with population's properties giving
        property values just for this person. Most properties can also be accessed
        directly as properties of the person object, provided there is no name
        clash. So you can do, for instance, both `person.props['is_alive']` and
        `person.is_alive`.
    """

    # These are the explicit attributes, so we can check whether a name is a dynamic property
    __slots__ = ('index', 'population', 'sim')

    def __init__(self, population, index):
        """Create a new individual within a population.

        :param population: the Population this person is part of
        :param index: the index of this person with the population
        """
        self.population = population
        self.index = index
        self.sim = population.sim

    @property
    def props(self):
        """A view of this person's properties within the population.

        :returns: a view on the single row of the overall DataFrame.
        """
        # We pass the row index as a singleton list to force Pandas to give us a DataFrame output
        # Otherwise it would convert the row to a Series
        return self.population.props.loc[[self.index], :]

    def __getattr__(self, name):
        """Get the value of the given property of this individual.

        :param name: the name of the property to access
        """
        return self.population.props.at[self.index, name]

    def __setattr__(self, name, value):
        """Set the value of a property of this individual.

        :param name: the name of the property to access
        :param value: the new value
        """
        try:
            super().__setattr__(name, value)
        except AttributeError:
            if name in self.population.props.columns:
                self.population.props.at[self.index, name] = value
            else:
                raise


class Population:
    """A complete population of individuals.

    Useful properties of a population:

    `sim`
        The Simulation instance controlling this population

    `props`
        A pandas DataFrame with the properties of all individuals as columns

    `people`
        A list of Person objects representing the individuals in the population
    """

    __slots__ = ('people', 'props', 'sim')

    def __init__(self, sim, initial_size):
        """Create a new population.

        This will create the required Person objects and initialise their
        properties with 'empty' values. The simulation will then ask disease
        modules to fill in suitable starting values.

        :param sim: the Simulation containing this population
        :param initial_size: the initial population size
        """
        self.sim = sim
        # Create empty property arrays
        props = self.props = pd.DataFrame()
        for module in sim.modules.values():
            for prop_name, prop in module.PROPERTIES.items():
                props[prop_name] = prop.create_series(prop_name, initial_size)
        # Create Person objects to provide individual-based access
        self.people = []
        for i in range(initial_size):
            self.people.append(Person(self, i))

    def __len__(self):
        """:return: the size of the population."""
        return len(self.people)

    def __getitem__(self, key):
        """Get one or more people from the population."""
        return self.people[key]

    def __iter__(self):
        """Iterate over the people in a population."""
        return iter(self.people)

    def __getattr__(self, name):
        """Get the values of the given property over the population.

        :param name: the name of the property to access
        """
        return self.props.loc[:, name]

    def __setattr__(self, name, value):
        """Set the values of a property over the population.

        :param name: the name of the property to access
        :param value: the new values
        """
        try:
            super().__setattr__(name, value)
        except AttributeError:
            if name in self.props.columns:
                self.props.loc[:, name] = value
            else:
                raise
