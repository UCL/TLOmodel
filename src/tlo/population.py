"""The Person and Population classes."""
from functools import lru_cache

import numpy as np
import pandas as pd


class Person:
    """An individual within the population.

    Useful attributes of a person:

    `population`
        The population this person is part of.

    `index`
        The index of this person within the population.

    `sim`
        The Simulation instance controlling this population.

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
        :param index: the index of this person within the population
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

    def __str__(self):
        """Return a human-readable summary of this person."""
        return '<Person {}>'.format(self.index)

    __repr__ = __str__

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
        The Simulation instance controlling this population.

    `props`
        A pandas DataFrame with the properties of all individuals as columns.

    `people`
        A list of Person objects representing the individuals in the population.
    """

    __slots__ = ('people', 'props', 'sim', 'age_ranges')

    MINIMUM_AGE = 0
    MAXIMUM_AGE = 120
    AGE_RANGE_SIZE = 5
    AGE_RANGE_LIMIT = 100

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
        self.props = self._create_props(initial_size)
        self.props.index.name = 'person'
        # Create Person objects to provide individual-based access
        self.people = []
        for i in range(initial_size):
            self.people.append(Person(self, i))

        self.age_ranges = Population.make_age_range_lookup()

    @staticmethod
    def make_age_range_lookup():
        # Create a mapping from age (in years) to age range
        # i.e. { 0: '0-4', 1: '0-4', ..., 119: '115-119', 120: '120-120' }
        def chunks(items, n):
            """Takes a list and divides it into parts of size n"""
            for index in range(0, len(items), n):
                yield items[index:index + n]

        # split all the ages from min to limit (100 years) into 5 year ranges
        parts = chunks(range(Population.MINIMUM_AGE, Population.AGE_RANGE_LIMIT),
                       Population.AGE_RANGE_SIZE)
        age_ranges = {}
        # loop over each range and map all ages falling within the range to the range
        for part in parts:
            start = part.start
            end = part.stop - 1
            value = '%s-%s' % (start, end)
            for i in range(start, part.stop):
                age_ranges[i] = value

        # all ages over the limit (100 years) get the same age range category
        for i in range(Population.AGE_RANGE_LIMIT, Population.MAXIMUM_AGE + 1):
            age_ranges[i] = '%s+' % Population.AGE_RANGE_LIMIT

        return age_ranges

    def __len__(self):
        """:return: the size of the population."""
        return len(self.people)

    def __getitem__(self, key):
        """Get a person or set of properties from the population.

        What is returned depends on the type of key looked up:

        * ``int``: a single :py:class:`Person` object is returned, e.g. ``pop[2]``
        * ``str``: a Series is returned giving the value of a single named property
          for the whole population, e.g. ``pop['is_alive']``
        * ``slice``: a DataFrame is returned giving the values of all properties for
          the given range of people, e.g. ``pop[1:3]``
        * otherwise (e.g. ``tuple``): the key is passed to ``props.loc``, to extract a
          sub-frame of the properties DataFrame, e.g. ``pop[1:2, 'is_alive']``

        Note that due to the way Pandas labelled indexing works, slices here are
        *inclusive* of the end point, unlike indexing Python lists. So ``pop[0:2]``
        will return properties for *3* people: those at positions 0, 1 and 2.

        :param key: the item(s) to look up in the population
        """
        if isinstance(key, int):
            return self.people[key]
        elif isinstance(key, str):
            return self.props.loc[:, key]
        elif isinstance(key, slice):
            return self.props.loc[key, :]
        else:
            return self.props.loc[key]

    def __setitem__(self, key, values):
        """Set properties for people in the population.

        This provides transparent label-based access to change property values in bulk,
        for improved performance. For example:

        * ``pop[:, ('is_alive', 'is_pregnant')] = False``
        * ``pop[bool_array, 'is_depressed'] = True``

        :param key: index(es) for the properties to set, as for pandas.DataFrame.loc
        :param values: the value(s) to set
        """
        self.props.loc[key] = values

    def __iter__(self):
        """Iterate over the people in a population."""
        return iter(self.people)

    def __getattr__(self, name):
        """Get the values of the given property over the population.

        :param name: the name of the property to access
        """
        # TODO: If over-allocating make sure to change the end index here and in __setattr__!
        if name in self.props.columns:
            return self.props.loc[:, name]
        else:
            return getattr(self.props, name)

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

    def _create_props(self, size):
        """Internal helper function to create a properties dataframe.

        :param size: the number of rows to create
        """
        props = pd.DataFrame()
        for module in self.sim.modules.values():
            for prop_name, prop in module.PROPERTIES.items():
                props[prop_name] = prop.create_series(prop_name, size)
        return props

    @lru_cache(maxsize=1)
    def __get_age(self, timestamp):
        """
        Creates a dataframe holding age information. This private method is decorated with an LRU
        cache of size 1. This means the age dataframe for the most recent timestamp is cached.

        Module/events can repeatedly access the population.age property as the simulation runs
        forward in time, the age for the current timestemp will only be calculated once.

        See the 'age()' method.

        :param timestamp: a numpy datetime
        :return: Pandas dataframe with age of individuals in population
        """
        age = pd.DataFrame({'days': timestamp - self.sim.population.props.date_of_birth})
        age.index.name = 'person'
        age['years_exact'] = age.days / np.timedelta64(1, 'Y')
        age['years'] = age.years_exact.astype(int)
        age['age_range'] = age.years.map(self.age_ranges).astype('category')
        return age

    @property
    def age(self):
        """
        Returns age of individuals based on the current simulation date. The dataframe returned
        has columns 'days', 'years_exact' and 'years' (whole years).

        The index of the age dataframe is the same as the population dataframe.

        :return: A Pandas dataframe of age information of the population
        """
        return self.__get_age(self.sim.date)

    def do_birth(self):
        """Create a new person within the population.

        TODO: This will over-allocate capacity in the population dataframe for efficiency.

        :return: the new person
        """
        new_index = len(self)
        extra_props = self._create_props(1)
        self.props = self.props.append(extra_props, ignore_index=True, sort=False)
        new_person = Person(self, new_index)
        self.people.append(new_person)
        return new_person

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
        self.props[name] = prop.create_series(name, len(self))
