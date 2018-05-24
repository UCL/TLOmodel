"""Core framework classes.

This contains things that didn't obviously go in their own file, such as
specification for parameters and properties, and the base Module class for
disease modules.
"""

from enum import Enum, auto

import numpy as np
import pandas as pd


class Types(Enum):
    """Possible types for parameters and properties.

    This lets us hide the details of numpy & pandas dtype strings and give
    users an easy list to reference instead.

    Most of these should be intuitive. The DATE type can actually represent
    date+time values, but we are only concerned with day precision at present.
    The CATEGORICAL type is useful for things like sex where there are a fixed
    number of options to choose from. The LIST type is used for properties
    where the value is a collection, e.g. the set of children of a person.
    """
    DATE = auto()
    BOOL = auto()
    INT = auto()
    REAL = auto()
    CATEGORICAL = auto()
    LIST = auto()


class Specifiable:
    """Base class for Parameter and Property."""

    """Map our Types to pandas dtype specifications."""
    PANDAS_TYPE_MAP = {
        Types.DATE: 'datetime64[ns]',
        Types.BOOL: bool,
        Types.INT: int,
        Types.REAL: float,
        Types.CATEGORICAL: 'category',
        Types.LIST: object,
    }

    """Map our Types to Python types."""
    PYTHON_TYPE_MAP = {
        Types.DATE: pd.Timestamp,
        Types.BOOL: bool,
        Types.INT: int,
        Types.REAL: float,
        Types.CATEGORICAL: str,
        Types.LIST: list,
    }

    def __init__(self, type_, description):
        """Create a new Specifiable.

        :param type_: an instance of Types giving the type of allowed values
        :param description: textual description of what this Specifiable represents
        """
        assert type_ in Types
        self.type_ = type_
        self.description = description

    @property
    def python_type(self):
        return self.PYTHON_TYPE_MAP[self.type_]

    @property
    def pandas_type(self):
        return self.PANDAS_TYPE_MAP[self.type_]


class Parameter(Specifiable):
    """Used to specify parameters for disease modules etc."""


class Property(Specifiable):
    """Used to specify properties of individuals."""

    def __init__(self, type_, description, *, optional=False):
        """Create a new property specification.

        :param type_: an instance of Types giving the type of allowed values of this property
        :param description: textual description of what this property represents
        :param optional: whether a value needs to be given for this property
        """
        super().__init__(type_, description)
        self.optional = optional


class Module:
    """The base class for disease modules.

    This declares the methods which individual modules must implement, and contains the
    core functionality linking modules into a simulation. Useful properties available
    on instances are:

    `name`:
        The unique name of this module within the simulation.

    `parameters`
        A dictionary of module parameters, derived from specifications in the PARAMETERS
        class attribute on a subclass. These parameters are also available as object
        attributes in their own right(where their names do not clash) so you can do both
        `m.parameters['name']` and `m.name`.

    `rng`
        A random number generator specific to this module, with its own internal state.
        It's an instance of `numpy.random.RandomState`; see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
    """

    # Subclasses may declare this dictionary to specify module-level parameters.
    # We give an empty definition here as default.
    PARAMETERS = {}

    # Subclasses may declare this dictionary to specify properties of individuals.
    # We give an empty definition here as default.
    PROPERTIES = {}

    # The explicit attributes of the module. We list these so we can distinguish dynamic
    # parameters created from the PARAMETERS specification.
    __slots__ = ('name', 'parameters', 'rng')

    def __init__(self, name=None):
        """Construct a new disease module ready to be included in a simulation.

        Initialises an empty parameters dictionary and module-specific random number
        generator.

        :param name: the name to use for this module. Defaults to the concrete subclass' name.
        """
        self.parameters = {}
        self.rng = np.random.RandomState()
        self.name = name or self.__class__.__name__

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Must be implemented by subclasses.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        raise NotImplementedError

    def initialise_population(self, population):
        """Set our property values for the initial population.

        Must be implemented by subclasses.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in its PROPERTIES dictionary.

        TODO: We probably need to declare somehow which properties we 'read' here, so the
        simulation knows what order to initialise modules in!

        :param population: the population of individuals
        """
        raise NotImplementedError

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        Must be implemented by subclasses.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        Must be implemented by subclasses.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError

    def __getattr__(self, name):
        """Look up a module parameter as though it is an object property.

        :param name: the parameter name
        """
        try:
            return self.parameters[name]
        except KeyError:
            raise AttributeError('Attribute %s not found' % name)

    def __setattr__(self, name, value):
        """Set a module parameter as though it is an object property.

        :param name: the parameter name
        :param value: the new value for the parameter
        """
        try:
            super().__setattr__(name, value)
        except AttributeError:
            if name in self.PARAMETERS:
                assert isinstance(value, self.PARAMETERS[name].python_type)
                self.parameters[name] = value
            else:
                raise
