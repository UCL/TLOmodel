"""Core framework classes.

This contains things that didn't obviously go in their own module,
such as specification for parameters and properties.
"""

from enum import Enum, auto


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
    TYPE_MAP = {
        Types.DATE: 'datetime64[ns]',
        Types.BOOL: bool,
        Types.INT: int,
        Types.REAL: float,
        Types.CATEGORICAL: 'category',
        Types.LIST: object,
    }

    def __init__(self, type_, description):
        """Create a new Specifiable.

        :param type_: an instance of Types giving the type of allowed values
        :param description: textual description of what this Specifiable represents
        """
        assert type_ in Types
        self.type_ = type_
        self.description = description


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
