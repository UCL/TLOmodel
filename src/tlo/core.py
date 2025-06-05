"""Core framework classes.

This contains things that didn't obviously go in their own file, such as
specification for parameters and properties, and the base Module class for
disease modules.
"""
from __future__ import annotations

import json
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Optional

    from tlo.methods import Metadata
    from tlo.methods.causes import Cause
    from tlo.population import Population
    from tlo.simulation import Simulation

class Types(Enum):
    """Possible types for parameters and properties.

    This lets us hide the details of numpy & Pandas dtype strings and give
    users an easy list to reference instead.

    Most of these should be intuitive. The CATEGORICAL type is useful for things like
    sex where there are a fixed number of options to choose from. The LIST type is used
    for properties where the value is a collection, e.g. the set of children of a person.
    """
    DATE = auto()
    BOOL = auto()
    INT = auto()
    REAL = auto()
    CATEGORICAL = auto()
    LIST = auto()
    SERIES = auto()
    DATA_FRAME = auto()
    STRING = auto()
    DICT = auto()
    BITSET = auto()


class Specifiable:
    """Base class for Parameter and Property."""

    # Map our Types to pandas dtype specifications
    # Individuals have Property. Property Types map to Pandas dtypes
    PANDAS_TYPE_MAP = {
        Types.DATE: 'datetime64[ns]',
        Types.BOOL: bool,
        Types.INT: 'int64',
        Types.REAL: float,
        Types.CATEGORICAL: 'category',
        Types.LIST: object,
        Types.SERIES: object,
        Types.DATA_FRAME: object,
        Types.STRING: object,
        Types.DICT: object,
        Types.BITSET: np.uint32,
    }

    # Map our Types to Python types
    # Modules has Parameter. Parameter Types map to Python types
    PYTHON_TYPE_MAP = {
        Types.DATE: pd.Timestamp,
        Types.BOOL: bool,
        Types.INT: int,
        Types.REAL: float,
        Types.CATEGORICAL: pd.Categorical,
        Types.LIST: list,
        Types.SERIES: pd.Series,
        Types.DATA_FRAME: pd.DataFrame,
        Types.STRING: object,
        Types.DICT: dict,
        Types.BITSET: int,
    }

    def __init__(self, type_: Types, description: str, categories: List[str] = None):
        """Create a new Specifiable.

        :param type_: an instance of Types giving the type of allowed values
        :param description: textual description of what this Specifiable represents
        :param categories: list of strings which will be the available categories
        """
        assert type_ in Types
        self.type_ = type_
        self.description = description

        # Save the categories for a categorical property
        if self.type_ is Types.CATEGORICAL:
            if not categories:
                raise ValueError("CATEGORICAL types require the 'categories' argument")
            self.categories = categories

    @property
    def python_type(self) -> type:
        """Return the Python type corresponding to this Specifiable."""
        return self.PYTHON_TYPE_MAP[self.type_]

    @property
    def pandas_type(self) -> type:
        """Return the Pandas type corresponding to this Specifiable."""
        return self.PANDAS_TYPE_MAP[self.type_]

    def __repr__(self) -> str:
        """Return detailed description of Specifiable."""

        delimiter = " === "

        if self.type_ == Types.CATEGORICAL:
            return f'{self.type_.name}{delimiter}{self.description} (Possible values are: {self.categories})'

        return f'{self.type_.name}{delimiter}{self.description}'


class Parameter(Specifiable):
    """Used to specify parameters for disease modules etc."""


class Property(Specifiable):
    """Used to specify properties of individuals."""

    # Default values to use for Series of different Pandas dtypes
    PANDAS_TYPE_DEFAULT_VALUE_MAP = {
        "datetime64[ns]": pd.NaT,
        bool: False,
        "int64": 0,
        float: float("nan"),
        "category": float("nan"),
        object: float("nan"),
        np.uint32: 0,
    }
    _default_value_override: Any

    def __init__(
        self,
        type_: Types,
        description: str,
        categories: List[str] = None,
        *,
        ordered: bool = False,
        default_value: Optional[Any] = None,
    ) -> None:
        """Create a new property specification.

        :param type_: An instance of ``Types`` giving the type of allowed values of this
            property.
        :param description: Textual description of what this property represents.
        :param categories: Set of categories this property can take if ``type_`` is
            ``Types.CATEGORICAL``.
        :param ordered: Whether categories are ordered  if ``type_`` is
            ``Types.CATEGORICAL``.
        :param default_value: The default value for the property.
        """
        if type_ in [Types.SERIES, Types.DATA_FRAME]:
            raise TypeError("Property cannot be of type SERIES or DATA_FRAME.")

        super().__init__(type_, description, categories)
        self.ordered = ordered
        # Use _default_value setter method to set property initial value
        self._default_value = default_value

    @property
    def _default_value(self) -> Any:
        """
        Default value for this property, which will be used to fill the respective columns
        of the population dataframe, for example.

        If not explicitly set, it will fall back on the ``PANDAS_TYPE_DEFAULT_TYPE_MAP``.
        If a value is provided, it must:

        - Be of the corresponding TYPE for the property.
        - If ``type_`` is ``Types.CATEGORICAL``, it must also be a possible category.
        """
        return (
            self.PANDAS_TYPE_DEFAULT_VALUE_MAP[self.pandas_type]
            if self._default_value_override is None
            else self._default_value_override
        )

    @_default_value.setter
    def _default_value(self, new_val: Any) -> None:
        if new_val is not None:
            # Check for valid category
            if self.type_ is Types.CATEGORICAL:
                if new_val not in self.categories:
                    raise ValueError(
                        f"Value {new_val} is not a valid category, so cannot be set as the default."
                    )
            # If not categorical, check for valid data type for default
            elif not isinstance(new_val, self.python_type):
                raise ValueError(
                    f"Trying to set a default value of type {type(new_val).__name__}, "
                    f"which is different from Property's type of {type(self.python_type).__name__}."
                )
        # Outside block so that providing new_val = None reverts to Property-wide default.
        self._default_value_override = new_val

    def create_series(self, name: str, size: int) -> pd.Series:
        """Create a Pandas Series for this property.

        The values will be left uninitialised.

        :param name: The name for the series.
        :param size: The length of the series.
        """
        # Series of Categorical are set up differently
        if self.type_ is Types.CATEGORICAL:
            dtype = pd.CategoricalDtype(
                categories=self.categories, ordered=self.ordered
            )
        else:
            dtype = self.pandas_type

        return pd.Series(
            data=[self._default_value] * size,
            name=name,
            index=range(size),
            dtype=dtype,
        )


class Module:
    """The base class for disease modules.

    This declares the methods which individual modules must implement, and contains the
    core functionality linking modules into a simulation. Useful properties available
    on instances are:

    `name`
        The unique name of this module within the simulation.

    `parameters`
        A dictionary of module parameters, derived from specifications in the PARAMETERS
        class attribute on a subclass.

    `rng`
        A random number generator specific to this module, with its own internal state.
        It's an instance of `numpy.random.RandomState`

    `sim`
        The simulation this module is part of, once registered.
    """

    # Subclasses can override this to declare the set of initialisation dependencies
    # Declares modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES: FrozenSet[str] = frozenset()

    # Subclasses can override this to declare the set of optional init. dependencies
    # Declares modules that need to be registered in simulation and initialised before
    # this module if they are present, but are not required otherwise
    OPTIONAL_INIT_DEPENDENCIES: FrozenSet[str] = frozenset()

    # Subclasses can override this to declare the set of additional dependencies
    # Declares any modules that need to be registered in simulation in addition to those
    # in INIT_DEPENDENCIES to allow running simulation
    ADDITIONAL_DEPENDENCIES: FrozenSet[str] = frozenset()

    # Subclasses can override this to declare the set of modules that this module can be
    # used in place of as a dependency
    ALTERNATIVE_TO: FrozenSet[str] = frozenset()

    # Subclasses can override this set to add metadata tags to their class
    # See tlo.methods.Metadata class
    METADATA: FrozenSet[Metadata] = frozenset()

    # Subclasses can override this dict to declare the causes death that this module contributes to
    # This is a dict of the form {<name_used_by_the_module : Cause()}: see core.Cause
    CAUSES_OF_DEATH: Dict[str, Cause] = {}

    # Subclasses can override this set to declare the causes disability that this module contributes to
    # This is a dict of the form {<name_used_by_the_module : Cause()}: see core.Cause
    CAUSES_OF_DISABILITY: Dict[str, Cause] = {}

    # Subclasses may declare this dictionary to specify module-level parameters.
    # We give an empty definition here as default.
    PARAMETERS: Dict[str, Parameter] = {}

    # Subclasses may declare this dictionary to specify properties of individuals.
    # We give an empty definition here as default.
    PROPERTIES: Dict[str, Property] = {}

    # The explicit attributes of the module. We list these to distinguish dynamic
    # parameters created from the PARAMETERS specification.
    __slots__ = ('name', 'parameters', 'rng', 'sim')

    def __init__(self, name: Optional[str] = None) -> None:
        """Construct a new disease module ready to be included in a simulation.

        Initialises an empty parameters dictionary and module-specific random number
        generator.

        :param name: the name to use for this module. Defaults to the concrete subclass' name.
        """
        self.parameters = {}
        self.rng: Optional[np.random.RandomState] = None
        self.name = name or self.__class__.__name__
        self.sim: Optional[Simulation] = None

    def load_parameters_from_dataframe(self, resource: pd.DataFrame) -> None:
        """Automatically load parameters from resource dataframe, updating the class parameter dictionary

        Goes through parameters dict self.PARAMETERS and updates the self.parameters with values
        Automatically updates the values of data types:
        - Integers
        - Real numbers
        - Lists
        - Categorical
        - Strings
        - Dates (Any numbers will be converted into dated without warnings)
        - Booleans (An value in the csv '0', 0, 'False', 'false', 'FALSE', or None will be interpreted as False;
            everything else as True)

        Will also make the parameter_name the index of the resource DataFrame.

        :param DataFrame resource: DataFrame with a column of the parameter_name and a column of `value`
        """
        resource.set_index('parameter_name', inplace=True)
        skipped_data_types = ('DATA_FRAME', 'SERIES')
        # for each supported parameter, convert to the correct type
        for parameter_name in resource.index[resource.index.notnull()]:
            parameter_definition = self.PARAMETERS[parameter_name]

            if parameter_definition.type_.name in skipped_data_types:
                continue

            # For each parameter, raise error if the value can't be coerced
            parameter_value = resource.at[parameter_name, 'value']
            error_message = (
                f"The value of '{parameter_value}' for parameter '{parameter_name}' "
                f"could not be parsed as a {parameter_definition.type_.name} data type"
            )
            if parameter_definition.python_type is list:
                try:
                    # chose json.loads instead of save_eval
                    # because it raises error instead of joining two strings without a comma
                    parameter_value = json.loads(parameter_value)
                    assert isinstance(parameter_value, list)
                except (json.decoder.JSONDecodeError, TypeError, AssertionError) as exception:
                    raise ValueError(error_message) from exception
            elif parameter_definition.python_type == pd.Categorical:
                categories = parameter_definition.categories
                assert parameter_value in categories, f"{error_message}\nvalid values: {categories}"
                parameter_value = pd.Categorical([parameter_value], categories=categories)
            elif parameter_definition.type_.name == 'STRING':
                parameter_value = parameter_value.strip()
            elif parameter_definition.type_.name == 'BOOL':
                parameter_value = False if (
                    parameter_value in (0, '0', None, 'FALSE', 'False', 'false') or pd.isna(parameter_value)
                ) else True
            else:
                # All other data types, assign to the python_type defined in Parameter class
                try:
                    parameter_value = parameter_definition.python_type(parameter_value)
                except Exception as exception:
                    raise ValueError(error_message) from exception

            # Save the values to the parameters
            self.parameters[parameter_name] = parameter_value

    def read_parameters(self, data_folder: str | Path) -> None:
        """Read parameter values from file, if required.

        Must be implemented by subclasses.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically, modules would read a particular file within here.
        """
        raise NotImplementedError

    def initialise_population(self, population: Population) -> None:
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in its PROPERTIES dictionary.

        By default, all ``Property``s in ``self.PROPERTIES`` will have
        their columns in the population dataframe set to the default value.

        Modules that wish to implement this behaviour do not need to implement this method,
        it will be inherited automatically. Modules that wish to perform additional steps
        during the initialise_population stage should reimplement this method and call

        ```python
        super().initialise_population(population=population)
        ```

        at the beginning of the method, then proceed with their additional steps. Modules that
        do not wish to inherit this default behaviour should re-implement initialise_population
        without the call to ``super()`` above.

        TODO: We probably need to declare somehow which properties we 'read' here, so the
        simulation knows what order to initialise modules in!

        :param population: The population of individuals in the simulation.
        """
        df = population.props

        for property_name, property in self.PROPERTIES.items():
            df.loc[df.is_alive, property_name] = (
                property._default_value
            )

    def initialise_simulation(self, sim: Simulation) -> None:
        """Get ready for simulation start.

        Must be implemented by subclasses.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def pre_initialise_population(self) -> None:
        """Carry out any work before any populations have been initialised

        This optional method allows access to all other registered modules, before any of
        the modules have initialised a population. This is expected to be useful for
        when a module's properties rely upon information from other modules.
        """

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """Initialise our properties for a newborn individual.

        Must be implemented by subclasses.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the person id for the mother of this child (can be -1 if the mother is not identified).
        :param child_id: the person id of new child
        """
        raise NotImplementedError

    def on_simulation_end(self) -> None:
        """This is called after the simulation has ended.
        Modules do not need to declare this."""
