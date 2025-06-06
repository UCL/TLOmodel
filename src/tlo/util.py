"""This file contains helpful utility functions."""
import ast
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, DateOffset
from pandas._typing import DtypeArg

from tlo import Population, Property, Types

# Default mother_id value, assigned to individuals initialised as adults at the start of the simulation.
DEFAULT_MOTHER_ID = -1e7


def create_age_range_lookup(min_age: int, max_age: int, range_size: int = 5) -> (list, Dict[int, str]):
    """Create age-range categories and a dictionary that will map all whole years to age-range categories

    If the minimum age is not zero then a below minimum age category will be made,
    then age ranges until maximum age will be made by the range size,
    all other ages will map to the greater than maximum age category.

    :param min_age: Minimum age for categories,
    :param max_age: Maximum age for categories, a greater than maximum age category will be made
    :param range_size: Size of each category between minimum and maximum ages
    :returns:
        age_categories: ordered list of age categories available
        lookup: Default dict of integers to maximum age mapping to the age categories
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit
    parts = chunks(range(min_age, max_age), range_size)

    default_category = f'{max_age}+'
    lookup = defaultdict(lambda: default_category)
    age_categories = []

    # create category for minimum age
    if min_age > 0:
        under_min_age_category = f'0-{min_age}'
        age_categories.append(under_min_age_category)
        for i in range(0, min_age):
            lookup[i] = under_min_age_category

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = f'{start}-{end}'
        age_categories.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    age_categories.append(default_category)

    return age_categories, lookup


def transition_states(initial_series: pd.Series, prob_matrix: pd.DataFrame, rng: np.random.RandomState) -> pd.Series:
    """Transition a series of states based on probability matrix

    This should carry out all state transitions for a Series (i.e. column in DataFrame)
    based on the probability of state-transition matrix.

    Timing values for 1M rows per state, 4 states, 100 times:
    - Looping through groups: [59.5, 58.7, 59.5]
    - Using apply: [84.2, 83.3, 84.4]
    Because of this, looping through the groups was chosen

    :param Series initial_series: the initial state series
    :param DataFrame prob_matrix: DataFrame of state-transition probabilities
        columns are the original state, rows are the new state. values are the probabilities
    :param RandomState rng: RandomState from the disease module
    :return: Series with states changed according to probabilities
    """
    # Create final_series with index so that we are sure it's the same size as the original
    final_states = pd.Series(None, index=initial_series.index, dtype=initial_series.dtype)

    # for each state, get the random choice states and add to the final_states Series
    state_indexes = initial_series.groupby(initial_series).groups
    all_states = prob_matrix.index.tolist()
    for state, state_index in state_indexes.items():
        if not state_index.empty:
            new_states = rng.choice(all_states, len(state_index), p=prob_matrix[state])
            final_states[state_index] = new_states
    return final_states


def sample_outcome(probs: pd.DataFrame, rng: np.random.RandomState):
    """ Helper function to randomly sample an outcome for each individual in a population from a set of probabilities
    that are specific to each individual.
    :param probs: Each row of this dataframe represents the person and the columns are the possible outcomes. The
    values are the probability of that outcome for that individual. For each individual, the probabilities of each
    outcome are assumed to be independent and mutually exclusive (but not necessarily exhaustive). If they sum to more
    than 1.0, then they are (silently) scaled so that they do sum to 1.0.
    :param rng: Random Number state to use for the generation of random numbers.
    :return: A dict of the form {<index>:<outcome>} where an outcome is selected.
    """

    # Scaling to ensure that the sum in each row not exceed 1.0
    probs = probs.apply(lambda row: (row / row.sum() if row.sum() >= 1.0 else row), axis=1)
    assert (probs.sum(axis=1) < (1.0 + 1e-6)).all(), "Probabilities across columns cannot sum to more than 1.0"

    # Compare uniform deviate to cumulative sum across columns, after including a "null" column (for no event).
    _probs = probs.copy()
    _probs['_'] = 1.0 - _probs.sum(axis=1)  # add implied "none of these events" category
    cumsum = _probs.cumsum(axis=1)
    draws = pd.Series(rng.rand(len(cumsum)), index=cumsum.index)
    y = cumsum.gt(draws, axis=0)
    outcome = y.idxmax(axis=1)

    # return as a dict of form {person_id: outcome} only in those cases where the outcome is one of the events.
    return outcome.loc[outcome != '_'].to_dict()


BitsetDType = Property.PANDAS_TYPE_MAP[Types.BITSET]


class BitsetHandler:
    """Provides methods to operate on int column(s) in the population dataframe as a bitset"""

    def __init__(self, population: Population, column: Optional[str], elements: List[str]):
        """
        :param population: The TLO Population object (not the props dataframe).
        :param column: The integer property column that will be used as a bitset. If
            set to ``None`` then the optional `columns` argument to methods which act
            on the population dataframe __must__ be specified.
        :param elements: A list of strings specifying the elements of the bitset.
        :returns: Instance of BitsetHandler for supplied arguments.
        """
        assert isinstance(population, Population), (
            'First argument is the population object (not the `props` dataframe)'
        )
        dtype_bitwidth = BitsetDType(0).nbytes * 8
        assert len(elements) <= dtype_bitwidth, (
            f"A maximum of {dtype_bitwidth} elements are supported"
        )
        self._elements = elements
        self._element_to_int_map = {el: 2 ** i for i, el in enumerate(elements)}
        self._population = population
        if column is not None:
            assert column in population.props.columns, (
                'Column not found in population dataframe'
            )
            assert population.props[column].dtype == BitsetDType, (
                f'Column must be of {BitsetDType} type'
            )
        self._column = column

    @property
    def df(self) -> pd.DataFrame:
        return self._population.props

    def element_repr(self, *elements: str) -> BitsetDType:
        """Returns integer representation of the specified element(s)"""
        return BitsetDType(sum(self._element_to_int_map[el] for el in elements))

    def to_strings(self, integer: BitsetDType) -> Set[str]:
        """Given an integer value, returns the corresponding set of strings.

        :param integer: The integer value for the bitset.
        :return: Set of strings corresponding to integer value.
        """
        bin_repr = format(integer, 'b')
        return {
            self._elements[index]
            for index, bit in enumerate(reversed(bin_repr)) if bit == '1'
        }

    def _get_columns(self, columns):
        if columns is None and self._column is None:
            raise ValueError(
                'columns argument must be specified as not set when constructing handler'
            )
        return self._column if columns is None else columns

    def set(self, where, *elements: str, columns: Optional[Union[str, List[str]]] = None):
        """Set (i.e. set to True) the bits corersponding to the specified elements.

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call.
        It can be index  items, a boolean logical condition, or list of row indices e.g. "[0]".

        The elements are one of more valid items from the list of elements for this bitset.

        :param where: Condition to filter rows that will be set.
        :param elements: One or more elements to set to True.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        """
        self.df.loc[where, self._get_columns(columns)] |= self.element_repr(*elements)

    def unset(self, where, *elements: str, columns: Optional[Union[str, List[str]]] = None):
        """Unset (i.e. set to False) the bits corresponding the specified elements.

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call.
        It can be index items, a boolean logical condition, or list of row indices e.g. "[0]".

        The elements are one of more valid items from the list of elements for this bitset.

        :param where: Condition to filter rows that will be unset.
        :param elements: one or more elements to set to False.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        """
        self.df.loc[where, self._get_columns(columns)] &= ~self.element_repr(*elements)  # pylint: disable=E1130

    def clear(self, where, columns: Optional[Union[str, List[str]]] = None):
        """Clears all the bits for the specified rows.

        :param where: Condition to filter rows that will cleared.
        :param columns: Optional argument specifying column(s) containing bitsets to
            clear. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        """
        self.df.loc[where, self._get_columns(columns)] = 0

    def has(
        self,
        where,
        element: str,
        first: bool = False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, bool]:
        """Test whether bit(s) for a specified element are set.

        :param where: Condition to filter rows that will checked.
       :param element: Element string to test if bit is set for.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Boolean value(s) indicating whether element bit(s) are set.
        """
        int_repr = self._element_to_int_map[element]
        matched = (self.df.loc[where, self._get_columns(columns)] & int_repr) != 0
        return matched.iloc[0] if first else matched

    def has_all(
        self,
        where,
        *elements: str,
        first: bool = False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, bool]:
        """Check whether individual(s) have all the elements given set to True.

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call.
        It can be index  items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset.

        :param where: Condition to filter rows that will checked.
        :param elements: One or more elements to set to True.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Boolean value(s) indicating whether all element bit(s) are set.
        """
        int_repr = self.element_repr(*elements)
        matched = (self.df.loc[where, self._get_columns(columns)] & int_repr) == int_repr
        return matched.iloc[0] if first else matched

    def has_any(
        self,
        where,
        *elements: str,
        first: bool = False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, bool]:
        """Check whether individual(s) have any of the elements given set to True.

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call.
        It can be index items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset.

        :param where: Condition to filter rows that will checked.
        :param elements: One or more elements to set to True.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Boolean value(s) indicating whether any element bit(s) are set.
        """
        int_repr = self.element_repr(*elements)
        matched = (self.df.loc[where, self._get_columns(columns)] & int_repr) != 0
        return matched.iloc[0] if first else matched

    def get(
        self,
        where,
        first: bool = False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, Set[str]]:
        """Returns a series or dataframe with set of string elements where bit is True.

        The where argument is used verbatim as the first item in a `df.loc[x, y]` call.
        It can be index items, a boolean logical condition, or list of row indices e.g. "[0]"

        The elements are one of more valid items from the list of elements for this bitset

        :param where: Condition to filter rows that will returned.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Set(s) of strings corresponding to elements with bits set to True.
        """
        columns = self._get_columns(columns)
        if isinstance(columns, str):
            sets = self.df.loc[where, columns].apply(self.to_strings)
        else:
            sets = self.df.loc[where, columns].applymap(self.to_strings)
        return sets.iloc[0] if first else sets

    def uncompress(
        self,
        where=None,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Returns an exploded representation of the bitset(s).

        Each element bit becomes a column and each column is a bool indicating whether
        the bit is set for the element.

        :param where: Condition to filter rows that an exploded representation will be
            returned for.
        :param columns: Optional argument specifying column(s) containing bitsets to
            return exploded representation for. If set to ``None`` (the default) a
            ``column`` argument must have been specified when constructing the
            ``BitsetHandler`` object.
        :return: If ``columns`` is not set or is set to a single string, then a
            dataframe is returned with a column for each element in set and boolean
            values indicating whether the corresponding bit is set; if ``columns`` is
            specified as a list of multiple column names a dictionary keyed by column
            name and with the corresponding value a dataframe corresponding to the
            exploded representation of the column bitset is returned.
        """
        if where is None:
            where = self.df.index
        columns = self._get_columns(columns)
        uncompressed = {}
        for column in [columns] if isinstance(columns, str) else columns:
            collect = dict()
            for element in self._elements:
                collect[element] = self.has(where, element, columns=column)
            uncompressed[column] = pd.DataFrame(collect)
        return uncompressed[columns] if isinstance(columns, str) else uncompressed

    def not_empty(
        self,
        where,
        first=False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, bool]:
        """Returns Series of bool indicating whether the BitSet entry is not empty.

        True is set is not empty, False otherwise.

        :param where: Condition to filter rows that will checked.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Boolean value(s) indicating whether any elements bits are set.
        """
        return ~self.is_empty(where, first=first, columns=columns)

    def is_empty(
        self,
        where,
        first=False,
        columns: Optional[Union[str, List[str]]] = None
    ) -> Union[pd.DataFrame, pd.Series, bool]:
        """Returns Series of bool indicating whether the BitSet entry is empty.

        True if the set is empty, False otherwise.

        :param where: Condition to filter rows that will checked.
        :param first: Boolean keyword argument specifying whether to return only the
            first item / row in the computed column / dataframe.
        :param columns: Optional argument specifying column(s) containing bitsets to
            update. If set to ``None`` (the default) a ``column`` argument must have
            been specified when constructing the ``BitsetHandler`` object.
        :return: Boolean value(s) indicating whether all elements bits are not set.
        """
        empty = self.df.loc[where, self._get_columns(columns)] == 0
        return empty.iloc[0] if first else empty


def random_date(start, end, rng):
    if start >= end:
        raise ValueError("End date equal to or earlier than start date")
    return start + DateOffset(days=rng.randint(0, (end - start).days))


def str_to_pandas_date(date_string):
    """Convert a string with the format YYYY-MM-DD to a pandas Timestamp (aka TLO Date) object."""
    return pd.to_datetime(date_string, format="%Y-%m-%d")


def hash_dataframe(dataframe: pd.DataFrame):
    def coerce_lists_to_tuples(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns in a pd.DataFrame that are lists to tuples. This step is needed before hashing a pd.DataFrame
        as list are not hashable."""
        return df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    return hashlib.sha1(pd.util.hash_pandas_object(coerce_lists_to_tuples(dataframe)).values).hexdigest()


def get_person_id_to_inherit_from(child_id, mother_id, population_dataframe, rng):
    """Get index of person to inherit properties from.
    """

    if mother_id == DEFAULT_MOTHER_ID:
        # Get indices of alive persons and try to drop child_id from these indices if
        # present, ignoring any errors if child_id not currently in population dataframe
        alive_persons_not_including_child = population_dataframe.index[
            population_dataframe.is_alive
        ].drop(child_id, errors="ignore")
        return rng.choice(alive_persons_not_including_child)
    elif 0 > mother_id > DEFAULT_MOTHER_ID:
        return abs(mother_id)
    elif mother_id >= 0:
        return mother_id


def convert_excel_files_to_csv(
    folder: Path, files: Optional[list[str]] = None, *, delete_excel_files: bool = False
) -> None:
    """convert Excel files to csv files.

    :param folder: Folder containing Excel files.
    :param files: List of Excel file names to convert to csv files. When `None`, all Excel files in the folder and
                  subsequent folders within this folder will be converted to csv files with Excel file name becoming
                  folder name and sheet names becoming csv file names.
    :param delete_excel_files: When true, the Excel file we are generating csv files from will get deleted.
    """
    # get path to Excel files
    if files is None:
        excel_file_paths = sorted(folder.rglob("*.xlsx"))
    else:
        excel_file_paths = [folder / file for file in files]
    # exit function if no Excel file is given or found within the path
    if excel_file_paths is None:
        return

    for excel_file_path in excel_file_paths:
        sheet_dataframes: dict[Any, DataFrame] = pd.read_excel(excel_file_path, sheet_name=None)
        excel_file_directory: Path = excel_file_path.with_suffix("")
        # Create a container directory for per sheet CSVs
        if excel_file_directory.exists():
            print(f"Directory {excel_file_directory} already exists")
        else:
            excel_file_directory.mkdir()
        # Write a CSV for each worksheet
        for sheet_name, dataframe in sheet_dataframes.items():
            dataframe.to_csv(f'{excel_file_directory / sheet_name}.csv', index=False)

        if delete_excel_files:
            # Remove no longer needed Excel file
            Path(folder/excel_file_path).unlink()


def read_csv_files(folder: Path,
                   dtype: DtypeArg | dict[str, DtypeArg] | None = None,
                   files: str | int | list[str] | None = 0) -> DataFrame | dict[str, DataFrame]:
    """
    A function to read CSV files in a similar way pandas reads Excel files (:py:func:`pandas.read_excel`).

    NB: Converting Excel files to csv files caused all columns that had no relevant data to simulation (i.e.
    parameter descriptions or data references) to be named `Unnamed1, Unnamed2, ....., UnnamedN` in the csv files.
    We are therefore using :py:func:`pandas.filter` to track all unnamed columns and silently drop them using
    :py:func:`pandas.drop`.

    :param folder: Path to folder containing CSV files to read.
    :param dtype: allows passing in a dictionary of datatypes in cases where you want different datatypes per column
    :param files: preferred csv file name(s). This is the same as sheet names in Excel file. Note that if None(no files
                  selected) then all csv files in the containing folder will be read

                  Please take note of the following behaviours:
                  -----------------------------------------------
                   - if files argument is initialised to zero(default) and the folder contains one or multiple files,
                     this method will return a dataframe. If the folder contain multiple files, it is good to
                     specify file names or initialise files argument with None to ensure correct files are selected
                   - if files argument is initialised to None and the folder contains one or multiple files, this method
                     will return a dataframe dictionary
                   - if the folder contains multiple files and files argument is initialised with one file name this
                     method will return a dataframe. it will return a dataframe dictionary when files argument is
                     initialised with a list of multiple file names

    """
    all_data: dict[str, DataFrame] = {}  # dataframes dictionary

    def clean_dataframe(dataframes_dict: dict[str, DataFrame]) -> None:
        """ silently drop all columns that have no relevant data to simulation (all columns with a name starting with
        Unnamed
        :param dataframes_dict: Dictionary of dataframes to clean
        """
        for _key, dataframe in dataframes_dict.items():
            all_data[_key] = dataframe.drop(dataframe.filter(like='Unnamed'), axis=1)  # filter and drop Unnamed columns

    return_dict = False # a flag that will determine whether the output should be a dictionary or a DatFrame
    if isinstance(files, list):
        return_dict = True
    elif isinstance(files, int) or files is None:
        return_dict = files is None
        files = [f_name.stem for f_name in folder.glob("*.csv")]
    elif isinstance(files, str):
        files = [files]
    else:
        raise TypeError(f"Value passed for files argument {files} is not one of expected types.")

    for f_name in files:
        all_data[f_name] = pd.read_csv((folder / f_name).with_suffix(".csv"), dtype=dtype)
    # clean and return the dataframe dictionary
    clean_dataframe(all_data)
    # return a dictionary if return_dict flag is set to True else return a dataframe
    return all_data if return_dict else next(iter(all_data.values()))


def parse_csv_values_for_columns_with_mixed_datatypes(value: Any):
    """ Pandas :py:func:`pandas.read_csv` function handles columns with mixed data types by defaulting to the object
    data type, which often results in values being interpreted as strings. The most common place for this in TLO is
    when we are reading parameters. This is not a problem when the parameters are read in read parameters method
    using load_parameters_from_dataframe method as parameter values are mapped to their defined datatypes.

    Problems arise when you're trying to directly use the output from the csv files like it is within some few files
    in TLO. This method tries to provide a fix by parsing the parameter values in those few places to their best
    possible data types

    :param value: mixed datatype column value
    """
    # if value is not a string then return value
    if not isinstance(value, str):
        return value

    value = value.strip()  # Remove leading/trailing whitespace
    # It is important to catch booleans early to avoid int(value) which will convert them into an interger value
    # 0(False) or 1(True)
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    try:
        return int(value) # try converting the value to an interger, throw excepetion otherwise
    except ValueError:
        try:
            return float(value) # try converting the value to a float, throw excepetion otherwise
        except ValueError:
            # Check if it's a list using `ast.literal_eval`
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            return value  # Return as a string if no other type fits
