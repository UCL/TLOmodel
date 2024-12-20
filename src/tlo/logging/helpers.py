from collections.abc import Collection, Iterable
from typing import Dict, List, Optional, Union

import pandas as pd
from pandas.api.types import is_extension_array_dtype


def get_dataframe_row_as_dict_for_logging(
    dataframe: pd.DataFrame,
    row_label: Union[int, str],
    columns: Optional[Iterable[str]] = None,
) -> dict:
    """Get row of a pandas dataframe in a format suitable for logging.
    
    Retrieves entries for all or a subset of columns for a particular row in a dataframe
    and returns a dict keyed by column name, with values NumPy or pandas extension types
    which should be the same for all rows in dataframe.
    
    :param dataframe: Population properties dataframe to get properties from.
    :param row_label: Unique index label identifying row in dataframe.
    :param columns: Set of column names to extract - if ``None``, the default, all
        column values will be returned.
    :returns: Dictionary with column names as keys and corresponding entries in row as
        values.
    """
    dataframe = dataframe.convert_dtypes(convert_integer=False, convert_floating=False)
    columns = dataframe.columns if columns is None else columns
    row_index = dataframe.index.get_loc(row_label)
    return {
        column_name:
        dataframe[column_name].values[row_index]
        # pandas extension array datatypes such as nullable types and categoricals, will
        # be type unstable if a scalar is returned as NA / NaT / NaN entries will have a
        # different type from non-missing entries, therefore use a length 1 array of
        # relevant NumPy or pandas extension type in these cases to ensure type
        # stability across different rows.
        if not is_extension_array_dtype(dataframe[column_name].dtype) else
        dataframe[column_name].values[row_index:row_index+1]
        for column_name in columns
    }


def grouped_counts_with_all_combinations(
    dataframe: pd.DataFrame,
    group_by_columns: List[str],
    column_possible_values: Optional[Dict[str, Collection]] = None,
) -> pd.Series:
    """Perform group-by count in which all combinations of column values are included.

    As all combinations are included irrespective of whether they are present in data
    (and so have a non-zero count), this gives a multi-index series output of fixed
    structure suitable for logging.

    Attempts to convert all columns to categorical datatype, with bool(ean) columns
    automatically converted, and other non-categorical columns needing to have set of
    possible values specified (which requires that this set is finite).

    :param dataframe: Dataframe to perform group-by counts on.
    :param group_by_columns: Columns to perform grouping on.
    :param column_possible_values: Dictionary mapping from column names to set of
        possible values for all columns not of categorical or bool(ean) data type.
    :returns: Multi-index series with values corresponding to grouped counts.
    """
    subset = dataframe[group_by_columns].copy()
    # Convert any bool(ean) columns to categoricals
    for column_name in group_by_columns:
        if subset[column_name].dtype in ("bool", "boolean"):
            subset[column_name] = pd.Categorical(
                subset[column_name], categories=[True, False]
            )
    # For other non-categorical columns possible values need to be explicitly stated
    if column_possible_values is not None:
        for column_name, possible_values in column_possible_values.items():
            subset[column_name] = pd.Categorical(
                subset[column_name], categories=possible_values
            )
    if not (subset.dtypes == "category").all():
        msg = "At least one column not convertable to categorical dtype:\n" + str(
            {subset.dtypes[subset.dtypes != "categorical"]}
        )
        raise ValueError(msg)
    return subset.groupby(by=group_by_columns).size()
