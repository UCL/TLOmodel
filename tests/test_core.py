import os
import pathlib

import pandas as pd
import pytest

from tlo import Module, Parameter, Property, Types


def test_categorical():
    categories = ['M', 'F']
    p = Property(Types.CATEGORICAL, description='Biological Sex', categories=categories)
    s = p.create_series('Sex', 10)
    assert s.dtype.name == 'category'
    assert list(s.cat.categories) == categories
    # by default if ordered is not set categories will be unordered. test this is true
    assert not s.cat.ordered
    # assignment of valid category
    try:
        s[0] = 'M'
    except ValueError:
        pytest.fail('Unexpected ValueError')

    # assignment of invalid category
    with pytest.raises(ValueError):
        s[0] = 'A'

    # test we can create ordered categories.
    # set ordered argument to True
    prop = Property(Types.CATEGORICAL, description='Ordered Categories', categories=[0, 1, 2, 3], ordered=True)
    ser = prop.create_series('ordered_cat', 10)
    # check we now have created ordered categories
    assert ser.cat.ordered, 'categories are not ordered'


def test_dict():
    p = Property(Types.DICT, description="Dictionary property")
    s = p.create_series('Dicts', 10)
    assert s.dtype.name == 'object'
    s.values[:] = s.apply(lambda x: dict())
    s[0]['name'] = 'world'
    assert 'name' in s[0] and s[0]['name'] == 'world'
    assert len(s[1]) == 0
    assert dict() == s[1] == s[2]


class TestLoadParametersFromDataframe:
    """Tests for the load_parameters_from_dataframe method

    No explicit tests for correct parsing of boolean because of how python evaluates truthiness
    """
    def setup(self):
        class ParameterModule(Module):
            def __init__(self):
                super().__init__(name=None)
                self.PARAMETERS = {
                    'int_basic': Parameter(Types.INT, 'int'),
                    'real_basic': Parameter(Types.REAL, 'real'),
                    'categorical_basic': Parameter(Types.CATEGORICAL, 'categorical',
                                                   categories=["category_1", "category_2"]),
                    'list_basic_int': Parameter(Types.LIST, 'list_int'),
                    'list_basic_real': Parameter(Types.LIST, 'list_real'),
                    'string_basic': Parameter(Types.STRING, 'string'),
                    'bool_true': Parameter(Types.BOOL, 'string'),
                    'bool_false': Parameter(Types.BOOL, 'string'),
                    'date': Parameter(Types.DATE, 'string'),
                }

        self.module = ParameterModule()

        test_resource = pathlib.Path(os.path.dirname(__file__)) / "resources/ResourceFile_load-parameters.xlsx"

        self.resource = pd.read_excel(test_resource, sheet_name="parameter_values")

    def test_happy_path(self):
        """Simple case parsing from excel file dataframe"""

        self.module.load_parameters_from_dataframe(self.resource)

        assert isinstance(self.module.parameters['int_basic'], int)
        assert isinstance(self.module.parameters['real_basic'], float)
        assert isinstance(self.module.parameters['categorical_basic'], pd.Categorical)
        assert isinstance(self.module.parameters['list_basic_int'], list)
        assert isinstance(self.module.parameters['list_basic_real'], list)
        assert isinstance(self.module.parameters['string_basic'], str)
        assert isinstance(self.module.parameters['bool_true'], bool)
        assert isinstance(self.module.parameters['bool_false'], bool)
        assert isinstance(self.module.parameters['date'], pd.Timestamp)
        assert self.module.parameters['bool_true'] is True
        assert self.module.parameters['bool_false'] is False
        assert self.module.parameters['date'] == pd.Timestamp("2015-02-01")

    def test_string_stripping(self):
        """Strings should have left and right whitespace trimmed"""
        resource = self.resource.copy()
        resource.loc[resource.parameter_name == 'string_basic', 'value'] = ' string_with_space    '
        self.module.load_parameters_from_dataframe(resource)

        assert self.module.parameters['string_basic'] == 'string_with_space'

    def test_parameter_df_not_in_class(self):
        """Resource df is missing a parameter

        should raise an Exception"""
        resource = self.resource.copy()
        resource = resource.append({"parameter_name": "new_value", "value": 1}, ignore_index=True)
        with pytest.raises(KeyError):
            self.module.load_parameters_from_dataframe(resource)

    def test_invalid_numeric(self):
        """Non-numeric value given to a numeric field

        should raise an Exception"""
        resource = self.resource.copy()
        resource.loc[resource.parameter_name == 'real_basic', 'value'] = "a"
        with pytest.raises(ValueError):
            self.module.load_parameters_from_dataframe(resource)

    def test_invalid_category(self):
        """Category is given in sheet which has not been defined in the parameter

        should raise an Exception"""
        resource = self.resource.copy()
        resource.loc[resource.parameter_name == 'categorical_basic', 'value'] = "invalid"
        with pytest.raises(AssertionError):
            self.module.load_parameters_from_dataframe(resource)

    @pytest.mark.parametrize("list_value", ["[1, 2, 3", "[1 2, 3]", "['a' 'b', 'c']", 'a', '2'])
    def test_invalid_list(self, list_value):
        """List input isn't parsable as a list

        should raise an Exception"""
        resource = self.resource.copy()
        resource.loc[resource.parameter_name == 'list_basic_int', 'value'] = list_value
        with pytest.raises(ValueError):
            self.module.load_parameters_from_dataframe(resource)

    @pytest.mark.parametrize("date_value", ["[1, 2, 3]", "monday", False])
    def test_invalid_date(self, date_value):
        """Non-valid dates

        should raise an Exception"""
        resource = self.resource.copy()
        resource.loc[resource.parameter_name == 'date', 'value'] = date_value
        with pytest.raises(ValueError):
            self.module.load_parameters_from_dataframe(resource)

    def test_skip_dataframe_and_series(self):
        """Dataframe and Series should not be added to the module parameters"""
        self.module.PARAMETERS['data_frame'] = Parameter(Types.DATA_FRAME, "data_frame")
        self.module.PARAMETERS['series'] = Parameter(Types.SERIES, "series")

        self.module.load_parameters_from_dataframe(self.resource)
        assert 'data_frame' in self.module.PARAMETERS.keys()
        assert 'data_frame' not in self.module.parameters.keys()
        assert 'series' in self.module.PARAMETERS.keys()
        assert 'series' not in self.module.parameters.keys()
