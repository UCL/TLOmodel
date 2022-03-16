"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.
See https://github.com/UCL/TLOmodel/wiki/Diagnostic-Tests-(DxTest)-and-the-Diagnostic-Tests-Manager-(DxManager)
"""
import json
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_float_dtype

from tlo import logging
from tlo.events import IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxManager:
    """
    This is the Diagnostic Tests Manager (DxManager).
    It simplifies the process of conducting diagnostic tests on a person.
    It can store and then apply Diagnostic Tests (DxTest) and return the result.

    :param hs_module: an instance of HealthSystem module
    """

    def __init__(self, hs_module):
        self.hs_module = hs_module
        self.dx_tests: Dict[str, Tuple[DxTest]] = dict()

    def register_dx_test(self, **dict_of_tests_to_register):
        """
        Diagnostics tests are registered in the DxManager by passing in a dictionary of one of more DxTests:
        """
        for name, dx_test in dict_of_tests_to_register.items():
            # Examine the proposed name of the dx_test:
            assert isinstance(name, str), f'Name is not a string: {name}'

            # Examine the proposed dx_test:
            # Make each item provided into a tuple of DxTest objects.
            if not isinstance(dx_test, tuple):
                dx_test = (dx_test,)

            # Check that the objects given are each a DxTest object
            assert all([isinstance(d, DxTest) for d in dx_test]), 'One of the passed objects is not a DxTest object.'

            # Checks on each DxTest
            df = self.hs_module.sim.population.props
            for d in dx_test:
                assert isinstance(d, DxTest), 'One of the passed objects is not a DxTest object'
                assert d.property in df.columns, f'Column {d.property} does exist in population dataframe'
                # if property is category, check target categories have been provided
                if d.target_categories is not None:
                    assert is_categorical_dtype(df[d.property]), f'{d.property} is not categorical'
                    assert isinstance(d.target_categories, list), 'target_categories must be list of categories'
                    property_categories = df[d.property].cat.categories
                    assert all(elem in property_categories
                               for elem in d.target_categories), 'not all target_categories are valid categories'

            # Ensure that name is unique and will not over-write a test already registered
            if name not in self.dx_tests:
                # Add the list of DxTests to the dict of registered DxTests (name is not already used)
                self.dx_tests[name] = dx_test
            else:
                # Determine whether to throw an error due to the same name being used as a test already registered
                try:
                    # If the name is already used, then the DxTest must be identical to the one already registered
                    assert self.dx_tests[name] == dx_test
                except (AssertionError):
                    raise ValueError(
                        "The same name have been registered previously for a different DxTest.")

    def print_info_about_dx_test(self, name_of_dx_test):
        assert name_of_dx_test in self.dx_tests, f'This DxTest is not recognised: {name_of_dx_test}'
        the_dx_test = self.dx_tests[name_of_dx_test]
        print()
        print('----------------------')
        print(f'** {name_of_dx_test} **')
        for num, test in enumerate(the_dx_test):
            print(f'   Position in tuple #{num}')
            print(f'consumables item_codes: {test.item_codes}')
            print(f'sensitivity: {test.sensitivity}')
            print(f'specificity: {test.specificity}')
            print(f'property: {test.property}')
            print('----------------------')

    def print_info_about_all_dx_tests(self):
        for dx_test in self.dx_tests:
            self.print_info_about_dx_test(dx_test)

    def run_dx_test(self, dx_tests_to_run, hsi_event, use_dict_for_single=False, report_dxtest_tried=False):
        from tlo.methods.healthsystem import HSI_Event

        # Check that the thing passed to hsi_event is usable as an hsi_event
        assert isinstance(hsi_event, HSI_Event)
        assert hasattr(hsi_event, 'TREATMENT_ID')

        # Make dx_tests_to_run into a list if it is not already one
        if not isinstance(dx_tests_to_run, list):
            dx_tests_to_run = [dx_tests_to_run]

        assert all([name in self.dx_tests for name in dx_tests_to_run]), 'A DxTest name is not recognised.'

        # Create the dict() of results that will be returned
        result_dict_for_list_of_dx_tests = dict()

        # Create the dict of test that were tried (True for worked, False for failed)
        the_dxtests_tried = dict()

        for dx_test in dx_tests_to_run:
            test_result = False

            # Loop through the list of DxTests that are registered under this name:
            for i, test in enumerate(self.dx_tests[dx_test]):
                test_result = test.apply(hsi_event, self.hs_module)

                if test_result is not None:
                    # The DxTest was successful. Log the use of that DxTest
                    # Logging using the name of the DxTest and the number of the test that was tried within it
                    the_dxtests_tried[f"{dx_test}_{i}"] = True
                    break
                else:
                    # The DxTest was not successful. Log the use of that DxTest
                    the_dxtests_tried[f"{dx_test}_{i}"] = False

            result_dict_for_list_of_dx_tests[dx_test] = test_result

        # Decide on type of return:
        if (len(dx_tests_to_run) == 1) and (use_dict_for_single is False):
            result = result_dict_for_list_of_dx_tests[dx_tests_to_run[0]]
        else:
            result = result_dict_for_list_of_dx_tests

        if report_dxtest_tried:
            return result, the_dxtests_tried

        return result


def _assert_float_or_none(value, msg):
    assert (value is None) or isinstance(value, float), msg


def _default_if_none(value, default):
    return value if value is not None else default


class DxTest:
    """
    This is the helper class that contains information about a Diagnostic Test.
    It is specified by passing at initialisation:
    * Mandatory:
    :param property: the column in the sim.population.props that the diagnostic will observe
    * Optional:
    Use of consumable - specify either
        :param item_codes: the item code(s) (and quantities) of the consumables that are required for the
        test to be done.(Follows same format as `get_consumables` in the HSI_Event base class.)
    Performance of test:
        Specify any of the following if the property's dtype is bool
            :param sensitivity: the sensitivity of the test (probability that a true value will be observed as true)
            :param specificity: the specificity of the test (probability that a false value will be observed as false)
        Specify any of the following if the property's dtype is numeric
            :param measure_error_stdev: the standard deviation of the normally distributed (and zero-centered) error in
                                        the observation of a continuous property
            :param threshold: the observed value of a continuous property above which the result of the test is True.
            :param target_categories: if property is categorical, a list of categories corresponding
                                      to a positive result.
    """
    def __init__(self,
                 property: str,
                 item_codes: Union[np.integer, int, list, set, dict] = None,
                 sensitivity: float = None,
                 specificity: float = None,
                 measure_error_stdev: float = None,
                 threshold: float = None,
                 target_categories: List[str] = None
                 ):

        # Store the property on which it acts (This is the only required parameter)
        assert isinstance(property, str), 'argument "property" is required'
        self.property = property

        # Store consumable code (None means that no consumables are required)
        if item_codes is not None:
            assert isinstance(item_codes, (np.integer, int, list, set, dict)), 'item_codes in incorrect format.'
        self.item_codes = item_codes

        # Store performance characteristics (if sensitivity and specificity are not supplied than assume perfect)
        _assert_float_or_none(sensitivity, 'Sensitivity is given in incorrect format.')
        _assert_float_or_none(specificity, 'Sensitivity is given in incorrect format.')
        _assert_float_or_none(measure_error_stdev, 'measure_error_stdev is given in incorrect format.')
        _assert_float_or_none(threshold, 'threshold is given in incorrect format.')

        self.sensitivity = _default_if_none(sensitivity, 1.0)
        self.specificity = _default_if_none(specificity, 1.0)
        self.measure_error_stdev = _default_if_none(measure_error_stdev, 0.0)
        self.threshold = threshold
        self.target_categories = target_categories

    def __hash_key(self):
        if isinstance(self.item_codes, (dict, list)):
            item_codes_key = json.dumps(self.item_codes, sort_keys=True)
        elif isinstance(self.item_codes, set):
            item_codes_key = frozenset(self.item_codes)
        elif isinstance(self.item_codes, np.integer):
            item_codes_key = int(self.item_codes)
        else:
            item_codes_key = self.item_codes
        return (
            self.__class__,
            item_codes_key,
            self.property,
            self.sensitivity,
            self.specificity,
            self.measure_error_stdev
        )

    def __hash__(self):
        return hash(self.__hash_key())

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.__hash_key() == other.__hash_key()
        return NotImplemented

    def apply(self, hsi_event, hs_module):
        """
        This is where the test is applied.
        If this test returns None this means the test has failed due to there not being the required consumables.

        :param hsi_event:
        :return: value of test or None.
        """
        # Must be an individual level HSI and not a population level HSI
        assert isinstance(hsi_event, IndividualScopeEventMixin), 'DxManager requires individual-level HSI_Event'
        assert isinstance(hsi_event.target, int), 'DxManager requires individual-level HSI_Event'
        person_id = hsi_event.target

        # Get the "true value" of the property being examined
        df: pd.DataFrame = hs_module.sim.population.props
        assert self.property in df.columns, \
            f'The property "{self.property}" is not found in the population dataframe'
        true_value = df.at[person_id, self.property]

        # If a consumable is required and it is not available, return None
        if self.item_codes is not None:
            if not hsi_event.get_consumables(item_codes=self.item_codes):
                return None

        # Apply the test:
        if is_bool_dtype(df[self.property]):
            if true_value:
                # Apply the sensitivity:
                test_value = hs_module.rng.rand() < self.sensitivity
            else:
                # Apply the specificity:
                test_value = not (hs_module.rng.rand() < self.specificity)

        elif is_float_dtype(df[self.property]):
            # Apply the normally distributed zero-mean error
            reading = true_value + hs_module.rng.normal(0.0, self.measure_error_stdev)

            # If no threshold value is provided, then return the reading; otherwise apply the threshold
            if self.threshold is None:
                test_value = float(reading)
            else:
                test_value = bool(reading >= self.threshold)

        elif self.target_categories is not None:
            # Categorical property: compare the value to the 'target_categories' if its specified
            is_match_to_cat = (true_value in self.target_categories)

            if is_match_to_cat:
                # Apply the sensitivity:
                test_value = hs_module.rng.rand() < self.sensitivity
            else:
                # Apply the specificity:
                test_value = not (hs_module.rng.rand() < self.specificity)

        else:
            test_value = true_value

        # Return test_value
        return test_value
