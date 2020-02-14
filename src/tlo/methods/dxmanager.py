"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.
See https://github.com/UCL/TLOmodel/wiki/Diagnostic-Tests-(DxTest)-and-the-Diagnostic-Tests-Manager-(DxManager)
"""
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tlo import logging
from tlo.events import IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxManager:
    """
    The is the Diagnostic Tests Manager (DxManager).
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
            assert all([isinstance(d, DxTest) for d in dx_test]), f'One of the passed objects is not a DxTest object.'

            # Check if this tuple of DxTests is a duplicate of something already registered.
            if (dx_test in self.dx_tests.values()) or (name in self.dx_tests):
                try:
                    assert self.dx_tests[name] == dx_test
                except:
                    raise ValueError(
                        "The same Dx_Test or the same name have been registered previously against a different name "
                        "or DxTest.")

            # Add the list of DxTests to the dict of registered DxTests
            self.dx_tests.update({name: dx_test})

    def print_info_about_dx_test(self, name_of_dx_test):
        assert name_of_dx_test in self.dx_tests, f'This DxTest is not recognised: {name_of_dx_test}'
        the_dx_test = self.dx_tests[name_of_dx_test]
        print()
        print(f'----------------------')
        print(f'** {name_of_dx_test} **')
        for num, test in enumerate(the_dx_test):
            print(f'   Position in tuple #{num}')
            print(f'consumables: {test.cons_req_as_footprint}')
            print(f'sensitivity: {test.sensitivity}')
            print(f'specificity: {test.specificity}')
            print(f'property: {test.property}')
            print(f'----------------------')

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

        assert all([name in self.dx_tests for name in dx_tests_to_run]), f'A DxTest name is not recognised.'

        # Create the dict() of results that will be returned
        result_dict_for_list_of_dx_tests = dict()

        # Create the dict of test that were tried (True for worked, False for failed)
        the_dxtests_tried = dict()

        for dx_test in dx_tests_to_run:
            test_result = False

            # Loop through the list of DxTests that are registered under this name:
            for test in self.dx_tests[dx_test]:
                test_result = test.apply(hsi_event, self.hs_module)

                if test_result is not None:
                    # The DxTest was successful. Log the use of that DxTest
                    the_dxtests_tried[test] = True
                    break
                else:
                    # The DxTest was not successful. Log the use of that DxTest
                    the_dxtests_tried[test] = False

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
    Use of consumable - specify either:
        :param cons_req_as_footprint: the footprint of the consumables that are required for the test to be done.
        or:
        :param cons_req_as_footprint: the item code of the consumables that are required for the test to be done.
    Performance of test:
        Specify any of the following if the property's dtype is bool
            :param sensitivity: the sensitivity of the test (probability that a true value will be observed as true)
            :param specificity: the specificity of the test (probabilit that a false value will be observed as false)
        Specify any of the following if the property's dtype is numeric
            :param measure_error_stdev: the standard deviation of the normally distributed (and zero-centered) error in
                                        the observation of a continuous property
            :param threshold: the observed value of a continuous property above which the result of the test is True.
    """

    def __init__(self,
                 property: str,
                 cons_req_as_footprint=None,
                 cons_req_as_item_code=None,
                 sensitivity: float = None,
                 specificity: float = None,
                 measure_error_stdev: float = None,
                 threshold: float = None
                 ):

        # Store the property on which it acts (This is the only required parameter)
        assert isinstance(property, str), 'argument "property" is required'
        self.property = property

        # Store consumable code (None means that no consumables are required)
        self.cons_req_as_footprint = None
        if (cons_req_as_footprint is not None) and (cons_req_as_item_code is not None):
            raise ValueError('Consumable requirement was provided as both item code and footprint.')
        elif cons_req_as_footprint is not None:
            self.cons_req_as_footprint = cons_req_as_footprint
        elif cons_req_as_item_code is not None:
            self.cons_req_as_footprint = {
                'Intervention_Package_Code': {},
                'Item_Code': {cons_req_as_item_code: 1},
            }

        # Store performance characteristics (if sensitivity and specificity are not supplied than assume perfect)
        _assert_float_or_none(sensitivity, 'Sensitivity is given in incorrect format.')
        _assert_float_or_none(specificity, 'Sensitivity is given in incorrect format.')
        _assert_float_or_none(measure_error_stdev, 'measure_error_stdev is given in incorrect format.')
        _assert_float_or_none(threshold, 'threshold is given in incorrect format.')

        self.sensitivity = _default_if_none(sensitivity, 1.0)
        self.specificity = _default_if_none(specificity, 1.0)
        self.measure_error_stdev = _default_if_none(measure_error_stdev, 0.0)
        self.threshold = threshold

    def __hash_key(self):
        return (
            self.__class__,
            json.dumps(self.cons_req_as_footprint, sort_keys=True),
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
        if self.cons_req_as_footprint is not None:
            # check availability of consumable
            rtn_from_health_system = hs_module.request_consumables(hsi_event,
                                                                   self.cons_req_as_footprint,
                                                                   to_log=True)

            if not (all(rtn_from_health_system['Intervention_Package_Code'].values()) and
                    all(rtn_from_health_system['Item_Code'].values())):
                return None

        # Apply the test:
        if df[self.property].dtype == np.dtype('bool'):
            if true_value:
                # Apply the sensitivity:
                test_value = hs_module.rng.choice([True, False], p=[self.sensitivity, 1 - self.sensitivity])
            else:
                # Apply the specificity:
                test_value = hs_module.rng.choice([False, True], p=[self.specificity, 1 - self.specificity])

        elif df[self.property].dtype == np.dtype('float'):
            # Apply the normally distributed zero-mean error
            reading = true_value + hs_module.rng.normal(0.0, self.measure_error_stdev)

            # If no threshold value is provided, then return the reading; otherwise apply the threshold
            if self.threshold is None:
                test_value = float(reading)
            else:
                test_value = bool(reading >= self.threshold)
        else:
            test_value = true_value

        # Return test_value
        return test_value
