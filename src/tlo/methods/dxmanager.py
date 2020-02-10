"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.
See https://github.com/UCL/TLOmodel/wiki/Diagnostic-Tests-(DxTest)-and-the-Diagnostic-Tests-Manager-(DxManager)
"""

import numpy as np

from tlo import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxManager:
    """
    The is the Diagnostic Tests Manager (DxManager).
    It simplifies the process of conducting diagnostic tests on a person.
    It can store and then apply Diagnostic Tests (DxTest) and return the result.
    """

    def __init__(self):
        self.dx_tests = dict()
        self.dx_test_hash = set()

    def register_dx_test(self, **dict_of_tests_to_register):
        """
        Diagnostics tests are registered in the DxManager by passing in a dictionary of one of more DxTests:
        """

        for name, dx_test in dict_of_tests_to_register.items():
            # Examine the proposed name of the dx_test:
            assert isinstance(name, str), f'Name is not a string: {name}'
            assert name not in self.dx_tests, 'Test name already in use'

            # Examine the proposed dx_test:
            # Make each item provided into a list of DxTest objects.
            if not isinstance(dx_test, list):
                dx_test = [dx_test]

            # Check that the objects given are each a DxTest object
            assert all([isinstance(d, DxTest) for d in dx_test]), f'One of the passed objects is not a DxTest object.'

            # Check if this List of DxTests is a duplicate
            hash_of_dx_test = hash(tuple([hash(d) for d in dx_test]))
            if hash_of_dx_test in self.dx_test_hash:
                logger.warning('This exact same DxTest or list of DxTests has already been registered.')
            else:
                # Add the list of DxTests to the dict of registered DxTests
                self.dx_tests.update({name: dx_test})
                self.dx_test_hash.add(hash_of_dx_test)

    def print_info_about_dx_test(self, name_of_dx_test):
        assert name_of_dx_test in self.dx_tests, f'This DxTest is not recognised: {name_of_dx_test}'
        the_dx_test = self.dx_tests[name_of_dx_test]
        print()
        print(f'----------------------')
        print(f'** {name_of_dx_test} **')
        for n, t in enumerate(the_dx_test):
            print(f'   Position in List #{n}')
            print(f'consumbales: {t.cons_req_as_footprint}')
            print(f'sensitivity: {t.sensitivity}')
            print(f'specificity: {t.specificity}')
            print(f'property: {t.property}')
            print(f'----------------------')

    def print_info_about_all_dx_tests(self):
        for dx_test in self.dx_tests:
            self.print_info_about_dx_test(dx_test)

    def run_dx_test(self, dx_tests_to_run, hsi_event, use_dict_for_single=False, report_DxTest_tried=False):

        # Check that the thing passed to hsi_event is usable as an hsi_event
        assert 'TREATMENT_ID' in dir(hsi_event)
        assert 'rng' in dir(hsi_event.module)
        assert 'HSI_' in str(hsi_event.__class__)

        # Make dx_tests_to_run into a list if it is not already one
        if not isinstance(dx_tests_to_run, list):
            dx_tests_to_run = [dx_tests_to_run]

        assert all(
            [name in self.dx_tests for name in dx_tests_to_run]
        ), f'A DxTest name is not recognised.'

        # Create the dict() of results that will be returned
        result_dict_for_list_of_dx_tests = dict()

        # Create the dict of test that were tried (True for worked, False for failed)
        the_dxtests_tried = dict()

        for dx_test in dx_tests_to_run:

            # Loop through the list of DxTests that are registered under this name:
            for t in self.dx_tests[dx_test]:
                t_res = t.apply(hsi_event)

                if t_res is not None:
                    # The DxTest was successful. Log the use of that DxTest
                    the_dxtests_tried[t] = True
                    break
                else:
                    # The DxTest was not successful. Log the use of that DxTest
                    the_dxtests_tried[t] = False

            result_dict_for_list_of_dx_tests[dx_test] = t_res

        # Decide on type of return:
        if (len(dx_tests_to_run) == 1) and (use_dict_for_single is False):
            result = result_dict_for_list_of_dx_tests[dx_tests_to_run[0]]
        else:
            result = result_dict_for_list_of_dx_tests

        if report_DxTest_tried:
            return result, the_dxtests_tried
        else:
            return result


class DxTest:
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
        if (property is not None) and isinstance(property, str):
            self.property = property

        # Store consumable code (None means that no consumables are required)
        if (cons_req_as_footprint is not None) and (cons_req_as_item_code is not None):
            raise ValueError('Consumable requirement was provided as both item code and footprint.')
        elif (cons_req_as_footprint is not None) and (cons_req_as_item_code is None):
            self.cons_req_as_footprint = cons_req_as_footprint
        elif (cons_req_as_footprint is None) and (cons_req_as_item_code is not None):
            self.cons_req_as_footprint = {
                'Intervention_Package_Code': {},
                'Item_Code': {cons_req_as_item_code: 1},
            }
        else:
            self.cons_req_as_footprint = None

        # Store performance characteristics (if sensitivity and specificity are not supplied than assume perfect)
        assert (sensitivity is None) or isinstance(sensitivity, float), 'Sensitivity is given in incorrect format.'
        assert (specificity is None) or isinstance(specificity, float), 'Sensitivity is given in incorrect format.'
        assert (measure_error_stdev is None) or isinstance(measure_error_stdev, float), \
            'measure_error_stdev is given in incorrect format.'
        assert (threshold is None) or isinstance(threshold, float), \
            'threshold is given in incorrect format.'

        if (sensitivity is not None):
            self.sensitivity = sensitivity
        else:
            self.sensitivity = 1.0

        if specificity is not None:
            self.specificity = specificity
        else:
            self.specificity = 1.0

        if measure_error_stdev is not None:
            self.measure_error_stdev = measure_error_stdev
        else:
            self.measure_error_stdev = 0.0

        self.threshold = threshold  # store the value of threshold (could be None or a float)

    def __hash__(self):
        if self.cons_req_as_footprint is not None:
            string_of_reqs = ''
            for t in self.cons_req_as_footprint.values():
                for k, v in t.items():
                    string_of_reqs = string_of_reqs + f'{k}_{v}'
            hash_of_cons_req_as_footprint = hash(string_of_reqs)
        else:
            hash_of_cons_req_as_footprint = hash(None)

        return hash((
            hash(self.property),
            hash_of_cons_req_as_footprint,
            hash(self.sensitivity),
            hash(self.specificity),
            hash(self.measure_error_stdev)
        ))

    def apply(self, hsi_event):
        """
        This is where the test is applied.
        If this test returns None this means the test has failed due to there not being the required consumables.

        :param hsi_event:
        :return: value of test or None.
        """

        health_system_module = hsi_event.module.sim.modules['HealthSystem']

        # Must be an individual level HSI and not a population level HSI
        assert not isinstance(hsi_event.target,
                              health_system_module.sim.population.__class__), \
            'HSI_Event is not individual level but it must be to use the DxManager'
        person_id = hsi_event.target

        # Get the "true value" of the property being examined
        df = health_system_module.sim.population.props
        assert self.property in df.columns, 'The property is not found in the sim.population.props dataframe'
        true_value = df.loc[person_id, self.property]

        # If a consumable is required and it is not available, return None
        if self.cons_req_as_footprint is not None:
            # check availability of consumable
            rtn_from_health_system = health_system_module.request_consumables(hsi_event,
                                                                              self.cons_req_as_footprint,
                                                                              to_log=True)

            if not (all(rtn_from_health_system['Intervention_Package_Code'].values()) and
                    all(rtn_from_health_system['Item_Code'].values())):
                return None

        # Apply the test:
        if isinstance(true_value, np.bool_):
            if true_value:
                # Apply the sensitivity:
                test_value = health_system_module.rng.choice([True, False], p=[self.sensitivity, 1 - self.sensitivity])
            else:
                # Apply the specificity:
                test_value = health_system_module.rng.choice([False, True], p=[self.specificity, 1 - self.specificity])

        elif isinstance(true_value, np.float_):
            # Apply the normally distributed zero-mean error
            reading = true_value + health_system_module.rng.normal(0.0, self.measure_error_stdev)

            # If no threshold value is provided, then return the reading; otherwise apply the threshold
            if self.threshold is None:
                test_value = float(reading)
            else:
                test_value = bool(reading >= self.threshold)

        else:
            test_value = true_value

        # Return test_value
        return test_value
