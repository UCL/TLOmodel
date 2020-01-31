"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.
See https://github.com/UCL/TLOmodel/wiki/Diagnostic-Tests-(DxTest)-and-the-Diagnostic-Tests-Manager-(DxManager)
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxManager:
    """
    The is the Diagnostic Tests Manager (DxManager).
    It simplifies the process of conducting diagnostic tests on a person.
    It can store and then apply Diagnostic Tests (DxTest) and return the result.
    """

    def __init__(self, healthsystem_module):
        self.dx_tests = dict()
        self.dx_test_hash = set()

        # Check that the HealthSystem module has passed itself correctly
        assert 'HealthSystem' == healthsystem_module.name
        self.healthsystem_module = healthsystem_module

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
            assert all([isinstance(d, DxTest) for d in dx_test]), f'Object is not a DxTest object: {d}'

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

    def run_dx_test(self, dx_tests_to_run, hsi_event, use_dict_for_single=True):

        # Make dx_tests_to_run into a list if it is not already one
        if not isinstance(dx_tests_to_run, list):
            dx_tests_to_run = [dx_tests_to_run]

        assert all(
            [name in self.dx_tests for name in dx_tests_to_run]
        ), f'A DxTest name is not recognised.'

        # Create the dict() that will be returned
        result_dict_for_list_of_dx_tests = dict()

        for dx_test in dx_tests_to_run:

            # Loop through the list of DxTests that are registered under this name:
            for t in self.dx_tests[dx_test]:
                t_res = t.apply(hsi_event, self.healthsystem_module)
                if t_res is not None:
                    break

            result_dict_for_list_of_dx_tests[dx_test] = t_res

        # Decide on type of return:
        if (len(dx_tests_to_run)==1) and use_dict_for_single:
            return result_dict_for_list_of_dx_tests[dx_tests_to_run[0]]
        else:
            return result_dict_for_list_of_dx_tests


class DxTest:
    def __init__(self,
                 property: str,
                 cons_req_as_footprint=None,
                 cons_req_as_item_code=None,
                 sensitivity: float = None,
                 specificity: float = None
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

        # TODO; format checking on consumable footprint

        # Store performance characteristics (if sensitivity and specificity are not supplied than assume perfect)
        assert (sensitivity is None) or isinstance(sensitivity, float), 'Sensitivity is given in incorrect format.'
        assert (specificity is None) or isinstance(specificity, float), 'Sensitivity is given in incorrect format.'

        if (sensitivity is not None):
            self.sensitivity = sensitivity
        else:
            self.sensitivity = 1.0

        if specificity is not None:
            self.specificity = specificity
        else:
            self.specificity = 1.0

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
            hash(self.specificity)
        ))

    def apply(self, hsi_event, health_system_module):
        """
        This is where the test is applied.
        If this test returns None this means the test has failed due to there not being the required consumables.

        :param hsi_event:
        :param health_system_module:
        :return: value of test or None.
        """
        print('Apply the diagnostic test and return the result')

        # Must be an individual level HSI and not a population level HSI
        assert not isinstance(hsi_event.target, health_system_module.sim.population.__class__), 'HSI_Event is not individual level but it must be to use the DxManager'
        person_id = hsi_event.target

        # Get the "true value" of the property being examined
        df = health_system_module.sim.population.props
        true_value = df.at[person_id, self.property]

        # Check for the availability of the consumable code
        if self.cons_req_as_footprint is not None:
            rtn_from_health_system = health_system_module.request_consumables(hsi_event, self.cons_req_as_footprint,
                                                                              to_log=True)

            cons_available = all(rtn_from_health_system['Intervention_Package_Code'].values()) \
                             and all(rtn_from_health_system['Item_Code'].values())
        else:
            cons_available = True

        # Apply the test:
        test_value = true_value
        # TODO: insert logic about erroneous tests.

        if cons_available:
            # Consumables available, return test value
            return test_value
        else:
            # Consumables not available, return test value
            return None

        # TODO: elaborate for continuous tests
