"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.

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
        self.healthsystem_module = healthsystem_module
        self.dx_test_hash = set()

    def register_dx_test(self, **kwargs):
        """
        Diagnostics tests are registerd in the DxManager by pass key-word arguements:
        e.g. register_dx_test(name_to_use_for_a_test = my_dx_test)
        MUST BE A LIST OF TESTS
        #TODO: make this into passing a dictionary in

        :param kwargs:
        :return:
        """
        for name, dx_test in kwargs.items():
            # Examine the proposed name of the dx_test
            assert isinstance(name, str), f'Name is not a string: {name}'
            assert name not in self.dx_tests, 'Test name already in use'

            # Examine the proposed dx_test
            # Make each test provided into a list of test
            if not isinstance(dx_test, list):
                dx_test = [dx_test]

            assert all([isinstance(d, DxTest) for d in dx_test]), f'Object is not a DxTest object: {d}'

            # Check if this is a duplicate dx_test
            hash_of_dx_test = hash(tuple([hash(d) for d in dx_test]))

            if hash_of_dx_test in self.dx_test_hash:
                logger.warning('This exact same dx_test has already been registered.')
            else:
                # Add the test the dict of registered dx_tests
                self.dx_tests.update({name: dx_test})
                self.dx_test_hash.add(hash_of_dx_test)

    def print_info_about_dx_test(self, name_of_dx_test):
        assert name_of_dx_test in self.dx_tests, f'This dx_test is not recognised: {name_of_dx_test}'
        the_dx_test = self.dx_tests[name_of_dx_test]
        print()
        print(f'----------------------')
        print(f'** {name_of_dx_test} **')
        for n, t in enumerate(the_dx_test):
            print(f'   Line #{n}')
            print(f'consumbales: {t.cons_req_as_footprint}')
            print(f'sensitivity: {t.sensitivity}')
            print(f'specificity: {t.specificity}')
            print(f'property: {t.property}')
            print(f'----------------------')

    def print_info_about_all_dx_tests(self):
        for dx_test in self.dx_tests:
            self.print_info_about_dx_test(dx_test)

    def run_dx_test(self, name_of_dx_test, hsi_event):

        # Make the name_of_dx_test into list if it is not already
        if not isinstance(name_of_dx_test, list):
            name_of_dx_test = list(name_of_dx_test)

        assert all(
            [name in self.dx_tests for name in name_of_dx_test]), f'This dx_test is not recognised: {name_of_dx_test}'

        result_for_alg = dict()

        for dx_test in name_of_dx_test:

            list_of_tests = self.dx_tests[dx_test]

            result_for_this_dx_test = None
            for test in list_of_tests:
                result_for_this_dx_test = test.apply(hsi_event, self.healthsystem_module)
                if result_for_this_dx_test is not None:
                    break
            result_for_alg[dx_test] = result_for_this_dx_test

        return result_for_alg


class DxTest:
    def __init__(self,
                 property: str,
                 cons_req_as_footprint=None,
                 cons_req_as_item_code=None,
                 sensitivity: float = None,
                 specificity: float = None
                 ):

        # Store the property on which it acts (This is the only required parameter)
        if property is not None:
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
        if sensitivity is not None:
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

        print('Apply the diagnostic test and return the result')

        # assuming its an individual level HSI (TODO)
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
