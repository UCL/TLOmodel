"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.

"""

class DxManager:
    """
    The is the Diagnostic Tests Manager (DxManager).
    It simplifies the process of conducting diagnostic tests on a person.
    It can store and then apply Diagnostic Tests (DxTest) and return the result.
    """

    def __init__(self, healthsystem_module):
        self.dx_tests = dict()
        self.healthsystem_module = healthsystem_module

    def register_dx_test(self, **kwargs):
        """
        Diagnostics tests are registerd in the DxManager by pass key-word arguements:
        e.g. register_dx_test(name_to_use_for_a_test = my_dx_test)
        MUST BE A LIST OF TESTS
        #TODO: make this into passing a dictionary in

        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            assert isinstance(key, str), f'Name is not a string: {key}'

            # Make each test provded into a list of test
            if not isinstance(value, list):
                value = [value]

            assert all([isinstance(v, DxTest) for v in value]), f'Object is not a DxTest object: {value}'

            self.dx_tests.update({key: value})

    def print_info_about_dx_test(self, name_of_dx_test):
        assert name_of_dx_test in self.dx_tests, f'This dx_test is not recognised: {name_of_dx_test}'
        the_dx_test = self.dx_tests[name_of_dx_test]
        print()
        print(f'----------------------')
        print(f'** {name_of_dx_test} **')
        print(f'consumbale_code: {the_dx_test.consumable_code}')
        print(f'sensitivity: {the_dx_test.sensitivity}')
        print(f'specificity: {the_dx_test.specificity}')
        print(f'property: {the_dx_test.property}')
        print(f'----------------------')

    def print_info_about_all_dx_tests(self):
        for dx_test in self.dx_tests:
            self.print_info_about_dx_test(dx_test)

    def run_dx_test(self, name_of_dx_test, hsi_event):

        assert name_of_dx_test in self.dx_tests, f'This dx_test is not recognised: {name_of_dx_test}'

        list_of_tests = self.dx_tests[name_of_dx_test]

        for test in list_of_tests:
            result = test.apply(hsi_event, self.healthsystem_module)
            if result is not None:
                break

        return result

class DxTest:
    def __init__(self,
                 property: str,
                 cons_req_as_footprint = None,
                 sensitivity: float = None,
                 specificity: float = None
                 ):

        # Store the property on which it acts (This is the only required parameter)
        if property is not None:
            self.property = property

        # Store consumable code (None means that no consumables are required)
        self.cons_req_as_footprint = cons_req_as_footprint


        # Store performance characteristics (if sensitivity and specifity are not supplied than assume perfect)
        if sensitivity is not None:
            self.sensitivity = sensitivity
        else:
            self.sensitivity = 1.0

        if specificity is not None:
            self.specificity = specificity
        else:
            self.specificity = 1.0


    def apply(self, hsi_event, health_system_module):

        print('Apply the diagnostic test and return the result')

        # assuming its an individual level HSI (TODO)
        person_id = hsi_event.target

        # Get the "true value" of the property being examined
        df = health_system_module.sim.population.props
        true_value = df.at[person_id, self.property]

        # Check for the availability of the consumable code
        if self.cons_req_as_footprint is not None:
            rtn_from_health_system = health_system_module.request_consumables(hsi_event, self.cons_req_as_footprint, to_log=True)

            cons_available = all(rtn_from_health_system['Intervention_Package_Code'].values()) \
                                                and all(rtn_from_health_system['Item_Code'].values())
        else:
            cons_available = True



        # Apply the test:
        test_value = true_value
        #TODO: insert logic about erroneous tests.

        if cons_available:
            # Consumables available, return test value
            return true_value
        else:
            # Consumabls not available, return test value
            return None



        # TODO: check that the symptom is in the SymtomManager
        # TODO: elaborate for continuous tests
