"""
The is the Diagnostic Tests Manager (DxManager). It simplifies the process of conducting diagnostic tests on a person.

"""
import logging

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class DxManager:
    """
    The is the Diagnostic Tests Manager (DxManager).
    It simplifies the process of conducting diagnostic tests on a person.
    It can store and then apply Diagnostic Tests (DxTest) and return the result
    """

    def __init__(self):
        self.dx_tests = dict()
        print()
        print('DxManager initiated')
        #TODO: should accept a rng

        pass

    def register_dx_test(self, **kwargs):
        """
        Diagnostics tests are registerd in the DxManager by pass key-word arguements:
        e.g. register_dx_test(name_to_use_for_a_test = my_dx_test)

        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            assert isinstance(key, str), f'Name is not a string: {key}'
            assert isinstance(value, DxTest), f'Object is not a DxTest object: {value}'
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


    def run_dx_test(self, name_of_dx_test: str, rng_of_module):

        assert name_of_dx_test in self.dx_tests, f'This dx_test is not recognised: {name_of_dx_test}'
        # TODO: assert isinstance(rng_of_module, correct type)

        # TODO: elaborate to include more than one test

        self.dx_tests[name_of_dx_test].apply(rng_of_module)


class DxTest:
    def __init__(self,
                 consumable_code: int,
                 sensitivity: float,
                 specificity: float,
                 property: str,
                 consumable_name: str = None):

        # Store consumable code
        if consumable_code is not None:

            #TODO: check that consumable code is sensible
            self.consumable_code = consumable_code

        elif consumable_name is not None:

            #TODO: look up the consumable and generate the code
            self.consumable_code = -99

        else:
            raise Exception("Neither a consumable code or a consumable name is specified")


        # Store performance characteristics
        if sensitivity is not None:
            self.sensitivity = sensitivity

        if specificity is not None:
            self.specificity = specificity


        # Store the property on which it acts:
        if property is not None:
            #TODO: check that the property is in the df.
            self.property = property

        # # Store the symptom on which it acts:
        # if symptom_string is not None:
        #     #TODO: check that the symptom is in the SymtomManager
        #     self.symptom_string = symptom_string

        # TODO: elaborate for continuous tests

    def apply(self, rng):
        print('Apply the diagnostic test and return the result')
        return rng.choice([True, False], p=0.5)
        pass
