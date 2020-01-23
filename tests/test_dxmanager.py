
import pytest
import numpy as np
from tlo.methods.dxmanager import DxManager, DxTest


def test_create_dx_test():

    my_test = DxTest(
        consumable_code=100,
        sensitivity=0.98,
        specificity=0.95,
        property='mi_status'
    )

    print(f'my_test: consumable code = {my_test.consumable_code}')
    print(f'my_test: sensitivity = {my_test.sensitivity}')
    print(f'my_test: specificity = {my_test.specificity}')
    print(f'my_test: property = {my_test.property}')



def test_create_dx_test_and_register():

    my_test1 = DxTest(
        consumable_code=100,
        sensitivity=0.98,
        specificity=0.95,
        property='mi_status'
    )

    my_test2 = DxTest(
        consumable_code=200,
        sensitivity=0.50,
        specificity=0.26,
        property='cs_status'
    )

    my_dx_manager = DxManager()

    my_dx_manager.register_dx_test(
        my_test1=my_test1,
        my_test2=my_test2
    )

    my_dx_manager.print_info_about_all_dx_tests()


def test_create_dx_test_and_run():

    #todo
    pass


