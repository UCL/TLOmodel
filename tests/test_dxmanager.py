
import pytest
import numpy as np
import os
from tlo.events import IndividualScopeEventMixin
from tlo.methods.dxmanager import DxManager, DxTest
from pathlib import Path
from tlo import Date, Simulation
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    symptommanager,
)
from tlo.methods.healthsystem import HSI_Event


# --------------------------------------------------------------------------
# Create a simulation object for next batch of tests
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=2010, month=1, day=31)
popsize = 200

# Establish the simulation object
sim = Simulation(start_date=start_date)

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Create a dummy HSI event from which the use of diagnostics can be tested
class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Dummy'
        self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        pass

hsi_event = HSI_Dummy(module = sim.modules['Mockitis'], person_id=-99)


# Create consumable codes that are always and never available
cons = sim.modules['HealthSystem'].cons_item_code_availability_today
item_code_for_consumable_that_is_not_available = 0
item_code_for_consumable_that_is_available = 1

cons.loc[item_code_for_consumable_that_is_not_available, cons.columns] = False
cons.loc[item_code_for_consumable_that_is_available, cons.columns] = True

assert sim.modules['HealthSystem'].cons_item_code_availability_today.loc[item_code_for_consumable_that_is_available ].all()
assert not sim.modules['HealthSystem'].cons_item_code_availability_today.loc[item_code_for_consumable_that_is_not_available ].any()

cons_req_as_footprint_for_consumable_that_is_not_available = consumables_needed = {
                                                    'Intervention_Package_Code': {},
                                                    'Item_Code': {item_code_for_consumable_that_is_not_available: 1},
                                                }

cons_req_as_footprint_for_consumable_that_is_available = consumables_needed = {
                                                    'Intervention_Package_Code': {},
                                                    'Item_Code': {item_code_for_consumable_that_is_available: 1},
                                                }

# --------------------------------------------------------------------------


def test_create_dx_test():
    my_test = DxTest(
        property='mi_status'
    )

    assert isinstance(my_test, DxTest)

    # Check hash
    hashed = hash(my_test)
    assert isinstance(hashed, int)

def test_create_dx_test_and_register():
    my_test1 = DxTest(
        property='mi_status'
    )

    my_test2 = DxTest(
        property='cs_status'
    )

    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager

    dx_manager.register_dx_test(
        my_test1=my_test1,
        my_test2=my_test2
    )

    dx_manager.register_dx_test(
        my_compound_test=[my_test1, my_test2]
    )

    dx_manager.register_dx_test(
        my_test2_diff_name_should_not_be_added=my_test2
    )

    dx_manager.print_info_about_all_dx_tests()
    assert 3 == len(dx_manager.dx_tests)

def test_create_duplicate_test_that_should_be_ignored():
    my_test1_property_only = DxTest(
        property='mi_status'
    )

    my_test1_property_and_consumable = DxTest(
        property='mi_status',
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
    )

    my_test1_property_and_consumable_and_sensspec = DxTest(
        property='mi_status',
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        sensitivity=0.99,
        specificity=0.95
    )

    # Give the same test but under a different name: name only
    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_only,
        my_test1_copy=my_test1_property_only
    )
    assert len(dx_manager.dx_tests) == 1

    # Give the same test but under a different name: name and consumbales provided
    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_and_consumable,
        my_test1_copy=my_test1_property_and_consumable
    )
    assert len(dx_manager.dx_tests) == 1

    # Give the same test but under a different name: name and consumbales provided and sens/spec provided
    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_and_consumable_and_sensspec,
        my_test1_copy=my_test1_property_and_consumable_and_sensspec
    )
    assert len(dx_manager.dx_tests) == 1

    # Give duplicated compound tests under different name: only one should be added
    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager
    dx_manager.register_dx_test(
        my_compound_test1=[my_test1_property_and_consumable_and_sensspec, my_test1_property_only],
        my_compound_test1_copy=[my_test1_property_and_consumable_and_sensspec, my_test1_property_only]
    )
    assert len(dx_manager.dx_tests) == 1

    # Give compound test that use the same test components but in different order: both should be added
    dx_manager = DxManager(sim.modules['HealthSystem']) # get new DxManager
    dx_manager.register_dx_test(
        my_compound_test1=[my_test1_property_and_consumable_and_sensspec, my_test1_property_only],
        my_compound_test1_copy=[my_test1_property_only, my_test1_property_and_consumable_and_sensspec]
    )
    dx_manager.print_info_about_all_dx_tests()
    assert len(dx_manager.dx_tests) == 2


def test_create_dx_test_and_run():

    # Create the test:
    my_test1 = DxTest(
        property='mi_status'
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register DxTest with DxManager:
    dx_manager.register_dx_test(my_test1 = my_test1)

    # Run it and check the result:
    df = sim.population.props
    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id
        result_for_dx_manager = sim.modules['HealthSystem'].dx_manager.run_dx_test(name_of_dx_test='my_test1',
                                                                                   hsi_event=hsi_event)
        assert result_for_dx_manager == df.at[person_id, 'mi_status']


def test_create_dx_tests_with_consumable_useage():

    # Create the test:
    my_test1_not_available = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_not_available,
        property='mi_status'
    )

    my_test2_is_available = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        property='mi_status'
    )

    my_test3_is_no_consumable_needed = DxTest(
        cons_req_as_footprint=None,  # No consumable code: means that the consumable will not be
        property='mi_status'
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register the single and the compound tests with DxManager:
    sim.modules['HealthSystem'].dx_manager.register_dx_test(
        my_test1=my_test1_not_available,
        my_test2=my_test2_is_available,
        my_test3=my_test3_is_no_consumable_needed,

    )

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    # Confirm the my_test1 does not give result
    assert None is dx_manager.run_dx_test(name_of_dx_test='my_test1',
                                                                      hsi_event=hsi_event)

    # Confirm that my_test2 and my_test3 does give result
    assert None is not dx_manager.run_dx_test(name_of_dx_test='my_test2',
                                                                          hsi_event=hsi_event)

    assert None is not dx_manager.run_dx_test(name_of_dx_test='my_test3',
                                                                          hsi_event=hsi_event)


def test_create_dx_tests_with_consumable_useage_given_by_item_code_only():

    # Create the test:
    my_test1_not_available = DxTest(
        cons_req_as_item_code=item_code_for_consumable_that_is_not_available,
        property='mi_status'
    )

    my_test2_is_available = DxTest(
        cons_req_as_item_code=item_code_for_consumable_that_is_available,
        property='mi_status'
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register the single and the compound tests with DxManager:
    sim.modules['HealthSystem'].dx_manager.register_dx_test(
        my_test1=my_test1_not_available,
        my_test2=my_test2_is_available,
    )

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    # Confirm the my_test1 does not give result
    assert None is dx_manager.run_dx_test(name_of_dx_test='my_test1',
                                                                      hsi_event=hsi_event)

    # Confirm that my_test2 does give result
    assert None is not dx_manager.run_dx_test(name_of_dx_test='my_test2',
                                                                          hsi_event=hsi_event)

def test_hash_from_footprint_and_hash_from_item_code():
    my_test_using_item_code = DxTest(
        cons_req_as_item_code=item_code_for_consumable_that_is_available,
        property='mi_status'
    )

    my_test_using_footprint = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        property='mi_status'
    )

    assert hash(my_test_using_item_code) == hash(my_test_using_footprint)

def test_run_batch_of_dx_test():

    # Create the dx_test
    my_test1 = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        property='mi_status'
    )

    my_test2 = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        property='cs_has_cs'
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register compound tests with DxManager:
    dx_manager.register_dx_test(my_test1=my_test1, my_test2=my_test2)

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    result = dx_manager.run_dx_test(hsi_event=hsi_event, name_of_dx_test=['my_test1','my_test2'])

    assert isinstance(result, dict)
    assert list(result.keys()) == ['my_test1','my_test2']

    df = sim.population.props
    assert result['my_test1'] == df.at[person_id, 'mi_status']
    assert result['my_test2'] == df.at[person_id, 'cs_has_cs']


    # TODO: ***** Start wiki and sort out naming conventions and commenting *****
    # DxTest - individual unit: one test, one consumable and one result
    # DxProcedure - composed of one of more 'DxTest's
    # Batch - composed of one of more DxProcedures


def test_create_set_of_dx_tests_which_fail_and_require_chain_execution():

    # Create the test:
    my_test1_not_available = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_not_available,
        property='mi_status'
    )

    my_test2_is_available = DxTest(
        cons_req_as_footprint=cons_req_as_footprint_for_consumable_that_is_available,
        property='mi_status'
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register compound tests with DxManager:
    dx_manager.register_dx_test(single_test_not_available=
                                                            [
                                                                my_test1_not_available
                                                            ])

    dx_manager.register_dx_test(compound_test_with_first_not_available=
                                                            [
                                                                my_test1_not_available,
                                                                my_test2_is_available,
                                                            ])

    dx_manager.register_dx_test(compound_test_with_first_available=
                                                            [
                                                                my_test2_is_available,
                                                                my_test1_not_available,
                                                            ])

    # pick a person
    person_id = 0
    hsi_event.target = person_id


    # Run the non-compound test with a test that is not available: should not return result
    result = dx_manager.run_dx_test(
        name_of_dx_test='single_test_not_available',
        hsi_event=hsi_event)
    assert result is None

    # Run the compound test (when the first test fails): should return a result having run test2
    result = sim.modules['HealthSystem'].dx_manager.run_dx_test(name_of_dx_test='compound_test_with_first_not_available',
                                                                hsi_event=hsi_event)
    assert None is not result
    assert result == sim.population.props.at[person_id, 'mi_status']

    # Run the compound test (when the first test fails): should return a result having run test2 and not tried test1
    result = dx_manager.run_dx_test(name_of_dx_test='compound_test_with_first_available',
                                                                hsi_event=hsi_event)
    assert None is not result
    assert result == sim.population.props.at[person_id, 'mi_status']
    # TODO: report which test provided the result
    # TODO; check that a second test is not run if the first proviedes result.
