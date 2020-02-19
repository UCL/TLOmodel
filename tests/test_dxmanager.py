import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    mockitis,
    symptommanager,
)
from tlo.methods.dxmanager import DxManager, DxTest
from tlo.methods.healthsystem import HSI_Event

# --------------------------------------------------------------------------
# Create a very short-run simulation for use in the tests
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

# Establish the simulation object
sim = Simulation(start_date=Date(year=2010, month=1, day=1))

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(labour.Labour(resourcefilepath=resourcefilepath))
sim.register(mockitis.Mockitis())
sim.register(chronicsyndrome.ChronicSyndrome())

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=2000)
sim.simulate(end_date=Date(year=2010, month=1, day=31))


# Create a dummy HSI event from which the use of diagnostics can be tested
class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'Dummy'
        self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        pass


hsi_event = HSI_Dummy(module=sim.modules['Mockitis'], person_id=-99)

# Create consumable codes that are always and never available
cons = sim.modules['HealthSystem'].cons_item_code_availability_today
item_code_for_consumable_that_is_not_available = 0
item_code_for_consumable_that_is_available = 1

cons.loc[item_code_for_consumable_that_is_not_available, cons.columns] = False
cons.loc[item_code_for_consumable_that_is_available, cons.columns] = True

assert sim.modules['HealthSystem'].cons_item_code_availability_today.loc[
    item_code_for_consumable_that_is_available].all()
assert not sim.modules['HealthSystem'].cons_item_code_availability_today.loc[
    item_code_for_consumable_that_is_not_available].any()

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

    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager

    dx_manager.register_dx_test(
        my_test1=my_test1,
        my_test2=my_test2
    )

    dx_manager.register_dx_test(
        my_tuple_of_tests=(my_test1, my_test2)
    )

    # try to add the same test again under exactly the same name- should not throw error but not add another
    dx_manager.register_dx_test(
        my_test1=my_test1,
    )

    dx_manager.print_info_about_all_dx_tests()
    assert 3 == len(dx_manager.dx_tests)

    # Create duplicate of a test with a different name and same DxTest: should fail and not add a test
    try:
        dx_manager.register_dx_test(my_test2_diff_name_should_not_be_added=my_test2)
    except ValueError:
        pass
    assert 3 == len(dx_manager.dx_tests)

    # Create a duplicate of test: same name and different DxTest: should fail and not add a test
    try:
        dx_manager.register_dx_test(my_test1=DxTest(property='is_alive'))
    except ValueError:
        pass
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

    # Give the same test but under a different name: only name provided - should fail and not add test
    try:
        dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
        dx_manager.register_dx_test(
            my_test1=my_test1_property_only,
            my_test1_copy=my_test1_property_only
        )
    except ValueError:
        pass
    assert len(dx_manager.dx_tests) == 1

    # Give the same test but under a different name: name and consumbales provided - should fail and not add test
    try:
        dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
        dx_manager.register_dx_test(
            my_test1=my_test1_property_and_consumable,
            my_test1_copy=my_test1_property_and_consumable
        )
    except ValueError:
        pass
    assert len(dx_manager.dx_tests) == 1

    # Give the same test but under a different name: name and consumbales provided and sens/spec provided:
    #       --- should fail and not add test
    try:
        dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
        dx_manager.register_dx_test(
            my_test1=my_test1_property_and_consumable_and_sensspec,
            my_test1_copy=my_test1_property_and_consumable_and_sensspec
        )
    except ValueError:
        pass
    assert len(dx_manager.dx_tests) == 1

    # Give duplicated list of tests under different name: only one should be added
    #       --- should throw error but add the one test
    try:
        dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
        dx_manager.register_dx_test(
            my_list_of_tests1=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only),
            my_list_of_tests1_copy=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only)
        )
    except ValueError:
        pass
    assert len(dx_manager.dx_tests) == 1

    # Give list of test that use the same test components but in different order: both should be added, no errors
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_list_of_tests1=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only),
        my_list_of_tests1_copy=(my_test1_property_only, my_test1_property_and_consumable_and_sensspec)
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
    dx_manager.register_dx_test(my_test1=my_test1)

    # Run it and check the result:
    df = sim.population.props
    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test1',
            hsi_event=hsi_event,
        )
        assert result_from_dx_manager == df.at[person_id, 'mi_status']

    # Run it and check the result - getting a dict returned rather than a single value
    df = sim.population.props
    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test1',
            hsi_event=hsi_event,
            use_dict_for_single=True
        )
        assert isinstance(result_from_dx_manager, dict)
        assert result_from_dx_manager['my_test1'] == df.at[person_id, 'mi_status']


def test_create_dx_tests_with_consumable_usage():
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
    dx_manager.register_dx_test(
        my_test1=my_test1_not_available,
        my_test2=my_test2_is_available,
        my_test3=my_test3_is_no_consumable_needed,
    )

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    # Confirm the my_test1 does not give result
    assert None is dx_manager.run_dx_test(dx_tests_to_run='my_test1',
                                          hsi_event=hsi_event,
                                          )

    # Confirm that my_test2 and my_test3 does give result
    assert None is not dx_manager.run_dx_test(dx_tests_to_run='my_test2',
                                              hsi_event=hsi_event,
                                              )

    assert None is not dx_manager.run_dx_test(dx_tests_to_run='my_test3',
                                              hsi_event=hsi_event,
                                              )


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
    dx_manager.register_dx_test(
        my_test1=my_test1_not_available,
        my_test2=my_test2_is_available,
    )

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    # Confirm the my_test1 does not give result
    assert None is dx_manager.run_dx_test(dx_tests_to_run='my_test1',
                                          hsi_event=hsi_event,
                                          )

    # Confirm that my_test2 does give result
    assert None is not dx_manager.run_dx_test(dx_tests_to_run='my_test2',
                                              hsi_event=hsi_event,
                                              )


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


def test_run_batch_of_dx_test_in_one_call():
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

    result = dx_manager.run_dx_test(hsi_event=hsi_event, dx_tests_to_run=['my_test1', 'my_test2'])

    assert isinstance(result, dict)
    assert list(result.keys()) == ['my_test1', 'my_test2']

    df = sim.population.props
    assert result['my_test1'] == df.at[person_id, 'mi_status']
    assert result['my_test2'] == df.at[person_id, 'cs_has_cs']


def test_create_tuple_of_dx_tests_which_fail_and_require_chain_execution():
    # Create the tests:
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

    # Register tuple of tests with DxManager:
    dx_manager.register_dx_test(single_test_not_available=my_test1_not_available)

    dx_manager.register_dx_test(tuple_of_tests_with_first_not_available=(
        my_test1_not_available,
        my_test2_is_available
    ))

    dx_manager.register_dx_test(tuple_of_tests_with_first_available=(
        my_test2_is_available,
        my_test1_not_available
    ))

    # pick a person
    person_id = 0
    hsi_event.target = person_id

    # Run the single test which requires a consumable that is not available: should not return result
    result = dx_manager.run_dx_test(
        dx_tests_to_run='single_test_not_available',
        hsi_event=hsi_event
    )
    assert result is None

    # Run the tuple of test (when the first test fails): should return a result having run test1 and test2
    result, tests_tried = dx_manager.run_dx_test(
        dx_tests_to_run='tuple_of_tests_with_first_not_available',
        hsi_event=hsi_event,
        report_dxtest_tried=True
    )
    assert result is not None
    assert result == sim.population.props.at[person_id, 'mi_status']
    assert len(tests_tried) == 2
    assert tests_tried[my_test1_not_available] is False
    assert tests_tried[my_test2_is_available] is True

    # Run the tuple of tests (when the first test fails): should return a result having run test2 and not tried test1
    result, tests_tried = dx_manager.run_dx_test(
        dx_tests_to_run='tuple_of_tests_with_first_available',
        hsi_event=hsi_event,
        report_dxtest_tried=True
    )
    assert result is not None
    assert result, tests_tried == sim.population.props.at[person_id, 'mi_status']
    assert len(tests_tried) == 1
    assert tests_tried[my_test2_is_available] is True
    assert my_test1_not_available not in tests_tried


def test_create_dx_test_and_run_with_imperfect_sensitivity():
    # Create a property in the sim.population.props dataframe for testing
    df = sim.population.props
    df['AllTrue'] = True

    # Create the tests and determine performance
    def run_test_sensitivity_and_specificty(sens=1.0, spec=1.0):
        my_test = DxTest(
            property='AllTrue',
            sensitivity=sens,
            specificity=spec
        )

        # Register DxTest with DxManager:
        dx_manager = DxManager(sim.modules['HealthSystem'])
        dx_manager.register_dx_test(my_test=my_test)

        # Run it on all people and get a list of the results
        results = list()
        for person_id in df.index:
            hsi_event.target = person_id
            result_from_dx_manager = dx_manager.run_dx_test(
                dx_tests_to_run='my_test',
                hsi_event=hsi_event
            )
            results.append(result_from_dx_manager)

        return results

    # Test sensitivity:
    # 0% sensitivity and perfect specificity: no positive results
    results = run_test_sensitivity_and_specificty(sens=0.0)
    assert sum(results) == 0.0

    # 100% sensitivity and perfect specificity: all positive results
    results = run_test_sensitivity_and_specificty(sens=1.0)
    assert sum(results) == len(results)

    # 50% sensitivity and perfect specificity: some (but not all) positive results
    results = run_test_sensitivity_and_specificty(sens=0.5)
    assert 0.0 < sum(results) < len(results)


def test_create_dx_test_and_run_with_bool_dx_and_imperfect_specificity():
    # Create a property in the sim.population.props dataframe for testing
    df = sim.population.props
    df['AllFalse'] = False

    # Create the tests and determine performance
    def run_test_sensitivity_and_specificty(sens=1.0, spec=1.0):
        my_test = DxTest(
            property='AllFalse',
            sensitivity=sens,
            specificity=spec
        )

        # Register DxTest with DxManager:
        dx_manager = DxManager(sim.modules['HealthSystem'])
        dx_manager.register_dx_test(my_test=my_test)

        # Run it on all people and get a list of the results
        results = list()
        for person_id in df.index:
            hsi_event.target = person_id
            result_from_dx_manager = dx_manager.run_dx_test(
                dx_tests_to_run='my_test',
                hsi_event=hsi_event
            )

            results.append(result_from_dx_manager)

        return results

    # Test specificity:
    # 100% specificity: no positive results
    results = run_test_sensitivity_and_specificty(spec=1.0)
    assert sum(results) == 0.0

    # 0% specificity: all positive results
    results = run_test_sensitivity_and_specificty(spec=0.0)
    assert sum(results) == len(results)

    # 50% sensitivity and perfect specificity: some (but not all) positive results
    results = run_test_sensitivity_and_specificty(spec=0.5)
    assert 0.0 < sum(results) < len(results)


def test_create_dx_test_and_run_with_cont_value_and_cutoff():
    # This test is for a property represented by a continuous variable and the DxTest should return True if the value
    # is above a certain threshold.

    # Create a dataframe that has values known to be above or below the threshold (=0.0)
    df = sim.population.props
    df['AboveThreshold'] = 1.0
    df['BelowThreshold'] = -1.0

    my_test_on_above_threshold = DxTest(
        property='AboveThreshold',
        threshold=0.0
    )

    my_test_on_below_threshold = DxTest(
        property='BelowThreshold',
        threshold=0.0
    )

    # Register DxTest with DxManager:
    dx_manager = DxManager(sim.modules['HealthSystem'])
    dx_manager.register_dx_test(my_test_on_above_threshold=my_test_on_above_threshold,
                                my_test_on_below_threshold=my_test_on_below_threshold
                                )

    # Run it on one person and confirm the result is as expected
    person_id = 0
    hsi_event.target = person_id
    result = dx_manager.run_dx_test(
        dx_tests_to_run='my_test_on_above_threshold',
        hsi_event=hsi_event
    )
    assert result is True

    result = dx_manager.run_dx_test(
        dx_tests_to_run='my_test_on_below_threshold',
        hsi_event=hsi_event
    )
    assert result is False


def test_create_dx_test_and_run_with_cont_dx_and_error():
    # Create a property in the sim.population.props dataframe for testing.
    # Make it zero for everyone
    df = sim.population.props
    df['AllZero'] = 0.0

    # Create the tests:
    my_test_zero_stdev = DxTest(
        property='AllZero',
        measure_error_stdev=0.0
    )

    my_test_nonzero_stdev = DxTest(
        property='AllZero',
        measure_error_stdev=1.0
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register DxTest with DxManager:
    dx_manager.register_dx_test(
        my_test_zero_stdev=my_test_zero_stdev,
        my_test_nonzero_stdev=my_test_nonzero_stdev,
    )

    # Run each test and check the result:
    df = sim.population.props
    result_from_test_with_zero_stdev = list()
    result_from_test_with_nonzero_stdev = list()

    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run=['my_test_zero_stdev', 'my_test_nonzero_stdev'],
            hsi_event=hsi_event,
        )

        result_from_test_with_zero_stdev.append(result_from_dx_manager['my_test_zero_stdev'])
        result_from_test_with_nonzero_stdev.append(result_from_dx_manager['my_test_nonzero_stdev'])

    assert all([0.0 == e for e in result_from_test_with_zero_stdev])
    assert sum([abs(e) for e in result_from_test_with_nonzero_stdev]) > 0
