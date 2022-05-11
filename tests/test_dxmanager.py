import collections
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.consumables import Consumables, create_dummy_data_for_cons_availability
from tlo.methods.dxmanager import DxManager, DxTest
from tlo.methods.healthsystem import HSI_Event

# --------------------------------------------------------------------------
# Create a very short-run simulation for use in the tests
# --------------------------------------------------------------------------
# Create a very short-run simulation for use in the tests
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


@pytest.fixture
def bundle(seed):
    Bundle = collections.namedtuple('Bundle',
                                    ['simulation',
                                     'hsi_event',
                                     'item_code_for_consumable_that_is_not_available',
                                     'item_code_for_consumable_that_is_available'])
    # Establish the simulation object
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(
                     resourcefilepath=resourcefilepath,
                     disable_and_reject_all=True
                 ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),

                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=2000)
    sim.simulate(end_date=sim.start_date)

    # Create a dummy HSI event from which the use of diagnostics can be tested
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = '0'
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    hsi_event = HSI_Dummy(module=sim.modules['Mockitis'], person_id=0)
    hsi_event.initialise()
    # Force that the Facility_ID associated is facility_id=0 (as this is the facility for which availability of
    #  consumables is manipulated in the below).
    hsi_event.facility_info = sim.modules['HealthSystem']._facility_by_facility_id[0]

    # Update Consumables module with  consumable codes that are always and never available
    item_code_for_consumable_that_is_not_available = 0
    item_code_for_consumable_that_is_available = 1

    sim.modules['HealthSystem'].consumables = Consumables(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability={
                item_code_for_consumable_that_is_not_available: 0.0,
                item_code_for_consumable_that_is_available: 1.0},
            facility_ids=[0],
            months=[sim.date.month]),
        rng=sim.modules['HealthSystem'].rng,
        availability='default'
    )
    sim.modules['HealthSystem'].consumables.on_start_of_day(sim.date)

    assert hsi_event.get_consumables(item_codes=item_code_for_consumable_that_is_available)
    assert not hsi_event.get_consumables(item_codes=item_code_for_consumable_that_is_not_available)

    return Bundle(sim,
                  hsi_event,
                  item_code_for_consumable_that_is_not_available,
                  item_code_for_consumable_that_is_available)


# --------------------------------------------------------------------------


def test_create_dx_test():
    my_test = DxTest(
        property='mi_status'
    )

    assert isinstance(my_test, DxTest)

    # Check hash
    hashed = hash(my_test)
    assert isinstance(hashed, int)


def test_create_dx_test_and_register(bundle):
    sim = bundle.simulation

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

    # Try to register the same test again under exactly the same name: should not throw error but not add a duplicate
    dx_manager.register_dx_test(
        my_test1=my_test1,
    )
    dx_manager.print_info_about_all_dx_tests()
    assert 3 == len(dx_manager.dx_tests)

    # Register the same test under a different name: should be accepted
    # (tests of different names can be functionally identical).
    dx_manager.register_dx_test(my_test2_diff_name_should_not_be_added=my_test2)
    assert 4 == len(dx_manager.dx_tests)

    # Attempt to over-write a test: same name but different DxTest:
    #  -> should not add a test and raise an ValueError
    with pytest.raises(ValueError):
        dx_manager.register_dx_test(my_test1=DxTest(property='is_alive'))
    assert 4 == len(dx_manager.dx_tests)


def test_create_duplicate_test_that_should_be_allowed(bundle):
    sim = bundle.simulation
    item_code_for_consumable_that_is_available = \
        bundle.item_code_for_consumable_that_is_available

    my_test1_property_only = DxTest(
        property='mi_status'
    )

    my_test1_property_and_consumable = DxTest(
        property='mi_status',
        item_codes=item_code_for_consumable_that_is_available,
    )

    my_test1_property_and_consumable_and_sensspec = DxTest(
        property='mi_status',
        item_codes=item_code_for_consumable_that_is_available,
        sensitivity=0.99,
        specificity=0.95
    )

    # Give the same test but under a different name: only name provided - should fail and not add test
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_only,
        my_test1_copy=my_test1_property_only
    )
    assert len(dx_manager.dx_tests) == 2

    # Give the same test but under a different name: name and consumables provided - should fail and not add test
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_and_consumable,
        my_test1_copy=my_test1_property_and_consumable
    )
    assert len(dx_manager.dx_tests) == 2

    # Give the same test but under a different name: name and consumables provided and sens/spec provided:
    #       --- should fail and not add test
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_test1=my_test1_property_and_consumable_and_sensspec,
        my_test1_copy=my_test1_property_and_consumable_and_sensspec
    )
    assert len(dx_manager.dx_tests) == 2

    # Give duplicated list of tests under different name: only one should be added
    #       --- should throw error but add the one test
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_list_of_tests1=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only),
        my_list_of_tests1_copy=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only)
    )
    assert len(dx_manager.dx_tests) == 2

    # Give list of test that use the same test components but in different order: both should be added, no errors
    dx_manager = DxManager(sim.modules['HealthSystem'])  # get new DxManager
    dx_manager.register_dx_test(
        my_list_of_tests1=(my_test1_property_and_consumable_and_sensspec, my_test1_property_only),
        my_list_of_tests1_copy=(my_test1_property_only, my_test1_property_and_consumable_and_sensspec)
    )

    dx_manager.print_info_about_all_dx_tests()
    assert len(dx_manager.dx_tests) == 2


def test_create_dx_test_and_run(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

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


def test_create_dx_tests_with_consumable_useage_given_by_item_code_only(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event
    item_code_for_consumable_that_is_not_available = bundle.item_code_for_consumable_that_is_not_available
    item_code_for_consumable_that_is_available = bundle.item_code_for_consumable_that_is_available

    # Create the test:
    my_test1_not_available = DxTest(
        item_codes=item_code_for_consumable_that_is_not_available,
        property='mi_status'
    )

    my_test2_is_available = DxTest(
        item_codes=item_code_for_consumable_that_is_available,
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


def test_run_batch_of_dx_test_in_one_call(bundle):
    sim = bundle.simulation
    item_code_for_consumable_that_is_available = bundle.item_code_for_consumable_that_is_available
    hsi_event = bundle.hsi_event

    # Create the dx_test
    my_test1 = DxTest(
        item_codes=item_code_for_consumable_that_is_available,
        property='mi_status'
    )

    my_test2 = DxTest(
        item_codes=item_code_for_consumable_that_is_available,
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


def test_create_tuple_of_dx_tests_which_fail_and_require_chain_execution(bundle):
    sim = bundle.simulation
    item_code_for_consumable_that_is_not_available = bundle.item_code_for_consumable_that_is_not_available
    item_code_for_consumable_that_is_available = bundle.item_code_for_consumable_that_is_available
    hsi_event = bundle.hsi_event

    # Create the tests:
    my_test1_not_available = DxTest(
        item_codes=item_code_for_consumable_that_is_not_available,
        property='mi_status'
    )

    my_test2_is_available = DxTest(
        item_codes=item_code_for_consumable_that_is_available,
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
    assert tests_tried['tuple_of_tests_with_first_not_available_0'] is False
    assert tests_tried['tuple_of_tests_with_first_not_available_1'] is True

    # Run the tuple of tests (when the first test fails): should return a result having run test2 and not tried test1
    result, tests_tried = dx_manager.run_dx_test(
        dx_tests_to_run='tuple_of_tests_with_first_available',
        hsi_event=hsi_event,
        report_dxtest_tried=True
    )
    assert result is not None
    assert result, tests_tried == sim.population.props.at[person_id, 'mi_status']
    assert len(tests_tried) == 1
    assert tests_tried['tuple_of_tests_with_first_available_0'] is True
    assert 'tuple_of_tests_with_first_available_1' not in tests_tried


def test_create_dx_test_and_run_with_imperfect_sensitivity(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

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


def test_create_dx_test_and_run_with_bool_dx_and_imperfect_specificity(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

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


def test_create_dx_test_and_run_with_cont_value_and_cutoff(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

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


def test_create_dx_test_and_run_with_cont_dx_and_error(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

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


def test_dx_with_categorial(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

    df = sim.population.props
    df['CategoricalProperty'] = pd.Series(
        data=sim.rng.choice(['level0', 'level1', 'level2'], len(df), [0.1, 0.1, 0.8]),
        dtype="category"
    )

    # Create the test - with no error:
    my_test = DxTest(
        property='CategoricalProperty',
        target_categories=['level2']
    )

    # Create the test - with no sensitivity:
    my_test_w_no_sens = DxTest(
        property='CategoricalProperty',
        target_categories=['level2'],
        sensitivity=0.0
    )

    # Create the test - with no specificity:
    my_test_w_no_spec = DxTest(
        property='CategoricalProperty',
        target_categories=['level2'],
        specificity=0.0
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register DxTest with DxManager:
    dx_manager.register_dx_test(my_test=my_test,
                                my_test_w_no_sens=my_test_w_no_sens,
                                my_test_w_no_spec=my_test_w_no_spec
                                )

    # Run it and check the result:
    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id

        # Test with perfect sensitivity and specificity
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test',
            hsi_event=hsi_event,
        )
        assert result_from_dx_manager == (df.at[person_id, 'CategoricalProperty'] == 'level2')

        # Test with no sensitivity: will never detect the category when it is correct
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test_w_no_sens',
            hsi_event=hsi_event,
        )
        assert result_from_dx_manager is False

        # Test with no specificity: will always detect the category if it not correct
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test_w_no_spec',
            hsi_event=hsi_event,
        )
        assert result_from_dx_manager is True


def test_dx_with_categorial_multiple_levels_accepted(bundle):
    sim = bundle.simulation
    hsi_event = bundle.hsi_event

    df = sim.population.props
    df['CategoricalProperty'] = pd.Series(
        data=sim.rng.choice(['level0', 'level1', 'level2'], len(df), [0.3, 0.3, 0.4]),
        dtype="category"
    )

    # Create the test - with no error:
    my_test = DxTest(
        property='CategoricalProperty',
        target_categories=['level2', 'level0']
    )

    # Create new DxManager
    dx_manager = DxManager(sim.modules['HealthSystem'])

    # Register DxTest with DxManager:
    dx_manager.register_dx_test(my_test=my_test,
                                )
    # Run it and check the result:
    for person_id in df.loc[df.is_alive].index:
        hsi_event.target = person_id

        # Test with perfect sensitivity and specificity
        result_from_dx_manager = dx_manager.run_dx_test(
            dx_tests_to_run='my_test',
            hsi_event=hsi_event,
        )
        assert result_from_dx_manager == (df.at[person_id, 'CategoricalProperty'] == 'level2') or (
                df.at[person_id, 'CategoricalProperty'] == 'level0')
