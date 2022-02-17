import datetime
import os
from pathlib import Path

import numpy
import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.consumables import (
    Consumables,
    create_dummy_data_for_cons_availability,
    get_item_code_from_item_name,
    get_item_codes_from_package_name,
)
from tlo.methods.healthsystem import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)


def find_level_of_facility_id(facility_id: int) -> str:
    """Returns the level of a Facility_ID"""
    return mfl.set_index('Facility_ID').loc[facility_id].Facility_Level


def any_warnings_about_item_code(recorded_warnings):
    """Helper function to determine if any of the recorded warnings is the one created when an Item_Code is not
    recognised."""
    return len([_r for _r in recorded_warnings if str(_r.message).startswith('Item_Code')]) > 0


def test_using_recognised_item_codes():
    """Test the functionality of the `Consumables` class with a recognising item_code."""
    # Prepare inputs for the Consumables class (normally provided by the `HealthSystem` module).
    data = create_dummy_data_for_cons_availability(
        intrinsic_availability={0: 0.0, 1: 1.0},
        months=[1],
        facility_ids=[0])
    rng = numpy.random
    date = datetime.datetime(2010, 1, 1)

    # Initiate Consumables class
    cons = Consumables(data=data, rng=rng)

    # Start a new day (this trigger is usually called by the event `HealthSystemScheduler`).
    cons.processing_at_start_of_new_day(date=date)

    # Make requests for consumables (which would normally come from an instance of `HSI_Event`).
    with pytest.warns(None) as recorded_warnings:
        rtn = cons._request_consumables(
            item_codes={0: 1, 1: 1},
            facility_id=0
        )

    assert {0: False, 1: True} == rtn
    assert not any_warnings_about_item_code(recorded_warnings)


def test_unrecognised_item_code_leads_to_warning():
    """Check that when using an item_code that is not recognised, a working result is returned but a warning is
    issued."""
    # Prepare inputs for the Consumables class (normally provided by the `HealthSystem` module).
    data = create_dummy_data_for_cons_availability(
        intrinsic_availability={0: 0.0, 1: 1.0},
        months=[1],
        facility_ids=[0])
    rng = numpy.random
    date = datetime.datetime(2010, 1, 1)

    # Initiate Consumables class
    cons = Consumables(data=data, rng=rng)

    # Start a new day (this trigger usually called by the event `HealthSystemScheduler`).
    cons.processing_at_start_of_new_day(date=date)

    # Make requests for consumables (which would normally come from an instance of `HSI_Event`).
    with pytest.warns(None) as recorded_warnings:
        rtn = cons._request_consumables(
            item_codes={99: 1},
            facility_id=0
        )

    assert isinstance(rtn[99], bool)
    assert any_warnings_about_item_code(recorded_warnings)


def test_consumables_availability_options():
    """Check that the options for `availability` in the Consumables class work as expected for recognised and
    unrecognised item_codes."""
    intrinsic_availability = {0: 0.0, 1: 1.0}
    data = create_dummy_data_for_cons_availability(
        intrinsic_availability=intrinsic_availability,
        months=[1, 2],
        facility_ids=[0, 1])
    rng = numpy.random
    date = datetime.datetime(2010, 1, 1)

    # Define the items to be requested, including some unrecognised item_codes
    all_items_request = list(intrinsic_availability.keys()) + [98, 99]

    # Determine the expected results given the option
    options_and_expected_results = {
        "all": {_i: True for _i in all_items_request},
        "none": {_i: False for _i in all_items_request},
    }

    # Check that for each option for `availability` the result is as expected.
    for _cons_availability_option, _expected_result in options_and_expected_results.items():
        cons = Consumables(data=data, rng=rng, availability=_cons_availability_option)
        cons.processing_at_start_of_new_day(date=date)

        assert _expected_result == cons._request_consumables(
            item_codes={_item_code: 1 for _item_code in all_items_request}, to_log=False, facility_id=0
        )


@pytest.mark.slow
def test_consumables_available_at_right_frequency():
    """Check that the availability of consumables following a request is as expected."""
    # Define known set of probabilities with which each item is available
    p_known_items = dict(zip(range(4), [0.0, 0.2, 0.8, 1.0]))
    requested_items = {**{_i: 1 for _i in p_known_items}, **{4: 1}}  # request for item_code=4 is not recognised.
    average_availability_of_known_items = sum(p_known_items.values()) / len(p_known_items.values())

    data = create_dummy_data_for_cons_availability(
        intrinsic_availability=p_known_items,
        months=[1],
        facility_ids=[0])
    rng = numpy.random
    date = datetime.datetime(2010, 1, 1)

    # Initiate Consumables class
    cons = Consumables(data=data, rng=rng)

    # Make requests for consumables (which would normally come from an instance of `HSI_Event`).
    n_trials = 10_000
    counter = {_i: 0 for _i in requested_items}

    for _ in range(n_trials):
        cons.processing_at_start_of_new_day(date=date)
        rtn = cons._request_consumables(
            item_codes=requested_items,
            facility_id=0,
        )
        for _i in requested_items:
            counter[_i] += (1 if (rtn[_i]) else 0)

    assert 0 == counter[0]
    assert n_trials == counter[3]

    for _i, _p in p_known_items.items():
        assert np.isclose(_p, counter[_i] / n_trials, rtol=0.05)

    # Check that the availability of the unknown item is the average of the known items
    assert np.isclose(average_availability_of_known_items, counter[4] / n_trials, rtol=0.02)


def get_sim_with_dummy_module_registered(tmpdir=None, run=True, data=None):
    """Return an initialised simulation object with a Dummy Module registered. If the `data` argument is provided,
    the parameter in HealthSystem that holds the data on consumables availability is over-written."""

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create simulation with the HealthSystem and DummyModule
    if tmpdir is not None:
        _log_config = {
            'filename': 'tmp',
            'directory': tmpdir,
        }
    else:
        _log_config = None

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0, log_config=_log_config)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
        # Disable sorting + checks to avoid error due to missing dependencies
        sort_modules=False,
        check_all_dependencies=False
    )

    if data is not None:
        sim.modules['HealthSystem'].parameters['availability_estimates'] = data

    sim.make_initial_population(n=100)

    if run:
        sim.simulate(end_date=start_date)

    return sim


def get_dummy_hsi_event_instance(module, facility_id=None):
    """Make an HSI Event that runs for person_id=0 in a particular facility_id and requests consumables,
    and for which its parent is the identified module."""

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = find_level_of_facility_id(facility_id)
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1}) \
                if self.ACCEPTED_FACILITY_LEVEL == '0' else self.make_appt_footprint({'Over5OPD': 1})

            self.ALERT_OTHER_DISEASES = []
            self._facility_id = facility_id

        def apply(self, person_id, squeeze_factor):
            """Requests all recognised consumables."""
            self.get_consumables(
                item_codes=list(self.sim.modules['HealthSystem'].consumables.item_codes),
                to_log=True,
                return_individual_results=False
            )

    return HSI_Dummy(module=module, person_id=0)


def test_use_get_consumables_by_hsi_method_get_consumables():
    """Test that the helper function 'get_consumables' in the base class of the HSI works as expected with different
    forms of input for item_codes."""

    # Create data with item_codes known to be available or not.
    item_code_is_available = [0, 1]
    item_code_not_available = [2, 3]
    intrinsic_availability = {**{_i: 1.0 for _i in item_code_is_available},
                              **{_i: 0.0 for _i in item_code_not_available}}

    sim = get_sim_with_dummy_module_registered(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability=intrinsic_availability,
            months=[1],
            facility_ids=[0])
    )
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'], facility_id=0)

    # Test using item_codes in different input format and with different output formats..
    # -- input as `int`
    assert True is hsi_event.get_consumables(item_codes=item_code_is_available[0])
    assert False is hsi_event.get_consumables(item_codes=item_code_not_available[0])
    assert {item_code_not_available[0]: False} == hsi_event.get_consumables(
        item_codes=item_code_not_available[0], return_individual_results=True)

    # -- input as `list`
    assert True is hsi_event.get_consumables(item_codes=item_code_is_available)
    assert False is hsi_event.get_consumables(item_codes=item_code_not_available)
    assert False is hsi_event.get_consumables(item_codes=item_code_is_available + item_code_not_available)
    assert {item_code_is_available[0]: True, item_code_not_available[0]: False} == hsi_event.get_consumables(
        item_codes=[item_code_is_available[0], item_code_not_available[0]], return_individual_results=True)

    # -- input as `dict`
    assert True is hsi_event.get_consumables(
        item_codes={i: 10 for i in item_code_is_available}
    )
    assert False is hsi_event.get_consumables(
        item_codes={i: 10 for i in item_code_not_available}
    )
    assert {item_code_is_available[0]: True, item_code_not_available[0]: False} == hsi_event.get_consumables(
        item_codes={item_code_is_available[0]: 10, item_code_not_available[0]: 10},
        return_individual_results=True
    )

    #  Using `optional_item_codes` argument in the `get_consumables` method on the HSI Base Class should result in those
    #  consumables being checked for availability (and the request logged), but the availability/non-availability of
    #  these items does not affect the summary result (a `bool` returned indicating availability/non-availability of the
    #  items requested).

    # Request both consumables in usual fashion: as one in not available, overall result is False
    assert False is hsi_event.get_consumables(item_codes=[item_code_is_available[0], item_code_not_available[0]])

    # Make request with the non-available consumable being optional: as the non-optional one is available, result is
    #  True.
    assert True is hsi_event.get_consumables(item_codes=item_code_is_available[0],
                                             optional_item_codes=item_code_not_available[0])

    # Make request with the non-available consumable being non-optional: result is False
    assert False is hsi_event.get_consumables(item_codes=item_code_not_available[0],
                                              optional_item_codes=item_code_is_available[0])

    # If the only consumables requested are optional, then the result is always True
    assert True is hsi_event.get_consumables(item_codes=[], optional_item_codes=item_code_not_available[0])
    assert True is hsi_event.get_consumables(item_codes=None, optional_item_codes=item_code_not_available[0])
    assert True is hsi_event.get_consumables(optional_item_codes=item_code_not_available[0])

    # Check that option `return_individual_results` works as expected when using `optional_item_codes`
    assert {item_code_is_available[0]: True, item_code_not_available[0]: False} == hsi_event.get_consumables(
        item_codes=item_code_is_available[0],
        optional_item_codes=item_code_not_available[0],
        return_individual_results=True
    )


def test_outputs_to_log(tmpdir):
    """Check that logging from Consumables is as expected."""
    intrinsic_availability = {0: True, 1: False}

    sim = get_sim_with_dummy_module_registered(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability=intrinsic_availability,
            months=[1],
            facility_ids=[0]),
        tmpdir=tmpdir,
        run=False
    )

    # Edit the `initialise_simulation` method of DummyModule so that, during the simulation, an HSI is run that requests
    # consumables.
    def schedule_hsi_that_will_request_consumables(sim):
        """Drop-in replacement for `initialise_simulation` in the DummyModule module."""
        # Make the district for person_id=0 such that the HSI will be served by facility_id=0
        sim.population.props.at[0, 'district_of_residence'] = mfl.set_index('Facility_ID').loc[0].District

        # Schedule the HSI event for person_id=0
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=get_dummy_hsi_event_instance(module=sim.modules['DummyModule'], facility_id=0),
            topen=sim.start_date,
            tclose=None,
            priority=0
        )

    sim.modules['DummyModule'].initialise_simulation = schedule_hsi_that_will_request_consumables

    # Simulate for one day
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=1))

    # Check that log is created and the content is as expected.
    cons_log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']['Consumables']
    assert len(cons_log)
    assert "{0: 1}" == cons_log.loc[cons_log.index[0], 'Item_Available']  # Item 0 (1 requested) is available
    assert "{1: 1}" == cons_log.loc[cons_log.index[0], 'Item_NotAvailable']  # Item 1 (1 requested) is not available


# ----------------------------------------------------------------------------
# Checks involving the actual ResourceFile used by default in the simulations
# ----------------------------------------------------------------------------

def check_format_of_consumables_file(df):
    """Check that we have a complete set of estimates, for every region & facility_type, as defined in the model."""
    months = set(range(1, 13))
    item_codes = set(df.item_code.unique())

    assert set(df.columns) == {'Facility_ID', 'month', 'item_code', 'available_prop'}

    # Check that all permutations of Facility_ID, month and item_code are present
    pd.testing.assert_index_equal(
        df.set_index(['Facility_ID', 'month', 'item_code']).index,
        pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code']),
        check_order=False
    )

    # Check that every entry for a probability is a float on [0,1]
    assert (df.available_prop <= 1.0).all() and (df.available_prop >= 0.0).all()
    assert not pd.isnull(df.available_prop).any()


def test_check_format_of_consumables_file():
    """Run the check on the file used by default for the Consumables data"""
    check_format_of_consumables_file(pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_availability_small.csv'
    ))


@pytest.mark.slow
def test_every_declared_consumable_for_every_possible_hsi_using_actual_data():
    """Check that every item_code that is declared can be requested from a person at every district and facility_level.
    """

    sim = get_sim_with_dummy_module_registered(run=True)
    hs = sim.modules['HealthSystem']
    item_codes = hs.consumables.item_codes

    with pytest.warns(None) as recorded_warnings:
        for month in range(1, 13):
            sim.date = Date(2010, month, 1)
            hs.consumables._refresh_availability_of_consumables(date=sim.date)

            for _district in sim.modules['Demography'].PROPERTIES['district_of_residence'].categories:
                # Change the district of person 0 (for whom the HSI is created.)
                sim.population.props.at[0, 'district_of_residence'] = _district
                for _facility_id in fac_ids:
                    hsi_event = get_dummy_hsi_event_instance(
                        module=sim.modules['DummyModule'],
                        facility_id=_facility_id
                    )
                    for _item_code in item_codes:
                        hsi_event.get_consumables(item_codes=_item_code)

    assert not any_warnings_about_item_code(recorded_warnings)


def test_get_item_code_from_item_name():
    """Check that can use `get_item_code_from_item_name` to retrieve the correct `item_code`."""
    lookup_df = pd.read_csv(
        resourcefilepath / "healthsystem" / "consumables" / "ResourceFile_Consumables_Items_and_Packages.csv"
    )

    example_item_names = [
        "Syringe, autodisposable, BCG, 0.1 ml, with needle",
        "Pentavalent vaccine (DPT, Hep B, Hib)",
        "Pneumococcal vaccine"
    ]

    for _item_name in example_item_names:
        _item_code = get_item_code_from_item_name(lookup_df=lookup_df, item=_item_name)
        assert isinstance(_item_code, int)
        assert lookup_df.loc[lookup_df.Item_Code == _item_code].Items.values[0] == _item_name


def test_get_item_codes_from_package_name():
    """Check that can use `get_item_codes_from_package_name` to retrieve the correct `item_code`."""
    lookup_df = pd.read_csv(
        resourcefilepath / "healthsystem" / "consumables" / "ResourceFile_Consumables_Items_and_Packages.csv"
    )

    example_package_names = [
        "Measles rubella vaccine",
        "HPV vaccine",
        "Tetanus toxoid (pregnant women)"
    ]

    for _pkg_name in example_package_names:
        _item_codes = get_item_codes_from_package_name(lookup_df=lookup_df, package=_pkg_name)
        assert isinstance(_item_codes, dict)

        res_from_lookup = \
            lookup_df.loc[lookup_df.Intervention_Pkg == _pkg_name].set_index('Item_Code').sort_index()[
                'Expected_Units_Per_Case'].astype(int)

        pd.testing.assert_series_equal(
            res_from_lookup.groupby(res_from_lookup.index).sum(),
            pd.Series(_item_codes).sort_index(),
            check_names=False
        )
