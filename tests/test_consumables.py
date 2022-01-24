import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    Metadata,
    demography,
    healthsystem,

)
from tlo.methods.consumables import Consumables
from tlo.methods.healthsystem import HSI_Event

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def get_sim_with_dummy_module_registered(tmpdir=None, run=True):
    """Return an initialised and run simulation object with a Dummy Module registered."""

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
        _log_config=None

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

    if run:
        sim.make_initial_population(n=100)
        sim.simulate(end_date=start_date)

    return sim

def get_dummy_hsi_event_instance(module, accepted_facility_level='1a'):
    """Make an HSI Event that runs for person_id=0 and request consumables, and for which its parent is the
    identified module."""
    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
            self.ACCEPTED_FACILITY_LEVEL = accepted_facility_level
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            self.get_consumables(
                item_codes=list(self.sim.modules['HealthSystem'].consumables.item_codes),
                to_log=True,
                return_individual_results=False
            )

    return HSI_Dummy(module=module, person_id=0)

def create_dummy_df_for_cons(intrinsic_availability: dict = None) -> pd.DataFrame:
    """Returns a pd.DataFrame that is a dummy for the imported `ResourceFile_Consumables.csv`, which has two items,
    one of which is always available, and one of which is never available."""

    if not intrinsic_availability:
        _intrinsic_availability = {0: False, 1: True}
    else:
        _intrinsic_availability = intrinsic_availability

    list_of_items = []
    for k, v in _intrinsic_availability.items():
        list_of_items.append({
            'Item_Code': k,
            'Available_Facility_Level_0': 1.0 if v else 0.0,
            'Available_Facility_Level_1a': 1.0 if v else 0.0,
            'Available_Facility_Level_1b': 1.0 if v else 0.0,
            'Available_Facility_Level_2': 1.0 if v else 0.0,
            'Available_Facility_Level_3': 1.0 if v else 0.0,
            'Intervention_Pkg_Code': 0,
            'Expected_Units_Per_Case': 1
        })

    return pd.DataFrame(data=list_of_items)

def test_consumables_availability_options():
    """Check that the options for `cons_availability` in the Consumables class work as expected."""
    # todo refactor to pass in the 'original' availability and let that form the part of the test

    # Create simulation and hsi_event
    sim = get_sim_with_dummy_module_registered()
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'])

    # Create dataframe for the availability of consumables
    intrinsic_availability = {0: True, 1: False}
    df = create_dummy_df_for_cons(intrinsic_availability)

    # Determine the expected results given the option
    options_and_expected_results = {
        "all": {_i: True for _i in intrinsic_availability.keys()},
        "none": {_i: False for _i in intrinsic_availability.keys()},
        "default": intrinsic_availability
    }

    # Check that for each option for `cons_availability` the result is as expected.
    for _cons_availability_option, _expected_result in options_and_expected_results.items() :

        cons = Consumables(sim.modules['HealthSystem'], cons_availabilty=_cons_availability_option)
        cons.process_consumables_df(df)
        cons.processing_at_start_of_new_day()

        assert _expected_result == cons._request_consumables(
            hsi_event, item_codes={_item_code: 1 for _item_code in range(2)}, to_log=False
        )

def test_outputs_to_log(tmpdir):
    """Check that logging from Consumables is as expected."""

    # Create simulation
    sim = get_sim_with_dummy_module_registered(tmpdir=tmpdir, run=False)

    # Edit the `initialise_simulation` method of DummyModule so that, during the simulation, an HSI is run which
    # requests consumables.
    def schedule_hsi_that_will_request_consumables(sim):
        """Drop-in replacement for `initialise_simulation` in the DummyModule module."""
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=get_dummy_hsi_event_instance(module=sim.modules['DummyModule']),
            topen=sim.start_date,
            tclose=None,
            priority=0
        )
    sim.modules['DummyModule'].initialise_simulation = schedule_hsi_that_will_request_consumables

    # Simulate
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=1))

    # Check that log is created
    cons_log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']['Consumables']
    assert len(cons_log)

def test_every_declared_consumable():
    """Check that every item_code that is declared can be requested from a person at every district and facility_level.
    """
    sim = get_sim_with_dummy_module_registered(run=True)
    hs = sim.modules['HealthSystem']
    cons = hs.consumables
    hs.consumables._refresh_availability_of_consumables()

    with pytest.warns(None) as recorded_warnings:
        for _disrict in sim.modules['Demography'].PROPERTIES['district_of_residence'].categories:
            # Change the district of person 0 (for whom the HSI is created.)
            sim.population.props.at[0, 'district_of_residence'] = _disrict
            for _accepted_facility_level in (hs._facility_levels - {'4'}):
                for _item_code in cons.item_codes:
                    hsi_event = get_dummy_hsi_event_instance(
                        module=sim.modules['DummyModule'],
                        accepted_facility_level=_accepted_facility_level
                    )
                    hsi_event.get_consumables(item_codes=_item_code)

    assert 0 == len(recorded_warnings)

def test_unrecognised_consumables_lead_to_warning():
    """Check that every item_code that is declared can be requested from a person at every district and facility_level.
    """
    sim = get_sim_with_dummy_module_registered(run=True)
    hs = sim.modules['HealthSystem']
    cons = hs.consumables
    hs.consumables._refresh_availability_of_consumables()

    _item_code_that_is_not_recognised = max(cons.item_codes) + 10000000

    with pytest.warns(UserWarning):
        hsi_event = get_dummy_hsi_event_instance(
            module=sim.modules['DummyModule'],
        )
        hsi_event.get_consumables(item_codes=_item_code_that_is_not_recognised)


def test_use_get_consumables_with_different_inputs_for_item_codes():
    """Test that the helper function 'get_consumables' in the base class of the HSI works as expectewd with different
    forms of input for item_codes."""

    sim = get_sim_with_dummy_module_registered()
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'])
    hs = sim.modules['HealthSystem']

    # Manually edit availability probabilities to force some items to be (not) available
    item_code_is_available = [0, 1]
    item_code_not_available = [2, 3]
    hs.consumables.prob_item_codes_available.loc[item_code_is_available] = 1
    hs.consumables.prob_item_codes_available.loc[item_code_not_available] = 0
    hs.consumables._refresh_availability_of_consumables()

    # Test using item_codes in different input format and with different output formats
    # -- as `int`
    assert True is hsi_event.get_consumables(item_codes=item_code_is_available[0])
    assert False is hsi_event.get_consumables(item_codes=item_code_not_available[0])

    # -- as `list`
    assert True is hsi_event.get_consumables(item_codes=item_code_is_available)
    assert False is hsi_event.get_consumables(item_codes=item_code_not_available)
    assert False is hsi_event.get_consumables(item_codes=item_code_is_available + item_code_not_available)
    assert {item_code_is_available[0]: True, item_code_not_available[0]: False} == hsi_event.get_consumables(
        item_codes=[item_code_is_available[0], item_code_not_available[0]], return_individual_results=True)

    # -- as `dict`
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




