import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from tlo import Date, Module, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.consumables import Consumables
from tlo.methods.healthsystem import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def any_warnings_about_item_code(recorded_warnings):
    return len([_r for _r in recorded_warnings if str(_r.message).startswith('Item_Code')]) > 0


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
            """Requests all consumables."""
            self.get_consumables(
                item_codes=list(self.sim.modules['HealthSystem'].consumables.item_codes),
                to_log=True,
                return_individual_results=False
            )

    return HSI_Dummy(module=module, person_id=0)


def create_dummy_df_for_cons(intrinsic_availability: Dict[int, bool] = {0: False, 1: True},
                             districts: List[str] = None,
                             months: List[int] = [1],
                             facility_levels: List[int] = ['1a']
                             ) -> pd.DataFrame:
    """Returns a pd.DataFrame that is a dummy for the imported `ResourceFile_Consumables.csv`, which has two items,
    one of which is always available, and one of which is never available."""
    list_of_items = []
    for _item, _avail in intrinsic_availability.items():
        for _district in districts:
            for _month in months:
                for _fac_level in facility_levels:
                    list_of_items.append({
                        'item_code': _item,
                        'district': _district,
                        'month': _month,
                        'fac_type_tlo': _fac_level,
                        'available_prop': _avail
                    })

    return pd.DataFrame(data=list_of_items)


def test_consumables_availability_options():
    """Check that the options for `cons_availability` in the Consumables class work as expected."""

    # Create simulation and hsi_event
    sim = get_sim_with_dummy_module_registered()
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'])

    # Create dataframe for the availability of consumables
    intrinsic_availability = {0: True, 1: False}

    df = create_dummy_df_for_cons(intrinsic_availability,
                                  districts=[sim.population.props.at[0, 'district_of_residence']],
                                  months=[1],
                                  facility_levels=['1a']
                                  )

    # Determine the expected results given the option
    options_and_expected_results = {
        "all": {_i: True for _i in intrinsic_availability.keys()},
        "none": {_i: False for _i in intrinsic_availability.keys()},
        "default": intrinsic_availability
    }

    # Check that for each option for `cons_availability` the result is as expected.
    for _cons_availability_option, _expected_result in options_and_expected_results.items():
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
    sim.make_initial_population(n=100)

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

    # Change the Consumables availability
    intrinsic_availability = {0: True, 1: False}
    sim.modules['HealthSystem'].consumables.process_consumables_df(
        create_dummy_df_for_cons(intrinsic_availability,
                                 districts=[sim.population.props.at[0, 'district_of_residence']],
                                 months=[1],
                                 facility_levels=['1a'])
    )

    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=1))

    # Check that log is created
    cons_log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem']['Consumables']
    assert len(cons_log)
    assert cons_log.loc[cons_log.index[0], 'Item_Available'] == "{0: 1}"  # Item 0 (1 requested) is available
    assert cons_log.loc[cons_log.index[0], 'Item_NotAvailable'] == "{1: 1}"  # Item 1 (1 requested) is not available


@pytest.mark.slow
def test_every_declared_consumable_for_every_possible_hsi():
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

    assert not any_warnings_about_item_code(recorded_warnings)


def test_unrecognised_consumables_lead_to_warning_and_controlled_behaviour():
    """Check that when using an item_code that is not recognised, a working result is returned but a warning is
    issued."""
    sim = get_sim_with_dummy_module_registered()
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'])

    # Create dataframe for the availability of consumables (only one).
    df = create_dummy_df_for_cons(intrinsic_availability={0: True},
                                  districts=[sim.population.props.at[0, 'district_of_residence']],
                                  months=[1],
                                  facility_levels=['1a']
                                  )
    item_code_that_is_not_recognised = 1

    # Determine the expected results given the option
    options_for_cons_availability = {"all", "none", "default"}

    # Check that for each option for `cons_availability` the result is as expected.
    for _option in options_for_cons_availability:
        cons = Consumables(sim.modules['HealthSystem'], cons_availabilty=_option)
        cons.process_consumables_df(df)
        cons.processing_at_start_of_new_day()

        with pytest.warns(UserWarning) as recorded_warnings:
            _result = cons._request_consumables(
                hsi_event, item_codes={item_code_that_is_not_recognised: 1}, to_log=False
            )
            assert isinstance(_result, dict)
            assert isinstance(_result[item_code_that_is_not_recognised], bool)

            if _option == "all":
                assert _result[item_code_that_is_not_recognised] is True
            elif _option == "none":
                assert _result[item_code_that_is_not_recognised] is False

        assert any_warnings_about_item_code(recorded_warnings)


def test_use_get_consumables_with_different_inputs_for_item_codes():
    """Test that the helper function 'get_consumables' in the base class of the HSI works as expected with different
    forms of input for item_codes."""

    sim = get_sim_with_dummy_module_registered()
    hsi_event = get_dummy_hsi_event_instance(module=sim.modules['DummyModule'])
    hs = sim.modules['HealthSystem']

    # Manually edit availability probabilities to force some items to be (not) available
    item_code_is_available = [0, 1]
    item_code_not_available = [2, 3]
    intrinsic_availability = {**{_i: 1.0 for _i in item_code_is_available},
                              **{_i: 0.0 for _i in item_code_not_available}}

    hs.consumables.process_consumables_df(
        df=create_dummy_df_for_cons(intrinsic_availability=intrinsic_availability,
                                    districts=[sim.population.props.at[0, 'district_of_residence']],
                                    months=[1],
                                    facility_levels=['1a'])
    )
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


# todo strict mode to cause errors

# todo - _get_item_codes_from_package_name(self, package: str) -> dict:

# todo - _get_item_code_from_item_name
