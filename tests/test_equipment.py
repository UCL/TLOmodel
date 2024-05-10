"""This file contains all the tests to do with Equipment use logging and availability checks."""
import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.equipment import Equipment
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"


def test_core_functionality_of_equipment_class(seed):
    """Test that the core functionality of the equipment class works on toy data."""

    # Create toy data
    catalogue = pd.DataFrame(
        [
            {"Item_Description": "ItemZero", "Item_Code": 0, "Pkg_Name": float('nan')},
            {"Item_Description": "ItemOne", "Item_Code": 1, "Pkg_Name": float('nan')},
            {"Item_Description": "ItemTwo", "Item_Code": 2, "Pkg_Name": 'PkgWith2+3'},
            {"Item_Description": "ItemThree", "Item_Code": 3, "Pkg_Name": 'PkgWith2+3'},
        ]
    )
    data_availability = pd.DataFrame(
        # item 0 is not available anywhere; item 1 is available everywhere; item 2 is available only at facility_id=1
        # No data for fac_id=2
        [
            {"Item_Code": 0, "Facility_ID": 0, "Pr_Available": 0.0},
            {"Item_Code": 0, "Facility_ID": 1, "Pr_Available": 0.0},
            {"Item_Code": 1, "Facility_ID": 0, "Pr_Available": 1.0},
            {"Item_Code": 1, "Facility_ID": 1, "Pr_Available": 1.0},
            {"Item_Code": 2, "Facility_ID": 0, "Pr_Available": 0.0},
            {"Item_Code": 2, "Facility_ID": 1, "Pr_Available": 1.0},
        ]
    )

    mfl = pd.DataFrame(
        [
            {'District': 'D0', 'Facility_Level': '1a', 'Region': 'R0', 'Facility_ID': 0, 'Facility_Name': 'Fac0'},
            {'District': 'D0', 'Facility_Level': '1b', 'Region': 'R0', 'Facility_ID': 1, 'Facility_Name': 'Fac1'},
            {'District': 'D0', 'Facility_Level': '2', 'Region': 'R0', 'Facility_ID': 2, 'Facility_Name': 'Fac2'},
        ]
    )

    # Create instance of the Equipment class with these toy data and check availability of equipment...
    # -- when using `default` behaviour:
    eq_default = Equipment(
        catalogue=catalogue,
        data_availability=data_availability,
        rng=np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed))),
        master_facilities_list=mfl,
        availability="default",
    )

    # Checks on parsing equipment items
    # - using single integer for one item_code
    assert {1} == eq_default.parse_items(1)
    # - using list of integers for item_codes
    assert {1, 2} == eq_default.parse_items([1, 2])
    # - using single string for one item descriptor
    assert eq_default.parse_items('ItemOne')
    # - using list of strings for item descriptors
    assert eq_default.parse_items(['ItemOne', 'ItemTwo'])
    # - an empty iterable of equipment should always be work whether expressed as list/tuple/set
    assert set() == eq_default.parse_items(list())
    assert set() == eq_default.parse_items(tuple())
    assert set() == eq_default.parse_items(set())

    # - Calling for unrecognised item_codes (should raise warning)
    with pytest.warns():
        eq_default.parse_items(10001)
    with pytest.warns():
        eq_default.parse_items('ItemThatIsNotDefined')

    # Testing checking on available of items
    # - calling when all items available (should be true)
    assert eq_default.is_all_items_available(item_codes={1, 2}, facility_id=1)
    # - calling when no items available (should be false)
    assert not eq_default.is_all_items_available(item_codes={0, 2}, facility_id=0)
    # - calling when some items available (should be false)
    assert not eq_default.is_all_items_available(item_codes={1, 2}, facility_id=0)
    # - calling for empty set of equipment (should always be available)
    assert eq_default.is_all_items_available(item_codes=set(), facility_id=0)

    # - calling an item for which data on availability is not provided (should not raise error)
    eq_default.is_all_items_available(item_codes={3}, facility_id=1)
    # - calling an item at a facility that for which data is not provided (should give average behaviour for other
    #   facilities)
    assert not eq_default.is_all_items_available(item_codes={0}, facility_id=2)
    assert eq_default.is_all_items_available(item_codes={1}, facility_id=2)
    # - calling a recognised item for which no data at a facility with no data (should not error)
    eq_default.is_all_items_available(item_codes={3}, facility_id=2)
    # -- calling for an unrecognised facility_id (should error)
    with pytest.raises(ValueError):
        eq_default.is_all_items_available(item_codes={1}, facility_id=1001)

    # -- when using `none` availability behaviour: everything should not be available!
    eq_none = Equipment(
        catalogue=catalogue,
        data_availability=data_availability,
        rng=np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed))),
        availability="none",
        master_facilities_list=mfl,
    )
    # - calling when all items available (should be false because using 'none' behaviour)
    assert not eq_none.is_all_items_available(item_codes={1, 2}, facility_id=1)
    # - calling when no items available (should be false)
    assert not eq_none.is_all_items_available(item_codes={0, 2}, facility_id=0)
    # - calling when some items available (should be false)
    assert not eq_none.is_all_items_available(item_codes={1, 2}, facility_id=0)
    # - calling for empty set of equipment (should always be available)
    assert eq_none.is_all_items_available(item_codes=set(), facility_id=0)

    # -- when using `all` availability behaviour: everything should not be available!
    eq_all = Equipment(
        catalogue=catalogue,
        data_availability=data_availability,
        rng=np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed))),
        availability="all",
        master_facilities_list=mfl,
    )
    # - calling when all items available (should be true)
    assert eq_all.is_all_items_available(item_codes={1, 2}, facility_id=1)
    # - calling when no items available (should be true because using 'all' behaviour)
    assert eq_all.is_all_items_available(item_codes={0, 2}, facility_id=0)
    # - calling when some items available (should be true because using 'all' behaviour)
    assert eq_all.is_all_items_available(item_codes={1, 2}, facility_id=0)
    # - calling for empty set of equipment (should always be available)
    assert eq_all.is_all_items_available(item_codes=set(), facility_id=0)

    # Check recording use of equipment
    # - Add records, using calls with integers and list to different facility_id
    eq_default.record_use_of_equipment(item_codes={1}, facility_id=0)
    eq_default.record_use_of_equipment(item_codes={0, 1}, facility_id=0)
    eq_default.record_use_of_equipment(item_codes={0, 1}, facility_id=1)
    # - Check that internal record is as expected
    assert dict(eq_default._record_of_equipment_used_by_facility_id) == {0: {0: 1, 1: 2}, 1: {0: 1, 1: 1}}

    # Lookup the item_codes that belong in a particular package.
    # - When package is recognised
    assert {2, 3} == eq_default.lookup_item_codes_from_pkg_name(pkg_name='PkgWith2+3')  # these items are in the same
    #                                                                                     package
    # - Error thrown when package is not recognised
    with pytest.raises(ValueError):
        eq_default.lookup_item_codes_from_pkg_name(pkg_name='')



equipment_item_code_that_is_available = [0, 1, ]
equipment_item_code_that_is_not_available = [2, 3,]

def run_simulation_and_return_log(
    seed, tmpdir, essential_equipment: Iterable[str], other_equipment: Iterable[str]
) -> Dict:
    """Returns the parsed logs from `tlo.methods.healthsystem.summary:EquipmentEverUsed_ByFacilityID` from a
    simulation object in which a single event has been run with the specified equipment usage. The
    availability of equipment has been manipulated so that the item_codes given in
    `equipment_item_code_that_is_available` and `equipment_item_code_that_is_not_available` are as expected. """

    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(
            self,
            module,
            person_id,
            level,
            essential_equipment,
            other_equipment,
        ):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSIEvent"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = level
            self.add_equipment(essential_equipment)  # Declaration at init will mean that these items are considered
            #                                          essential.
            self._other_equipment = other_equipment

        def apply(self, person_id, squeeze_factor):
            if self._other_equipment is not None:
                self.add_equipment(self._other_equipment)

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

        def __init__(self, essential_equipment, other_equipment, name=None):
            super().__init__(name)
            self.essential_equipment = essential_equipment
            self.other_equipment = other_equipment

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule the HSI_Event to occur on the first day of the simulation
            sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=DummyHSIEvent(
                    person_id=0,
                    level="2",
                    module=sim.modules["DummyModule"],
                    essential_equipment=self.essential_equipment,
                    other_equipment=self.other_equipment,
                ),
                do_hsi_event_checks=False,
                topen=sim.date,
                tclose=None,
                priority=0,
            )

    log_config = {"filename": "log", "directory": tmpdir}
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(
            essential_equipment=essential_equipment, other_equipment=other_equipment
        ),
    )

    # Manipulate availability of equipment
    df = sim.modules["HealthSystem"].parameters["equipment_availability_estimates"]
    df.loc[df['Item_Code'].isin(equipment_item_code_that_is_available), 'Pr_Available'] = 1.0
    df.loc[df['Item_Code'].isin(equipment_item_code_that_is_not_available), 'Pr_Available'] = 0.0

    # Run the simulation
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(months=1))

    # Return the parsed log of `tlo.methods.healthsystem.summary`
    return parse_log_file(sim.log_filepath)["tlo.methods.healthsystem.summary"]



def test_equipment_use_is_logged(seed, tmpdir):
    """Check that an HSI that after an HSI is run, the logs reflect the use of the equipment (and correctly record the
     name of the HSI and the facility_level at which ran).
     This is repeated for:
        * An HSI that declares use of equipment during its `apply` method (but no essential equipment);
        * An HSI that declare use of essential equipment but nothing in its `apply` method`;
        * An HSI that declare use of essential equipment and equipment during its `apply` method;
        * An HSI that declares not use of any equipment.
     """
    the_item_code = equipment_item_code_that_is_available[0]
    another_item_code = equipment_item_code_that_is_available[1]

    def all_equipment_ever_used(log: Dict) -> set:
        """With the log of equipment used in the simulation, return a set of the equipments item that have been used
        (at any facility)."""
        s = set()
        for i in log["EquipmentEverUsed_ByFacilityID"]['EquipmentEverUsed']:
            s.update(eval(i))
        return s

    # * An HSI that declares no use of any equipment (logs should be empty).
    assert set() == all_equipment_ever_used(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=set(),
            other_equipment=set(),
        )
    )

    # * An HSI that declares use of equipment during its `apply` method (but no essential equipment)
    assert {the_item_code} == all_equipment_ever_used(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=set(),
            other_equipment=the_item_code,
        )
    )

    # * An HSI that declare use of essential equipment but nothing in its `apply` method`;
    assert {the_item_code} == all_equipment_ever_used(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=the_item_code,
            other_equipment=set(),
        )
    )

    # * An HSI that declare use of essential equipment and equipment during its `apply` method;
    assert {the_item_code, another_item_code} == all_equipment_ever_used(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=the_item_code,
            other_equipment=another_item_code,
        )
    )


def test_hsi_does_not_run_if_essential_equipment_is_not_available(seed, tmpdir):
    """Check that an HSI which declares an item of equipment that is essential does run if that item is available
    and does not run if that item is not available."""

    def did_the_hsi_run(log: Dict) -> bool:
        """Read the log to work out if the `DummyHSIEvent` ran or not."""
        it_did_run = len(log['hsi_event_counts'].iloc[0]['hsi_event_key_to_counts']) > 0
        it_did_not_run = len(log['never_ran_hsi_event_counts'].iloc[0]['never_ran_hsi_event_key_to_counts']) > 0

        # Check that there if it did not run, it has had its "never_ran" function called
        assert it_did_run != it_did_not_run

        # Return indication of whether it did run
        return it_did_run


    # HSI_Event that requires equipment that is available --> will run
    assert did_the_hsi_run(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=equipment_item_code_that_is_available,
            other_equipment=set(),
        )
    )

    # HSI_Event that requires equipment that is not available --> will not run
    assert not did_the_hsi_run(
        run_simulation_and_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=equipment_item_code_that_is_not_available,
            other_equipment=set(),
        )
    )


def test_change_equipment_availability(seed):
    """Test that we can change the availability of equipment midway through the simulation."""
    # Set-up simulation that starts with `all` availability and then changes to  `none` after one year. In the
    # simulation a DummyModule schedules a DummyHSI that runs every month and tries to get a piece of equipment.
    # Check that this piece of equipment is available for the first year but not the second year of the simulation.

    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(
            self,
            module,
            person_id,
        ):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSIEvent"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = '1a'
            self.store_of_equipment_checks = dict()

        def apply(self, person_id, squeeze_factor):
            # Check availability of a piece of equipment, with item_code = 0
            self.store_of_equipment_checks.update(
                {
                    self.sim.date: self.is_equipment_available(item_codes={0})
                }
            )

            # Schedule recurrence of this event in one month's time
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=self,
                do_hsi_event_checks=False,
                topen=self.sim.date + pd.DateOffset(months=1),
                tclose=None,
                priority=0,
            )

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule the HSI_Event to occur on the first day of the simulation (it will schedule its own repeats)
            self.the_hsi_event = DummyHSIEvent(person_id=0, module=self)

            sim.modules["HealthSystem"].schedule_hsi_event(
                hsi_event=self.the_hsi_event,
                do_hsi_event_checks=False,
                topen=sim.date,
                tclose=None,
                priority=0,
            )

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        DummyModule(),
    )
    # Modify the parameters of the healthsystem to effect a change in the availability of equipment
    sim.modules['HealthSystem'].parameters['equip_availability'] = 'all'
    sim.modules['HealthSystem'].parameters['equip_availability_postSwitch'] = 'none'
    sim.modules['HealthSystem'].parameters['year_equip_availability_switch'] = 2011

    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(years=2))

    # Get store & check for availabilities of the equipment
    log = pd.Series(sim.modules['DummyModule'].the_hsi_event.store_of_equipment_checks)
    assert log[log.index < Date(2011, 1, 1)].all()
    assert not log[log.index >= Date(2011, 1, 1)].any()
