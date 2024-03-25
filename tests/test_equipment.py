"""This file contains all the tests to do with Equipment use logging and availability checks."""
import os
from pathlib import Path
from typing import Union, Dict, Iterable

import pandas as pd

from tlo import Simulation, Module, Date
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

equipment_item_code_that_is_available = [0, 1, ]
equipment_item_code_that_is_not_available = [2, 3, ]


def run_simulation_return_log(seed, tmpdir, essential_equipment: Iterable[str], other_equipment: Iterable[str]) -> Dict:
    """Returns a parsed logs from `tlo.methods.healthsystem.summary` from a simulation object, in which a single
    event has been scheduled with the specified equipment usage, and the availability of equipment has been manipulated.
    """

    class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
        def __init__(self,
                     module,
                     person_id,
                     level,
                     essential_equipment: Union[int, None],
                     other_equipment: Union[int, None]
                     ):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSIEvent"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = level
            self.ESSENTIAL_EQUIPMENT = str(essential_equipment) if essential_equipment is not None else set()
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
            sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=DummyHSIEvent(
                    person_id=0,
                    level='2',
                    module=sim.modules['DummyModule'],
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
        DummyModule(essential_equipment=essential_equipment, other_equipment=other_equipment),
    )

    # Manipulate availability of equipment
    df = sim.modules['HealthSystem'].parameters['Equipment']
    col_for_availability = df.columns[df.columns.str.startswith('Avail_')]
    df.loc[df['Equip_Code'].isin(equipment_item_code_that_is_available),  col_for_availability] = True
    df.loc[df['Equip_Code'].isin(equipment_item_code_that_is_not_available),  col_for_availability] = False
    df['Equip_Item'] = df['Equip_Code'].astype(str)
    sim.modules['HealthSystem'].parameters['Equipment'] = df.loc[df['Equip_Code'].isin(set(equipment_item_code_that_is_available) | set(equipment_item_code_that_is_not_available))]

    sim.make_initial_population(n=100)
    sim.simulate(end_date=pd.DateOffset(months=1))

    return parse_log_file(sim.log_filepath)['tlo.methods.healthsystem.summary']




def test_equipment_use_is_logged(seed, tmpdir):
    """Check that an HSI that after an HSI is run, the logs reflect the use of the equipment (and correctly record the
     name of the HSI and the facility_level at which ran).
     This is repeated for:
        * An HSI that declares use of equipment during its `apply` method (but no essential equipment);
        * An HSI that declare use of essential equipment but nothing in its `apply` method`;
        * An HSI that declare use of essential equipment and equipment during its `apply` method;
        * An HSI that declares not use of any equipment (logs should be empty).
     """

    def logged_equipment_used(sim: Dict) -> pd.DataFrame:
        """Read the log to work out what equipment usage has been logged."""
        # @Eva - I think this will somehow use the function that is currently in `src/scripts/healthsystem/equipment/equipment_catalogue.py`
        pass

    def get_sim(essential_equipment, other_equipment):
        """Pass-through to `run_simulation_return_log` to make call simpler."""
        return run_simulation_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=essential_equipment,
            other_equipment=other_equipment,
        )

    # Check that the log matches expectation under each permutation
    item_available_as_set_of_str = {str(equipment_item_code_that_is_available[0])}

    # * An HSI that declares use of equipment during its `apply` method (but no essential equipment)
    expected_df = pd.DataFrame()  # <-- fill in what we expect it to look like
    assert expected_df.equals(logged_equipment_used(get_sim(
        essential_equipment={},
        other_equipment=item_available_as_set_of_str,
    )))

    # * An HSI that declare use of essential equipment but nothing in its `apply` method`;
    expected_df = pd.DataFrame()  # <-- fill in what we expect it to look like
    assert expected_df.equals(logged_equipment_used(get_sim(
        essential_equipment=item_available_as_set_of_str,
        other_equipment={},
    )))

    # * An HSI that declare use of essential equipment and equipment during its `apply` method;
    expected_df = pd.DataFrame()  # <-- fill in what we expect it to look like
    assert expected_df.equals(logged_equipment_used(get_sim(
        essential_equipment=item_available_as_set_of_str,
        other_equipment=item_available_as_set_of_str,
    )))

    # * An HSI that declares not use of any equipment (logs should be empty).
    expected_df = pd.DataFrame()  # <-- fill in what we expect it to look like
    assert expected_df.equals(logged_equipment_used(get_sim(
        essential_equipment={},
        other_equipment={},
    )))


def test_hsi_does_not_run_if_essential_equipment_is_not_available(seed, tmpdir):
    """Check that an HSI which declares an item of equipment that is essential does run if that item is available
    and does not run if that item is not available."""

    def did_hsi_run(sim: Dict) -> bool:
        """Read the log to work out if the Dummy HSI Event ran or not."""
        pass

    def get_sim(essential_equipment):
        """Pass-through to `run_simulation_return_log` to make call simpler."""
        return run_simulation_return_log(
            seed=seed,
            tmpdir=tmpdir,
            essential_equipment=essential_equipment,
            other_equipment=None
        )

    # HSI_Event that requires equipment that is available --> will run
    assert did_hsi_run(
        get_sim(
            essential_equipment=set(str(equipment_item_code_that_is_available[0]))
        )
    )

    # HSI_Event that requires equipment that is not available --> will not run
    assert not did_hsi_run(
        get_sim(
            essential_equipment=set(str(equipment_item_code_that_is_not_available[0]))
        )
    )



