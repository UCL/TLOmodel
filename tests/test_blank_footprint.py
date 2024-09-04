import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Module, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.consumables import create_dummy_data_for_cons_availability
from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

def run_simulation_and_return_healthsystem_summary_log(tmpdir: Path, blank_footprint: bool) -> dict:
    """Return the `healthsystem.summary` logger for a simulation. In that simulation, there is HSI_Event run on the
    first day of the simulation and its `EXPECTED_APPT_FOOTPRINT` may or may not be blank. The simulation is run for one
    year in order that the summary logger is active (it runs annually)."""

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, _is_footprint_blank):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = '0'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({}) if blank_footprint \
                else self.make_appt_footprint({'ConWithDCSA': 1})

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            hsi_event = HSI_Dummy(module=self, person_id=0, _is_footprint_blank=blank_footprint)
            sim.modules['HealthSystem'].schedule_hsi_event(hsi_event=hsi_event, topen=sim.date, priority=0)

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0, log_config={'filename': 'tmp', 'directory': tmpdir})
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=0),
        DummyModule(),
        # Disable sorting + checks to avoid error due to missing dependencies
        sort_modules=False,
        check_all_dependencies=False
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(years=1))

    return parse_log_file(sim.log_filepath)['tlo.methods.healthsystem.summary']


def test_logging_of_only_hsi_events_with_non_blank_footprints(tmpdir):
    """Run the simulation with an HSI_Event that may have a blank_footprint and examine the healthsystem.summary logger.
     * If the footprint is blank, the HSI event should be recorded in the usual loggers but not the 'no_blank' logger
     * If the footprint is non-blank, the HSI event should be recorded in the usual and the 'no_blank' loggers.
     """

    # When the footprint is blank:
    log = run_simulation_and_return_healthsystem_summary_log(tmpdir, blank_footprint=True)
    assert log['HSI_Event']['TREATMENT_ID'].iloc[0] == {'Dummy': 1}  # recorded in usual logger
    assert log['HSI_Event_non_blank_appt_footprint']['TREATMENT_ID'].iloc[0] == {}  # not recorded in 'non-blank' logger

    # When the footprint is non-blank:
    log = run_simulation_and_return_healthsystem_summary_log(tmpdir, blank_footprint=False)
    assert (
        log['HSI_Event']['TREATMENT_ID'].iloc[0]
        == log['HSI_Event_non_blank_appt_footprint']['TREATMENT_ID'].iloc[0]
        == {'Dummy': 1}
        # recorded in both in the usual and the 'non-blank' logger
    )
