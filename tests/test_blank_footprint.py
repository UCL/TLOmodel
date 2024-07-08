import heapq as hp
import os
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.hsi_events import get_details_of_defined_hsi_events
from tlo.analysis.utils import get_filtered_treatment_ids, parse_log_file
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    mockitis,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.methods.consumables import Consumables, create_dummy_data_for_cons_availability
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthsystem import HealthSystem, HealthSystemChangeParameters
from tlo.methods.hsi_event import HSI_Event
from tlo.util import BitsetDType

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200




def test_all_appt_types_can_run(seed):
    """Check that if an appointment type is declared as one that can run at a facility-type of level `x` that it can
    run at the level for persons in any district."""

    # Create Dummy Module to host the HSI
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    # Create a dummy HSI event class
    class DummyHSIEvent_no_blank(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, appt_type, level):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'DummyHSIEvent'
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({appt_type: 1})
            self.ACCEPTED_FACILITY_LEVEL = level

            self.this_hsi_event_ran = False

    class DummyHSIEvent_with_blank(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id, appt_type, level):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'DummyHSIEvent'
            self.EXPECTED_APPT_FOOTPRINT = self.make_blank_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = level

            self.this_hsi_event_ran = False
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the core modules and simulate for 0 days
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=1,
                                           use_funded_or_actual_staffing='funded_plus'),
                 # <-- hard constraint (only HSI events with no squeeze factor can run)
                 # <-- using the 'funded_plus' number/distribution of officers
                 DummyModule()
                 )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)

    # Get pointer to the HealthSystemScheduler event
    healthsystemscheduler = sim.modules['HealthSystem'].healthsystemscheduler

    # For each type of appointment, for a person in each district, create the HSI, schedule the HSI and check it runs
    error_msg = list()

    def check_appt_works(district, level, appt_type):
        sim.modules['HealthSystem'].reset_queue()

        hsi_no_blank = DummyHSIEvent_no_blank(module=sim.modules['DummyModule'],
                            person_id=0,
                            appt_type=appt_type,
                            level=level)
        hsi_no_blank.make_appt_footprint({})
        print(hsi_no_blank.appt_footprint)

        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_no_blank,
            topen=sim.date,
            tclose=sim.date + pd.DateOffset(days=1),
            priority=1
        )
        hsi_blank = DummyHSIEvent_with_blank(module=sim.modules['DummyModule'],
                            person_id=0,
                            appt_type=appt_type,
                            level=level)
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_blank,
            topen=sim.date,
            tclose=sim.date + pd.DateOffset(days=1),
            priority=1
        )
        print(hsi_blank.appt_footprint)

        healthsystemscheduler.apply(sim.population)



    if len(error_msg):
        for _line in error_msg:
            print(_line)

    assert 0 == len(error_msg)

