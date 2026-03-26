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
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthsystem import HealthSystem, HealthSystemChangeParameters
from tlo.methods.hsi_event import HSI_Event
from tlo.util import BitsetDType

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 2)
popsize = 200

"""
Test whether the system runs under multiple configurations of the healthsystem.

This test file is focussed on the overall function of the module and its behaviour in Mode 1.
"""


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


## TEST DELAYS
def test_weather_disruption_delays_hsi_by_correct_days(seed, tmpdir):
    """An HSI disrupted by weather should be rescheduled and run 28 days later."""

    from tlo.methods import weather_disruptions

    class DummyHSI(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSI"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyModule(Module):
        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.modules["HealthSystem"].schedule_hsi_event(
                DummyHSI(self, person_id=0),
                topen=sim.date,
                tclose=None,
                priority=1,
            )

    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {
            "tlo.methods.healthsystem": logging.DEBUG,
            "tlo.methods.weather_disruptions": logging.INFO,
        },
    }

    sim = Simulation(
        start_date=Date(2025, 1, 1), seed=seed, log_config=log_config, resourcefilepath=resourcefilepath
    )

    wd = weather_disruptions.WeatherDisruptions(
        services_affected_precip="all",
        year_effective_climate_disruptions=2025,
        scale_factor_prob_disruption=1.0,  # ensure disruption always occurs
        delay_in_seeking_care_weather=28.0,  # 28-day delay
        scale_factor_severity_disruption_and_delay=1.0,
        scale_factor_reseeking_healthcare_post_disruption=100.0,  # ensure always reseeks
        prop_supply_side_disruptions=0.0,  # always demand-side, simpler
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            capabilities_coefficient=10000.0,  #
            cons_availability="all",
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            force_any_symptom_to_lead_to_healthcareseeking=True
        ),
        symptommanager.SymptomManager(),
        wd,
        DummyModule(),
        check_all_dependencies=False,
    )

    # Override disruption probability to 1.0 for all facilities
    # after build_disruption_probabilities has run
    sim.make_initial_population(n=100)

    sim.modules["WeatherDisruptions"].parameters["scale_factor_appointment_urgency"] = 1.0

    disruption_fired = [False]  # if p(disruption) = 1, it will keep firing

    def always_disrupted_once(hsi_event_item, current_date):
        if disruption_fired[0]:
            return False, False
        disruption_fired[0] = True
        wd = sim.modules["WeatherDisruptions"]
        district = sim.population.props.at[
            hsi_event_item.hsi_event.target, "district_of_residence"
        ]
        treatment_id = getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown")
        is_supply_side = wd._handle_disruption(
            hsi_event_item=hsi_event_item,
            prob_disruption=1.0,
            current_date=current_date,
            facility_id="test_facility",
            district=str(district),
            treatment_id=str(treatment_id),
        )
        return True, is_supply_side

    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = always_disrupted_once

    original_date = sim.start_date
    expected_run_date = original_date + pd.DateOffset(days=35)  # 28 base + 7 window

    sim.simulate(end_date=expected_run_date + pd.DateOffset(days=5))

    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)["tlo.methods.healthsystem"]["HSI_Event"]
    dummy_log = log[log["TREATMENT_ID"] == "DummyHSI"]

    assert len(dummy_log) == 1, "HSI should have run exactly once (after rescheduling)"
    assert dummy_log["date"].iloc[0] == expected_run_date, (
        f"Expected HSI to run on {expected_run_date}, "
        f"but ran on {dummy_log['date'].iloc[0]}"
    )


## TEST LOGGING

def test_summary_logger_for_never_ran_hsi_event(seed, tmpdir):
    """Check that under a mode_appt_constraints = 2 with zero resources, HSIs with a tclose
    soon after topen will be correctly recorded in the summary logger, and that this can
    be parsed correctly when a different set of HSI are never ran."""

    # Create a dummy disease module (to be the parent of the dummy HSI)
    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # In 2010: Dummy1 only
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy1(self, person_id=0),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=2),
                priority=0,
            )
            # In 2011: Dummy2 & Dummy3
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy2(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(years=1),
                tclose=self.sim.date + pd.DateOffset(years=1) + pd.DateOffset(days=2),
                priority=0,
            )
            sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Dummy3(self, person_id=0),
                topen=self.sim.date + pd.DateOffset(years=1),
                tclose=self.sim.date + pd.DateOffset(years=1) + pd.DateOffset(days=2),
                priority=0,
            )

    # Create two different dummy HSI events:
    class HSI_Dummy1(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy1"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy2(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy2"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class HSI_Dummy3(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "Dummy3"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1b"

        def apply(self, person_id, squeeze_factor):
            pass

    # Set up simulation:
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            "filename": "tmpfile",
            "directory": tmpdir,
            "custom_levels": {
                "tlo.methods.healthsystem": logging.DEBUG,
                "tlo.methods.healthsystem.summary": logging.INFO,
            },
        },
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=2,
            capabilities_coefficient=0.0,  # <--- Ensure all events postponed
        ),
        DummyModule(),
        sort_modules=False,
        check_all_dependencies=False,
    )
    sim.make_initial_population(n=1000)

    sim.simulate(end_date=start_date + pd.DateOffset(years=2))
    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)

    # Summary log:
    summary_hsi_event = log["tlo.methods.healthsystem.summary"]["Never_ran_HSI_Event"]
    # In 2010, should have recorded one instance of Dummy1 having never ran
    assert summary_hsi_event.loc[summary_hsi_event["date"] == Date(2010, 12, 31), "TREATMENT_ID"][0] == {"Dummy1": 1}
    # In 2011, should have recorded one instance of Dummy2 and one of Dummy3 having never ran
    assert summary_hsi_event.loc[summary_hsi_event["date"] == Date(2011, 12, 31), "TREATMENT_ID"][1] == {
        "Dummy2": 1,
        "Dummy3": 1,
    }
