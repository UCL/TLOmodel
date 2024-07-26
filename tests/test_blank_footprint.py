import logging
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

# Set up logging configuration
log_file_path = 'simulation.log'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
                    filename=log_file_path,
                    filemode='w')

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
facility_info_0 = 0
facility_id = 0
blank_footprint = True
did_run=True
def find_level_of_facility_id(facility_id: int) -> str:
    """Returns the level of a Facility_ID"""
    return mfl.set_index('Facility_ID').loc[facility_id].Facility_Level


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


def get_dummy_hsi_event_instance(module, facility_id=None, blank = False):
    """Make an HSI Event that runs for person_id=0 in a particular facility_id and requests consumables,
    and for which its parent is the identified module."""

    _facility_level = find_level_of_facility_id(facility_id)

    class HSI_Dummy_no_blank(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = _facility_level
            self._facility_id = facility_id
            self.blank = blank_footprint
            if self.blank:
                self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            else:
                self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})



        def apply(self, person_id, squeeze_factor):
            if squeeze_factor != np.inf:
                # Check that this appointment is being run and run not with a squeeze_factor that signifies that a cadre
                # is not at all available.
                self.did_run = True


    hsi_dummy = HSI_Dummy_no_blank(module=module, person_id=0)
    hsi_dummy.initialise()
    hsi_dummy.facility_info = module.sim.modules['HealthSystem']._facility_by_facility_id[facility_id]
    return hsi_dummy


def get_dummy_hsi_event_instance_blank_footprint(module, facility_id=None):
    """Make an HSI Event that runs for person_id=0 in a particular facility_id and requests consumables,
    and for which its parent is the identified module."""

    _facility_level = find_level_of_facility_id(facility_id)

    class HSI_Dummy_with_blank(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.ACCEPTED_FACILITY_LEVEL = _facility_level
            self.EXPECTED_APPT_FOOTPRINT = self.make_blank_appt_footprint()
            self._facility_id = facility_id

        def apply(self, person_id, squeeze_factor):
            if squeeze_factor != np.inf:
                # Check that this appointment is being run and run not with a squeeze_factor that signifies that a cadre
                # is not at all available.
                self.did_run = True
        def make_blank_appt_footprint(self):
            return {}

    hsi_dummy = HSI_Dummy_with_blank(module=module, person_id=0)
    hsi_dummy.initialise()
    hsi_dummy.facility_info = module.sim.modules['HealthSystem']._facility_by_facility_id[facility_id]
    return hsi_dummy
def set_person_district_id(sim):
      # sets facility and person to 0
      sim.population.props.at[0, 'district_of_residence'] = mfl.set_index('Facility_ID').loc[0].District
      return sim

def test_outputs_to_log_no_blank(tmpdir):
    """Check that logging from Consumables is as expected."""
    intrinsic_availability = {0: 1.0, 1: 0.0}

    sim = get_sim_with_dummy_module_registered(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability=intrinsic_availability,
            months=[1],
            facility_ids=[0]),
        tmpdir=tmpdir,
        run=False
    )

   # Edit the `initialise_simulation` method of DummyModule so that, during the simulation, an HSI is run.
    def schedule_hsi(sim):
        """Drop-in replacement for `initialise_simulation` in the DummyModule module."""
        # Make the district for person_id=0 such that the HSI will be served by facility_id=0
        sim = set_person_district_id(sim)

        # Schedule the HSI event with a blank footprint for person_id=0
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=get_dummy_hsi_event_instance(module=sim.modules['DummyModule'], facility_id=0, blank = blank_footprint),
            topen=sim.start_date,
            tclose=None,
            priority=0
        )


    sim.modules['DummyModule'].initialise_simulation = schedule_hsi

        # Simulate for one day
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=1))

        # Check that log is created and the content is as expected.
    hsi_log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem.summary']['hsi_event_details']
    assert hsi_log.iloc[0,1]['0']['appt_footprint'] == []




def test_outputs_to_log_blank(tmpdir):
    """Check that logging from Consumables is as expected."""
    intrinsic_availability = {0: 1.0, 1: 0.0}

    sim = get_sim_with_dummy_module_registered(
        data=create_dummy_data_for_cons_availability(
            intrinsic_availability=intrinsic_availability,
            months=[1],
            facility_ids=[0]),
        tmpdir=tmpdir,
        run=False
    )

    # Edit the `initialise_simulation` method of DummyModule so that, during the simulation, an HSI is run.
    def schedule_hsi(sim):
        """Drop-in replacement for `initialise_simulation` in the DummyModule module."""
        # Make the district for person_id=0 such that the HSI will be served by facility_id=0
        sim = set_person_district_id(sim)

        # Schedule the HSI event with no blank footprint for person_id=0
        sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=get_dummy_hsi_event_instance_blank_footprint(module=sim.modules['DummyModule'], facility_id=0),
            topen=sim.start_date,
            tclose=None,
            priority=0
        )

    sim.modules['DummyModule'].initialise_simulation = schedule_hsi

    # Simulate for one day
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=1))

    # Check that log is created and the content is as expected.
    hsi_log = parse_log_file(sim.log_filepath)['tlo.methods.healthsystem.summary']['hsi_event_details']
    assert hsi_log.iloc[0,1]['0']['appt_footprint'] == []

