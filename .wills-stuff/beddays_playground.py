import os
import shutil
from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation, logging
from tlo.methods import demography, healthsystem, Metadata
from tlo.events import RegularEvent, PopulationScopeEventMixin, IndividualScopeEventMixin
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.bed_days import BedDays

start_date = Date(2010, 1, 1)
seed = 83563095832589325021
_bed_type = 'general_bed'
days_simulation_duration = 20
resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

tmpdir = "./.beddays_tmp"
os.makedirs(tmpdir, exist_ok=True)

class DummyModule(Module):
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        # Schedule event that will query the status of the property 'is_inpatient' each day
        self.sim.schedule_event(
            QueryInPatientStatus(self),
            self.sim.date
        )
        self.in_patient_status = pd.DataFrame(
            index=pd.date_range(self.sim.start_date,
                                self.sim.start_date + pd.DateOffset(days=days_simulation_duration)
                                ),
            columns=[0, 1],
            data=False
        )

        # Schedule person_id=0 and person_id=1 to attend care on 3rd January for 10 days
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Dummy(self, person_id=0),
            topen=Date(2010, 1, 3),
            tclose=None,
            priority=0)

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Dummy(self, person_id=1),
            topen=Date(2010, 1, 3),
            tclose=None,
            priority=0)

        # Schedule person_id=0 to die on 6th January
        self.sim.schedule_event(
            demography.InstantaneousDeath(self.sim.modules['Demography'], 0, 'Other'),
            Date(2010, 1, 6)
        )

class QueryInPatientStatus(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(days=1))

    def apply(self, population):
        self.module.in_patient_status.loc[self.sim.date] = \
            population.props.loc[[0, 1], 'hs_is_inpatient'].values

# Create a dummy HSI with both-types of Bed Day specified
class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'Dummy'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({_bed_type: 10})

    def apply(self, person_id, squeeze_factor):
        pass

if __name__ == "__main__":

    explore_rework = True

    # Create simulation with the health system and DummyModule
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            'tlo.methods.healthsystem': logging.INFO,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, beds_availability='all'),
        DummyModule()
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date + pd.DateOffset(days=days_simulation_duration))

    #  Debugging code here now now
    bd: BedDays = sim.modules["HealthSystem"].bed_days

    print(dir(bd))

    shutil.rmtree(tmpdir)
