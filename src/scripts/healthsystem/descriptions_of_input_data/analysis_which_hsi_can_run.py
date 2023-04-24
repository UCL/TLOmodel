"""This script produces a pd.DataFrame that shows which ApptTypes can run at each facility, under the different modes
 of the HealthSystem and the different assumptions for the HR resources."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from tlo import Date, Module, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import Metadata, demography, healthsystem
from tlo.methods.healthsystem import HSI_Event

resourcefilepath = Path('resources')
outputpath = Path('outputs')


class DummyModule(Module):
    """Dummy Module to host the HSI"""
    METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHSYSTEM}

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, appt_type, level):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = 'DummyHSIEvent'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({appt_type: 1})
        self.ACCEPTED_FACILITY_LEVEL = level

        self.this_hsi_event_ran = False
        self.squeeze_factor_of_this_hsi = None

    def apply(self, person_id, squeeze_factor):
        self.squeeze_factor_of_this_hsi = squeeze_factor
        self.this_hsi_event_ran = True


# For each Mode and assumption on HR resources, test whether each type of appointment can run in each district at each
# level for which it is defined.

results = list()

for mode_appt_constraints in (0, 1, 2):
    for use_funded_or_actual_staffing in ('actual', 'funded', 'funded_plus'):
        sim = Simulation(start_date=Date(2010, 1, 1), seed=0)

        # Register the core modules and simulate for 0 days
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                               capabilities_coefficient=1.0,
                                               mode_appt_constraints=mode_appt_constraints,
                                               use_funded_or_actual_staffing=use_funded_or_actual_staffing),
                     DummyModule(),
                     )
        sim.make_initial_population(n=100)
        sim.simulate(end_date=sim.start_date)

        # Get pointer to the HealthSystemScheduler event
        healthsystemscheduler = sim.modules['HealthSystem'].healthsystemscheduler

        # Get the table showing which types of appointment can occur at which level
        appt_types_offered = sim.modules['HealthSystem'].parameters['Appt_Offered_By_Facility_Level'].set_index(
            'Appt_Type_Code')

        # Get the all the districts in which a person could be resident, and allocate one person to each district
        person_for_district = {d: i for i, d in enumerate(sim.population.props['district_of_residence'].cat.categories)}
        sim.population.props.loc[person_for_district.values(), 'is_alive'] = True
        sim.population.props.loc[person_for_district.values(), 'district_of_residence'] = list(
            person_for_district.keys())


        def check_appt_works(district, level, appt_type) -> Tuple:
            sim.modules['HealthSystem'].reset_queue()

            hsi = DummyHSIEvent(
                module=sim.modules['DummyModule'],
                person_id=person_for_district[district],
                appt_type=appt_type,
                level=level
            )

            sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=hsi,
                topen=sim.date,
                tclose=sim.date + pd.DateOffset(days=1),
                priority=1
            )

            healthsystemscheduler.run()

            return hsi.this_hsi_event_ran, hsi.squeeze_factor_of_this_hsi


        for _district in person_for_district:
            for _facility_level_col_name in appt_types_offered.columns:
                for _appt_type in (
                    appt_types_offered[_facility_level_col_name].loc[appt_types_offered[_facility_level_col_name]].index
                ):
                    _level = _facility_level_col_name.split('_')[-1]
                    hsi_did_run, sqz = check_appt_works(district=_district, level=_level, appt_type=_appt_type)

                    results.append(dict(
                        mode_appt_constraints=mode_appt_constraints,
                        use_funded_or_actual_staffing=use_funded_or_actual_staffing,
                        level=_level,
                        appt_type=_appt_type,
                        district=_district,
                        hsi_did_run=hsi_did_run,
                        sqz=sqz,
                    ))

        results = pd.DataFrame(results)



# print to console
print(results)

# copy to clipboard
results.to_clipboard(excel=True)

# save to a file
results.to_csv(outputpath / 'which_hsi_can_run.csv')
