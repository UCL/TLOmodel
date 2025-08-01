import os
from pathlib import Path

import pandas as pd
import pytest
from pandas import DateOffset

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.epi import HSI_BcgVaccine, HSI_HpvVaccine, HSI_RotaVaccine

start_date = Date(2010, 1, 1)
end_date = Date(2021, 1, 1)
popsize = 500

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# check epi module does schedule hsi events
@pytest.mark.slow
@pytest.mark.group2
def test_epi_scheduling_hsi_events(tmpdir, seed):
    log_config = {
        'filename': 'test_log',
        'directory': tmpdir,
        'custom_levels': {"*": logging.FATAL, "tlo.methods.epi": logging.INFO}
    }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable=True
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)
    df = sim.population.props

    # check vaccine coverage is above zero for all vaccine types
    ep_out = output["tlo.methods.epi"]["ep_vaccine_coverage"]

    # check vaccine coverage is above 0 for all vaccine types
    assert (ep_out.epBcgCoverage > 0).any()
    assert (ep_out.epDtp3Coverage > 0).any()
    assert (ep_out.epOpv3Coverage > 0).any()
    assert (ep_out.epHib3Coverage > 0).any()
    assert (ep_out.epHep3Coverage > 0).any()
    assert (ep_out.epPneumo3Coverage > 0).any()
    assert (ep_out.epRota2Coverage > 0).any()
    assert (ep_out.epMeaslesCoverage > 0).any()
    assert (ep_out.epRubellaCoverage > 0).any()  # begins in 2018
    assert (ep_out.epHpvCoverage > 0).any()  # begins in 2019

    # check only 3 doses max of dtp/pneumo
    assert (df.va_dtp <= 3).all()
    assert (df.va_pneumo <= 3).all()


@pytest.mark.slow
def test_all_doses_properties(seed):
    """check alignment between "number of doses" properties and "all_doses" properties"""

    # Make Dummy class and event to check alignment of the properties:
    class DummyModule(Module):

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            self.sim.schedule_event(
                CheckProperties(self.sim.modules['Epi']),
                self.sim.date
            )

        def on_birth(self, mother, child):
            pass

    class CheckProperties(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(days=1))

        def apply(self, population):
            """This checks that there is an alignment between the properties for the number of doses received of each
            vaccine and the all_doses_received properties
            """
            df = self.sim.population.props
            for _vacc, _max in self.module.all_doses.items():
                properties_aligned = (
                    df.loc[df.is_alive, f"va_{_vacc}_all_doses"] == (df.loc[df.is_alive, f"va_{_vacc}"] >= _max)
                ).all()
                assert properties_aligned, f"On {self.sim.date} and for vaccine {_vacc}, there is a mismatch between" \
                                           f" the all-doses and number-of-doses."

    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
        epi.Epi(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        DummyModule()
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


# check distribution of facility levels for vaccines
def test_facility_level_distribution(tmpdir, seed):
    log_config = {
        'filename': 'test_log',
        'directory': tmpdir,
        'custom_levels': {"*": logging.FATAL,
                          "tlo.methods.epi": logging.INFO,
                          "tlo.methods.healthsystem.summary": logging.INFO}
    }

    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=["*"],  # all treatment allowed
            mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
            cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
            ignore_priority=False,  # do not use the priority information in HSI event to schedule
            capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
            use_funded_or_actual_staffing="funded_plus",
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    )

    # change distribution of vaccine delivery
    # make all vaccines occur at level 3
    sim.modules['Epi'].parameters['prob_facility_level_for_vaccine'] = [0, 0, 0, 1.0]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=Date(2014, 1, 1))
    check_dtypes(sim)

    # check we can read the results
    output = parse_log_file(sim.log_filepath)

    facilities = output["tlo.methods.healthsystem.summary"]["HSI_Event"]

    tmp = facilities.Number_By_Appt_Type_Code_And_Level

    t1 = pd.DataFrame(tmp.values.tolist())
    t2 = t1.set_axis(["level0", "level1a", "level1b", "level2", "level3", "level4"], axis=1)

    epi_levels = pd.DataFrame(columns=["level0", "level1a", "level1b", "level2", "level3", "level4"])

    for i in range(len(t2.index)):
        out = [d.get('EPI') for d in t2.iloc[i]]
        epi_levels.loc[i] = out
    epi_levels = epi_levels.fillna(0.0)

    assert epi_levels.level0.sum() == 0
    assert epi_levels.level1a.sum() == 0
    assert epi_levels.level1b.sum() == 0

    # assert all epi appts that have occurred are at level 2
    assert epi_levels.level2.sum() == epi_levels.sum().sum()

    assert epi_levels.level3.sum() == 0
    assert epi_levels.level4.sum() == 0


def test_hsi_epi_footprint(seed):
    """
    Test that the HSI for vaccine delivery is returning the correct footprint
    * as an example, test BCG and rota - both are given in bundles at different time-points
    and should have a footprint of 0.5 EPI
    * test HPV vaccine, should have full EPI footprint
    """

    popsize = 10
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='default'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 )

    # set up initial population
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # Get target person and eligible for a childhood vaccine
    person_id = 0
    df.at[person_id, "age_years"] = 0

    # Run the BCG vaccine event
    t = HSI_BcgVaccine(module=sim.modules['Epi'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check the footprint returned by this event
    assert t.EXPECTED_APPT_FOOTPRINT.get('EPI') == 0.5

    # Run the Rotavirus vaccine event
    t = HSI_RotaVaccine(module=sim.modules['Epi'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check the footprint returned by this event
    assert t.EXPECTED_APPT_FOOTPRINT.get('EPI') == 0.5

    # Run the HPV vaccine event - this should have one full EPI appt as footprint
    # Get target person and eligible for a childhood vaccine
    person_id = 1
    df.at[person_id, "age_years"] = 9

    t = HSI_HpvVaccine(module=sim.modules['Epi'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    # Check the footprint returned by this event
    assert t.EXPECTED_APPT_FOOTPRINT.get('EPI') == 1
