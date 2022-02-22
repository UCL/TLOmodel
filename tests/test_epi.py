import os
from datetime import datetime
from pathlib import Path

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

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
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


# checking no vaccines administered through health system
# only hpv should stay at zero, other vaccines start as individual events (year=2010-2018)
# coverage should gradually decline for all after 2018
# hard constraints (mode=2) and zero capabilities
@pytest.mark.slow
@pytest.mark.group2
def test_no_health_system(tmpdir):
    log_config = {
        'filename': 'test_log',
        'directory': tmpdir,
        'custom_levels': {"*": logging.FATAL, "tlo.methods.epi": logging.INFO}
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable_and_reject_all=True
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check we can read the results
    parse_log_file(sim.log_filepath)
    df = sim.population.props

    # check no vaccines being administered through health system
    # only hpv currently, all others start as individual events
    assert (df.va_hpv == 0).all()

    # check all infants born after Jan 2019 have no bcg / penta etc. through HSIs
    assert not ((df.va_bcg > 0) & (df.date_of_birth > datetime(2019, 1, 1))).any()
    assert not ((df.va_dtp > 0) & (df.date_of_birth > datetime(2019, 1, 1))).any()


# check epi module does schedule hsi events
@pytest.mark.slow
@pytest.mark.group2
def test_epi_scheduling_hsi_events(tmpdir):

    log_config = {
        'filename': 'test_log',
        'directory': tmpdir,
        'custom_levels': {"*": logging.FATAL, "tlo.methods.epi": logging.INFO}
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

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

    # check vaccine coverage is above zero for all vaccine types post 2019
    # 2010-2018 vaccines administered through individual events
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
    assert (ep_out.epRubellaCoverage > 0).any()
    assert (ep_out.epHpvCoverage > 0).any()

    # check only 3 doses max of dtp/pneumo
    assert (df.va_dtp <= 3).all()
    assert (df.va_pneumo <= 3).all()


@pytest.mark.slow
def test_all_doses_properties():
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

    sim = Simulation(start_date=start_date)
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
