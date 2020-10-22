import os
import time
from pathlib import Path
import numpy as np
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    rti,
    symptommanager,
    healthseekingbehaviour,
    contraception,
    depression,
    epi,
    epilepsy,
    hiv,
    tb,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    pregnancy_supervisor,
    male_circumcision,
    Metadata
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    return sim


def test_run():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'

    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the RTI HSIs"""
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_health_system_events_wont_run():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_with_more_modules():
    sim = Simulation(start_date=start_date)

    # Register the core modules
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_sim_high_incidence():
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    orig_inc = params['base_rate_injrti']
    params['base_rate_injrti'] = orig_inc * 10
    params['allowed_interventions'] = 'none'

    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_tiny_population():
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)
    # Note that when n=1 an error was thrown up by the enhanced_lifestyle module when calculating bmi
    sim.make_initial_population(n=2)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

def test_no_capabilities():
    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 )

    sim.seed_rngs(0)
    # Note that when n=1 an error was thrown up by the enhanced_lifestyle module when calculating bmi
    sim.make_initial_population(n=popsize)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = 'none'
    sim.simulate(end_date=end_date)

    check_dtypes(sim)
if __name__ == '__main__':
    t0 = time.time()
    test_run()
    t1 = time.time()
    print('Time taken', t1 - t0)
