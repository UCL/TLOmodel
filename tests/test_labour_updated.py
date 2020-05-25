import os
import datetime
from pathlib import Path


from tlo import Date, Simulation
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor, symptommanager, healthseekingbehaviour
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 5000


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = Simulation(start_date=start_date)

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the Labour HSIs to ensure women
    who want to deliver in a facility, but cant, due to lacking capacity, have the correct events scheduled to continue
    their labour"""
    sim = Simulation(start_date=start_date)

    # Register the core modules

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2))
    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_health_system_events_wont_run():
    """This test runs a simulation in which no scheduled HSIs will run.. Therefore it tests the logic in the
    not_available functions of the Labour HSIs to ensure women who want to deliver in a facility, but cant, due to the
    service being unavailble, have the correct events scheduled to continue their labour"""
    sim = Simulation(start_date=start_date)

    # Register the core modules

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                               service_availability=[]))
    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    # todo: some logic here to confirm that all women are now delivering at home due to high squeeze
