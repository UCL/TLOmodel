import datetime
import os
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    symptommanager, hiv, tb, male_circumcision, postnatal_supervisor
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
end_date = Date(2013, 1, 1)
popsize = 1000


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.group2
def test_run():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 #healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    sim.make_initial_population(n=popsize)
    params_ps = sim.modules['PregnancySupervisor'].parameters
    params_labour = sim.modules['Labour'].parameters

    #params_ps['prob_gest_htn_per_month'] = 1
    #params_ps['prob_pre_eclampsia_per_month'] = 1
    # params_ps['probability_htn_persists'] = 1
    # params_labour['prob_progression_gest_htn'] = 1
    # params_labour['prob_progression_mild_pre_eclamp'] = 1
    # params_labour['prob_progression_severe_pre_eclamp'] = 1

    sim.simulate(end_date=end_date)

    check_dtypes(sim)


@pytest.mark.group2
def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the Labour HSIs to ensure women
    who want to deliver in a facility, but cant, due to lacking capacity, have the correct events scheduled to continue
    their labour"""
    sim = Simulation(start_date=start_date)

    # Register the core modules

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 #healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


@pytest.mark.group2
def test_run_health_system_events_wont_run():
    """This test runs a simulation in which no scheduled HSIs will run.. Therefore it tests the logic in the
    not_available functions of the Labour HSIs to ensure women who want to deliver in a facility, but cant, due to the
    service being unavailble, have the correct events scheduled to continue their labour"""
    sim = Simulation(start_date=start_date)

    # Register the core modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 #healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

test_run()
# test_run_health_system_events_wont_run()
# test_run_health_system_high_squeeze()
