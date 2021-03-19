import os
from pathlib import Path

from tlo import Simulation, Date
from tlo.methods import (
    demography,
    simplified_births,
    enhanced_lifestyle,
    healthsystem,
    symptommanager,
    healthseekingbehaviour,
    healthburden,
    oesophagealcancer,
    bladder_cancer,
    diarrhoea,
    epilepsy,
    hiv,
    malaria,
    tb
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 100


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim():
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.Simplifiedbirths(resourcefilepath=resourcefilepath)
                 )

    # Make the population
    sim.make_initial_population(n=popsize)
    return sim


def test_simplified_births_module_with_other_modules():
    """this is a test to see whether we can use simplified_births module inplace of contraception, labour and
    pregnancy supervisor modules"""
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.Simplifiedbirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_breastfeeding_simplified_birth_logic():
    """This is a simple test to ensure that breastfeeding status is applied to all newly generated individuals on
     birth"""
    sim = get_sim()
    initial_pop_size = 100
    sim.make_initial_population(n=initial_pop_size)

    # increasing number of women likely to get pregnant by setting pregnancy probability to 1
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 1

    # Force the probability of exclusive breastfeeding to 1, no other 'types' should occur from births in this sim run
    sim.modules['Simplifiedbirths'].parameters['prob_breastfeeding_type'] = [0, 0, 1]

    # Run the sim until end date, clear event queue
    sim.simulate(end_date=end_date)
    sim.event_queue.queue.clear()

    # define the dataframe
    df = sim.population.props

    # selecting the number of women who have given birth during simulation and compare it to the number of newborn
    number_of_women_ever_pregnant = df.loc[(df.sex == 'F') & (df.si_date_of_delivery < end_date)]
    number_of_ever_newborns = df.loc[df.date_of_birth.notna() & (df.mother_id >= 0)]

    # check that there are births happening
    assert len(number_of_ever_newborns) > 0

    # check that the number of women hasn't exceeded the number of newborns
    assert len(number_of_women_ever_pregnant) <= len(number_of_ever_newborns)

    # Finally we check to make sure all newborns have their breastfeeding status set to exclusive
    assert (df.loc[number_of_ever_newborns.index, 'nb_breastfeeding_status'] == 'exclusive').all().all()
