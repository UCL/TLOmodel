import os
from pathlib import Path

import pytest
import pandas as pd

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
end_date = Date(2011, 1, 1)
popsize = 1000


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim():
    start_date = Date(2010, 1, 1)
    popsize = 100
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

    # set delivery period to 0 days allowing pregnant women deliver same day
    # sim.modules['Simplifiedbirths'].parameters['days_until_delivery'] = pd.DateOffset(days=0)

    # Force the probability of exclusive breastfeeding to 1, no other 'types' should occur from births in this sim run
    sim.modules['Simplifiedbirths'].parameters['prob_breastfeeding_type'] = [0, 0, 1]

    # Run the sim until end date, clear event queue
    sim.simulate(end_date=end_date)
    sim.event_queue.queue.clear()

    # Run the Simplified Pregnancy Event on the selected population
    birth_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    birth_event.apply(sim.population.props)

    # define the dataframe
    df = sim.population.props

    # Ensure births are happening and the dataframe has grown
    assert len(df) > initial_pop_size

    # selecting the number of women who have given birth during simulation and compare it to the number of newborn
    selected_women = df.loc[(df.sex == 'F') & df.is_alive & df.is_pregnant & (df.si_date_of_delivery <= end_date)]
    new_borns = df.loc[df.is_alive & (df.mother_id >= 0)]

    # selected_women = df.loc[(df.sex == 'F') & df.is_pregnant & (df.si_date_of_delivery <= end_date)]
    # new_borns = df.loc[(df.mother_id >= 0)]
    # print(selected_women)
    # print(new_borns)

    assert len(selected_women) == len(new_borns)

    # Finally we check to make sure all newborns have their breastfeeding status set to exclusive
    assert (df.loc[new_borns.index, 'nb_breastfeeding_status'] == 'exclusive').all().all()
