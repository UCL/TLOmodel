import os
from pathlib import Path

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


# def test_simplified_births_module_with_other_modules():
#     """this is a test to see whether we can use simplified_births module inplace of contraception, labour and
#     pregnancy supervisor modules"""
#     sim = Simulation(start_date=start_date, seed=0)
#
#     # Register the appropriate modules
#     sim.register(demography.Demography(resourcefilepath=resourcefilepath),
#                  simplified_births.Simplifiedbirths(resourcefilepath=resourcefilepath),
#                  enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#                  healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
#                  symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#                  healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#                  healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#                  oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
#                  bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
#                  diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
#                  epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
#                  hiv.Hiv(resourcefilepath=resourcefilepath),
#                  malaria.Malaria(resourcefilepath=resourcefilepath),
#                  tb.Tb(resourcefilepath=resourcefilepath)
#                  )
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=end_date)
#     check_dtypes(sim)


def check_property_integrity(sim):
    # check that the properties are as expected

    # define dataframe
    df = sim.population.props

    # check that individuals who are dead do not become pregnant
    assert not (~df.is_alive & df.is_pregnant).any(), 'a dead person can not become pregnant'

    # check that a date of last pregnancy is not assigned to individuals who are not pregnant
    assert pd.isnull(df.loc[~df.is_pregnant, 'date_of_last_pregnancy']).all(), "date of last pregnancy assigned to a " \
                                                                               "female not pregnant"

    # check that no date of delivery is assigned to individuals who are not pregnant
    assert pd.isnull(df.loc[~df.is_pregnant, 'si_date_of_delivery']).all(), "date of delivery assigned to a " \
                                                                            "female not pregnant"

    # check that all pregnant women have been assigned a date of last pregnancy
    assert not pd.isnull(df.loc[df.is_pregnant, 'date_of_last_pregnancy']).any(), "date of last pregnancy not " \
                                                                                  "assigned to a pregnant woman"
    # check that all pregnant women have been assigned a date of delivery
    assert not pd.isnull(df.loc[df.is_pregnant, 'si_date_of_delivery']).any(), "date of delivery not " \
                                                                               "assigned to a pregnant woman"


def test_pregnancy_logic_at_max_pregnancy_probability():
    # a test to check whether pregnancies are happening as expected
    sim = get_sim()
    initial_pop_size = 1
    sim.make_initial_population(n=initial_pop_size)

    # increasing number of women likely to get pregnant by setting pregnancy probability to 1
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 1

    # define dataframe
    df = sim.population.props
    df.at[df.index, 'age_years'] = 40
    df.at[df.index, 'sex'] = 'F'

    # number of eligible females before pregnancy event
    eligible_females_before_pregnancy_event = df

    # check property configuration before any event is run
    check_property_integrity(sim)

    # check population to see if anyone gets pregnant before pregnancy Event is run
    assert not eligible_females_before_pregnancy_event.is_pregnant.any()

    # Run the Simplified Pregnancy Event on the selected population
    pregnancy_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    pregnancy_event.apply(df)

    # get the number of females who got pregnant
    pregnant_females_after_pregnancy_event = df.loc[df.is_pregnant]

    """ since we have set pregnancy probability at 1 then all eligible individuals should get pregnant
    after Pregnancy Event has been fired """
    assert len(eligible_females_before_pregnancy_event) == len(pregnant_females_after_pregnancy_event)

    # check property configuration after an event is run
    check_property_integrity(sim)

    # check woman's date of delivery if correct assert (pd.to_datetime(sim.date + pd.DateOffset(months=9)).date() ==
    # df.loc[df.index, 'si_date_of_delivery']).all()


def test_run_pregnancy_logic_at_zero_pregnancy_probability():
    # running pregnancy event with zero pregnancy probability
    sim = get_sim()
    initial_pop_size = 1
    sim.make_initial_population(n=initial_pop_size)

    # ensuring no pregnancies happen by setting pregnancy probability to zero
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 0

    # define dataframe
    df = sim.population.props
    df.at[df.index, 'age_years'] = 40
    df.at[df.index, 'sex'] = 'F'

    # number of eligible females to get pregnant
    eligible_pop = df

    # check property configuration before any event is run
    check_property_integrity(sim)

    # check population to see if anyone gets pregnant before pregnancy Event is run
    assert not eligible_pop.is_pregnant.any(), "an individual can not become pregnant before pregnant event is run"

    # Run the Simplified Pregnancy Event on the selected population
    pregnancy_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    pregnancy_event.apply(df)

    # get the number of females who got pregnant
    pregnancies_after_pregnancy_event = df.loc[df.is_pregnant]

    """ since we have set pregnancy probability to zero confirm we have no pregnancies"""
    assert len(pregnancies_after_pregnancy_event) == 0, "There can not be pregnancies when probability is set to zero"

    # check property configuration after an event is run
    check_property_integrity(sim)


def test_pregnancy_logic_on_a_dead_individual():
    # running pregnancy event on a dead population
    sim = get_sim()
    initial_pop_size = 1
    sim.make_initial_population(n=initial_pop_size)

    # ensuring no pregnancies happen by setting pregnancy probability to zero
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 1

    # select population
    df = sim.population.props

    # modify person's age and sex
    df.at[df.index, 'age_years'] = 20
    df.at[df.index, 'sex'] = 'F'

    # kill the person
    df.at[df.index, 'is_alive'] = False

    # check property configuration before any event is run
    check_property_integrity(sim)

    # Run the Simplified Pregnancy Event on the dead population
    pregnancy_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    pregnancy_event.apply(df)

    # get the number of females who got pregnant
    pregnancies_after_pregnancy_event = df.loc[df.is_pregnant]

    """ since we are running pregnancy event on a dead population, no one should get pregnant"""
    assert len(pregnancies_after_pregnancy_event) == 0, "a dead person can not become pregnant"

    # check property configuration after an event is run
    check_property_integrity(sim)


def test_pregnancy_logic_on_males():
    sim = get_sim()
    initial_pop_size = 1
    sim.make_initial_population(n=initial_pop_size)

    # ensuring no pregnancies happen by setting pregnancy probability to zero
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 1

    # select population
    df = sim.population.props

    # modify person's age and sex
    df.at[df.index, 'age_years'] = 30
    df.at[df.index, 'sex'] = 'M'

    # check property configuration before any event is run
    check_property_integrity(sim)

    # Run the Simplified Pregnancy event on population
    pregnancy_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    pregnancy_event.apply(df)

    # get the number of females who got pregnant
    pregnancies_after_pregnancy_event = df.loc[df.is_pregnant]

    assert len(pregnancies_after_pregnancy_event) == 0, "a male person can not become pregnant"

    # check property configuration after an event is run
    check_property_integrity(sim)


def test_pregnancy_logic_on_an_individual_whose_age_is_outside_pregnancy_range():
    sim = get_sim()
    initial_pop_size = 1
    sim.make_initial_population(n=initial_pop_size)

    # ensuring no pregnancies happen by setting pregnancy probability to zero
    sim.modules['Simplifiedbirths'].parameters['pregnancy_prob'] = 1

    # select population
    df = sim.population.props

    # modify person's age and sex
    df.at[df.index, 'age_years'] = 10
    df.at[df.index, 'sex'] = 'F'

    # check property configuration before any event is run
    check_property_integrity(sim)

    # Run the Simplified Pregnancy event on population
    pregnancy_event = simplified_births.SimplifiedPregnancyEvent(module=sim.modules['Simplifiedbirths'])
    pregnancy_event.apply(df)

    # get the number of females who got pregnant
    pregnancies_after_pregnancy_event = df.loc[df.is_pregnant]

    assert len(pregnancies_after_pregnancy_event) == 0, "a person of that age can not become pregnant"

    # check property configuration after an event is run
    check_property_integrity(sim)


def test_breastfeeding_simplified_birth_logic():
    """This is a simple test to ensure that breastfeeding status is applied to all newly generated individuals on
     birth"""
    sim = get_sim()
    initial_pop_size = 100
    sim.make_initial_population(n=initial_pop_size)

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
