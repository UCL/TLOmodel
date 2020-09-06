"""Test for for the HIV Module."""

import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malecircumcision,  # todo - remove dependency
    tb, # todo - think about removing dependency
    labour,
    pregnancy_supervisor,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def check_configuration_of_properties(sim):
    """check that the properties are ok"""
    df = sim.population.props

    # # todo - check properties


    #  check config of         "hv_status" and "hiv_art" and "hv_on_cotrim"
    # check "hv_date_inf" is consistent with hv_status and consistent with data and date of birth

    #hv_is_sexworker only for women and proportion is low.
    # hv_is_circ only for men?
    #

    #check that tested, diagnosed, hiv infected and ART startus make sense


    # ANALYSES: BASELINE HIV PREVALENCE, BASELINE ART COVERAGE,
    # ANALYSES: INCIDENCE RATE AND





    # # assert ((df.sex == 'M') & (df.hv_sexual_risk == 'low')).all()  # no sex work
    # assert not any((df.sex == "M") & (df.hv_sexual_risk == "sex_work"))
    #
    # assert not ((df.hv_number_tests >= 1) & ~df.hv_ever_tested).any()
    #
    # assert not (df.mc_is_circumcised & (df.sex == "F")).any()
    #
    # # check if HIV-TB co-infected, hv_specific_symptoms=aids
    # assert not any(df.tb_diagnosed & df.hv_inf & (df.hv_specific_symptoms == "none"))
    #
    # # only on cotrim if hiv is diagnosed [hv_date_cotrim = DATE and hv_diagnosed = True]
    # assert not any(df.hv_date_cotrim.notnull() & ~df.hv_diagnosed)


def test_basic_run_with_default_parameters():
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)


# -- OTHER TESTS PLANNED --

# todo test that if everyone on ART --- no new infections
