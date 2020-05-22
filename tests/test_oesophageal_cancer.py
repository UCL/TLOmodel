import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophagealcancer,
    pregnancy_supervisor,
    labour,
    healthseekingbehaviour,
    symptommanager
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


def make_simulation_healthsystemdisabled():

    start_date = Date(2010, 1, 1)
    popsize = 5000

    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

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
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    return sim

def make_simulation_nohsi():

    start_date = Date(2010, 1, 1)
    popsize = 5000

    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    return sim




def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = \
        [0.0] * len(sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'])
    return sim

def incr_rates_of_progression_and_effect_of_treatment(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.05

    # Rates of cancer progression per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5

    # Effect of treatment in reducing progression: set so that treatment prevent progression
    sim.modules['OesophagealCancer'].parameters['rr_high_grade_dysp_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage1_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage2_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage3_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage4_undergone_curative_treatment'] = 0.0

    return sim

def test_initial_configuration_of_population():
    """Tests of the the way the population is configured"""
    sim = make_simulation_healthsystemdisabled()
    test_dtypes(sim)

    # Further tests:


def test_run_from_zero_init():
    """Tests on the population following simulation"""
    sim = make_simulation_healthsystemdisabled()
    sim = zero_out_init_prev(sim)
    end_date = Date(2020, 1, 1)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # Further tests:

def test_run_from_nonzero_init():
    """Tests on the population following simulation"""
    sim = make_simulation_healthsystemdisabled()
    end_date = Date(2020, 1, 1)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # Further tests:

def test_run_from_zero_init_nohsi():
    """Tests on the population following simulation"""
    sim = make_simulation_nohsi()
    sim = zero_out_init_prev(sim)
    end_date = Date(2020, 1, 1)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # Further test


def test_run_from_nonzero_init_nohsi():
    """Tests on the population following simulation"""
    sim = make_simulation_nohsi()
    end_date = Date(2020, 1, 1)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # Further tests:



def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()






"""Other tests:

* that the ca_oesophagus and ca_oesophagus_any properties alwasy correspond - initiatio or after simulation

* that no one has oes_cancer at age less than 20 -- initiation or after simulation

* treamtnt is none for all those w/o cancer and w/o diagnossi

* that treatment will lead to a piling up of people in a particular stage

* That the dates of thing in the properites are in right order where appropriate: ca_date_oes_cancer_diagnosis < ca_date_treatment_oesophageal_cancer < ca_date_palliative_care

* To check the working of the HSI, good shortcut would be to increase the baseline risk of cancer:

* at initiation, that date diagnosed is consistent with the age of the person

* at initiation, that not treatment or dianogsis for those with none stages

* check that treatment reduced risk of progression

* check that progression works:

* no dx w/o health system

* lots of dx, treatment etc w/ health system

*

"""
