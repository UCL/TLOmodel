import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    demography,
    depression,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    cardio_metabolic_disorders,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    depression,
    tb,
    contraception,
    simplified_births,
    rti,
    symptommanager,
)
from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstAppt

# create simulation parameters
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200

@pytest.mark.slow
def test_data_harvesting(seed):
    """
    This test runs a simulation to print all individual events of specific individuals
    """
    
    module_of_interest = 'RTI'
    # create sim object
    sim = create_basic_sim(popsize, seed)
    
    dependencies_list = sim.modules[module_of_interest].ADDITIONAL_DEPENDENCIES.union(sim.modules[module_of_interest].INIT_DEPENDENCIES)
    
    # Check that all dependencies are included
    for dep in dependencies_list:
        if dep not in sim.modules:
            print("WARNING: dependency ", dep, "not included")
            exit(-1)

    # run simulation
    sim.simulate(end_date=end_date, generate_event_chains = True)
    exit(-1)

def create_basic_sim(population_size, seed):
    # create the basic outline of an rti simulation object
    sim = Simulation(start_date=start_date, seed=seed)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
               # contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
                 rti.RTI(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
               #  epi.Epi(resourcefilepath=resourcefilepath),
               #  hiv.Hiv(resourcefilepath=resourcefilepath),
               #  tb.Tb(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                # newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                # pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                # care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                # labour.Labour(resourcefilepath=resourcefilepath),
                 #postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 )

    sim.make_initial_population(n=population_size)
    return sim

