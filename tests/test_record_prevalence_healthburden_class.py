import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from tlo import DAYS_IN_YEAR, Date, Module, Simulation, logging
from tlo.analysis.utils import get_mappers_in_fullmodel, parse_log_file
from tlo.events import Event, IndividualScopeEventMixin
from tlo.methods import (
    care_of_women_during_pregnancy,
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    healthburden,
    hiv,
    hsi_generic_first_appts,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath, age_at_date
from tlo.methods.diarrhoea import increase_risk_of_death, make_treatment_perfect
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthburden import Get_Current_DALYS

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10000


def extract_mapper(key):
    return pd.Series(key.drop(columns={'date'}).loc[0]).to_dict()


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected.
    """

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': tmpdir})

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 # Disable check to avoid error due to lack of Contraception module
                 check_all_dependencies=False,
                 )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    # correctly configured index (outputs on 31st december in each year of simulation for each age/sex group)
    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']
    print(prevalence)
