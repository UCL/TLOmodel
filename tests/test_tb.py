""" Tests for for the TB Module """


import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    simplified_births,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    epi,
    hiv,
    tb
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

# set up the log config
outputpath = Path("./outputs")  # folder for convenience of storing outputs
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.tb': logging.INFO,
        'tlo.methods.demography': logging.INFO
    }
}

start_date = Date(2010, 1, 1)


def register_sim():

    sim = Simulation(start_date=start_date, seed=100, log_config=log_config)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )
    return sim


# simple checks
def test_basic_run():
    """ test basic run and properties assigned correctly """
    end_date = Date(2012, 12, 31)
    popsize = 1000

    sim = register_sim()

    # set high transmission rate and all are fast progressors
    sim.modules['Tb'].parameters['transmission_rate'] = 0.5
    sim.modules['Tb'].parameters['prop_fast_progressor'] = 1.0

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    df = sim.population.props

    # check properties assigned correctly for baseline population
    # should be some latent infections, maybe few active infections
    num_latent = len(df[(df.tb_inf == 'latent') & df.is_alive])
    prev_latent = num_latent / len(df[df.is_alive])
    assert prev_latent > 0

    assert not pd.isnull(df.loc[~df.date_of_birth.isna(), [
        'tb_inf',
        'tb_strain',
        'tb_date_latent']
    ]).all().all()

    # no-one should be on tb treatment yet
    assert pd.isnull(df.loc[~df.date_of_birth.isna(), [
        'tb_on_treatment',
        'tb_date_treated',
        'tb_ever_treated',
        'tb_treatment_failure',
        'tb_treated_mdr',
        'tb_date_treated_mdr',
        'tb_on_ipt',
        'tb_date_ipt']
    ]).all().all()

    # run the simulation
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props  # updated dataframe

    # some should have treatment dates
    # some mdr cases should be occurring
    assert not pd.isnull(df.loc[~df.date_of_birth.isna(), [
        'tb_on_treatment',
        'tb_date_treated',
        'tb_ever_treated',
        'tb_diagnosed',
        'tb_diagnosed_mdr']
    ]).all().all()





# test overall proportion of new latent cases which progress to active
# ahould be 14% fast progressors, 67% hiv+ fast progressors
# overall lifetime risk 5-10%

# check treatment failure
# start high active infection rate
# assign treatment to all
# check proportion treatment failure

# infect one person
# check all properties
# check smear status

# check risk of relapse

