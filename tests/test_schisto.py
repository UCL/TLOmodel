import os
import time
from pathlib import Path

import numpy as np
import pytest

from tlo import Date, Simulation
from tlo.methods import demography, healthburden, healthsystem, schisto, simplified_births, symptommanager

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 10_000

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# todo - healthcare seeking?
# make one class to declare when want both to be used - done
# symtom manager and hSI
# switches to log or not
# use consumables
# test with HSI and symptoms on
# check dtypes (and that columns are addressed with right ss_sm/ ss_sh prefix)


def simulation_both(seed):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 schisto.Schisto(resourcefilepath=resourcefilepath),
                 )
    return sim

def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run(seed):
    sim = simulation_both(seed)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

#
# def test_one_schisto_type(simulation_haem):
#     """Check that there is no columns starting with 'sm' when only haematobium is registered."""
#     simulation_haem.make_initial_population(n=popsize)
#     simulation_haem.simulate(end_date=end_date)
#     df = simulation_haem.population.props
#     for col in df.columns:
#         assert col[:2] != 'sm'
#
#
# def test_no_symptoms_or_HSI_or_MDA(simulation_haem):
#     """Check that with the symptoms turned off there will be no PZQ ever administered
#     # and that the symptoms will all be nans."""
#     df = simulation_haem.population.props
#     assert(len(df.ss_last_PZQ_date.unique()) == 1)
#     assert(df.ss_last_PZQ_date.unique()[0] == np.datetime64('1900-01-01T00:00:00.000000000'))
#     assert df.sh_symptoms.isnull().all()


# if __name__ == '__main__':
#     t0 = time.time()
#     simulation = simulation_both()
#     test_run(simulation)
#     t1 = time.time()
#     print('Time taken', t1 - t0)
