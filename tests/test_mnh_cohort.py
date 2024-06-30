import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import DAYS_IN_MONTH, DAYS_IN_YEAR, Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel


# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(mnh_cohort_module.MaternalNewbornHealthCohort(resourcefilepath=resourcefilepath),
                 *fullmodel(resourcefilepath=resourcefilepath),

                 )

def test_run_sim_with_mnh_cohort(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

    register_modules(sim)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

# def test_mnh_cohort_module_updates_properties_as_expected(tmpdir, seed):
#     sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})
#
#     register_modules(sim)
#     sim.make_initial_population(n=1000)
#     sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
#     # to do: check properties!!
