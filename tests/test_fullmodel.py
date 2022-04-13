"""A set of tests on the models with all the disease modules registered."""

import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.profiling import shared
from tlo import Date, Simulation
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_dtypes_and_checksum(seed):
    """Check that types of all properties are as expected and that the checksum can be generated."""

    for _use_simplified_births in (True, False):
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(*fullmodel(resourcefilepath=resourcefilepath, use_simplified_births=_use_simplified_births))
        sim.make_initial_population(n=1000)
        sim.simulate(end_date=start_date + pd.DateOffset(months=1))
        check_dtypes(sim)
        shared.print_checksum(sim)
