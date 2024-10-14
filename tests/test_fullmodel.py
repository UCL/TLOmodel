"""A set of tests on the models with all the disease modules registered."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.methods.fullmodel import fullmodel
from tlo.util import hash_dataframe

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)

logger = logging.getLogger(f'tlo.{__name__}')  # = tlo.tests.test_fullmodel


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@contextmanager
def temporarily_change_working_directory(new_working_directory: Path):
    """Context manager for temporarily changing working directory."""
    previous_working_directory = os.getcwd()
    os.chdir(new_working_directory)
    try:
        yield
    finally:
        os.chdir(previous_working_directory)


@pytest.mark.slow
def test_dtypes_and_checksum(seed):
    """Check that types of all properties are as expected and that the checksum can be generated."""

    for _use_simplified_births in (True, False):
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(*fullmodel(resourcefilepath=resourcefilepath, use_simplified_births=_use_simplified_births))
        sim.make_initial_population(n=10_000)
        sim.simulate(end_date=start_date + pd.DateOffset(months=12))
        check_dtypes(sim)
        logger.info(key="msg", data=f"Population checksum: {hash_dataframe(sim.population.props)}")


def test_resourcefilepath(seed):
    # Try register all fullmodel modules with current working directory set to a
    # temporary directory but resourcefilepath set correctly to ensure that it is being
    # used to construct paths to resource files as expected
    with TemporaryDirectory() as temporary_directory:
        with temporarily_change_working_directory(temporary_directory):
            sim = Simulation(start_date=start_date, seed=seed)
            sim.register(*fullmodel(resourcefilepath=resourcefilepath))
