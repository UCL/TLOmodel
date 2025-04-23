import os

import pandas as pd

from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.methods import service_integration
from tlo.methods.fullmodel import fullmodel
from tlo.analysis.utils import parse_log_file

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(*fullmodel(resourcefilepath=resourcefilepath),
                  service_integration.ServiceIntegration(resourcefilepath=resourcefilepath))

def test_run_sim_service_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels":{
                "*": logging.DEBUG},"directory": tmpdir})

    register_modules(sim)
    sim.make_initial_population(n=500)
    sim.simulate(end_date=Date(2011, 1, 2))

    output= parse_log_file(sim.log_filepath)

    # The resource files
