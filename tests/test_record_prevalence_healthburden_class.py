import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tlo import Date, Simulation
from tlo.analysis.utils import extract_results, parse_log_file

from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 1000
seed = 42

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
    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})

    # Register the appropriate modules
    sim.register(*fullmodel(
        resourcefilepath=resourcefilepath,
        use_simplified_births=False,))

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']
    prevalence_tb_function = prevalence['Tb']
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]["tbPrevActive"] + output['tlo.methods.tb']["tb_prevalence"]["tbPrevLatent"]

    print(prevalence_tb_function)
    assert prevalence_tb_function[0] ==  prevalence_tb_log[0]
