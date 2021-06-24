"""Test file for the Alri module (alri.py)"""


import os
from pathlib import Path

import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)


# Path to the resource files used by the disease and intervention methods
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


# %% Run the Simulation

def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""

    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 4, 1)
    pop_size = 100

    sim = Simulation(start_date=start_date, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "Alri": logging.INFO}
    })

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=True)
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # Read the log for the one individual being tracked:
    log = parse_log_file(sim.log_filepath)['tlo.methods.alri']['log_individual']
    log['date'] = pd.to_datetime(log['date'])
    log = log.set_index('date')
    assert log.index.equals(pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1)))
    assert set(log.columns) == set(sim.modules['Alri'].PROPERTIES.keys())




# TODO -- @ines: We need some tests here to make sure everything is working, like in the diarrhoea code.
#  I have done some basic one above to check on the mechanics of the logging etc. But we need more for the 'biology:
#  Some examples below:

# 1) The progression of natural history for one person

# 2) Show that treatment, when provided and has 100% effectiveness, prevents deaths

# 3) Show that if treatment not provided, and CFR is 100%, every case results in a death




