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


def get_sim(tmpdir, popsize=100, dur=pd.DateOffset(months=3)):

    start_date = Date(2010, 1, 1)
    end_date = start_date + dur

    sim = Simulation(start_date=start_date, seed=0, show_progress_bar=True, log_config={
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
        alri.Alri(resourcefilepath=resourcefilepath, log_indivdual=True, do_checks=True)
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim

def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run(tmpdir):
    """Short run of the module using default parameters with check on dtypes"""
    dur = pd.DateOffset(months=1)
    popsize = 100
    sim = get_sim(tmpdir, popsize=popsize, dur=dur)
    check_dtypes(sim)


def test_nat_hist_progression(tempdir):
    """Cause infection --> ALRI onset --> complication --> death and check it is logged correctly"""
    pass


def test_basic_run_lasting_two_years(tmpdir):
    """Check logging results in a run of the model for two years, with daily property config checking"""
    dur = pd.DateOffset(years=2)
    popsize = 100
    sim = get_sim(tmpdir, popsize=popsize, dur=dur)

    # Read the log for the population counts of incidence:
    log_counts = parse_log_file(sim.log_filepath)['tlo.methods.alri']['event_counts']
    log_path_breakdown = parse_log_file(sim.log_filepath)['tlo.methods.alri']['incidence_count_by_age_and_pathogen']

    # Read the log for the one individual being tracked:
    log_one_person = parse_log_file(sim.log_filepath)['tlo.methods.alri']['log_individual']
    log_one_person['date'] = pd.to_datetime(log_one_person['date'])
    log_one_person = log_one_person.set_index('date')
    assert log_one_person.index.equals(pd.date_range(sim.start_date, sim.end_date - pd.DateOffset(days=1)))
    assert set(log_one_person.columns) == set(sim.modules['Alri'].PROPERTIES.keys())


# todo - Need some kind of test bed so that Ines can see the effects of the linear models she is programming.



# TODO -- @ines: We need some tests here to make sure everything is working, like in the diarrhoea code.
#  I have done some basic one above to check on the mechanics of the logging etc. But we need more for the 'biology:
#  Some examples below:

# 1) The progression of natural history for one person

# 2) Show that treatment, when provided and has 100% effectiveness, prevents deaths

# 3) Show that if treatment not provided, and CFR is 100%, every case results in a death


# check that things are being logged

# todo  - test use of cure event


