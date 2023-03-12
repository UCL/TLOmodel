"""Test file for the COPD module."""

# todo Emmanuel - let's add some more tests here! For example...
#  1) If everyone starts at co_lungfunction=0 and then is high progression rate --> everyone ends up in category 6
#  2) Zero risk of exacerbations --> No exacerbations scheduled; High risk --> Many exacerbations scheduled
#  3) Exacerbation (moderate) --> leads to moderate symptoms --> leads to non-emergency care seeking --> gets inhaler
#  4) Exacerbation (severe) --> leads to severe symptoms --> leads to emergency care seeking --> gets treatment
#  5) Zero death rate --> No death; High death rate --> Many deaths; High death rate but perfect treatment -> No deaths
#  6) ...

import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import (
    copd,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run(tmpdir, seed):
    """Run the simulation with the Copd module and read the log from the Copd module."""

    start_date = Date(2010, 1, 1)
    end_date = start_date + pd.DateOffset(months=3)
    popsize = 1000
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'bed_days',
            'directory': tmpdir,
        },
    )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 copd.Copd(resourcefilepath=resourcefilepath),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    log = parse_log_file(sim.log_filepath)['tlo.methods.copd']

    # Unpack logged prevalence
    log_prev_copd = log['copd_prevalence']

    def unflatten(date):
        select_record = lambda df, _date: df.loc[df['date'] == _date].drop(columns=['date'])  # noqa: E731
        return unflatten_flattened_multi_index_in_logging(select_record(log_prev_copd, date)).iloc[0].T.unstack()

    print(unflatten(log_prev_copd['date'].values[0]))
    print(unflatten(log_prev_copd['date'].values[-1]))
