"""Test file for the COPD module."""

# todo Emmanuel - let's add some more tests here! For example...
#  1) If everyone starts at ch_lungfunction=0 and then is high progression rate --> everyone ends up in category 6
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

start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(months=1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run(tmpdir, seed):
    """Run the simulation with the Copd module and read the log from the Copd module."""

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


def get_simulation(pop_size):
    """ Return a simulation object

    :param pop_size: total number of individuals at the start of simulation """
    sim = Simulation(
        start_date=start_date
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
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    return sim


def test_ch_lungfunction():
    """ test everyone ends up in lung function category 6 if there is high progression rate """
    sim = get_simulation(10)
    df = sim.population.props

    # set everyone to start at lung function category 0
    df.loc[df.index, 'ch_lungfunction'] = 0

    # confirm they are all at category zero
    assert (df['ch_lungfunction'] == 0).all(), 'not all are category 0'

    copd_module = sim.modules['Copd']
    # set probability of progressing to a higher category to 1. This will ensure everyone progresses
    # to a higher category
    copd_module.parameters['prob_progress_to_next_cat'] = 1.0

    #   call a function to make individuals progress to the next category
    # data = copd.CopdModels(sim.modules['Copd'].parameters, sim.rng).will_progres_to_next_cat_of_lungfunction(df)
    sim.schedule_event(copd.Copd_PollEvent(copd_module), sim.date + pd.DateOffset(days=1))
    print(f'the data is {df["ch_lungfunction"]}')


def test_exacerbations():
    """ test copd exacerbations. """
    sim = get_simulation(10)
    df = sim.population.props
