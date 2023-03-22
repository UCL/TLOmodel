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

import numpy as np
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
    """ test everyone ends up in lung function category 6 if high progression rate is set """
    sim = get_simulation(10)
    df = sim.population.props

    copd_module = sim.modules['Copd']

    # make all individuals qualify for progressing to the next lung function
    df.loc[df.index, 'is_alive'] = True
    df.loc[df.index, 'ch_lungfunction'] = 0
    df.loc[df.index, 'age_years'] = np.random.choice(range(20, 50), len(df))

    # check they're all eligible to progress to the next lung function
    assert all(copd.eligible_to_progress_to_next_lung_function(df)), 'some are still not eligible to progress to ' \
                                                                     'next lung function'

    # set probability of progressing to next lung function to 1. This will ensure everyone progresses
    # to the next lung function
    copd_module.parameters['prob_progress_to_next_cat'] = 1.0

    # Run a function to progress to next lung function six times and ensure all individuals have progressed to a higher
    # lung function(6)
    for _range in range(6):
        copd.CopdPollEvent(module=copd_module).progress_to_next_lung_function(df)
    # all individuals should progress to the highest lung function which in this case is 6
    assert all(df['ch_lungfunction'] == 6)


def test_exacerbations():
    """ test copd exacerbations. Zero risk of exacerbation should lead to no exacerbation event scheduled and higher
    risk of exacerbation should lead to many exacerbation events scheduled"""
    sim = get_simulation(1)  # get simulation object
    copd_module = sim.modules['Copd']  # get copd module

    # 1)--------------- NO RISK OF EXACERBATION
    # reset individual properties to zero risk exacerbations.
    # reset age to <15 and lung function to 0
    df = sim.population.props
    df.loc[df.index, 'age_years'] = 10
    df.loc[df.index, 'ch_lungfunction'] = 0

    # clear the event queue
    sim.event_queue.queue = []

    # schedule copd poll event
    _event = copd.CopdPollEvent(copd_module)
    _event.apply(sim.population)

    # confirm no event on an individual has been scheduled
    _individual_events = sim.find_events_for_person(df.index[0])
    assert 0 == len(_individual_events), f'one or more events was scheduled for this ' \
                                         f'person {_individual_events}'

    # 2)----------  HIGH RISK EXACERBATION
    # reset individual properties to higher risk exacerbations.
    # reset age to >15 and lung function to 6
    df = sim.population.props
    df.loc[df.index, 'age_years'] = 20
    df.loc[df.index, 'ch_lungfunction'] = 6

    # set severe and moderate exacerbation probability to maximum(1). This ensures all exacerbation events are schedules
    # on all eligible individuals
    copd_module.parameters['prob_mod_exacerb_lung_func_6'] = 1.0
    copd_module.parameters['prob_sev_exacerb_lung_func_6'] = 1.0

    # clear the event queue
    sim.event_queue.queue = []

    # schedule copd poll event
    _event = copd.CopdPollEvent(copd_module)
    _event.apply(sim.population)

    # confirm more than one event has been scheduled
    _individual_events = sim.find_events_for_person(df.index[0])
    assert 1 < len(_individual_events), f'not all events have been scheduled {_individual_events}'


def test_moderate_exacerbation():
    """ test moderate exacerbation leads to;
          i) moderate symptoms
         ii) non-emergency care seeking
        iii) getting inhaler """
