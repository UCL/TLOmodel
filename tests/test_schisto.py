"""This file has been created to allow a check
that the model is working as originally intended."""

import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    schisto,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)


def get_simulation(seed, start_date, mda_execute=True):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           cons_availability='all'),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 schisto.Schisto(resourcefilepath=resourcefilepath, mda_execute=mda_execute, single_district=True),
                 )
    return sim


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.mark.slow
def test_run_without_mda(seed):
    """Run the Schisto module with default parameters for one year on a population of 10_000, with no MDA"""

    end_date = start_date + pd.DateOffset(years=1)
    popsize = 10_000

    sim = get_simulation(seed=seed, start_date=start_date, mda_execute=False)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


@pytest.mark.slow
def test_run_with_mda(seed):
    """Run the Schisto module with default parameters for 20 years on a population of 1_000, with MDA"""

    end_date = start_date + pd.DateOffset(years=20)
    popsize = 5_000

    sim = get_simulation(seed=seed, start_date=start_date, mda_execute=True)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)


def test_diagnosis_and_treatment(seed):
    """ infect person with high worm burden (mansoni) and test whether
    diagnosed correctly, referred for treatment, and treatment effectively reduces
    worm burden in individual """

    infecting_worms = 500

    sim = get_simulation(seed=seed, start_date=start_date, mda_execute=True)
    # set symptoms to high probability
    sim.modules['Schisto'].parameters["sm_symptoms"] = {key: 1 for key in
                                                        sim.modules['Schisto'].parameters["sm_symptoms"]}

    sim.make_initial_population(n=1)
    sim.simulate(end_date=start_date + pd.DateOffset(days=1))

    df = sim.population.props
    person_id = 0

    # give person high S. mansoni worm burden
    df.at[person_id, "ss_sm_infection_status"] = 'Non-infected'
    df.at[person_id, "ss_sm_aggregate_worm_burden"] = 0
    df.at[person_id, "ss_sm_susceptibility"] = 1
    df.at[person_id, "ss_sm_harbouring_rate"] = 0.5

    mature_worms = schisto.SchistoMatureWorms(
            module=sim.modules['Schisto'],
            species=sim.modules['Schisto'].species['mansoni'],
            person_id=0,
            number_of_worms_that_mature=infecting_worms,
        )
    mature_worms.apply(person_id)

    # check symptoms assigned
    symptom_list = {'anemia', 'fever', 'ascites', 'diarrhoea', 'vomiting', 'hepatomegaly'}
    assert symptom_list.issubset(sim.modules['SymptomManager'].has_what(person_id))

    # refer for test
    test_appt = schisto.HSI_Schisto_TestingFollowingSymptoms(person_id=person_id,
                                                 module=sim.modules['Schisto'])
    test_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # check referral for treatment is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], schisto.HSI_Schisto_TreatmentFollowingDiagnosis)
    ][0]
    assert date_event > sim.date

    # check tx administered
    tx_appt = schisto.HSI_Schisto_TreatmentFollowingDiagnosis(person_id=person_id,
                                                 module=sim.modules['Schisto'])
    tx_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'ss_last_PZQ_date'] != pd.NaT

    # check worm burden now reduced
    assert df.at[person_id, 'ss_sm_aggregate_worm_burden'] < infecting_worms






