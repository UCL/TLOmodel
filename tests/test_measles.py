import os
from pathlib import Path
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    dx_algorithm_child,
    healthseekingbehaviour,
    symptommanager,
    antenatal_care,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    epi,
    measles,
    postnatal_supervisor)
from tlo.methods.healthsystem import HSI_Event

try:
    resources = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resources = "resources"


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


log_config = {
    "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs/",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.measles": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}


def make_sim():
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0, log_config=None)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=["*"],  # all treatment IDs allowed
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,  # multiplier for capabilities of health officer
            disable=False,
        ),
        # disables the health system constraints so all HSI events run
        symptommanager.SymptomManager(resourcefilepath=resources),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
        dx_algorithm_child.DxAlgorithmChild(),
        healthburden.HealthBurden(resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    return sim


def test_single_person():
    """
    run sim for one person
    assign infection
    check symptoms scheduled
    check symptoms resolved correctly
    """
    sim = make_sim()

    # set high death rate - change all symptom probabilities to 1
    sim.modules['Measles'].parameters["symptom_prob"]["probability"] = 1

    sim.make_initial_population(n=2)  # why does this throw an error if n=1??
    # ValueError: Wrong number of items passed 5, placement implies 1
    df = sim.population.props
    person_id = 0
    df.at[person_id, "me_has_measles"] = True

    # measles onset event
    inf_event = measles.MeaslesOnsetEvent(person_id=person_id, module=sim.modules['Measles'])
    inf_event.apply(person_id)
    assert not pd.isnull(df.at[person_id, "me_date_measles"])

    # check measles symptom resolve event and death scheduled
    events_for_this_person = sim.find_events_for_person(person_id)
    assert len(events_for_this_person) > 0
    next_event_date, next_event_obj = events_for_this_person[0]
    assert (isinstance(next_event_obj, measles.MeaslesSymptomResolveEvent) or isinstance(next_event_obj, measles.MeaslesDeathEvent))


def test_measles_cases_and_hsi_occurring():
    """ Run the measles module
    check dtypes consistency
    check infections occurring
    check measles onset event scheduled
    check symptoms assigned
    check treatments occurring
    """

    end_date = Date(2011, 12, 31)
    popsize = 1000

    sim = make_sim()

    # set high transmission probability
    sim.modules['Measles'].parameters['beta_baseline'] = 1.0

    # set high death rate - change all symptom probabilities to 1
    sim.modules['Measles'].parameters["symptom_prob"]["probability"] = 1

    # Make the population
    sim.make_initial_population(n=popsize)

    # check data types
    check_dtypes(sim)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props

    # check people getting measles
    assert df['me_has_measles'].values.sum() > 0  # current cases of measles

    # check that everyone who is currently infected gets a measles onset or symptom resolve event
    # they can have multiple symptom resolve events scheduled (by symptom onset and by treatment)
    inf = df.loc[df.is_alive & df.me_has_measles].index.tolist()

    for idx in inf:
        events_for_this_person = sim.find_events_for_person(idx)
        assert len(events_for_this_person) > 0
        # assert measles event in event list for this person
        assert "tlo.methods.measles" in str(events_for_this_person)
        # find the first measles event
        measles_event_date = [date for (date, event) in events_for_this_person if "tlo.methods.measles" in str(event)]
        assert measles_event_date[0] >= df.loc[idx, "me_date_measles"]

    # check symptoms assigned
    # there is an incubation period, so infected people may not have rash immediately
    # if on treatment for measles, must have rash for diagnosis
    has_rash = sim.modules['SymptomManager'].who_has('rash')
    current_measles_tx = df.index[df.is_alive & df.me_has_measles & df.me_on_treatment]
    if current_measles_tx.any():
        assert set(current_measles_tx) <= set(has_rash)

    # check if any measles deaths occurred
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()


def test_measles_zero_death_rate():

    end_date = Date(2010, 12, 31)
    popsize = 100

    sim = make_sim()

    # set zero death rate - change all symptom probabilities to 0
    sim.modules['Measles'].parameters["symptom_prob"]["probability"] = 0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    df = sim.population.props

    # no symptoms should equal no treatment (unless other rash has prompted incorrect tx: unlikely)
    assert not (df.loc[df.is_alive, 'me_on_treatment']).all()

    # check that there have been no deaths caused by measles
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()



