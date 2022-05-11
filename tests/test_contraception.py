import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import contraception, demography, enhanced_lifestyle, healthsystem, symptommanager
from tlo.methods.hiv import DummyHivModule


def run_sim(tmpdir,
            seed,
            use_healthsystem=False,
            disable=False,
            healthsystem_disable_and_reject_all=False,
            consumables_available=True,
            run=True,
            no_discontinuation=False,
            incr_prob_of_failure=False,
            popsize=1000,
            end_date=Date(2011, 12, 31),
            no_changes_in_contraception=False,
            no_initial_contraception_use=False,
            equalised_risk_of_preg=None,
            unlimited_runs_of_hsi=False,
            max_days_delay_between_decision_to_change_method_and_hsi_scheduled=28
            ):
    """Run basic checks on function of contraception module"""

    def __check_properties(df):
        """basic checks on configuration of properties"""
        assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & (df.co_contraception != 'not_using')).any()
        assert not ((~df.date_of_birth.isna()) & (df.age_years < 15) & (df.co_contraception != 'not_using')).any()
        assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & df.is_pregnant).any()

    def __check_dtypes(simulation):
        """check types of columns"""
        df = simulation.population.props
        orig = simulation.population.new_row
        assert (df.dtypes == orig.dtypes).all()

    # Determine availability of consumables (True --> all available; False --> none available; other --> custom arg.)
    if consumables_available is True:
        _cons_available = 'all'
    elif consumables_available is False:
        _cons_available = 'none'
    else:
        _cons_available = consumables_available

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    start_date = Date(2010, 1, 1)

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            "tlo.methods.contraception": logging.INFO,
            "tlo.methods.demography": logging.INFO
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config, seed=seed)

    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=disable,
                                  disable_and_reject_all=healthsystem_disable_and_reject_all,
                                  cons_availability=_cons_available,
                                  ),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=use_healthsystem),
        contraception.SimplifiedPregnancyAndLabour(),

        # - Dummy HIV module (as contraception requires the property hv_inf): but set prevalence to be 0%
        DummyHivModule(hiv_prev=0.0)
    )
    states = sim.modules['Contraception'].all_contraception_states

    if no_initial_contraception_use:
        sim.modules['Contraception'].parameters['Method_Use_In_2010'].loc[:, 'not_using'] = 1.0
        sim.modules['Contraception'].parameters['Method_Use_In_2010'].loc[:, list(states - {'not_using'})] = 0.0

    sim.make_initial_population(n=popsize)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    if no_discontinuation:
        # Let there be no discontinuation of any method
        sim.modules['Contraception'].processed_params['p_stop_per_month'] = zero_param(
            sim.modules['Contraception'].processed_params['p_stop_per_month'])

    if incr_prob_of_failure:
        # Let the probability of failure of contraceptives be high
        sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'] = \
            (sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'] * 100).clip(
                upper=1.0)

    if no_changes_in_contraception:
        # Make there be no changes over time in the risk of starting/stopping a contraceptive
        sim.modules['Contraception'].processed_params['p_start_per_month'] = zero_param(
            sim.modules['Contraception'].processed_params['p_start_per_month']
        )
        sim.modules['Contraception'].processed_params['p_stop_per_month'] = zero_param(
            sim.modules['Contraception'].processed_params['p_stop_per_month']
        )
        sim.modules['Contraception'].processed_params['p_switch_from_per_month'] *= 0.0

        sim.modules['Contraception'].processed_params['p_start_after_birth']['not_using'] = 1.0
        sim.modules['Contraception'].processed_params['p_start_after_birth'][list(states - {'not_using'})] = 0.0

    if equalised_risk_of_preg is not None:
        sim.modules['Contraception'].processed_params['p_pregnancy_no_contraception_per_month'].loc[:, :] = \
            equalised_risk_of_preg
        sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'].loc[:, :] = \
            equalised_risk_of_preg

    if unlimited_runs_of_hsi:
        sim.modules['Contraception'].parameters['max_number_of_runs_of_hsi_if_consumable_not_available'] = 10_000

    sim.modules['Contraception'].parameters['max_days_delay_between_decision_to_change_method_and_hsi_scheduled'] = \
        max_days_delay_between_decision_to_change_method_and_hsi_scheduled

    if not run:
        return sim

    sim.simulate(end_date=end_date)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    return sim


def zero_param(p):
    return {k: 0.0 * v for k, v in p.items()}


def incr_param(p):
    return {k: 100.0 * v for k, v in p.items()}


def __check_some_starting_switching_and_stopping(sim):
    """Check that there is at least some usage of contraceptives and some starting, switching and stopping."""

    logs = parse_log_file(sim.log_filepath)

    # Check that summary logs are as expected and that some use of contraception is happening:
    ys = logs['tlo.methods.contraception']['contraception_use_summary']
    ys = ys.set_index('date')
    assert set(ys.columns) == sim.modules['Contraception'].all_contraception_states
    assert (ys.drop(columns=['not_using']).sum(axis=1) > 0).all()

    # Check that there is some starting, switching and stopping:
    contraception_change = logs['tlo.methods.contraception']['contraception_change']

    # some starting
    assert len(contraception_change.loc[contraception_change.switch_from == "not_using"])

    # some stopping
    assert len(contraception_change.loc[contraception_change.switch_to == "not_using"])

    # some switching
    assert len(contraception_change.loc[
                   (contraception_change.switch_from != "not_using") & (contraception_change.switch_to != "not_using")])


def __check_no_illegal_switches(sim):
    """Check that there are no illegal switches happening (from `female_sterilization` or to if if age <30).
    This is especially important when the configuration is such that people are defaulting from methods when
    HealthSystem/consumables are not available."""
    logs = parse_log_file(sim.log_filepath)

    # Check for no illegal changes
    if 'tlo.methods.contraception' in logs:
        if 'contraception_change' in logs['tlo.methods.contraception']:
            con = logs['tlo.methods.contraception']['contraception_change']
            assert not (con.switch_from == 'female_sterilization').any()  # no switching from female_sterilization
            assert not (con.loc[con['age_years'] < 30, 'switch_to'] == 'female_sterilization').any()  # no switching to
            # No female_sterilization if age less than 30


@pytest.mark.slow
def test_starting_and_stopping_contraceptive_use(seed):
    """Check that initiation and discontinuation rates work as expected."""
    popsize = 10_000

    def create_dummy_sim():
        """Create dummy simulation"""
        resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
        start_date = Date(2010, 1, 1)
        _sim = Simulation(start_date=start_date, seed=seed)
        _sim.register(
            # - core modules:
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
            contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=False),
            contraception.SimplifiedPregnancyAndLabour(),

            # - Dummy HIV module (as contraception requires the property hv_inf)
            DummyHivModule()
        )
        return _sim

    def sim_contraceptive_poll(sim):
        """Do a "manual" simulation of the contraceptive poll"""
        poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'], run_do_pregnancy=False)
        _usage_by_age = dict()
        for date in pd.date_range(sim.date, Date(2019, 12, 31), freq=pd.DateOffset(months=1)):
            sim.date = date

            _usage_by_age[date] = sim.population.props.loc[(sim.population.props.sex == 'F') & (
                sim.population.props.age_years.between(15, 49))].groupby(by=['co_contraception', 'age_range']).size()

            poll.apply(sim.population)
        return _usage_by_age

    # Set rates of initiation and discontinuation to zero: --> no changes in contraceptive use
    sim = create_dummy_sim()
    sim.make_initial_population(n=popsize)
    sim.modules['Contraception'].processed_params['p_start_per_month'] = zero_param(
        sim.modules['Contraception'].processed_params['p_start_per_month'])
    sim.modules['Contraception'].processed_params['p_stop_per_month'] = zero_param(
        sim.modules['Contraception'].processed_params['p_stop_per_month'])
    sim.modules['Contraception'].processed_params['p_switch_from_per_month'] *= 0.0
    usage = sim_contraceptive_poll(sim)
    assert all([(usage[Date(2010, 1, 1)] == usage[d]).all() for d in usage])

    # Set rates of initiation to "high" and rates of discontinuation to zero: --> all on contraception
    sim = create_dummy_sim()
    sim.make_initial_population(n=popsize)
    sim.modules['Contraception'].processed_params['p_start_per_month'] = incr_param(
        sim.modules['Contraception'].processed_params['p_start_per_month'])
    sim.modules['Contraception'].processed_params['p_stop_per_month'] = zero_param(
        sim.modules['Contraception'].processed_params['p_stop_per_month'])
    sim.modules['Contraception'].processed_params['p_switch_from_per_month'] *= 0.0
    usage = sim_contraceptive_poll(sim)
    end_usage = usage[list(usage.keys())[-1]].unstack()
    assert 0 == end_usage.loc['not_using'].sum()

    # Set rates of initiation to zero and rates of discontinuation to "high": --> all off contraception
    sim = create_dummy_sim()
    sim.make_initial_population(n=popsize)
    sim.modules['Contraception'].processed_params['p_start_per_month'] = zero_param(
        sim.modules['Contraception'].processed_params['p_start_per_month'])
    sim.modules['Contraception'].processed_params['p_stop_per_month'] = incr_param(
        sim.modules['Contraception'].processed_params['p_stop_per_month'])
    sim.modules['Contraception'].processed_params['p_switch_from_per_month'] *= 0.0
    usage = sim_contraceptive_poll(sim)
    end_usage = usage[list(usage.keys())[-1]].unstack()
    assert 0 == end_usage.drop(index=['not_using', 'female_sterilization']).sum().sum()


@pytest.mark.slow
def test_pregnancies_and_births_occurring(tmpdir, seed):
    """Test that pregnancies occur for those who are on contraception and those who are not."""
    # Run simulation without use of HealthSystem stuff and with high risk of failure of contraceptive
    sim = run_sim(tmpdir=tmpdir, seed=seed, use_healthsystem=False, disable=True, incr_prob_of_failure=True)

    # Check pregnancies
    logs = parse_log_file(sim.log_filepath)
    pregs = logs['tlo.methods.contraception']['pregnancy']

    assert len(pregs) > 0
    assert (pregs['contraception'] == "not_using").any()
    assert (pregs['contraception'] != "not_using").any()

    # Check births
    births = logs['tlo.methods.demography']['on_birth']
    assert len(births) > 0

    # Check that births are occurring during the first 9 months of the simulation (from unidentified mothers).
    after9months = pd.to_datetime(births.date) >= (sim.start_date + pd.DateOffset(months=9))
    assert len(births[~after9months])

    # Check that unidentified mothers are given as the mother for some (but not all) of the births before 9 months.
    assert -1 in births.loc[~after9months, 'mother'].values

    # Check that after 9 months, every birth has a specific mother identified (i.e. not mother_id = -1)
    assert (births.loc[after9months, 'mother'] != -1).all()

    # Check that, for any birth associated with a mother, the mother was pregnant
    assert (set(births.loc[after9months, 'mother']) - {-1}).issubset(set(pregs['woman_id']))


def test_woman_starting_contraceptive_after_birth(tmpdir, seed):
    """Check that woman re-start the same contraceptive after birth."""
    sim = run_sim(tmpdir=tmpdir, seed=seed, run=False)

    # Manipulate probabilities of starting contraception:
    contraception = sim.modules['Contraception']
    contraception.parameters['Initiation_AfterBirth'].loc[0] = \
        1.0 / len(contraception.parameters['Initiation_AfterBirth'].loc[0].values)
    contraception.processed_params = contraception.process_params()

    # Select a woman to be a mother
    person_id = 0
    _props = {
        "is_alive": True,
        "sex": "F",
        "age_years": 30
    }
    sim.population.props.loc[person_id, _props.keys()] = _props.values()

    # Run `select_contraceptive_following_birth` for the woman many times
    co_after_birth = list()
    for _ in range(1000):
        # Reset woman to be "not_using"
        sim.population.props.at[person_id, 'co_contraception'] = "not_using"

        # Run `select_contraceptive_following_birth`
        sim.modules['Contraception'].select_contraceptive_following_birth(person_id)

        # Get new status
        co_after_birth.append(sim.population.props.at[person_id, 'co_contraception'])

    # Check that updated contraceptive status is not "not_using" on at least some occasions
    assert any([x != "not_using" for x in co_after_birth])


def test_occurrence_of_HSI_for_maintaining_on_and_switching_to_methods(tmpdir, seed):
    """Check HSI for the maintenance of a person on a contraceptive are scheduled as expected.."""

    # Create a simulation that has run for zero days and clear the event queue
    sim = run_sim(tmpdir,
                  seed=seed,
                  use_healthsystem=True,
                  disable=False,
                  consumables_available=True,
                  end_date=Date(2010, 1, 1)
                  )
    sim.event_queue.queue = []
    sim.modules['HealthSystem'].reset_queue()

    # Let there be no chance of switching or discontinuing
    pp = sim.modules['Contraception'].processed_params
    pp['p_stop_per_month'] = zero_param(pp['p_stop_per_month'])
    pp['p_switch_from_per_month'] = zero_param(pp['p_switch_from_per_month'])

    # Set that person_id=0 is a woman on a contraceptive for longer than six months
    person_id = 0
    df = sim.population.props
    original_props = {
        'sex': 'F',
        'age_years': 30,
        'date_of_birth': sim.date - pd.DateOffset(years=30),
        'co_contraception': 'pill',  # <-- requires appointments for maintenance
        'is_pregnant': False,
        'date_of_last_pregnancy': pd.NaT,
        'co_unintended_preg': False,
        'co_date_of_last_fp_appt': sim.date - pd.DateOffset(months=7)
    }
    df.loc[person_id, original_props.keys()] = original_props.values()

    # Run the ContraceptivePoll
    poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'])
    poll.apply(sim.population)

    # Confirm that an HSI_FamilyPlanningAppt has been made for her (within 28 days as she is due an appointment already)
    events = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert 1 == len(events)
    ev = events[0]
    assert isinstance(ev[1], contraception.HSI_Contraception_FamilyPlanningAppt)

    date_of_hsi = ev[0]
    assert date_of_hsi <= (sim.date + pd.DateOffset(days=28))

    # Run that HSI_FamilyPlanningAppt and confirm there is no change in her state except that the date of last
    # appointment has been updated.
    sim.date = date_of_hsi
    ev[1].apply(person_id=person_id, squeeze_factor=0.0)

    df = sim.population.props  # update shortcut df
    props_to_be_same = [k for k in original_props.keys() if k != "co_date_of_last_fp_appt"]
    assert list(df.loc[person_id, props_to_be_same].values) == [original_props[p] for p in props_to_be_same]
    assert sim.population.props.at[person_id, "co_date_of_last_fp_appt"] == sim.date

    # CLear the HealthSystem queue and run the ContraceptivePoll again
    sim.modules['HealthSystem'].reset_queue()
    poll.apply(sim.population)

    # Confirm that no HSI_FamilyPlanningAppt has been scheduled (now that there is less time elapsed since her last
    # appointment)
    assert not len(sim.modules['HealthSystem'].find_events_for_person(person_id))


@pytest.mark.slow
def test_defaulting_off_method_if_no_healthsystem_or_consumable_at_individual_level(tmpdir, seed):
    """Check that if someone is on a method that requires an HSI and consumable for maintenance, but that HSI do not
    occur or consumable are not available, then that the person defaults to "not_using" as they become due for a
    maintenance appointment."""

    def check_that_persons_on_contraceptive_default(sim):
        """Before simulation starts, put women on a contraceptive. Then run the simulation. Check that those who are on
         a contraceptive that requires HSI and consumables and were due to have an appointment, default to "not_using"
         by the end of the simulation. NB. All defaulters will move to "not_using" because no other kind of natural
         switching is allowed in this simulation."""

        df = sim.population.props
        contraceptives = sorted(sim.modules['Contraception'].all_contraception_states)
        n_contraceptives = len(contraceptives)
        states_that_may_require_HSI_to_maintain_on = \
            sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on

        # Set that some woman two women are on each of the contraceptive, one of whom is due an appointment next month
        initial_conditions = pd.DataFrame({
            'method': contraceptives * 2,
            'due_appt': [True] * n_contraceptives + [False] * n_contraceptives
        })
        for _person_id, _row in initial_conditions.iterrows():
            _props = {
                'sex': 'F',
                'age_years': 30,
                'date_of_birth': sim.date - pd.DateOffset(years=30),
                'co_contraception': _row.method,
                'is_pregnant': False,
                'date_of_last_pregnancy': pd.NaT,
                'co_unintended_preg': False,
                'co_date_of_last_fp_appt': sim.date - (
                    pd.DateOffset(months=5) if _row.due_appt else pd.DateOffset(days=1)
                )
            }
            df.loc[_person_id, _props.keys()] = _props.values()

        # Run simulation
        sim.simulate(end_date=sim.start_date + pd.DateOffset(months=3))
        __check_no_illegal_switches(sim)

        # Check method that the women are now on.
        method_after_sim = sim.population.props.loc[initial_conditions.index, "co_contraception"]

        # - Those originally on a method that did not require an appointment, are still on it
        on_a_method_that_did_not_require_appointment = ~initial_conditions.method.isin(
            states_that_may_require_HSI_to_maintain_on)
        assert (
            method_after_sim.loc[on_a_method_that_did_not_require_appointment] ==
            initial_conditions.method.loc[on_a_method_that_did_not_require_appointment]
        ).all()

        # - Those originally on a method that did not require an appointment and were not due an appointment, are still
        # on it
        on_a_method_that_required_appointment_but_appointment_not_due = (
            initial_conditions.method.isin(states_that_may_require_HSI_to_maintain_on)
            & ~initial_conditions.due_appt
        )
        assert (
            method_after_sim.loc[on_a_method_that_required_appointment_but_appointment_not_due] ==
            initial_conditions.method.loc[on_a_method_that_required_appointment_but_appointment_not_due]
        ).all()

        # - Those originally on a method that did not require an appointment and were due an appointment, have defaulted
        # to "not_using"
        on_a_method_that_required_appointment_and_appointment_was_due = (
            initial_conditions.method.isin(states_that_may_require_HSI_to_maintain_on)
            & initial_conditions.due_appt
        )
        assert (
            method_after_sim.loc[on_a_method_that_required_appointment_and_appointment_was_due] == "not_using"
        ).all()

    # Check when no HSI occur
    sim = run_sim(tmpdir,
                  seed=seed,
                  use_healthsystem=True,
                  healthsystem_disable_and_reject_all=True,
                  consumables_available=True,
                  no_changes_in_contraception=True,
                  run=False,
                  popsize=50
                  )
    check_that_persons_on_contraceptive_default(sim)

    # Check when HSI occur but consumables are not available
    sim = run_sim(tmpdir,
                  seed=seed,
                  use_healthsystem=True,
                  disable=False,
                  consumables_available=False,
                  no_changes_in_contraception=True,
                  run=False,
                  popsize=50
                  )
    check_that_persons_on_contraceptive_default(sim)


@pytest.mark.slow
def test_defaulting_off_method_if_no_healthsystem_at_population_level(tmpdir, seed):
    """Check that if switching and initiation use the HealthSystem but no HSI can occur, then all those already
     on a contraceptive requiring an HSI to maintain use will default to not_using, and there is no initiation or
     switching to any contraceptive that requires an HSI."""

    # Run simulation whereby contraception requires HSI but the HealthSystem prevent HSI occurring
    sim = run_sim(tmpdir=tmpdir, seed=seed, use_healthsystem=True, healthsystem_disable_and_reject_all=True)
    __check_no_illegal_switches(sim)

    log = parse_log_file(sim.log_filepath)['tlo.methods.contraception']

    # Check there is no record of persons being maintained on contraceptives that require an HSI
    states_that_may_require_HSI_to_maintain_on = sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on
    ys = log['contraception_use_summary']
    after_everyone_has_appt = pd.to_datetime(ys['date']) > (sim.start_date + pd.DateOffset(months=7))  # 7 months allow
    # time for an appointment to become due for everyone (allowing for the monthly occurrence of the poll.)
    assert (ys.loc[after_everyone_has_appt, states_that_may_require_HSI_to_maintain_on] == 0).all().all()

    # Check there is no record of starting/switching-to contraception of anything that requires an HSI
    states_that_may_require_HSI_to_switch_to = sim.modules['Contraception'].states_that_may_require_HSI_to_switch_to
    changes = log["contraception_change"]
    assert not changes["switch_to"].isin(states_that_may_require_HSI_to_switch_to).any()

    # Check that all switches from things that require an HSI are to not something that does not require HSI
    states_that_do_require_HSI_to_switch_to = \
        sim.modules['Contraception'].all_contraception_states - states_that_may_require_HSI_to_switch_to
    assert changes.loc[changes["switch_from"].isin(states_that_may_require_HSI_to_maintain_on), "switch_to"].isin(
        states_that_do_require_HSI_to_switch_to).all()


@pytest.mark.slow
def test_defaulting_off_method_if_no_consumables_at_population_level(tmpdir, seed):
    """Check that if switching and initiation use the HealthSystem but there are no consumables, then all those already
     on a contraceptive requiring a consumable to maintain use will default to not_using, and there is no initiation or
      switching to any contraceptive that requires a consumable."""

    # Run simulation whereby contraception requires HSI, HSI run, but there are no consumables
    # Let there be no discontinuation (so that every would otherwise stay on contraception)
    sim = run_sim(tmpdir=tmpdir, seed=seed, use_healthsystem=True, disable=False, consumables_available=False,
                  no_discontinuation=True)
    __check_no_illegal_switches(sim)

    log = parse_log_file(sim.log_filepath)['tlo.methods.contraception']

    states_that_may_require_HSI_to_switch_to = sim.modules['Contraception'].states_that_may_require_HSI_to_switch_to
    states_that_may_require_HSI_to_maintain_on = sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on
    states_that_do_require_HSI_to_switch_to = \
        sim.modules['Contraception'].all_contraception_states - states_that_may_require_HSI_to_switch_to

    # Check that, after six months of simulation time, no one is on a contraceptive that requires a consumable for
    # maintenance.
    num_on_contraceptives = log['contraception_use_summary']
    after_everyone_has_appt = pd.to_datetime(num_on_contraceptives['date']) > (sim.start_date + pd.DateOffset(months=7))
    # (7 months allow time for an appointment to become due for everyone (allowing for the monthly occurrence of the
    # poll.)
    assert (num_on_contraceptives.loc[after_everyone_has_appt, states_that_may_require_HSI_to_maintain_on] == 0
            ).all().all()

    # Check that people are not switching to those contraceptives that require consumables to switch to.
    changes = log["contraception_change"]
    assert not changes["switch_to"].isin(states_that_may_require_HSI_to_switch_to).any()

    # ... but are only switching_from them to something that does not require an HSI to switch to (mostly "not_using",
    # but others if the switch was "natural")
    assert changes["switch_from"].isin(states_that_may_require_HSI_to_maintain_on).any()
    assert changes.loc[changes["switch_from"].isin(states_that_may_require_HSI_to_maintain_on), "switch_to"].isin(
        states_that_do_require_HSI_to_switch_to).all()


@pytest.mark.slow
def test_outcomes_same_if_using_or_not_using_healthsystem(tmpdir, seed):
    """Test that the contraception module has the same effects when either using `use_healthsystem=False` or
     `use_healthsystem=True` and all consumables available."""

    # Run basic check, for the case when the model is using the healthsystem and when not and check the logs
    sim_does_not_use_healthsystem = run_sim(run=True, tmpdir=tmpdir, seed=seed, use_healthsystem=False, disable=True,
                                            max_days_delay_between_decision_to_change_method_and_hsi_scheduled=0)
    __check_no_illegal_switches(sim_does_not_use_healthsystem)
    __check_some_starting_switching_and_stopping(sim_does_not_use_healthsystem)

    sim_uses_healthsystem = run_sim(run=True, tmpdir=tmpdir, seed=seed, use_healthsystem=True, disable=True,
                                    max_days_delay_between_decision_to_change_method_and_hsi_scheduled=0)
    __check_no_illegal_switches(sim_uses_healthsystem)
    __check_some_starting_switching_and_stopping(sim_uses_healthsystem)

    # Check that the output of these two simulations are the same
    def sort_log(_log):
        """Do some sorting on the logs to enable comparisons."""
        return _log.sort_values(['date', 'woman_id']).reset_index(drop=True)

    for key in {'pregnancy', 'contraception_change'}:
        pd.testing.assert_frame_equal(
            sort_log(parse_log_file(sim_uses_healthsystem.log_filepath)['tlo.methods.contraception'][key]),
            sort_log(parse_log_file(sim_does_not_use_healthsystem.log_filepath)['tlo.methods.contraception'][key])
        )


def test_correct_number_of_live_births_created(tmpdir, seed):
    """Check that the actual number of births simulated (in one month) matches expectations"""

    # Run a simulation in which every woman has the same chance of becoming pregnant.
    _risk_of_pregnancy = 0.05
    sim = run_sim(tmpdir,
                  seed=seed,
                  end_date=Date(2010, 11, 1),
                  popsize=100_000,
                  disable=True,
                  equalised_risk_of_preg=_risk_of_pregnancy
                  )
    log = parse_log_file(sim.log_filepath)

    age_group_lookup = sim.modules['Demography'].AGE_RANGE_LOOKUP
    adult_age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

    def get_births_in_a_month_by_age_range_of_mother_at_pregnancy(_df, year, month):
        _df = _df.drop(_df.index[_df.mother == -1])  # ignore pregnancies generated in first 9mo by other method
        _df = _df.assign(year=_df['date'].dt.year)
        _df = _df.assign(month=_df['date'].dt.month)
        _df['mother_age_range'] = _df['mother_age_at_pregnancy'].map(age_group_lookup)
        return _df.loc[(_df.year == year) & (_df.month == month)].groupby(by='mother_age_range').size()

    def get_num_adult_women_in_a_year_by_age_range(_df, year):
        _df = _df.assign(year=_df['date'].dt.year)
        _df = _df.set_index(_df['year'], drop=True)
        return _df.loc[year, adult_age_groups]

    # Actual number of births in simulation (in October 2010, the first month when pregnancies could occur under the
    #  control of the Contraception Module).
    actual_num_births_in_Oct2010 = get_births_in_a_month_by_age_range_of_mother_at_pregnancy(
        log["tlo.methods.demography"]["on_birth"], year=2010, month=10).sum()

    # Expected number of births:
    av_num_adult_women_in_2010 = get_num_adult_women_in_a_year_by_age_range(
        log["tlo.methods.demography"]["age_range_f"], year=2010)

    _prob_live_birth = sim.modules['Labour'].parameters['prob_live_birth']

    expected_births = av_num_adult_women_in_2010.sum() * _risk_of_pregnancy * _prob_live_birth

    assert np.isclose(
        actual_num_births_in_Oct2010,
        expected_births,
        atol=4.0 * np.sqrt(expected_births)  # Rough guide of allowable tolerance
    )


def test_initial_distribution_of_contraception(tmpdir, seed):
    """Check that the initial population distribution has the expected distribution of use of contraceptive methods."""

    # large simulation, run just to initialise pop
    sim = run_sim(tmpdir, seed=seed, end_date=Date(2010, 1, 1), popsize=100_000)

    df = sim.population.props
    pp = sim.modules['Contraception'].processed_params

    adult_age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
    age_group_lookup = sim.modules['Demography'].AGE_RANGE_LOOKUP

    # 1) Check that initial distribution of use of contraceptives matches the parameters for initial distribution
    expected = pp['initial_method_use']
    expected_by_age_range = expected.groupby(expected.index.map(age_group_lookup)).mean()

    actual = df.loc[df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
                    ].groupby(by=['co_contraception', 'age_range']).size().sort_index().unstack().T.apply(
        lambda row: row / row.sum(), axis=1
    ).loc[adult_age_groups]
    assert (abs(actual - expected_by_age_range) < 0.03).all().all()


def test_contraception_coverage_with_use_healthsystem(tmpdir, seed):
    """Check that the same patterns (approximately) of usage of contraception is achieved when `use_healthsystem=True`
    as when `use_healthsystem=False` (despite the possibility of consumables not being always available when using the
    healthsystem)."""

    def report_availability_of_consumables():
        """Helper function to find the availability of consumables used in the Contraception module."""
        sim = run_sim(tmpdir, seed, run=False, consumables_available='default')
        item_codes = sim.modules['Contraception'].get_item_code_for_each_contraceptive()
        cons = sim.modules['HealthSystem'].consumables._prob_item_codes_available

        def find_average_availability(items: List, level: str):
            """Find the probability that all the items are available at level 1a."""
            facilities = set([x.id for x in sim.modules['HealthSystem']._facilities_for_each_district[level].values()])
            return np.prod(
               [cons.loc[(slice(None), facilities, _item)].mean() for _item in items]
            )

        for fac_level in ('1a', '1b', '2'):
            av_availability = {
                k: find_average_availability(items=set(v.keys()) if isinstance(v, dict) else set(v), level=fac_level)
                for k, v in item_codes.items()
            }
            print(f'Probability of all items being available at {fac_level}: {av_availability}')

    report_availability_of_consumables()

    def summarize_contraception_use(sim):
        """Summarize the pattern of contraception currently in the population."""
        df = sim.population.props
        return df.loc[df.is_alive, 'co_contraception'].value_counts().sort_index().to_dict()

    contraception_use_healthsystem_true = summarize_contraception_use(
        run_sim(tmpdir,
                seed,
                use_healthsystem=True,
                consumables_available='default',
                popsize=5_000,
                end_date=Date(2011, 12, 31),
                equalised_risk_of_preg=0.0,
                unlimited_runs_of_hsi=True
                )
    )

    contraception_use_healthsystem_false = summarize_contraception_use(
        run_sim(tmpdir,
                seed,
                use_healthsystem=False,
                consumables_available='default',
                popsize=5_000,
                end_date=Date(2011, 12, 31),
                equalised_risk_of_preg=0.0,
                unlimited_runs_of_hsi=True
                )
    )

    def compare_dictionaries(A: Dict, B: Dict, tol: float):
        """True if the elements of A and B are equal within some tolerance (expressed as the fraction of the sum of the
         values in the dict). """

        def equals(a: int, b: int, tol: int):
            """True if the difference between a and b is less than tol (expressed as the absolute difference)."""
            return abs(a - b) < tol

        _tol = int(tol * np.mean([sum(_x.values()) for _x in (A, B)]))

        return all([equals(A[k], B[k], tol=_tol) for k in set(A.keys() & B.keys())])

    assert compare_dictionaries(contraception_use_healthsystem_true, contraception_use_healthsystem_false, tol=0.01)
