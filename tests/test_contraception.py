import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import get_medium_variant_asfr_from_wpp_resourcefile, parse_log_file
from tlo.methods import contraception, demography, enhanced_lifestyle, healthsystem, symptommanager
from tlo.methods.hiv import DummyHivModule


def run_sim(tmpdir,
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
            equalise_all_risk_of_preg=False,   # todo - this can go.
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

    _cons_available = 'all' if consumables_available else 'none'

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

    sim = Simulation(start_date=start_date, log_config=log_config, seed=0)

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
    if no_initial_contraception_use:
        states = sim.modules['Contraception'].all_contraception_states
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
        sim.modules['Contraception'].processed_params['p_start_after_birth'] *= 0.0

    if equalise_all_risk_of_preg:
        sim.modules['Contraception'].processed_params['p_pregnancy_no_contraception_per_month'].loc[:, :] = 0.02
        sim.modules['Contraception'].processed_params['p_pregnancy_with_contraception_per_month'].loc[:, :] = 0.02

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
            assert not (con.loc[con['age_years'] <= 30, 'switch_to'] == 'female_sterilization').any()  # no switching to
            # female_sterilization if age less than 30 (or equal to, in case they have aged since an HSI was scheduled)


def test_starting_and_stopping_contraceptive_use():
    """Check that initiation and discontinuation rates work as expected."""
    popsize = 10_000

    def create_dummy_sim():
        """Create dummy simulation"""
        resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
        start_date = Date(2010, 1, 1)
        _sim = Simulation(start_date=start_date, seed=0)
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


def test_pregnancies_and_births_occurring(tmpdir):
    """Test that pregnancies occur for those who are on contraception and those who are not."""
    # Run simulation without use of HealthSystem stuff and with high risk of failure of contraceptive
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=False, disable=True, incr_prob_of_failure=True)

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


def test_woman_starting_contraceptive_after_birth(tmpdir):
    """Check that woman re-start the same contraceptive after birth."""
    sim = run_sim(tmpdir=tmpdir, run=False)

    # Select a woman to be a mother
    person_id = 0
    sim.population.props.loc[person_id, [
        "is_alive",
        "sex",
        "age_years"]
    ] = (
        True,
        "F",
        30
    )

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


def test_occurrence_of_HSI_for_maintaining_on_and_switching_to_methods(tmpdir):
    """Check HSI for the maintenance of a person on a contraceptive are scheduled as expected.."""

    # Create a simulation that has run for zero days and clear the event queue
    sim = run_sim(tmpdir,
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


def test_defaulting_off_method_if_no_healthsystem_or_consumable_at_individual_level(tmpdir):
    """Check that if someone is on a method that requires an HSI for maintenance, and if consumable is not available
     and/or the health system cannot do the appointment, then that the person defaults to not using after they become
     due for a maintenance appointment."""

    def check_that_persons_on_contraceptive_default(sim):
        """Before simulaton starts, put women on a contraceptive, and make some due an appointment. Then run the
        simulation. Check that those who are on a contraceptive that requires HSI and consumables default to "not_using"
        by the end of the simulation."""

        df = sim.population.props
        contraceptives = list(sim.modules['Contraception'].all_contraception_states)

        # Set that person_id=0-10 are woman on each of the contraceptive and are due an appointment next month (these
        # women will default if on a contraceptive that requires a consumable).
        person_ids_due_appt = list(range(len(contraceptives)))

        # Set that person_id=12-25 are women each of the contraceptives and are not due an appointment during the
        # simulation. These women will not default.
        person_ids_not_due_appt = [i + len(contraceptives) for i in person_ids_due_appt]

        for i, contraceptive in enumerate(contraceptives):
            original_props = {
                'sex': 'F',
                'age_years': 30,
                'date_of_birth': sim.date - pd.DateOffset(years=30),
                'co_contraception': contraceptive,
                'is_pregnant': False,
                'date_of_last_pregnancy': pd.NaT,
                'co_unintended_preg': False,
                'co_date_of_last_fp_appt': sim.date - pd.DateOffset(months=5)
                # <-- due for an appointment in 1 mo
            }
            df.loc[person_ids_due_appt[i], original_props.keys()] = original_props.values()

            original_props['co_date_of_last_fp_appt'] = sim.date - pd.DateOffset(days=1)
            # <--not due an appointment
            df.loc[person_ids_not_due_appt[i], original_props.keys()] = original_props.values()

        # Check they are using the correct contraceptive
        for i, _c in enumerate(contraceptives):
            assert df.at[person_ids_due_appt[i], "co_contraception"] == _c
            assert df.at[person_ids_not_due_appt[i], "co_contraception"] == _c

        # Run simulation
        sim.simulate(end_date=sim.start_date + pd.DateOffset(months=3))
        __check_no_illegal_switches(sim)

        # Those on a contraceptive that requires HSI for maintenance should have defaulted to "not_using".
        # NB. All defaulters will move to "not_using" because not other kind of natural switching is allowed in this
        #  simulation.
        for i, _c in enumerate(contraceptives):

            if _c in sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on:
                assert df.at[person_ids_due_appt[i], "co_contraception"] == "not_using"
            else:
                assert df.at[person_ids_due_appt[i], "co_contraception"] == _c

            # Those not due an appointment will not have defaulted (were not due an appointment)
            assert df.at[person_ids_not_due_appt[i], "co_contraception"] == _c

    # Check when no HSI occur
    sim = run_sim(tmpdir,
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
                  use_healthsystem=True,
                  disable=False,
                  consumables_available=False,
                  no_changes_in_contraception=True,
                  run=False,
                  popsize=50
                  )
    check_that_persons_on_contraceptive_default(sim)


def test_defaulting_off_method_if_no_healthsystem_at_population_level(tmpdir):
    """Check that if switching and initiation use the HealthSystem but no HSI can occur, then all those already
     on a contraceptive requiring an HSI to maintain use will default to not_using, and there is no initiation or
     switching to any contraceptive that requires an HSI."""

    # Run simulation whereby contraception requires HSI but the HealthSystem prevent HSI occurring
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=True, healthsystem_disable_and_reject_all=True)
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


def test_defaulting_off_method_if_no_consumables_at_population_level(tmpdir):
    """Check that if switching and initiation use the HealthSystem but there are no consumables, then all those already
     on a contraceptive requiring a consumable to maintain use will default to not_using, and there is no initiation or
      switching to any contraceptive that requires a consumable."""

    # Run simulation whereby contraception requires HSI, HSI run, but there are no consumables
    # Let there be no discontinuation (so that every would otherwise stay on contraception)
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=True, disable=False, consumables_available=False,
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


def test_outcomes_same_if_using_or_not_using_healthsystem(tmpdir):
    """Test that the contraception module functions and that exactly the same patterns of usage, switching, etc occur
    when action do not use the HealthsSystem as when they do (and the HealthSystem allow every change to occur)."""

    # Run basic check, for the case when the model is using the healthsystem and when not and check the logs
    sim_does_not_use_healthsystem = run_sim(run=True, tmpdir=tmpdir, use_healthsystem=False, disable=True)
    __check_no_illegal_switches(sim_does_not_use_healthsystem)
    __check_some_starting_switching_and_stopping(sim_does_not_use_healthsystem)

    sim_uses_healthsystem = run_sim(run=True, tmpdir=tmpdir, use_healthsystem=True, disable=True)
    __check_no_illegal_switches(sim_uses_healthsystem)
    __check_some_starting_switching_and_stopping(sim_uses_healthsystem)

    # Check that the output of these two simulations are the same (apart from day of the month, which may change as
    # HSI dates are intentionally scattered over the month.)

    def format_log(_log):
        """Format the log so that date is replaced with only the month and year"""
        _log["year_month"] = pd.to_datetime(_log['date']).dt.to_period('M')
        return _log.drop(columns=['date', 'age_years']).sort_values(['year_month', 'woman_id']).reset_index(drop=True)

    for key in {'pregnancy', 'contraception_change'}:
        pd.testing.assert_frame_equal(
            format_log(parse_log_file(sim_uses_healthsystem.log_filepath)['tlo.methods.contraception'][key]),
            format_log(parse_log_file(sim_does_not_use_healthsystem.log_filepath)['tlo.methods.contraception'][key])
        )


def test_initial_distribution_of_contraception_and_implied_asfr(tmpdir):
    """Check that the initial population distribution has the expected distribution of use of contraceptive methods and
    that this induces an age-specific fertility rate that is consistent with the WPP calibration data in that year."""

    sim = run_sim(tmpdir, end_date=Date(2010, 1, 1), popsize=100_000)  # large simulation just to initialise pop

    df = sim.population.props
    contraception = sim.modules['Contraception']
    pp = contraception.processed_params
    states = contraception.all_contraception_states
    prob_live_births = sim.modules['Labour'].parameters['prob_live_birth']

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

    # 2) Check that the induced age-specific fertility rates match expected number of births per month (by mother age)

    # - pregnancies for those not on contraception
    prob_preg_no_contraception = df.loc[(df.sex == "F")
                                        & df.age_years.between(15, 49)
                                        & df.is_alive
                                        & ~df.is_pregnant
                                        & (df.co_contraception == 'not_using')
                                        ].merge(
        pp['p_pregnancy_no_contraception_per_month'], left_on='age_years', right_on='age_years')
    [['age_range', 'hv_inf', 'hv_inf_False', 'hv_inf_True']]
    prob_preg_no_contraception['prob'] = prob_preg_no_contraception.apply(
        lambda row: row.hv_inf_True if row.hv_inf else row.hv_inf_False, axis=1)
    expected_pregs_no_contraception = prob_preg_no_contraception.groupby('age_range')['prob'].sum()[
        adult_age_groups].to_dict()

    # - pregnancies for those on contraception
    prob_preg_on_contraception = df.loc[
        (df.sex == "F") & df.age_years.between(15, 49) & df.is_alive & ~df.is_pregnant & (
            df.co_contraception != 'not_using')].merge(
        pp['p_pregnancy_with_contraception_per_month'].stack().reset_index(),
        left_on=['co_contraception', 'age_years'], right_on=['level_1', 'level_0']
    )
    expected_preg_on_contraception = prob_preg_on_contraception.groupby('age_range')[0].sum()[
        adult_age_groups].to_dict()

    # Expected Age-Specific Fertility Rates (per month), given actual pattern of contraceptive usage in the model
    model_asfr = {
        k: prob_live_births * (
            (expected_pregs_no_contraception[k] + expected_preg_on_contraception[k])
            / len(df.loc[df.is_alive & (df.sex == "F") & (df.age_range == k)])
        ) for k in adult_age_groups
    }

    # Get the "expected" age-specific fertility rates (per month) implied deterministically by the parameters for
    # usage of contraception usage in 2010
    assumption_init_use = pp['initial_method_use'].groupby(
        by=pp['initial_method_use'].index.map(age_group_lookup)).mean()
    ex_preg_per_month = pd.DataFrame(index=range(15, 50), columns=sorted(states))
    ex_preg_per_month['not_using'] = \
        pp['p_pregnancy_no_contraception_per_month']['hv_inf_False']  # (for simplicity, assume all HIV-negative)
    ex_preg_per_month.loc[:, sorted(states - {'not_using'})] = \
        pp['p_pregnancy_with_contraception_per_month'].loc[:, sorted(states - {'not_using'})]
    ex_preg_per_month = ex_preg_per_month.groupby(by=ex_preg_per_month.index.map(age_group_lookup)).mean()
    ex_asfr_per_month = ((assumption_init_use * ex_preg_per_month).sum(axis=1) * prob_live_births).to_dict()

    # Check that model approximately matches the deterministic expectation
    assert all([np.isclose(model_asfr[k], ex_asfr_per_month[k], rtol=0.10) for k in adult_age_groups])
    assert 0.05 > (
        abs(sum(model_asfr.values()) - sum(ex_asfr_per_month.values())) / sum(ex_asfr_per_month.values())
    )

    # Check age-specific fertility rates against WPP
    wpp_fert_per_month_2010 = get_medium_variant_asfr_from_wpp_resourcefile(
        sim.modules['Contraception'].parameters['age_specific_fertility_rates'], months_exposure=1)[2010]
    # assert all([np.isclose(model_asfr[k], wpp_fert_per_month_2010[k], rtol=0.10) for k in adult_age_groups])
    # todo - put this back to working :-)

    # 3) Check that the actual number of births simulated over 2010-2014 matches expectations:
    sim_run_to_end_2014 = run_sim(tmpdir,
                                  end_date=Date(2014, 12, 31),
                                  popsize=20_000,
                                  disable=True,
                                  # no_changes_in_contraception=True,
                                  # no_initial_contraception_use=True,
                                  # equalise_all_risk_of_preg=True
                                  )
    log = parse_log_file(sim_run_to_end_2014.log_filepath)

    def get_births_by_year_and_age_range_of_mother_at_pregnancy(_df):
        _df = _df.drop(_df.index[_df.mother == -1])
        _df = _df.assign(year=_df['date'].dt.year)
        _df['mother_age_range'] = _df['mother_age_at_pregnancy'].map(age_group_lookup)
        return _df.groupby(['year', 'mother_age_range'])['year'].count()

    def get_num_adult_women_by_age_range(_df):
        _df = _df.assign(year=_df['date'].dt.year)
        _df = _df.set_index(_df['year'], drop=True)
        _df = _df.drop(columns='date')
        select_col = adult_age_groups
        ser = _df[select_col].stack()
        ser.index.names = ['year', 'mother_age_range']
        return ser

    def convert_annual_prob_to_monthly_prob(p_annual):
        # todo - do this properly and using helper function
        return 1 - np.exp(np.log(1 - p_annual) / 12)

    av_num_adult_women = get_num_adult_women_by_age_range(log["tlo.methods.demography"]["age_range_f"]).unstack().mean()
    num_births = get_births_by_year_and_age_range_of_mother_at_pregnancy(
        log["tlo.methods.demography"]["on_birth"]).loc[(slice(2011, 2014), slice(None))].unstack()
    av_births_per_year = num_births.mean()
    asfr_sim_per_year = av_births_per_year.div(av_num_adult_women).to_dict()
    asfr_sim_per_month = {_a: convert_annual_prob_to_monthly_prob(asfr_sim_per_year[_a]) for _a in asfr_sim_per_year}

    import matplotlib.pyplot as plt
    plt.plot(asfr_sim_per_month.keys(), asfr_sim_per_month.values(), 'b', label='In Simulation')
    plt.plot(model_asfr.keys(), model_asfr.values(), 'r', label='Expected ASFR given actual contraception')
    plt.plot(ex_asfr_per_month.keys(), ex_asfr_per_month.values(), 'r--',
             label='Expected ASFR given parameters for contraception')
    plt.plot(wpp_fert_per_month_2010.keys(), wpp_fert_per_month_2010.values(), 'k-', label='WPP')
    plt.title('ASFR per month')
    plt.ylabel('Live births per woman per month')
    plt.xlabel('Age at pregnancy')
    plt.legend()
    plt.show()

    # correction_factor = {
    #     _age:  model_asfr[_age] / asfr_sim_per_month[_age] for _age in asfr_sim_per_month
    # }

    # todo - we need a further correction because the calibration factor is from the script which doesn't recognise that
    #  people become pregnant.... putting this in heuristically for now

    # todo remove plot and make into a formal test.

    # todo (when gestation doesn't last very long, we see another bias --- to solve!)
    # todo - do it when there is a constant asfr and we can test if the number of births in all age-groups matches it.
