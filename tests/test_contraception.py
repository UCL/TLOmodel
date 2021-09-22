import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.hiv import DummyHivModule


def __check_properties(df):
    # basic checks on configuration of properties
    assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & (df.co_contraception != 'not_using')).any()
    assert not ((~df.date_of_birth.isna()) & (df.age_years < 15) & (df.co_contraception != 'not_using')).any()
    assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & df.is_pregnant).any()


def __check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def check_logs(sim):
    """Do the checks on a logfile generated by the simulation"""
    logs = parse_log_file(sim.log_filepath)['tlo.methods.contraception']
    assert set(logs.keys()) == {'contraception_use_yearly_summary', 'pregnancy', 'contraception_change'}

    # check that pregnancies are happening and that some of from those on a contraceptive and some from those not on
    # contraception
    assert len(logs['pregnancy'])
    # assert (logs['pregnancy']['contraception'] != "not_using").any()  # <-- only works on big enough runs
    assert (logs['pregnancy']['contraception'] == "not_using").any()

    # check that yearly-summary logs are as expected and that some use of contraception is happening:
    ys = logs['contraception_use_yearly_summary']
    ys = ys.set_index('date')
    assert set(ys.columns) == sim.modules['Contraception'].all_contraception_states
    assert (ys.drop(columns=['not_using']).sum(axis=1) > 0).all()

    # check that there is some starting/stopping/switching of contraceptive:
    con = logs['contraception_change']

    # some starting
    assert len(con.loc[con.switch_from == "not_using"])

    # some stopping
    assert len(con.loc[con.switch_to == "not_using"])

    # some switching
    assert len(con.loc[(con.switch_from != "not_using") & (con.switch_to != "not_using")])

    # no switching to female_sterilization if age less than 30 (or equal to, in case they have aged since an HSI was
    # scheduled)
    assert not (con.loc[con['age'] <= 30, 'switch_to'] == 'female_sterilization').any()

    # no switching from female_sterilization
    assert not (con.switch_from == 'female_sterilization').any()


def run_sim(tmpdir, use_healthsystem=False, healthsystem_disable_and_reject_all=False):
    """Run basic checks on function of contraception module"""

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 12, 31)
    popsize = 1000

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.contraception': logging.INFO
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config, seed=0)

    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=(not healthsystem_disable_and_reject_all),
                                  disable_and_reject_all=healthsystem_disable_and_reject_all,
                                  ignore_cons_constraints=True,
                                  ),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=use_healthsystem),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

        # - Dummy HIV module (as contraception requires the property hv_inf)
        DummyHivModule()
    )

    sim.make_initial_population(n=popsize)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    # Make most of the population women
    df = sim.population.props
    df.loc[df.is_alive, 'sex'] = sim.modules['Demography'].rng.choice(['M', 'F'], p=[0.5, 0.5],
                                                                      size=df.is_alive.sum())
    df.loc[(df.sex == 'M'), "co_contraception"] = "not_using"

    sim.simulate(end_date=end_date)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    return sim


def test_contraception_use_and_not_using_healthsystem(tmpdir):
    """Test that the contraception module function and that what comes out in log is as expected when initiation and
    switching is NOT going through the HealthSystem."""

    # Run basic check, for the case when the model is using the healthsystem and when not and check the logs
    sim_does_not_use_healthsystem = run_sim(tmpdir=tmpdir, use_healthsystem=False)
    check_logs(sim_does_not_use_healthsystem)

    sim_uses_healthsystem = run_sim(tmpdir=tmpdir, use_healthsystem=True)
    check_logs(sim_uses_healthsystem)

    # Check that the output of these two simulations are the same (apart from day of the month, which may change as
    # HSI dates are intentionally scattered over the month.)

    def format_log(_log):
        """Format the log so that date is replaced with the only the month and year"""
        _log["year_month"] = pd.to_datetime(_log['date']).dt.to_period('M')
        return _log.drop(columns=['date', 'age']).sort_values('year_month').reset_index(drop=True)

    for key in {'contraception_use_yearly_summary', 'pregnancy', 'contraception_change'}:
        pd.testing.assert_frame_equal(
            format_log(parse_log_file(sim_uses_healthsystem.log_filepath)['tlo.methods.contraception'][key]),
            format_log(parse_log_file(sim_does_not_use_healthsystem.log_filepath)['tlo.methods.contraception'][key])
        )

    # u = format_log(parse_log_file(sim_uses_healthsystem.log_filepath)[
    #                    'tlo.methods.contraception']['contraception_change'])
    # n = format_log(parse_log_file(sim_does_not_use_healthsystem.log_filepath)[
    #                    'tlo.methods.contraception']['contraception_change'])


def test_contraception_using_healthsystem_but_no_capability(tmpdir):
    """Check that if switching and initiation use the HealthSystem but that no HSI occur that there is no initiation or
     switching of anything that requires an HSI"""

    # Run simulation whereby contraception requires HSI but the Healthsystem prevent HSI occurring
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=True, healthsystem_disable_and_reject_all=True)

    log = parse_log_file(sim.log_filepath)['tlo.methods.contraception']

    # No record of starting/switching-to contraception of anything that requires an HSI
    assert not log["contraception_change"]["switch_to"].isin(
        sim.modules['Contraception'].states_that_may_require_HSI_to_switch_to
    ).any()
