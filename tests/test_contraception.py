import os
from pathlib import Path

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

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 1000


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


def test_contraception(tmpdir):
    """Test that the contraception module function and that what comes out in log is as expected"""
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.contraception': logging.INFO
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config)
    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath),
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

    sim.simulate(end_date=end_date)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    logs = parse_log_file(sim.log_filepath)['tlo.methods.contraception']
    assert set(logs.keys()) == {'contraception_use_yearly_summary', 'contraception_costs_yearly_summary', 'pregnancy',
                                'contraception'}

    # check that pregnancies are happening:
    assert len(logs['pregnancy'])

    # check that yearly-summary logs are as expected and that some use of contraception is happening:
    ys = logs['contraception_use_yearly_summary']
    ys = ys.set_index('date')
    assert set(ys.columns) == sim.modules['Contraception'].all_contraception_states
    assert (ys.drop(columns=['not_using']).sum(axis=1) > 0).all()

    # check that there is some starting/stopping/switching of contraceptive:
    con = logs['contraception']

    # some starting
    assert len(con.loc[con.switch_from == "not_using"])

    # some starting after pregnancy
    assert len(con.loc[con.init_after_pregnancy])
    assert (con.loc[con.init_after_pregnancy].switch_from == "not_using").all()
    assert (con.loc[con.init_after_pregnancy].switch_to != "not_using").any()

    # some switching
    assert len(con.loc[(con.switch_from != "not_using") & (con.switch_to != "not_using")])

    # no switching from female_sterilization
    assert not (con.switch_from == 'female_sterilization').any()
