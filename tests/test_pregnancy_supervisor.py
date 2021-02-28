import pytest
import os
from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

seed = 560

log_config = {
    "filename": "pregnancy_supervisor_test",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # warning  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.contraception": logging.DEBUG,
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def set_all_women_as_pregnant(sim):
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)


def registering_modules():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    return sim

# =========================================== PREGNANCY SUPERVISOR TESTS ==============================================


@pytest.mark.group2
def test_run_with_normal_allocation_of_pregnancy():
    sim = registering_modules()

    sim.make_initial_population(n=1000)

    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


@pytest.mark.group2
def test_run_with_modules_linked_via_interventions():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=1000)

    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


@pytest.mark.group2
def test_run_with_high_volumes_of_pregnancy():
    sim = registering_modules()
    sim.make_initial_population(n=1000)

    set_all_women_as_pregnant(sim)

    sim.simulate(end_date=Date(2011, 1, 1))


@pytest.mark.group2
def test_twins_logic():
    sim = registering_modules()
    sim.make_initial_population(n=1000)
    set_all_women_as_pregnant(sim)

    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['multiples'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.5)

    sim.simulate(end_date=Date(2011, 1, 1))
    df = sim.population.props

    twins = df.nb_is_twin
    assert (df.loc[twins.loc[twins].index, 'nb_twin_sibling_id'] != -1).all().all()
    # todo more


@pytest.mark.group2
def test_ensure_spont_abortion_stops_pregnancies():
    sim = registering_modules()

    sim.make_initial_population(n=100)
    set_all_women_as_pregnant(sim)

    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))

    df = sim.population.props
    assert len(df) == 100


# TODO: this isnt working, need to check why
# @pytest.mark.group2
# def test_ensure_induced_abortion_stops_pregnancies():
#    sim = registering_modules()

#    sim.make_initial_population(n=100)
#    set_all_women_as_pregnant(sim)

#    params = sim.modules['PregnancySupervisor'].parameters
#    params['ps_linear_equations']['induced_abortion'] = \
#        LinearModel(
#            LinearModelType.MULTIPLICATIVE,
#            1)

#    sim.simulate(end_date=Date(2011, 1, 1))

#    df = sim.population.props
#    assert len(df) == 100


@pytest.mark.group2
def test_ensure_ectopics_stops_pregnancies():
    sim = registering_modules()

    sim.make_initial_population(n=100)
    set_all_women_as_pregnant(sim)

    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))

    df = sim.population.props
    assert len(df) == 100


test_run_with_normal_allocation_of_pregnancy()
