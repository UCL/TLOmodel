import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

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
    measles
)

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 500

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


def test_measles_cases_and_hsi_occurring(tmpdir):
    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=["*"],  # no treatment IDs allowed
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
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check we can read the results
    log_df = parse_log_file(sim.log_filepath)
    df = sim.population.props

    # check people getting measles
    # assert df["me_has_measles"].sum > 0  # current cases of measles
    total_inc = log_df["tlo.methods.measles"]["incidence"]["inc_1000py"]
    assert total_inc.sum > 0

    # check people die of measles
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()

    # check symptoms assigned - all those with measles should have rash
    has_symptoms = set(sim.modules['SymptomManager'].who_has('rash'))
    current_measles = df.index[df.is_alive & df.me_has_measles]
    assert set(current_measles) < set(has_symptoms)

    # check measles HSI occurring
    assert len(log_df['tlo.methods.healthsystem']['HSI_Event']['Measles_Treatment']) > 0


def test_measles_high_death_rate(tmpdir):

    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=["*"],  # no treatment IDs allowed
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
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    # Increase death:
    symptom_prob = sim.modules['Measles'].parameters["symptom_prob"]
    symptom_prob.loc[symptom_prob.symptom == "death", "probability"].values[0] = 1.0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    df = sim.population.props

    # check that there have been deaths caused by Diarrhoea
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()

    # check all cases of measles also had a measles death
    assert not df.loc[~df.is_alive & df.me_has_measles & (df.cause_of_death != "measles")].any()


def test_measles_zero_death_rate(tmpdir):
    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=["*"],  # no treatment IDs allowed
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
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    # Change death rate to zero
    symptom_prob = sim.modules['Measles'].parameters["symptom_prob"]
    symptom_prob.loc[symptom_prob.symptom == "death", "probability"].values[0] = 0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check that there have been no deaths caused by Diarrhoea
    df = sim.population.props
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()


# checking no vaccines administered through health system
# only hpv should stay at zero, other vaccines start as individual events (year=2010-2018)
# coverage should gradually decline for all after 2018
# hard constraints (mode=2) and zero capabilities
def test_no_vaccine_measles_rebound(tmpdir):
    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=[" "],  # no treatment IDs allowed
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=0.0,  # multiplier for capabilities of health officer
            disable=False,
        ),
        # disables the health system constraints so all HSI events run
        symptommanager.SymptomManager(resourcefilepath=resources),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
        dx_algorithm_child.DxAlgorithmChild(),
        # dx_algorithm_adult.DxAlgorithmAdult(),
        healthburden.HealthBurden(resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check we can read the results
    log_df = parse_log_file(sim.log_filepath)
    df = sim.population.props

    # check no vaccine administered from 2019 onwards (through HSIs)
    # vaccines pre-2019 are assigned based on coverage and not dependent on health system capacity
    # check coverage in children born after 2019

    # eligible children born after 2019
    def get_coverage(condition, subset):
        total = sum(subset)
        has_condition = sum(condition & subset)
        coverage = has_condition / total * 100 if total else 0
        assert coverage <= 100
        return coverage

    # age should be <= end_date.year - 2019
    max_age = end_date.year - 2019
    susceptible_cohort = (df.age_years <= max_age)
    unvaccinated = get_coverage(df.va_measles == 0, susceptible_cohort)
    assert unvaccinated == 100

    # check measles incidence (in the unvaccinated cohort) is at pre-EPI levels
    output = log_df["tlo.methods.measles"]["incidence"]
    model_inc = output["inc_1000py"]

    assert model_inc.loc[len(model_inc)-1] > 2  # baseline incidence per 1000py




