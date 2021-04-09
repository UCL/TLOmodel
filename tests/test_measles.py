import os
from pathlib import Path

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

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)

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


# def test_measles_cases_and_hsi_occurring(tmpdir):
#
#     log_config = {
#         "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
#         "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
#         "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
#             "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
#             "tlo.methods.measles": logging.INFO,
#             "tlo.methods.healthsystem": logging.INFO,
#         }
#     }
#     popsize = 1000
#
#     sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
#
#     sim.register(
#         demography.Demography(resourcefilepath=resources),
#         healthsystem.HealthSystem(
#             resourcefilepath=resources,
#             service_availability=["*"],  # all treatment IDs allowed
#             mode_appt_constraints=0,
#             ignore_cons_constraints=True,
#             ignore_priority=True,
#             capabilities_coefficient=1.0,  # multiplier for capabilities of health officer
#             disable=False,
#         ),
#         # disables the health system constraints so all HSI events run
#         symptommanager.SymptomManager(resourcefilepath=resources),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
#         dx_algorithm_child.DxAlgorithmChild(),
#         healthburden.HealthBurden(resourcefilepath=resources),
#         contraception.Contraception(resourcefilepath=resources),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
#         labour.Labour(resourcefilepath=resources),
#         antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
#         pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
#         postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
#         newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
#         epi.Epi(resourcefilepath=resources),
#         measles.Measles(resourcefilepath=resources),
#     )
#
#     sim.make_initial_population(n=popsize)
#     sim.simulate(end_date=end_date)
#     check_dtypes(sim)
#
#     # check we can read the results
#     log_df = parse_log_file(sim.log_filepath)
#     df = sim.population.props
#
#     # check people getting measles
#     # assert df["me_has_measles"].sum > 0  # current cases of measles
#     total_inc = log_df["tlo.methods.measles"]["incidence"]["inc_1000people"]
#     assert total_inc.sum() > 0
#
#     # check symptoms assigned - all those currently with measles should have rash
#     # there is an incubation period, so infected people may not have rash immediately
#     # if on treatment, must have rash for diagnosis
#     has_rash = sim.modules['SymptomManager'].who_has('rash')
#     current_measles_tx = df.index[df.is_alive & df.me_on_treatment]
#     if current_measles_tx.any():
#         assert set(current_measles_tx) < set(has_rash)


def test_measles_high_death_rate(tmpdir):

    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }
    popsize = 100

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

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

    # Increase death:
    symptom_prob = sim.modules['Measles'].parameters["symptom_prob"]
    symptom_prob.loc[:, 'probability'] = 1

    # set beta to 0 so no new cases occurring
    sim.modules['Measles'].parameters['beta_baseline'] = 0

    sim.make_initial_population(n=popsize)
    df = sim.population.props

    # set all people in baseline population to have measles infection
    df.loc[df.is_alive, "me_has_measles"] = True

    sim.simulate(end_date=Date(2010, 6, 1))

    # check if any measles deaths occurred
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()

    # check all cases of measles also had a measles death
    assert df.cause_of_death.loc[~df.is_alive & df.me_has_measles].str.startswith('measles').all()


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

    # Change death rate to zero
    symptom_prob = sim.modules['Measles'].parameters["symptom_prob"]
    symptom_prob.loc[symptom_prob.symptom == "death", "probability"].values[0] = 0

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check that there have been no deaths caused by measles
    df = sim.population.props
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('measles').any()
