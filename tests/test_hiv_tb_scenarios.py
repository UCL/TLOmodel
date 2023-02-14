""" Tests for setting up the HIV and TB scenarios used for projections """

import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def get_sim(seed):
    """
    register all necessary modules for the tests to run
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=["*"],  # all treatment allowed
            mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
            cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
            ignore_priority=True,  # do not use the priority information in HSI event to schedule
            capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
            disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
            disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        hiv.Hiv(resourcefilepath=resourcefilepath),
        tb.Tb(resourcefilepath=resourcefilepath),
    )

    return sim


def test_scenario_parameters(seed):

    sim = get_sim(seed=seed)
    sim.modules["Tb"].parameters["scenario"] = 3
    sim.make_initial_population(n=10)

    scenario_change_event = tb.ScenarioSetupEvent(module=sim.modules['Tb'])
    scenario_change_event.apply(sim.population)

    # check parameters have changed for scenario 3
    assert sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] == 0.5
    assert sim.modules["Hiv"].parameters["prob_prep_for_agyw"] == 0.1
    assert sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] == 0.75
    assert sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] == 0.25
    assert sim.modules["Tb"].parameters["age_eligibility_for_ipt"] == 100
    assert sim.modules["Tb"].parameters["ipt_coverage"]["coverage_plhiv"].all() >= 0.6
    assert (sim.modules["Tb"].parameters["ipt_coverage"]["coverage_paediatric"] == 80).all()


@pytest.mark.slow
def test_scenario_ipt_expansion(seed):
    """ test scenario IPT expansion is set up correctly
    should be expanded age eligibility in scenario 3
    otherwise only ages <5 are eligible
    """

    popsize = 100

    sim = get_sim(seed=seed)

    # stop PLHIV getting IPT for purpose of tests
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_plhiv = 0
    # set coverage of IPT for TB contacts to 1.0
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_paediatric = 100

    # Make the population
    sim.make_initial_population(n=popsize)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # check default scenario is set to 0
    assert sim.modules['Tb'].parameters['scenario'] == 0

    # ipt eligibility should be limited to ages <=5 years
    # assign all population into one district - so all are eligible as contacts
    df.loc[df.is_alive, 'district_of_residence'] = 'Blantyre'

    # assign active TB to person 0
    person_id = 0

    # assign person_id 0 active tb
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'age_exact_years'] = 20
    df.at[person_id, 'age_years'] = 20

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # run diagnosis (HSI_Tb_ScreeningAndRefer) for person 0
    assert "tb_sputum_test_smear_positive" in sim.modules["HealthSystem"].dx_manager.dx_tests
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert pd.notnull(df.at[person_id, 'tb_date_tested'])
    assert df.at[person_id, 'tb_diagnosed']

    # check ages of those scheduled for HSI_Tb_Start_or_Continue_Ipt
    list_of_events = list()

    for ev_tuple in sim.modules['HealthSystem'].HSI_EVENT_QUEUE:
        date = ev_tuple[1]  # this is the 'topen' value
        event = ev_tuple[4]
        if isinstance(event, tb.HSI_Tb_Start_or_Continue_Ipt):
            list_of_events.append((date, event, event.target))

    idx_of_ipt_candidates = [x[2] for x in list_of_events]
    ages_of_ipt_candidates = df.loc[idx_of_ipt_candidates, "age_exact_years"]
    assert (ages_of_ipt_candidates < 6).all()

    # run ScenarioSetupEvent - should not change parameter "age_eligibility_for_ipt"
    progression_event = tb.ScenarioSetupEvent(module=sim.modules['Tb'])
    progression_event.apply(population=sim.population)
    assert sim.modules["Tb"].parameters["age_eligibility_for_ipt"] == 5.0

    # ---------- change scenario to 3 ---------- #
    # reset population
    sim = get_sim(seed=seed)

    # change scenario to 3 (expanded access to IPT: all ages)
    sim.modules['Tb'].parameters['scenario'] = 3

    # Make the population
    sim.make_initial_population(n=popsize)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # check scenario is set to 3
    assert sim.modules['Tb'].parameters['scenario'] == 3

    # assign all population into one district - so all are eligible as contacts
    df.loc[df.is_alive, 'district_of_residence'] = 'Blantyre'

    # run ScenarioSetupEvent - should change parameter "age_eligibility_for_ipt"
    progression_event = tb.ScenarioSetupEvent(module=sim.modules['Tb'])
    progression_event.apply(population=sim.population)

    # these parameters are changed during ScenarioSetupEvent so reset them
    # stop PLHIV getting IPT for purpose of tests
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_plhiv = 0
    # set coverage of IPT for TB contacts to 1.0
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_paediatric = 100

    assert sim.modules["Tb"].parameters["age_eligibility_for_ipt"] >= 5.0

    # assign another person active TB
    person_id = 3

    # assign person_id active tb
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'age_exact_years'] = 20
    df.at[person_id, 'age_years'] = 20

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # run HSI_Tb_ScreeningAndRefer for person 3
    # check ages again of those scheduled for HSI_Tb_Start_or_Continue_Ipt
    assert "tb_sputum_test_smear_positive" in sim.modules["HealthSystem"].dx_manager.dx_tests
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert pd.notnull(df.at[person_id, 'tb_date_tested'])
    assert df.at[person_id, 'tb_diagnosed']

    # check ages of those scheduled for HSI_Tb_Start_or_Continue_Ipt
    list_of_events = list()

    for ev_tuple in sim.modules['HealthSystem'].HSI_EVENT_QUEUE:
        date = ev_tuple[1]  # this is the 'topen' value
        event = ev_tuple[4]
        if isinstance(event, tb.HSI_Tb_Start_or_Continue_Ipt):
            list_of_events.append((date, event, event.target))

    idx_of_ipt_candidates = [x[2] for x in list_of_events]
    ages_of_ipt_candidates = df.loc[idx_of_ipt_candidates, "age_exact_years"]
    # make sure at least one candidate is over 5 years old
    assert (ages_of_ipt_candidates > 5.0).any()


@pytest.mark.slow
def test_check_tb_test_under_each_scenario(seed):
    """ test correct test is scheduled under each scenario
    """

    popsize = 10

    sim = get_sim(seed=seed)

    # Make the population
    sim.make_initial_population(n=popsize)

    sim.modules['Tb'].parameters["prop_presumptive_mdr_has_xpert"] = 1.0  # xpert always available
    sim.modules['Tb'].parameters["sens_xpert"] = 1.0  # increase sensitivity of xpert testing

    # ------------------------- scenario 0 ------------------------- #
    sim.modules['Tb'].parameters['scenario'] = 0

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # assign person_id active tb
    hiv_neg_person = 0
    hiv_pos_person = 1
    both_people = [hiv_neg_person, hiv_pos_person]

    df.at[both_people, 'tb_inf'] = 'active'
    df.at[both_people, 'tb_strain'] = 'ds'
    df.at[both_people, 'tb_date_active'] = sim.date
    df.at[both_people, 'tb_smear'] = True
    df.at[both_people, 'age_exact_years'] = 20
    df.at[both_people, 'age_years'] = 20

    # set HIV status
    df.at[hiv_neg_person, 'hv_inf'] = False
    df.at[hiv_pos_person, 'hv_inf'] = True
    df.at[hiv_neg_person, 'hv_diagnosed'] = False  # this is used for tb test selection
    df.at[hiv_pos_person, 'hv_diagnosed'] = True

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    sim.modules["SymptomManager"].change_symptom(
        person_id=both_people,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # select test for each person under baseline scenario - standard guidelines
    assert "sputum" == sim.modules["Tb"].select_tb_test(hiv_neg_person)
    assert "xpert" == sim.modules["Tb"].select_tb_test(hiv_pos_person)

    # screen and test hiv_neg_person
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=hiv_neg_person,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=hiv_neg_person, squeeze_factor=0.0)

    assert pd.notnull(df.at[hiv_neg_person, 'tb_date_tested'])
    assert df.at[hiv_neg_person, 'tb_diagnosed']
    assert not df.at[hiv_neg_person, 'tb_diagnosed_mdr']

    # screen and test hiv_pos_person
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=hiv_pos_person,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=hiv_pos_person, squeeze_factor=0.0)

    assert pd.notnull(df.at[hiv_pos_person, 'tb_date_tested'])
    assert df.at[hiv_pos_person, 'tb_diagnosed']
    assert not df.at[hiv_pos_person, 'tb_diagnosed_mdr']

    # apply scenario change, re-test, should be same
    scenario_change_event = tb.ScenarioSetupEvent(module=sim.modules['Tb'])
    scenario_change_event.apply(sim.population)

    # select test for each person under baseline scenario - standard guidelines
    # this person should still qualify for sputum as they have not been treated
    assert "sputum" == sim.modules["Tb"].select_tb_test(hiv_neg_person)
    assert "xpert" == sim.modules["Tb"].select_tb_test(hiv_pos_person)

    # ------------------------- scenario 1 ------------------------- #
    sim = get_sim(seed=seed)

    # Make the population
    sim.make_initial_population(n=popsize)

    sim.modules['Tb'].parameters["prop_presumptive_mdr_has_xpert"] = 1.0  # xpert always available
    sim.modules['Tb'].parameters["sens_xpert"] = 1.0  # increase sensitivity of xpert testing
    sim.modules['Tb'].parameters['scenario'] = 1

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # assign person_id active tb
    hiv_neg_person = 0
    hiv_pos_person = 1
    both_people = [hiv_neg_person, hiv_pos_person]

    df.at[both_people, 'tb_inf'] = 'active'
    df.at[both_people, 'tb_strain'] = 'ds'
    df.at[both_people, 'tb_date_active'] = sim.date
    df.at[both_people, 'tb_smear'] = True
    df.at[both_people, 'age_exact_years'] = 20
    df.at[both_people, 'age_years'] = 20
    # set HIV status
    df.at[hiv_neg_person, 'hv_inf'] = False
    df.at[hiv_pos_person, 'hv_inf'] = True
    df.at[hiv_pos_person, 'hv_diagnosed'] = True

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    sim.modules["SymptomManager"].change_symptom(
        person_id=both_people,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # select test for each person under scenario 1, should be standard at first
    assert "sputum" == sim.modules["Tb"].select_tb_test(hiv_neg_person)
    assert "xpert" == sim.modules["Tb"].select_tb_test(hiv_pos_person)

    # apply scenario change, re-test, should be xpert for all
    scenario_change_event = tb.ScenarioSetupEvent(module=sim.modules['Tb'])
    scenario_change_event.apply(sim.population)

    # select test for each person under changed guidelines
    assert "xpert" == sim.modules["Tb"].select_tb_test(hiv_neg_person)
    assert "xpert" == sim.modules["Tb"].select_tb_test(hiv_pos_person)
