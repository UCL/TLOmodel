""" Tests for setting up the HIV and TB scenarios used for projections """

from pathlib import Path
import os

import pandas as pd

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


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim():
    """
    register all necessary modules for the tests to run
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0)

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
            store_hsi_events_that_have_run=False,  # convenience function for debugging
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        hiv.Hiv(resourcefilepath=resourcefilepath),
        tb.Tb(resourcefilepath=resourcefilepath),
    )

    return sim


def test_scenario_ipt_expansion():
    """ test scenario IPT expansion is set up correctly
    should be expanded age eligibility in scenarios 2 and 4
    otherwise only ages <5 are eligible
    """

    end_date = Date(2012, 12, 31)
    popsize = 1000

    sim = get_sim()

    # stop PLHIV getting IPT for purpose of tests
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_plhiv = 0
    # set coverage of IPT for TB contacts to 1.0
    sim.modules['Tb'].parameters['ipt_coverage'].coverage_paediatric = 0

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

    # assign person_id active tb
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'age_exact_years'] = 20
    df.at[person_id, 'age_years'] = 20

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    for symptom in symptom_list:
        sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string=symptom,
            add_or_remove="+",
            disease_module=sim.modules['Tb'],
            duration_in_days=None,
        )

    # run diagnosis (HSI_Tb_ScreeningAndRefer) for person 0
    assert "tb_sputum_test" in sim.modules["HealthSystem"].dx_manager.dx_tests
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_ever_tested']
    assert df.at[person_id, 'tb_diagnosed']

    # check ages of those scheduled for HSI_Tb_Start_or_Continue_Ipt
    list_of_events = list()

    for ev_tuple in sim.modules['HealthSystem'].HSI_EVENT_QUEUE:
        date = ev_tuple[1]  # this is the 'topen' value
        event = ev_tuple[4]
        if isinstance(event, tb.HSI_Tb_Start_or_Continue_Ipt):
            list_of_events.append((date, event, event.target))

    # change sim.date to post-scenario start date
    # run ScenarioSetupEvent - should change parameter "age_eligibility_for_ipt"
    # run HSI_Tb_ScreeningAndRefer
    # check ages again of those scheduled for HSI_Tb_Start_or_Continue_Ipt


    # ---------- change scenario to 2 ---------- #
    # reset population
    sim.make_initial_population(n=popsize)

    df = sim.population.props

    # change scenario to 2 (expanded access to IPT: all ages)
    sim.modules['Tb'].parameters['scenario'] = 2

















