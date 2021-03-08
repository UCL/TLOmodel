import pytest
import os
import pandas as pd
import numpy as np
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


def set_all_women_as_pregnant_and_reset_baseline_parity(sim):
    """Force all women of reproductive age to be pregnant at the start of the simulation"""
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)

    all_women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]
    df.loc[all_women.index, 'la_parity'] = 0

def turn_off_antenatal_pregnancy_loss(sim):
    params = sim.modules['PregnancySupervisor'].parameters

    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['induced_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['antenatal_stillbirth'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)

def register_core_modules():
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


def register_all_modules():
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

    return sim


@pytest.mark.group2
def test_run_core_modules_normal_allocation_of_pregnancy():
    """Runs the simulation using only core modules without maniuplation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = register_core_modules()
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


@pytest.mark.group2
def test_run_all_modules_normal_allocation_of_pregnancy():
    """Runs the simulation with all modules registered that can be called during pregnancy/labour/postnatal care
    without maniuplation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = register_all_modules()
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


@pytest.mark.group2
def test_run_all_modules_high_volumes_of_pregnancy():
    """Runs the simulation with the core modules and all women of reproductive age being pregnant at the start of the
    simulation"""

    sim = register_all_modules()
    sim.make_initial_population(n=1000)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=Date(2011, 1, 1))


@pytest.mark.group2
def test_run_with_all_births_as_twins():
    """Runs the simulation with the core modules, all reproductive age women as pregnant and forces all pregnancies to
    be twins"""
    sim = register_core_modules()
    sim.make_initial_population(n=500)

    # Force all reproductive age women to be pregnant and force all pregnancies to lead to twins
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    params = sim.modules['PregnancySupervisor'].parameters
    params_lab = sim.modules['Labour'].parameters

    params['ps_linear_equations']['multiples'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)
    params_lab['la_labour_equations']['intrapartum_still_birth'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    sim.simulate(end_date=Date(2011, 1, 1))
    df = sim.population.props

    # For any women pregnant at the end of the run, make sure they are pregnant with twins
    pregnant_at_end_of_sim = df.is_pregnant
    assert (df.loc[pregnant_at_end_of_sim.loc[pregnant_at_end_of_sim].index, 'ps_multiple_pregnancy']).all().all()

    # As intrapartum stillbirth is switched off, we check that all live births lead to 2 newborns (and that parity is
    # being recorded correctly)
    had_live_birth = (df.pn_id_most_recent_child > 0)
    assert (df.loc[had_live_birth.loc[had_live_birth].index, 'la_parity'] >= 2).all().all()

    # For all births check theyre twins and are matched to a sibling
    new_borns = df.date_of_birth > sim.start_date
    assert (df.loc[new_borns.loc[new_borns].index, 'nb_is_twin']).all().all()
    assert (df.loc[new_borns.loc[new_borns].index, 'nb_twin_sibling_id'] != -1).all().all()

    # TODO: linked death associated with stillbirth
    # TODO: check breastfeeding status is the same  (should this be in newborn tests)
    # todo: check PNC logic - although if this runs with no errors then should be ok


@pytest.mark.group2
def test_run_all_births_end_in_miscarriage():
    """Runs the simulation with the core modules and all women of reproductive age as pregnant. Sets miscarriage risk
    to 1 and runs checks """
    sim = register_core_modules()
    starting_population = 500

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    # Risk of ectopic applied before miscarriage so set to 0 so that only miscarriages lead to pregnancy loss
    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)

    # Set risk of miscarriage to 1
    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))
    df = sim.population.props

    # Check that there are no newborns
    possible_newborns = df.date_of_birth > sim.start_date
    assert possible_newborns.loc[possible_newborns].empty
    assert len(df) == starting_population

    # Check that all women have passed through abortion functions and this has been captured, check parity remains
    # static
    women_ever_pregnant = df.loc[~df.is_pregnant & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)
                                 & ~pd.isnull(df.date_of_last_pregnancy)]
    assert (df.loc[women_ever_pregnant.index, 'ps_prev_spont_abortion']).all().all()
    assert (df.loc[women_ever_pregnant.index, 'la_parity'] == 0).all().all()

    # Check that any woman who is still pregnant has not yet had risk of miscarriage applied (applied first at week 4)
    pregnant_at_end_of_sim = df.loc[df.is_pregnant]
    assert (df.loc[pregnant_at_end_of_sim.index, 'ps_gestational_age_in_weeks'] < 5).all().all()

    # todo: make sure no pregnancy complications are ever set
    # todo: force everyone to seek care and check they get care? (test in antenatal care)


@pytest.mark.group2
def test_run_all_births_end_in_abortion():
    """Runs the simulation with the core modules and all women of reproductive age as pregnant. Sets abortion risk
    to 1 and runs checks """
    sim = register_core_modules()
    starting_population = 500

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    # Define women of interest in the population
    df = sim.population.props
    women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]

    # Set parity to 0 for an additional check for births
    df.loc[women.index, 'la_parity'] = 0
    # Set all births as unintended to ensure women are at risk of abortion
    df.loc[women.index, 'co_unintended_preg'] = True

    # Risk of ectopic applied before miscarriage so set to 0
    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)

    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)

    params['ps_linear_equations']['induced_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))

    df = sim.population.props

    # Check that there are no newborns
    possible_newborns = df.date_of_birth > sim.start_date
    assert possible_newborns.loc[possible_newborns].empty
    assert len(df) == starting_population

    women_ever_pregnant = df.loc[~df.is_pregnant & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)
                                 & ~pd.isnull(df.date_of_last_pregnancy)]

    assert (df.loc[women_ever_pregnant.index, 'la_parity'] == 0).all().all()

    pregnant_at_end_of_sim = df.loc[df.is_pregnant]
    assert (df.loc[pregnant_at_end_of_sim.index, 'ps_gestational_age_in_weeks'] < 9).all().all()


@pytest.mark.group2
def test_run_all_births_end_antenatal_still_birth():
    """Runs the simulation with the core modules and all women of reproductive age as pregnant. Sets antenatal still
    birth risk to 1 and runs checks """
    sim = register_core_modules()
    starting_population = 500

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    # Only allow pregnancy loss from stillbirth
    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['induced_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['early_onset_labour'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['antenatal_stillbirth'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))
    df = sim.population.props

    # Check that there are no newborns
    possible_newborns = df.date_of_birth > sim.start_date
    assert possible_newborns.loc[possible_newborns].empty
    assert len(df) == starting_population

    # Check that all women who were ever pregnant didnt deliver a live birth and that previous still birth is captured
    # for all women
    women_ever_pregnant = df.loc[~df.is_pregnant & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)
                                 & ~pd.isnull(df.date_of_last_pregnancy)]

    assert (df.loc[women_ever_pregnant.index, 'la_parity'] == 0).all().all()
    assert (df.loc[women_ever_pregnant.index, 'ps_prev_stillbirth']).all().all()

    # If anyone remains pregnant at the end of sim, assert then are not greater than 28 weeks pregnant, where risk of
    # still birth is first applied
    pregnant_at_end_of_sim = df.loc[df.is_pregnant]
    assert (df.loc[pregnant_at_end_of_sim.index, 'ps_gestational_age_in_weeks'] < 28).all().all()


def test_run_all_births_end_ectopic_no_care_seeking():
    """Run the simulation with core modules forcing all pregnancies to be ectopic. We also remove care seeking and set
    risk of death to 1. Check no new births, all women who are pregnant at the start of the sim experience rupture and
    die. Any ongoing pregnancies should not exceed 9 weeks gestation"""
    sim = register_core_modules()
    starting_population = 500
    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    # We force all pregnancies to be ectopic, never trigger care seeking, always lead to rupture and death
    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['ectopic'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)
    params['ps_linear_equations']['care_seeking_pregnancy_loss'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['ectopic_pregnancy_death'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.simulate(end_date=Date(2011, 1, 1))

    # Check that there are no newborns
    df = sim.population.props
    possible_newborns = df.date_of_birth > sim.start_date
    assert possible_newborns.loc[possible_newborns].empty
    assert len(df) == starting_population

    women_ever_pregnant = df.loc[~df.is_pregnant & (df.sex == 'F') & ~pd.isnull(df.date_of_last_pregnancy)]
    women_still_pregnant = df.loc[df.is_pregnant]

    assert (False == (df.loc[women_ever_pregnant.index, 'is_alive'])).all()
    assert (df.loc[women_ever_pregnant.index, 'ps_ectopic_pregnancy'] == 'ruptured').all().all()
    assert (df.loc[women_ever_pregnant.index, 'la_parity'] == 0).all().all()

    # Check that all women, who were ever pregnant have died of their ectopic
    assert (df.loc[women_still_pregnant.index, 'ps_gestational_age_in_weeks'] < 9).all().all()

    # TODO: check treatment before rupture completely blocks death (anc tests)
    # TODO: check treatment after rupture reduces risk of death? (anc tests)


def test_preterm_labour_logic():
    """Runs sim to fully initialise the population. Sets all women pregnant and risk of preterm labour to 1. Checks
     logic and scheduling of preterm labour through the PregnancySupervisor event. Checks labour events correctly
     scheduled"""

    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    params = sim.modules['PregnancySupervisor'].parameters

    # We force risk of preterm birth to be 1, meaning all women will go into labour at month 5
    params['ps_linear_equations']['early_onset_labour'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    # And stop attendance to ANC (could block labour)
    params['prob_first_anc_visit_gestational_age'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Stop any pregnancies ending in pregnancy loss
    turn_off_antenatal_pregnancy_loss(sim)

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Clear events, including original scheduling of labour
    sim.event_queue.queue.clear()
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])

    df = sim.population.props
    pregnant_women = df.loc[df.is_pregnant]

    # Run the event to generate MNI dictionary 1 week after the sim has started
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup.apply(sim.population)

    # Move the sim date forward
    sim.date = sim.date + pd.DateOffset(weeks=19)

    # Run the event again to apply risk of preterm birth
    pregnancy_sup.apply(sim.population)

    # check due date for all women has been reset to occur in less than one months time
    latest_labour_can_happen = sim.date + pd.DateOffset(days=((27 - 22) * 7))
    assert (df.loc[pregnant_women.index, 'la_due_date_current_pregnancy'] <= latest_labour_can_happen).all().all()

    # Check that each woman has had a new labour onset event scheduled (original event has been cleared so if logic
    # hasnt held no labour onset event would have been set)
    mother_id = pregnant_women.index[0]
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.LabourOnsetEvent in events

    # move time forward to due date
    sim.date = df.at[mother_id, 'la_due_date_current_pregnancy']

    # Call checker function from labour which returns BOOl if labour can proceed and check labour can proceed for
    # all women
    can_labour_now_go_ahead = sim.modules['Labour'].check_labour_can_proceed(mother_id)

    # todo this might crash if the mother dies? maybe not needed?
    assert can_labour_now_go_ahead

    # Next we run the labour onset event and check that the key events are scheduled (birth and death events)
    labour_event = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_event.apply(mother_id)

    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.LabourDeathAndStillBirthEvent in events
    assert labour.BirthEvent in events

    # TODO: check newborns are all preterm (in newborn module?)
    # TODO: check for different gestational age at onset? i.e. force labour onset later


def test_check_first_anc_visit_scheduling():
    """Runs sim to fully initialise the population. Check that when pregnancy supervisor event is ran on pregnancy
    population of the correct gestational age that anentanatal care is always scheduled when prob set to 1"""

    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)

    # Set parameters so that women will attend ANC in 1 months time
    params = sim.modules['PregnancySupervisor'].parameters
    params['prob_first_anc_visit_gestational_age'] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Run sim and clear event queue
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    df = sim.population.props

    # Define and run the pregnancy supervisor event (move date forward 1 week)
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup.apply(sim.population)

    # Check the ps_date_of_anc1 property as its used to schedule ANC. Make sure ANC will occur in one month and before
    # two months
    earliest_anc_can_happen = sim.date + pd.DateOffset(days=30)
    latest_anc_can_happen = sim.date + pd.DateOffset(days=59)

    pregnant_women = df.loc[df.is_pregnant]
    assert (df.loc[pregnant_women.index, 'ps_date_of_anc1'] >= earliest_anc_can_happen).all().all()
    assert (df.loc[pregnant_women.index, 'ps_date_of_anc1'] <= latest_anc_can_happen).all().all()

    # Finally check that the HSI event has been correctly scheduled
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact in hsi_events


def test_store_dalys_in_mni_function_and_daly_calculations():
    """This tesst checks how we calcualte dalys in the model."""

    # Set up sim and run for 0 days
    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Store functions from pregnancy supervisor as variables
    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    store_dalys_in_mni = sim.modules['PregnancySupervisor'].store_dalys_in_mni
    params = sim.modules['PregnancySupervisor'].parameters

    # Select pregnant woman from dataframe
    pregnant_women = df.loc[df.is_pregnant]
    mother_id = pregnant_women.index[0]

    # Generate the MNI dictionary (full dictionary has to be created as functions cycle through each key)
    # TODO: just run preg sup event for woman to generate mni (might not be empty though?)
    mni[mother_id] = {'delete_mni': False, 'abortion_onset': pd.NaT, 'abortion_haem_onset': pd.NaT,
                      'abortion_sep_onset': pd.NaT, 'eclampsia_onset': pd.NaT, 'mild_mod_aph_onset': pd.NaT,
                      'severe_aph_onset': pd.NaT, 'chorio_onset': pd.NaT, 'ectopic_onset': pd.NaT,
                      'ectopic_rupture_onset': pd.NaT, 'gest_diab_onset': pd.NaT, 'gest_diab_diagnosed_onset': pd.NaT,
                      'gest_diab_resolution': pd.NaT, 'mild_anaemia_onset': pd.NaT, 'mild_anaemia_resolution': pd.NaT,
                      'moderate_anaemia_onset': pd.NaT, 'moderate_anaemia_resolution': pd.NaT,
                      'severe_anaemia_onset': pd.NaT, 'severe_anaemia_resolution': pd.NaT,
                      'mild_anaemia_pp_onset': pd.NaT, 'mild_anaemia_pp_resolution': pd.NaT,
                      'moderate_anaemia_pp_onset': pd.NaT, 'moderate_anaemia_pp_resolution': pd.NaT,
                      'severe_anaemia_pp_onset': pd.NaT, 'severe_anaemia_pp_resolution': pd.NaT,
                      'hypertension_onset': pd.NaT, 'hypertension_resolution': pd.NaT,
                      'obstructed_labour_onset': pd.NaT, 'sepsis_onset': pd.NaT, 'uterine_rupture_onset': pd.NaT,
                      'mild_mod_pph_onset': pd.NaT, 'severe_pph_onset': pd.NaT, 'secondary_pph_onset': pd.NaT,
                      'vesicovaginal_fistula_onset': pd.NaT,  'vesicovaginal_fistula_resolution': pd.NaT,
                      'rectovaginal_fistula_onset': pd.NaT, 'rectovaginal_fistula_resolution': pd.NaT}

    # First we test the logic for 'acute' complications
    # Call store_dalys_in_mni function which is called when any woman experiences one of the complications stored in
    # the mni
    store_dalys_in_mni(mother_id, 'ectopic_onset')
    store_dalys_in_mni(mother_id, 'ectopic_rupture_onset')

    # Check that the correct date of onset has been captured
    assert mni[mother_id]['ectopic_onset'] == sim.date
    assert mni[mother_id]['ectopic_rupture_onset'] == sim.date

    # Call the report_daly_function which contains the functionality to calculate the daly weight this woman should have
    # following her complications
    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()
    assert dalys_from_pregnancy.loc[mother_id] > 0

    # We assume any of the 'acute' complications cause women to accrue 7 days of disability weight
    ectopic_preg_weight = (params['ps_daly_weights']['ectopic'] / 52)
    ectopic_rupture_weight = (params['ps_daly_weights']['ectopic_rupture'] / 52)

    # Therefore for this month this woman should have acrussed the sum of these two weight values
    # (rounded to allow comparison)
    total_weight = round((ectopic_preg_weight + ectopic_rupture_weight), 4)

    # Now we use the output returned from the report_daly_values function and check that the function has correctly
    # calculated the weight for this month and store that in the output against this mothers index
    rounded_dalys = round(dalys_from_pregnancy.loc[mother_id], 4)
    assert total_weight == rounded_dalys

    # Chronic complications
    # Now we check the logic for complications that acrue daly weights every day
    # Store onset of severe anaemia and check the correct date has been stored
    store_dalys_in_mni(mother_id, 'severe_anaemia_onset')
    assert mni[mother_id]['severe_anaemia_onset'] == sim.date

    # Move the date forward 1 month and run the report_daly_values
    sim.date = sim.date + pd.DateOffset(months=1)
    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()

    # Check a daly weight has been stored for this woman
    assert dalys_from_pregnancy.loc[mother_id] > 0

    # This woman has had this complication for the entire month (01/01/2010 - 01/02/2010) and it has not resolved,
    # therefore we expect her to have accrued 1 months weight
    sev_anemia_weight = params['ps_daly_weights']['severe_anaemia']  # TODO: im rethinking this weight
    reported_weight = dalys_from_pregnancy.loc[mother_id]
    assert sev_anemia_weight == reported_weight

    # Move the date forward 2 weeks and set the date of resolution for the complication
    sim.date = sim.date + pd.DateOffset(weeks=2)
    store_dalys_in_mni(mother_id, 'severe_anaemia_resolution')

    # Move the date forward another 2 weeks and call report_daly_values. This woman will only have experienced the
    # complication for half the month as it was resolved 2 weeks prior to report_daly_values being called
    sim.date = sim.date + pd.DateOffset(weeks=2)
    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()

    # Ensure some weight was captured
    assert dalys_from_pregnancy.loc[mother_id] > 0
    # Ensure the complication variables are reset within the report_daly_values
    assert pd.isnull(mni[mother_id]['severe_anaemia_onset'])
    assert pd.isnull(mni[mother_id]['severe_anaemia_resolution'])

    # We know she has experience 15 days of complication this month, check the function returns the correct daly weight
    sev_anemia_weight = round((params['ps_daly_weights']['severe_anaemia'] / 365) * 15, 3)
    reported_weight = round(dalys_from_pregnancy.loc[mother_id], 3)

    assert sev_anemia_weight == reported_weight


def test_calculation_of_gestational_age():
    """This is a simple test to check that when called, the pregnancy supervisor event updates the age of all womens
    gestational age correctly"""
    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    df = sim.population.props

    pregnant_women = df.is_pregnant & df.is_alive

    # todo: replace for loop
    for person in pregnant_women.loc[pregnant_women].index:
        random_days = sim.modules['PregnancySupervisor'].rng.randint(1, 274)
        df.at[person, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(days=random_days)

    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # todo: THIS TEST DOSENT REALY WORK, ITS LIKE THE PREGNANCY SUPERVISOR EVEN DOESNT RUN ON THE UPDATED VERSION OF
    #  THE DATAFRAME

    preg_new_slice =df.loc[df.is_pregnant & df.is_alive]

    assert (df.loc[preg_new_slice.index, 'ps_gestational_age_in_weeks'] != 0).all().all()

    for person in preg_new_slice.index:
        foetal_age_weeks = np.ceil((sim.date - df.at[person, 'date_of_last_pregnancy']) / np.timedelta64(1, 'W'))
        assert df.at[person, 'ps_gestational_age_in_weeks'] == (foetal_age_weeks + 2)


def test_pregnancy_supervisor_anaemia():
    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)

    # Set the risk of nutritional deficiences to 1
    params = sim.modules['PregnancySupervisor'].parameters
    params['prob_iron_def_per_month'] = 1
    params['prob_folate_def_per_month'] = 1
    params['prob_b12_def_per_month'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    df = sim.population.props

    # Move the date forward 1 week and run pregnancy supervisor event to generate mni dictionary (essential for event to
    # run at higher gestations)
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Move the date forward again, this will mean when the event is run women will be 4 weeks pregnant, this is the
    # time point at which risk of anaemia is first applied
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup.apply(sim.population)

    # All pregnant women should now have be deficient of iron, folate and B12 (as risk == 1)
    pregnant_women = df.loc[df.is_pregnant]

    assert (sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(
        df.is_pregnant & df.is_alive, 'iron')).all().all()
    assert (sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(
        df.is_pregnant & df.is_alive, 'folate')).all().all()
    assert (sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(
        df.is_pregnant & df.is_alive, 'b12')).all().all()

    # Reset anaemia status
    df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] = 'none'

    # Set the relative risk of anaemia due to deficiences as very high to force anaemia on all women (women have
    # baseline risk which is increased by nutrient deficiencies)
    params['rr_anaemia_if_iron_deficient'] = 10
    params['rr_anaemia_if_folate_deficient'] = 10
    params['rr_anaemia_if_b12_deficient'] = 10

    # set probability that any cases of anaemia should be severe
    params['prob_mild_mod_sev_anaemia'] = [0, 0, 1]

    # Move the date so that women are now 8 weeks pregnant (next time risk of anaemia is applied in the event)
    sim.date = sim.date + pd.DateOffset(weeks=4)
    pregnancy_sup.apply(sim.population)

    # Check all women have developed severe anaemia as they should have and that the correct dates have been stored in
    # the mni
    assert (df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] == 'severe').all().all()
    for person in pregnant_women.index:
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_onset'] == sim.date)
        assert pd.isnull(sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_'
                                                                                            'resolution'])

    # Reset anaemia status in the women of interest and set that they are receiving iron and folic acid treatment, which
    # should reduce the risk of iron or folate deficiency (which increase risk of anaemia)
    df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] = 'none'
    df.loc[pregnant_women.index, 'ac_receiving_iron_folic_acid'] = True

    # Unset deficiencies
    sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.unset(pregnant_women.index, 'iron', 'folate',
                                                                       'b12')

    # Set treatment effect to 0 - this should mean that when the even runs, despite risk of iron/folate deficiency
    # being 1, women on treatment shoudl be prevented from experiencing deficiency
    params['treatment_effect_iron_def_ifa'] = 0
    params['treatment_effect_folate_def_ifa'] = 0

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    # Check no women are deficient of iron/folate i.e. treatment is working as expected
    assert (~sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(
        df.is_pregnant & df.is_alive, 'iron')).all().all()
    assert (~sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(
        df.is_pregnant & df.is_alive, 'folate')).all().all()


def test_pregnancy_supervisor_placental_conditions_and_antepartum_haemorrhage():
    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)

    params = sim.modules['PregnancySupervisor'].parameters
    params['ps_linear_equations']['placenta_praevia'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)
    params['ps_linear_equations']['placental_abruption'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)
    params['ps_linear_equations']['care_seeking_pregnancy_complication'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    params['prob_aph_placenta_praevia'] = 1
    params['prob_aph_placental_abruption'] = 1
    params['prob_mod_sev_aph'] = [0, 1]
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup.apply(sim.population)

    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    assert (df.loc[pregnant_women.index, 'ps_placenta_praevia']).all().all()

    sim.date = sim.date + pd.DateOffset(weeks=19)
    pregnancy_sup.apply(sim.population)
    assert (df.loc[pregnant_women.index, 'ps_placental_abruption']).all().all()
    assert (df.loc[pregnant_women.index, 'ps_antepartum_haemorrhage'] == 'severe').all().all()
    for person in pregnant_women.index:
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_aph_onset'] == sim.date)

    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment in hsi_events

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df.loc[pregnant_women.index, 'ps_antepartum_haemorrhage'] = 'none'
    df.loc[pregnant_women.index, 'ps_emergency_event'] = False

    params['ps_linear_equations']['care_seeking_pregnancy_complication'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)
    params['ps_linear_equations']['antepartum_haemorrhage_death'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    mother_id = pregnant_women.index[0]
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert demography.InstantaneousDeath in events

    # todo check mni delted
    # todo check stillbirth

test_pregnancy_supervisor_placental_conditions_and_antepartum_haemorrhage()
def test_pregnancy_supervisor_hypertensive_disorders():
    pass
def test_pregnancy_supervisor_gdm():
    pass
def test_pregnancy_supervisor_chorio_and_prom():
    pass
