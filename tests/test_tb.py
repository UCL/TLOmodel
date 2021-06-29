""" Tests for for the TB Module """

from pathlib import Path
import os

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    healthburden,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    simplified_births,
    symptommanager,
    epi,
    hiv,
    tb
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


def get_sim(use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True):
    """
    get sim with the checks for configuration of properties running in the TB module
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    if use_simplified_birth:
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=disable_HS,
                         ignore_cons_constraints=ignore_con_constraints),
                     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                     dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                     epi.Epi(resourcefilepath=resourcefilepath),
                     hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False),
                     tb.Tb(resourcefilepath=resourcefilepath),
                     )
    else:
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                     care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                     labour.Labour(resourcefilepath=resourcefilepath),
                     newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                     postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True,
                         ignore_cons_constraints=True),
                     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                     dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                     epi.Epi(resourcefilepath=resourcefilepath),
                     hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False),
                     tb.Tb(resourcefilepath=resourcefilepath),
                     )

    return sim


# simple checks
def test_basic_run():
    """ test basic run and properties assigned correctly
    """

    end_date = Date(2012, 12, 31)
    popsize = 1000

    sim = get_sim(use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # set high transmission rate and all are fast progressors
    sim.modules['Tb'].parameters['transmission_rate'] = 0.5
    sim.modules['Tb'].parameters['prop_fast_progressor'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)

    df = sim.population.props

    # check properties assigned correctly for baseline population
    # should be some latent infections, no active infections
    num_latent = len(df[(df.tb_inf == 'latent') & df.is_alive])
    prev_latent = num_latent / len(df[df.is_alive])
    assert prev_latent > 0

    # todo check this works
    assert not pd.isnull(df.loc[~df.date_of_birth.isna() & df.is_alive, [
        'tb_inf',
        'tb_strain',
        'tb_date_latent']
                         ]).all().all()

    # no-one should be on tb treatment yet
    assert not df.tb_on_treatment.any()
    assert pd.isnull(df.tb_date_treated).all()

    # run the simulation
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props  # updated dataframe

    # some should have treatment dates
    assert not pd.isnull(df.loc[~df.date_of_birth.isna() & df.is_alive, [
        'tb_on_treatment',
        'tb_date_treated',
        'tb_ever_treated',
        'tb_diagnosed']
                         ]).all().all()


# check natural history of TB infection
def test_natural_history():
    """
    test natural history and progression
    need to have disable_HS=False to ensure events enter queue
    otherwise they do run, but without entering queue
    find_events_for_person() checks the event queue
    disable=true, runs all hsi events but doesn't use queue so you won't find them
    """

    popsize = 10

    sim = get_sim(use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # set all to be fast progressors
    sim.modules['Tb'].parameters['prop_fast_progressor'] = 1.0
    sim.modules['Tb'].parameters['prop_smear_positive'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)
    # simulate for 0 days, just get everthing set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # select an adult who is alive with latent tb
    person_id = df.loc[df.is_alive & (df.tb_inf == 'latent') &
                       df.age_years.between(15, 80)].index[0]
    assert person_id  # check person_id has been identified

    # set tb strain to ds
    df.at[person_id, 'tb_strain'] = 'ds'
    # set hiv status to uninfected
    df.at[person_id, 'hv_inf'] = False

    # run TB polling event to schedule progression to active stage
    progression_event = tb.TbRegularPollingEvent(module=sim.modules['Tb'])
    progression_event.apply(population=sim.population)

    # check if TbActiveEvent was scheduled
    date_active_event, active_event = \
        [ev for ev in sim.find_events_for_person(person_id) if isinstance(ev[1], tb.TbActiveEvent)][0]
    assert date_active_event >= sim.date

    # run TbActiveEvent
    active_event_run = tb.TbActiveEvent(module=sim.modules['Tb'], person_id=person_id)
    active_event_run.apply(person_id)

    # check properties set
    assert df.at[person_id, 'tb_inf'] == 'active'
    assert df.at[person_id, 'tb_date_active'] == sim.date
    assert df.at[person_id, 'tb_smear']

    # check symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    assert set(sim.modules['SymptomManager'].has_what(person_id)) == symptom_list

    # run HSI_Tb_ScreeningAndRefer and check outcomes
    # this schedules the event
    sim.modules['HealthSystem'].schedule_hsi_event(
        tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb']),
        topen=sim.date,
        tclose=None,
        priority=0
    )

    # Check person_id has a ScreeningAndRefer event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_ScreeningAndRefer)
    ][0]
    assert date_event == sim.date

    list_of_hsi = [
        'tb.HSI_Tb_ScreeningAndRefer',
        'tb.HSI_Tb_StartTreatment'
    ]

    for name_of_hsi in list_of_hsi:
        hsi_event = eval(name_of_hsi +
                         "(person_id=" +
                         str(person_id) +
                         ", "
                         "module=sim.modules['Tb'],"
                         ""
                         ")"
                         )
        hsi_event.run(squeeze_factor=0)

    assert df.at[person_id, 'tb_ever_tested']
    assert df.at[person_id, 'tb_diagnosed']


def test_treatment_schedule():
    """ test treatment schedules
    check dates of follow-up appts following schedule
    check treatment ends at appropriate time
    """

    popsize = 10

    # disable HS, all HSI events will run, but won't be in the HSI queue
    # they will enter the sim.event_queue
    sim = get_sim(use_simplified_birth=True, disable_HS=True, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # change prob treatment success for tb treatment end checks
    sim.modules['Tb'].parameters['prob_tx_success_new'] = 1.0

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
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

    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_ever_tested']
    assert df.at[person_id, 'tb_diagnosed']
    assert not df.at[person_id, 'tb_diagnosed_mdr']

    # schedule treatment start
    # this calls clinical_monitoring which should schedule all follow-up appts
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # clinical monitoring
    # default clinical monitoring schedule for first infection ds-tb
    follow_up_times = sim.modules['Tb'].parameters['followup_times']
    clinical_fup = follow_up_times['ds_clinical_monitor'].dropna()
    number_fup_appts = len(clinical_fup)

    # count events scheduled currently
    t1 = len(sim.event_queue.queue)

    # this will schedule future follow-up appts depdendent on strain/retreatment status
    sim.modules['Tb'].clinical_monitoring(person_id=person_id)
    # check should be 6 more events now
    t2 = len(sim.event_queue.queue)
    assert t2 - t1 == number_fup_appts

    # end treatment, change sim.date so person will be ready to stop treatment
    sim.date = Date(2010, 12, 31)
    tx_end_event = tb.TbEndTreatmentEvent(module=sim.modules['Tb'])

    tx_end_event.apply(sim.population)

    # check individual properties consistent with treatment end
    assert not df.at[person_id, 'tb_on_treatment']
    assert not df.at[person_id, 'tb_treated_mdr']
    assert df.at[person_id, 'tb_inf'] == 'latent'
    assert df.at[person_id, 'tb_strain'] == 'none'
    assert not df.at[person_id, 'tb_smear']


def test_treatment_failure():
    """
    test treatment failure occurs and
    retreatment proerties / follow-up occurs correctly
    """

    popsize = 10

    # allow HS to run and queue events
    sim = get_sim(use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # change prob treatment success - all treatment will fail
    sim.modules['Tb'].parameters['prob_tx_success_new'] = 0.0

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    person_id = 0

    # assign person_id active tb
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'age_exact_years'] = 20
    df.at[person_id, 'age_years'] = 20
    # change district so facilities available
    df.at[person_id, 'district_of_residence'] = 'Lilongwe'

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

    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_ever_tested']
    assert df.at[person_id, 'tb_diagnosed']
    assert not df.at[person_id, 'tb_diagnosed_mdr']

    # schedule treatment start
    # this calls clinical_monitoring which should schedule all follow-up appts
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # end treatment, change sim.date so person will be ready to stop treatment
    sim.date = Date(2010, 12, 31)

    # make treatment fail
    tx_end_event = tb.TbEndTreatmentEvent(module=sim.modules['Tb'])

    tx_end_event.apply(sim.population)

    # check individual properties consistent with treatment failure
    assert df.at[person_id, 'tb_treatment_failure']
    assert df.at[person_id, 'tb_ever_treated']

    # HSI_Tb_ScreeningAndRefer should be scheduled
    # check referral for screening/testing again
    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    test = screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # should schedule xpert - if available
    # check that the event returned a footprint for an xpert test
    assert test == screening_appt.make_appt_footprint({'Over5OPD': 1, 'LabMolec': 1})

    # start treatment
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # clinical monitoring
    # default clinical monitoring schedule for first infection ds-tb and retreatment
    follow_up_times = sim.modules['Tb'].parameters['followup_times']
    clinical_fup = follow_up_times['ds_clinical_monitor'].dropna()
    clinical_fup_retx = follow_up_times['ds_retreatment_clinical'].dropna()
    number_fup_appts = len(clinical_fup) + len(clinical_fup_retx)

    # count how many events are tb.HSI_Tb_FollowUp
    count = 0
    all_future_events = sim.modules['HealthSystem'].find_events_for_person(person_id)
    for idx in range(len(all_future_events)):
        if isinstance(all_future_events[idx][1], tb.HSI_Tb_FollowUp):
            count += 1

    assert count == number_fup_appts
    # check final scheduled event is at least 8 months after sim.date (might be 8 months + 1day)
    last_event_date, last_event_obj = all_future_events[-1]
    assert last_event_date >= (sim.date + pd.DateOffset(months=8))





# def test_children_referrals():
#     """
#     check referrals for children
#     should be x-ray at screening/testing
#     """
#
#
def test_latent_prevalence():
    """
    test overall proportion of new latent cases which progress to active
    should be 14% fast progressors, 67% hiv+ fast progressors
    overall lifetime risk 5-10%
    """

    #set up population
    popsize = 100

    # allow HS to run and queue events
    sim = get_sim(use_simplified_birth=True, disable_HS=True, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # assign latent status to all population
    # all HIV- adults
    df.loc[df.is_alive, 'tb_inf'] = 'latent'
    df.loc[df.is_alive, 'tb_strain'] = 'ds'
    df.loc[df.is_alive, 'tb_date_latent'] = sim.date
    df.loc[df.is_alive, 'hv_inf'] = False
    df.loc[df.is_alive, 'age_years'] = 25

    sim.modules['Tb'].progression_to_active(sim.population)
    # get all events
    test = sim.event_queue.queue
    # count how many scheduled to progress to active disease
    count = 0
    for tmp in range(len(test)):
        if isinstance(test[tmp][2], tb.TbActiveEvent):
            count += 1

    # check proportion HIV- adults who have active scheduled
    assert (count / len(df.loc[df.is_alive])) < 0.25

    # how many are progressing fast  (~14% fast)
    count2 = 0
    for tmp in range(len(test)):
        if test[tmp][0] == sim.date:
            count2 += 1

    prop_fast = count2 / len(df.loc[df.is_alive])
    assert 0.1 <= prop_fast <= 0.3

    # ------------------ HIV-positive progression risk ---------------- #
    sim = get_sim(use_simplified_birth=True, disable_HS=True, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props

    # assign latent status to all population
    # all HIV+ adults
    df.loc[df.is_alive, 'tb_inf'] = 'latent'
    df.loc[df.is_alive, 'tb_strain'] = 'ds'
    df.loc[df.is_alive, 'tb_date_latent'] = sim.date
    df.loc[df.is_alive, 'hv_inf'] = True
    df.loc[df.is_alive, 'age_years'] = 25

    sim.modules['Tb'].progression_to_active(sim.population)
    # get all events
    test = sim.event_queue.queue
    # count how many scheduled to progress to active disease
    count = 0
    for tmp in range(len(test)):
        if isinstance(test[tmp][2], tb.TbActiveEvent):
            count += 1

    # check proportion HIV+ adults who have active scheduled
    assert (count / len(df.loc[df.is_alive])) > 0.5

    # how many are progressing fast  (~67% fast)
    count2 = 0
    for tmp in range(len(test)):
        if test[tmp][0] == sim.date:
            count2 += 1

    prop_fast = count2 / len(df.loc[df.is_alive])
    assert 0.5 <= prop_fast <= 1.0
#
#
# def test_relapse_risk():
#     """
#     check risk of relapse
#     """
#
#
# def test_run_without_hiv():
#     """
#     test running without hiv
#     check smear positive rates in hiv- and hiv+ to confirm process still working with dummy property
#     """
#
#
# def test_ipt_to_child_of_tb_mother():
#     """
#     if child born to mother with diagnosed tb, check give ipt
#     """
#
#
# def test_ipt():
#     """
#     check ipt administered and ended correctly
#     in HIV+ and paediatric contacts of cases
#       also the effect of ipt? i.e. if perfect - no disease;
#       if 0% efficacy and high risk of disease, then everyone gets disease?
#     """
