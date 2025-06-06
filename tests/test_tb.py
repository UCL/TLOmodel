""" Tests for the TB Module """

import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
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


def get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True):
    """
    get sim with the checks for configuration of properties running in the TB module
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    if use_simplified_birth:
        sim.register(demography.Demography(),
                     simplified_births.SimplifiedBirths(),
                     enhanced_lifestyle.Lifestyle(),
                     healthsystem.HealthSystem(disable=disable_HS,cons_availability="all",
                         # mode for consumable constraints (if ignored, all consumables available)
                     ),
                     healthburden.HealthBurden(),
                     symptommanager.SymptomManager(),
                     healthseekingbehaviour.HealthSeekingBehaviour(),
                     epi.Epi(),
                     hiv.Hiv(run_with_checks=False),
                     tb.Tb(),
                     )
    else:
        sim.register(demography.Demography(),
                     pregnancy_supervisor.PregnancySupervisor(),
                     care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
                     labour.Labour(),
                     newborn_outcomes.NewbornOutcomes(),
                     postnatal_supervisor.PostnatalSupervisor(),
                     enhanced_lifestyle.Lifestyle(),
                     healthsystem.HealthSystem(disable=True, cons_availability="all",
                         # mode for consumable constraints (if ignored, all consumables available)
                     ),
                     healthburden.HealthBurden(),
                     symptommanager.SymptomManager(),
                     healthseekingbehaviour.HealthSeekingBehaviour(),
                     epi.Epi(),
                     hiv.Hiv(run_with_checks=False),
                     tb.Tb(),
                     )

    return sim


# simple checks
def test_basic_run(seed):
    """ test basic run and properties assigned correctly
    """

    end_date = Date(2012, 12, 31)
    popsize = 1000

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    df = sim.population.props

    # check properties assigned correctly for baseline population
    # should be no latent infections, no active infections
    num_latent = len(df[(df.tb_inf == 'latent') & df.is_alive])
    assert num_latent == 0

    assert not pd.isnull(df.loc[~df.date_of_birth.isna(), [
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
def test_natural_history(seed):
    """
    test natural history and progression
    need to have disable_HS=False to ensure events enter queue
    otherwise they do run, but without entering queue
    find_events_for_person() checks the event queue
    disable=true, runs all hsi events but doesn't use queue so you won't find them
    """

    popsize = 1000

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # todo change active testing rate
    # set very high incidence rates for poll
    sim.modules['Tb'].parameters['scaling_factor_WHO'] = 50
    sim.modules["Tb"].parameters["rate_testing_active_tb"]["treatment_coverage"] = 100
    sim.modules['Tb'].parameters['prop_smear_positive'] = 1.0
    sim.modules['Tb'].parameters['prop_smear_positive_hiv'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # run TB polling event to schedule progression to active stage
    progression_event = tb.TbActiveCasePoll(module=sim.modules['Tb'])
    progression_event.apply(population=sim.population)

    # some should have TbActiveEvent date scheduled
    assert not pd.isnull(df.loc[~df.date_of_birth.isna() & df.is_alive, [
        'tb_scheduled_date_active']]).all().all()

    # select one person with scheduled active date
    tb_case = df.loc[df.is_alive & ~pd.isnull(df.tb_scheduled_date_active)].index[0]

    # change scheduled date active to now
    df.loc[tb_case, "tb_scheduled_date_active"] = sim.date

    # run TbActiveEvent
    active_event_run = tb.TbActiveEvent(module=sim.modules['Tb'])
    active_event_run.apply(population=sim.population)

    # check properties set
    assert df.at[tb_case, 'tb_inf'] == 'active'
    assert df.at[tb_case, 'tb_date_active'] == sim.date
    assert df.at[tb_case, 'tb_smear']

    # check for TB-related symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    assert symptom_list.issubset(sim.modules['SymptomManager'].has_what(tb_case))

    # Check person_id has a ScreeningAndRefer event scheduled by TbActiveEvent
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(tb_case) if
        isinstance(ev[1], tb.HSI_Tb_ScreeningAndRefer)
    ][0]
    assert date_event > sim.date

    # test and treat this person
    list_of_hsi = [
        'tb.HSI_Tb_ScreeningAndRefer',
        'tb.HSI_Tb_StartTreatment'
    ]

    for name_of_hsi in list_of_hsi:
        hsi_event = eval(name_of_hsi +
                         "(person_id=" +
                         str(tb_case) +
                         ", "
                         "module=sim.modules['Tb'],"
                         ""
                         ")"
                         )
        hsi_event.run(squeeze_factor=0)

    assert pd.notnull(df.at[tb_case, 'tb_date_tested'])
    assert df.at[tb_case, 'tb_diagnosed']
    assert df.at[tb_case, "tb_on_treatment"]
    assert df.at[tb_case, "tb_date_treated"] == sim.date


def test_treatment_schedule(seed):
    """ test treatment schedules
    check dates of follow-up appts following schedule
    check treatment ends at appropriate time
    """

    popsize = 10

    # disable HS, all HSI events will run, but won't be in the HSI queue
    # they will enter the sim.event_queue
    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

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
    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert pd.notnull(df.at[person_id, 'tb_date_tested'])
    assert df.at[person_id, 'tb_diagnosed']
    assert not df.at[person_id, 'tb_diagnosed_mdr']

    # schedule treatment start
    # this schedules HSI_Tb_FollowUp
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # check follow-up event is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_FollowUp)
    ][0]
    assert date_event > sim.date

    # end treatment, change sim.date so person will be ready to stop treatment
    sim.date = Date(2010, 12, 31)
    sim.modules['Tb'].end_treatment(sim.population)

    # check individual properties consistent with treatment end
    assert not df.at[person_id, 'tb_on_treatment']
    assert not df.at[person_id, 'tb_treated_mdr']
    assert df.at[person_id, 'tb_strain'] == 'ds'  # should not have changed


def test_record_of_appt_of_tb_start_treatment_hsi(tmpdir, seed):
    """
    This is to test the appointment footprint recorded with the trigger of HSI_Tb_StartTreatment:
    if consumables are available, the HSI is scheduled only once and the footprint should be TBNew;
    if consumables are not available, the HSI is scheduled repeatedly where the first footprint is TBNew
    and the rest should be PharmDispensing.
    """

    def get_sim_for_appt_test_only(tmpdir, seed, use_simplified_birth=True, disable_HS=False,
                                   ignore_con_constraints=True, consumables_availability='all'):
        """
        get sim with the checks for configuration of properties running in the TB module
        """

        start_date = Date(2010, 1, 1)

        # configurate the log
        log_config = {
            'filename': 'temp',
            'directory': tmpdir,
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem": logging.DEBUG,
            }
        }

        sim = Simulation(start_date=start_date, log_config=log_config, seed=seed, resourcefilepath=resourcefilepath)

        # Register the appropriate modules
        if use_simplified_birth:
            sim.register(demography.Demography(),
                         simplified_births.SimplifiedBirths(),
                         enhanced_lifestyle.Lifestyle(),
                         healthsystem.HealthSystem(disable=disable_HS,cons_availability=consumables_availability,
                             # mode for consumable constraints (if ignored, all consumables available)
                         ),
                         healthburden.HealthBurden(),
                         symptommanager.SymptomManager(),
                         healthseekingbehaviour.HealthSeekingBehaviour(),
                         epi.Epi(),
                         hiv.Hiv(run_with_checks=False),
                         tb.Tb(),
                         )
        else:
            sim.register(demography.Demography(),
                         pregnancy_supervisor.PregnancySupervisor(),
                         care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
                         labour.Labour(),
                         newborn_outcomes.NewbornOutcomes(),
                         postnatal_supervisor.PostnatalSupervisor(),
                         enhanced_lifestyle.Lifestyle(),
                         healthsystem.HealthSystem(disable=True, cons_availability=consumables_availability,
                             # mode for consumable constraints (if ignored, all consumables available)
                         ),
                         healthburden.HealthBurden(),
                         symptommanager.SymptomManager(),
                         healthseekingbehaviour.HealthSeekingBehaviour(),
                         epi.Epi(),
                         hiv.Hiv(run_with_checks=False),
                         tb.Tb(),
                         )

        return sim

    def get_appt_footprints(_consumables_availability):
        """
        Return a list of the APPT_FOOTPRINTS that are logged for one person
        following the trigger of HSI_Tb_StartTreatment.
        """
        popsize = 1

        # disable HS, all HSI events will run, but won't be in the HSI queue
        # they will enter the sim.event_queue
        sim = get_sim_for_appt_test_only(tmpdir, seed, use_simplified_birth=True, disable_HS=False,
                                         ignore_con_constraints=False,
                                         consumables_availability=_consumables_availability)

        # Make the population
        sim.make_initial_population(n=popsize)

        df = sim.population.props
        person_id = 0

        # assign person_id active tb and diagnosed, not on treatment, etc.
        df.at[person_id, 'tb_inf'] = 'active'
        df.at[person_id, 'tb_strain'] = 'ds'
        df.at[person_id, 'tb_date_active'] = sim.date
        df.at[person_id, 'tb_smear'] = True
        df.at[person_id, 'age_exact_years'] = 20
        df.at[person_id, 'age_years'] = 20
        df.at[person_id, 'tb_diagnosed'] = True
        df.at[person_id, 'tb_on_treatment'] = False
        df.at[person_id, 'tb_diagnosed_mdr'] = False
        df.at[person_id, 'is_alive'] = True

        # schedule treatment start
        from tlo.methods.tb import HSI_Tb_StartTreatment
        hsi_event = HSI_Tb_StartTreatment(
            module=sim.modules['Tb'],
            person_id=person_id
        )
        sim.modules['HealthSystem'].schedule_hsi_event(hsi_event=hsi_event, topen=sim.start_date, priority=0.0)

        # let the simulation run 2 months so that the HSI could be rescheduled.
        # the maximum reschedule number is 5, requiring a period of 4 weeks (NB. the first run
        # date is the sim.start_date)
        sim.simulate(end_date=sim.start_date + pd.DateOffset(months=2))

        # find the appt footprint list
        hsi_run = parse_log_file(sim.log_filepath, level=logging.DEBUG)["tlo.methods.healthsystem"]["HSI_Event"]
        return hsi_run.loc[
            hsi_run.did_run
            & (hsi_run['Person_ID'] == person_id)
            & (hsi_run['TREATMENT_ID'] == 'Tb_Treatment'), 'Number_By_Appt_Type_Code'
        ].to_list()

    # 1) If consumables available, the HSI will only be run once and the appt footprint should be TBNew:
    assert [{'TBNew': 1}] == get_appt_footprints(_consumables_availability='all')
    # 2) If consumables not available, there should be multiple footprints where the first is TBNew
    # and the rest is PharmDispensing
    appt_list = get_appt_footprints(_consumables_availability='none')
    assert len(appt_list) > 1
    assert appt_list[0] == {'TBNew': 1}
    assert len(appt_list) - 1 == len([_x for _i, _x in enumerate(appt_list) if _x == {'PharmDispensing': 1}])


def test_treatment_failure(seed):
    """
    test treatment failure occurs and properties set correctly
    treatment failure will schedule referral for xpert test at level 2
    """

    popsize = 10

    # allow HS to run and queue events
    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # change prob treatment success - all treatment will fail
    sim.modules['Tb'].parameters['prob_tx_success_ds'] = 0.0
    sim.modules['Tb'].parameters["prop_presumptive_mdr_has_xpert"] = 1.0

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
    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert pd.notnull(df.at[person_id, 'tb_date_tested'])
    assert df.at[person_id, 'tb_diagnosed']
    assert not df.at[person_id, 'tb_diagnosed_mdr']

    # schedule treatment start
    # this calls clinical_monitoring which should schedule all follow-up appts
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_on_treatment']

    # end treatment, change sim.date so person will be ready to stop treatment
    sim.date = Date(2010, 12, 31)

    # make treatment fail
    sim.modules['Tb'].end_treatment(sim.population)

    # check individual properties consistent with treatment failure
    assert df.at[person_id, 'tb_treatment_failure']
    assert df.at[person_id, 'tb_ever_treated']

    # HSI_Tb_ScreeningAndRefer should be scheduled
    # check referral for screening/testing again
    # screen and test person_id
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'],
                                                 facility_level="1a")
    test = screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # should schedule a referral for xpert testing at facility level 2
    # check that the event returned a footprint Over5OPD
    assert test == screening_appt.make_appt_footprint({'Over5OPD': 1})

    # check tb.HSI_Tb_ScreeningAndRefer scheduled
    followup_test = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_ScreeningAndRefer)
    ][-1]

    assert followup_test[0] > sim.date

    # schedule follow-up test at level 2 to get xpert
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'],
                                                 facility_level="2")
    test = screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # assert now should be diagnosed as active TB again
    assert df.at[person_id, 'tb_diagnosed']


def test_children_referrals(seed):
    """
    check referrals for children
    should be x-ray at screening/testing
    """
    popsize = 10

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # make clinical diagnosis perfect
    sim.modules['Tb'].parameters["sens_clinical"] = 1.0
    sim.modules['Tb'].parameters["spec_clinical"] = 1.0
    sim.modules['Tb'].parameters["sens_xray_smear_negative"] = 1.0
    sim.modules['Tb'].parameters["sens_xray_smear_positive"] = 1.0
    sim.modules['Tb'].parameters["spec_xray_smear_negative"] = 1.0
    sim.modules['Tb'].parameters["spec_xray_smear_positive"] = 1.0
    sim.modules['Tb'].parameters["probability_access_to_xray"] = 1.0

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    person_id = 0

    # set tb strain to ds
    df.at[person_id, 'age_years'] = 2
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'hv_inf'] = False

    # give the symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}

    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    assert set(sim.modules['SymptomManager'].has_what(person_id=person_id)) == symptom_list

    # run HSI_Tb_ScreeningAndRefer and check outcomes
    sim.modules['HealthSystem'].schedule_hsi_event(
        tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb']),
        topen=sim.date,
        tclose=None,
        priority=0
    )

    hsi_event = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb'])
    hsi_event.run(squeeze_factor=0)

    # Check person_id has a HSI_Tb_Xray event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_Xray_level1b)
    ][0]
    assert date_event == sim.date

    # run HSI_Tb_Xray and check outcomes
    sim.modules['HealthSystem'].schedule_hsi_event(
        tb.HSI_Tb_Xray_level1b(person_id=person_id, module=sim.modules['Tb']),
        topen=sim.date,
        tclose=None,
        priority=0
    )

    hsi_event = tb.HSI_Tb_Xray_level1b(person_id=person_id, module=sim.modules['Tb'])
    hsi_event.run(squeeze_factor=0)

    # should be diagnosed by x-ray
    assert df.at[person_id, 'tb_diagnosed']


def test_relapse_risk(seed):
    """
    check risk of relapse
    """

    # apply linear model of relapse risk
    # set properties to ensure high risk
    # set up population
    popsize = 10

    # allow HS to run and queue events
    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)
    sim.modules['Tb'].parameters['monthly_prob_relapse_tx_incomplete'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    person_id = 0

    # risk of relapse <2 years after active onset
    df.at[person_id, 'tb_inf'] = 'latent'
    df.at[person_id, 'tb_ever_treated'] = True
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_date_treated'] = sim.date + pd.DateOffset(days=30)
    df.at[person_id, 'tb_treatment_failure'] = True
    df.at[person_id, 'hv_inf'] = False
    df.at[person_id, 'age_years'] = 25

    # run relapse event
    sim.modules['Tb'].relapse_event(sim.population)

    # check relapse to active tb is scheduled to occur
    assert pd.notnull(df.at[person_id, 'tb_scheduled_date_active'])


def test_ipt_to_child_of_tb_mother(seed):
    """
    if child born to mother with diagnosed tb, check give ipt
    """
    popsize = 10

    # allow HS to run and queue events
    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # allow IPT to be given on_birth prior to 2014
    sim.modules['Tb'].parameters['ipt_start_date'] = 2010

    # make IPT protection against active disease perfect
    sim.modules['Tb'].parameters['rr_ipt_child'] = 0.0

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mother_id = 0
    child_id = 1

    # give mother active tb
    df.at[mother_id, 'tb_inf'] = 'active'
    df.at[mother_id, 'tb_strain'] = 'ds'
    df.at[mother_id, 'hv_inf'] = False
    df.at[mother_id, 'age_years'] = 25
    df.at[mother_id, 'tb_diagnosed'] = True

    # check HSI_Tb_Start_or_Continue_Ipt scheduled for child
    sim.modules['Tb'].on_birth(mother_id=mother_id, child_id=child_id)

    # check HSI IPT event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
        isinstance(ev[1], tb.HSI_Tb_Start_or_Continue_Ipt)
    ][0]
    assert date_event == sim.date

    # give IPT to child
    ipt_appt = tb.HSI_Tb_Start_or_Continue_Ipt(person_id=child_id,
                                               module=sim.modules['Tb'])
    ipt_appt.apply(person_id=child_id, squeeze_factor=0.0)

    assert df.at[child_id, 'tb_on_ipt']
    assert df.at[child_id, 'tb_date_ipt'] == sim.date

    # give child latent tb, ipt should prevent progression to active
    df.at[child_id, 'tb_inf'] = 'latent'
    active_event_run = tb.TbActiveCasePoll(module=sim.modules['Tb'])
    active_event_run.apply(sim.population)
    assert not df.at[child_id, 'tb_scheduled_date_active'] == pd.NaT

    # child should have Tb_DecisionToContinueIPT scheduled
    date_event, event = [
        ev for ev in sim.find_events_for_person(child_id) if
        isinstance(ev[1], tb.Tb_DecisionToContinueIPT)
    ][0]
    assert date_event == sim.date + pd.DateOffset(months=6)


def test_mdr(seed):
    """
    mdr infection to first-line treatment
    failure on first line
    MDR to treatment
    death
    checking that death is logged in the right way (two permutations with and without HIV)
    """

    popsize = 10

    # disable HS, all HSI events will run, but won't be in the HSI queue
    # they will enter the sim.event_queue
    sim = get_sim(seed, use_simplified_birth=True, disable_HS=True, ignore_con_constraints=True)

    # change sensitivity of xpert test to ensure mdr diagnosis on treatment failure
    sim.modules['Tb'].parameters['sens_xpert'] = 1.0
    sim.modules['Tb'].parameters["prop_presumptive_mdr_has_xpert"] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    person_id = 0

    # assign person_id active tb
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_strain'] = 'mdr'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'age_exact_years'] = 20
    df.at[person_id, 'age_years'] = 20
    df.at[person_id, 'hv_inf'] = False

    # assign symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    # screen and test person_id
    # no previous infection and HIV-negative so will get sputum smear test
    # should be incorrect diagnosis of ds-tb
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'])
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_date_tested'] != pd.NaT
    assert df.at[person_id, 'tb_diagnosed']
    assert not df.at[person_id, 'tb_diagnosed_mdr']

    # schedule treatment start
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'])
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # end treatment, change sim.date so person will be ready to stop treatment
    sim.date = Date(2010, 12, 31)
    sim.modules['Tb'].end_treatment(sim.population)

    # check individual properties consistent with treatment failure
    assert df.at[person_id, 'tb_treatment_failure']
    assert df.at[person_id, 'tb_ever_treated']

    # next screening should pick up case as retreatment / mdr
    screening_appt = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                                 module=sim.modules['Tb'],
                                                 facility_level='2')
    screening_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, 'tb_diagnosed_mdr']

    # schedule mdr treatment start
    # this calls clinical_monitoring which should schedule all follow-up appts
    tx_start = tb.HSI_Tb_StartTreatment(person_id=person_id,
                                        module=sim.modules['Tb'],
                                        facility_level='2')
    tx_start.apply(person_id=person_id, squeeze_factor=0.0)

    # check treatment appropriate for mdr-tb
    assert df.at[person_id, 'tb_treated_mdr']


def test_cause_of_death(seed):
    """
    schedule death for people with tb and tb-hiv
    check causes of death are assigned correctly
    """

    popsize = 10

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # set mortality rates to 1
    sim.modules['Tb'].parameters['death_rate_smear_pos_untreated'] = 1.0

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # create person with tb, no HIV
    person_id0 = 0
    df.at[person_id0, 'tb_inf'] = 'active'
    df.at[person_id0, 'tb_strain'] = 'ds'
    df.at[person_id0, 'tb_date_active'] = sim.date
    df.at[person_id0, 'tb_smear'] = True
    df.at[person_id0, 'hv_inf'] = False

    # schedule death through TbDeathEvent
    result = sim.modules['Tb'].lm['death_rate'].predict(df.loc[[person_id0]], rng=sim.rng)

    if result:
        sim.modules['Demography'].do_death(
            individual_id=person_id0, cause='TB',
            originating_module=sim.modules['Tb'])

    assert not df.at[person_id0, 'is_alive']
    assert df.at[person_id0, 'cause_of_death'] == "TB"

    # create person with tb-hiv
    person_id1 = 1
    df.at[person_id1, 'tb_inf'] = 'latent'
    df.at[person_id1, 'tb_strain'] = 'ds'
    df.at[person_id1, 'tb_scheduled_date_active'] = sim.date
    # df.at[person_id1, 'tb_smear'] = True
    df.at[person_id1, 'hv_inf'] = True

    # check AIDS onset scheduled through TbActiveEvent
    active_event_run = tb.TbActiveEvent(module=sim.modules['Tb'])
    active_event_run.apply(sim.population)

    # check properties set
    assert df.at[person_id1, 'tb_inf'] == 'active'
    assert df.at[person_id1, 'tb_date_active'] == sim.date

    # find the AIDS onset event for this person
    date_aids_event, aids_event = \
        [ev for ev in sim.find_events_for_person(person_id1) if isinstance(ev[1], hiv.HivAidsOnsetEvent)][0]
    assert date_aids_event == sim.date

    # run the AIDS onset event for this person - this will schedule AIDS death:
    aids_event.apply(person_id=person_id1)
    assert "aids_symptoms" in sim.modules['SymptomManager'].has_what(person_id1)

    # schedule AIDS death - cause AIDS_TB
    date_aids_death_event, aids_death_event = \
        [ev for ev in sim.find_events_for_person(person_id1) if isinstance(ev[1], hiv.HivAidsTbDeathEvent)][0]
    assert date_aids_death_event > sim.date

    # run the AIDS death event for this person:
    aids_death_event.apply(person_id1)

    # confirm the person is dead
    assert False is bool(df.at[person_id1, "is_alive"])
    assert sim.date == df.at[person_id1, "date_of_death"]
    assert "AIDS_TB" == df.at[person_id1, "cause_of_death"]


def test_active_tb_linear_model(seed):
    """
    check the weighting for active tb applied to sub-groups
    using linear model
    """

    popsize = 10

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    tb_module = sim.modules['Tb']

    # set parameters
    tb_module.parameters['scaling_factor_WHO'] = 5

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # set properties - no risk factors
    df.loc[df.is_alive, 'tb_inf'] = 'uninfected'
    df.loc[df.is_alive, 'age_years'] = 25
    df.loc[df.is_alive, 'va_bcg_all_doses'] = True

    df.loc[df.is_alive, 'li_bmi'] = 1  # 4=obese
    df.loc[df.is_alive, 'li_ex_alc'] = False
    df.loc[df.is_alive, 'li_tob'] = False

    df.loc[df.is_alive, 'tb_on_ipt'] = False

    df.loc[df.is_alive, 'hv_inf'] = False
    df.loc[df.is_alive, 'sy_aids_symptoms'] = 0
    df.loc[df.is_alive, 'hv_art'] = "not"

    # no risk factors
    person_id0 = 0
    rr_base = tb_module.lm["active_tb"].predict(df.loc[[person_id0]])
    assert rr_base.values[0] == 1.0

    # hiv+, no tx
    person_id1 = 1
    df.at[person_id1, 'hv_inf'] = True
    df.at[person_id1, 'sy_aids_symptoms'] = 0
    df.at[person_id1, 'hv_art'] = "not"
    rr_hiv = tb_module.lm["active_tb"].predict(df.loc[[person_id1]])
    assert rr_hiv.values[0] > rr_base.values[0]

    # hiv+, ART and virally suppressed
    person_id2 = 2
    df.at[person_id2, 'hv_inf'] = True
    df.at[person_id2, 'sy_aids_symptoms'] = 0
    df.at[person_id2, 'hv_art'] = "on_VL_suppressed"
    rr_hiv_art = tb_module.lm["active_tb"].predict(df.loc[[person_id2]])
    assert rr_hiv_art.values[0] < rr_hiv.values[0]  # protective effect vs untreated hiv

    # hiv+, ART and virally suppressed, on IPT
    person_id3 = 3
    df.at[person_id3, 'hv_inf'] = True
    df.at[person_id3, 'sy_aids_symptoms'] = 0
    df.at[person_id3, 'hv_art'] = "on_VL_suppressed"
    df.at[person_id3, 'tb_on_ipt'] = True
    rr_hiv_art_ipt = tb_module.lm["active_tb"].predict(df.loc[[person_id3]])
    assert rr_hiv_art_ipt.values[0] < rr_hiv.values[0]  # protective effect vs untreated hiv


@pytest.mark.slow
def test_basic_run_with_default_parameters(seed):
    """Run the TB module with check and check dtypes consistency"""
    end_date = Date(2010, 6, 30)

    sim = get_sim(seed=seed)
    sim.make_initial_population(n=1000)

    check_dtypes(sim)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)
    # confirm configuration of properties at the end of the simulation:
    sim.modules['Tb'].check_config_of_properties()


@pytest.mark.slow
def test_use_dummy_version(seed):
    """check that the dummy version of the TB module works with the dummy HIV version
    """
    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(),
                 epi.Epi(),
                 hiv.DummyHivModule(hiv_prev=1.0),
                 tb.DummyTbModule(active_tb_prev=0.01),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=Date(2014, 12, 31))

    check_dtypes(sim)


def test_hsi_scheduling(seed):
    """
    check HSI_Tb_ScreeningAndRefer schedules the correct events for children / adults / adults with HIV

    children should have an xray and hiv test scheduled
    adults should have treatment and hiv test scheduled
    adults already diagnosed with hiv should not have further hiv test scheduled

    assert multiple tests not being scheduled accidentally in each HSI_Tb_ScreeningAndRefer call

    """
    popsize = 10

    sim = get_sim(seed, use_simplified_birth=True, disable_HS=False, ignore_con_constraints=True)

    # Make the population
    sim.make_initial_population(n=popsize)

    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    person_id = 0

    # child under 5yrs
    df.at[person_id, 'age_years'] = 2
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'hv_inf'] = False

    # give the symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}

    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    assert set(sim.modules['SymptomManager'].has_what(person_id=person_id)) == symptom_list

    hsi_event = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb'])
    hsi_event.run(squeeze_factor=0)

    # Check person_id has a HSI_Tb_Xray event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_Xray_level1b)
    ][0]
    assert date_event == sim.date

    # check HIV test scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], hiv.HSI_Hiv_TestAndRefer)
    ][0]
    assert date_event == sim.date

    # check these are the only two events scheduled
    tmp = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert len(tmp) == 2

    # repeat checks for person over 5 years
    person_id = 1

    df.at[person_id, 'age_years'] = 25
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_smear'] = True
    df.at[person_id, 'hv_inf'] = False

    # give the symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}

    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    assert set(sim.modules['SymptomManager'].has_what(person_id=person_id)) == symptom_list

    hsi_event = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb'])
    hsi_event.run(squeeze_factor=0)

    # Check person_id has a treatment event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_StartTreatment)
    ][0]
    assert date_event == sim.date

    # check HIV test scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], hiv.HSI_Hiv_TestAndRefer)
    ][0]
    assert date_event == sim.date

    # check these are the only two events scheduled
    tmp = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert len(tmp) == 2

    # repeat checks for person over 5 years, HIV+, smear-ve
    person_id = 2

    df.at[person_id, 'age_years'] = 25
    df.at[person_id, 'tb_inf'] = 'active'
    df.at[person_id, 'tb_date_active'] = sim.date
    df.at[person_id, 'tb_strain'] = 'ds'
    df.at[person_id, 'tb_smear'] = False
    df.at[person_id, 'hv_inf'] = True
    df.at[person_id, 'hv_diagnosed'] = True

    # give the symptoms
    symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}

    sim.modules["SymptomManager"].change_symptom(
        person_id=person_id,
        symptom_string=symptom_list,
        add_or_remove="+",
        disease_module=sim.modules['Tb'],
        duration_in_days=None,
    )

    assert set(sim.modules['SymptomManager'].has_what(person_id=person_id)) == symptom_list

    hsi_event = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id, module=sim.modules['Tb'])
    hsi_event.run(squeeze_factor=0)

    # person is HIV+ and will be referred for xpert - only available at level 2
    # Check person_id has a further testing event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_ScreeningAndRefer)
    ][0]
    assert date_event >= sim.date

    # run testing event for xpert
    hsi_event = tb.HSI_Tb_ScreeningAndRefer(person_id=person_id,
                                            module=sim.modules['Tb'],
                                            facility_level='2')
    hsi_event.run(squeeze_factor=0)

    # person should now have treatment event scheduled
    # Check person_id has a treatment event scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], tb.HSI_Tb_StartTreatment)
    ][0]
    assert date_event == sim.date
