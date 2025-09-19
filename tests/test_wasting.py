"""
 Basic tests for the Wasting Module
 """
import os
from pathlib import Path

import pandas as pd
import pytest
from pandas import DateOffset

from tlo import Date, Module, Simulation, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    Metadata,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hsi_generic_first_appts,
    simplified_births,
    symptommanager,
    wasting,
)
from tlo.methods.healthseekingbehaviour import HealthSeekingBehaviourPoll
from tlo.methods.wasting import (
    HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM,
    HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
    Wasting_FullRecovery_Event,
    Wasting_IncidencePoll,
    Wasting_ProgressionToSevere_Event,
    Wasting_RecoveryToMAM_Event,
    Wasting_SevereAcuteMalnutritionDeath_Event,
)

# Path to the resource files used by the disease and intervention methods
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

# Default date for the start of simulations
start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)


def get_sim(tmpdir):
    """
    Return simulation objection with Wasting and other necessary
    modules registered.
    """
    sim = Simulation(start_date=start_date, seed=0,
                     resourcefilepath=resourcefilepath,
                     show_progress_bar=False,
                     log_config={
                         'filename': 'tmp',
                         'directory': tmpdir,
                         'custom_levels': {
                             "*": logging.WARNING,
                             "tlo.methods.wasting": logging.INFO}
                     })

    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(disable=False, cons_availability="all", equip_availability="all"),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(force_any_symptom_to_lead_to_healthcareseeking=True),
        healthburden.HealthBurden(),
        simplified_births.SimplifiedBirths(),
        wasting.Wasting(),
    )
    return sim


@pytest.mark.slow
def test_integrity_of_properties_of_wasting_module(tmpdir):
    """ Run the simulation and do some daily checks on dtypes and properties integrity. """
    class DummyModule(Module):
        """ A Dummy module that ensure wasting properties are as expected on a daily basis """
        METADATA = {Metadata.DISEASE_MODULE}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # schedule check property integrity event
            sim.schedule_event(CheckPropertyIntegrityEvent(self), sim.date)

        def on_birth(self, mother_id, child_id):
            pass

    class CheckPropertyIntegrityEvent(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            """schedule to run every day
            :param module: the module that created this event
            """
            self.repeat_days = 1
            super().__init__(module, frequency=DateOffset(days=self.repeat_days))
            assert isinstance(module, DummyModule)

        def apply(self, population):
            """ Apply this event to the population.
               :param population: the current population
           """
            # check datatypes
            self.check_dtypes(population)

            # check properties are as expected
            self.check_configuration_of_properties(population)

        def check_dtypes(self, population):
            # Check types of columns
            df = population.props
            orig = population.new_row
            assert df.dtypes.eq(orig.dtypes).all()

        def check_configuration_of_properties(self, population):
            """ check wasting properties on a daily basis to ensure integrity """
            df = population.props
            under5_sam = df.index[df.is_alive & (df.age_exact_years < 5) &
                                  (df.un_clinical_acute_malnutrition == 'SAM')]

            # Those that were never wasted, should have normal WHZ score:
            assert (df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna(), 'un_WHZ_category'].eq('WHZ>=-2')).all()

            # Those for whom the death date has past should be dead
            assert not df.loc[(df['un_sam_death_date'] < self.sim.date), 'is_alive'].any()
            # Those who died due to SAM should have SAM
            assert df.loc[(df['un_sam_death_date']) < self.sim.date, 'un_clinical_acute_malnutrition'].eq('SAM').all()

            # Check that those in a current episode have symptoms of wasting
            # [caused by the wasting module] but not others (among those alive)
            has_symptoms_of_wasting = set(self.sim.modules['SymptomManager'].who_has('weight_loss'))

            has_symptoms = set([p for p in has_symptoms_of_wasting if
                                'Wasting' in sim.modules['SymptomManager'].causes_of(p, 'weight_loss')])

            in_current_episode_before_recovery = df.is_alive & df.un_ever_wasted & (df.un_last_wasting_date_of_onset <=
                                                                                    self.sim.date) & (self.sim.date <=
                                                                                                 df.un_am_recovery_date)
            set_of_person_id_in_current_episode_before_recovery = set(in_current_episode_before_recovery[
                                                                          in_current_episode_before_recovery].index)

            in_current_episode_before_death = df.is_alive & df.un_ever_wasted & (df.un_last_wasting_date_of_onset <=
                                                                                 self.sim.date) & (
                                                          self.sim.date <= df.un_sam_death_date)
            set_of_person_id_in_current_episode_before_death = set(in_current_episode_before_death[
                                                                       in_current_episode_before_death].index)

            assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
                set_of_person_id_in_current_episode_before_death)

            # WHZ standard deviation of -3, MUAC < 115mm, and oedema should cause severe acute malnutrition
            whz_index = df.index[df['un_WHZ_category'] == 'WHZ<-3']
            muac_index = df.index[df['un_am_MUAC_category'] == '<115mm']
            oedema_index = df.index[df['un_am_nutritional_oedema']]
            assert df.loc[whz_index, 'un_clinical_acute_malnutrition'].eq("SAM").all()
            assert df.loc[muac_index, 'un_clinical_acute_malnutrition'].eq("SAM").all()
            assert df.loc[oedema_index, 'un_clinical_acute_malnutrition'].eq("SAM").all()

            # all SAM individuals should have symptoms of wasting
            assert set(under5_sam).issubset(has_symptoms)

            # All MAM individuals should have no symptoms of wasting
            assert set(df.index[df.is_alive & (df.age_exact_years < 5) &
                                (df.un_clinical_acute_malnutrition == 'MAM')]) not in has_symptoms

    popsize = 10_000
    sim = get_sim(tmpdir)
    sim.register(DummyModule())
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)


def test_wasting_incidence(tmpdir):
    """ Check incidence of wasting is happening as expected. """
    # get simulation object:
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date) # zero duration

    # Reset properties of all individuals so that they are well-nourished
    df = sim.population.props
    df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not wasted
    df.loc[df.is_alive, 'un_am_MUAC_category'] = '>=125mm'
    df.loc[df.is_alive, 'un_am_nutritional_oedema'] = False
    df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
    df.loc[df.is_alive, 'un_ever_wasted'] = False
    df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT

    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # Run polling event: check that all children should now have moderate wasting
    polling = Wasting_IncidencePoll(wmodule)
    polling.apply(sim.population)

    # Check properties of individuals: should now be moderately wasted with onset in previous month
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    assert all(under5s['un_ever_wasted'])
    assert (under5s['un_WHZ_category'].eq('-3<=WHZ<-2')).all()
    days_in_month = (sim.date - pd.DateOffset(days=1)).days_in_month
    assert (under5s['un_last_wasting_date_of_onset'] < sim.date).all() and \
           (under5s['un_last_wasting_date_of_onset'] >= sim.date - pd.to_timedelta(days_in_month, unit='D')).all()


def test_report_daly_weights(tmpdir):
    """ Check if daly weights reporting is done as expected. Four checks are made:
    1. For an individual who is well (No weight is expected/must be 0.0)
    2. For an individual with moderate wasting and oedema (expected daly weight is 0.051)
    3. For an individual with severe wasting and oedema (expected daly weight is 0.172)
    4. For an individual with severe wasting without oedema (expected daly weight is 0.128) """
    popsize = 1
    sim = get_sim(tmpdir)
    sim.modules['Demography'].parameters['max_age_initial'] = 4.9
    # get wasting module
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date) # zero duration

    # Dict to hold the DALY weights
    daly_wts = dict()

    # Get person to use
    df = sim.population.props
    person_id = df.index[0]
    df.at[person_id, 'is_alive'] = True

    # 1. Check daly weight for well-nourished person (weight is 0.0)
    # Reset diagnostic properties
    df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.at[person_id, 'un_am_nutritional_oedema'] = False
    df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'

    # Verify diagnosis - an individual should be well
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'well'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Verify that individual has no daly weight
    assert daly_weights_reported.at[person_id] == 0.0

    get_daly_weights = sim.modules['HealthBurden'].get_daly_weight

    # 2. Check daly weight for person with moderate wasting and oedema (weight is 0.051)
    # Reset diagnostic properties
    df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
    df.at[person_id, 'un_am_nutritional_oedema'] = True

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get daly weight of moderate wasting with oedema
    daly_wts['mod_wasting_with_oedema'] = get_daly_weights(sequlae_code=461)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['mod_wasting_with_oedema'] == daly_weights_reported.at[person_id]

    # 3. Check daly weight for person with severe wasting and oedema (weight is 0.172)
    # Reset diagnostic properties
    df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.at[person_id, 'un_am_nutritional_oedema'] = True

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get daly weight of severe wasting with oedema
    daly_wts['sev_wasting_with_oedema'] = get_daly_weights(sequlae_code=463)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_with_oedema'] == daly_weights_reported.at[person_id]

    # 4. Check daly weight for person with severe wasting without oedema (weight is 0.128)
    # Reset diagnosis
    df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.at[person_id, 'un_am_nutritional_oedema'] = False

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get day weight of severe wasting without oedema
    daly_wts['sev_wasting_w/o_oedema'] = get_daly_weights(sequlae_code=462)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_w/o_oedema'] == daly_weights_reported.at[person_id]


def test_nat_recovery_moderate_wasting(tmpdir):
    """ Check natural recovery after onset of moderate wasting with MAM diagnosis. """
    for am_state_expected in ['MAM', 'SAM']:
        popsize = 1000
        sim = get_sim(tmpdir)
        # get wasting module
        wmodule = sim.modules['Wasting']
        p = wmodule.parameters

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)  # zero duration
        sim.event_queue.queue = []  # clear the queue

        # Get person to use:
        df = sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        # Reset properties of this individual to be well-nourished
        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not wasted
        df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
        df.at[person_id, 'un_am_nutritional_oedema'] = False
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well' # well-nourished
        df.at[person_id, 'un_ever_wasted'] = False
        df.at[person_id, 'un_last_wasting_date_of_onset'] = pd.NaT

        # Set moderate wasting incidence rate at 100% and rate of progression to severe wasting at 0%.
        # (Hence, all children with normal wasting should get onset of moderate wasting and be scheduled for natural
        # recovery.)
        wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
        wmodule.wasting_models.severe_wasting_progression_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
        if am_state_expected == 'MAM':
            # Set probability of MUAC < 115mm with moderate wasting, and probability of oedema with moderate wasting
            # at 0% in order to have MAM with onset of wasting
            p['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = 0.0
            p['proportion_WHZ<-2_with_oedema'] = 0.0
        else:  # am_state_expected == 'SAM'
            # Set probability of oedema with moderate wasting at 100% in order to have SAM with onset of wasting
            p['proportion_WHZ<-2_with_oedema'] = 1.0

        # Run Wasting Polling event: This event should cause all young children to be moderately wasted
        polling = Wasting_IncidencePoll(module=wmodule)
        polling.apply(sim.population)

        # Check properties of this individual: should now be moderately wasted with MAM or SAM respectively and
        # onset in previous month
        person = df.loc[person_id]
        assert person['un_ever_wasted']
        assert person['un_WHZ_category'] == '-3<=WHZ<-2'
        days_in_month = (sim.date - pd.DateOffset(days=1)).days_in_month
        assert (person['un_last_wasting_date_of_onset'] < sim.date) and \
               (person['un_last_wasting_date_of_onset'] >= sim.date - pd.to_timedelta(days_in_month, unit='D'))
        assert pd.isnull(person['un_am_tx_start_date'])
        assert pd.isnull(person['un_am_recovery_date'])
        assert df.at[person_id, 'un_clinical_acute_malnutrition'] == am_state_expected

        # Check that there is a natural recovery event scheduled:
        #  Wasting_FullRecovery_Event if this person has MAM, Wasting_RecoveryToMAM_Event if this person has SAM
        if am_state_expected == 'MAM':
            recov_event_type = Wasting_FullRecovery_Event
        else:  # am_state_expected == 'SAM'
            recov_event_type = Wasting_RecoveryToMAM_Event
        recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                             if isinstance(event_tuple[1], recov_event_type)][0]
        date_of_scheduled_recov = recov_event_tuple[0]
        recov_event = recov_event_tuple[1]
        assert date_of_scheduled_recov > sim.date

        # Check no progression to severe wasting is scheduled
        progress_event_tuple = next((event_tuple for event_tuple in sim.find_events_for_person(person_id)
                                     if isinstance(event_tuple[1], Wasting_ProgressionToSevere_Event)), None)
        assert not progress_event_tuple

        # Run the natural recovery event:
        sim.date = date_of_scheduled_recov
        recov_event.apply(person_id)

        # Check properties of this individual, if recovered from MAM should be well, if recovered from SAM should be MAM
        person = df.loc[person_id]
        if am_state_expected == 'MAM':  # with moderate wasting
            assert person['un_WHZ_category'] == 'WHZ>=-2'
            assert person['un_am_MUAC_category'] == '>=125mm'
            assert not person['un_am_nutritional_oedema']
            assert person['un_clinical_acute_malnutrition'] == 'well'
            assert not person['un_sam_with_complications']
            assert person['un_am_recovery_date'] == sim.date
            assert pd.isnull(person['un_sam_death_date'])
        else:  # am_state_expected == 'SAM' with moderate wasting
            assert not person['un_am_nutritional_oedema']
            assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM'
            assert not person['un_sam_with_complications']
            assert pd.isnull(person['un_am_recovery_date'])
            assert pd.isnull(person['un_sam_death_date'])


def test_tx_recovery_to_MAM_severe_acute_malnutrition_without_complications(tmpdir):
    """ Assume: untreated SAM causes certain death, no complications occur, incidence and progression to severe wasting
    are certain, and treatment prevents death.
    Check: child progresses to SAM without complications, death is scheduled but cancelled by treatment, and recovery to
    MAM follows treatment. """
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']
    p = wmodule.parameters

    # Set death due to untreated SAM at 100% for all, hence no natural recovery from severe wasting
    p['base_death_rate_untreated_SAM'] = 1.0
    p['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)  # zero duration
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to be well
    df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.at[person_id, 'un_am_nutritional_oedema'] = False
    df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
    df.at[person_id, 'un_sam_with_complications'] = False
    df.at[person_id, 'un_sam_death_date'] = pd.NaT

    # Ensure the individual has no complications when SAM occurs
    p['prob_complications_in_SAM'] = 0.0
    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    # Set progress to severe wasting at 100% as well, hence no natural recovery from moderate wasting
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()
    # Set complete recovery from SAM to zero. We want those with SAM to recover to MAM with treatment
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    # Set probability of death after treatment at 0% (hence recovery to MAM with treatment at 100%)
    p['prob_death_after_SAMcare'] = 0.0

    # Run Wasting Polling event to get new incident cases:
    polling = Wasting_IncidencePoll(module=wmodule)
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    assert df.at[person_id, 'un_WHZ_category'] == '-3<=WHZ<-2'

    # Check that there is a Wasting_ProgressionToSevere_Event scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                               if isinstance(event_tuple[1], Wasting_ProgressionToSevere_Event)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id)

    # Check this individual has symptom (weight loss) caused by Wasting (SAM only)
    assert 'weight_loss' in sim.modules['SymptomManager'].has_what(
        person_id=person_id, disease_module=wmodule
    )

    # Check properties of this individual
    # (should now be severely wasted, diagnosed as SAM
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'

    # Check death is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_SevereAcuteMalnutritionDeath_Event)
    assert not pd.isnull(person['un_sam_death_date'])
    # get date of death and death event
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # Check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)
    # run the created instance of HSI_GenericNonEmergencyFirstAppt and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # Check HSI event is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)
    # run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # Check death was canceled with treatment
    assert pd.isnull(df.at[person_id, 'un_sam_death_date'])

    # Check recovery to MAM due to treatment is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[2][1], Wasting_RecoveryToMAM_Event)
    # get date of recovery to MAM and the recovery event
    sam_recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_RecoveryToMAM_Event)][0]
    date_of_scheduled_recovery_to_mam = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

   # Run death event (death should not happen) & recovery to MAM in correct order
    sim.date = min(date_of_scheduled_death, date_of_scheduled_recovery_to_mam)
    if sim.date == date_of_scheduled_death:
        death_event.apply(person_id)
        sim.date = date_of_scheduled_recovery_to_mam
        sam_recovery_event.apply(person_id)
    else:
        sam_recovery_event.apply(person_id)
        sim.date = date_of_scheduled_death
        death_event.apply(person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['is_alive']
    assert pd.isnull(person['un_sam_death_date'])
    assert person['un_clinical_acute_malnutrition'] == 'MAM'
    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id=person_id, disease_module=wmodule))


def test_tx_full_recovery_severe_acute_malnutrition_with_complications(tmpdir):
    """ Check the onset of symptoms with complicated SAM, check full recovery with treatment when the natural death due
    to SAM is certain but canceled with treatment, and symptoms resolved when fully recovered with treatment. """
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']
    p = wmodule.parameters

    # Set death due to untreated SAM at 100% for all, hence no natural recovery from severe wasting
    p['base_death_rate_untreated_SAM'] = 1.0
    p['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)  # zero duration
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to have SAM
    df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.at[person_id, 'un_last_wasting_date_of_onset'] = sim.date

    # Ensure the individual has complications due to SAM
    p['prob_complications_in_SAM'] = 1.0
    # Set full recovery rate to 100% so that this individual should fully recover with treatment
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative()

    # Assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # Check properties of this individual:
    person = df.loc[person_id]
    # should be diagnosed as SAM due to severe wasting
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    # should have complications
    assert person['un_sam_with_complications']
    # symptoms should be applied
    assert person_id in set(sim.modules['SymptomManager'].who_has('weight_loss'))

    # Check death is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1], Wasting_SevereAcuteMalnutritionDeath_Event)
    assert not pd.isnull(df.at[person_id, 'un_sam_death_date'])
    # get date of death and death event
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run care seeking event and ensure HSI for complicated SAM is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()
    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericNonEmergencyFirstAppt and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)
    # check HSI event for complicated SAM is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # Check scheduled death was canceled due to treatment
    assert pd.isnull(df.at[person_id, 'un_sam_death_date'])

    # Check full recovery due to treatment is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_FullRecovery_Event)
    # get date of full recovery and the recovery event
    sam_recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_FullRecovery_Event)][0]
    date_of_scheduled_full_recovery = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_full_recovery > sim.date

    # Run death event (death should not happen) & full recovery in correct order
    sim.date = min(date_of_scheduled_death, date_of_scheduled_full_recovery)
    if sim.date == date_of_scheduled_death:
        death_event.apply(person_id)
        sim.date = date_of_scheduled_full_recovery
        sam_recovery_event.apply(person_id)
    else:
        sam_recovery_event.apply(person_id)
        sim.date = date_of_scheduled_death
        death_event.apply(person_id)

    # Check properties of this individual. Should now be well and alive
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert person['un_am_MUAC_category'] == '>=125mm'
    assert not person['un_am_nutritional_oedema']
    assert person['un_clinical_acute_malnutrition'] == 'well'
    assert pd.isnull(person['un_sam_death_date'])
    assert person['is_alive']

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id=person_id, disease_module=wmodule))


def test_nat_death_overwritten_by_tx_death(tmpdir):
    """ Check if the risk of death when untreated is 100%, the person is scheduled to die due to natural history. But
     with treatment the natural death is cancelled. Check if also chance to fully recover with treatment is 0%, and
     risk of death when treated is 100%, the person will die. Test for uncomplicated SAM."""
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']
    p = wmodule.parameters

    # Set death due to untreated SAM at 100% for all, hence no natural recovery from severe wasting,
    # hence all SAM cases should die without treatment
    p['base_death_rate_untreated_SAM'] = 1.0
    p['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)  # zero duration
    sim.event_queue.queue = []  # clear the queue

    # Set full recovery with treatment at 0%
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    # Set death rate with treatment at 100%, hence all SAM cases should die with treatment
    p['prob_death_after_SAMcare'] = 1.0
    # Ensure the individual has no complications when SAM occurs
    p['prob_complications_in_SAM'] = 0.0

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to have SAM due to severe wasting, hence natural death should be applied
    df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.at[person_id, 'un_last_wasting_date_of_onset'] = sim.date

    # Assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # Check properties of this individual:
    person = df.loc[person_id]
    # should be diagnosed as SAM due to severe wasting
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    # symptoms should be applied
    assert person_id in set(sim.modules['SymptomManager'].who_has('weight_loss'))
    # natural death should be scheduled
    assert not pd.isnull(person['un_sam_death_date'])

    # Get the natural death date
    nat_death_date = person['un_sam_death_date']

    # Run health seeking behavior two days later
    sim.date = sim.date + DateOffset(weeks=1)
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()
    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericNonEmergencyFirstAppt
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
          if isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)
    # check outpatient care event is scheduled for uncomplicated SAM
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
              if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # Check death event is scheduled for another day than natural death was scheduled for since there is no recovery
    # with treatment
    assert isinstance(sim.find_events_for_person(person_id)[0][1], Wasting_SevereAcuteMalnutritionDeath_Event)
    assert df.at[person_id, 'un_sam_death_date'] != nat_death_date

    # Load list of all death events scheduled for the person
    death_events_list = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                         isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)]

    # Run the first scheduled death event (ie natural death), nothing should happen, person should be still alive:
    nat_death_event_tuple = death_events_list[0]
    assert nat_death_date == nat_death_event_tuple[0]
    nat_death_event = nat_death_event_tuple[1]
    assert nat_death_date > sim.date
    sim.date = nat_death_date
    nat_death_event.apply(person_id)
    # check properties of this individual: (should be still alive, with death scheduled to another day)
    person = df.loc[person_id]
    assert person['un_sam_death_date'] != nat_death_date
    assert person['is_alive']

    # Run the second scheduled death event (ie death with treatment), ensure the person is now dead:
    death_with_tx_event_tuple = death_events_list[1]
    death_with_tx_date = death_with_tx_event_tuple[0]
    death_with_tx_event = death_with_tx_event_tuple[1]
    assert death_with_tx_date != nat_death_date
    assert death_with_tx_date > sim.date
    sim.date = death_with_tx_date
    death_with_tx_event.apply(person_id)
    # check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not pd.isnull(person['un_sam_death_date'])
    assert person['un_sam_death_date'] == sim.date
    assert not person['is_alive']


def test_tx_recovery_before_nat_recovery_moderate_wasting_scheduled(tmpdir):
    """ Show that if recovered with a treatment event before the person was going to recover naturally from moderate
    wasting with moderate or severe acute malnutrition, it causes the episode to end earlier, natural recovery is
    cancelled.
    Test for MAM and complicated SAM. """
    for am_state_expected in ['MAM', 'SAM']:
        popsize = 1000
        sim = get_sim(tmpdir)
        # get wasting module
        wmodule = sim.modules['Wasting']
        p = wmodule.parameters

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)  # zero duration
        sim.event_queue.queue = []  # clear the queue

        # Set moderate wasting incidence rate at 100% and rate of progression to severe wasting at 0%.
        # (Hence, all children with no wasting should get onset of moderate wasting and be scheduled for natural
        # recovery.)
        wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
        wmodule.wasting_models.severe_wasting_progression_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
        # Set probs of full recovery from MAM and SAM at 100%, so with treatment they always fully recover
        wmodule.wasting_models.acute_malnutrition_recovery_mam_lm = LinearModel.multiplicative()
        wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative()

        # Get person to use:
        df = sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        # Manually set this individual properties to be well
        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
        df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
        df.at[person_id, 'un_am_nutritional_oedema'] = False
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_sam_death_date'] = pd.NaT

        if am_state_expected == 'MAM':
            # Set probability of MUAC < 115mm with moderate wasting, and probability of oedema with moderate wasting
            # at 0% in order to have MAM with onset of wasting
            p['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = 0.0
            p['proportion_WHZ<-2_with_oedema'] = 0.0
        else:  # am_state_expected == 'SAM'
            # Set probability of oedema with moderate wasting at 100% in order to have SAM with onset of wasting
            p['proportion_WHZ<-2_with_oedema'] = 1.0
            # Ensure the individual has always complications when SAM occurs
            p['prob_complications_in_SAM'] = 1.0

        # Run Wasting Polling event to get new incident cases:
        polling = Wasting_IncidencePoll(module=wmodule)
        polling.apply(sim.population)

        # Check that there is a natural recovery event scheduled:
        #  Wasting_FullRecovery_Event if this person has MAM, Wasting_RecoveryToMAM_Event if this person has SAM
        if am_state_expected == 'MAM':
            recov_event_type = Wasting_FullRecovery_Event
        else:  # am_state_expected == 'SAM'
            recov_event_type = Wasting_RecoveryToMAM_Event
        nat_recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                             if isinstance(event_tuple[1], recov_event_type)][0]
        date_of_scheduled_nat_recov = nat_recov_event_tuple[0]
        nat_recov_event = nat_recov_event_tuple[1]

        # Start appropriate treatment
        if am_state_expected == 'MAM':
            wmodule.do_when_am_treatment(person_id, treatment='SFP')
        else: # complicated SAM
            wmodule.do_when_am_treatment(person_id, treatment='ITC')
        assert df.at[person_id, 'un_am_tx_start_date'] == sim.date

        # Check full recovery with treatment is scheduled before the natural recovery
        full_recov_events = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                                 if isinstance(event_tuple[1], Wasting_FullRecovery_Event)]
        if am_state_expected == 'MAM':
            assert len(full_recov_events) == 2, (f"Two full recovery events should be scheduled (natural and following "
                                                 f"treatment), but {len(full_recov_events)} is/are scheduled.")
            # check the natural full recovery is at position 1;
            # hence full recovery following treatment will always be at position 0
            nat_recov_event_tuple = full_recov_events[1]
            date_of_scheduled_nat_recov_to_confirm = nat_recov_event_tuple[0]
            assert date_of_scheduled_nat_recov_to_confirm == date_of_scheduled_nat_recov
        else: # complicated SAM
            assert len(full_recov_events) == 1, (f"One full recovery event should be scheduled (following treatment),"
                                                 f"but {len(full_recov_events)} is/are scheduled.")
        # full recovery following treatment at position 0
        tx_recov_event_tuple = full_recov_events[0]
        date_of_scheduled_tx_recov = tx_recov_event_tuple[0]
        tx_recov_event = tx_recov_event_tuple[1]
        assert date_of_scheduled_tx_recov > sim.date
        assert date_of_scheduled_tx_recov < date_of_scheduled_nat_recov

        # Run a recovery event due to treatment first
        sim.date = date_of_scheduled_tx_recov
        tx_recov_event.apply(person_id)
        # check properties of this individual, should have recovered today, is not wasted, is well-nourished and alive
        person = df.loc[person_id]
        assert person['un_am_recovery_date'] == sim.date
        assert person['un_WHZ_category'] == 'WHZ>=-2'
        assert person['un_am_MUAC_category'] == '>=125mm'
        assert not person['un_am_nutritional_oedema']
        assert person['un_clinical_acute_malnutrition'] == 'well'
        assert person['is_alive']
        assert pd.isnull(person['un_sam_death_date'])

        # Check natural recovery is going to be cancelled
        if am_state_expected == 'MAM':
            assert  pd.isnull(person['un_full_recov_date'])
        else: # complicated SAM
            assert pd.isnull(person["un_recov_to_mam_date"])
        # Run the natural recovery, this should have no effect
        sim.date = date_of_scheduled_nat_recov
        nat_recov_event.apply(person_id)
        # check properties of this individual are still exact the same
        person = df.loc[person_id]
        assert person['un_am_recovery_date'] == date_of_scheduled_tx_recov
        assert date_of_scheduled_tx_recov < sim.date
        assert person['un_WHZ_category'] == 'WHZ>=-2'
        assert person['un_am_MUAC_category'] == '>=125mm'
        assert not person['un_am_nutritional_oedema']
        assert person['un_clinical_acute_malnutrition'] == 'well'
        assert person['is_alive']
        assert pd.isnull(person['un_sam_death_date'])


def test_recovery_before_death_scheduled(tmpdir):
    """ Test that if a recovery event following a treatment occurs before a scheduled death due to untreated SAM, the
    person recovers and does not die."""
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']
    p = wmodule.parameters

    # Set death due to untreated SAM at 0% for all, so natural recovery from SAM is always scheduled
    p['base_death_rate_untreated_SAM'] = 0.0
    p['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)  # zero duration
    sim.event_queue.queue = []  # clear the queue

    # Set moderate wasting incidence, progression to severe wasting, and death rate after SAM care at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()
    p['prob_death_after_SAMcare'] = 1.0
    # Set full recovery with SAM care at 0%. With a 100% death rate after SAM care, there will be no recovery to MAM;
    # all individuals will die after receiving SAM care.
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    # Ensure the individual has no complications when SAM occurs
    p['prob_complications_in_SAM'] = 0.0

    # Get person to use
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to be well
    df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.at[person_id, 'un_am_nutritional_oedema'] = False
    df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
    df.at[person_id, 'un_sam_with_complications'] = False
    df.at[person_id, 'un_sam_death_date'] = pd.NaT

    # Run Wasting Polling event to get new incident cases:
    polling = Wasting_IncidencePoll(module=wmodule)
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted  with onset in previous month
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_clinical_acute_malnutrition'] != 'well'
    assert not df.at[person_id, 'un_sam_with_complications']
    days_in_month = (sim.date - pd.DateOffset(days=1)).days_in_month
    assert (person['un_last_wasting_date_of_onset'] < sim.date) and \
           (person['un_last_wasting_date_of_onset'] >= sim.date - pd.to_timedelta(days_in_month, unit='D'))
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])  # no death due to untreated SAM

    # Check that there is a Wasting_ProgressionToSevere_Event scheduled for this person
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                               if isinstance(event_tuple[1], Wasting_ProgressionToSevere_Event)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: should now be severely wasted
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert not df.at[person_id, 'un_sam_with_complications']
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    # since there is no natural death, no death date should be scheduled,
    assert pd.isnull(person['un_sam_death_date'])  # no death due to untreated SAM
    # Check that there is a natural recovery to MAM scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_RecoveryToMAM_Event)

    # Run health seeking behavior and ensure non-emergency event is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()
    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericNonEmergencyFirstAppt and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # Check outpatient care event is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # Check death event is scheduled since there is zero recovery rate with treatment
    assert isinstance(sim.find_events_for_person(person_id)[2][1], Wasting_SevereAcuteMalnutritionDeath_Event)

    # Check a date of scheduled death
    # we assume OTC treatment to be longer than natural recovery from sev. wasting,
    # (if decided in future to change this assumption, the test will need to be updated)
    assert p['tx_length_weeks_OutpatientSAM'] * 7 > p['duration_of_untreated_sev_wasting']
    # hence the death should be scheduled before the natural recovery
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    nat_recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], Wasting_RecoveryToMAM_Event)][0]
    date_of_scheduled_nat_recov = nat_recov_event_tuple[0]
    assert date_of_scheduled_death > sim.date
    assert date_of_scheduled_death > date_of_scheduled_nat_recov

    # Run a full recovery event (it is not scheduled though)
    df.at[person_id, 'un_full_recov_date'] = sim.date
    full_recov_event = Wasting_FullRecovery_Event(person_id=person_id, module=wmodule)
    full_recov_event.apply(person_id=person_id)

    # Check that the person is fully recovered and is still alive
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert person['un_am_MUAC_category'] == '>=125mm'
    assert not person['un_am_nutritional_oedema']
    assert person['un_clinical_acute_malnutrition'] == 'well'
    assert not person['un_sam_with_complications']
    assert person['un_am_recovery_date'] == sim.date
    assert pd.isnull(person['un_sam_death_date'])
    assert pd.isnull(person['un_recov_to_mam_date'])
    assert pd.isnull(person['un_full_recov_date'])
    assert pd.isnull(person['un_progression_date'])
    assert person['is_alive']

    # Run the death event that was originally scheduled - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    assert df.at[person_id, 'is_alive']

def test_no_wasting_after_recent_recovery(tmpdir):
    """ Test that a person who recovered from wasting 5 days ago does not become wasted again. (The 5-day interval is
    used as an example within the assumed 14-day relapse-free window.) """
    popsize = 1000
    sim = get_sim(tmpdir)
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date)  # zero duration
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # Manually set this individual properties to be well and recovered 5 days ago
    df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.at[person_id, 'un_am_nutritional_oedema'] = False
    df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
    df.at[person_id, 'un_am_recovery_date'] = sim.date - pd.DateOffset(days=5)

    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # Run Wasting Polling event
    polling = Wasting_IncidencePoll(module=wmodule)
    polling.apply(sim.population)

    # Check properties of this individual: should still be well
    assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'well'
