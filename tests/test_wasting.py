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
                     show_progress_bar=False,
                     log_config={
                         'filename': 'tmp',
                         'directory': tmpdir,
                         'custom_levels': {
                             "*": logging.WARNING,
                             "tlo.methods.wasting": logging.INFO}
                     })

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all',
                                           equip_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath)
                 )
    return sim


@pytest.mark.slow
def test_basic_run(tmpdir):
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

            # WHZ standard deviation of -3, MUAC <115mm, and oedema should cause severe acute malnutrition
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
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)

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

    # Check properties of individuals: should now be moderately wasted
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    assert all(under5s['un_ever_wasted'])
    assert (under5s['un_WHZ_category'].eq('-3<=WHZ<-2')).all()
    assert (under5s['un_last_wasting_date_of_onset'].eq(sim.date)).all()


def test_report_daly_weights(tmpdir):
    """ Check if daly weights reporting is done as expected. Four checks are made:
    1. For an individual who is well (No weight is expected/must be 0.0)
    2. For an individual with moderate wasting and oedema (expected daly weight is 0.051)
    3. For an individual with severe wasting and oedema (expected daly weight is 0.172)
    4. For an individual with severe wasting without oedema (expected daly weight is 0.128) """

    dur = pd.DateOffset(days=0)
    popsize = 1
    sim = get_sim(tmpdir)
    sim.modules['Demography'].parameters['max_age_initial'] = 4.9
    # get wasting module
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)

    # Dict to hold the DALY weights
    daly_wts = dict()

    # Get person to use
    df = sim.population.props
    person_id = df.index[0]
    df.at[person_id, 'is_alive'] = True

    # 1. Check daly weight for well-nourished person (weight is 0.0)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.loc[person_id, 'un_am_nutritional_oedema'] = False
    df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'

    # Verify diagnosis - an individual should be well
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'well'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Verify that individual has no daly weight
    assert daly_weights_reported.loc[person_id] == 0.0

    get_daly_weights = sim.modules['HealthBurden'].get_daly_weight

    # 2. Check daly weight for person with moderate wasting and oedema (weight is 0.051)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
    df.loc[person_id, 'un_am_nutritional_oedema'] = True

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get daly weight of moderate wasting with oedema
    daly_wts['mod_wasting_with_oedema'] = get_daly_weights(sequlae_code=461)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['mod_wasting_with_oedema'] == daly_weights_reported.loc[person_id]

    # 3. Check daly weight for person with severe wasting and oedema (weight is 0.172)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_am_nutritional_oedema'] = True

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get daly weight of severe wasting with oedema
    daly_wts['sev_wasting_with_oedema'] = get_daly_weights(sequlae_code=463)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_with_oedema'] == daly_weights_reported.loc[person_id]

    # 4. Check daly weight for person with severe wasting without oedema (weight is 0.128)
    # Reset diagnosis
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_am_nutritional_oedema'] = False

    # Verify diagnosis - an individual should have SAM
    wmodule.clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = wmodule.report_daly_values()

    # Get day weight of severe wasting without oedema
    daly_wts['sev_wasting_w/o_oedema'] = get_daly_weights(sequlae_code=462)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_w/o_oedema'] == daly_weights_reported.loc[person_id]


def test_nat_recovery_moderate_wasting(tmpdir):
    """ Check natural recovery after onset of moderate wasting with MAM diagnosis. """
    for am_state_expected in ['MAM', 'SAM']:
        dur = pd.DateOffset(days=0)
        popsize = 1000
        sim = get_sim(tmpdir)
        # get wasting module
        wmodule = sim.modules['Wasting']

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date + dur)
        sim.event_queue.queue = []  # clear the queue

        # Get person to use:
        df = sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        person_id = under5s.index[0]
        # Reset properties of this individual to be well-nourished
        df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not wasted
        df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'
        df.loc[person_id, 'un_am_nutritional_oedema'] = False
        df.loc[person_id, 'un_clinical_acute_malnutrition'] = 'well' # well-nourished
        df.loc[person_id, 'un_ever_wasted'] = False
        df.loc[person_id, 'un_last_wasting_date_of_onset'] = pd.NaT

        # Set incidence of moderate wasting at 100%
        wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
        # Set progression rate to severe wasting at 0% (hence, natural recovery always scheduled)
        wmodule.wasting_models.severe_wasting_progression_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
        if am_state_expected == 'MAM':
            # Set probability of MUAC < 115mm with moderate wasting, and probability of oedema with moderate wasting
            # at 0% in order to have MAM with onset of wasting
            wmodule.parameters['proportion_-3<=WHZ<-2_with_MUAC<115mm'] = 0.0
            wmodule.parameters['proportion_WHZ<-2_with_oedema'] = 0.0
        else:  # am_state_expected == 'SAM'
            # Set probability of oedema with moderate wasting at 100% in order to have SAM with onset of wasting
            wmodule.parameters['proportion_WHZ<-2_with_oedema'] = 1.0

        # Run Wasting Polling event: This event should cause all young children to be moderately wasted
        polling = Wasting_IncidencePoll(module=wmodule)
        polling.apply(sim.population)

        # Check properties of this individual: should now be moderately wasted with MAM or SAM respectively
        person = df.loc[person_id]
        assert person['un_ever_wasted']
        assert person['un_WHZ_category'] == '-3<=WHZ<-2'
        assert person['un_last_wasting_date_of_onset'] == sim.date
        assert pd.isnull(person['un_am_tx_start_date'])
        assert pd.isnull(person['un_am_recovery_date'])
        if am_state_expected == 'MAM':
            assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM'
        else:
            assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

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


def test_tx_recovery_severe_acute_malnutrition_without_complications(tmpdir):
    """ Check the onset of symptoms with SAM, check recovery to MAM with tx when
    the progression to severe wasting is certain, hence no natural recovery from moderate wasting,
    the natural death due to SAM is certain, hence no natural recovery from severe wasting,
    and check natural death canceled w\ tx and symptoms resolved when recovered to MAM with tx. """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)
    # get wasting module
    wmodule = sim.modules['Wasting']

    # Set death due to untreated SAM at 100% for all, hence no natural recovery from severe wasting
    wmodule.parameters['base_death_rate_untreated_SAM'] = 1.0
    wmodule.parameters['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to be well
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.loc[person_id, 'un_am_nutritional_oedema'] = False
    df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
    df.at[person_id, 'un_sam_with_complications'] = False
    df.at[person_id, 'un_sam_death_date'] = pd.NaT

    # Ensure the individual has no complications when SAM occurs
    wmodule.parameters['prob_complications_in_SAM'] = 0.0
    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    # Set progress to severe wasting at 100% as well, hence no natural recovery from moderate wasting
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()
    # Set complete recovery from SAM to zero. We want those with SAM to recover to MAM with tx
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    # Set prob of death after tx at 0% (hence recovery to MAM w\ tx at 100%)
    wmodule.parameters['prob_death_after_SAMcare'] = 0.0

    # Run Wasting Polling event to get new incident cases:
    polling = Wasting_IncidencePoll(module=wmodule)
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'

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
    # run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
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

    # Check death was canceled with tx
    assert pd.isnull(df.loc[person_id]['un_sam_death_date'])

    # Check recovery to MAM due to tx is scheduled
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


def test_recovery_severe_acute_malnutrition_with_complications(tmpdir):
    """ test individual's recovery from wasting with complications """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    # get wasting module
    wmodule = sim.modules['Wasting']

    # set death due to untreated SAM at 100% for all, hence no natural recovery from severe wasting
    wmodule.parameters['base_death_rate_untreated_SAM'] = 1.0
    wmodule.parameters['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # Manually set this individual properties to have severe acute malnutrition
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_last_wasting_date_of_onset'] = sim.date
    # ensure the individual has complications due to SAM
    wmodule.parameters['prob_complications_in_SAM'] = 1.0
    # assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # by having severe wasting, this individual should be diagnosed as SAM
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'
    # should have complications
    assert df.at[person_id, 'un_sam_with_complications']
    # symptoms should be applied
    assert person_id in set(sim.modules['SymptomManager'].who_has('weight_loss'))

    # Check death is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1], Wasting_SevereAcuteMalnutritionDeath_Event)
    assert not pd.isnull(df.loc[person_id]['un_sam_death_date'])
    # get date of death and death event
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # make full recovery rate to 100% so that this individual should fully recover with tx
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative()

    # run care seeking event and ensure HSI for complicated SAM is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
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

    # Check scheduled death was canceled due to tx
    person = df.loc[person_id]
    assert pd.isnull(person['un_sam_death_date'])

    # Check full recovery due to tx is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_ClinicalAcuteMalnutritionRecovery_Event)
    # get date of full recovery and the recovery event
    sam_recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], Wasting_ClinicalAcuteMalnutritionRecovery_Event)][0]
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
    assert (person['un_am_MUAC_category'] == '>=125mm')
    assert pd.isnull(person['un_sam_death_date'])
    assert person['is_alive']

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id=person_id, disease_module=sim.modules['Wasting']))


def test_nat_hist_death(tmpdir):
    """ Check: Wasting onset --> death """
    """ Check if the risk of death is 100% does everyone with SAM die? """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # get wasting module
    wmodule = sim.modules['Wasting']

    # Set death rate with tx at 100%
    wmodule.parameters['prob_death_after_SAMcare'] = 1.0

    # make zero recovery rate. reset recovery linear model
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Get the children to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # make an individual diagnosed as SAM by WHZ category.
    # We want to make this individual qualify for death
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'

    # assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # apply wasting symptoms to this individual
    wmodule.wasting_clinical_symptoms(person_id)

    # check symptoms are applied
    assert person_id in set(sim.modules['SymptomManager'].who_has('weight_loss'))

    # run health seeking behavior and ensure non-emergency event is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
          if isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # check outpatient care event is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id)
              if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # since there is zero recovery rate, check death event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1], Wasting_SevereAcuteMalnutritionDeath_Event)

    # # Run the acute death event and ensure the person is now dead:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be dead)
    person = df.loc[person_id]
    assert not pd.isnull(person['un_sam_death_date'])
    assert person['un_sam_death_date'] == sim.date
    assert not person['is_alive']


def test_nat_hist_cure_if_recovery_scheduled(tmpdir):
    """ Show that if a cure event is run before when a person was going to recover naturally, it causes the episode
    to end earlier. """

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    wmodule = sim.modules['Wasting']
    p = wmodule.parameters
    # set prob of death after tx at 0% (hence recovery to MAM at 100%)
    p['prob_death_after_SAMcare'] = 0.0

    # increase wasting incidence rate to 100% and reduce rate of progress to severe wasting to zero. We don't want
    # individuals to progress to SAM as we are testing for MAM natural recovery
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = Wasting_IncidencePoll(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted without progression to severe)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a Wasting_NaturalRecovery_Event scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], Wasting_NaturalRecovery_Event)][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run a Cure Event after the length of the treatment
    def get_tx_length(in_person_id):
        if df.at[in_person_id, 'un_sam_with_complications']:
            tx_length = p['tx_length_weeks_InpatientSAM']
        elif df.at[in_person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
            tx_length = p['tx_length_weeks_OutpatientSAM']
        else:  # df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM':
            tx_length = p['tx_length_weeks_SuppFeedingMAM']
        return tx_length
    sim.date = sim.date + DateOffset(weeks=get_tx_length(person_id))
    assert sim.date < date_of_scheduled_recov
    cure_event = Wasting_ClinicalAcuteMalnutritionRecovery_Event(person_id=person_id, module=sim.modules['Wasting'])
    cure_event.apply(person_id=person_id)

    # Check the natural recovery was cancelled with the cure:
    assert date_of_scheduled_recov in df.at[person_id, 'un_nat_recov_to_cancel']

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the recovery event that was originally scheduled - this should have no effect
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_nat_hist_cure_if_death_scheduled(tmpdir):
    """Show that if a cure event is run before when a person was going to die, it causes the episode to end without
    the person dying."""

    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    # get wasting module
    wmodule = sim.modules['Wasting']

    # set death due to untreated SAM at 0% for all, hence always scheduled natural recovery from severe wasting
    wmodule.parameters['base_death_rate_untreated_SAM'] = 0.0
    wmodule.parameters['rr_death_rate_by_agegp'] = [1, 1, 1, 1, 1, 1]

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # increase to 100% wasting incidence, progress to severe wasting, and death rate after SAM care,
    # set full recovery with SAM care at 0% (and as 100% death rate after SAM care, no recovery to MAM)
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    wmodule.parameters['prob_death_after_SAMcare'] = 1.0

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # Manually set this individual properties to be well
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.loc[person_id, 'un_am_nutritional_oedema'] = False
    df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
    df.at[person_id, 'un_sam_with_complications'] = False
    df.at[person_id, 'un_sam_death_date'] = pd.NaT

    # Run Wasting Polling event to get new incident cases:
    polling = Wasting_IncidencePoll(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a Wasting_ProgressionToSevere_Event scheduled for this person
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                               if isinstance(event_tuple[1], Wasting_ProgressionToSevere_Event)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check properties of this individual: (should now be severely wasted and without a scheduled death date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # run health seeking behavior and ensure non-emergency event is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # check outpatient care event is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_OutpatientTherapeuticProgramme_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # since there is no natural death, natural recovery should be scheduled, and
    # since there is zero recovery rate with tx, death event after care should be scheduled
    print(f"{sim.find_events_for_person(person_id)=}")
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_NaturalRecovery_Event) or \
            isinstance(sim.find_events_for_person(person_id)[2][1], Wasting_NaturalRecovery_Event)
    assert isinstance(sim.find_events_for_person(person_id)[1][1], Wasting_SevereAcuteMalnutritionDeath_Event) or \
            isinstance(sim.find_events_for_person(person_id)[2][1], Wasting_SevereAcuteMalnutritionDeath_Event)

    # Check a date of death is scheduled. it should be any date in the future:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], Wasting_SevereAcuteMalnutritionDeath_Event)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event now
    cure_event = Wasting_ClinicalAcuteMalnutritionRecovery_Event(person_id=person_id, module=sim.modules['Wasting'])
    cure_event.apply(person_id=person_id)

    # Check that the person is not wasted and is alive still:
    person = df.loc[person_id]
    assert person['is_alive']
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert not pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Run the death event that was originally scheduled - this should have no effect and the person should not die
    sim.date = date_of_scheduled_death
    death_event.apply(person_id=person_id)
    person = df.loc[person_id]
    assert person['is_alive']

def test_no_wasting_after_recent_recovery(tmpdir):
    """ Test that a person who recovered from wasting 5 days ago does not become wasted again. """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)
    wmodule = sim.modules['Wasting']

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # Manually set this individual properties to be well and recovered 5 days ago
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'
    df.loc[person_id, 'un_am_nutritional_oedema'] = False
    df.loc[person_id, 'un_clinical_acute_malnutrition'] = 'well'
    df.loc[person_id, 'un_am_recovery_date'] = sim.date - pd.DateOffset(days=5)

    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # Run Wasting Polling event
    polling = Wasting_IncidencePoll(module=wmodule)
    polling.apply(sim.population)

    # Check properties of this individual: should still be well
    person = df.loc[person_id]
    assert person['un_clinical_acute_malnutrition'] == 'well'
