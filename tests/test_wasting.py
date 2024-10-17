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
    HSI_Wasting_InpatientCareForComplicated_SAM,
    HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
    WastingClinicalAcuteMalnutritionRecoveryEvent,
    WastingIncidencePollingEvent,
    WastingNaturalRecoveryEvent,
    WastingProgressionToSevereEvent,
    WastingSevereAcuteMalnutritionDeathEvent,
    WastingUpdateToMAM,
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
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath)
                 )
    return sim


@pytest.mark.slow
def test_basic_run(tmpdir):
    """Run the simulation and do some daily checks on dtypes and properties integrity """
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
            assert (df.dtypes == orig.dtypes).all()

        def check_configuration_of_properties(self, population):
            """ check wasting properties on a daily basis to ensure integrity """
            df = population.props
            under5_sam = df.index[df.is_alive & (df.age_exact_years < 5) &
                                  (df.un_clinical_acute_malnutrition == 'SAM')]

            # Those that were never wasted, should have normal WHZ score:
            assert (df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna(), 'un_WHZ_category'] == 'WHZ>=-2').all()

            # Those for whom the death date has past should be dead
            assert not df.loc[df.un_ever_wasted & (df['un_sam_death_date'] < self.sim.date), 'is_alive'].any()
            assert not df.loc[
                (df.un_clinical_acute_malnutrition == 'SAM') & (df['un_sam_death_date'] < self.sim.date),
                'is_alive'].any()

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

            # WHZ standard deviation of -3, oedema, and MUAC <115mm should cause severe acute malnutrition
            whz_index = df.index[df['un_WHZ_category'] == 'WHZ<-3']
            oedema_index = df.index[df['un_am_bilateral_oedema']]
            muac_index = df.index[df['un_am_MUAC_category'] == '<115mm']
            assert (df.loc[whz_index, 'un_clinical_acute_malnutrition'] == "SAM").all()
            assert (df.loc[oedema_index, 'un_clinical_acute_malnutrition'] == "SAM").all()
            assert (df.loc[muac_index, 'un_clinical_acute_malnutrition'] == "SAM").all()

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
    """Check Incidence of wasting is happening as expected """
    # get simulation object:
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)

    # reset properties of all individuals so that they are not wasted
    df = sim.population.props
    df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
    df.loc[df.is_alive, 'un_ever_wasted'] = False
    df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT

    # Set incidence of wasting at 100%
    sim.modules['Wasting'].wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # Run polling event: check that all children should now have moderate wasting:
    polling = WastingIncidencePollingEvent(sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of individuals: should now be moderately wasted
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    assert all(under5s['un_ever_wasted'])
    assert all(under5s['un_WHZ_category'] == '-3<=WHZ<-2')
    assert all(under5s['un_last_wasting_date_of_onset'] == sim.date)


def test_report_daly_weights(tmpdir):
    """Check if daly weights reporting is done as expected. Four checks are made:
    1. For an individual who is well (No weight is expected/must be 0.0)
    2. For an individual with moderate wasting and oedema (expected daly weight is 0.051)
    3. For an individual with severe wasting and oedema (expected daly weight is 0.172)
    4. For an individual with severe wasting without oedema (expected daly weight is 0.128)"""

    dur = pd.DateOffset(days=0)
    popsize = 1
    sim = get_sim(tmpdir)
    sim.modules['Demography'].parameters['max_age_initial'] = 4.9
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)

    # Dict to hold the DALY weights
    daly_wts = dict()

    # Get person to use
    df = sim.population.props
    person_id = df.index[0]
    df.at[person_id, 'is_alive'] = True

    # Check daly weight for not undernourished person (weight is 0.0)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    df.loc[person_id, 'un_am_bilateral_oedema'] = False
    df.loc[person_id, 'un_am_MUAC_category'] = '>=125mm'

    # Verify diagnosis - an individual should be well
    sim.modules["Wasting"].clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'well'

    # Report daly weight for this individual
    daly_weights_reported = sim.modules["Wasting"].report_daly_values()

    # Verify that individual has no daly weight
    assert daly_weights_reported.loc[person_id] == 0.0

    get_daly_weights = sim.modules['HealthBurden'].get_daly_weight

    # Check daly weight for person with moderate wasting and oedema (weight is 0.051)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
    df.loc[person_id, 'un_am_bilateral_oedema'] = True

    # Verify diagnosis - an individual should be SAM
    sim.modules["Wasting"].clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = sim.modules["Wasting"].report_daly_values()

    # Get daly weight of moderate wasting with oedema
    daly_wts['mod_wasting_with_oedema'] = get_daly_weights(sequlae_code=461)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['mod_wasting_with_oedema'] == daly_weights_reported.loc[person_id]

    # Check daly weight for person with severe wasting and oedema (weight is 0.172)
    # Reset diagnostic properties
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_am_bilateral_oedema'] = True

    # Verify diagnosis - an individual should be SAM
    sim.modules["Wasting"].clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = sim.modules["Wasting"].report_daly_values()

    # Get daly weight of severe wasting with oedema
    daly_wts['sev_wasting_with_oedema'] = get_daly_weights(sequlae_code=463)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_with_oedema'] == daly_weights_reported.loc[person_id]

    # Check daly weight for person with severe wasting without oedema (weight is 0.128)
    # Reset diagnosis
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'
    df.loc[person_id, 'un_am_bilateral_oedema'] = False

    # Verify diagnosis - an individual should be SAM
    sim.modules["Wasting"].clinical_acute_malnutrition_state(person_id, df)
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # Report daly weight for this individual
    daly_weights_reported = sim.modules["Wasting"].report_daly_values()

    # Get day weight of severe wasting without oedema
    daly_wts['sev_wasting_w/o_oedema'] = get_daly_weights(sequlae_code=462)

    # Compare the daly weight of this individual with the daly weight obtained from HealthBurden module
    assert daly_wts['sev_wasting_w/o_oedema'] == daly_weights_reported.loc[person_id]


def test_recovery_moderate_wasting(tmpdir):
    """Check natural recovery of moderate wasting """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # make this individual have no wasting
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    # confirm wasting property is reset. This individual should have no wasting
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Set incidence of wasting at 100%
    sim.modules['Wasting'].wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # Run Wasting Polling event: This event should cause all young children to be moderate wasting
    polling = WastingIncidencePollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a WastingNaturalRecoveryEvent scheduled
    # for this person
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], WastingNaturalRecoveryEvent)][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run the recovery event:
    sim.date = date_of_scheduled_recov
    recov_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'well':
        assert person['un_am_recovery_date'] == sim.date
    else:
        assert pd.isnull(df.at[person_id, 'un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])


def test_recovery_severe_wasting_without_complications(tmpdir):
    """ Check recovery to MAM by removing death rate for those with severe wasting, and check the onset of
    symptoms with SAM and revolving of symptoms when recovered to MAM """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    # make this individual have no wasting
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
    # confirm wasting property is reset. This individual should have no wasting
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # get wasting module
    wmodule = sim.modules['Wasting']

    # Set incidence of wasting at 100%
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()

    # set progress to severe wasting at 100% as well
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()

    # set complete recovery from wasting to zero. We want those with SAM to recover to MAM
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Set death rate at 0% and recovery to MAM at 100%
    wmodule.parameters['prob_death_after_care'] = 0.0
    wmodule.parameters['prob_mam_after_care'] = 1.0

    # Run Wasting Polling event to get new incident cases:
    polling = WastingIncidencePollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: should now be moderately wasted
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'

    # Check that there is a WastingProgressionToSevereEvent scheduled for this person:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                               if isinstance(event_tuple[1], WastingProgressionToSevereEvent)][0]
    date_of_scheduled_progression = progression_event_tuple[0]
    progression_event = progression_event_tuple[1]
    assert date_of_scheduled_progression > sim.date

    # Run the progression to severe wasting event:
    sim.date = date_of_scheduled_progression
    progression_event.apply(person_id=person_id)

    # Check this individual has symptom(weight loss) caused by Wasting (SAM only)
    assert 'weight_loss' in sim.modules['SymptomManager'].has_what(
        person_id=person_id, disease_module=sim.modules['Wasting']
    )

    # Check properties of this individual: (should now be severely wasted, diagnosed as SAM and without a scheduled
    # death date)
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ<-3'
    assert person['un_clinical_acute_malnutrition'] == 'SAM'
    assert pd.isnull(person['un_sam_death_date'])

    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # check HSI event is scheduled
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

    # check recovery event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], WastingUpdateToMAM)

    # Run the recovery event and check the individual has recovered from SAM:
    sam_recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id) if
                                isinstance(event_tuple[1], WastingUpdateToMAM)][0]

    date_of_scheduled_recovery_to_mam = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

    # Run SAM recovery
    sim.date = date_of_scheduled_recovery_to_mam
    sam_recovery_event.apply(person_id=person_id)

    # Check properties of this individual
    person = df.loc[person_id]
    assert person['un_clinical_acute_malnutrition'] == 'MAM'
    assert pd.isnull(person['un_sam_death_date'])

    # check they have no symptoms:
    assert 0 == len(sim.modules['SymptomManager'].has_what(person_id=person_id, disease_module=sim.modules['Wasting']))


def test_recovery_severe_wasting_with_complications(tmpdir):
    """ test individual's recovery from wasting with complications """
    dur = pd.DateOffset(days=0)
    popsize = 1000
    sim = get_sim(tmpdir)

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]

    # get wasting module
    wmodule = sim.modules['Wasting']

    # Manually set this individual properties to have
    # severe acute malnutrition with complications
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ<-3'

    # make the individual have wasting complications
    wmodule.parameters['prob_complications_in_SAM'] = 1.0

    # assign diagnosis
    wmodule.clinical_acute_malnutrition_state(person_id, df)

    # by having severe wasting, this individual should be diagnosed as SAM
    assert df.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'

    # symptoms should be applied
    assert person_id in set(sim.modules['SymptomManager'].who_has('weight_loss'))

    # should have complications
    assert df.at[person_id, 'un_sam_with_complications']

    # make recovery rate to 100% and death rate to zero so that
    # this individual should recover
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative()
    wmodule.parameters['prob_death_after_care'] = 0.0

    # run care seeking event and ensure HSI for complicated SAM is scheduled
    hsp = HealthSeekingBehaviourPoll(sim.modules['HealthSeekingBehaviour'])
    hsp.run()

    # check non-emergency care event is scheduled
    assert isinstance(sim.modules['HealthSystem'].find_events_for_person(person_id)[0][1],
                      hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)

    # Run the created instance of HSI_GenericFirstApptAtFacilityLevel0 and check care was sought
    # make inpatient care coverage 100%
    wmodule.parameters['coverage_inpatient_care'] = 1.0
    ge = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
          isinstance(ev[1], hsi_generic_first_appts.HSI_GenericNonEmergencyFirstAppt)][0]
    ge.run(squeeze_factor=0.0)

    # check HSI event for complicated SAM is scheduled
    hsi_event_scheduled = [
        ev
        for ev in sim.modules["HealthSystem"].find_events_for_person(person_id)
        if isinstance(ev[1], HSI_Wasting_InpatientCareForComplicated_SAM)
    ]
    assert 1 == len(hsi_event_scheduled)

    # Run the created instance of HSI_Wasting_OutpatientTherapeuticProgramme_SAM and check care was sought
    sam_ev = [ev[1] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
              isinstance(ev[1], HSI_Wasting_InpatientCareForComplicated_SAM)][0]
    sam_ev.run(squeeze_factor=0.0)

    # check recovery event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[0][1], WastingClinicalAcuteMalnutritionRecoveryEvent)

    # Run the recovery event and check the individual has recovered from SAM:
    sam_recovery_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                                if isinstance(event_tuple[1], WastingClinicalAcuteMalnutritionRecoveryEvent)][0]

    date_of_scheduled_recovery_to_mam = sam_recovery_event_tuple[0]
    sam_recovery_event = sam_recovery_event_tuple[1]
    assert date_of_scheduled_recovery_to_mam > sim.date

    # Run SAM recovery
    sim.date = date_of_scheduled_recovery_to_mam
    sam_recovery_event.apply(person_id=person_id)

    # Check properties of this individual. Should now be well
    person = df.loc[person_id]
    assert person['un_WHZ_category'] == 'WHZ>=-2'
    assert (person['un_am_MUAC_category'] == '>=125mm')
    assert pd.isnull(person['un_sam_death_date'])

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

    # Set death rate at 100%
    wmodule.parameters['prob_death_after_care'] = 1.0
    wmodule.parameters['prob_mam_after_care'] = 0.0

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
    # make outpatient care coverage 100%
    wmodule.parameters['coverage_outpatient_therapeutic_care'] = 1.0
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
    assert isinstance(sim.find_events_for_person(person_id)[0][1], WastingSevereAcuteMalnutritionDeathEvent)

    # # Run the acute death event and ensure the person is now dead:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], WastingSevereAcuteMalnutritionDeathEvent)][0]
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
    # Make 0% death rate by replacing with empty linear model 0.0
    wmodule.parameters['prob_death_after_care'] = 0.0
    wmodule.parameters['prob_mam_after_care'] = 1.0

    # increase wasting incidence rate to 100% and reduce rate of progress to severe wasting to zero.We don't want
    # individuals to progress to SAM as we are testing for MAM natural recovery
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    assert df.loc[person_id, 'un_WHZ_category'] == 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingIncidencePollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted without progression to severe)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a WastingNaturalRecoveryEvent scheduled for this person:
    recov_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], WastingNaturalRecoveryEvent)][0]
    date_of_scheduled_recov = recov_event_tuple[0]
    recov_event = recov_event_tuple[1]
    assert date_of_scheduled_recov > sim.date

    # Run a Cure Event
    cure_event = WastingClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id, module=sim.modules['Wasting'])
    cure_event.apply(person_id=person_id)

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

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=start_date + dur)
    sim.event_queue.queue = []  # clear the queue

    # get wasting module parameters
    wmodule = sim.modules['Wasting']

    # increase to 100% death rate, incidence and progress to severe wasting
    wmodule.wasting_models.wasting_incidence_lm = LinearModel.multiplicative()
    wmodule.wasting_models.severe_wasting_progression_lm = LinearModel.multiplicative()
    wmodule.wasting_models.acute_malnutrition_recovery_sam_lm = LinearModel(LinearModelType.MULTIPLICATIVE, 0.0)
    wmodule.parameters['prob_mam_after_care'] = 0.0
    wmodule.parameters['prob_death_after_care'] = 1.0

    # Get person to use:
    df = sim.population.props
    under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
    person_id = under5s.index[0]
    df.loc[person_id, 'un_WHZ_category'] = 'WHZ>=-2'

    # Run Wasting Polling event to get new incident cases:
    polling = WastingIncidencePollingEvent(module=sim.modules['Wasting'])
    polling.apply(sim.population)

    # Check properties of this individual: (should now be moderately wasted with a scheduled progression to severe date)
    person = df.loc[person_id]
    assert person['un_ever_wasted']
    assert person['un_WHZ_category'] == '-3<=WHZ<-2'
    assert person['un_last_wasting_date_of_onset'] == sim.date
    assert pd.isnull(person['un_am_tx_start_date'])
    assert pd.isnull(person['un_am_recovery_date'])
    assert pd.isnull(person['un_sam_death_date'])

    # Check that there is a WastingProgressionToSevereEvent scheduled for this person and if the person is scheduled for
    # death due to SAM, the death happens after the progression to severe wasting:
    progression_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                               if isinstance(event_tuple[1], WastingProgressionToSevereEvent)][0]
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

    # since there is zero recovery rate, check death event is scheduled
    assert isinstance(sim.find_events_for_person(person_id)[1][1], WastingSevereAcuteMalnutritionDeathEvent)

    # Check a date of death is scheduled. it should be any date in the future:
    death_event_tuple = [event_tuple for event_tuple in sim.find_events_for_person(person_id)
                         if isinstance(event_tuple[1], WastingSevereAcuteMalnutritionDeathEvent)][0]
    date_of_scheduled_death = death_event_tuple[0]
    death_event = death_event_tuple[1]
    assert date_of_scheduled_death > sim.date

    # Run a Cure Event now
    cure_event = WastingClinicalAcuteMalnutritionRecoveryEvent(person_id=person_id, module=sim.modules['Wasting'])
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
