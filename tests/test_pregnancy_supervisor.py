import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    wasting,
)

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
    """Force all women of reproductive age to be pregnant at the start of the simulation and overrides parity set at
     initialisation of simulation """
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)

    all_women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]
    df.loc[all_women.index, 'la_parity'] = 0


def turn_off_antenatal_pregnancy_loss(sim):
    """Set all parameters which output probability of pregnancy loss to 0"""
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 0.0
    params['prob_induced_abortion_per_month'] = 0.0
    params['prob_still_birth_per_month'] = 0.0

    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),  # went set disable=true, cant check HSI queue
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule(),
                 )


def test_pregnancy_supervisor_chorio_and_prom(seed):
    """Tests the application of risk of chorioamnionitis and PROM within the pregnancy supervisor event"""

    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    turn_off_antenatal_pregnancy_loss(sim)

    # Set risk of PROM to 1
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_prom_per_month'] = 0.0
    params['prob_chorioamnionitis'] = 1.0
    params['prob_seek_care_pregnancy_complication'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the event
    sim.date = sim.date + pd.DateOffset(weeks=20)
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check that despite risk of chorio being 1, not one should develop is because PROM cannot occur
    assert not df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'].any().any()

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    assert not df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'].any().any()
    assert not df.loc[pregnant_women.index, 'ps_chorioamnionitis'].any().any()

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_prom_per_month'] = 1.0
    params['prob_chorioamnionitis'] = 1.0

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Roll back gestational age, set risk of stillbirth to 1
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'] = True
    pregnancy_sup.apply(sim.population)

    # Check women have correctly developed chorio
    assert (df.loc[pregnant_women.index, 'ps_chorioamnionitis']).all().all()

    # Check care seeking has occured as expected
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment in hsi_events

    # Now clear the event queue
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df.loc[pregnant_women.index, 'ps_chorioamnionitis'] = False

    # Finally, block care seeking and set risk of death to high
    params['prob_seek_care_pregnancy_complication'] = 0.0
    params['prob_antenatal_sepsis_death'] = 1.0

    # prevent preterm birth which can effect care seeking by updating la_due_date_current_pregnancy
    params['baseline_prob_early_labour_onset'] = [0.0, 0.0, 0.0, 0.0]

    pregnancy_sup.apply(sim.population)

    # Check women from the series has correctly died
    assert (df.loc[pregnant_women.index, 'ps_chorioamnionitis']).all().all()
    assert not (df.loc[pregnant_women.index, 'is_alive']).any().any()
    for person in pregnant_women.index:
        assert person not in list(sim.modules['PregnancySupervisor'].mother_and_newborn_info)


def test_induction_of_labour_logic(seed):
    """Tests the that woman who are post-term are seeking care for induction of labour"""

    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_seek_care_induction'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the event, asume all women are now 41 weeks pregnant
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=39)
    pregnancy_sup.apply(sim.population)

    # Check care seeking for induction has occured
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour in hsi_events

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Roll back gestational age, set risk of stillbirth to 1
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    # block care seeking for induction
    params['prob_seek_care_induction'] = 0.0

    # module code divides monthly risk by weeks (assuming 4.5 weeks in a month) so we set the intercept
    # 4.5 times greater than 1 to assure sb will happen
    params['prob_still_birth_per_month'] = 4.5

    # Check that instead of seeking care for induction women have experience post term stillbirth
    pregnancy_sup.apply(sim.population)
    assert not (df.loc[pregnant_women.index, 'is_pregnant']).any().any()
    assert (df.loc[pregnant_women.index, 'ps_prev_stillbirth']).all().all()
    for person in pregnant_women.index:
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['delete_mni']
