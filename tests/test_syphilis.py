import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    syphilis,
    symptommanager,
)


try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def register_modules(sim):
    """Register the maternal/perinatal modules needed by Syphilis."""
    sim.register(
        demography.Demography(),
        contraception.Contraception(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
        symptommanager.SymptomManager(),
        healthsystem.HealthSystem(service_availability=['*'], cons_availability='all'),
        newborn_outcomes.NewbornOutcomes(),
        pregnancy_supervisor.PregnancySupervisor(),
        syphilis.Syphilis(),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
        labour.Labour(),
        postnatal_supervisor.PostnatalSupervisor(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        hiv.DummyHivModule(),
    )


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def get_sim(seed, population_size=100):
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=population_size)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    return sim


def setup_pregnant_woman_for_syphilis_test(sim, gestational_age=22):
    df = sim.population.props
    ps = sim.modules['PregnancySupervisor']
    mother_id = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)].index[0]

    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = gestational_age
    df.at[mother_id, 'ps_ectopic_pregnancy'] = 'none'
    pregnancy_helper_functions.update_mni_dictionary(ps, mother_id)

    return df, ps.mother_and_newborn_info, ps, mother_id


def test_syphilis_module_initialises_properties_and_parameters(seed):
    sim = get_sim(seed)
    df = sim.population.props
    syph = sim.modules['Syphilis']

    for property_name in ['ps_syphilis_state', 'ps_syphilis_treated', 'ps_congenital_syphilis']:
        assert property_name in df.columns

    alive = df.is_alive
    assert (df.loc[alive, 'ps_syphilis_state'] == 'none').all()
    assert not df.loc[alive, 'ps_syphilis_treated'].any()
    assert not df.loc[alive, 'ps_congenital_syphilis'].any()

    for parameter_name in syphilis.Syphilis.PARAMETERS:
        assert parameter_name in syph.current_parameters


def test_pre_existing_syphilis_probability_zero_leaves_woman_uninfected(seed):
    sim = get_sim(seed)
    df, mni, _, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    syph_module.current_parameters['prob_pre_existing_syphilis_at_pregnancy_start'] = 0.0
    syph_module.initialise_pre_existing_syphilis_for_pregnancy(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'none'
    assert not df.at[mother_id, 'ps_syphilis_treated']
    assert mni[mother_id]['syphilis_infection_origin'] == 'none'
    assert pd.isnull(mni[mother_id]['syphilis_next_progression_date'])


def test_pre_existing_syphilis_probability_one_assigns_infection_and_origin(seed):
    sim = get_sim(seed)
    df, mni, _, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    syph_module.current_parameters['prob_pre_existing_syphilis_at_pregnancy_start'] = 1.0
    syph_module.current_parameters['pre_existing_syphilis_stage_distribution'] = [1.0, 0.0, 0.0, 0.0]
    syph_module.initialise_pre_existing_syphilis_for_pregnancy(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] in syphilis.DETECTABLE_STAGES
    assert not df.at[mother_id, 'ps_syphilis_treated']
    assert mni[mother_id]['syphilis_infection_origin'] == 'pre_existing'
    assert not pd.isnull(mni[mother_id]['syphilis_stage_onset_date'])


@pytest.mark.parametrize(
    "stage_distribution, expected_stage",
    [
        ([1.0, 0.0, 0.0, 0.0], 'primary'),
        ([0.0, 1.0, 0.0, 0.0], 'secondary'),
        ([0.0, 0.0, 1.0, 0.0], 'early_latent'),
        ([0.0, 0.0, 0.0, 1.0], 'late_latent'),
    ],
)
def test_pre_existing_stage_distribution_maps_to_expected_stage(seed, stage_distribution, expected_stage):
    sim = get_sim(seed)
    df, mni, _, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    syph_module.current_parameters['prob_pre_existing_syphilis_at_pregnancy_start'] = 1.0
    syph_module.current_parameters['pre_existing_syphilis_stage_distribution'] = stage_distribution
    syph_module.initialise_pre_existing_syphilis_for_pregnancy(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == expected_stage
    assert not df.at[mother_id, 'ps_syphilis_treated']
    assert mni[mother_id]['syphilis_infection_origin'] == 'pre_existing'
    if expected_stage == 'late_latent':
        assert pd.isnull(mni[mother_id]['syphilis_next_progression_date'])
    else:
        assert not pd.isnull(mni[mother_id]['syphilis_next_progression_date'])


@pytest.mark.parametrize("stage_distribution", [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def test_invalid_pre_existing_stage_distribution_is_safe_noop(seed, stage_distribution):
    sim = get_sim(seed)
    df, mni, _, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    syph_module.current_parameters['prob_pre_existing_syphilis_at_pregnancy_start'] = 1.0
    syph_module.current_parameters['pre_existing_syphilis_stage_distribution'] = stage_distribution
    syph_module.initialise_pre_existing_syphilis_for_pregnancy(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'none'
    assert mni[mother_id]['syphilis_infection_origin'] == 'none'


def test_incident_syphilis_sets_primary_origin_progression_and_resets_treatment(seed):
    sim = get_sim(seed)
    df, mni, _, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'none'
    df.at[mother_id, 'ps_syphilis_treated'] = True
    mni[mother_id]['pred_syph_infect'] = sim.date

    event = syphilis.SyphilisInPregnancyEvent(syph_module, mother_id)
    event.apply(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'primary'
    assert not df.at[mother_id, 'ps_syphilis_treated']
    assert mni[mother_id]['syphilis_infection_origin'] == 'incident_pregnancy'
    assert mni[mother_id]['syphilis_stage_onset_date'] == sim.date
    assert mni[mother_id]['syphilis_next_progression_date'] == sim.date + pd.Timedelta(
        weeks=syph_module.current_parameters['duration_primary_weeks'])


def test_progression_event_moves_one_valid_stage_and_schedules_next(seed):
    sim = get_sim(seed)
    df, mni, ps, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'primary'
    df.at[mother_id, 'ps_syphilis_treated'] = False
    mni[mother_id]['syphilis_infection_origin'] = 'incident_pregnancy'
    mni[mother_id]['syphilis_stage_onset_date'] = sim.date - pd.Timedelta(
        weeks=syph_module.current_parameters['duration_primary_weeks'])
    mni[mother_id]['syphilis_next_progression_date'] = sim.date

    event = syphilis.SyphilisProgressionEvent(syph_module, mother_id, from_stage='primary', to_stage='secondary')
    event.apply(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'secondary'
    assert ps.mnh_outcome_counter['syphilis_progression_primary_secondary'] == 1
    assert ps.mnh_outcome_counter['syphilis_progression_secondary_latent'] == 0
    assert mni[mother_id]['syphilis_next_progression_date'] == sim.date + pd.Timedelta(
        weeks=syph_module.current_parameters['duration_secondary_weeks'])


def test_progression_event_ignores_invalid_state(seed):
    sim = get_sim(seed)
    df, mni, ps, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'secondary'
    df.at[mother_id, 'ps_syphilis_treated'] = False
    mni[mother_id]['syphilis_infection_origin'] = 'incident_pregnancy'
    mni[mother_id]['syphilis_next_progression_date'] = sim.date

    event = syphilis.SyphilisProgressionEvent(syph_module, mother_id, from_stage='primary', to_stage='secondary')
    event.apply(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'secondary'
    assert ps.mnh_outcome_counter['syphilis_progression_primary_secondary'] == 0


def test_treatment_cures_infection_records_flag_and_blocks_progression(seed):
    sim = get_sim(seed)
    df, mni, ps, mother_id = setup_pregnant_woman_for_syphilis_test(sim)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'primary'
    df.at[mother_id, 'ps_syphilis_treated'] = False
    mni[mother_id]['syphilis_infection_origin'] = 'incident_pregnancy'
    mni[mother_id]['syphilis_stage_onset_date'] = sim.date - pd.Timedelta(
        weeks=syph_module.current_parameters['duration_primary_weeks'])
    mni[mother_id]['syphilis_next_progression_date'] = sim.date

    syph_module.treat_maternal_syphilis(mother_id)
    event = syphilis.SyphilisProgressionEvent(syph_module, mother_id, from_stage='primary', to_stage='secondary')
    event.apply(mother_id)

    assert df.at[mother_id, 'ps_syphilis_state'] == 'none'
    assert df.at[mother_id, 'ps_syphilis_treated']
    assert mni[mother_id]['syphilis_treatment_date'] == sim.date
    assert pd.isnull(mni[mother_id]['syphilis_next_progression_date'])
    assert ps.mnh_outcome_counter['syphilis_progression_primary_secondary'] == 0


def test_vertical_transmission_assigns_current_fetal_infection_state(seed):
    sim = get_sim(seed)
    df, mni, ps, mother_id = setup_pregnant_woman_for_syphilis_test(sim, gestational_age=22)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'primary'
    df.at[mother_id, 'ps_syphilis_treated'] = False
    df.at[mother_id, 'ps_congenital_syphilis'] = False
    syph_module.current_parameters['prob_congenital_transmission_primary'] = 1.0
    syph_module.current_parameters['prob_stillbirth_from_congenital_syphilis'] = 0.0

    syph_module.apply_risk_of_congenital_syphilis(gestation_of_interest=22)

    assert df.at[mother_id, 'ps_congenital_syphilis']
    assert mni[mother_id]['congenital_syphilis_stillbirth_week'] is None
    assert ps.mnh_outcome_counter['congenital_syphilis_infection'] == 1


def test_vertical_transmission_is_not_reapplied_after_fetal_infection(seed):
    sim = get_sim(seed)
    df, _, ps, mother_id = setup_pregnant_woman_for_syphilis_test(sim, gestational_age=22)
    syph_module = sim.modules['Syphilis']

    df.at[mother_id, 'ps_syphilis_state'] = 'primary'
    df.at[mother_id, 'ps_syphilis_treated'] = False
    df.at[mother_id, 'ps_congenital_syphilis'] = True
    syph_module.current_parameters['prob_congenital_transmission_primary'] = 1.0

    syph_module.apply_risk_of_congenital_syphilis(gestation_of_interest=22)

    assert ps.mnh_outcome_counter['congenital_syphilis_infection'] == 0


@pytest.mark.slow
def test_syphilis_integrates_in_large_maternal_perinatal_run(seed, tmpdir):
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={"filename": "log", "directory": tmpdir},
        resourcefilepath=resourcefilepath,
    )
    register_modules(sim)
    sim.make_initial_population(n=1000)

    assert 'Syphilis' in sim.modules
    sim.simulate(end_date=Date(2010, 1, 2))

    check_dtypes(sim)
    df = sim.population.props
    syph_module = sim.modules['Syphilis']
    ps = sim.modules['PregnancySupervisor']

    for property_name in ['ps_syphilis_state', 'ps_syphilis_treated', 'ps_congenital_syphilis']:
        assert property_name in df.columns
    assert syph_module.current_parameters
    for counter_name in [
        'syphilis',
        'congenital_syphilis_infection',
        'congenital_syphilis_stillbirth',
        'congenital_syphilis_live_birth',
    ]:
        assert counter_name in ps.mnh_outcome_counter
        assert ps.mnh_outcome_counter[counter_name] >= 0
