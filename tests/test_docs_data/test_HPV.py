import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation, logging
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hpv,
    hiv,
    simplified_births,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    resourcefilepath = "resources"

log_config = {
    "filename": "hpv_test",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs/",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.hpv": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}

HPV_GROUPS = hpv.HPV.HPV_GROUPS
DATE_COLS = [f"hp_date_infected_{group}" for group in HPV_GROUPS]

def make_sim(seed,log_config=None):
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        healthsystem.HealthSystem(
            disable=True,  # disables the health system constraints so all HSI events run
        ),
        epi.Epi(),
        hiv.Hiv(),
        hpv.HPV(),
    )

    return sim

@pytest.fixture
def sim(seed):
    return make_sim(seed=seed, log_config=None)

def check_dtypes(simulation):
    """
    Check that population property dtypes remain consistent.
    """
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def get_any_hpv_infected(df):
    """Return whether each person is infected with any HPV group."""
    return df[DATE_COLS].notna().any(axis=1)


def assert_hpv_state_consistent(simulation):
    """Check that HPV date-based states, durations, and persistence flags are internally consistent."""

    module = simulation.modules["HPV"]
    df = simulation.population.props

    module._update_persistence_status()

    eligible = df.is_alive & (df.age_years >= 15)

    for group in HPV_GROUPS:
        date_col = f"hp_date_infected_{group}"
        dur_col = f"hp_duration_{group}"
        pers_col = f"hp_persistent_{group}"

        infected = df[date_col].notna()
        non_infected = df[date_col].isna()

        # Non-infected people should not be persistent
        assert not df.loc[non_infected, pers_col].fillna(False).any()

        # Eligible infected people should have non-negative duration
        assert (df.loc[eligible & infected, dur_col] >= 0).all()

        # Eligible non-infected people should have duration -1.0
        assert (df.loc[eligible & non_infected, dur_col] == -1.0).all()

        # Persistent people must also be infected
        assert df.loc[df[pers_col].fillna(False), date_col].notna().all()

        # Persistent status should match duration threshold
        threshold = module.parameters["persistent_threshold_months"]
        expected_persistent = eligible & infected & (df[dur_col] >= threshold)
        observed_persistent = df[pers_col].fillna(False)

        assert (observed_persistent == expected_persistent).all()


def test_hpv_initial_population_date_based_state(sim):

    module = sim.modules["HPV"]
    module.parameters["init_prev_hpv_hr1"] = 1.0
    module.parameters["init_prev_hpv_hr2"] = 0.0
    module.parameters["init_prev_hpv_hr3"] = 0.0

    sim.make_initial_population(n=500)
    df = sim.population.props

    eligible = df.is_alive & (df.age_years >= 15)
    under15 = df.is_alive & (df.age_years < 15)

    # All eligible people should be infected with hr1
    assert df.loc[eligible, "hp_date_infected_hr1"].notna().all()

    # Nobody should be infected with hr2/hr3
    assert df.loc[df.is_alive, "hp_date_infected_hr2"].isna().all()
    assert df.loc[df.is_alive, "hp_date_infected_hr3"].isna().all()

    # Under-15 people should not be infected
    assert df.loc[under15, "hp_date_infected_hr1"].isna().all()

    # Initial infection durations should be between 0 and 23 months
    assert (df.loc[eligible, "hp_duration_hr1"] >= 0).all()
    assert (df.loc[eligible, "hp_duration_hr1"] < 24).all()

    # Persistent status should match duration threshold
    threshold = module.parameters["persistent_threshold_months"]
    expected_persistent = df.loc[eligible, "hp_duration_hr1"] >= threshold
    observed_persistent = df.loc[eligible, "hp_persistent_hr1"]

    assert (observed_persistent == expected_persistent).all()

    check_dtypes(sim)

def test_hpv_add_and_clear_single_group(sim):

    module = sim.modules["HPV"]
    module.parameters["init_prev_hpv_hr1"] = 0.0
    module.parameters["init_prev_hpv_hr2"] = 0.0
    module.parameters["init_prev_hpv_hr3"] = 0.0

    sim.make_initial_population(n=500)
    df = sim.population.props

    eligible_idx = df.index[df.is_alive & (df.age_years >= 15)]

    if len(eligible_idx) == 0:
        pytest.skip("No eligible person aged 15+ in the test population.")

    person_id = eligible_idx[0]

    # Initially not infected
    assert not module._hp_is_infected(person_id)

    # Add hr1 infection
    module._add_new_infection_groups(person_id, {"hr1"})

    assert module._is_group_infected(person_id, "hr1")
    assert df.at[person_id, "hp_date_infected_hr1"] == sim.date
    assert df.at[person_id, "hp_duration_hr1"] == 0
    assert not df.at[person_id, "hp_persistent_hr1"]

    # Clear hr1 infection
    module._clear_single_group(person_id, "hr1")

    assert not module._is_group_infected(person_id, "hr1")
    assert pd.isna(df.at[person_id, "hp_date_infected_hr1"])
    assert df.at[person_id, "hp_duration_hr1"] == -1
    assert not df.at[person_id, "hp_persistent_hr1"]


def test_hpv_clearance_probability_in_valid_range(sim):

    module = sim.modules["HPV"]

    module.parameters["init_prev_hpv_hr1"] = 0.0
    module.parameters["init_prev_hpv_hr2"] = 0.0
    module.parameters["init_prev_hpv_hr3"] = 0.0

    sim.make_initial_population(n=500)
    df = sim.population.props

    eligible_idx = df.index[df.is_alive & (df.age_years >= 15)]

    if len(eligible_idx) == 0:
        pytest.skip("No eligible person aged 15+ in the test population.")

    person_id = eligible_idx[0]

    p_clear = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    assert 0.0 <= p_clear <= 1.0


def test_hiv_can_modify_hpv_clearance_probability(sim):

    module = sim.modules["HPV"]

    module.parameters["init_prev_hpv_hr1"] = 0.0
    module.parameters["init_prev_hpv_hr2"] = 0.0
    module.parameters["init_prev_hpv_hr3"] = 0.0
    module.parameters["rr_clear_hiv_no_art"] = 0.5

    sim.make_initial_population(n=500)
    df = sim.population.props

    eligible_idx = df.index[df.is_alive & (df.age_years >= 15)]

    if len(eligible_idx) == 0:
        pytest.skip("No eligible person aged 15+ in the test population.")

    person_id = eligible_idx[0]

    # Baseline: HIV negative
    if "hv_inf" in df.columns:
        df.at[person_id, "hv_inf"] = False

    p_clear_baseline = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    # HIV positive and not on ART / not virally suppressed
    if "hv_inf" not in df.columns:
        pytest.skip("HIV infection column hv_inf not available.")

    df.at[person_id, "hv_inf"] = True

    if "hv_art" in df.columns:
        df.at[person_id, "hv_art"] = "not"

    p_clear_hiv = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    assert 0.0 <= p_clear_hiv <= 1.0
    assert p_clear_hiv <= p_clear_baseline


@pytest.mark.slow
def test_hpv_simulation_runs_and_states_remain_consistent(sim):

    module = sim.modules["HPV"]

    module.parameters["init_prev_hpv_hr1"] = 0.2
    module.parameters["init_prev_hpv_hr2"] = 0.2
    module.parameters["init_prev_hpv_hr3"] = 0.2

    popsize = 1000
    end_date = Date(2011, 1, 1)

    sim.make_initial_population(n=popsize)
    check_dtypes(sim)

    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    assert_hpv_state_consistent(sim)

    df = sim.population.props
    alive_15plus = df.is_alive & (df.age_years >= 15)

    any_hpv = get_any_hpv_infected(df)
    total_prev = any_hpv.loc[alive_15plus].mean()

    assert 0.0 <= total_prev <= 1.0


@pytest.mark.slow
def test_hpv_logging_columns_are_consistent(seed, tmp_path):

    test_log_config = {
        "filename": "hpv_test",
        "directory": tmp_path,
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.hpv": logging.INFO,
        },
    }

    sim = make_sim(seed=seed, log_config=test_log_config)

    module = sim.modules["HPV"]
    module.parameters["init_prev_hpv_hr1"] = 0.2
    module.parameters["init_prev_hpv_hr2"] = 0.2
    module.parameters["init_prev_hpv_hr3"] = 0.2

    sim.make_initial_population(n=1000)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        sim.simulate(end_date=Date(2012, 1, 1))

    inconsistent_column_warnings = [
        w for w in caught_warnings
        if (
            "InconsistentLoggedColumnsWarning" in w.category.__name__
            or "Inconsistent columns in logged values" in str(w.message)
        )
    ]

    assert len(inconsistent_column_warnings) == 0

