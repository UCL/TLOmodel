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
    tb,
)
from tlo.analysis.utils import parse_log_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
resourcefilepath = PROJECT_ROOT / "resources"

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

HPV_GROUPS = tuple(hpv.HPV.HPV_GROUPS)
DATE_COLS = tuple(f"hp_date_infected_{group}" for group in HPV_GROUPS)
DURATION_COLS = tuple(f"hp_duration_{group}" for group in HPV_GROUPS)
PERSISTENT_COLS = tuple(f"hp_persistent_{group}" for group in HPV_GROUPS)
INITIAL_PREVALENCE_PREFIXES = ("hiv_neg", "hiv_pos")

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
        tb.Tb(),
    )

    return sim

@pytest.fixture
def seed():
    return 12345

@pytest.fixture
def sim(seed):
    return make_sim(seed=seed, log_config=None)

def set_initial_prevalence(module, prevalence_by_group=None, default=0.0):

    prevalence_by_group = prevalence_by_group or {}

    for group in HPV_GROUPS:
        prevalence = float(prevalence_by_group.get(group, default))

        for prefix in INITIAL_PREVALENCE_PREFIXES:
            parameter_name = f"{prefix}_init_prev_hpv_{group}"
            module.parameters[parameter_name] = prevalence


def initialise_simulation(simulation, population_size):
    """Create the population and run zero days to initialise scheduled events."""

    simulation.make_initial_population(n=population_size)

    simulation.simulate(
        end_date=simulation.date + pd.DateOffset(days=0)
    )

    return simulation.population.props


def get_eligible_ids(df, number=1):
    """Return IDs of alive people aged 15 years or older."""

    eligible_ids = df.index[df.is_alive & (df.age_years >= 15)]

    assert len(eligible_ids) >= number, (
        f"The test needs at least {number} alive people aged 15+, "
        f"but only {len(eligible_ids)} were generated."
    )

    return eligible_ids[:number].tolist()


def reset_hpv_state(df, person_ids=None):
    """Reset selected people to a completely HPV-uninfected state."""

    if person_ids is None:
        person_ids = df.index

    for group in HPV_GROUPS:
        df.loc[person_ids, f"hp_date_infected_{group}"] = pd.NaT
        df.loc[person_ids, f"hp_duration_{group}"] = -1.0
        df.loc[person_ids, f"hp_persistent_{group}"] = False

    df.loc[person_ids, "hp_duration_all_clear"] = -1.0

def set_group_infection(
    simulation,
    person_id,
    group,
    infection_date=None,
):
    """Assign one HPV group infection directly for a deterministic test scenario."""

    assert group in HPV_GROUPS

    df = simulation.population.props

    if infection_date is None:
        infection_date = simulation.date

    df.at[person_id, f"hp_date_infected_{group}"] = infection_date
    df.at[person_id, f"hp_duration_{group}"] = (
        max(0.0, (simulation.date - infection_date).days / 30.5)
    )
    df.at[person_id, f"hp_persistent_{group}"] = False


def get_any_hpv_infected(df):
    """Return whether each person is infected with any HPV group."""

    return (
        df.loc[:, list(DATE_COLS)]
        .notna()
        .any(axis=1)
    )

def check_dtypes(simulation):
    """Check that population property dtypes remain consistent."""
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def assert_hpv_state_consistent(simulation):
    """Check that HPV dates, durations, and persistence flags agree."""

    module = simulation.modules["HPV"]
    df = simulation.population.props

    eligible = df.is_alive & (df.age_years >= 15)
    threshold = float(
        module.parameters["persistent_threshold_months"]
    )

    for group in HPV_GROUPS:
        date_col = f"hp_date_infected_{group}"
        duration_col = f"hp_duration_{group}"
        persistent_col = f"hp_persistent_{group}"

        infected = df[date_col].notna()
        non_infected = ~infected

        observed_persistent = (
            df[persistent_col]
            .fillna(False)
            .astype(bool)
        )

        # 未感染者不能被标记为持续感染
        assert not observed_persistent.loc[
            non_infected
        ].any()

        # 符合条件的感染者持续时间必须非负
        assert (
            df.loc[
                eligible & infected,
                duration_col,
            ] >= 0.0
        ).all()

        # 符合条件的未感染者持续时间应为 -1
        assert (
            df.loc[
                eligible & non_infected,
                duration_col,
            ] == -1.0
        ).all()

        # 持续感染者必须仍然处于感染状态
        assert df.loc[
            observed_persistent,
            date_col,
        ].notna().all()

        # 持续感染标记应符合持续时间阈值
        expected_persistent = (
            infected
            & (df[duration_col] >= threshold)
        )

        mismatch = (
            observed_persistent.loc[eligible]
            != expected_persistent.loc[eligible]
        )

        if mismatch.any():
            mismatch_ids = mismatch.index[mismatch]

            diagnostic_columns = [
                "is_alive",
                "age_years",
                date_col,
                duration_col,
                persistent_col,
            ]

            pytest.fail(
                f"Persistence mismatch for {group}:\n"
                f"{df.loc[mismatch_ids, diagnostic_columns]}"
            )

# 5. Module contract and initialisation tests
def test_hpv_group_contract():
    """The test suite should follow the HPV groups declared by the module."""
    assert HPV_GROUPS == ("hr1", "hr2", "hr3", "hr4", "hr5", "hr6")

def test_hpv_required_property_columns_exist(sim):
    """All date, duration and persistence properties should exist after population creation."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=100)

    expected_columns = {
        *DATE_COLS,
        *DURATION_COLS,
        *PERSISTENT_COLS,
        "hp_duration_all_clear",
    }

    assert expected_columns.issubset(df.columns)

def test_hpv_initial_population_date_based_state(sim):
    module = sim.modules["HPV"]

    set_initial_prevalence(
        module,
        prevalence_by_group={"hr1": 1.0},
    )

    sim.make_initial_population(n=500)
    df = sim.population.props

    eligible = df.is_alive & (df.age_years >= 15)
    under15 = df.is_alive & (df.age_years < 15)

    # All eligible people should be infected with hr1
    assert df.loc[eligible, "hp_date_infected_hr1"].notna().all()

    # Nobody should be infected with hr2/hr3
    for group in HPV_GROUPS:
        if group == "hr1":
            continue

        assert df.loc[
            df.is_alive,
            f"hp_date_infected_{group}",
        ].isna().all()

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

def test_initial_prevalence_is_applied_by_group_and_age(sim):

    module = sim.modules["HPV"]

    set_initial_prevalence(module,
        prevalence_by_group={"hr1": 1.0},
        default=0.0,
    )

    df = initialise_simulation(sim, population_size=500)

    eligible = df.is_alive & (df.age_years >= 15)
    under_15 = df.is_alive & (df.age_years < 15)

    # Every eligible person should have hr1 because both HIV strata are set to 1.
    assert df.loc[eligible, "hp_date_infected_hr1"].notna().all()

    # All other groups were configured with zero initial prevalence.
    for group in HPV_GROUPS:
        if group == "hr1":
            continue

        assert df.loc[df.is_alive, f"hp_date_infected_{group}"].isna().all()

    # People younger than 15 should not be initialised with HPV.
    assert df.loc[under_15, list(DATE_COLS)].isna().all().all()

    # Initial infections are sampled from 0 to 23 completed months before start.
    assert (
        df.loc[eligible, "hp_duration_hr1"] >= 0.0
    ).all()
    assert (
        df.loc[eligible, "hp_duration_hr1"] < 24.0
    ).all()

    assert_hpv_state_consistent(sim)
    check_dtypes(sim)

def test_hpv_on_birth_resets_all_hpv_properties(sim):
    """A newborn should begin without an HPV infection or persistence state."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=10)

    mother_id = df.index[0]
    child_id = df.index[1]

    # Deliberately contaminate the future child row before calling on_birth.
    for group in HPV_GROUPS:
        set_group_infection(sim, child_id, group)

    df.at[child_id, "hp_duration_all_clear"] = 20.0

    module.on_birth(mother_id=mother_id, child_id=child_id)

    assert df.loc[child_id, list(DATE_COLS)].isna().all()
    assert (df.loc[child_id, list(DURATION_COLS)] == -1.0).all()
    assert not df.loc[child_id, list(PERSISTENT_COLS)].any()
    assert df.at[child_id, "hp_duration_all_clear"] == -1.0

def test_hpv_add_and_clear_single_group(sim):
    """Adding and clearing one group should update all related properties."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    reset_hpv_state(df, [person_id])

    assert not module._hp_is_infected(person_id)

    module._add_new_infection_groups(person_id, {"hr1"})

    assert module._is_group_infected(person_id, "hr1")
    assert df.at[person_id, "hp_date_infected_hr1"] == sim.date
    assert df.at[person_id, "hp_duration_hr1"] == 0.0
    assert not df.at[person_id, "hp_persistent_hr1"]
    assert df.at[person_id, "hp_duration_all_clear"] == -1.0

    module._clear_single_group(person_id, "hr1")

    assert not module._is_group_infected(person_id, "hr1")
    assert pd.isna(df.at[person_id, "hp_date_infected_hr1"])
    assert df.at[person_id, "hp_duration_hr1"] == -1.0
    assert not df.at[person_id, "hp_persistent_hr1"]
    assert df.at[person_id, "hp_duration_all_clear"] >= 0.0

def test_clearing_one_group_does_not_clear_other_groups(sim):
    """Clearing hr1 must not accidentally clear a concurrent hr2 infection."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    reset_hpv_state(df, [person_id])
    module._add_new_infection_groups(person_id, {"hr1", "hr2"})

    module._clear_single_group(person_id, "hr1")

    assert not module._is_group_infected(person_id, "hr1")
    assert module._is_group_infected(person_id, "hr2")
    assert df.at[person_id, "hp_duration_all_clear"] == -1.0

def test_duplicate_add_does_not_reset_existing_infection_date(sim):
    """Re-adding an already infected group should leave its original date unchanged."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    reset_hpv_state(df, [person_id])

    original_date = sim.date - pd.DateOffset(months=6)
    set_group_infection(sim, person_id, "hr1", original_date)

    module._add_new_infection_groups(person_id, {"hr1"})

    assert df.at[person_id, "hp_date_infected_hr1"] == original_date

@pytest.mark.parametrize(
    ("days_ago", "expected_persistent"),
    [
        (365, False),
        (366, True),
        (367, True),
    ],
)

def test_persistence_threshold_boundary(
    sim,
    days_ago,
    expected_persistent,
):
    """Persistence should switch on at the configured 12-month boundary."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)
    module.parameters["persistent_threshold_months"] = 12.0

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    reset_hpv_state(df, [person_id])

    infection_date = sim.date - pd.DateOffset(days=days_ago)
    set_group_infection(sim, person_id, "hr1", infection_date)

    module._update_persistence_status()

    expected_duration = days_ago / 30.5

    assert df.at[person_id, "hp_duration_hr1"] == pytest.approx(
        expected_duration
    )
    assert bool(df.at[person_id, "hp_persistent_hr1"]) is expected_persistent

def test_clearance_probability_is_valid_and_zero_for_zero_interval(sim):
    """Clearance probability should be bounded and equal zero over no elapsed interval."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    p_six_months = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    p_zero_interval = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=0.0,
    )

    assert 0.0 <= p_six_months <= 1.0
    assert p_zero_interval == pytest.approx(0.0)

def test_longer_interval_has_at_least_as_much_clearance_probability(sim):
    """At a fixed end time, a longer retrospective interval contains more cumulative hazard."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    p_one_month = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=1.0,
    )

    p_six_months = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    assert p_six_months >= p_one_month

def test_hiv_and_art_modify_hpv_clearance_probability(sim):
    """Untreated/unsuppressed HIV should reduce clearance; suppression restores baseline."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)
    module.parameters["rr_clear_hiv_no_art"] = 0.5

    df = initialise_simulation(sim, population_size=500)
    person_id = get_eligible_ids(df, number=1)[0]

    # [修改-6] hv_inf and hv_art are core interfaces because HPV declares Hiv as a dependency.
    assert "hv_inf" in df.columns
    assert "hv_art" in df.columns

    df.at[person_id, "hv_inf"] = False

    p_hiv_negative = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_art"] = "not"

    p_hiv_no_art = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    df.at[person_id, "hv_art"] = "on_not_VL_suppressed"

    p_hiv_unsuppressed = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    df.at[person_id, "hv_art"] = "on_VL_suppressed"

    p_hiv_suppressed = module._get_clearance_probability(
        group="hr1",
        person_id=person_id,
        duration_months=12.0,
        interval_months=6.0,
    )

    assert p_hiv_no_art < p_hiv_negative
    assert p_hiv_unsuppressed == pytest.approx(p_hiv_no_art)
    assert p_hiv_suppressed == pytest.approx(p_hiv_negative)

def test_infection_rr_for_age_vaccination_and_hiv(sim):
    """Check the direction and multiplication of HPV acquisition modifiers."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    module.parameters["rr_hpv_age50plus"] = 0.7
    module.parameters["rr_hr1_vaccinated"] = 0.2
    module.parameters["rr_hpv_hiv_no_art"] = 1.5

    df = initialise_simulation(
        sim,
        population_size=500,
    )

    person_id = get_eligible_ids(
        df,
        number=1,
    )[0]

    # These columns are required by this test.
    assert "hv_inf" in df.columns
    assert "hv_art" in df.columns
    assert "va_hpv" in df.columns

    # Baseline: age 25, HIV negative, unvaccinated.
    df.at[person_id, "age_years"] = 25
    df.at[person_id, "hv_inf"] = False
    df.at[person_id, "va_hpv"] = 0

    baseline_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert baseline_rr == pytest.approx(1.0)

    # Age 50+ effect.
    df.at[person_id, "age_years"] = 50

    age_50_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert age_50_rr == pytest.approx(0.7)

    # Restore baseline age.
    df.at[person_id, "age_years"] = 25

    # HPV vaccination effect.
    df.at[person_id, "va_hpv"] = 1

    vaccinated_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert vaccinated_rr == pytest.approx(0.2)

    # Restore unvaccinated state before testing HIV.
    df.at[person_id, "va_hpv"] = 0

    # HIV positive and not receiving ART.
    df.at[person_id, "hv_inf"] = True
    df.at[person_id, "hv_art"] = "not"

    hiv_no_art_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert hiv_no_art_rr == pytest.approx(1.5)

    # HIV positive with viral suppression.
    df.at[person_id, "hv_art"] = "on_VL_suppressed"

    suppressed_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert suppressed_rr == pytest.approx(1.0)

    # People younger than 15 should not acquire HPV.
    df.at[person_id, "age_years"] = 14

    underage_rr = module._get_infection_rr(
        person_id,
        "hr1",
    )

    assert underage_rr == pytest.approx(0.0)

def test_age_mixing_matrix_rows_sum_to_one(sim):
    """Every recipient age group's mixing weights should form a probability distribution."""

    module = sim.modules["HPV"]
    set_initial_prevalence(module)

    initialise_simulation(sim, population_size=100)

    matrix = module.age_mixing_matrix

    assert list(matrix.index) == list(module.AGE_LABELS)
    assert list(matrix.columns) == list(module.AGE_LABELS)
    assert (matrix >= 0.0).all().all()
    assert matrix.sum(axis=1).to_numpy() == pytest.approx(
        [1.0] * len(module.AGE_LABELS)
    )

def test_invalid_age_mixing_parameters_raise_value_error(sim):
    """Age-mixing proportions that do not sum to one should be rejected."""

    module = sim.modules["HPV"]

    with pytest.raises(ValueError, match="must sum to 1.0"):
        module._build_age_mixing_matrix(
            within=0.5,
            adjacent=0.3,
            distant=0.3,
        )

def prepare_event_test_population(simulation, population_size=500):
    """Initialise a zero-prevalence population for deterministic event tests."""

    module = simulation.modules["HPV"]

    set_initial_prevalence(module)

    # Use only within-age-group mixing so a same-age source-target pair is deterministic.
    module.parameters["age_mixing_within"] = 1.0
    module.parameters["age_mixing_adjacent"] = 0.0
    module.parameters["age_mixing_distant"] = 0.0

    df = initialise_simulation(simulation, population_size)

    reset_hpv_state(df)

    return module, df

def test_infection_event_can_force_clearance(sim, monkeypatch):
    """When clearance probability is forced to one, an existing infection must clear."""

    module, df = prepare_event_test_population(sim)
    person_id = get_eligible_ids(df, number=1)[0]

    df.at[person_id, "age_years"] = 25
    set_group_infection(
        sim,
        person_id,
        "hr1",
        sim.date - pd.DateOffset(months=6),
    )

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 1.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    assert not module._is_group_infected(person_id, "hr1")
    assert module._last_event_counts["Clear_hr1"] >= 1
    assert module._last_event_counts["Clear_Total"] >= 1

def test_infection_event_can_force_no_clearance(sim, monkeypatch):
    """When clearance probability is forced to zero, infection must remain."""

    module, df = prepare_event_test_population(sim)
    person_id = get_eligible_ids(df, number=1)[0]

    df.at[person_id, "age_years"] = 25
    set_group_infection(
        sim,
        person_id,
        "hr1",
        sim.date - pd.DateOffset(months=6),
    )

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 0.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    assert module._is_group_infected(person_id, "hr1")
    assert module._last_event_counts["Clear_hr1"] == 0

def test_infection_event_transmits_from_male_to_female(sim, monkeypatch):
    """High transmission with positive male prevalence should infect a susceptible female."""

    module, df = prepare_event_test_population(sim)
    source_id, target_id = get_eligible_ids(df, number=2)

    df.at[source_id, "sex"] = "M"
    df.at[source_id, "age_years"] = 25
    df.at[target_id, "sex"] = "F"
    df.at[target_id, "age_years"] = 25

    df.at[target_id, "hv_inf"] = False

    if "va_hpv" in df.columns:
        df.at[target_id, "va_hpv"] = 0

    set_group_infection(
        sim,
        source_id,
        "hr1",
        sim.date - pd.DateOffset(months=1),
    )

    module.parameters["b_hpv"] = 1.0e9

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 0.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    assert module._is_group_infected(target_id, "hr1")
    assert module._last_event_counts["NewInf_hr1"] >= 1
    assert module._last_event_counts["NewInf_F"] >= 1

def test_infection_event_does_not_transmit_when_beta_is_zero(
    sim,
    monkeypatch,
):
    """A zero transmission coefficient should prevent all new infections."""

    module, df = prepare_event_test_population(sim)
    source_id, target_id = get_eligible_ids(df, number=2)

    df.at[source_id, "sex"] = "M"
    df.at[source_id, "age_years"] = 25
    df.at[target_id, "sex"] = "F"
    df.at[target_id, "age_years"] = 25
    df.at[target_id, "hv_inf"] = False

    if "va_hpv" in df.columns:
        df.at[target_id, "va_hpv"] = 0

    set_group_infection(
        sim,
        source_id,
        "hr1",
        sim.date - pd.DateOffset(months=1),
    )

    module.parameters["b_hpv"] = 0.0

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 0.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    assert not module._is_group_infected(target_id, "hr1")
    assert module._last_event_counts["NewInf_Total"] == 0

def test_infection_event_excludes_underage_and_dead_people(
    sim,
    monkeypatch,
):
    """People younger than 15 or not alive should not acquire infection in the event."""

    module, df = prepare_event_test_population(sim)
    source_id, underage_id, dead_id = get_eligible_ids(df, number=3)

    df.at[source_id, "sex"] = "M"
    df.at[source_id, "age_years"] = 25

    df.at[underage_id, "sex"] = "F"
    df.at[underage_id, "age_years"] = 14
    df.at[underage_id, "is_alive"] = True

    df.at[dead_id, "sex"] = "F"
    df.at[dead_id, "age_years"] = 25
    df.at[dead_id, "is_alive"] = False

    set_group_infection(
        sim,
        source_id,
        "hr1",
        sim.date - pd.DateOffset(months=1),
    )

    module.parameters["b_hpv"] = 1.0e9

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 0.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    assert not module._is_group_infected(underage_id, "hr1")
    assert not module._is_group_infected(dead_id, "hr1")

def test_event_count_totals_match_group_counts(sim, monkeypatch):
    """Total event counters should equal the sum of their group-specific counters."""

    module, df = prepare_event_test_population(sim)
    source_id, target_id = get_eligible_ids(df, number=2)

    df.at[source_id, "sex"] = "M"
    df.at[source_id, "age_years"] = 25
    df.at[target_id, "sex"] = "F"
    df.at[target_id, "age_years"] = 25
    df.at[target_id, "hv_inf"] = False

    if "va_hpv" in df.columns:
        df.at[target_id, "va_hpv"] = 0

    set_group_infection(sim, source_id, "hr1")
    module.parameters["b_hpv"] = 1.0e9

    monkeypatch.setattr(
        module,
        "_get_clearance_probability",
        lambda **kwargs: 0.0,
    )

    event = hpv.HpvInfectionEvent(
        module=module,
        frequency_months=6,
    )
    event.apply(sim.population)

    counts = module._last_event_counts

    new_infection_group_sum = sum(
        counts[f"NewInf_{group}"]
        for group in HPV_GROUPS
    )
    clearance_group_sum = sum(
        counts[f"Clear_{group}"]
        for group in HPV_GROUPS
    )

    assert counts["NewInf_Total"] == new_infection_group_sum
    assert counts["Clear_Total"] == clearance_group_sum

@pytest.mark.slow
def test_hpv_simulation_runs_and_states_remain_consistent(sim):
    """Run the default event machinery and check end-state invariants."""

    module = sim.modules["HPV"]

    set_initial_prevalence(
        module,
        prevalence_by_group={
            group: 0.10
            for group in HPV_GROUPS
        },
    )

    sim.make_initial_population(n=1000)
    check_dtypes(sim)

    sim.simulate(end_date=Date(2011, 1, 1))

    check_dtypes(sim)
    assert_hpv_state_consistent(sim)

    df = sim.population.props
    alive_15_plus = df.is_alive & (df.age_years >= 15)
    any_hpv = get_any_hpv_infected(df)

    total_prevalence = any_hpv.loc[alive_15_plus].mean()

    assert pd.notna(total_prevalence)
    assert 0.0 <= total_prevalence <= 1.0

    # At least one scheduled HpvInfectionEvent should have populated these counters.
    assert "NewInf_Total" in module._last_event_counts
    assert "Clear_Total" in module._last_event_counts


@pytest.mark.slow
def test_hpv_logging_columns_and_required_content_are_consistent(
    seed,
    tmp_path,
):
    """HPV logging should keep a stable schema and contain core summary columns."""

    test_log_config = {
        "filename": "hpv_test",
        "directory": tmp_path,
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.hpv": logging.INFO,
        },
    }

    simulation = make_sim(
        seed=seed,
        log_config=test_log_config,
    )

    module = simulation.modules["HPV"]

    set_initial_prevalence(
        module,
        prevalence_by_group={
            group: 0.10
            for group in HPV_GROUPS
        },
    )

    simulation.make_initial_population(n=1000)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        simulation.simulate(end_date=Date(2012, 1, 1))

    inconsistent_column_warnings = [
        warning
        for warning in caught_warnings
        if (
            "InconsistentLoggedColumnsWarning"
            in warning.category.__name__
            or "Inconsistent columns in logged values"
            in str(warning.message)
        )
    ]

    assert len(inconsistent_column_warnings) == 0

    parsed_logs = parse_log_file(
        simulation.log_filepath,
        level=logging.INFO,
    )

    hpv_summary = parsed_logs["tlo.methods.hpv"]["summary"]

    required_log_columns = {
        "EligibleN",
        "TotalInf",
        "TotalPrev",
        "NewInf_Total",
        "Clear_Total",
    }

    assert required_log_columns.issubset(hpv_summary.columns)

    non_missing_prevalence = hpv_summary["TotalPrev"].dropna()

    assert not non_missing_prevalence.empty
    assert non_missing_prevalence.between(0.0, 1.0).all()
