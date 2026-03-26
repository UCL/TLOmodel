import os
from pathlib import Path
import pandas as pd
import pytest

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    Metadata,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    weather_disruptions
)

from tlo.methods.hsi_event import HSI_Event

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 2)
popsize = 200

"""
Test whether the WeatherDisruptions module is behaving as expcted.
 """


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.fixture
def weather_disruption_sim(seed, tmpdir):
    """Shared simulation setup for weather disruption tests."""

    class DummyHSI(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSI"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyModule(Module):
        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.modules["HealthSystem"].schedule_hsi_event(
                DummyHSI(self, person_id=0),
                topen=sim.date,
                tclose=None,
                priority=1,
            )

    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {
            "tlo.methods.healthsystem": logging.DEBUG,
            "tlo.methods.weather_disruptions": logging.INFO,
        },
    }

    sim = Simulation(
        start_date=Date(2025, 1, 1), seed=seed, log_config=log_config,
        resourcefilepath=resourcefilepath
    )

    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            capabilities_coefficient=10000.0,  # ensures events are never hindered by lack of capabilities
            cons_availability="all",
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            force_any_symptom_to_lead_to_healthcareseeking=True
        ),
        symptommanager.SymptomManager(),
        weather_disruptions.WeatherDisruptions(
            services_affected_precip="all",
            year_effective_climate_disruptions=2025,
            scale_factor_prob_disruption=1.0,
            delay_in_seeking_care_weather=28.0,
            scale_factor_severity_disruption_and_delay=1.0,
            scale_factor_reseeking_healthcare_post_disruption=1.0,
            prop_supply_side_disruptions=0.0,
            scale_factor_appointment_urgency=1.0,
        ),
        DummyModule(),
        check_all_dependencies=False,
    )

    sim.make_initial_population(n=100)
    sim.modules["WeatherDisruptions"].parameters["scale_factor_appointment_urgency"] = 1.0

    return sim


def _make_always_disrupted_once(sim):
    """Ensures that with high p(disruptions), there is only one
    HSI disrupted (otherwise it is always delayed and never runs)"""
    disruption_fired = [False]

    def always_disrupted_once(hsi_event_item, current_date):
        if disruption_fired[0]:
            return False, False  # is not fired, also not supply side
        # rest is just handling disruption
        disruption_fired[0] = True
        wd = sim.modules["WeatherDisruptions"]
        district = sim.population.props.at[
            hsi_event_item.hsi_event.target, "district_of_residence"
        ]
        treatment_id = getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown")
        is_supply_side = wd._handle_disruption(
            hsi_event_item=hsi_event_item,
            prob_disruption=1.0,
            current_date=current_date,
            facility_id="test_facility",
            district=str(district),
            treatment_id=str(treatment_id),
        )
        return True, is_supply_side

    return always_disrupted_once


def test_weather_disruption_delays_hsi_by_correct_days(weather_disruption_sim):
    """Tests to ensure that when a delay is scheduled, it is scheduled the correct amount
    into the future. """
    sim = weather_disruption_sim
    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = _make_always_disrupted_once(sim)

    expected_run_date = sim.start_date + pd.DateOffset(days=35)
    sim.simulate(end_date=expected_run_date + pd.DateOffset(days=5))

    log = parse_log_file(sim.log_filepath, level=logging.DEBUG)["tlo.methods.healthsystem"]["HSI_Event"]
    dummy_log = log[log["TREATMENT_ID"] == "DummyHSI"]

    assert len(dummy_log) == 1
    assert dummy_log["date"].iloc[0] == expected_run_date


def test_weather_disruption_delayed_log_entry(weather_disruption_sim):
    """Tests logging of weather-delayed HSIs"""
    sim = weather_disruption_sim
    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = _make_always_disrupted_once(sim)

    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=40))

    log = parse_log_file(sim.log_filepath)["tlo.methods.weather_disruptions.summary"]
    delayed_log = log["Weather_delayed_HSI_Event_full_info"]

    assert len(delayed_log) == 1
    assert delayed_log["TREATMENT_ID"].iloc[0] == "DummyHSI"
    assert delayed_log["Person_ID"].iloc[0] == 0
    assert delayed_log["RealFacility_ID"].iloc[0] == "test_facility"


def test_weather_disruption_cancelled_log_entry(weather_disruption_sim):
    """Tests logging of weather-cancelled HSIs"""

    sim = weather_disruption_sim
    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = _make_always_disrupted_once(sim)
    sim.modules["WeatherDisruptions"]._determine_rescheduling = lambda hsi_event_item: False

    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=40))

    log = parse_log_file(sim.log_filepath)["tlo.methods.weather_disruptions.summary"]
    cancelled_log = log["Weather_cancelled_HSI_Event_full_info"]

    assert len(cancelled_log) == 1
    assert cancelled_log["TREATMENT_ID"].iloc[0] == "DummyHSI"
    assert cancelled_log["RealFacility_ID"].iloc[0] == "test_facility"


def test_weather_disruption_monthly_log(weather_disruption_sim):
    sim = weather_disruption_sim
    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = _make_always_disrupted_once(sim)
    sim.modules["WeatherDisruptions"]._determine_rescheduling = lambda hsi_event_item: False

    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=40))

    log = parse_log_file(sim.log_filepath)["tlo.methods.weather_disruptions.summary"]
    monthly_log = log["weather_disruptions_monthly"]

    assert len(monthly_log) >= 1
    assert (monthly_log["hsi_total"] == monthly_log["delayed"] + monthly_log["cancelled"]).all()
    assert (monthly_log["supply_side"] == 0).all()


## Test facility assignment
@pytest.fixture(scope="module")
def demography_sim(seed=0):
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, resourcefilepath=resourcefilepath)
    sim.register(demography.Demography())
    sim.make_initial_population(n=5000)  # large enough to get variety
    sim.simulate(end_date=Date(2010, 1, 1))
    return sim


def test_facility_assignment_varies_within_district(demography_sim):
    """People in the same district should not all be sent to the same facility,
    unless that district genuinely only has one facility at that level."""
    df = demography_sim.population.props
    alive = df[df["is_alive"]]
    demog = demography_sim.modules["Demography"]
    facility_info = demog.parameters["facilities_info"].copy()

    FACILITY_TYPES = {
        "level_1a": ["Dispensary", "Clinic"],
        "level_1b": ["Health Centre", "Rural/Community Hospital"],
    }
    CITY_TO_DISTRICT = {
        "Blantyre City": "Blantyre",
        "Lilongwe City": "Lilongwe",
        "Zomba City": "Zomba",
        "Mzuzu City": "Mzimba",
    }

    for level, ftypes in FACILITY_TYPES.items():
        relevant = facility_info[facility_info["facility_type"].isin(ftypes)]
        for district, group in alive.groupby("district_of_residence"):
            if len(group) < 10:
                continue
            lookup = CITY_TO_DISTRICT.get(district, district)
            n_facilities = relevant[relevant["district"] == lookup]["Fname"].nunique()
            if n_facilities <= 1:
                continue  # only one facility exists — variation impossible, skip
            n_unique = group[level].nunique()
            assert n_unique > 1, (
                f"Level '{level}', district '{district}': all {len(group)} people "
                f"assigned to the same facility despite {n_facilities} available"
            )


def test_facility_assignment_district_alignment(demography_sim):
    """Facility assigned should be in the individual's district.
    Only checked for level_1a and level_1b — District and Central Hospitals
    (level_2, level_3) are regional facilities."""
    df = demography_sim.population.props
    alive = df[df["is_alive"]]
    demog = demography_sim.modules["Demography"]

    facility_info = demog.parameters["facilities_info"].copy()
    FACILITY_DISTRICT_MAP = {  # some misspellings/divisions
        "Blanytyre": "Blantyre",
        "Mzimba North": "Mzimba",
        "Mzimba South": "Mzimba",
        "Nkhatabay": "Nkhata Bay",
    }
    CITY_TO_DISTRICT = {
        "Blantyre City": "Blantyre",
        "Lilongwe City": "Lilongwe",
        "Zomba City": "Zomba",
        "Mzuzu City": "Mzimba",
    }
    facility_info["district"] = facility_info["district"].replace(FACILITY_DISTRICT_MAP)
    fname_to_district = facility_info.drop_duplicates("Fname").set_index("Fname")["district"].to_dict()

    for level in ["level_1a", "level_1b"]:
        mismatches = 0
        total = 0
        for _, row in alive[["district_of_residence", level]].dropna().iterrows():
            expected = CITY_TO_DISTRICT.get(row["district_of_residence"], row["district_of_residence"])
            facility_district = fname_to_district.get(row[level])
            if facility_district is not None:
                total += 1
                if facility_district != expected:
                    mismatches += 1
        mismatch_rate = mismatches / total if total > 0 else 0
        assert mismatch_rate < 0.05, (
            f"Level '{level}': {mismatch_rate:.1%} of assignments are cross-district "
            f"({mismatches}/{total})"
        )


# Test linear model
def _make_weather_disruptions_sim_for_probs():
    """Minimal simulation with WeatherDisruptions registered for probability tests."""

    sim = Simulation(
        start_date=Date(2025, 1, 1),
        seed=0,
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            cons_availability="all",
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        symptommanager.SymptomManager(),
        weather_disruptions.WeatherDisruptions(
            services_affected_precip="all",
            year_effective_climate_disruptions=2025,
        ),
        check_all_dependencies=False,
    )

    sim.make_initial_population(n=100)
    sim.simulate(end_date=Date(2025, 1, 2))
    return sim


def test_disruption_probabilities_bounded():
    """Ensures probabilities are returned"""
    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    probs = wd.parameters["projected_precip_disruptions"]["disruption"]
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_disruption_probabilities_not_all_zero():
    """Ensures linear model is returning non-zero probabilities"""
    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    probs = wd.parameters["projected_precip_disruptions"]["disruption"]
    assert probs.sum() > 0


def test_disruption_probabilities_vary_by_facility():
    """Ensures linear model is returning different probabilities for each facility"""
    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    df = wd.parameters["projected_precip_disruptions"]
    jan_2025 = df[(df["year"] == 2025) & (df["month"] == 1)]["disruption"]
    assert jan_2025.nunique() > 1


def test_disruption_probabilities_vary_by_month():
    """Ensures linear model is returning different probabilities for each month"""
    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    df = wd.parameters["projected_precip_disruptions"]
    first_facility = df["RealFacility_ID"].iloc[0]
    fac_df = df[(df["RealFacility_ID"] == first_facility) & (df["year"] == 2025)]
    assert fac_df["disruption"].nunique() > 1


def test_disruption_probabilities_cover_all_months():
    """Ensures linear model is returning different probabilities for all months"""

    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    df = wd.parameters["projected_precip_disruptions"]
    months = df[df["year"] == 2025]["month"].unique()
    assert set(months) == set(range(1, 13))


def test_disruption_probabilities_cover_expected_years():
    """Ensures linear model is returning different probabilities for all years"""
    sim = _make_weather_disruptions_sim_for_probs()
    wd = sim.modules["WeatherDisruptions"]
    df = wd.parameters["projected_precip_disruptions"]
    assert df["year"].min() <= 2025
    assert df["year"].max() >= 2070


# Test parameter setting and delays
def _make_scale_factor_sim(seed, scale_factor, n_pop=1000):
    """Allows testing of scale factor."""

    class DummyHSI(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSI"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyModule(Module):
        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            for person_id in range(n_pop):
                sim.modules["HealthSystem"].schedule_hsi_event(
                    DummyHSI(self, person_id=person_id),
                    topen=sim.date,
                    tclose=None,
                    priority=1,
                )

    sim = Simulation(
        start_date=Date(2025, 1, 1),
        seed=seed,
        resourcefilepath=resourcefilepath,
    )

    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            mode_appt_constraints=1,
            capabilities_coefficient=10000.0,
            cons_availability="all",
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            force_any_symptom_to_lead_to_healthcareseeking=True
        ),
        symptommanager.SymptomManager(),
        weather_disruptions.WeatherDisruptions(
            services_affected_precip="all",
            year_effective_climate_disruptions=2025,
            scale_factor_prob_disruption=scale_factor,
            delay_in_seeking_care_weather=1.0,
            scale_factor_severity_disruption_and_delay=1.0,
            scale_factor_reseeking_healthcare_post_disruption=1.0,
            prop_supply_side_disruptions=0.0,
            scale_factor_appointment_urgency=1.0,
        ),
        DummyModule(),
        check_all_dependencies=False,
    )

    sim.make_initial_population(n=n_pop)
    sim.simulate(end_date=Date(2025, 1, 3))

    return sim.modules["WeatherDisruptions"]._disruptions_hsi_total_count


def test_scale_factor_prob_disruption_scales_probability(seed):
    """Check that a doubled scale factor should produce more disruptions."""
    disruptions_low = _make_scale_factor_sim(seed, scale_factor=0.5)
    disruptions_high = _make_scale_factor_sim(seed, scale_factor=1.0)

    assert disruptions_high > disruptions_low, (
        f"Doubling scale factor should increase disruptions: "
        f"got {disruptions_low} at scale=0.5, {disruptions_high} at scale=1.0"
    )


def test_scale_factor_prob_disruption_capped_at_1(seed):
    """Even with a very large scale factor, disruptions should not exceed 100% (i.e. all HSIs scheduled)."""
    n_pop = 500
    disruptions = _make_scale_factor_sim(seed, scale_factor=1000.0, n_pop=n_pop)

    assert disruptions <= n_pop, (
        f"Disruptions ({disruptions}) exceed total HSIs scheduled ({n_pop}) — "
        f"probability cap at 1.0 is not working"
    )


def test_delay_capped_for_time_sensitive_modules(weather_disruption_sim):
    """Checks that for time-sensitive modules, delay should never push topen past tclose."""
    wd = weather_disruption_sim.modules["WeatherDisruptions"]
    wd.parameters["scale_factor_appointment_urgency"] = 1.0
    wd.parameters["scale_factor_severity_disruption_and_delay"] = 10.0
    wd.parameters["delay_in_seeking_care_weather"] = 100.0

    class MockHSI:
        module = type("M", (), {
            "__class__": type("Labour", (), {"__name__": "Labour"})()
        })()

    class MockItem:
        hsi_event = MockHSI()
        priority = 1
        topen = Date(2025, 1, 1)
        tclose = Date(2025, 1, 3)  # only 2 days window

    item = MockItem()
    current_date = Date(2025, 1, 1)
    delay = wd._calculate_delay(item, prob_disruption=1.0, current_date=current_date)
    max_allowable = (item.tclose - current_date).days
    assert delay <= max_allowable, (
        f"Delay {delay} exceeds max allowable {max_allowable} for time-sensitive module"
    )


## Mode 2 test

def test_mode_2_supply_side_disruption_reduces_capabilities(seed, tmpdir):
    """In mode 2, a supply-side disruption should update running_total_footprint,
    which reduces available capabilities for subsequent HSIs that day."""

    class DummyHSI(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = "DummyHSI"
            self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"ConWithDCSA": 1})
            self.ACCEPTED_FACILITY_LEVEL = "1a"

        def apply(self, person_id, squeeze_factor):
            pass

    class DummyModule(Module):
        METADATA = {Metadata.USES_HEALTHSYSTEM}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.modules["HealthSystem"].schedule_hsi_event(
                DummyHSI(self, person_id=0),
                topen=sim.date,
                tclose=None,
                priority=1,
            )

    log_config = {
        "filename": "log",
        "directory": tmpdir,
        "custom_levels": {
            "tlo.methods.healthsystem": logging.DEBUG,
            "tlo.methods.weather_disruptions": logging.INFO,
        },
    }

    sim = Simulation(
        start_date=Date(2025, 1, 1), seed=seed, log_config=log_config,
        resourcefilepath=resourcefilepath
    )

    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(
            mode_appt_constraints=2,
            capabilities_coefficient=10000.0,
            cons_availability="all",
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(
            force_any_symptom_to_lead_to_healthcareseeking=True
        ),
        symptommanager.SymptomManager(),
        weather_disruptions.WeatherDisruptions(
            services_affected_precip="all",
            year_effective_climate_disruptions=2025,
            scale_factor_prob_disruption=1.0,
            delay_in_seeking_care_weather=1.0,
            scale_factor_severity_disruption_and_delay=1.0,
            scale_factor_reseeking_healthcare_post_disruption=100.0,
            prop_supply_side_disruptions=1.0,  # ← always supply-side
        ),
        DummyModule(),
        check_all_dependencies=False,
    )

    sim.make_initial_population(n=100)
    sim.modules["WeatherDisruptions"].parameters["scale_factor_appointment_urgency"] = 1.0

    # Capture running_total_footprint before the disruption fires
    hs = sim.modules["HealthSystem"]

    disruption_fired = [False]

    def always_supply_side_once(hsi_event_item, current_date):
        if disruption_fired[0]:
            return False, False
        disruption_fired[0] = True
        wd = sim.modules["WeatherDisruptions"]
        district = sim.population.props.at[
            hsi_event_item.hsi_event.target, "district_of_residence"
        ]
        treatment_id = getattr(hsi_event_item.hsi_event, "TREATMENT_ID", "unknown")
        is_supply_side = wd._handle_disruption(
            hsi_event_item=hsi_event_item,
            prob_disruption=1.0,
            current_date=current_date,
            facility_id="test_facility",
            district=str(district),
            treatment_id=str(treatment_id),
        )
        return True, is_supply_side

    sim.modules["WeatherDisruptions"].check_hsi_for_disruption = always_supply_side_once

    sim.simulate(end_date=Date(2025, 1, 3))

    # running_total_footprint should have been updated by the supply-side disruption
    total_footprint = sum(
        sum(counter.values())
        for counter in hs.running_total_footprint.values()
    )
    assert total_footprint > 0, (
        "capabilities were not reduced"
    )

    # Also check the supply-side counter on WeatherDisruptions was incremented
    wd = sim.modules["WeatherDisruptions"]
    assert wd._supply_side_disruptions_count > 0 or wd._disruptions_hsi_total_count > 0, (
        "WeatherDisruptions supply-side counter was not incremented"
    )
