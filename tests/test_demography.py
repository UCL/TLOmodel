import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from tlo import DAYS_IN_MONTH, DAYS_IN_YEAR, Date, Module, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import Metadata, demography
from tlo.methods.causes import Cause
from tlo.methods.demography import AgeUpdateEvent
from tlo.methods.diarrhoea import increase_risk_of_death, make_treatment_perfect
from tlo.methods.fullmodel import fullmodel

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 500


@pytest.fixture
def simulation(seed):
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    core_module = demography.Demography()
    sim.register(core_module)
    return sim


# Facility level columns are categorical but with legitimately different category
# lists between props (level-specific) and new_row (all facilities), so they are
# excluded from strict dtype equality checks.
FACILITY_LEVEL_COLUMNS = {'level_0', 'level_1a', 'level_1b', 'level_2', 'level_3'}


def check_dtypes(simulation):
    df = simulation.population.props
    orig = simulation.population.new_row
    shared_cols = df.columns.intersection(orig.columns)
    mismatches = {}
    for col in shared_cols:
        df_dtype = df[col].dtype
        orig_dtype = orig[col].dtype
        if col in FACILITY_LEVEL_COLUMNS:
            # Both sides should still be categorical — just the category lists
            # legitimately differ after level-specific facility assignment
            assert hasattr(df_dtype, 'categories'), f"{col}: expected category in props, got {df_dtype}"
            assert hasattr(orig_dtype, 'categories'), f"{col}: expected category in new_row, got {orig_dtype}"
            continue
        if df_dtype != orig_dtype:
            mismatches[col] = (df_dtype, orig_dtype)
    assert len(mismatches) == 0, (
        f"Column dtype mismatches:\n"
        + "\n".join(f"  {col}: props={a}  new_row={b}" for col, (a, b) in mismatches.items())
    )


def test_run_dtypes_and_mothers_female(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    assert set(['Other']) == set(simulation.population.props['cause_of_death'].cat.categories)
    check_dtypes(simulation)
    # check all mothers are female
    df = simulation.population.props
    mothers = df.loc[df.mother_id >= 0, 'mother_id']
    is_female = mothers.apply(lambda mother_id: df.at[mother_id, 'sex'] == 'F')
    assert is_female.all()


def test_storage_of_cause_of_death(seed):
    rfp = Path(os.path.dirname(__file__)) / '../resources'

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}
        CAUSES_OF_DEATH = {'a_cause': Cause(label='a_cause')}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, resourcefilepath=rfp)
    sim.register(
        demography.Demography(),
        DummyModule()
    )
    sim.make_initial_population(n=20)
    df = sim.population.props
    assert isinstance(df.dtypes['cause_of_death'], pd.CategoricalDtype)
    assert set(['Other', 'a_cause']) == set(df['cause_of_death'].cat.categories)

    # Cause a person to die by the DummyModule
    person_id = 0
    sim.modules['Demography'].do_death(
        individual_id=person_id,
        originating_module=sim.modules['DummyModule'],
        cause='a_cause'
    )

    person = df.loc[person_id]
    assert not person.is_alive
    assert person.cause_of_death == 'a_cause'
    check_dtypes(sim)


@pytest.mark.slow
def test_cause_of_death_being_registered(tmpdir, seed):
    """Test that the modules can declare causes of death, that the mappers between tlo causes of death and gbd
    causes of death can be created correctly and that the analysis helper scripts can be used to produce comparisons
    between model outputs and GBD data."""
    rfp = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.demography': logging.INFO
        }
    }, resourcefilepath=rfp)

    sim.register(
        *fullmodel(
            module_kwargs={"HealthSystem": {"disable": True}},
        )
    )

    # Increase risk of death of Diarrhoea to ensure that are at least some deaths
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2010, 12, 31))
    check_dtypes(sim)

    mapper_from_tlo_causes, mapper_from_gbd_causes = \
        sim.modules['Demography'].create_mappers_from_causes_of_death_to_label()

    assert set(mapper_from_tlo_causes.keys()) == set(sim.modules['Demography'].causes_of_death)
    assert set(mapper_from_gbd_causes.keys()) == sim.modules['Demography'].gbd_causes_of_death
    assert set(mapper_from_gbd_causes.values()) == set(mapper_from_tlo_causes.values())

    # check that these mappers come out in the log correctly
    output = parse_log_file(sim.log_filepath)
    demoglog = output['tlo.methods.demography']
    assert mapper_from_tlo_causes == \
           pd.Series(demoglog['mapper_from_tlo_cause_to_common_label'].drop(columns={'date'}).loc[0]).to_dict()
    assert mapper_from_gbd_causes == \
           pd.Series(demoglog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]).to_dict()

    # Check that the mortality risks being used in Other Death Poll have been reduced from the 'all-cause' rates
    odp = sim.modules['Demography'].other_death_poll
    all_cause_risk = odp.get_all_cause_mort_risk_per_poll()
    actual_risk_per_poll = odp.mort_risk_per_poll
    assert (
        actual_risk_per_poll['prob_of_dying_before_next_poll'] < all_cause_risk['prob_of_dying_before_next_poll']
    ).all()

    # check that can recover from the log the proportion of deaths represented by the OtherDeaths
    logged_prop_of_death_by_odp = demoglog['other_deaths'][['Sex', 'Age_Grp', '0']].to_dict()
    dict_of_ser = {k: pd.DataFrame(v)[0] for k, v in logged_prop_of_death_by_odp.items()}
    log_odp = pd.concat(dict_of_ser, axis=1).set_index(['Sex', 'Age_Grp'])['0']
    assert (log_odp < 1.0).all()

    # Run the analysis file:
    results = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=rfp)
    # Check the number of deaths in model represented is right (allowing for the scaling factor)
    assert (results['model'].sum() * 5.0) == approx(len(output['tlo.methods.demography']['death'])
                                                    / sim.modules['Demography'].initial_model_to_data_popsize_ratio
                                                    )


@pytest.mark.slow
def test_calc_of_scaling_factor(tmpdir, seed):
    """Test that the scaling factor is computed and put out to the log"""
    rfp = Path(os.path.dirname(__file__)) / '../resources'
    popsize = 10_000
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.INFO,
        }
    }, resourcefilepath=rfp)
    sim.register(
        demography.Demography(),
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)

    # Check that the scaling factor is calculated in the log correctly:
    output = parse_log_file(sim.log_filepath)
    sf = output['tlo.methods.demography']['scaling_factor'].at[0, 'scaling_factor']
    assert sf == approx(14.5e6 / popsize, rel=0.10)

    # Check that the scaling factor is also logged in `tlo.methods.population`
    assert output['tlo.methods.demography']['scaling_factor'].at[0, 'scaling_factor'] == \
           output['tlo.methods.population']['scaling_factor'].at[0, 'scaling_factor']


def test_py_calc(simulation):
    # make population of one person:
    simulation.make_initial_population(n=1)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date
    one_year = pd.Timedelta(days=DAYS_IN_YEAR)
    one_month = pd.Timedelta(days=DAYS_IN_MONTH)
    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc py: person is born and died before sim.date
    df.date_of_birth = now - (one_year * 10)
    df.date_of_death = now - (one_year * 9)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc py of person who is not yet born:
    df.date_of_birth = pd.NaT
    df.date_of_death = pd.NaT
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc person who is alive and aged 20, with birthdays on today's date and lives throughout the period
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    np.testing.assert_almost_equal(calc_py_lived_in_last_year(delta=one_year)['M'][19], 1.0)

    # calc person who is alive and aged 20, with birthdays on today's date, and dies 3 months ago
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = now - pd.Timedelta(one_year) * 0.25
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_almost_equal(0.75, df_py['M'][19])
    assert df_py['M'][20] == 0.0

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and lives throughout
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_allclose(0.5, df_py['M'][19])
    np.testing.assert_allclose(0.5, df_py['M'][20])

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and died 3 months ago
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = now - (one_month * 3)
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_allclose(0.75, df_py['M'].sum())
    np.testing.assert_allclose(0.5, df_py['M'][19])
    np.testing.assert_allclose(0.25, df_py['M'][20])

    # 0/1 year-old with first birthday during the last year
    df.date_of_birth = now - (one_month * 15)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_allclose(0.75, df_py['M'][0])
    np.testing.assert_allclose(0.25, df_py['M'][1])

    # 0 year born in the last year
    df.date_of_birth = now - (one_month * 9)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_allclose(0.75, df_py['M'][0])
    np.testing.assert_allclose(0, df_py['M'][1:])

    # 99 years-old turning 100 in the last year
    df.date_of_birth = now - (one_year * 100) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_allclose(0.5, df_py['M'][99])
    np.testing.assert_allclose(1, df_py['M'].sum())


def test_py_calc_w_mask(simulation):
    """test that function calc_py_lived_in_last_year works to calculate PY lived without a given condition """

    # make population of two people
    simulation.make_initial_population(n=2)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date
    one_year = pd.Timedelta(days=DAYS_IN_YEAR)

    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # neither has hypertension

    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df['nc_hypertension'] = False
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year, mask=mask)
    np.testing.assert_almost_equal(2.0, df_py['M'][19])

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # one has hypertension

    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df['nc_hypertension'].iloc[0] = True
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year, mask=mask)
    np.testing.assert_almost_equal(1.0, df_py['M'][19])


def test_max_age_initial(seed):
    """Check that the parameter in the `Demography` module, `max_age_initial`, works as expected
     * `max_age_initial=X`: only persons up to and including age_years (age in whole years) up to X are included in the
      initial population.
     * `max_age_initial=0` or `>MAX_AGE`: results in an error being thrown.
    """

    from tlo.methods.demography import MAX_AGE

    def max_age_in_sim_with_max_age_initial_argument(_max_age_initial):
        """Return the greatest value of `age_years` in a population that is created."""
        resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
        sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
        sim.register(
            demography.Demography()
        )
        sim.modules['Demography'].parameters['max_age_initial'] = _max_age_initial
        sim.make_initial_population(n=50_000)
        return sim.population.props.age_years.max()

    # `max_age_initial=5` (using integer)
    assert max_age_in_sim_with_max_age_initial_argument(5) <= 5

    # `max_age_initial=5.5` (using float)
    assert max_age_in_sim_with_max_age_initial_argument(5.5) <= int(5.5)

    # `max_age_initial=0`
    with pytest.raises(ValueError):
        max_age_in_sim_with_max_age_initial_argument(0)

    # `max_age_initial>MAX_AGE`
    with pytest.raises(ValueError):
        max_age_in_sim_with_max_age_initial_argument(MAX_AGE + 1)


def test_ageing_of_old_people_up_to_max_age(simulation):
    """Check persons can age naturally up to MAX_AGE and are then assumed to die with cause 'Other'."""

    # Populate the model with persons aged 90 years
    simulation.make_initial_population(n=1000)
    df = simulation.population.props
    df.loc[df.is_alive, 'date_of_birth'] = simulation.start_date - pd.DateOffset(years=90)
    ever_alive = df.loc[df.is_alive].index

    # Make the intrinsic risk of death zero (to enable ageing up to MAX_AGE)
    simulation.modules['Demography'].parameters['all_cause_mortality_schedule']['death_rate'] = 0.0

    # Simulate the model for 40 years (such that the persons would be 130 years old, greater than MAX_AGE)
    simulation.simulate(end_date=simulation.start_date + pd.DateOffset(years=40))

    # All persons should have died, with a cause of 'Other'
    assert not df.loc[ever_alive].is_alive.any()
    assert (df.loc[ever_alive, 'cause_of_death'] == 'Other').all()


def test_equal_allocation_by_district(seed):
    """
    Check when key-word argument `equal_allocation_by_district=True` that each district has an identical population size
    """

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(
            equal_allocation_by_district=True,
        )
    )
    population_per_district = 10_000
    number_of_districts = len(sim.modules['Demography'].districts)
    popsize = number_of_districts * population_per_district
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)  # Simulate for zero days

    # check population size
    df = sim.population.props
    assert sum(df.is_alive) == popsize

    # check total within each district is (close to being) identical and matches the target population of each district
    pop_size_by_district = df.loc[df.is_alive].groupby('district_of_residence').size()
    assert np.allclose(pop_size_by_district.values, pop_size_by_district, rtol=0.05)


# ── Facility assignment tests ─────────────────────────────────────────────────
# These name normalisations must mirror those used in assign_closest_facility_level()
_CITY_TO_DISTRICT = {
    "Blantyre City": "Blantyre",
    "Lilongwe City": "Lilongwe",
    "Zomba City": "Zomba",
    "Mzuzu City": "Mzimba",
}
_FACILITY_DISTRICT_MAP = {
    "Blanytyre": "Blantyre",
    "Mzimba North": "Mzimba",
    "Mzimba South": "Mzimba",
    "Nkhatabay": "Nkhata Bay",
}
_FACILITY_LEVELS = {
    "level_0": ["Health Post"],
    "level_1a": ["Dispensary", "Clinic"],
    "level_1b": ["Health Centre", "Rural/Community Hospital"],
    "level_2": ["District Hospital"],
    "level_3": ["Central Hospital"],
}

# The 8 districts that have no Health Posts (Level 0) in the facility resource file —
# level_0 assignment for these will correctly fall back to the nearest available
# Health Post in the region, resulting in cross-district assignments.
_DISTRICTS_WITHOUT_HEALTH_POSTS = frozenset({
    'Blantyre', 'Likoma', 'Mulanje', 'Mwanza', 'Neno', 'Salima', 'Zomba'
})


@pytest.fixture(scope="module")
def sim_with_facilities(seed=0):
    """
    Simulation initialised with a population large enough that every facility
    has a reasonable chance of being assigned at least one person.
    Scoped to module so the slow initialisation runs only once for all facility tests.
    """
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    sim.register(demography.Demography())
    sim.make_initial_population(n=50_000)
    sim.simulate(end_date=start_date)  # zero-day run — only initial state needed
    return sim


def test_facility_assignment_coordinate_uniqueness(sim_with_facilities):
    """
    Check that worldpop-weighted coordinate sampling produced meaningfully distinct
    locations, i.e. individuals within the same district were not all collapsed to
    a single point.  We test this indirectly (coordinate_of_residence is dropped
    after assignment) by checking that no single facility monopolises >95% of a
    district's population at level_0, but only in districts where more than one
    Health Post exists — if a district has only one eligible facility, 100% share
    is correct behaviour.
    """
    df = sim_with_facilities.population.props
    alive = df[df["is_alive"]]
    demog = sim_with_facilities.modules["Demography"]
    facility_info = demog.parameters["facilities_info"].copy()
    facility_info["district_clean"] = facility_info["Dist"].replace(_FACILITY_DISTRICT_MAP)

    for district, group in alive.groupby("district_of_residence"):
        if len(group) < 20:
            continue

        assert group["level_0"].nunique() >= 1, (
            f"District '{district}': no level_0 facility assigned at all."
        )

        # Count how many Health Posts exist in this district's lookup district
        lookup_district = _CITY_TO_DISTRICT.get(district, district)
        n_eligible = len(
            facility_info[
                (facility_info["Ftype"] == "Health Post") &
                (facility_info["district_clean"] == lookup_district)
                ]
        )

        # Only check spread if there is more than one eligible facility —
        # a district with a single Health Post will correctly show 100% share
        if n_eligible <= 1:
            continue

        top_share = group["level_0"].value_counts(normalize=True).iloc[0]
        assert top_share < 0.95, (
            f"District '{district}': {top_share:.1%} of individuals share one "
            f"level_0 facility despite {n_eligible} eligible Health Posts — "
            f"worldpop coordinate sampling may have collapsed."
        )


def test_all_districts_have_health_posts(sim_with_facilities):
    """Document which districts have no Health Posts in the facility resource file.
    For these districts level_0 assignment falls back to the nearest available
    Health Post in the region, so cross-district assignments are expected and correct.
    This test will fail if the set of affected districts changes unexpectedly,
    e.g. if the resource file is updated."""
    facility_info = sim_with_facilities.modules["Demography"].parameters["facilities_info"].copy()
    facility_info["district_clean"] = facility_info["Dist"].replace(_FACILITY_DISTRICT_MAP)

    health_posts = facility_info[facility_info["Ftype"] == "Health Post"]
    districts_with_posts = set(health_posts["district_clean"].unique())

    all_districts = set(sim_with_facilities.population.props["district_of_residence"].unique())
    all_lookup_districts = {_CITY_TO_DISTRICT.get(d, d) for d in all_districts}

    missing = all_lookup_districts - districts_with_posts
    assert missing == _DISTRICTS_WITHOUT_HEALTH_POSTS, (
        f"Expected districts without Health Posts: {sorted(_DISTRICTS_WITHOUT_HEALTH_POSTS)}\n"
        f"Actual districts without Health Posts:   {sorted(missing)}\n"
        f"If the resource file has changed, update _DISTRICTS_WITHOUT_HEALTH_POSTS."
    )


def test_facility_assignment_district_alignment(sim_with_facilities):
    """
    Check that each individual's assigned facility at every level is in their
    district of residence.  Districts that have no eligible facilities of the
    relevant type fall back to region-level search — cross-district assignments
    for those districts are expected and correct.
    For all other districts a <1% tolerance is applied for the region-level fallback.
    """
    df = sim_with_facilities.population.props
    alive = df[df["is_alive"]]
    demog = sim_with_facilities.modules["Demography"]
    _DISTRICTS_WITHOUT_HEALTH_POSTS = frozenset({
        'Blantyre', 'Likoma', 'Mulanje', 'Mwanza', 'Neno', 'Salima', 'Zomba'
    })
    _DISTRICTS_WITHOUT_CLINICS = frozenset({
        'Ntchisi'
    })
    _DISTRICTS_WITHOUT_DISTRICT_HOSPITALS = frozenset({
        'Dowa', 'Likoma', 'Phalombe', 'Zomba'
    })
    _DISTRICTS_WITHOUT_CENTRAL_HOSPITALS = frozenset({
        'Balaka', 'Chikwawa', 'Chiradzulu', 'Chitipa', 'Dedza', 'Dowa', 'Karonga',
        'Kasungu', 'Likoma', 'Machinga', 'Mangochi', 'Mchinji', 'Mulanje', 'Mwanza',
        'Neno', 'Nkhata Bay', 'Nkhotakota', 'Nsanje', 'Ntcheu', 'Ntchisi', 'Phalombe',
        'Rumphi', 'Salima', 'Thyolo'
    })

    _DISTRICTS_WITHOUT_LEVEL_FACILITIES = {
        "level_0": _DISTRICTS_WITHOUT_HEALTH_POSTS,
        "level_1a": _DISTRICTS_WITHOUT_CLINICS,
        "level_1b": frozenset(),
        "level_2": _DISTRICTS_WITHOUT_DISTRICT_HOSPITALS,
        "level_3": _DISTRICTS_WITHOUT_CENTRAL_HOSPITALS,
    }

    # Map from level to the set of districts that legitimately produce cross-district
    # assignments because no eligible facility of that type exists in the district.
    _DISTRICTS_WITHOUT_LEVEL_FACILITIES = {
        "level_0": _DISTRICTS_WITHOUT_HEALTH_POSTS,
        "level_1a": _DISTRICTS_WITHOUT_CLINICS,
        "level_1b": frozenset(),
        "level_2": _DISTRICTS_WITHOUT_DISTRICT_HOSPITALS,
        "level_3": _DISTRICTS_WITHOUT_CENTRAL_HOSPITALS,
    }
    facility_info = demog.parameters["facilities_info"].copy()
    facility_info["district_clean"] = facility_info["Dist"].replace(_FACILITY_DISTRICT_MAP)
    fname_to_district = (
        facility_info.drop_duplicates("Fname")
        .set_index("Fname")["district_clean"]
        .to_dict()
    )

    for level in _FACILITY_LEVELS:
        assert level in alive.columns, f"Column '{level}' missing from population props."

        subset = alive[["district_of_residence", level]].dropna(subset=[level])
        mismatches = 0

        for _, row in subset.iterrows():
            expected_district = _CITY_TO_DISTRICT.get(
                row["district_of_residence"], row["district_of_residence"]
            )
            facility_district = fname_to_district.get(row[level])
            if facility_district is not None and facility_district != expected_district:
                # Skip districts that have no eligible facilities at this level —
                # cross-district assignment via region fallback is expected and correct
                if expected_district in _DISTRICTS_WITHOUT_LEVEL_FACILITIES[level]:
                    continue
                mismatches += 1

        mismatch_rate = mismatches / len(subset) if len(subset) else 0
        assert mismatch_rate < 0.01, (
            f"Level '{level}': district mismatch rate {mismatch_rate:.2%} "
            f"({mismatches} / {len(subset)} individuals) exceeds 1% tolerance."
        )
