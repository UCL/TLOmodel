import fnmatch
from random import shuffle
from typing import Any, Dict, List, Literal, Tuple, TypeAlias

import numpy as np
import pandas as pd
import pytest

from tlo import Date
from tlo.methods.bed_days import BedDays, BedDaysFootprint, BedOccupancy

IssueBedDaysDataType: TypeAlias = Dict[
    str,
    BedDaysFootprint
    | Dict[str, List[BedOccupancy] | BedDaysFootprint]
    | int
    | int
    | Date,
]

@pytest.fixture
def bed_days_table() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "Facility_ID": [0, 1, 2],
            "bed_1": [2, 4, 0],
            "bed_2": [4, 8, 16],
            "bed_3": [2, 0, 0],
            "non_bed_space": [0, 0, 0],
        }
    )

@pytest.fixture
def reindexed_table(bed_days_table: pd.DataFrame) -> pd.DataFrame:
    return bed_days_table.set_index("Facility_ID", inplace=False)

@pytest.fixture(scope="function")
def BD(bed_days_table: pd.DataFrame) -> BedDays:
    return BedDays(bed_capacities=bed_days_table, capacity_scaling_factor='all')


@pytest.mark.parametrize(
    ["fill_values", "is_invalid"],
    [
        pytest.param({}, False, id="Blank footprint"),
        pytest.param(
            {"bed_1": 1, "bed_2": 2, "bed_3": 3, "non_bed_space": 0},
            False,
            id="Pre-filled values (valid)",
        ),
        pytest.param(
            {"bed_1": 0, "bed_2": 0, "bed_3": 0, "non_bed_space": 1},
            True,
            id="Invalid footprint",
        ),
    ],
)
def test_footprint_methods(
    BD: BedDays, fill_values: Dict[str, int], is_invalid: bool
) -> None:
    """
    Test that the get_blank_footprint method correctly returns a footprint whose
    keys are set according to the set bed-types.

    Also check that the assert_valid_footprint method correctly flags footprints
    with non-zero allocations to the non-bed-bed-type as invalid.
    """
    total_days = sum(fill_values.values())
    
    footprint = BD.get_blank_beddays_footprint(**fill_values)
    assert len(footprint) == len(
        BD.bed_types
    ), "Bed types in blank footprint and in BedDays instance don't match."
    if total_days == 0:
        assert footprint.total_days == total_days, "Different total number of days allocated."
    else:
        assert footprint == fill_values, "Footprint was not created with correct attribute assignments."
    if is_invalid:
        with pytest.raises(AssertionError):
            BD.assert_valid_footprint(footprint)
    else:
        BD.assert_valid_footprint(footprint)


@pytest.fixture(scope="function")
def BD_max_1_capacity(bed_days_table: pd.DataFrame) -> BedDays:
    """
    Returns a BedDays object that is identical to BD,
    expect that all raw max capacities greater than 1 are set to 1.
    """
    return BedDays(bed_capacities=bed_days_table, capacity_scaling_factor=1/100)

@pytest.mark.parametrize(
    "missing_df_col, error_type, error_string",
    [
        pytest.param(
            "non_bed_space",
            AssertionError,
            "Lowest priority bed type (corresponding to no available bed space)*",
            id="Lowest-priority bed type",
        ),
        pytest.param(
            "Facility_ID",
            KeyError,
            "None of * are in the columns*",
            id="Facility IDs"
        )
    ],
)
def test_init_capacities_must_contain(
    bed_days_table: pd.DataFrame,
    missing_df_col: str,
    error_type: Exception,
    error_string: str,
) -> None:
    """Check that a BedDays instance cannot be initialised without the
    lowest-priority bed type (no bed space) being provided.
    """
    bad_bed_types = bed_days_table.loc[
        :, [col for col in bed_days_table.columns if col != missing_df_col]
    ]
    with pytest.raises(
        error_type,
        match=fnmatch.translate(error_string),
    ):
        BedDays(bed_capacities=bad_bed_types)


def test_init_stores_raw_capacties(
    bed_days_table: pd.DataFrame,
    reindexed_table: pd.DataFrame,
) -> None:
    assert BedDays(bed_days_table)._raw_max_capacities.equals(
        reindexed_table
    ), "BedDays init method does not store the raw maximum bed capacities."


def test_bed_type_to_priority(BD: BedDays, reindexed_table: pd.DataFrame) -> None:
    for position, bed_type in enumerate(reindexed_table.columns):
        given_priority = BD.bed_type_to_priority(bed_type)
        assert (
            position == given_priority
        ), f"{bed_type} is given priority {given_priority} but should be given priority {position}."


@pytest.mark.parametrize(
    "capacity_scaling_factor, expected_factor",
    [
        pytest.param("none", 0, id="none"),
        pytest.param("all", 1, id="all"),
        pytest.param(0.5, 0.5, id="Scale by half"),
        pytest.param("default", 0.25, id="Use fallback value"),
    ],
)
def test_set_max_capacities(
    BD: BedDays,
    reindexed_table: pd.DataFrame,
    capacity_scaling_factor: Literal["none", "all"] | float | int,
    expected_factor: float | int,
):
    expected_capacities = (reindexed_table * expected_factor).apply(np.ceil).astype(int)
    BD.set_max_capacities(capacity_scaling_factor=capacity_scaling_factor, fallback_value=0.25)
    assert BD.max_capacities.equals(
        expected_capacities
    ), f"Setting capacity scaling factor to {capacity_scaling_factor} does not correctly set effective bed capacities."
    assert BD._raw_max_capacities.equals(
        reindexed_table
    ), "Setting max capacities has edited the raw capacities that are stored!"


@pytest.mark.parametrize(
    "expected_error, expected_msg, args_to_max_capacities",
    [
        pytest.param(
            ValueError,
            "Cannot interpret capacity scaling factor:*",
            {"capacity_scaling_factor": []},
            id="Bad capacity scaling factor",
        ),
    ],
)
def test_set_max_capacities_error_cases(
    BD: BedDays,
    expected_error: Exception,
    expected_msg: str,
    args_to_max_capacities: Dict[str, str | float | int | None],
):
    with pytest.raises(expected_error, match=fnmatch.translate(expected_msg)):
        BD.set_max_capacities(**args_to_max_capacities)


@pytest.fixture(scope="module")
def simple_non_conflicting_occupancies() -> List[BedOccupancy]:
    o1 = BedOccupancy("bed_1", 0, Date("2010-01-05"), 2, Date("2010-01-01"))
    o2 = BedOccupancy("bed_2", 1, Date("2010-01-05"), 1, Date("2010-01-01"))
    o3 = BedOccupancy("bed_3", 2, Date("2010-01-05"), 0, Date("2010-01-01"))
    return [o1, o2, o3]


def test_schedule_occupancies_and_end_occupancies(
    BD: BedDays, simple_non_conflicting_occupancies: List[BedOccupancy]
) -> None:
    shuffle(simple_non_conflicting_occupancies)

    BD.schedule_occupancies(*simple_non_conflicting_occupancies)
    assert len(BD.occupancies) == len(simple_non_conflicting_occupancies)
    assert all(o in BD.occupancies for o in simple_non_conflicting_occupancies)

    BD.end_occupancies(*simple_non_conflicting_occupancies[:-1])
    assert len(BD.occupancies) == 1
    assert simple_non_conflicting_occupancies[-1] in BD.occupancies and all(
        o not in BD.occupancies for o in simple_non_conflicting_occupancies[:-1]
    )

    BD.end_occupancies(simple_non_conflicting_occupancies[-1])
    assert len(BD.occupancies) == 0

def test_is_inpatient(BD: BedDays) -> None:
    person_id = 0
    o1 = BedOccupancy("bed_1", 0, Date("2010-01-02"), person_id, Date("2010-01-01"))
    o2 = BedOccupancy("bed_2", 0, Date("2010-02-02"), person_id, Date("2010-02-01"))
    assert len(BD.occupancies) == 0, "BedDays occupancies should be empty before running this test"

    # No occupancies scheduled = no-one is an inpatient
    assert not BD.is_inpatient(
        person_id
    ), f"Person {person_id} is incorrectly flagged as an inpatient"

    # Schedule person_id an occupancy to make them an inpatient
    BD.schedule_occupancies(o1)
    assert BD.is_inpatient(
        person_id
    ), f"Person {person_id} should be flagged as an inpatient"
    # Scheduling another, independent occupancy should not affect their inpatient status
    BD.schedule_occupancies(o2)
    assert BD.is_inpatient(
        person_id
    ), f"Person {person_id} should be flagged as an inpatient"
    # Ending the earlier occupancy should also mean person_id is still flagged as an inpatient
    BD.end_occupancies(o1)
    assert BD.is_inpatient(
        person_id
    ), f"Person {person_id} should be flagged as an inpatient"

    # Ending person_id's remaining occupancies should no longer have them flagged as an inpatient
    BD.end_occupancies(o2)
    assert not BD.is_inpatient(
        person_id
    ), f"Person {person_id} is incorrectly flagged as an inpatient"


def test_remove_patient_footprint(
    BD: BedDays, simple_non_conflicting_occupancies: List[BedOccupancy]
) -> None:
    person_id = 0
    BD.schedule_occupancies(*simple_non_conflicting_occupancies)

    assert BD.is_inpatient(person_id)
    BD.remove_patient_footprint(person_id)

    assert not BD.is_inpatient(person_id)

@pytest.fixture
def find_occupancies_list() -> List[BedOccupancy]:
    """
    Collection of occupancies that are to be used when testing find_occupancies.
    Organised by index:
    0 - Person 0 @ facility 0 between 2010-01-01 and 2010-01-05,
    1 - Person 0 @ facility 0 between 2010-01-06 and 2010-01-07,
    2 - Person 1 @ facility 1 between 2010-02-01 and 2010-02-02,
    3 - Person 2 @ facility 2 between 2010-01-15 and 2010-02-27,
    """
    return [
        BedOccupancy("bed_1", 0, Date("2010-01-05"), 0, Date("2010-01-01")),
        BedOccupancy("bed_2", 0, Date("2010-01-07"), 0, Date("2010-01-06")),
        BedOccupancy("bed_3", 1, Date("2010-02-02"), 1, Date("2010-02-01")),
        BedOccupancy("bed_2", 2, Date("2010-02-27"), 2, Date("2010-01-15")),
    ]

@pytest.mark.parametrize(
    "args_to_fn, event_indices_that_should_be_found",
    [
        pytest.param({"patient_id": 0}, [0, 1], id="Person 0's occupancies"),
        pytest.param(
            {"patient_id": [0, 1]}, [0, 1, 2], id="Person 0 & 1's occupancies"
        ),
        pytest.param(
            {"patient_id": 0, "facility": 1}, [], id="Person 0 AND facility 1"
        ),
        pytest.param(
            {"patient_id": 0, "facility": 1, "logical_or": True},
            [0, 1, 2],
            id="Person 0 OR facility 1",
        ),
        pytest.param(
            {"patient_id": 0, "start_on_or_after": Date("2010-01-03")},
            [1],
            id="Person 1 and starting after Jan 2nd",
        ),
        pytest.param(
            {
                "patient_id": 0,
                "start_on_or_after": Date("2010-01-03"),
                "logical_or": True,
            },
            [0, 1, 2, 3],
            id="Person 1 OR starting after Jan 2nd",
        ),
        pytest.param(
            {"end_on_or_before": Date("2010-01-31")},
            [0, 1],
            id="End no later than Jan 31st",
        ),
        pytest.param(
            {"start_on_or_after": Date("2010-01-04")},
            [1, 2, 3],
            id="Start after Jan 3rd",
        ),
        pytest.param({"on_date": Date("2010-02-01")}, [2, 3], id="Occur on Feb 1st"),
        pytest.param(
            {"occurs_between_dates": (Date("2010-01-31"), Date("2010-02-03"))},
            [2, 3],
            id="Occurs between Jan 31st and Feb 3rd",
        ),
    ],
)
def test_find_occupancies(
    BD: BedDays,
    find_occupancies_list: List[BedOccupancy],
    args_to_fn: Dict[str, Any],
    event_indices_that_should_be_found: List[int],
) -> None:
    # Schedule the occupancies we want to use in this test
    BD.schedule_occupancies(*find_occupancies_list)

    # Request that occupancies matching the given criteria are found
    found_occupancies = BD.find_occupancies(**args_to_fn)

    # Assert that the occupancies with the correct indices were indeed found
    for e_index in event_indices_that_should_be_found:
        should_have_found = find_occupancies_list[e_index]
        assert (
            should_have_found in found_occupancies
        ), f"Event {should_have_found} did not get found under search conditions: {args_to_fn}"
    # Catch cases when we expect to find no occupancies, but due to errors might have
    # found some.
    assert len(event_indices_that_should_be_found) == len(
        found_occupancies
    ), "Did not find the expected number of occupancies."


def test_get_inpatient_appts(BD: BedDays, find_occupancies_list: List[BedOccupancy], simple_non_conflicting_occupancies: List[BedOccupancy]) -> None:
    """
    TODO: This method should move into the HealthSystem class maybe? It doesn't make use of the beds at all

    NB: It shouldn't matter that some people are - with this combination of occupancies -
    occupying multiple beds during this time. At least for the purposes of this test.
    """
    BD.schedule_occupancies(*find_occupancies_list)
    BD.schedule_occupancies(*simple_non_conflicting_occupancies)

    report_5th_jan = BD.get_inpatient_appts(date=Date("2010-01-05"))
    assert report_5th_jan == {
        0: {"InpatientDays": 2},
        1: {"InpatientDays": 1},
        2: {"InpatientDays": 1},
    }


@pytest.mark.parametrize(
    "incoming_occs, current_occs, solution",
    [
        pytest.param(
            [
                BedOccupancy(
                    "bed_2",
                    0,
                    Date("2010-01-05"),
                    0,
                    Date("2010-01-01"),
                )
            ],
            [
                BedOccupancy(
                    "bed_1",
                    0,
                    Date("2010-01-05"),
                    0,
                    Date("2010-01-01"),
                )
            ],
            [
                BedOccupancy(
                    "bed_1",
                    0,
                    Date("2010-01-05"),
                    0,
                    Date("2010-01-01"),
                )
            ],
            id="Higher-priority bed overwrites lower priority",
        ),
        pytest.param(
            [
                BedOccupancy(
                    "bed_2",
                    0,
                    Date("2010-01-05"),
                    0,
                    Date("2010-01-01"),
                )
            ],
            [
                BedOccupancy(
                    "bed_1",
                    0,
                    Date("2010-01-07"),
                    0,
                    Date("2010-01-03"),
                )
            ],
            [
                BedOccupancy(
                    "bed_1",
                    0,
                    Date("2010-01-07"),
                    0,
                    Date("2010-01-03"),
                ),
                BedOccupancy(
                    "bed_2",
                    0,
                    Date("2010-01-02"),
                    0,
                    Date("2010-01-01"),
                ),
            ],
            id="Higher-priority bed overwrites lower priority for portion of the stay which overlaps",
        ),
        pytest.param(
            [
                BedOccupancy("bed_1", 0, Date("2010-01-05"), 0, Date("2010-01-01")),
            ],
            [
                BedOccupancy("bed_1", 0, Date("2010-01-10"), 0, Date("2010-01-05")),
            ],
            [
                BedOccupancy("bed_1", 0, Date("2010-01-10"), 0, Date("2010-01-01")),
            ],
            id="Occupancies that want the same bed type are combined into one.",
        ),
    ],
)
def test_resolve_overlapping_dependencies(BD: BedDays, incoming_occs: List[BedOccupancy], current_occs: List[BedOccupancy], solution: List[BedOccupancy]) -> None:
    resolution = BD.resolve_overlapping_occupancies(
        incoming_occupancies=incoming_occs, current_occupancies=current_occs
    )

    assert len(solution) == len(resolution), "Resolved occupancies do not match solution!"
    for occ in resolution:
        assert occ in solution, "Resolution contains unexpected occupancy, not found in the expected solution"

@pytest.fixture
def forecasting_data() -> Tuple[List[BedOccupancy], pd.DataFrame]:
    """
    A collection of occupancies to use to test the forecast function.
    The occupancies span 2010-01-01 to 2010-01-07, and represent the following
    allocations to facility 0:
    2010-01-01: 1 bed_1, 2 bed_2,
    2010-01-02: 1 bed_1, 2 bed_2,
    2010-01-03: 2 bed_1, 2 bed_2,
    2010-01-04: 2 bed_1, 1 bed_2,
    2010-01-05: 2 bed_1, 1 bed_2,
    2010-01-06: 1 bed_1, 1 bed_2,
    2010-01-07: 1 bed_1, 0 bed_2,
    """
    facility = 0
    occs = [
        BedOccupancy("bed_1", facility, Date("2010-01-05"), 0, Date("2010-01-01")),
        BedOccupancy("bed_1", facility, Date("2010-01-07"), 1, Date("2010-01-03")),
        BedOccupancy("bed_2", facility, Date("2010-01-06"), 2, Date("2010-01-01")),
        BedOccupancy("bed_2", facility, Date("2010-01-03"), 3, Date("2010-01-01")),
    ]

    trusted_forecast = pd.DataFrame(
        data={
            "bed_1": [1, 1, 0, 0, 0, 1, 1],
            "bed_2": [2, 2, 2, 3, 3, 3, 4],
            "bed_3": [2] * 7,
            "non_bed_space": [0] * 7,
        },
        index=pd.date_range(Date("2010-01-01"), Date("2010-01-07"), freq="D"),
    )
    return occs, trusted_forecast

def test_forecast_availability(BD: BedDays, forecasting_data: Tuple[List[BedOccupancy], pd.DataFrame]) -> None:
    occs = forecasting_data[0]
    trusted_forecast = forecasting_data[1]
    trusted_boolean_forecast = (trusted_forecast > 0).reset_index(drop=True)

    BD.schedule_occupancies(*occs)

    forecast = BD.forecast_availability(Date("2010-01-01"), n_days=6, facility_id=0)
    assert len(forecast.index) == 7, "Did not receive a 7 day forecast"
    assert forecast.index[-1] == Date("2010-01-07"), "Forecast did not extend to the expected end date"
    assert forecast.index[0] == Date("2010-01-01"), "Forecast did not start at the given start date"

    boolean_forecast = BD.forecast_availability(
        Date("2010-01-01"), n_days=6, facility_id=0, as_bool=True, int_indexing=True
    )
    assert (
        boolean_forecast.index[0] == 0 and boolean_forecast.index[-1] == 6
    ), "int_indexing option does not return integer-indexed forecast"
    assert (
        boolean_forecast.dtypes  # noqa: E721
        == bool
        # (we are actually doing an instance evaluation,
        # as we want to know if every value in a series is THE TYPE bool)
    ).all(), "as_bool argument does not return boolean occupancy table"

    # We now check the returned forecasts themselves
    assert forecast.equals(trusted_forecast), "Incorrect forecast returned"
    assert boolean_forecast.equals(trusted_boolean_forecast), "Incorrect boolean forecast returned"

@pytest.fixture
def beddays_footprint_to_impose(BD: BedDays) -> BedDaysFootprint:
    """
    Returns a BedDays footprint requesting
    - 2 days in bed_1
    - 1 day  in bed_2
    - 2 days in bed_3
    """
    return BD.get_blank_beddays_footprint(bed_1=2, bed_2=1, bed_3=2)


def test_impose_beddays_footprint(
    BD: BedDays, beddays_footprint_to_impose: BedDaysFootprint
) -> None:
    facility = 0
    person_id = 0

    BD.impose_beddays_footprint(
        beddays_footprint_to_impose,
        facility=facility,
        first_day=Date("2010-01-01"),
        patient_id=person_id,
    )

    # Should now be precisely 3 bed occupancies scheduled
    assert len(BD.occupancies) == 3, "Not all expected occupancies were scheduled."
    # And they should follow one after the other, starting on 2010-01-01, which we check for
    first_occupancy = BedOccupancy(
        "bed_1", facility, Date("2010-01-02"), person_id, Date("2010-01-01")
    )
    second_occupancy = BedOccupancy(
        "bed_2", facility, Date("2010-01-03"), person_id, Date("2010-01-03")
    )
    third_occupancy = BedOccupancy(
        "bed_3", facility, Date("2010-01-05"), person_id, Date("2010-01-04")
    )
    for i, occ in enumerate([first_occupancy, second_occupancy, third_occupancy]):
        assert occ in BD.occupancies, f"The {i}th occupancy was not present after imposing the footprint."

    # Now, attempt to impose the same footprint again, starting on 2010-01-04
    # This will conflict with the occupancies we just scheduled:
    # - The bed_3 occupancy between 2010-01-04 through 2010-01-05 will be removed
    # - A bed_1 occupancy from 2010-01-04 through 2010-01-05 will be added
    # - A bed_2 occupancy from 2010-01-06 through 2010-01-06 will be added
    # - A bed_3 occupancy from 2010-01-07 through 2010-01-08 will be added
    BD.impose_beddays_footprint(
        beddays_footprint_to_impose,
        facility=facility,
        first_day=Date("2010-01-04"),
        patient_id=person_id,
    )
    assert len(BD.occupancies) == 5, "Not all expected occupancies were scheduled."

    new_third_occupancy = BedOccupancy(
        "bed_1", facility, Date("2010-01-05"), person_id, Date("2010-01-04")
    )
    fourth_occupancy = BedOccupancy(
        "bed_2", facility, Date("2010-01-06"), person_id, Date("2010-01-06")
    )
    fifth_occupancy = BedOccupancy(
        "bed_3", facility, Date("2010-01-08"), person_id, Date("2010-01-07")
    )
    for i, occ in enumerate(
        [
            first_occupancy,
            second_occupancy,
            new_third_occupancy,
            fourth_occupancy,
            fifth_occupancy,
        ]
    ):
        assert (
            occ in BD.occupancies
        ), f"The {i}th occupancy was not present after imposing the footprint with conflicts."
    assert (
        third_occupancy not in BD.occupancies
    ), "The old 3rd occupancy was not removed from the list of occupancies, it should have been overwritten in a conflict."


@pytest.fixture
def issue_bed_days_test_data(
    BD_max_1_capacity: BedDays,
) -> IssueBedDaysDataType:
    """
    Creates the data for testing the issue_bed_days_according_to_availability function.

    Returns a tuple with the following information:
    - Index 0: The footprint that will be requested, which is identical in each test case.
    - Index 1: A dictionary containing the input parameters (see below).
    - Index 2: The facility that beds should be requested at.
    - Index 3: The person who is causing the existing occupancies. Note that the issuing function should actually be agnostic to who is requesting the bed space, even if they already are assigned a bed.
    - Index 4: The date from which the footprint should be requested.

    The dictionary of test parameters contains key:value pairs as follows:
    - occs: A list of BedOccupancies that should be scheduled before attempting to issue bed days, representing existing demand on the BedDays system.
    - exp: The footprint that we expect to be returned after issuing the bed days, given the existing occupancies.
    """
    facility = 0
    start_date = Date("2010-01-01")
    patient_id = 0
    # The footprint we request will always be
    # bed_1: 4 days,
    # bed_2: 2 days,
    # bed_3: 3 days
    req_footprint = BD_max_1_capacity.get_blank_beddays_footprint(
        bed_1=4, bed_2=2, bed_3=3
    )
    scenarios = {}

    # Set 1: No existing occupancies, so returned footprint will be
    # the requested footprint.
    scenarios["No existing occupancies"] = {"occs": [], "req": req_footprint, "exp": req_footprint}

    # Set 2: No bed_1 spaces on the days this footprint would need them,
    # which should result in bed_2 days being allocated instead.
    occs_2 = [
        BedOccupancy(
            "bed_1", facility, Date("2010-01-04"), patient_id, start_date
        )
    ]
    exp_2 = BD_max_1_capacity.get_blank_beddays_footprint(bed_1=0, bed_2=6, bed_3=3)
    scenarios["No bed_1 space"] = {"occs": occs_2, "req": req_footprint, "exp": exp_2}

    # Set 3: bed_1 spaces for the first two days, but then none after.
    # Also no bed_2 days at any point.
    # Should result in bed_1 for the first 2 days, then bed_3 afterwards.
    occs_3 = [
        BedOccupancy("bed_1", facility, Date("2010-01-04"), patient_id, Date("2010-01-03")),
        BedOccupancy("bed_2", facility, Date("2010-01-10"), patient_id, start_date),
    ]
    exp_3 = BD_max_1_capacity.get_blank_beddays_footprint(bed_1=2, bed_2=0, bed_3=7)
    scenarios["No bed_2, some bed_1"] = {"occs": occs_3, "req": req_footprint, "exp": exp_3}

    # Set 4: bed_1 spaces for the requested time, no bed_2 spaces available.
    # No bed_3 spaces on the 7th day, which should result in the returned
    # allocation only consisting of 2 bed_1 days and 4 bed_3 days.
    # Note that:
    # - non_bed_space is still not allocated to, since the facility
    # has 0 "non_bed_spaces" to offer.
    # - despite there being bed_3 space on days 8-10, the algorithm being
    # used assumes no availability of a bed type after the first day it is
    # not available.
    occs_4 = [
        BedOccupancy("bed_2", facility, Date("2010-01-10"), patient_id, start_date),
        BedOccupancy("bed_3", facility, Date("2010-01-07"), patient_id, Date("2010-01-07")),
    ]
    exp_4 = BD_max_1_capacity.get_blank_beddays_footprint(
        bed_1=4, bed_2=0, bed_3=2, non_bed_space=0
    )
    scenarios["Result in non bed allocation"] = {"occs": occs_4, "req": req_footprint, "exp": exp_4}

    return (
        req_footprint, scenarios, facility, patient_id, start_date
    )

def test_issue_bed_days_according_to_availability(
    BD_max_1_capacity: BedDays, issue_bed_days_test_data: IssueBedDaysDataType,
) -> None:
    request_footprint = issue_bed_days_test_data[0]
    scenarios = issue_bed_days_test_data[1]
    facility = issue_bed_days_test_data[2]
    person_occupying_beds = issue_bed_days_test_data[3]
    start_date = issue_bed_days_test_data[4]

    for name, scenario in scenarios.items():
        # End all occupancies from the previous test case before starting
        BD_max_1_capacity.end_occupancies(
            *BD_max_1_capacity.find_occupancies(patient_id=person_occupying_beds)
        )
        assert (
            len(BD_max_1_capacity.occupancies) == 0
        ), f"Could not clear previous occupancies before starting scenario: {name}"

        # Schedule the pre-existing occupancies for this scenario
        BD_max_1_capacity.schedule_occupancies(*scenario["occs"])
        expected_footprint: BedDaysFootprint = scenario["exp"]

        provided_footprint = BD_max_1_capacity.issue_bed_days_according_to_availability(
            start_date=start_date,
            facility_id=facility,
            requested_footprint=request_footprint,
        )

        assert (
            provided_footprint.keys() == expected_footprint.keys()
        ), f"Scenario: {name} - Bed types allocated do not match!"
        for bed_type, n_days in provided_footprint.items():
            assert expected_footprint[bed_type] == n_days, (
                f"Scenario: {name} - Expected to be allocated {n_days} in {bed_type}, "
                f"but got {expected_footprint[bed_type]}"
            )


@pytest.fixture
def occupancies_to_convert() -> List[BedOccupancy]:
    return [
        BedOccupancy("bed_1", 0, Date("2010-01-02"), 0, Date("2010-01-01")),
        BedOccupancy("bed_2", 0, Date("2010-01-07"), 0, Date("2010-01-03")),
        BedOccupancy("bed_3", 0, Date("2010-01-10"), 0, Date("2010-01-08")),
    ]


@pytest.mark.parametrize(
    ["current_date", "expected"],
    [
        pytest.param(None, {"bed_1": 2, "bed_2": 5, "bed_3": 3}, id="No current date"),
        pytest.param(Date("2010-01-04"), {"bed_2": 4, "bed_3": 3}, id="Partial cut-off"),
        pytest.param(Date("2010-02-01"), {}, id="Entirely cut-off")
    ],
)
def test_occupancies_to_footprint(
    BD_max_1_capacity: BedDays, occupancies_to_convert: List[BedOccupancy], current_date: Date, expected: BedDaysFootprint
) -> None:
    """
    Test that a list of occupancies is correctly cast to a BedDaysFootprint.
    Test cases include:

    - Cast without a current date
    - Cast with a current date (loosing part of the footprint)
    - Cast with a current date (loosing the entire footprint)
    """
    expected = BD_max_1_capacity.get_blank_beddays_footprint(**expected)

    result = BD_max_1_capacity.occupancies_to_footprint(
        occupancies_to_convert, current_date=current_date
    )
    assert expected == result, "Allocations did not match."
