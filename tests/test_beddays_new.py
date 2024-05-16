"""Test file for the bed-days class"""

import pandas as pd
import pytest

from tlo import Date

from tlo.methods.bed_days import BedDays, BedOccupancy

start_date = Date(2010, 1, 1)


@pytest.fixture
def bed_days_data() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "Facility_ID": [0, 1, 2],
            "T1": 5,
            "T2": 100,
            "non_bed_space": 0,
        }
    )


@pytest.fixture
def bed_days(bed_days_data) -> BedDays:
    return BedDays(bed_days_data, "all")


def test_find_occupancies(bed_days) -> None:
    """ """
    one_day_after = start_date + pd.DateOffset(days=1)
    two_days_after = start_date + pd.DateOffset(days=2)
    three_days_after = start_date + pd.DateOffset(days=3)
    o1 = BedOccupancy(
        bed_type="T1",
        facility=0,
        freed_date=one_day_after,
        patient_id=0,
        start_date=start_date,
    )
    o2 = BedOccupancy(
        bed_type="T2",
        facility=1,
        freed_date=three_days_after,
        patient_id=1,
        start_date=one_day_after,
    )
    bed_days.schedule_occupancies(o1, o2)

    assert len(bed_days.occupancies) == 2
    # Should turn up 1 event for patients 0 and 1
    assert len(bed_days.find_occupancies(patient_id=0)) == 1
    assert bed_days.find_occupancies(patient_id=0)[0] == o1
    assert len(bed_days.find_occupancies(patient_id=1)) == 1
    assert bed_days.find_occupancies(patient_id=1)[0] == o2

    # Should turn up 1 event for facilities 0 and 1 as well
    assert len(bed_days.find_occupancies(facility=0)) == 1
    assert bed_days.find_occupancies(facility=0)[0] == o1
    assert len(bed_days.find_occupancies(facility=1)) == 1
    assert bed_days.find_occupancies(facility=1)[0] == o2

    # Can grab multiple patients / facilities
    assert len(bed_days.find_occupancies(patient_id=[0, 1])) == 2
    assert len(bed_days.find_occupancies(facility=[0, 1])) == 2

    # Filter by start date
    assert len(bed_days.find_occupancies(start_on_or_after=one_day_after)) == 1
    assert bed_days.find_occupancies(start_on_or_after=one_day_after)[0] == o2

    # Filter by end date
    assert len(bed_days.find_occupancies(end_on_or_before=two_days_after)) == 1
    assert bed_days.find_occupancies(end_on_or_before=two_days_after)[0] == o1

    # Filter by start and end date
    assert (
        len(
            bed_days.find_occupancies(
                end_on_or_before=two_days_after, start_on_or_after=two_days_after
            )
        )
        == 0
    )
    assert (
        len(
            bed_days.find_occupancies(
                end_on_or_before=two_days_after,
                start_on_or_after=two_days_after,
                logical_or=True,
            )
        )
        == 2
    )

    # Filter by on date
    assert len(bed_days.find_occupancies(on_date=one_day_after)) == 2
    assert len(bed_days.find_occupancies(on_date=start_date)) == 1
    assert bed_days.find_occupancies(on_date=start_date)[0] == o1

    # Filter by between dates
    assert (
        len(
            bed_days.find_occupancies(
                occurs_between_dates=[two_days_after, three_days_after]
            )
        )
        == 1
    )
    assert (
        bed_days.find_occupancies(
            occurs_between_dates=[two_days_after, three_days_after]
        )[0]
        == o2
    )
    assert (
        len(
            bed_days.find_occupancies(occurs_between_dates=[start_date, two_days_after])
        )
        == 2
    )
