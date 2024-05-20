"""Test file for the bed-days class"""
from typing import List

import pandas as pd
import pytest

from tlo import Date

from tlo.methods.bed_days import BedDays, BedOccupancy

start_date = Date(2010, 1, 1)
one_day_after = start_date + pd.DateOffset(days=1)
two_days_after = start_date + pd.DateOffset(days=2)
three_days_after = start_date + pd.DateOffset(days=3)

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
def premade_bed_occupancies() -> List[BedOccupancy]:
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
    return [o1, o2]


@pytest.fixture
def bed_days(bed_days_data) -> BedDays:
    return BedDays(bed_days_data, "all")


def test_find_occupancies(
    bed_days: BedDays, premade_bed_occupancies: List[BedOccupancy]
) -> None:
    """
    Test the different filters for the find_occupancies method.
    """
    occ0 = premade_bed_occupancies[0]
    occ1 = premade_bed_occupancies[1]
    bed_days.schedule_occupancies(occ0, occ1)

    assert len(bed_days.occupancies) == 2
    # Should turn up 1 event for patients 0 and 1
    assert (
        len(bed_days.find_occupancies(patient_id=0)) == 1
        and bed_days.find_occupancies(patient_id=0)[0] == occ0
    )
    assert (
        len(bed_days.find_occupancies(patient_id=1)) == 1
        and bed_days.find_occupancies(patient_id=1)[0] == occ1
    )

    # Should turn up 1 event for facilities 0 and 1 as well
    assert (
        len(bed_days.find_occupancies(facility=0)) == 1
        and bed_days.find_occupancies(facility=0)[0] == occ0
    )
    assert (
        len(bed_days.find_occupancies(facility=1)) == 1
        and bed_days.find_occupancies(facility=1)[0] == occ1
    )

    # Can grab multiple patients / facilities
    assert len(bed_days.find_occupancies(patient_id=[0, 1])) == 2
    assert len(bed_days.find_occupancies(facility=[0, 1])) == 2

    # Filter by start date
    assert (
        len(bed_days.find_occupancies(start_on_or_after=one_day_after)) == 1
        and bed_days.find_occupancies(start_on_or_after=one_day_after)[0] == occ1
    )

    # Filter by end date
    assert (
        len(bed_days.find_occupancies(end_on_or_before=two_days_after)) == 1
        and bed_days.find_occupancies(end_on_or_before=two_days_after)[0] == occ0
    )

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
    assert (
        len(bed_days.find_occupancies(on_date=start_date)) == 1
        and bed_days.find_occupancies(on_date=start_date)[0] == occ0
    )

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
        == occ1
    )
    assert (
        len(
            bed_days.find_occupancies(occurs_between_dates=[start_date, two_days_after])
        )
        == 2
    )

def test_impose_beddays_footprint() -> None:
    """
    """

def test_schedule_occupancies(bed_days: BedDays, premade_bed_occupancies: List[BedOccupancy]) -> None:
    """
    Test the scheduling methods for the BedDays class.
    """
    


def test_forecast_availability(bed_days: BedDays, small_bedday_dataset: List[BedOccupancy]):
    """Test the functionalities of BedDays class in the absence of HSI_Events"""
    # 1) Check if impose footprint works as expected
    footprint = bed_days.get_blank_beddays_footprint(bedtype1=2)

    bed_days.impose_beddays_footprint(
        footprint=footprint, facility=0, patient_id=0, first_day=start_date
    )
    # There should be a single occupancy now scheduled for the patient
    assert len(bed_days.occupancies) == 1

    forecast = bed_days.forecast_availability(
        start_date=start_date, n_days=0, facility_id=0, int_indexing=True
    )
    # One bed of type 1 should be occupied
    assert forecast.loc[0, "bedtype1"] == small_bedday_dataset.loc[0, "bedtype1"] - 1
    # No beds of type 2 should be occupied
    assert forecast.loc[0, "bedtype2"] == small_bedday_dataset.loc[0, "bedtype2"]

    # 3) Check that removing bed-days from a person without bed-days does nothing
    bed_days.remove_patient_footprint(1)
    forecast = bed_days.forecast_availability(
        start_date=start_date, n_days=0, facility_id=0, int_indexing=True
    )
    # One bed of type 1 should be occupied
    assert forecast.loc[0, "bedtype1"] == small_bedday_dataset.loc[0, "bedtype1"] - 1
    # No beds of type 2 should be occupied
    assert forecast.loc[0, "bedtype2"] == small_bedday_dataset.loc[0, "bedtype2"]

    # 2) Cause someone to die and relieve their footprint from the bed-days tracker
    bed_days.remove_patient_footprint(patient_id=0)
    forecast = bed_days.forecast_availability(
        start_date=start_date, n_days=0, facility_id=0, int_indexing=True
    )
    # Should have removed the one occupancy that we had
    assert len(bed_days.occupancies) == 0
    assert forecast.loc[0, "bedtype1"] == small_bedday_dataset.loc[0, "bedtype1"]
