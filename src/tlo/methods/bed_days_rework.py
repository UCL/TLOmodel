from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd

from tlo import Date

BedDaysFootprint: TypeAlias = Dict[str, float | int]

@dataclass
class BedOccupancy:
    """
    Logs an allocation of bed-days resources, and when it will be freed up.
    """

    bed_type: str
    facility: int
    freed_date: Date
    patient_id: int
    start_date: Date


class BedDaysRework:
    """
    Tracks bed days resources, in a better way than using dataframe columns
    """

    # Index: FacilityID, Cols: max capacity of given bed type
    _max_capacities: pd.DataFrame
    # List of bed occupancies, on removal from the queue, the resources an occupancy takes up
    # are returned to the DF
    occupancies: List[BedOccupancy]

    bed_types: Tuple[str]

    def __init__(
        self,
        bed_capacities: pd.DataFrame,
        capacity_scaling_factor: float | Literal["all", "none"] = 1.0,
    ) -> None:
        """
        bed_capacities is intended to replace self.hs_module.parameters["BedCapacity"]
        availability has been superseded by capacity_scaling_factor

        :param capacity_scaling_factor: Capacities read from resource files are scaled by this factor. "all" will map to 1.0, whilst "none" will map to 0.0 (no beds available anywhere). Default is 1.
        """
        # A dictionary to create a footprint according to facility bed days capacity
        self.available_footprint = {} # RM?

        # A dictionary to track inpatient bed days
        self.bed_tracker = {} # RM?
        self.list_of_cols_with_internal_dates = {} # RM?

        # List of bed-types
        self.bed_types = (x for x in bed_capacities.columns if x != "Facility_ID")

        # Determine the (scaled) bed capacity of each facility, and store the information
        if isinstance(capacity_scaling_factor, str):
            if capacity_scaling_factor == "all":
                capacity_scaling_factor = 1.0
            elif capacity_scaling_factor == "none":
                capacity_scaling_factor = 0.0
            else:
                raise ValueError(
                    f"Cannot interpret capacity scaling factor: {capacity_scaling_factor}."
                    " Provide a numeric value, 'all' (1.0), or 'none' (0.0)."
                )
        self._max_capacities = (
            (
                bed_capacities.set_index("Facility_ID", inplace=False)
                * capacity_scaling_factor
            )
            .apply(np.ceil)
            .astype(int)
        )
        # Create the dataframe that will track the current bed occupancy of each facility
        self.occupied_beds = pd.DataFrame(0, index=self._max_capacities.index.copy(), columns=self._max_capacities.columns.copy())

        # Set the initial list of bed occupancies as an empty list
        self.occupancies = []

    def is_inpatient(self, patient_id: int, on_date: Date) -> bool:
        """
        Return True if the person with the given index is currently
        an inpatient.
        
        A person is an inpatient if they are currently occupying a bed.

        :param patient_id: Index of the patient in the population DataFrame.
        :param on_date: The date on which to determine if they are an inpatient. 
        """
        return bool(self.find_occupancies(patient_id=patient_id, end_on_or_before=on_date))

    def occupancy_ends(self, occupancy: BedOccupancy) -> None:
        """
        Action taken when a bed occupancy in the list of occupancies
        is to end, for any reason.

        The occupancy is removed from the list of events.

        :param occupancy: The BedOccupancy that has ended.
        """
        self.occupancies.remove(occupancy)

    def start_of_day(self, todays_date: Date) -> None:
        """
        Actions to take at the start of a new day.

        - End any bed occupancies that are set to be freed today.
        """
        # End bed occupancies that expired yesterday
        for o in self.find_occupancies(
            end_on_or_before=todays_date - pd.DateOffset(days=1)
        ):
            self.occupancy_ends(o)

    def find_occupancies(
        self,
        end_on_or_before: Optional[Date] = None,
        facility: Optional[List[int] | int] = None,
        patient_id: Optional[List[int] | int] = None,
        start_on_or_after: Optional[Date] = None,
    ) -> List[BedOccupancy]:
        """
        Find all occupancies in the current list that match the criteria given.

        :param end_on_or_before: Only match occupancies that are scheduled to end on or before the date provided.
        :param facility: Only match occupancies that take place at the given facility (or facilities if list).
        :param patient_id: Only match occupancies scheduled for the given person (or persons if list).
        :param start_on_or_after: Only match occupancies that are scheduled to start on or after the date provided.
        """
        # Cast single-values to lists to make parsing easier
        if isinstance(patient_id, int):
            patient_id = [patient_id]
        if isinstance(facility, int):
            facility = [facility]

        matches = [
            o
            for o in self.occupancies
            if (patient_id is None or o.patient_id in patient_id)
            and (facility is None or o.facility in facility)
            and (start_on_or_after is None or o.start_date < start_on_or_after)
            and (end_on_or_before is None or end_on_or_before < o.freed_date)
        ]
        return matches

    # THESE MAY NOT BE NEEDED BUT ARE THERE TO KEEP THE HEALTHSYSTEM REWORK HEALTHY

    @property
    def availability(self) -> None:
        # don't think this attribute needs to be tracked anymore,
        # but there's a random event that thinks it should change it.
        # But then the codebase doesn't actually use the value
        # of this attribute AFTER that point in the simulation anyway.
        raise ValueError("Tried to access BedDays.availability")

    def remove_beddays_footprint(self, *args, **kwargs) -> None:
        pass

    def check_beddays_footprint_format(self, *args, **kwargs) -> None:
        pass

    def pre_initialise_population(self, *args, **kwargs) -> None:
        pass

    def get_facility_id_for_beds(self, *args, **kwargs) -> None:
        # should be a hs module method!
        pass
    
    def get_blank_beddays_footprint(self, *args, **kwargs) -> Dict[str, float | int]:
        # we shouldn't need this, just rework to not require all the fields...
        return {}

    def get_inpatient_appts(self, *args, **kwargs) -> None:
        pass

    def initialise_population(self, *args, **kwargs) -> None:
        pass

    def issue_bed_days_according_to_availability(self, *args, **kwargs) -> None:
        pass

    def on_birth(self, *args, **kwargs) -> None:
        pass

    def on_start_of_day(self, *args, **kwargs) -> None:
        pass

    def on_end_of_day(self, *args, **kwargs) -> None:
        pass

    def on_end_of_year(self, **kwargs) -> None:
        pass

    def on_simulation_end(self, *args, **kwargs) -> None:
        pass
