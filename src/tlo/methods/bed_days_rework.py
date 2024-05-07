from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from tlo import Date

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
    occupied_beds: pd.DataFrame

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

    def is_inpatient(self, person_id: int) -> bool:
        """
        Return True if the person with the given index is an inpatient.
        
        A person is an inpatient if they are currently occupying a bed
        """

    def occupancy_ends(self, occupancy: BedOccupancy) -> None:
        """
        Action taken when a bed occupancy in the list of occupancies
        is to end, for any reason.

        The occupancy is removed from the list of events, and the
        resources it consumed are made available again.

        :param occupancy: The BedOccupancy that has ended.
        """
        self.occupancies.remove(occupancy)
        # Reallocate the occupancy's resources
        self.occupied_beds.loc[occupancy.facility, occupancy.bed_type] -= 1

    def occupancy_starts(self, occupancy: BedOccupancy) -> None:
        """
        Action taken when a scheduled bed occupancy 
        """
        self.occupied_beds.loc[occupancy.facility, occupancy.bed_type] += 1

    def start_of_day(self, todays_date: Date) -> None:
        """
        Actions to take at the start of a new day.

        - End any bed occupancies that are set to be freed today.
        """
        # End bed occupancies that are set to expire today.
        for expired_occupancy in [
            o for o in self.occupancies if o.freed_date <= todays_date
        ]:
            self.occupancy_ends(expired_occupancy)

        # Start bed occupancies that are due to start today.
        for starting_occupancy in [
            o for o in self.occupancies if o.start_date 
        ]:
            self.occupied_beds

    def end_occupancies_for_people(
        self, *person_id: int,
    ) -> None:
        """
        End all bed occupancies for the given person(s).
        
        Typical use case is when a death occurs in the simulation.
        """
        for id in person_id:
            for occupancy in [o for o in self.occupancies if o.patient_id == id]:
                self.occupancy_ends(occupancy)
