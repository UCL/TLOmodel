"""
Todo:   - speed up
        - phase-out use of facility_name and replace with facility_id
        - re-organised resource files: bring in one united file, and then decompose following read_parameters
        - bed days parameterisation
        - let the level of the appointment be in the log
        - move things to the hsi base class
        - let the logger give times of each hcw
"""

import heapq as hp
import inspect
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

import tlo
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.dxmanager import DxManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FacilityInfo(NamedTuple):
    """Information about a specific health facility."""
    id: int
    name: str


class AppointmentSubunit(NamedTuple):
    """Component of an appointment relating to a specific officer type."""
    officer_type: str
    time_taken: float


class HSIEventDetails(NamedTuple):
    """Non-target specific details of a health system interaction event."""
    event_name: str
    module_name: str
    treatment_id: str
    facility_level: Optional[int]
    appt_footprint: Tuple


class HealthSystem(Module):
    """
    This is the Health System Module
    Version: September 2019
    The execution of all health systems interactions are controlled through this module.
    """

    # Define the types of bed that exist. The order give the sequences of bed that are used
    # (i.e. in descending order of intensiveness)
    bed_types = [
        "high_dependency_bed",
        "general_bed",
        "non_bed_space",
    ]

    PARAMETERS = {
        'Officer_Types': Parameter(Types.DATA_FRAME, 'The names of the types of health workers ("officers")'),
        'Daily_Capabilities': Parameter(
            Types.DATA_FRAME, 'The capabilities by facility and officer type available each day'
        ),
        'Appt_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the type of appointments with the health system'),
        'Appt_Time_Table': Parameter(
            Types.DICT, 'The time taken for each appointment, according to officer and facility type.'
        ),
        'ApptType_By_FacLevel': Parameter(
            Types.LIST, 'Indicates whether an appointment type can occur at a facility level.'
        ),
        'Master_Facilities_List': Parameter(Types.DATA_FRAME, 'Listing of all health facilities.'),
        'Facilities_For_Each_District': Parameter(
            Types.DICT,
            'Mapping between a district and all of the health facilities to which its \
                      population have access.',
        ),
        'BedCapacity': Parameter(Types.DATA_FRAME, 'Bed-days capacity by facility'),
        'Consumables': Parameter(Types.DATA_FRAME, 'List of consumables used in each intervention and their costs.'),
        'Consumables_Cost_List': Parameter(Types.DATA_FRAME, 'List of each consumable item and it' 's cost'),
        'Service_Availability': Parameter(Types.LIST, 'List of services to be available.')
    }

    PROPERTIES = {
        'hs_dist_to_facility': Property(
            Types.REAL, 'The distance for each person to their nearest clinic (of any type)'
        ),
        'hs_is_inpatient': Property(
            Types.BOOL, 'Whether or not the person is currently an in-patient at any medical facility'
        ),
    }

    # add in internal properties for entry/exit to different types of bed
    for bed_type in bed_types:
        PROPERTIES[f"hs_next_first_day_in_bed_{bed_type}"] = Property(
            Types.DATE, f"Date when person will next enter bed_type {bed_type}. (pd.NaT) is nothing scheduled")
        PROPERTIES[f"hs_next_last_day_in_bed_{bed_type}"] = Property(
            Types.DATE, f"Date of the last day in the next stay in bed_type {bed_type}. (pd.NaT) is nothing scheduled")

    def __init__(
        self,
        name=None,
        resourcefilepath=None,
        service_availability=None,  # must be a list of treatment_ids to allow
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        ignore_cons_constraints=False,  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        store_hsi_events_that_have_run=False,  # convenience function for debugging
        record_hsi_event_details=False,  # Whether to record details of HSI events used
    ):

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert type(ignore_cons_constraints) is bool
        self.ignore_cons_constraints = ignore_cons_constraints

        assert type(disable) is bool
        assert type(disable_and_reject_all) is bool
        assert not (disable and disable_and_reject_all), 'Cannot have both disable and disable_and_reject_all selected'
        self.disable = disable
        self.disable_and_reject_all = disable_and_reject_all

        assert mode_appt_constraints in [0, 1, 2]  # Mode of constraints
        # 0: no constraints -- all HSI Events run with no squeeze factor
        # 1: elastic -- all HSI Events run - with squeeze factor
        # 2: hard -- only HSI events with no squeeze factor run

        self.mode_appt_constraints = mode_appt_constraints

        self.ignore_priority = ignore_priority

        # Store the argument provided for service_availability
        self.arg_service_availabily = service_availability
        self.service_availability = ['*']  # provided so that there is a default even before simulation is run

        # Check that the capabilities coefficient is correct
        assert capabilities_coefficient >= 0
        assert type(capabilities_coefficient) is float
        self.capabilities_coefficient = capabilities_coefficient

        # Define (empty) list of registered disease modules (filled in at `initialise_simulation`)
        self.recognised_modules_names = []

        # Define the container for calls for health system interaction events
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0  # Counter to help with the sorting in the heapq

        # Check 'store_hsi_events_that_have_run': will store a running list of HSI events that have run
        # (for debugging)
        assert isinstance(store_hsi_events_that_have_run, bool)
        self.store_hsi_events_that_have_run = store_hsi_events_that_have_run
        if self.store_hsi_events_that_have_run:
            self.store_of_hsi_events_that_have_run = list()

        # If record_hsi_event_details == True, a set will be built during the simulation
        # containing HSIEventDetails tuples corresponding to all HSI_Event instances
        # used in the simulation
        self.record_hsi_event_details = record_hsi_event_details
        if record_hsi_event_details:
            self.hsi_event_details = set()

        # Create store for ending in-patient status
        self.list_of_cols_with_internal_dates = dict()
        self.list_of_cols_with_internal_dates['entries'] = [
            f"hs_next_first_day_in_bed_{bed_type}" for bed_type in self.bed_types]
        self.list_of_cols_with_internal_dates['exits'] = [
            f"hs_next_last_day_in_bed_{bed_type}" for bed_type in self.bed_types]
        self.list_of_cols_with_internal_dates['all'] = \
            self.list_of_cols_with_internal_dates['entries'] + self.list_of_cols_with_internal_dates['exits']

        # Create the Diagnostic Test Manager to store and manage all Diagnostic Test
        self.dx_manager = DxManager(self)

    def read_parameters(self, data_folder):

        self.parameters['Officer_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Officer_Types_Table.csv'
        )

        self.parameters['Appt_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Types_Table.csv'
        )
        self._appointment_types = set(
            self.parameters['Appt_Types_Table']['Appt_Type_Code'])

        appt_time_data = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Time_Table.csv'
        )
        facility_levels = set(appt_time_data['Facility_Level'].unique())
        # Check that facility levels are consecutive integers starting from 0
        assert facility_levels == set(range(len(facility_levels)))
        # Store facility levels in module for check in schedule_hsi_event and for
        # check against levels in facilities per district resource file
        self._facility_levels = facility_levels
        # Store data as tuple of dicts, with tuple indexed by integer facility level and
        # dict indexed by string type code with values corresponding to list of (named)
        # tuples of appointment officer type codes and time takens
        appt_times_per_level_and_type = tuple(defaultdict(list) for _ in facility_levels)
        for appt_time_tuple in appt_time_data.itertuples():
            appt_times_per_level_and_type[
                appt_time_tuple.Facility_Level
            ][
                appt_time_tuple.Appt_Type_Code
            ].append(
                AppointmentSubunit(
                    officer_type=appt_time_tuple.Officer_Type_Code,
                    time_taken=appt_time_tuple.Time_Taken
                )
            )
        assert (
            sum(
                len(appt_info_list)
                for level in facility_levels
                for appt_info_list in appt_times_per_level_and_type[level].values()
            ) == len(appt_time_data)
        )
        self.parameters['Appt_Time_Table'] = appt_times_per_level_and_type

        appt_type_per_level_data = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_ApptType_By_FacLevel.csv'
        )
        self.parameters['ApptType_By_FacLevel'] = [
            set(
                appt_type_per_level_data['Appt_Type_Code'][
                    appt_type_per_level_data[f'Facility_Level_{i}']
                ]
            )
            for i in facility_levels
        ]

        mfl = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Master_Facilities_List.csv')
        self.parameters['Master_Facilities_List'] = mfl.iloc[:, 1:]  # get rid of extra column

        facilities_per_district_data = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Facilities_For_Each_District.csv'
        )
        districts = set(facilities_per_district_data['District'].unique())
        facility_levels = set(facilities_per_district_data['Facility_Level'].unique())
        # Check facility levels match those from appointment time table
        assert facility_levels == self._facility_levels, (
            "Mismatch between facility levels in Facilities_For_Each_District "
            "resource file and Appt_Time_Table resource files"
        )
        # Store data as tuple of dicts, with tuple indexed by integer facility level and
        # dict indexed by district name with values corresponding to (named) tuples of
        # facility ID and name
        facilities_per_level_and_district = tuple({} for _ in facility_levels)
        for facility_tuple in facilities_per_district_data.itertuples():
            facilities_per_level_and_district[
                facility_tuple.Facility_Level
            ][
                facility_tuple.District
            ] = FacilityInfo(
                id=facility_tuple.Facility_ID,
                name=facility_tuple.Facility_Name
            )
        assert all(d.keys() == districts for d in facilities_per_level_and_district), (
            "Facilities_For_Each_District resource file does not contain facilities "
            "at all levels for all districts"
        )
        self.parameters['Facilities_For_Each_District'] = facilities_per_level_and_district

        caps = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Daily_Capabilities.csv')
        self.parameters['Daily_Capabilities'] = caps.iloc[:, 1:]
        self.reformat_daily_capabilities()  # Reformats this table to include zero where capacity is not available
        # Store set of officers with non-zero daily availability for checking scheduled
        # HSI events do not make appointment time requests of unavailable officers
        self._officers_with_availability = set(
            self.parameters['Daily_Capabilities'].index[
                (self.parameters['Daily_Capabilities']['Total_Minutes_Per_Day'] > 0)
            ]
        )

        # Read in ResourceFile_Consumables and then process it to create the data structures needed
        # NB. Modules can use this to look-up what consumables they need.
        self.parameters['Consumables'] = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Consumables.csv')
        self.process_consumables_file()

        # Read in in-patient capacity
        self.parameters['BedCapacity'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Bed_Capacity.csv'
        ).iloc[:, 1:].set_index('Facility_Name')
        assert all([bed_type in self.parameters['BedCapacity'].columns for bed_type in self.bed_types])

        # Set default parameter for Service Availablity (everthing available)
        self.parameters['Service_Availability'] = ['*']

    def process_consumables_file(self):
        """Helper function for processing the consumables data (stored as self.parameters['Consumables'])
        * Creates ```parameters['Consumables']```
        * Creates ```df_mapping_pkg_code_to_intv_code```
        * Creates ```prob_item_code_available```
        * Creates ```parameters['Consumables_Cost_List]```
        """
        # Load the 'raw' ResourceFile_Consumabes that is loaded in to self.parameters['Consumables']
        raw = self.parameters['Consumables']

        # -------------------------------------------------------------------------------------------------
        # Create a pd.DataFrame that maps pkg code (as index) to item code:
        # This is used to quickly look-up which items are required in each package
        df = raw[['Intervention_Pkg_Code', 'Item_Code', 'Expected_Units_Per_Case']]
        df = df.set_index('Intervention_Pkg_Code')
        self.df_mapping_pkg_code_to_intv_code = df

        # -------------------------------------------------------------------------------------------------
        # Make ```prob_item_codes_available```
        # This is a data-frame that organise the probabilities of individual consumables items being available
        # (by the item codes)
        unique_item_codes = pd.DataFrame(data={'Item_Code': pd.unique(raw['Item_Code'])})

        # merge in probabilities of being available
        filter_col = [col for col in raw if col.startswith('Available_Facility_Level_')]
        filter_col.append('Item_Code')
        prob_item_codes_available = unique_item_codes.merge(
            raw.drop_duplicates(['Item_Code'])[filter_col], on='Item_Code', how='inner'
        )
        assert len(prob_item_codes_available) == len(unique_item_codes)

        # set the index as the Item_Code and save
        self.prob_item_codes_available = prob_item_codes_available.set_index('Item_Code', drop=True)

        # -------------------------------------------------------------------------------------------------
        # Create ```parameters['Consumables_Cost_List]```
        # This is a pd.Series, with index item_code, giving the cost of each item.
        self.parameters['Consumables_Cost_List'] = pd.Series(
            raw[['Item_Code', 'Unit_Cost']].drop_duplicates().set_index('Item_Code')['Unit_Cost']
        )

    def initialise_population(self, population):
        df = population.props

        # Assign hs_dist_to_facility'
        # (For now, let this be a random number, but in future it may be properly informed based on \
        #  population density distribitions)
        # Note that this characteritic is inherited from mother to child.
        df.loc[df.is_alive, 'hs_dist_to_facility'] = self.rng.uniform(0.01, 5.00, df.is_alive.sum())
        df.loc[df.is_alive, 'hs_is_inpatient'] = False

        # Put pd.NaT for all the properties concerned with entry/exit of different types of bed.
        df.loc[df.is_alive, self.list_of_cols_with_internal_dates['all']] = pd.NaT

    def initialise_simulation(self, sim):

        # Capture list of disease modules:
        self.recognised_modules_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_HEALTHSYSTEM in m.METADATA
        ]

        # Check that set of districts of residence in population are subset o districts from
        # Facilities_For_Each_District resource file
        pop = self.sim.population.props
        districts_of_residence = set(pop[pop.is_alive]["district_of_residence"].unique())
        assert all(
            districts_of_residence.issubset(per_level_facilities.keys())
            for per_level_facilities in self.parameters["Facilities_For_Each_District"]
        ), (
            "At least one district_of_residence value in population not present in "
            "Facilities_For_Each_District resource file"
        )

        # Launch the healthsystem scheduler (a regular event occurring each day) [if not disabled]
        if not (self.disable or self.disable_and_reject_all):
            sim.schedule_event(HealthSystemScheduler(self), sim.date)

        # Update consumables available today:
        self.determine_availability_of_consumables_today()

        # Initialise the bed-days-tracker:
        self.initialise_beddays_tracker()

        # Schedule the first (repeating) event to update status of hs_is_in_patient
        sim.schedule_event(RefreshInPatient(self), sim.date)

        # Determine service_availability
        self.set_service_availability()

    def set_service_availability(self):
        """Set service availability. (Should be equal to what is specified by the parameter, but overwrite with what was
         provided in arguement if an argument was specified -- provided for backward compatibility.)"""

        if self.arg_service_availabily is None:
            service_availability = self.parameters['Service_Availability']
        else:
            service_availability = self.arg_service_availabily

        assert type(service_availability) is list
        self.service_availability = service_availability

        # Log the service_availability
        logger.info(key="message",
                    data=f"Running Health System With the Following Service Availability: "
                         f"{self.service_availability}"
                    )

    def on_birth(self, mother_id, child_id):

        # New child inherits the hs_dist_to_facility of the mother
        df = self.sim.population.props
        df.at[child_id, 'hs_dist_to_facility'] = df.at[mother_id, 'hs_dist_to_facility']
        df.at[child_id, 'hs_is_inpatient'] = False
        df.loc[child_id, self.list_of_cols_with_internal_dates['all']] = pd.NaT

    def register_disease_module(self, new_disease_module):
        """
        This is now deprecated. Disease modules do not need to register with the health system.
        """
        raise NotImplementedError

    def schedule_hsi_event(self, hsi_event, priority, topen, tclose=None):
        """
        Schedule the health system interaction event

        :param hsi_event: the hsi_event to be scheduled
        :param priority: the priority for the hsi event (0 (highest), 1 or 2 (lowest)
        :param topen: the earliest date at which the hsi event should run
        :param tclose: the latest date at which the hsi event should run
        """

        logger.debug(
            key='message',
            data=f"HealthSystem.schedule_event >> "
                 f"Logging a request for an HSI: {hsi_event.TREATMENT_ID} for person: {hsi_event.target}"
        )

        assert isinstance(hsi_event, HSI_Event)

        # 0) Check if healthsystem is disabled
        if self.disable and (not self.disable_and_reject_all):
            # If healthsystem is disabled (but HSI can still run), ...
            #   ... put this event straight into the normal simulation scheduler.
            wrapped_hsi_event = HSIEventWrapper(hsi_event=hsi_event)
            self.sim.schedule_event(wrapped_hsi_event, topen)
            return  # Terminate this functional call
        elif self.disable_and_reject_all:
            # If healthsystem is disabled and HSI should not run, do nothing.
            return

        # 1) Check that this is a legitimate health system interaction (HSI) event

        if isinstance(hsi_event.target, tlo.population.Population):  # check if hsi_event is population-scoped
            # This is a population-scoped HSI event...
            # ... So it only needs TREATMENT_ID (the other items are ignored)
            assert ('TREATMENT_ID' in dir(hsi_event)) and (hsi_event.TREATMENT_ID != '')

        else:
            # This is an individual-scoped HSI event.
            # It must have EXPECTED_APPT_FOOTPRINT, BEDDAYS_FOOTPRINT, ACCEPTED_FACILITY_LEVELS and
            # ALERT_OTHER_DISEASES defined

            # Correctly formatted footprint
            assert ('TREATMENT_ID' in dir(hsi_event)) and (hsi_event.TREATMENT_ID != '')

            # Correct formated EXPECTED_APPT_FOOTPRINT
            assert 'EXPECTED_APPT_FOOTPRINT' in dir(hsi_event)
            assert self.appt_footprint_is_valid(hsi_event.EXPECTED_APPT_FOOTPRINT)

            # That it has an 'ACCEPTED_FACILITY_LEVEL' attribute
            # (Integer specificying the facility level at which HSI_Event must occur)
            assert 'ACCEPTED_FACILITY_LEVEL' in dir(hsi_event)
            assert type(hsi_event.ACCEPTED_FACILITY_LEVEL) is int
            assert hsi_event.ACCEPTED_FACILITY_LEVEL in self._facility_levels

            assert 'BEDDAYS_FOOTPRINT' in dir(hsi_event)
            self.check_beddays_footrpint_format(hsi_event.BEDDAYS_FOOTPRINT)

            # That it has a list for the other disease that will be alerted when it is run and that this make sense
            assert 'ALERT_OTHER_DISEASES' in dir(hsi_event)
            assert type(hsi_event.ALERT_OTHER_DISEASES) is list

            if len(hsi_event.ALERT_OTHER_DISEASES) > 0:
                if not (hsi_event.ALERT_OTHER_DISEASES[0] == '*'):
                    for d in hsi_event.ALERT_OTHER_DISEASES:
                        assert d in self.recognised_modules_names

            # Check that this can accept the squeeze argument
            assert 'squeeze_factor' in inspect.getfullargspec(hsi_event.run).args

        # 2) Check that topen, tclose and priority are valid

        # If there is no specified tclose time then set this to a week after topen
        if tclose is None:
            tclose = topen + DateOffset(days=7)

        # Check topen is not in the past
        assert topen >= self.sim.date

        # Check that topen is before tclose are not the same date
        assert topen < tclose

        # Check that priority is either 0, 1 or 2
        assert priority in {0, 1, 2}

        # 3) Check that this request is allowable under current policy (i.e. included in service_availability)
        allowed = False
        if not self.service_availability:  # it's an empty list
            allowed = False
        elif self.service_availability[0] == '*':  # it's the overall wild-card, do anything
            allowed = True
        elif hsi_event.TREATMENT_ID in self.service_availability:
            allowed = True
        elif hsi_event.TREATMENT_ID is None:
            allowed = True  # (if no treatment_id it can pass)
        elif hsi_event.TREATMENT_ID.startswith('GenericFirstAppt'):
            allowed = True  # allow all GenericFirstAppts
        else:
            # check to see if anything provided given any wildcards
            for s in self.service_availability:
                if '*' in s:
                    stub = s.split('*')[0]
                    if hsi_event.TREATMENT_ID.startswith(stub):
                        allowed = True
                        break

        # Further checks for HSI which are not population level events:
        if type(hsi_event.target) is not tlo.population.Population:

            # 4) Check that at least one type of appointment is required
            assert len(hsi_event.EXPECTED_APPT_FOOTPRINT) > 0, (
                'No appointment types required in the EXPECTED_APPT_FOOTPRINT'
            )
            # 5) Check that the event does not request an appointment at a facility level which is not possible
            appt_type_to_check_list = hsi_event.EXPECTED_APPT_FOOTPRINT.keys()
            facility_appt_types = self.parameters['ApptType_By_FacLevel'][hsi_event.ACCEPTED_FACILITY_LEVEL]
            assert facility_appt_types.issuperset(appt_type_to_check_list), (
                f"An appointment type has been requested at a facility level for which "
                f"it is not possible: {hsi_event.TREATMENT_ID}"
            )
            # 6) Check that event (if individual level) is able to run with this
            # configuration of officers (i.e. check that this does not demand officers
            # that are never available at a particular facility)
            footprint = self.get_appt_footprint_as_time_request(hsi_event=hsi_event)
            if not self._officers_with_availability.issuperset(footprint.keys()):
                logger.warning(
                    key="message",
                    data=(
                        "The expected footprint is not possible with the configuration "
                        f"of officers: {hsi_event.TREATMENT_ID}."
                    )
                )

        #  Manipulate the priority level if needed
        # If ignoring the priority in scheduling, then over-write the provided priority information
        if self.ignore_priority:
            priority = 0  # set all event to priority 0
        # This is where could attach a different priority score according to the treatment_id (and other things)
        # in order to examine the influence of the prioritisation score.

        # If all is correct and the hsi event is allowed then add this request to the queue of HSI_EVENT_QUEUE
        if allowed:

            # Create a tuple to go into the heapq
            # (NB. the sorting is done ascending and by the order of the items in the tuple)
            # Pos 0: priority,
            # Pos 1: topen,
            # Pos 2: hsi_event_queue_counter,
            # Pos 3: tclose,
            # Pos 4: the hsi_event itself

            new_request = (priority, topen, self.hsi_event_queue_counter, tclose, hsi_event)
            self.hsi_event_queue_counter += 1

            hp.heappush(self.HSI_EVENT_QUEUE, new_request)

            logger.debug(
                key="message",
                data=f"HealthSystem.schedule_event >> "
                     f"HSI has been added to the queue: {hsi_event.TREATMENT_ID} for person: {hsi_event.target}"
            )

        else:
            # HSI is not available under the services_available parameter: call the hsi's not_available() method if it
            # exists:
            try:
                hsi_event.not_available()
                # TODO: should the healthsystem call this at the time that the HSI was intended to be run (i.e topen)?

            except AttributeError:
                pass

            logger.debug(
                key="message",
                data=f"A request was made for a service but it was not included in the service_availability list:"
                     f" {hsi_event.TREATMENT_ID}"
            )

    def appt_footprint_is_valid(self, appt_footprint):
        """
        Checks an appointment footprint to ensure it is in the correct format.
        :param appt_footprint: Appointment footprint to check.
        :return: True if valid and False otherwise.
        """
        # Check that all keys known appointment types and all values non-negative
        return isinstance(appt_footprint, dict) and all(
            k in self._appointment_types and v > 0
            for k, v in appt_footprint.items()
        )

    def broadcast_healthsystem_interaction(self, hsi_event):
        """
        This will alert disease modules than a treatment_id is occurring to a particular person.
        :param hsi_event: the health system interaction event
        """

        if not hsi_event.ALERT_OTHER_DISEASES:  # it's an empty list
            # There are no disease modules to alert, so do nothing
            pass

        else:
            # Alert some disease modules

            if hsi_event.ALERT_OTHER_DISEASES[0] == '*':
                alert_modules = self.recognised_modules_names
            else:
                alert_modules = hsi_event.ALERT_OTHER_DISEASES

            # Alert each of the modules
            for module_name in alert_modules:
                # Don't notify originating module
                if not hsi_event.module.name == module_name:
                    self.sim.modules[module_name].on_hsi_alert(
                        person_id=hsi_event.target, treatment_id=hsi_event.TREATMENT_ID
                    )

    def reformat_daily_capabilities(self):
        """
        This will updates the dataframe for the self.parameters['Daily_Capabilities'] so as to include
        every permutation of officer_type_code and facility_id, with zeros against permuations where no capacity
        is available.

        It also give the dataframe an index that is useful for merging on (based on Facility_ID and Officer Type)

        (This is so that its easier to track where demands are being placed where there is no capacity)
        """

        # Get the capabilities data as they are imported
        capabilities = self.parameters['Daily_Capabilities']

        # Create dataframe containing background information about facility and officer types
        facility_ids = self.parameters['Master_Facilities_List']['Facility_ID'].values
        officer_type_codes = self.parameters['Officer_Types_Table']['Officer_Type_Code'].values

        facs = list()
        officers = list()
        for f in facility_ids:
            for o in officer_type_codes:
                facs.append(f)
                officers.append(o)

        capabilities_ex = pd.DataFrame(data={'Facility_ID': facs, 'Officer_Type_Code': officers})

        # Merge in information about facility from Master Facilities List
        mfl = self.parameters['Master_Facilities_List']
        capabilities_ex = capabilities_ex.merge(mfl, on='Facility_ID', how='left')

        # Merge in information about officers
        officer_types = self.parameters['Officer_Types_Table'][['Officer_Type_Code', 'Officer_Type']]
        capabilities_ex = capabilities_ex.merge(officer_types, on='Officer_Type_Code', how='left')

        # Merge in the capabilities (minutes available) for each officer type (inferring zero minutes where
        # there is no entry in the imported capabilities table)
        capabilities_ex = capabilities_ex.merge(
            capabilities[['Facility_ID', 'Officer_Type_Code', 'Total_Minutes_Per_Day']],
            on=['Facility_ID', 'Officer_Type_Code'],
            how='left',
        )
        capabilities_ex = capabilities_ex.fillna(0)

        # Give the standard index:
        capabilities_ex = capabilities_ex.set_index(
            'FacilityID_'
            + capabilities_ex['Facility_ID'].astype(str)
            + '_Officer_'
            + capabilities_ex['Officer_Type_Code']
        )

        # Checks
        assert abs(capabilities_ex['Total_Minutes_Per_Day'].sum() - capabilities['Total_Minutes_Per_Day'].sum()) < 1e-7
        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)

        # Apply the capabilities coefficient
        capabilities_ex['Total_Minutes_Per_Day'] *= self.capabilities_coefficient

        # Updates the capabilities table with the reformatted version
        self.parameters['Daily_Capabilities'] = capabilities_ex

    def get_capabilities_today(self):
        """
        Get the capabilities of the health system today.
        returns: DataFrame giving minutes available for each officer type in each facility type

        Functions can go in here in the future that could expand the time available,
        simulating increasing efficiency (the concept of a productivitiy ratio raised
        by Martin Chalkley).

        For now this method does nothing.

        Note that the scaling by `capabilities_coefficient` is applied on the initial
        processing of the `Daily_Capabilities` data in `reformat_daily_capabilities`.
        """
        return self.parameters['Daily_Capabilities']

    def get_blank_appt_footprint(self):
        """
        This is a helper function so that disease modules can easily create their appt_footprints.
        It returns an empty Counter instance.

        """
        return Counter()

    def get_blank_cons_footprint(self):
        """
        This is a helper function so that disease modules can easily create their cons_footprints.
        It returns a dictionary containing the consumables information in the format that the /
        HealthSystemScheduler expects.

        Format is as follows:
            * dict with two keys; Intervention_Package_Code and Item_Code
            * the value for each is a dict of the form (package_code or item_code): quantity
            * the codes within each list must be unique and valid codes; quantities must be integer values >0

            e.g.
            cons_req_as_footprint = {
                        'Intervention_Package_Code': {my_pkg_code: 1},
                        'Item_Code': {my_item_code: 10, another_item_code: 1}
            }
        """

        blank_footprint = {'Intervention_Package_Code': {}, 'Item_Code': {}}
        return blank_footprint

    def get_prob_seek_care(self, person_id, symptom_code=0):
        """
        This is depracted. Report onset of generic acute symptoms to the symptom mananger.
        HealthSeekingBehaviour module will schedule a generic hsi.
        """
        raise Exception('Do not use get_prob_seek_care().')

    def get_facility_info(self, hsi_event) -> FacilityInfo:
        """Helper function to find the facility at which an HSI event will take place"""
        # Gather information about the HSI event
        the_district = self.sim.population.props.at[
            hsi_event.target, 'district_of_residence']
        the_level = hsi_event.ACCEPTED_FACILITY_LEVEL

        # Return the (one) health_facility available to this person (based on their
        # district), which is accepted by the hsi_event.ACCEPTED_FACILITY_LEVEL
        return self.parameters["Facilities_For_Each_District"][the_level][the_district]

    def get_appt_footprint_as_time_request(self, hsi_event, actual_appt_footprint=None):
        """
        This will take an HSI event and return the required appointments in terms of the
        time required of each Officer Type in each Facility ID.
        The index will identify the Facility ID and the Officer Type in the same format
        as is used in Daily_Capabilities.

        :param hsi_event: The HSI event
        :param actual_appt_footprint: The actual appt footprint (optional) if different
            to that in the HSI event.
        :return: A Counter that gives the times required for each officer-type in each
            facility_ID, where this time is non-zero.
        """
        # If specified use actual_appt_footprint otherwise use EXPECTED_APPT_FOOTPRINT
        the_appt_footprint = (
            hsi_event.EXPECTED_APPT_FOOTPRINT if actual_appt_footprint is None else
            actual_appt_footprint
        )
        # Check in time request cache in event if a time request corresponding to the
        # current relevant event attributes and appointment footprint has previously
        # been stored, and if so return this
        cache_key = (
            hsi_event.target,
            hsi_event.ACCEPTED_FACILITY_LEVEL,
            tuple(the_appt_footprint)
        )
        cached_time_request = hsi_event._cached_time_requests.get(cache_key)
        if cached_time_request is not None:
            return cached_time_request

        # No entry in cache so compute time request

        # Appointment times for each facility and officer combination
        appt_times = self.parameters['Appt_Time_Table']

        # Facility level required by this event
        the_facility_level = hsi_event.ACCEPTED_FACILITY_LEVEL

        # Get the (one) health_facility available to this person (based on their
        # district), which is accepted by the hsi_event.ACCEPTED_FACILITY_LEVEL:
        the_facility = self.get_facility_info(hsi_event)

        # Accumulate appointment times for specified footprint using times from
        # appointment times table
        appt_footprint_times = Counter()
        for appt_type in the_appt_footprint:
            try:
                appt_info_list = appt_times[the_facility_level][appt_type]
            except KeyError as e:
                raise KeyError(
                    f"The time needed for this appointment is not defined for this "
                    f"specified facility level in the Appt_Time_Table. "
                    f"Event treatment ID: {hsi_event.TREATMENT_ID}"
                ) from e
            for appt_info in appt_info_list:
                appt_footprint_times[
                    f"FacilityID_{the_facility.id}_Officer_{appt_info.officer_type}"
                ] += appt_info.time_taken

        # Cache the time request to avoid having to recompute on subsequent calls
        hsi_event._cached_time_requests[cache_key] = appt_footprint_times

        return appt_footprint_times

    def get_squeeze_factors(self, footprints_per_event, total_footprint, current_capabilities):
        """
        This will compute the squeeze factors for each HSI event from the list of all
        the calls on health system resources for the day.
        The squeeze factor is defined as (call/available - 1). ie. the highest
        fractional over-demand among any type of officer that is called-for in the
        appt_footprint of an HSI event.
        A value of 0.0 signifies that there is no squeezing (sufficient resources for
        the EXPECTED_APPT_FOOTPRINT).
        A value of 99.99 signifies that the call is for an officer_type in a
        health-facility that is not available.

        :param footprints_per_event: List, one entry per HSI event, containing the
            minutes required from each health officer in each health facility as a
            Counter (using the standard index)
        :param total_footprint: Counter, containing the total minutes required from
            each health officer in each health facility when non-zero, (using the
            standard index)
        :param current_capabilities: Dataframe giving the amount of time available from
            each health officer in each health facility (using the standard index)

        :return: squeeze_factors: an array of the squeeze factors for each HSI event
            (position in array matches that in the all_call_today list).
        """

        # 1) Compute the load factors for each officer type at each facility that is
        # called-upon in this list of HSIs
        total_available = current_capabilities['Total_Minutes_Per_Day']
        load_factor = {}
        for officer, call in total_footprint.items():
            availability = total_available.get(officer)
            if availability is None:
                load_factor[officer] = 99.99
            elif availability == 0:
                load_factor[officer] = float('inf')
            else:
                load_factor[officer] = max(call / availability - 1, 0)

        # 5) Convert these load-factors into an overall 'squeeze' signal for each HSI,
        # based on the highest load-factor of any officer required (or zero if event
        # has an empty footprint)
        squeeze_factor_per_hsi_event = np.array([
            max((load_factor[officer] for officer in footprint), default=0)
            for footprint in footprints_per_event
        ])

        assert (squeeze_factor_per_hsi_event >= 0).all()

        return squeeze_factor_per_hsi_event

    def request_consumables(self, hsi_event, cons_req_as_footprint, to_log=True):
        """
        This is where HSI events can check access to and log use of consumables.
        The healthsystem module will check if that consumable is available
        at this time and at that facility and return a True/False response. If a package is requested, it is considered
        to be available only if all of the constituent items are available.
        All requests are logged and, by default, it is assumed that if a requested consumable is available then it
        is used. Alternatively, HSI Events can just query if a
        consumable is available (without using it) by setting to_log=False.

        :param cons_req: The consumable that is requested, in the format specified in 'get_blank_cons_footprint()'
        :param to_log: Indicator to show whether this should not be logged (defualt: True)
        :return: In the same format of the provided footprint, giving a bool for each package or item returned
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Could check the format of the cons_req_as_footprint
        # It is removed as this is a time-consuming check that is rarely required
        # self.check_consumables_footprint_format(cons_req_as_footprint)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the healthsystem module is disabled, return True for all consumables without checking or logging
        if self.disable:
            return {
                'Intervention_Package_Code': dict(zip(
                    cons_req_as_footprint['Intervention_Package_Code'].keys(),
                    [True] * len(cons_req_as_footprint['Intervention_Package_Code'].keys()
                                 ))),
                'Item_Code': dict(zip(
                    cons_req_as_footprint['Item_Code'].keys(),
                    [True] * len(cons_req_as_footprint['Item_Code'].keys()
                                 )))
                }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine availability of each item and package:
        select_col = f'Available_Facility_Level_{hsi_event.ACCEPTED_FACILITY_LEVEL}'
        available = {
            'Intervention_Package_Code': self.cons_available_today['Intervention_Package_Code'].loc[
                cons_req_as_footprint['Intervention_Package_Code'].keys(), select_col].to_dict(),
            'Item_Code': self.cons_available_today['Item_Code'].loc[
                cons_req_as_footprint['Item_Code'].keys(), select_col].to_dict()
        }

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Logging:
        if to_log:
            treatment_id = hsi_event.TREATMENT_ID
            # NB. Casting the data to strings because logger complains with dict of varying sizes/keys
            logger.info(key='Consumables',
                        data={
                            'TREATMENT_ID': treatment_id,
                            'Item_Available': str({k: v for k, v in cons_req_as_footprint['Item_Code'].items()
                                                   if available['Item_Code'][k]}
                                                  ),
                            'Item_NotAvailable': str({k: v for k, v in cons_req_as_footprint['Item_Code'].items()
                                                      if not available['Item_Code'][k]}
                                                     ),
                            'Package_Available': str({k: v for k, v in
                                                      cons_req_as_footprint['Intervention_Package_Code'].items()
                                                      if available['Intervention_Package_Code'][k]}
                                                     ),
                            'Package_NotAvailable': str({k: v for k, v in
                                                         cons_req_as_footprint['Intervention_Package_Code'].items()
                                                         if not available['Intervention_Package_Code'][k]}
                                                        )
                        },
                        description="record of each consumable item that is requested by in an HSI"
                        )

        # return the result of the check on availability
        return available

    def check_consumables_footprint_format(self, cons_req_as_footprint):
        """
        This function runs some check on the cons_req_as_footprint to ensure its in the right format
        :param cons_req_as_footprint:
        :return:
        """
        # Format is as follows:
        #     * dict with two keys; Intervention_Package_Code and Item_Code
        #     * For each, there is list of dicts, each dict giving code (i.e. package_code or item_code):quantity
        #     * the codes within each list must be unique and valid codes, quantities must be integer values >0
        #     e.g.
        #     cons_req_as_footprint = {
        #                 'Intervention_Package_Code': {my_pkg_code: 1},
        #                 'Item_Code': {my_item_code: 10}, {another_item_code: 1}
        #     }

        # check basic formatting
        format_error_str = 'The consumable_footprint is not in the right format. ' \
                           'See check_consumables_footprint_format.'
        assert type(cons_req_as_footprint) is dict
        assert 'Intervention_Package_Code' in cons_req_as_footprint, format_error_str
        assert 'Item_Code' in cons_req_as_footprint, format_error_str
        assert type(cons_req_as_footprint['Intervention_Package_Code']) is dict, format_error_str
        assert type(cons_req_as_footprint['Item_Code']) is dict, format_error_str

        # Check that consumables being required are in the database

        # Check packages
        all_pkgs = self.df_mapping_pkg_code_to_intv_code.index.values
        for pkg_code, pkg_quant in cons_req_as_footprint['Intervention_Package_Code'].items():
            assert pkg_code in all_pkgs, f'Intervention_Package_Code {pkg_code} not recognised'
            assert pkg_code != -99, 'Intervention_Package_Code cannot be -99'
            assert pkg_quant > 0, format_error_str

        # Check items
        all_items = pd.unique(self.df_mapping_pkg_code_to_intv_code['Item_Code'])
        for itm_code, itm_quant in cons_req_as_footprint['Item_Code'].items():
            assert itm_code in all_items, f'Item_Code {itm_code} not recognised'
            assert itm_quant > 0, format_error_str

    def get_consumables_as_individual_items(self, cons_req_as_footprint):
        """
        Helper function to decompose a ```cons_req_as_footprint``` and return a pd.Series with the individual items
        (as the index) and the quantity needed (as the value).
        """

        # Unpack package_code (repeat package code if the package is required multiple times)
        pkgs = list()
        for k, v in cons_req_as_footprint['Intervention_Package_Code'].items():
            for i in range(v):
                pkgs.append(k)

        # Get the corresponding item codes and flip in to a series with index of Item_Code
        x = self.df_mapping_pkg_code_to_intv_code.loc[pkgs].set_index('Item_Code')['Expected_Units_Per_Case']

        # Add in individual consumables in one go as a pd.Series
        x = x.append(pd.Series(cons_req_as_footprint['Item_Code']))

        # Return de-duplicated Series (index=Item_Code, value=quantity)
        return x.groupby(x.index).sum()

    def determine_availability_of_consumables_today(self):
        """Helper function to determine availability of all items and packages"""

        # Determine the availability of the consumables *items* today
        if not self.ignore_cons_constraints:
            # Random draws: assume that availability of the same item is independent between different facility levels
            random_draws = self.rng.rand(
                len(self.prob_item_codes_available), len(self.prob_item_codes_available.columns)
            )
            items = self.prob_item_codes_available > random_draws
        else:
            # Make all true if ignoring consumables constraints
            items = self.prob_item_codes_available > 0.0

        # Determine the availability of packages today
        # (packages are made-up of the individual items: if one item is not available, the package is not available)
        pkgs = self.df_mapping_pkg_code_to_intv_code.merge(items, left_on='Item_Code', right_index=True)
        pkgs = pkgs.groupby(level=0)[pkgs.columns[pkgs.columns.str.startswith('Available_Facility_Level')]].all()

        self.cons_available_today = {
            "Item_Code": items,
            "Intervention_Package_Code": pkgs
        }

    def get_blank_beddays_footprint(self):
        """
        Helper function to generate a blank footprint for the bed-days required by an HSI
        :return: a footprint of the correct format, specifying no bed days
        """
        return dict(zip(self.bed_types, [0] * len(self.bed_types)))

    def check_beddays_footrpint_format(self, beddays_footprint):
        """Check that the format of the beddays footprint (provided by an HSI) is correct"""
        assert type(beddays_footprint) is dict
        assert len(self.bed_types) == len(beddays_footprint)
        assert all([(bed_type in beddays_footprint) for bed_type in self.bed_types])
        assert all([((v >= 0) and (type(v) is int)) for v in beddays_footprint.values()])
        assert beddays_footprint['non_bed_space'] == 0, "A request cannot be made for this type of bed"

    def log_hsi_event(self, hsi_event, actual_appt_footprint=None, squeeze_factor=None, did_run=True):
        """
        This will write to the log with a record that this HSI event has occured.
        If this is an individual-level HSI event, it will also record the actual appointment footprint
        :param hsi_event: The hsi event (containing the initial expectations of footprints)
        :param actual_appt_footprint: The actual appt footprint to log (if individual event)
        :param squeeze_factor: The squueze factor (if individual event)
        """

        if isinstance(hsi_event.target, tlo.population.Population):
            # Population HSI-Event
            log_info = dict()
            log_info['TREATMENT_ID'] = hsi_event.TREATMENT_ID
            log_info['Number_By_Appt_Type_Code'] = 'Population'  # remove the appt-types with zeros
            log_info['Person_ID'] = -1  # Junk code
            log_info['Squeeze_Factor'] = 0

        else:
            # Individual HSI-Event:
            assert actual_appt_footprint is not None
            assert squeeze_factor is not None

            log_info = dict()
            log_info['TREATMENT_ID'] = hsi_event.TREATMENT_ID
            # key appointment types that are non-zero
            log_info['Number_By_Appt_Type_Code'] = actual_appt_footprint
            log_info['Person_ID'] = hsi_event.target

            if squeeze_factor == np.inf:
                log_info['Squeeze_Factor'] = 100.0  # arbitrarily high value to replace infinity
            else:
                log_info['Squeeze_Factor'] = squeeze_factor

        log_info['did_run'] = did_run

        logger.info(key="HSI_Event",
                    data=log_info,
                    description="record of each HSI event")

        if self.store_hsi_events_that_have_run:
            log_info['date'] = self.sim.date
            self.store_of_hsi_events_that_have_run.append(log_info)
        if self.record_hsi_event_details:
            self.hsi_event_details.add(
                HSIEventDetails(
                    event_name=type(hsi_event).__name__,
                    module_name=type(hsi_event.module).__name__,
                    treatment_id=hsi_event.TREATMENT_ID,
                    facility_level=getattr(
                        hsi_event, 'ACCEPTED_FACILITY_LEVEL', None
                    ),
                    appt_footprint=(
                        tuple(actual_appt_footprint)
                        if actual_appt_footprint is not None
                        else tuple(getattr(hsi_event, 'EXPECTED_APPT_FOOTPRINT', {}))
                    )
                )
            )

    def log_current_capabilities(self, current_capabilities, total_footprint):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type
        NB. To get this per Officer_Type_Code, it would be possible to simply log the entire current_capabilities df.
        :param current_capabilities: the current_capabilities of the health system.
        :param total_footprint: Per-officer totals of footprints of all the HSI events that ran
        """

        # Combine the current_capabiliites and total_footprint per-officer totals
        total_calls_per_officer = pd.Series(total_footprint, dtype='float64')
        comparison = current_capabilities[['Facility_ID', 'Total_Minutes_Per_Day']].copy()
        comparison['Minutes_Used'] = total_calls_per_officer
        comparison['Minutes_Used'].fillna(0, inplace=True)
        total_calls = total_calls_per_officer.sum()

        assert len(comparison) == len(current_capabilities)
        assert np.isclose(comparison['Minutes_Used'].sum(), total_calls)

        # Sum within each Facility_ID using groupby (Index of 'summary' is Facility_ID)
        summary = comparison.groupby('Facility_ID')[['Total_Minutes_Per_Day', 'Minutes_Used']].sum()

        # Compute Fraction of Time Used Across All Facilities
        total_available = summary['Total_Minutes_Per_Day'].sum()
        fraction_time_used_across_all_facilities = (
            total_calls / total_available if total_available > 0
            else 0  # no capabilities or nan arising
        )

        # Compute Fraction of Time Used In Each Facility
        summary['Fraction_Time_Used'] = summary['Minutes_Used'] / summary['Total_Minutes_Per_Day']
        summary['Fraction_Time_Used'].replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        log_capacity = dict()
        log_capacity['Frac_Time_Used_Overall'] = fraction_time_used_across_all_facilities
        log_capacity['Frac_Time_Used_By_Facility_ID'] = summary['Fraction_Time_Used'].to_dict()

        logger.info(key='Capacity',
                    data=log_capacity,
                    description='daily summary of utilisation and capacity of health system resources')

    def find_events_for_person(self, person_id: int):
        """Find the events in the HSI_EVENT_QUEUE for a particular person.
        :param person_id: the person_id of interest
        :returns list of tuples (date_of_event, event) for that person_id in the HSI_EVENT_QUEUE.

        NB. This is for debugging and testing only - not for use in real simulations as it is slow
        """
        list_of_events = list()

        for ev_tuple in self.HSI_EVENT_QUEUE:
            date = ev_tuple[1]   # this is the 'topen' value
            event = ev_tuple[4]
            if isinstance(event.target, (int, np.integer)):
                if event.target == person_id:
                    list_of_events.append((date, event))

        return list_of_events

    def initialise_beddays_tracker(self):
        """Initialise the bed days tracker.
        Create a dataframe for each type of beds that give the total number of beds currently available in each facility
         (rows) by the date during the simulation (columns).

        It is decremented by 1 for each person occupying a bed for one day.

        The current implementation assumes that bed capacity is held constant throughout the simulation; but it could be
         changed through modifications here.
        """
        self.bed_tracker = dict()
        for bed_type in self.bed_types:
            df = pd.DataFrame(
                index=pd.date_range(self.sim.start_date, self.sim.end_date, freq='D'),
                columns=self.parameters['BedCapacity'].index,
                data=1
            )
            df = df.mul(self.parameters['BedCapacity'][bed_type], axis=1)
            assert not df.isna().any().any()
            self.bed_tracker[bed_type] = df

    def impose_beddays_footprint(self, hsi_event):
        """This is called to reflect that a new occupany of bed-days should be recorded:
        * Cause to be reflected in the bed_tracker that an hsi_event is being run that will cause bed to be
         occupied.
        * Update the property ```hs_is_inpatient``` to show that this person is now an in-patient

         NB. If multiple bed types are required, then it is assumed that these run in the sequence given in
         ```bed_types```.
         """
        df = self.sim.population.props
        new_footprint = hsi_event.BEDDAYS_FOOTPRINT

        if not type(hsi_event.target) is int:
            # If this is a population level event, do nothing
            return

        person_id = hsi_event.target

        def apply_footprint(footprint):
            """Edit the internal properties in the dataframe to reflect this in-patient stay"""
            # reset all internal properties about dates of transition between bed use states:
            df.loc[person_id, self.list_of_cols_with_internal_dates['all']] = pd.NaT

            # compute the entry/exit dates of bed use states:
            dates_of_bed_use = self.compute_dates_of_bed_use(footprint)

            # record these dates in the internal properties:
            df.loc[person_id, dates_of_bed_use.keys()] = dates_of_bed_use.values()

            # enter these to tracker
            self.edit_bed_tracker(
                dates_of_bed_use=dates_of_bed_use,
                person_id=person_id,
                add_footprint=True
            )

        if not df.at[person_id, 'hs_is_inpatient']:
            # apply the new footprint if the person is not already an in-patient
            apply_footprint(new_footprint)
            # label person as an in-patient
            df.at[person_id, 'hs_is_inpatient'] = True

        else:
            # if person is already an in-patient:
            # calculate how much time left in each bed remains for the person who is already an in-patient
            remaining_footprint = self.get_remaining_footprint(person_id)

            # combine the remaining footprint with the new footprint, with days in each bed-type running concurrently:
            combo_footprint = {bed_type: max(new_footprint[bed_type], remaining_footprint[bed_type])
                               for bed_type in self.bed_types
                               }

            # remove the old footprint and apply the combined footprint
            self.remove_beddays_footprint(person_id)
            apply_footprint(combo_footprint)

    def edit_bed_tracker(self, dates_of_bed_use, person_id, add_footprint=True):
        """Helper function to record the usage (or freeing-up) of beds in the bed-tracker
        dates_of_bed_use: the dates of entry/exit from beds of each type
        person_id: used to find the facility to use
        add_footprint: whether the footprint should be added (i.e. consume a bed), or the reversed (i.e. free a bed).
            The latter is used to when a footprint is removed when a person dies or before a new footprint is added.
        """
        # Exit silently if bed_tracker has not been initialised
        if 'bed_tracker' not in dir(self):
            return

        # determine the facility_id
        the_facility_name = 'National Hospital'  # todo - make this specific to the district/region using person_id
        operation = -1 if add_footprint else 1

        for bed_type in self.bed_types:
            date_start_this_bed = dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"]
            date_end_this_bed = dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"]
            self.bed_tracker[bed_type].loc[date_start_this_bed: date_end_this_bed, the_facility_name] += operation

    def compute_dates_of_bed_use(self, footprint):
        """Helper function to compute the dates of entry/exit from beds of each type according to a bed-days footprint
         (which provides information in terms of number of whole days).
        NB. It is always assumed that the footprint begins with today's date. """

        now = self.sim.date
        start_allbeds = now
        end_allbeds = now + pd.DateOffset(days=sum(footprint.values()) - 1)

        dates_of_bed_use = dict()

        start_this_bed = start_allbeds
        for bed_type in self.bed_types:
            if footprint[bed_type] > 0:
                end_this_bed = start_this_bed + pd.DateOffset(days=footprint[bed_type] - 1)

                # record these dates:
                dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"] = start_this_bed
                dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"] = end_this_bed

                # get ready for next bed type:
                start_this_bed = end_this_bed + pd.DateOffset(days=1)
            else:
                dates_of_bed_use[f"hs_next_first_day_in_bed_{bed_type}"] = pd.NaT
                dates_of_bed_use[f"hs_next_last_day_in_bed_{bed_type}"] = pd.NaT

        # check that dates run as expected
        assert (start_this_bed - pd.DateOffset(days=1)) == end_allbeds

        return dates_of_bed_use

    def get_remaining_footprint(self, person_id):
        """Helper function to work out how many days remaining in each bed-type of current stay for a given person."""
        df = self.sim.population.props
        now = self.sim.date

        d = df.loc[person_id, self.list_of_cols_with_internal_dates['all']]
        remaining_footprint = self.get_blank_beddays_footprint()

        for bed_type in self.bed_types:
            if d[f'hs_next_last_day_in_bed_{bed_type}'] >= now:
                remaining_footprint[bed_type] = 1 + \
                                                (
                                                    d[f'hs_next_last_day_in_bed_{bed_type}']
                                                    - max(now, d[f'hs_next_first_day_in_bed_{bed_type}'])
                                                ).days
                # NB. The '+1' accounts for the fact that 'today' is included

        return remaining_footprint

    def remove_beddays_footprint(self, person_id):
        """Helper function that will remove from the bed-days tracker the days of bed-days remaining for a person.
        This is called when the person dies or when a new footprint is imposed"""
        remaining_footprint = self.get_remaining_footprint(person_id)
        dates_of_bed_use = self.compute_dates_of_bed_use(remaining_footprint)
        self.edit_bed_tracker(
            dates_of_bed_use=dates_of_bed_use,
            person_id=person_id,
            add_footprint=False
        )

    def on_simulation_end(self):
        """Dump the record of the bed_tracker to the log"""
        for bed_type in self.bed_tracker:
            for index, row in self.bed_tracker[bed_type].iterrows():
                row.index = row.index.astype(str)
                row['Facility_Name'] = index

                logger.info(
                    key=f'bed_tracker_{bed_type}',
                    data=row,
                    description=f'dataframe of bed_tracker of type {bed_type}, broken down by day and facility'
                )


class RefreshInPatient(RegularEvent, PopulationScopeEventMixin):
    """Refresh the hs_in_patient status. This event runs every day"""
    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        df = self.sim.population.props
        exit_cols = self.module.list_of_cols_with_internal_dates['exits']

        # if any "date of last day in bed" in not null and in the future, then the person is an in-patient:
        df.loc[df.is_alive, "hs_is_inpatient"] = \
            ((~df.loc[df.is_alive, exit_cols].isnull()) & ~(df.loc[df.is_alive, exit_cols] < self.sim.date)).any(axis=1)


class HealthSystemScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This is the HealthSystemScheduler. It is an event that occurs every day, inspects the calls on the healthsystem
    and commissions event to occur that are consistent with the healthsystem's capabilities for the following day, given
    assumptions about how this decision is made.
    The overall Prioritation algorithm is:
        * Look at events in order (the order is set by the heapq: see schedule_event
        * Ignore is the current data is before topen
        * Remove and do nothing if tclose has expired
        * Run any  population-level HSI events
        * For an individual-level HSI event, check if there are sufficient health system capabilities to run the event

    If the event is to be run, then the following events occur:
        * The HSI event itself is run.
        * The occurence of the event is logged
        * The resources used are 'occupied' (if individual level HSI event)
        * Other disease modules are alerted of the occurence of the HSI event (if individual level HSI event)

    Here is where we can have multiple types of assumption regarding how these capabilities are modelled.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):

        # 0) Determine the availability of consumables today based on their probabilities
        self.module.determine_availability_of_consumables_today()

        # Create hold-over list (will become a heapq).
        # This will hold events that cannot occur today before they are added back to the heapq
        hold_over = list()

        # 1) Get the events that are due today:
        list_of_individual_hsi_event_tuples_due_today = list()
        list_of_population_hsi_event_tuples_due_today = list()

        while len(self.module.HSI_EVENT_QUEUE) > 0:

            next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)
            # Read the tuple and assemble into a dict 'next_event'

            # Structure of tuple is:
            # Pos 0: priority,
            # Pos 1: topen,
            # Pos 2: hsi_event_queue_counter,
            # Pos 3: tclose,
            # Pos 4: the hsi_event itself

            event = next_event_tuple[4]

            if self.sim.date > next_event_tuple[3]:
                # The event has expired (after tclose) having never been run. Call the 'never_ran' function
                event.never_ran()

            elif (type(event.target) is not tlo.population.Population) \
                    and (not self.sim.population.props.at[event.target, 'is_alive']):
                # if individual level event and the person who is the target is no longer alive, do nothing more
                pass

            elif self.sim.date < next_event_tuple[1]:
                # The event is not yet due (before topen), add to the hold-over list
                hp.heappush(hold_over, next_event_tuple)

                if next_event_tuple[0] == 2:
                    # Check the priority
                    # If the next event is not due and has low priority, then stop looking through the heapq
                    # as all other events will also not be due.
                    break

            else:
                # The event is now due to run today and the person is confirmed to be still alive
                # Add it to the list of events due today (individual or population level)
                # NB. These list is ordered by priority and then due date

                is_pop_level_hsi_event = type(event.target) is tlo.population.Population
                if is_pop_level_hsi_event:
                    list_of_population_hsi_event_tuples_due_today.append(next_event_tuple)
                else:
                    list_of_individual_hsi_event_tuples_due_today.append(next_event_tuple)

        # 2) Run all population-level HSI events
        while len(list_of_population_hsi_event_tuples_due_today) > 0:
            pop_level_hsi_event_tuple = list_of_population_hsi_event_tuples_due_today.pop()

            pop_level_hsi_event = pop_level_hsi_event_tuple[4]
            pop_level_hsi_event.run(squeeze_factor=0)
            self.module.log_hsi_event(hsi_event=pop_level_hsi_event)

        # 3) Get the capabilities that are available today and prepare dataframe to store all the calls for today
        current_capabilities = self.module.get_capabilities_today()

        if not list_of_individual_hsi_event_tuples_due_today:
            # Empty counter for log_current_capabilities call below
            total_footprint = Counter()
        else:
            # 4) Examine total call on health officers time from the HSI events that are due today

            # For all events in 'list_of_individual_hsi_event_tuples_due_today',
            # expand the appt-footprint of the event to give the demands on
            # each officer-type in each facility_id.

            footprints_of_all_individual_level_hsi_event = [
                self.module.get_appt_footprint_as_time_request(hsi_event=(event_tuple[4]))
                for event_tuple in list_of_individual_hsi_event_tuples_due_today
            ]

            # Compute total appointment footprint across all events
            total_footprint = Counter()
            for footprint in footprints_of_all_individual_level_hsi_event:
                # Counter.update method when called with dict-like argument adds counts
                # from argument to Counter object called from
                total_footprint.update(footprint)

            # 5) Estimate Squeeze-Factors for today
            if self.module.mode_appt_constraints == 0:
                # For Mode 0 (no Constraints), the squeeze factors are all zero.
                squeeze_factor_per_hsi_event = np.zeros(
                    len(footprints_of_all_individual_level_hsi_event))
            else:
                # For Other Modes, the squeeze factors must be computed
                squeeze_factor_per_hsi_event = self.module.get_squeeze_factors(
                    footprints_per_event=footprints_of_all_individual_level_hsi_event,
                    total_footprint=total_footprint,
                    current_capabilities=current_capabilities,
                )

            # 6) For each event, determine if run or not, and run if so.
            for ev_num in range(len(list_of_individual_hsi_event_tuples_due_today)):
                event = list_of_individual_hsi_event_tuples_due_today[ev_num][4]
                squeeze_factor = squeeze_factor_per_hsi_event[ev_num]

                ok_to_run = (
                    (self.module.mode_appt_constraints == 0)
                    or (self.module.mode_appt_constraints == 1)
                    or ((self.module.mode_appt_constraints == 2) and (squeeze_factor == 0.0))
                )

                # Mode 0: All HSI Event run, with no squeeze
                # Mode 1: All Run With Squeeze
                # Mode 2: Only if squeeze <1

                if ok_to_run:

                    # Compute the fraction of the bed-days in the footprint that is available and log the usage:
                    # todo - this provides exactly the beddays that were requested,
                    #  ... but this will be where the check on availability is gated
                    event._received_info_about_bed_days = event.BEDDAYS_FOOTPRINT

                    # Run the HSI event (allowing it to return an updated appt_footprint)
                    actual_appt_footprint = event.run(
                        squeeze_factor=squeeze_factor
                    )

                    # Check if the HSI event returned updated appt_footprint
                    if actual_appt_footprint is not None:
                        # The returned footprint is different to the expected footprint: so must update load factors

                        # check its formatting:
                        assert self.module.appt_footprint_is_valid(actual_appt_footprint)

                        # Update load factors:
                        updated_call = self.module.get_appt_footprint_as_time_request(
                            event, actual_appt_footprint)
                        original_call = footprints_of_all_individual_level_hsi_event[ev_num]
                        footprints_of_all_individual_level_hsi_event[ev_num] = updated_call
                        total_footprint -= original_call
                        total_footprint += updated_call
                        if self.module.mode_appt_constraints != 0:
                            # only need to recompute squeeze factors if running with constraints
                            # i.e. mode != 0
                            squeeze_factor_per_hsi_event = self.module.get_squeeze_factors(
                                footprints_per_event=footprints_of_all_individual_level_hsi_event,
                                total_footprint=total_footprint,
                                current_capabilities=current_capabilities,
                            )
                    else:
                        # no actual footprint is returned so take the expected initial declaration as the actual
                        actual_appt_footprint = event.EXPECTED_APPT_FOOTPRINT

                    # Write to the log
                    self.module.log_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=actual_appt_footprint,
                        squeeze_factor=squeeze_factor,
                        did_run=True,
                    )

                else:
                    # Do not run,
                    # Call did_not_run for the hsi_event
                    rtn_from_did_not_run = event.did_not_run()

                    # If received no response from the call to did_not_run, or a True signal, then
                    # add to the hold-over queue.
                    # Otherwise (disease module returns "FALSE") the event is not rescheduled and will not run.

                    if not (rtn_from_did_not_run is False):
                        # reschedule event
                        hp.heappush(hold_over, list_of_individual_hsi_event_tuples_due_today[ev_num])

                    # Log that the event did not run
                    self.module.log_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=event.EXPECTED_APPT_FOOTPRINT,
                        squeeze_factor=squeeze_factor,
                        did_run=False,
                    )

        # 7) Add back to the HSI_EVENT_QUEUE heapq all those events
        # which are still eligible to run but which did not run
        while len(hold_over) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(hold_over))

        # 8) After completing routine for the day, log total usage of the facilities
        self.module.log_current_capabilities(
            current_capabilities=current_capabilities,
            total_footprint=total_footprint
        )


class HSI_Event:
    """Base HSI event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined below, and implement at least an `apply` and `did_not_run` method.
    """

    def __init__(self, module, *args, **kwargs):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        """
        self.module = module
        self.sim = module.sim
        self.target = None  # Overwritten by the mixin
        # This is needed so mixin constructors are called
        super().__init__(*args, **kwargs)

        # Defaults for the HSI information:
        self.TREATMENT_ID = ''
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = None
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({})
        self._received_info_about_bed_days = None
        self._cached_time_requests = {}

    @property
    def bed_days_allocated_to_this_event(self):
        if self._received_info_about_bed_days is None:
            # default to the footprint if no information about bed-days is received
            return self.BEDDAYS_FOOTPRINT
        else:
            return self._received_info_about_bed_days

    def apply(self, squeeze_factor=0.0, *args, **kwargs):
        """Apply this event to the population.

        Must be implemented by subclasses.

        """
        raise NotImplementedError

    def did_not_run(self, *args, **kwargs):
        """Called when this event is due but it is not run. Return False to prevent the event being rescheduled, or True
        to allow the rescheduling. This is called each time that the event is tried to be run but it cannot be.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: did not run.")
        return True

    def never_ran(self):
        """Called when this event is was entered to the HSI Event Queue, but was never run.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: was never run.")

    def not_available(self):
        """Called when this event is passed to schedule_hsi_event but the TREATMENT_ID is not permitted by the
         parameter service_availability.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: was not admitted to the HSI queue because the "
                                         f"service is not available.")
        pass

    def post_apply_hook(self):
        """Do any required processing after apply() completes."""
        # Impose the bed-days footprint:
        self.module.sim.modules['HealthSystem'].impose_beddays_footprint(self)

    def run(self, squeeze_factor):
        """Make the event happen."""
        updated_appt_footprint = self.apply(self.target, squeeze_factor)
        self.post_apply_hook()
        return updated_appt_footprint

    def get_all_consumables(self, item_codes=None, pkg_codes=None, footprint=None):
        """Helper function to allow for getting and checking of entire set of consumables.
        It accepts a footprint, or an item_code, or a package_code, and returns True/False for whether all the items
         are available. It avoids the use of consumables 'footprints'."""

        # Turn the input arguments into the usual consumables footprint if it's not already provided as a footprint:
        if footprint is None:
            # Item Codes provided:
            if item_codes is not None:
                if not isinstance(item_codes, list):
                    item_codes = [item_codes]
                # turn into 'consumable footprint':
                footprint_items = dict(zip(item_codes, [1]*len(item_codes)))
            else:
                footprint_items = {}

            # Package Codes provided:
            if pkg_codes is not None:
                if not isinstance(pkg_codes, list):
                    pkg_codes = [pkg_codes]
                footprint_pkgs = dict(zip(pkg_codes, [1]*len(pkg_codes)))
            else:
                footprint_pkgs = {}

            # Make the total footprint
            footprint = {
                'Item_Code': footprint_items,
                'Intervention_Package_Code': footprint_pkgs,
            }
        else:
            self.sim.modules['HealthSystem'].check_consumables_footprint_format(footprint)

        # Check availability of consumables
        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, footprint)

        all_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )

        return all_available

    def make_beddays_footprint(self, dict_of_beddays):
        """Helper function to make a correctly-formed 'bed-days footprint'"""
        # get blank footprint
        footprint = self.sim.modules['HealthSystem'].get_blank_beddays_footprint()

        # do checks
        assert type(dict_of_beddays) is dict
        assert all([(k in footprint.keys()) for k in dict_of_beddays.keys()])
        assert all([type(v) in (float, int) for v in dict_of_beddays.values()])

        # make footprint (defaulting to zero where a type of bed-days is not specified)
        for k, v in dict_of_beddays.items():
            footprint[k] = v

        return footprint

    def is_all_beddays_allocated(self):
        """Check if the entire footprint requested is allocated"""
        return all(
            [self.bed_days_allocated_to_this_event[k] == self.BEDDAYS_FOOTPRINT[k] for k in self.BEDDAYS_FOOTPRINT]
        )

    def make_appt_footprint(self, dict_of_appts):
        """Helper function to make appointment footprint in format expected downstream.

        Should be passed a dictionary keyed by appointment type codes with non-negative
        values.
        """
        health_system = self.sim.modules['HealthSystem']
        if health_system.appt_footprint_is_valid(dict_of_appts):
            return Counter(dict_of_appts)
        else:
            raise ValueError(
                "Argument to make_appt_footprint should be a dictionary keyed by "
                "appointment type code strings in Appt_Types_Table with non-negative "
                "values"
            )


class HSIEventWrapper(Event):
    """This is wrapper that contains an HSI event.

    It is used when the healthsystem is 'disabled' and all HSI events sent to the health system scheduler should
    be passed to the main simulation scheduler.
    When this event is run (by the simulation scheduler) it runs the HSI event with squeeze_factor=0.0
    """
    def __init__(self, hsi_event, *args, **kwargs):
        super().__init__(hsi_event.module, *args, **kwargs)
        self.hsi_event = hsi_event
        self.target = hsi_event.target

    def run(self):
        # check that the person is still alive
        # (this check normally happens in the HealthSystemScheduler and silently do not run the HSI event)

        if isinstance(self.hsi_event.target, tlo.population.Population) \
                or (self.hsi_event.module.sim.population.props.at[self.hsi_event.target, 'is_alive']):

            # Run the event (with 0 squeeze_factor) and ignore the output
            _ = self.hsi_event.run(squeeze_factor=0.0)
