# """
# Remaining to do:
# - streamline input arguments
# - let the level of the appointment be in the log
# - let the logger give times of each hcw
# - bed days parameterization and use of HR capacity attaching automatically to beddays
# """

import heapq as hp
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import repeat
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

import tlo
from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.bed_days import BedDays
from tlo.methods.consumables import (
    Consumables,
    get_item_code_from_item_name,
    get_item_codes_from_package_name,
)
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
    facility_level: Optional[str]
    appt_footprint: Tuple[str]
    beddays_footprint: Tuple[Tuple[str, int]]


class HSIEventQueueItem(NamedTuple):
    """Properties of event added to health system queue.

    The order of the attributes in the tuple is important as the queue sorting is done
    by the order of the items in the tuple, i.e. first by `priority`, then `topen` and
    so on.
    """
    priority: int
    topen: Date
    queue_counter: int
    tclose: Date
    # Define HSI_Event type as string to avoid NameError exception as HSI_Event defined
    # later in module (see https://stackoverflow.com/a/36286947/4798943)
    hsi_event: 'HSI_Event'


def _accepts_argument(function: callable, argument: str) -> bool:
    """Helper to test if callable object accepts an argument with a given name.

    Compared to using `inspect.signature` or `inspect.getfullargspec` the approach here
    has significantly less overhead (as a full `Signature` or `FullArgSpec` object
    does not need to constructed) but is also less readable hence why it has been
    wrapped as a helper function despite being only one-line to make its functionality
    more obvious.

    :param function: Callable object to check if argument is present in.
    :param argument: Name of argument to check.
    :returns: ``True`` is ``argument`` is an argument of ``function`` else ``False``.
    """
    # co_varnames include both arguments to function and any internally defined variable
    # names hence we check only in the first `co_argcount` items which correspond to
    # just the arguments
    return argument in function.__code__.co_varnames[:function.__code__.co_argcount]


class HealthSystem(Module):
    """
    This is the Health System Module.
    The execution of all health systems interactions are controlled through this module.
    """

    INIT_DEPENDENCIES = {'Demography'}

    PARAMETERS = {
        # Organization of the HealthSystem
        'Master_Facilities_List': Parameter(Types.DATA_FRAME, 'Listing of all health facilities.'),

        # Definitions of the officers and appointment types
        'Officer_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the types of health workers ("officers")'),
        'Appt_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the type of appointments with the health system'),
        'Appt_Offered_By_Facility_Level': Parameter(
            Types.DATA_FRAME, 'Table indicating whether or not each appointment is offered at each facility level.'),
        'Appt_Time_Table': Parameter(Types.DATA_FRAME,
                                     'The time taken for each appointment, according to officer and facility type.'),

        # Capabilities of the HealthSystem (under alternative assumptions)
        'Daily_Capabilities_actual': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _estimated current_ number and distribution of staff estimated.'),
        'Daily_Capabilities_funded': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _potential_ number and distribution of staff estimated (i.e. those '
                              'positions that can be funded).'),
        'Daily_Capabilities_funded_plus': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _potential_ number and distribution of staff estimated, with adjustments '
                              'to permit each appointment type that should be run at facility level to do so in every '
                              'district.'),

        # Consumables
        'item_and_package_code_lookups': Parameter(
            Types.DATA_FRAME, 'Data imported from the OneHealth Tool on consumable items, packages and costs.'),
        'availability_estimates': Parameter(
            Types.DATA_FRAME, 'Estimated availability of consumables in the LMIS dataset.'),

        # Infrastructure and Equipment
        'BedCapacity': Parameter(
            Types.DATA_FRAME, "Data on the number of beds available of each type by facility_id"),

        # Service Availability
        'Service_Availability': Parameter(
            Types.LIST, 'List of services to be available.')
    }

    PROPERTIES = {
        'hs_is_inpatient': Property(
            Types.BOOL, 'Whether or not the person is currently an in-patient at any medical facility'
        ),
    }

    def __init__(
        self,
        name: Optional[str] = None,
        resourcefilepath: Optional[Path] = None,
        service_availability: Optional[List[str]] = None,
        mode_appt_constraints: int = 0,
        cons_availability: str = 'default',
        ignore_priority: bool = False,
        capabilities_coefficient: Optional[float] = None,
        use_funded_or_actual_staffing: Optional[str] = 'funded_plus',
        disable: bool = False,
        disable_and_reject_all: bool = False,
        store_hsi_events_that_have_run: bool = False,
        record_hsi_event_details: bool = False
    ):
        """
        :param name: Name to use for module, defaults to module class name if ``None``.
        :param resourcefilepath: Path to directory containing resource files.
        :param service_availability: A list of treatment IDs to allow.
        :param mode_appt_constraints: Integer code in ``{0, 1, 2}`` determining mode of
            constraints with regards to officer numbers and time - 0: no constraints,
            all HSI events run with no squeeze factor, 1: elastic constraints, all HSI
            events run with squeeze factor, 2: hard constraints, only HSI events with
            no squeeze factor run.
        :param cons_availability: If 'default' then use the availability specified in the ResourceFile; if 'none', then
        let no consumable be ever be available; if 'all', then all consumables are always available. When using 'all'
        or 'none', requests for consumables are not logged.
        :param ignore_priority: If ``True`` do not use the priority information in HSI
            event to schedule
        :param capabilities_coefficient: Multiplier for the capabilities of health
            officers, if ``None`` set to ratio of initial population to estimated 2010
            population.
        :param use_funded_or_actual_staffing: If `actual`, then use the numbers and distribution of staff estimated to
        be available currently; If `funded`, then use the numbers and distribution of staff that are potentially
        available.
        :param disable: If ``True``, disables the health system (no constraints and no
            logging) and every HSI event runs.
        :param disable_and_reject_all: If ``True``, disable health system and no HSI
            events run
        :param store_hsi_events_that_have_run: Convenience flag for debugging.
        :param record_hsi_event_details: Whether to record details of HSI events used.
        """

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert isinstance(disable, bool)
        assert isinstance(disable_and_reject_all, bool)
        assert not (disable and disable_and_reject_all), (
            'Cannot have both disable and disable_and_reject_all selected'
        )
        self.disable = disable
        self.disable_and_reject_all = disable_and_reject_all

        assert mode_appt_constraints in {0, 1, 2}

        self.mode_appt_constraints = mode_appt_constraints

        self.ignore_priority = ignore_priority

        # Store the argument provided for service_availability
        self.arg_service_availabily = service_availability
        self.service_availability = ['*']  # provided so that there is a default even before simulation is run

        # Check that the capabilities coefficient is correct
        if capabilities_coefficient is not None:
            assert capabilities_coefficient >= 0
            assert isinstance(capabilities_coefficient, float)
        self.capabilities_coefficient = capabilities_coefficient

        # Find which resourcefile to use - those for the actual staff available or the funded staff available
        assert use_funded_or_actual_staffing in ['actual', 'funded', 'funded_plus']
        self.use_funded_or_actual_staffing = use_funded_or_actual_staffing

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

        # Determine what the the `availability` parameter in the Consumables class should be: it should be `all`
        # if the HealthSystem is disabled.
        self._cons_availability = 'all' if self.disable else cons_availability

        # Create the Diagnostic Test Manager to store and manage all Diagnostic Test
        self.dx_manager = DxManager(self)

        # Create the instance of BedDays to record usage of in-patient bed days
        self.bed_days = BedDays(self)

        # Create the pointer that will be to the instance of Consumables used to determine availability of consumables.
        self.consumables = None

        # Create pointer for the HealthSystemScheduler event
        self.healthsystemscheduler = None

    def read_parameters(self, data_folder):

        path_to_resourcefiles_for_healthsystem = Path(self.resourcefilepath) / 'healthsystem'

        # Load basic information about the organization of the HealthSystem
        self.parameters['Master_Facilities_List'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

        # Load ResourceFiles that define appointment and officer types
        self.parameters['Officer_Types_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Officer_Types_Table.csv')
        self.parameters['Appt_Types_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Appt_Types_Table.csv')
        self.parameters['Appt_Offered_By_Facility_Level'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_ApptType_By_FacLevel.csv')
        self.parameters['Appt_Time_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Appt_Time_Table.csv')

        # Load 'Daily_Capabilities' (for both actual and funded)
        for _i in ['actual', 'funded', 'funded_plus']:
            self.parameters[f'Daily_Capabilities_{_i}'] = pd.read_csv(
                path_to_resourcefiles_for_healthsystem / 'human_resources' / f'{_i}' /
                'ResourceFile_Daily_Capabilities.csv')

        # Read in ResourceFile_Consumables
        self.parameters['item_and_package_code_lookups'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv')
        self.parameters['availability_estimates'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'consumables' / 'ResourceFile_Consumables_availability_small.csv')

        # Data on the number of beds available of each type by facility_id
        self.parameters['BedCapacity'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'infrastructure_and_equipment' / 'ResourceFile_Bed_Capacity.csv')

        # Set default parameter for Service Availability (everything available)
        self.parameters['Service_Availability'] = ['*']

    def pre_initialise_population(self):
        """Generate the accessory classes used by the HealthSystem and pass to them the data that has been read."""
        self.process_human_resources_files()
        self.bed_days.pre_initialise_population()
        self.consumables = Consumables(data=self.parameters['availability_estimates'],
                                       rng=self.rng,
                                       availability=self._cons_availability)

    def initialise_population(self, population):
        self.bed_days.initialise_population(population.props)

    def initialise_simulation(self, sim):
        # If capabilities coefficient was not explicitly specified, use initial population scaling factor
        if self.capabilities_coefficient is None:
            self.capabilities_coefficient = self.sim.modules['Demography'].initial_model_to_data_popsize_ratio

        # Set the tracker in preparation for the simulation
        self.bed_days.initialise_beddays_tracker()

        # Set the consumables modules in preparation for the simulation
        self.consumables.processing_at_start_of_new_day(sim.date)

        # Capture list of disease modules:
        self.recognised_modules_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_HEALTHSYSTEM in m.METADATA
        ]

        # Check that set of districts of residence in population are subset of districts from
        # `self._facilities_for_each_district`, which is derived from self.parameters['Master_Facilities_List']
        df = self.sim.population.props
        districts_of_residence = set(df.loc[df.is_alive, "district_of_residence"].cat.categories)
        assert all(
            districts_of_residence.issubset(per_level_facilities.keys())
            for per_level_facilities in self._facilities_for_each_district.values()
        ), (
            "At least one district_of_residence value in population not present in "
            "self._facilities_for_each_district resource file"
        )

        # Launch the healthsystem scheduler (a regular event occurring each day) [if not disabled]
        if not (self.disable or self.disable_and_reject_all):
            self.healthsystemscheduler = HealthSystemScheduler(self)
            sim.schedule_event(self.healthsystemscheduler, sim.date)

        # Determine service_availability
        self.set_service_availability()

    def on_birth(self, mother_id, child_id):
        self.bed_days.on_birth(self.sim.population.props, mother_id, child_id)

    def on_simulation_end(self):
        """Put out to the log the information from the tracker of the last day of the simulation"""
        self.bed_days.on_simulation_end()

    def process_human_resources_files(self):
        """Create the data-structures needed from the information read into the parameters."""

        # * Define Facility Levels
        self._facility_levels = set(self.parameters['Master_Facilities_List']['Facility_Level']) - {'5'}
        assert self._facility_levels == {'0', '1a', '1b', '2', '3', '4'}  # todo soft code this?

        # * Define Appointment Types
        self._appointment_types = set(self.parameters['Appt_Types_Table']['Appt_Type_Code'])

        # * Define the Officers Needed For Each Appointment
        # (Store data as dict of dicts, with outer-dict indexed by string facility level and
        # inner-dict indexed by string type code with values corresponding to list of (named)
        # tuples of appointment officer type codes and time taken.)
        appt_time_data = self.parameters['Appt_Time_Table']
        appt_times_per_level_and_type = {_facility_level: defaultdict(list) for _facility_level in
                                         self._facility_levels}
        for appt_time_tuple in appt_time_data.itertuples():
            appt_times_per_level_and_type[
                appt_time_tuple.Facility_Level
            ][
                appt_time_tuple.Appt_Type_Code
            ].append(
                AppointmentSubunit(
                    officer_type=appt_time_tuple.Officer_Category,
                    time_taken=appt_time_tuple.Time_Taken_Mins
                )
            )
        assert (
            sum(
                len(appt_info_list)
                for level in self._facility_levels
                for appt_info_list in appt_times_per_level_and_type[level].values()
            ) == len(appt_time_data)
        )
        self._appt_times = appt_times_per_level_and_type

        # * Define Which Appointments Are Possible At Each Facility Level
        appt_type_per_level_data = self.parameters['Appt_Offered_By_Facility_Level']
        self._appt_type_by_facLevel = {
            _facility_level: set(
                appt_type_per_level_data['Appt_Type_Code'][
                    appt_type_per_level_data[f'Facility_Level_{_facility_level}']
                ]
            )
            for _facility_level in self._facility_levels
        }

        # Also store data as dict of dicts, with outer-dict indexed by string facility level and
        # inner-dict indexed by district name with values corresponding to (named) tuples of
        # facility ID and name
        # Get look-up of the districts (by name) in each region (by name)
        districts_in_region = self.sim.modules['Demography'].parameters['districts_in_region']
        all_districts = set(self.sim.modules['Demography'].parameters['district_num_to_district_name'].values())

        facilities_per_level_and_district = {_facility_level: {} for _facility_level in self._facility_levels}
        for facility_tuple in self.parameters['Master_Facilities_List'].itertuples():

            if pd.notnull(facility_tuple.District):
                # A facility that is specific to a district:
                facilities_per_level_and_district[facility_tuple.Facility_Level][facility_tuple.District] = \
                    FacilityInfo(
                        id=facility_tuple.Facility_ID,
                        name=facility_tuple.Facility_Name
                    )

            elif pd.isnull(facility_tuple.District) and pd.notnull(facility_tuple.Region):
                # A facility that is specific to region (and not a district):
                for _district in districts_in_region[facility_tuple.Region]:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = \
                        FacilityInfo(
                            id=facility_tuple.Facility_ID,
                            name=facility_tuple.Facility_Name
                        )

            elif (
                pd.isnull(facility_tuple.District) and
                pd.isnull(facility_tuple.Region) and
                (facility_tuple.Facility_Level != '5')
            ):
                # A facility that is National (not specific to a region or a district) (ignoring level 5 (headquarters))
                for _district in all_districts:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = \
                        FacilityInfo(
                            id=facility_tuple.Facility_ID,
                            name=facility_tuple.Facility_Name
                        )

        # Check that there is facility of every level for every district:
        assert all(
            all_districts == facilities_per_level_and_district[_facility_level].keys()
            for _facility_level in self._facility_levels
        ), "There is not one of each facility type available to each district."

        self._facilities_for_each_district = facilities_per_level_and_district

        # * Store 'DailyCapabilities' in correct format and using the specified underlying assumptions
        self._daily_capabilities = self.format_daily_capabilities()

        # Also, store the set of officers with non-zero daily availability
        # (This is used for checking that scheduled HSI events do not make appointment requiring officers that are
        # never available.)
        self._officers_with_availability = set(self._daily_capabilities.index[self._daily_capabilities > 0])

    def format_daily_capabilities(self) -> pd.Series:
        """
        This will updates the dataframe for the self.parameters['Daily_Capabilities'] so as to include
        every permutation of officer_type_code and facility_id, with zeros against permutations where no capacity
        is available.

        It also give the dataframe an index that is useful for merging on (based on Facility_ID and Officer Type)

        (This is so that its easier to track where demands are being placed where there is no capacity)
        """

        # Get the capabilities data imported (according to the specified underlying assumptions).
        capabilities = self.parameters[f'Daily_Capabilities_{self.use_funded_or_actual_staffing}']
        capabilities = capabilities.rename(columns={'Officer_Category': 'Officer_Type_Code'})  # neaten

        # Create dataframe containing background information about facility and officer types
        facility_ids = self.parameters['Master_Facilities_List']['Facility_ID'].values
        officer_type_codes = set(self.parameters['Officer_Types_Table']['Officer_Category'].values)
        # todo - <-- avoid use of the file or define differnetly?

        # # naming to be not with _ within the name of an oficer
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
        # officer_types = self.parameters['Officer_Types_Table'][['Officer_Type_Code', 'Officer_Type']]
        # capabilities_ex = capabilities_ex.merge(officer_types, on='Officer_Type_Code', how='left')

        # Merge in the capabilities (minutes available) for each officer type (inferring zero minutes where
        # there is no entry in the imported capabilities table)
        capabilities_ex = capabilities_ex.merge(
            capabilities[['Facility_ID', 'Officer_Type_Code', 'Total_Mins_Per_Day']],
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

        # Rename 'Total_Minutes_Per_Day'
        capabilities_ex = capabilities_ex.rename(columns={'Total_Mins_Per_Day': 'Total_Minutes_Per_Day'})

        # Checks
        assert abs(capabilities_ex['Total_Minutes_Per_Day'].sum() - capabilities['Total_Mins_Per_Day'].sum()) < 1e-7
        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)

        # return the pd.Series of `Total_Minutes_Per_Day' indexed for each type of officer at each facility
        return capabilities_ex['Total_Minutes_Per_Day']

    def set_service_availability(self):
        """Set service availability. (Should be equal to what is specified by the parameter, but overwrite with what was
         provided in arguement if an argument was specified -- provided for backward compatibility.)"""

        if self.arg_service_availabily is None:
            service_availability = self.parameters['Service_Availability']
        else:
            service_availability = self.arg_service_availabily

        assert isinstance(service_availability, list)
        self.service_availability = service_availability

        # Log the service_availability
        logger.info(key="message",
                    data=f"Running Health System With the Following Service Availability: "
                         f"{self.service_availability}"
                    )

    def schedule_hsi_event(
        self, hsi_event, priority, topen, tclose=None, do_hsi_event_checks=True
    ):
        """
        Schedule a health system interaction (HSI) event.

        :param hsi_event: The HSI event to be scheduled.
        :param priority: The priority for the HSI event (0 (highest), 1 or 2 (lowest)
        :param topen: The earliest date at which the HSI event should run
        :param tclose: The latest date at which the HSI event should run. Set to one
           week after ``topen`` if ``None``.
        :param do_hsi_event_checks: Whether to perform sanity checks on the passed
            ``hsi_event`` argument to check that it constitutes a valid HSI event. This
            is intended for allowing disabling of these checks when scheduling multiple
            HSI events of the same ``HSI_Event`` subclass together, in which case
            typically performing these checks for each individual HSI event of the
            shared type will be redundant.
        """
        # If there is no specified tclose time then set this to a week after topen
        if tclose is None:
            tclose = topen + DateOffset(days=7)

        # Check topen is not in the past
        assert topen >= self.sim.date

        # Check that priority is in valid range
        assert priority in {0, 1, 2}

        # Check that topen is strictly before tclose
        assert topen < tclose

        # Check if healthsystem is disabled/disable_and_reject_all, and scheduled a the event with appropriate wrapper.
        if self.disable and (not self.disable_and_reject_all):
            # If healthsystem is disabled (but HSI can still run), ...
            #   ... put this event straight into the normal simulation scheduler.
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=True), topen)
            return

        if self.disable_and_reject_all:
            # If healthsystem is disabled the HSI will never run: schedule for the "never_ran" method for `tclose`.
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=False), tclose)
            return

        # Check that this is a legitimate health system interaction (HSI) event
        # These checks are only performed when the flag `do_hsi_event_checks` is set
        # to ``True`` to allow disabling when the checks are redundant for example when
        # scheduling multiple HSI events of same `HSI_Event` subclass
        if do_hsi_event_checks:

            assert isinstance(hsi_event, HSI_Event)

            # Check that non-empty treatment ID specified (required for both population
            # and individual scoped events)
            assert hsi_event.TREATMENT_ID != ''

            if not isinstance(hsi_event.target, tlo.population.Population):
                # This is an individual-scoped HSI event.
                # It must have EXPECTED_APPT_FOOTPRINT, BEDDAYS_FOOTPRINT,
                # ACCEPTED_FACILITY_LEVELS and ALERT_OTHER_DISEASES defined

                # Correct formatted EXPECTED_APPT_FOOTPRINT
                assert self.appt_footprint_is_valid(hsi_event.EXPECTED_APPT_FOOTPRINT)

                # That it has an acceptable 'ACCEPTED_FACILITY_LEVEL' attribute
                assert hsi_event.ACCEPTED_FACILITY_LEVEL in self._facility_levels, \
                    f"In the HSI with TREATMENT_ID={hsi_event.TREATMENT_ID}, the ACCEPTED_FACILITY_LEVEL (=" \
                    f"{hsi_event.ACCEPTED_FACILITY_LEVEL}) is not recognised."

                self.bed_days.check_beddays_footprint_format(hsi_event.BEDDAYS_FOOTPRINT)

                # That it has a list for the other disease that will be alerted when it
                # is run and that this make sense
                assert isinstance(hsi_event.ALERT_OTHER_DISEASES, Sequence)

                if len(hsi_event.ALERT_OTHER_DISEASES) > 0:
                    if not (hsi_event.ALERT_OTHER_DISEASES[0] == '*'):
                        for d in hsi_event.ALERT_OTHER_DISEASES:
                            assert d in self.recognised_modules_names

                # Check that this can accept the squeeze argument
                assert _accepts_argument(hsi_event.run, 'squeeze_factor')

                # Check that at least one type of appointment is required
                assert len(hsi_event.EXPECTED_APPT_FOOTPRINT) > 0, (
                    'No appointment types required in the EXPECTED_APPT_FOOTPRINT'
                )
                # Check that the event does not request an appointment at a facility
                # level which is not possible
                appt_type_to_check_list = hsi_event.EXPECTED_APPT_FOOTPRINT.keys()
                facility_appt_types = self._appt_type_by_facLevel[
                    hsi_event.ACCEPTED_FACILITY_LEVEL
                ]
                assert facility_appt_types.issuperset(appt_type_to_check_list), (
                    f"An appointment type has been requested at a facility level for "
                    f"which it is not possible: TREATMENT_ID={hsi_event.TREATMENT_ID}"
                )

        # Check that event (if individual level) is able to run with this configuration
        # of officers (i.e. check that this does not demand officers that are never
        # available at a particular facility). This is run irrespective of the value of
        # _do_hsi_event_checks as the appointment footprint time request depends on the
        # district of residence of the HSI event's target and so the time requests can
        # differ for each instance of an HSI_Event subclass even when their other
        # attributes are shared
        if not isinstance(hsi_event.target, tlo.population.Population):
            footprint = self.get_appt_footprint_as_time_request(hsi_event=hsi_event)
            if not self._officers_with_availability.issuperset(footprint.keys()):
                logger.warning(
                    key="message",
                    data=(
                        "The expected footprint is not possible with the configuration "
                        f"of officers: {hsi_event.TREATMENT_ID}."
                    )
                )

        # Check that this request is allowable under current policy (i.e. included in
        # service_availability)
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

        #  Manipulate the priority level if needed
        # If ignoring the priority in scheduling, then over-write the provided priority information
        if self.ignore_priority:
            priority = 0  # set all event to priority 0
        # This is where could attach a different priority score according to the treatment_id (and other things)
        # in order to examine the influence of the prioritisation score.

        # If all is correct and the hsi event is allowed then add this request to the queue of HSI_EVENT_QUEUE
        if allowed:

            if not isinstance(hsi_event.target, tlo.population.Population):
                # Write the facility_id at which this HSI will occur:
                hsi_event._facility_id = self.get_facility_info(hsi_event).id

            # Create a tuple to go into the heapq
            # (NB. the sorting is done ascending and by the order of the items in the tuple)
            # Pos 0: priority,
            # Pos 1: topen,
            # Pos 2: hsi_event_queue_counter,
            # Pos 3: tclose,
            # Pos 4: the hsi_event itself

            new_request = HSIEventQueueItem(
                priority, topen, self.hsi_event_queue_counter, tclose, hsi_event
            )
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

    def schedule_batch_of_individual_hsi_events(
        self, hsi_event_class, person_ids, priority, topen, tclose=None, **event_kwargs
    ):
        """Schedule a batch of individual-scoped HSI events of the same type.

        Only performs sanity checks on the HSI event for the first scheduled event
        thus removing the overhead of multiple redundant checks.

        :param hsi_event_class: The ``HSI_Event`` subclass of the events to schedule.
        :param person_ids: A sequence of person ID index values to use as the targets
            of the HSI events being scheduled.
        :param priority: The priority for the HSI events: 0 (highest), 1 or 2 (lowest).
            Either a single value for all events or an iterable of per-target values.
        :param topen: The earliest date at which the HSI events should run. Either a
            single value for all events or an iterable of per-target values.
        :param tclose: The latest date at which the HSI events should run. Set to one
           week after ``topen`` if ``None``. Either a single value for all events or an
           iterable of per-target values.
        :param event_kwargs: Any additional keyword arguments to pass to the
            ``hsi_event_class`` initialiser in addition to ``person_id``.
        """
        # If any of {priority, topen, tclose} are iterable assume correspond to per-
        # target values for corresponding arguments of schedule_hsi_event otherwise
        # use same value for all calls
        priorities = priority if isinstance(priority, Iterable) else repeat(priority)
        topens = topen if isinstance(topen, Iterable) else repeat(topen)
        tcloses = tclose if isinstance(tclose, Iterable) else repeat(tclose)
        for i, (person_id, priority, topen, tclose) in enumerate(
            zip(person_ids, priorities, topens, tcloses)
        ):
            self.schedule_hsi_event(
                hsi_event=hsi_event_class(person_id=person_id, **event_kwargs),
                priority=priority,
                topen=topen,
                tclose=tclose,
                # Only perform checks for first event
                do_hsi_event_checks=(i == 0)
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

    def get_capabilities_today(self) -> pd.Series:
        """
        Get the capabilities of the health system today.
        returns: pd.Series giving minutes available for each officer type in each facility type

        Functions can go in here in the future that could expand the time available,
        simulating increasing efficiency (the concept of a productivity ratio raised
        by Martin Chalkley).

        For now this method only multiplies the estimated minutes available by the `capabilities_coefficient` scale
        factor.
        """
        return self._daily_capabilities * self.capabilities_coefficient

    def get_blank_appt_footprint(self):
        """
        This is a helper function so that disease modules can easily create their appt_footprints.
        It returns an empty Counter instance.

        """
        return Counter()

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
        return self._facilities_for_each_district[the_level][the_district]

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
        appt_times = self._appt_times

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
                    f"The time needed for this appointment is not defined for the specified facility level: "
                    f"TREATMENT_ID={hsi_event.TREATMENT_ID}, "
                    f"appt_type={appt_type}, "
                    f"facility_level={the_facility_level}."
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
        :param current_capabilities: Series giving the amount of time available for
            each health officer in each health facility (using the standard index)

        :return: squeeze_factors: an array of the squeeze factors for each HSI event
            (position in array matches that in the all_call_today list).
        """

        # 1) Compute the load factors for each officer type at each facility that is
        # called-upon in this list of HSIs
        load_factor = {}
        for officer, call in total_footprint.items():
            availability = current_capabilities.get(officer)
            if availability is None:  # todo - does this ever happen?
                load_factor[officer] = 99.99
            elif availability == 0:
                load_factor[officer] = float('inf')
            else:
                load_factor[officer] = max(call / availability - 1, 0)

        # 2) Convert these load-factors into an overall 'squeeze' signal for each HSI,
        # based on the highest load-factor of any officer required (or zero if event
        # has an empty footprint)
        squeeze_factor_per_hsi_event = np.array([
            max((load_factor[officer] for officer in footprint), default=0)
            for footprint in footprints_per_event
        ])

        assert (squeeze_factor_per_hsi_event >= 0).all()

        return squeeze_factor_per_hsi_event

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
                    ),
                    beddays_footprint=tuple(sorted(hsi_event.BEDDAYS_FOOTPRINT.items()))
                )
            )

    def log_current_capabilities(self, current_capabilities, total_footprint):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type
        NB. To get this per Officer_Type_Code, it would be possible to simply log the entire current_capabilities df.
        :param current_capabilities: the current_capabilities of the health system.
        :param total_footprint: Per-officer totals of footprints of all the HSI events that ran
        """

        # Combine the current_capabilities and total_footprint per-officer totals
        comparison = pd.DataFrame(index=current_capabilities.index)
        comparison['Total_Minutes_Per_Day'] = current_capabilities
        comparison['Minutes_Used'] = pd.Series(total_footprint, dtype='float64')
        comparison['Minutes_Used'] = comparison['Minutes_Used'].fillna(0.0)
        assert len(comparison) == len(current_capabilities)

        # Sum within each Facility_ID
        facility_id = [_f.split('_')[1] for _f in comparison.index]
        summary = comparison.groupby(by=facility_id)[['Total_Minutes_Per_Day', 'Minutes_Used']].sum()

        # Compute Fraction of Time Used Across All Facilities
        total_available = summary['Total_Minutes_Per_Day'].sum()
        fraction_time_used_across_all_facilities = (
            summary['Minutes_Used'].sum() / total_available if total_available > 0 else 0
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

    def remove_beddays_footprint(self, person_id):
        # removing bed_days from a particular individual if any
        self.bed_days.remove_beddays_footprint(person_id=person_id)

    def find_events_for_person(self, person_id: int):
        """Find the events in the HSI_EVENT_QUEUE for a particular person.
        :param person_id: the person_id of interest
        :returns list of tuples (date_of_event, event) for that person_id in the HSI_EVENT_QUEUE.

        NB. This is for debugging and testing only - not for use in real simulations as it is slow
        """
        list_of_events = list()

        for ev_tuple in self.HSI_EVENT_QUEUE:
            date = ev_tuple[1]  # this is the 'topen' value
            event = ev_tuple[4]
            if isinstance(event.target, (int, np.integer)):
                if event.target == person_id:
                    list_of_events.append((date, event))

        return list_of_events

    def reset_queue(self):
        """Set the HSI event queue to be empty"""
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0

    def get_item_codes_from_package_name(self, package: str) -> dict:
        """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
         a given package name."""
        return get_item_codes_from_package_name(self.parameters['item_and_package_code_lookups'], package)

    def get_item_code_from_item_name(self, item: str) -> int:
        """Helper function to provide the item_code (an int) when provided with the name of the item"""
        return get_item_code_from_item_name(self.parameters['item_and_package_code_lookups'], item)


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

        # 0) Refresh information ready for new day:
        self.module.bed_days.processing_at_start_of_new_day()
        self.module.consumables.processing_at_start_of_new_day(self.sim.date)

        # - Create hold-over list (will become a heapq). This will hold events that cannot occur today before they are
        #  added back to the heapq
        hold_over = list()

        # 1) Get the events that are due today:
        list_of_individual_hsi_event_tuples_due_today = list()
        list_of_population_hsi_event_tuples_due_today = list()

        # To avoid repeated dataframe accesses in subsequent loop, assemble set of alive
        # person IDs as  one-off operation, exploiting the improved efficiency of
        # boolean-indexing of a Series compared to row-by-row access. From benchmarks
        # converting Series to list before converting to set is ~2x more performant than
        # direct conversion to set, while checking membership of set is ~10x quicker
        # than checking membership of Pandas Index object and ~25x quicker than checking
        # membership of list
        alive_persons = set(
            self.sim.population.props.index[self.sim.population.props.is_alive].to_list()
        )

        while len(self.module.HSI_EVENT_QUEUE) > 0:

            next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)
            # Read the tuple and assemble into a dict 'next_event'

            event = next_event_tuple.hsi_event

            if self.sim.date > next_event_tuple.tclose:
                # The event has expired (after tclose) having never been run. Call the 'never_ran' function
                event.never_ran()

            elif not (
                isinstance(event.target, tlo.population.Population)
                or event.target in alive_persons
            ):
                # if individual level event and the person who is the target is no longer alive, do nothing more
                pass

            elif self.sim.date < next_event_tuple.topen:
                # The event is not yet due (before topen), add to the hold-over list
                hp.heappush(hold_over, next_event_tuple)

                if next_event_tuple.priority == 2:
                    # Check the priority
                    # If the next event is not due and has low priority, then stop looking through the heapq
                    # as all other events will also not be due.
                    break

            else:
                # The event is now due to run today and the person is confirmed to be still alive
                # Add it to the list of events due today (individual or population level)
                # NB. These list is ordered by priority and then due date

                is_pop_level_hsi_event = isinstance(event.target, tlo.population.Population)
                if is_pop_level_hsi_event:
                    list_of_population_hsi_event_tuples_due_today.append(next_event_tuple)
                else:
                    list_of_individual_hsi_event_tuples_due_today.append(next_event_tuple)

        # 2) Run all population-level HSI events
        while len(list_of_population_hsi_event_tuples_due_today) > 0:
            pop_level_hsi_event_tuple = list_of_population_hsi_event_tuples_due_today.pop()

            pop_level_hsi_event = pop_level_hsi_event_tuple.hsi_event
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
                self.module.get_appt_footprint_as_time_request(hsi_event=(event_tuple.hsi_event))
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
            for ev_num, _ in enumerate(list_of_individual_hsi_event_tuples_due_today):
                event = list_of_individual_hsi_event_tuples_due_today[ev_num].hsi_event
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
                    # Compute the bed days that are allocated to this HSI and provide this information to the HSO
                    event._received_info_about_bed_days = \
                        self.module.bed_days.issue_bed_days_according_to_availability(
                            facility_id=self.module.bed_days.get_facility_id_for_beds(persons_id=event.target),
                            footprint=event.BEDDAYS_FOOTPRINT
                        )

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
        self._facility_id = None

    @property
    def bed_days_allocated_to_this_event(self):
        if self._received_info_about_bed_days is None:
            # default to the footprint if no information about bed-days is received
            return self.BEDDAYS_FOOTPRINT

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

    def post_apply_hook(self):
        """Impose the bed-days footprint (if target of the HSI is a person_id)"""
        if isinstance(self.target, int):
            self.module.sim.modules['HealthSystem'].bed_days.impose_beddays_footprint(
                person_id=self.target,
                footprint=self.bed_days_allocated_to_this_event
            )

    def run(self, squeeze_factor):
        """Make the event happen."""
        updated_appt_footprint = self.apply(self.target, squeeze_factor)
        self.post_apply_hook()
        return updated_appt_footprint

    def get_consumables(self,
                        item_codes: Union[None, np.integer, int, list, set, dict] = None,
                        optional_item_codes: Union[None, np.integer, int, list, set, dict] = None,
                        to_log: Optional[bool] = True,
                        return_individual_results: Optional[bool] = False
                        ) -> Union[bool, dict]:
        """Function to allow for getting and checking of entire set of consumables. All requests for consumables should
        use this function.
        :param item_codes: The item code(s) (and quantities) of the consumables that are requested and which determine
        the summary result for availability/non-availability. This can be an `int` (the item_code needed [assume
        quantity=1]), a `list` or `set` (the collection  of item_codes [for each assuming quantity=1]), or a `dict`
        (with key:value pairs `<item_code>:<quantity>`).
        :param optional_item_codes: The item code(s) (and quantities) of the consumables that are requested and which do
         not determine the summary result for availability/non-availability. (Same format as `item_codes`). This is
         useful when a large set of items may be used, but the viability of a subsequent operation depends only on a
         subset.
        :param return_individual_results: If True returns a `dict` giving the availability of each item_code requested
        (otherwise gives a `bool` indicating if all the item_codes requested are available).
        :param to_log: If True, logs the request.
        :returns A `bool` indicating whether every item is available, or a `dict` indicating the availability of each
         item.
        Note that disease module can use the `get_item_codes_from_package_name` and `get_item_code_from_item_name`
         methods in the `HealthSystem` module to find item_codes.
        """

        def _return_item_codes_in_dict(item_codes: Union[None, np.integer, int, list, set, dict]) -> dict:
            """Convert an argument for 'item_codes` (provided as int, list, set or dict) into the format
            dict(<item_code>:quantity)."""

            if item_codes is None:
                return {}

            if isinstance(item_codes, (int, np.integer)):
                return {int(item_codes): 1}

            elif isinstance(item_codes, list):
                if not all([isinstance(i, (int, np.integer)) for i in item_codes]):
                    raise ValueError("item_codes must be integers")
                return {int(i): 1 for i in item_codes}

            elif isinstance(item_codes, dict):
                if not all(
                    [(isinstance(code, (int, np.integer)) and isinstance(quantity, (int, np.integer)))
                     for code, quantity in item_codes.items()]
                ):
                    raise ValueError("item_codes must be integers and quantities must be integers.")
                return {int(i): int(q) for i, q in item_codes.items()}

            else:
                raise ValueError("The item_codes are given in an unrecognised format")

        hs_module = self.sim.modules['HealthSystem']

        _item_codes = _return_item_codes_in_dict(item_codes)
        _optional_item_codes = _return_item_codes_in_dict(optional_item_codes)

        # Determine if the request should be logged (over-ride argument provided if HealthSystem is disabled).
        _to_log = to_log if not hs_module.disable else False

        # Checking the availability and logging:
        rtn = hs_module.consumables._request_consumables(item_codes={**_item_codes, **_optional_item_codes},
                                                         to_log=_to_log,
                                                         facility_id=self._facility_id,
                                                         treatment_id=self.TREATMENT_ID)

        # Return result in expected format:
        if not return_individual_results:
            # Determine if all results for all the `item_codes` are True (discarding results from optional_item_codes).
            return all([v for k, v in rtn.items() if k in _item_codes])
        else:
            return rtn

    def make_beddays_footprint(self, dict_of_beddays):
        """Helper function to make a correctly-formed 'bed-days footprint'"""

        # get blank footprint
        footprint = self.sim.modules['HealthSystem'].bed_days.get_blank_beddays_footprint()

        # do checks on the dict_of_beddays provided.
        assert isinstance(dict_of_beddays, dict)
        assert all((k in footprint.keys()) for k in dict_of_beddays.keys())
        assert all(isinstance(v, (float, int)) for v in dict_of_beddays.values())

        # make footprint (defaulting to zero where a type of bed-days is not specified)
        for k, v in dict_of_beddays.items():
            footprint[k] = v

        return footprint

    def is_all_beddays_allocated(self):
        """Check if the entire footprint requested is allocated"""
        return all(
            self.bed_days_allocated_to_this_event[k] == self.BEDDAYS_FOOTPRINT[k] for k in self.BEDDAYS_FOOTPRINT
        )

    def make_appt_footprint(self, dict_of_appts):
        """Helper function to make appointment footprint in format expected downstream.

        Should be passed a dictionary keyed by appointment type codes with non-negative
        values.
        """
        health_system = self.sim.modules['HealthSystem']
        if health_system.appt_footprint_is_valid(dict_of_appts):
            return Counter(dict_of_appts)

        raise ValueError(
            "Argument to make_appt_footprint should be a dictionary keyed by "
            "appointment type code strings in Appt_Types_Table with non-negative "
            "values"
        )


class HSIEventWrapper(Event):
    """This is wrapper that contains an HSI event.

    It is used:
     1) When the healthsystem is in mode 'disabled=True' such that HSI events sent to the health system scheduler are
     passed to the main simulation scheduler for running on the date of `topen`. (Note, it is run with
     squeeze_factor=0.0.)
     2) When the healthsytsem is in mode `diable_and_reject_all=True` such that HSI are not run but the `never_ran`
     method is run on the date of `tclose`.
    """

    def __init__(self, hsi_event, run_hsi=True, *args, **kwargs):
        super().__init__(hsi_event.module, *args, **kwargs)
        self.hsi_event = hsi_event
        self.target = hsi_event.target
        self.run_hsi = run_hsi  # True to call the HSI's `run` method; False to call the HSI's `never_ran` method

    def run(self):
        """Do the appropriate action on the HSI event"""

        # Check that the person is still alive (this check normally happens in the HealthSystemScheduler and silently
        # do not run the HSI event)

        if isinstance(self.hsi_event.target, tlo.population.Population) or (
            self.hsi_event.module.sim.population.props.at[self.hsi_event.target, 'is_alive']
        ):

            if self.run_hsi:
                # Run the event (with 0 squeeze_factor) and ignore the output
                _ = self.hsi_event.run(squeeze_factor=0.0)
            else:
                self.hsi_event.never_ran()
