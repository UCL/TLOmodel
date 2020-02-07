import heapq as hp
import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import tlo
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods.dxmanager import DxManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HealthSystem(Module):
    """
    This is the Health System Module
    Version: September 2019
    The execution of all health systems interactions are controlled through this module.
    """

    PARAMETERS = {
        'Officer_Types': Parameter(Types.DATA_FRAME, 'The names of the types of health workers ("officers")'),
        'Daily_Capabilities': Parameter(
            Types.DATA_FRAME, 'The capabilities by facility and officer type available each day'
        ),
        'Appt_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the type of appointments with the health system'),
        'Appt_Time_Table': Parameter(
            Types.DATA_FRAME, 'The time taken for each appointment, according to officer and facility type.'
        ),
        'ApptType_By_FacLevel': Parameter(
            Types.DATA_FRAME, 'Indicates whether an appointment type can occur at a facility level.'
        ),
        'Master_Facilities_List': Parameter(Types.DATA_FRAME, 'Listing of all health facilities.'),
        'Facilities_For_Each_District': Parameter(
            Types.DATA_FRAME,
            'Mapping between a district and all of the health facilities to which its \
                      population have access.',
        ),
        'Consumables': Parameter(Types.DATA_FRAME, 'List of consumables used in each intervention and their costs.'),
        'Consumables_Cost_List': Parameter(Types.DATA_FRAME, 'List of each consumable item and it' 's cost'),
    }

    PROPERTIES = {
        'hs_dist_to_facility': Property(
            Types.REAL, 'The distance for each person to their nearest clinic (of any type)'
        )
    }

    def __init__(
        self,
        name=None,
        resourcefilepath=None,
        service_availability=None,  # must be a list of treatment_ids to allow
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        ignore_cons_constraints=False,  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        disable=False  # disables the healthsystem (no constraints and no logging).
    ):

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert type(ignore_cons_constraints) is bool
        self.ignore_cons_constraints = ignore_cons_constraints

        assert type(disable) is bool
        self.disable = disable

        assert mode_appt_constraints in [0, 1, 2]  # Mode of constraints
        # 0: no constraints -- all HSI Events run with no squeeze factor
        # 1: elastic -- all HSI Events run - with squeeze factor
        # 2: hard -- only HSI events with no squeeze factor run

        self.mode_appt_constraints = mode_appt_constraints

        self.ignore_priority = ignore_priority

        # Check that the service_availability list is specified correctly
        if service_availability is None:
            self.service_availability = ['*']
        else:
            assert type(service_availability) is list
            self.service_availability = service_availability

        # Check that the capabilities coefficident is correct
        assert capabilities_coefficient >= 0
        assert type(capabilities_coefficient) is float
        self.capabilities_coefficient = capabilities_coefficient

        # Define empty set of registered disease modules
        self.registered_disease_modules = {}

        # Define the dataframe that determines the availability of consumables on a daily basis
        self.cons_item_code_availability_today = pd.DataFrame()

        # Define the container for calls for health system interaction events
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0  # Counter to help with the sorting in the heapq

        logger.info('----------------------------------------------------------------------')
        logger.info("Setting up the Health System With the Following Service Availability:")
        logger.info(self.service_availability)
        logger.info('----------------------------------------------------------------------')

        # Create the Diagnostic Test Manager to store and manage all Diagnostic Test
        self.dx_manager = DxManager()

    def read_parameters(self, data_folder):

        self.parameters['Officer_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Officer_Types_Table.csv'
        )

        self.parameters['Appt_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Types_Table.csv'
        )

        self.parameters['Appt_Time_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Time_Table.csv'
        )

        self.parameters['ApptType_By_FacLevel'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_ApptType_By_FacLevel.csv'
        )

        mfl = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Master_Facilities_List.csv')
        self.parameters['Master_Facilities_List'] = mfl.iloc[:, 1:]  # get rid of extra column

        self.parameters['Facilities_For_Each_District'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Facilities_For_Each_District.csv'
        )

        self.parameters['Consumables'] = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Consumables.csv')

        self.parameters['Consumables_Cost_List'] = (
            self.parameters['Consumables'][['Item_Code', 'Unit_Cost']].drop_duplicates().set_index('Item_Code')
        )

        caps = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Daily_Capabilities.csv')
        self.parameters['Daily_Capabilities'] = caps.iloc[:, 1:]
        self.reformat_daily_capabilities()  # Reformats this table to include zero where capacity is not available

        # Make a dataframe that organsie the probabilities of individual consumables items being available
        # (by unique item codes)
        cons = self.parameters['Consumables']
        unique_item_codes = pd.DataFrame(data={'Item_Code': pd.unique(cons['Item_Code'])})

        # merge in probabilities of being available
        filter_col = [col for col in cons if col.startswith('Available_Facility_Level_')]
        filter_col.append('Item_Code')
        prob_unique_item_codes_available = unique_item_codes.merge(
            cons.drop_duplicates(['Item_Code'])[filter_col], on='Item_Code', how='inner'
        )
        assert len(prob_unique_item_codes_available) == len(unique_item_codes)

        # set the index as the Item_Code
        prob_unique_item_codes_available.set_index('Item_Code', drop=True, inplace=True)

        self.prob_unique_item_codes_available = prob_unique_item_codes_available

    def initialise_population(self, population):
        df = population.props

        # Assign hs_dist_to_facility'
        # (For now, let this be a random number, but in future it may be properly informed based on \
        #  population density distribitions)
        # Note that this characteritic is inherited from mother to child.
        df['hs_dist_to_facility'] = self.rng.uniform(0.01, 5.00, len(df))

    def initialise_simulation(self, sim):

        # Check that each person is being associated with a facility of each type
        pop = self.sim.population.props
        fac_per_district = self.parameters['Facilities_For_Each_District']
        mfl = self.parameters['Master_Facilities_List']
        self.Facility_Levels = pd.unique(mfl['Facility_Level'])

        for person_id in pop.index[pop.is_alive]:
            my_district = pop.at[person_id, 'district_of_residence']
            my_health_facilities = fac_per_district.loc[fac_per_district['District'] == my_district]
            my_health_facility_level = pd.unique(my_health_facilities.Facility_Level)
            assert len(my_health_facilities) == len(self.Facility_Levels)
            assert set(my_health_facility_level) == set(self.Facility_Levels)

        # Launch the healthsystem scheduler (a regular event occurring each day) [if not disabled]
        if not self.disable:
            sim.schedule_event(HealthSystemScheduler(self), sim.date)

    def on_birth(self, mother_id, child_id):

        # New child inherits the hs_dist_to_facility of the mother
        df = self.sim.population.props
        df.at[child_id, 'hs_dist_to_facility'] = df.at[mother_id, 'hs_dist_to_facility']

    def register_disease_module(self, new_disease_module):
        """
        Register Disease Module
        In order for a disease module to use the functionality of the health system it must be registered
        This list is also used to alert other disease modules when a health system interaction occurs

        :param new_disease_module: The pointer to the disease module
        """
        assert (
            new_disease_module.name not in self.registered_disease_modules
        ), 'A module named {} has already been registered'.format(new_disease_module.name)
        assert 'on_hsi_alert' in dir(new_disease_module)

        self.registered_disease_modules[new_disease_module.name] = new_disease_module
        logger.info('Registering disease module %s', new_disease_module.name)

    def schedule_hsi_event(self, hsi_event, priority, topen, tclose=None):
        """
        Schedule the health system interaction event

        :param hsi_event: the hsi_event to be scheduled
        :param priority: the priority for the hsi event (0 (highest), 1 or 2 (lowest)
        :param topen: the earliest date at which the hsi event should run
        :param tclose: the latest date at which the hsi event should run
        """

        logger.debug(
            'HealthSystem.schedule_event>>Logging a request for an HSI: %s for person: %s',
            hsi_event.TREATMENT_ID,
            hsi_event.target,
        )

        assert isinstance(hsi_event, HSI_Event)

        # 0) If healthsystem is disabled, put this event straight into the normal simulation scheduler.
        if self.disable:
            wrapped_hsi_event = HSIEventWrapper(hsi_event=hsi_event)
            self.sim.schedule_event(wrapped_hsi_event, topen)
            return  # Terrminate this functional call

        # 1) Check that this is a legitimate health system interaction (HSI) event

        if isinstance(hsi_event.target, tlo.population.Population):  # check if hsi_event is population-scoped
            # This is a population-scoped HSI event...
            # ... So it needs TREATMENT_ID
            # ... But it does not need APPT, CONS, ACCEPTED_FACILITY_LEVEL and ALERT_OTHER_DISEASES, or did_not_run().

            assert 'TREATMENT_ID' in dir(hsi_event)
            assert 'EXPECTED_APPT_FOOTPRINT_FOOTPRINT' not in dir(hsi_event)
            assert 'ACCEPTED_FACILITY_LEVEL' not in dir(hsi_event)
            assert 'ALERT_OTHER_DISEASES' not in dir(hsi_event)

        else:
            # This is an individual-scoped HSI event.
            # It must have APPT, CONS, ACCEPTED_FACILITY_LEVELS and ALERT_OTHER_DISEASES defined

            # Correctly formatted footprint
            assert 'TREATMENT_ID' in dir(hsi_event)

            # Correct formated EXPECTED_APPT_FOOTPRINT
            assert 'EXPECTED_APPT_FOOTPRINT' in dir(hsi_event)
            self.check_appt_footprint_format(hsi_event.EXPECTED_APPT_FOOTPRINT)

            # That it has an 'ACCEPTED_FACILITY_LEVEL' attribute
            # (Integer specificying the facility level at which HSI_Event must occur)
            assert 'ACCEPTED_FACILITY_LEVEL' in dir(hsi_event)
            assert type(hsi_event.ACCEPTED_FACILITY_LEVEL) is int
            assert hsi_event.ACCEPTED_FACILITY_LEVEL in list(
                pd.unique(self.parameters['Facilities_For_Each_District']['Facility_Level'])
            )

            # That it has a list for the other disease that will be alerted when it is run and that this make sense
            assert 'ALERT_OTHER_DISEASES' in dir(hsi_event)
            assert type(hsi_event.ALERT_OTHER_DISEASES) is list

            if len(hsi_event.ALERT_OTHER_DISEASES) > 0:
                if not (hsi_event.ALERT_OTHER_DISEASES[0] == '*'):
                    for d in hsi_event.ALERT_OTHER_DISEASES:
                        assert d in self.sim.modules['HealthSystem'].registered_disease_modules.keys()

            # Check that this can accept the squeeze argument
            assert 'squeeze_factor' in inspect.getfullargspec(hsi_event.run).args

        # 2) Check that topen, tclose and priority are valid

        # If there is no specified tclose time then set this is after the end of the simulation
        if tclose is None:
            tclose = self.sim.end_date + DateOffset(days=1)

        # Check topen is not in the past
        assert topen >= self.sim.date

        # Check that topen and tclose are not the same date
        assert not topen == tclose

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
            allowed = True  # allow all GenericFirstAppt's
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
            assert any(value > 0 for value in hsi_event.EXPECTED_APPT_FOOTPRINT.values()), \
                'No appointment types required in the EXPECTED_APPT_FOOTPRINT'

            # 5) Check that the event does not request an appointment at a facility level which is not possible
            appt_type_to_check_list = [k for k, v in hsi_event.EXPECTED_APPT_FOOTPRINT.items() if v > 0]
            assert all([self.parameters['ApptType_By_FacLevel'].loc[
                            self.parameters['ApptType_By_FacLevel']['Appt_Type_Code'] == appt_type_to_check,
                            self.parameters['ApptType_By_FacLevel'].columns.str.contains(
                                str(hsi_event.ACCEPTED_FACILITY_LEVEL))].all().all()
                        for appt_type_to_check in appt_type_to_check_list
                        ]), \
                "An appointment type has been requested at a facility level for which is it not possibe: " \
                + hsi_event.TREATMENT_ID

            # 6) Check that event (if individual level) is able to run with this configuration of officers
            # (ie. Check that this does not demand officers that are never available at a particular facility)
            caps = self.parameters['Daily_Capabilities']
            footprint = self.get_appt_footprint_as_time_request(hsi_event=hsi_event)

            footprint_is_possible = (len(footprint) > 0) & (
                caps.loc[caps.index.isin(footprint.index), 'Total_Minutes_Per_Day'] > 0).all()
            if not footprint_is_possible:
                logger.warning("The expected footprint is not possible with the configuration of officers.")

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
                'HealthSystem.schedule_event>>HSI has been added to the queue: %s for person: %s',
                hsi_event.TREATMENT_ID,
                hsi_event.target,
            )

        else:
            logger.debug(
                '%s| A request was made for a service but it was not included in the service_availability list: %s',
                self.sim.date,
                hsi_event.TREATMENT_ID,
            )

    def check_appt_footprint_format(self, appt_footprint):
        """
        This function runs some checks on the appt_footprint to ensure it is the right format
        :return: None
        """

        assert set(appt_footprint.keys()) == set(self.parameters['Appt_Types_Table']['Appt_Type_Code'])
        # All sensible numbers for the number of appointments requested (no negative and at least one appt required)

        assert all(np.asarray([(appt_footprint[k]) for k in appt_footprint.keys()]) >= 0)

        assert not all(value == 0 for value in appt_footprint.values())

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
                alert_modules = list(self.registered_disease_modules.keys())
            else:
                alert_modules = hsi_event.ALERT_OTHER_DISEASES

            # Remove the originating module from the list of modules to alert.

            # Get the name of the disease module that this event came from ultimately
            originating_disease_module_name = hsi_event.module.name
            if originating_disease_module_name in alert_modules:
                alert_modules.remove(originating_disease_module_name)

            for module_name in alert_modules:
                module = self.registered_disease_modules[module_name]
                module.on_hsi_alert(person_id=hsi_event.target, treatment_id=hsi_event.TREATMENT_ID)

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

        # apply the capabilities_coefficient
        capabilities['Total_Minutes_Per_Day'] = capabilities['Total_Minutes_Per_Day'] * self.capabilities_coefficient

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
        assert capabilities_ex['Total_Minutes_Per_Day'].sum() == capabilities['Total_Minutes_Per_Day'].sum()
        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)

        # Updates the capabilities table with the reformatted version
        self.parameters['Daily_Capabilities'] = capabilities_ex

    def get_capabilities_today(self):
        """
        Get the capabilities of the health system today
        returns: DataFrame giving minutes available for each officer type in each facility type

        Functions can go in here in the future that could expand the time available, simulating increasing efficiency.
        (The concept of a productivitiy ratio raised by Martin Chalkley). For now just have a single scaling value,
        named capabilities_coefficient.
        """

        # Get the capabilities data as they are imported
        capabilities_today = self.parameters['Daily_Capabilities']

        # apply the capabilities_coefficient
        capabilities_today['Total_Minutes_Per_Day'] = (
            capabilities_today['Total_Minutes_Per_Day'] * self.capabilities_coefficient
        )

        return capabilities_today

    def get_blank_appt_footprint(self):
        """
        This is a helper function so that disease modules can easily create their appt_footprints.
        It returns a dataframe containing the appointment footprint information in the format that /
        the HealthSystemScheduler expects.

        """

        keys = self.parameters['Appt_Types_Table']['Appt_Type_Code']
        values = np.zeros(len(keys))
        blank_footprint = dict(zip(keys, values))
        return blank_footprint

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
            cons_footprint = {
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

    def get_appt_footprint_as_time_request(self, hsi_event, actual_appt_footprint=None):
        """
        This will take an HSI event and return the required appointments in terms of the time required of each
        Officer Type in each Facility ID.
        The index will identify the Facility ID and the Officer Type in the same format as is used in
        Daily_Capabilities.

        :param hsi_event: The HSI event
        :param actual_appt_footprint: The actual appt footprint (optional) if different to that in the HSI_event
        :return: A series that gives the time required for each officer-type in each facility_ID
        """

        # Gather useful information
        df = self.sim.population.props
        mfl = self.parameters['Master_Facilities_List']
        fac_per_district = self.parameters['Facilities_For_Each_District']
        appt_types = self.parameters['Appt_Types_Table']['Appt_Type_Code'].values
        appt_times = self.parameters['Appt_Time_Table']

        # Gather information about the HSI event
        the_person_id = hsi_event.target
        the_district = df.at[the_person_id, 'district_of_residence']

        # Get the appt_footprint
        if actual_appt_footprint is None:
            # use the appt_footprint in the hsi_event
            the_appt_footprint = hsi_event.EXPECTED_APPT_FOOTPRINT
        else:
            # use the actual_appt_provided
            the_appt_footprint = actual_appt_footprint

        # Get the (one) health_facility available to this person (based on their district), which is accepted by the
        # hsi_event.ACCEPTED_FACILITY_LEVEL:
        the_facility_id = fac_per_district.loc[
            (fac_per_district['District'] == the_district)
            & (fac_per_district['Facility_Level'] == hsi_event.ACCEPTED_FACILITY_LEVEL),
            'Facility_ID',
        ].values[0]

        the_facility_level = mfl.loc[mfl['Facility_ID'] == the_facility_id, 'Facility_Level'].values[0]

        # Transform the treatment footprint into a demand for time for officers of each type, for this
        # facility level (it varies by facility level)
        appts_with_duration = [appt_type for appt_type in appt_types if the_appt_footprint[appt_type] > 0]
        df_appt_footprint = appt_times.loc[
            (appt_times['Facility_Level'] == the_facility_level) & appt_times.Appt_Type_Code.isin(appts_with_duration),
            ['Officer_Type_Code', 'Time_Taken'],
        ].copy()

        assert len(df_appt_footprint) > 0, \
            "The time needed for this appointment" \
            " is not defined for this specified facility level in the Appt_Time_Table. " \
            "And it should not go to this point" \
            ": " + hsi_event.TREATMENT_ID

        # Using f string or format method throws and error when df_appt_footprint is empty so hybrid used
        df_appt_footprint.set_index(
            f'FacilityID_{the_facility_id}_Officer_' + df_appt_footprint['Officer_Type_Code'].astype(str), inplace=True
        )

        # Create Series of summed required time for each officer type
        appt_footprint_as_time_request = df_appt_footprint['Time_Taken'].groupby(level=0).sum()

        # Check that indicies are unique
        assert not any(appt_footprint_as_time_request.index.duplicated())

        # Return
        return appt_footprint_as_time_request

    def get_squeeze_factors(self, all_calls_today, current_capabilities):
        """
        This will compute the squeeze factors for each HSI event from the dataframe that lists all the calls on health
        system resources for the day.
        The squeeze factor is defined as (call/available - 1). ie. the highest fractional over-demand among any type of
        officer that is called-for in the appt_footprint of an HSI event.
        A value of 0.0 signifies that there is no squeezeing (sufficient resources for the EXPECTED_APPT_FOOTPRINT).
        A value of 99.99 signifies that the call is for an officer_type in a health-facility that is not available.

        :param all_calls_today: Dataframe, one column per HSI event, containing the minutes required from each health
            officer in each health facility (using the standard index)
        :param current_capabilities: Dataframe giving the amount of time available from each health officer in each
            health facility (using the standard index)

        :return: squeeze_factors: a list of the squeeze factors for each HSI event
            (position in list matches column number in the all_call_today dataframe).
        """

        # 1) Compute the load factors
        total_call = all_calls_today.sum(axis=1)
        total_available = current_capabilities['Total_Minutes_Per_Day']

        load_factor = (total_call / total_available) - 1
        load_factor.loc[pd.isnull(load_factor)] = 99.99
        load_factor = load_factor.where(load_factor > 0, 0)

        # 5) Convert these load-factors into an overall 'squeeze' signal for each appointment_type requested
        squeeze_factor_per_hsi_event = list()  # The "squeeze factor" for each HSI event
        # [based on the highest load-factor of any officer required]

        for col_num in np.arange(0, len(all_calls_today.columns)):
            load_factor_per_officer_needed = list()
            officers_needed = all_calls_today.loc[all_calls_today[col_num] > 0, col_num].index.astype(str)
            assert len(officers_needed) > 0
            for officer in officers_needed:
                load_factor_per_officer_needed.append(load_factor.loc[officer])
            squeeze_factor_per_hsi_event.append(max(load_factor_per_officer_needed))

        assert len(squeeze_factor_per_hsi_event) == len(all_calls_today.columns)
        assert (np.asarray(squeeze_factor_per_hsi_event) >= 0).all()

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
        # 0) Check the format of the cons_req_as_footprint:
        self.check_consumables_footprint_format(cons_req_as_footprint)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.disable:
            # If the healthsystem module is disabled, return True for all consuambles
            # without checking or logging.
            packages_availability = dict()
            if not cons_req_as_footprint['Intervention_Package_Code'] == {}:
                for pkg_code in cons_req_as_footprint['Intervention_Package_Code'].keys():
                    packages_availability[pkg_code] = True

            # Iterate through the individual items that were requested
            items_availability = dict()
            if not cons_req_as_footprint['Item_Code'] == {}:
                for itm_code in cons_req_as_footprint['Item_Code'].keys():
                    items_availability[itm_code] = True

            # compile output
            output = dict()
            output['Intervention_Package_Code'] = packages_availability
            output['Item_Code'] = items_availability
            return output
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 0) Get information about the hsi_event
        the_facility_level = hsi_event.ACCEPTED_FACILITY_LEVEL
        the_treatment_id = hsi_event.TREATMENT_ID
        the_person_id = hsi_event.target

        # 1) Unpack the consumables footprint into the individual items
        items_req = self.get_consumables_as_individual_items(cons_req_as_footprint)
        n_items_req = len(items_req)

        # 2) Determine if these are available at the relevant facility level
        select_col = f'Available_Facility_Level_{the_facility_level}'
        availability = pd.DataFrame(data={'Available': self.cons_item_code_availability_today[select_col].copy()})
        items_req = items_req.merge(availability, left_on='Item_Code', right_index=True, how='left')
        assert len(items_req) == n_items_req

        # 3) Enter the the log (logs each item)
        # Do a groupby for the different consumables (there could be repeats of individual items which need /
        # to be summed)
        if to_log:
            items_req_to_log = pd.DataFrame(items_req.groupby('Item_Code').sum())
            items_req_to_log['Available'] = items_req_to_log['Available'] > 0  # restore to bool after sum in grouby()

            # Get the the cost of the each consumable item (could not do this merge until after model run)
            consumable_costs = self.parameters['Consumables_Cost_List']

            items_req_to_log = items_req_to_log.merge(consumable_costs, how='left', left_index=True, right_index=True)

            # Compute total cost (limiting to those items which were available)
            total_cost = (
                items_req_to_log.loc[items_req_to_log['Available'], ['Quantity_Of_Item', 'Unit_Cost']]
                .prod(axis=1).sum()
            )

            # Enter to the log
            items_req_to_log = items_req_to_log.drop(['Package_Code'], axis=1)  # drop from log for neatness

            log_consumables = items_req_to_log.to_dict()
            log_consumables['TREATMENT_ID'] = the_treatment_id
            log_consumables['Total_Cost'] = total_cost
            log_consumables['Person_ID'] = the_person_id

            logger.info('%s|Consumables|%s', self.sim.date, log_consumables)

        # 4) Format outcome into the CONS_FOOTPRINT format for return to HSI event
        # Iterate through the packages that were requested
        packages_availability = dict()
        if not cons_req_as_footprint['Intervention_Package_Code'] == {}:
            for package_code in cons_req_as_footprint['Intervention_Package_Code'].keys():
                packages_availability[package_code] = (
                    items_req.loc[items_req['Package_Code'] == package_code, 'Available'].all()
                )

        # Iterate through the individual items that were requested
        items_availability = dict()
        if not cons_req_as_footprint['Item_Code'] == {}:
            for item_code in cons_req_as_footprint['Item_Code'].keys():
                items_availability[item_code] = (
                    items_req.loc[items_req['Item_Code'] == item_code, 'Available'].values[0]
                )

        # compile output
        output = dict()
        output['Intervention_Package_Code'] = packages_availability
        output['Item_Code'] = items_availability

        return output

    def check_consumables_footprint_format(self, cons_req_as_footprint):
        """
        This function runs some check on the cons_footprint to ensure its in the right format
        :param cons_footprint:
        :return:
        """

        # Format is as follows:
        #     * dict with two keys; Intervention_Package_Code and Item_Code
        #     * For each, there is list of dicts, each dict giving code (i.e. package_code or item_code):quantity
        #     * the codes within each list must be unique and valid codes, quantities must be integer values >0
        #     e.g.
        #     cons_footprint = {
        #                 'Intervention_Package_Code': {my_pkg_code: 1},
        #                 'Item_Code': {my_item_code: 10}, {another_item_code: 1}
        #     }

        # check basic formatting
        assert 'Intervention_Package_Code' in cons_req_as_footprint
        assert 'Item_Code' in cons_req_as_footprint
        assert isinstance(cons_req_as_footprint['Intervention_Package_Code'], dict)
        assert isinstance(cons_req_as_footprint['Item_Code'], dict)

        # check that consumables being required are in the database:
        consumables = self.parameters['Consumables']

        # Check packages
        for pkg_code, pkg_quant in cons_req_as_footprint['Intervention_Package_Code'].items():
            assert pkg_code in consumables['Intervention_Pkg_Code'].values
            assert isinstance(pkg_quant, int)
            assert pkg_quant > 0

        # Check items
        for itm_code, itm_quant in cons_req_as_footprint['Item_Code'].items():
            assert itm_code in consumables['Item_Code'].values
            assert isinstance(itm_quant, int)
            assert itm_quant > 0

    def get_consumables_as_individual_items(self, cons_footprint):
        """
        This will look at the CONS_FOOTPRINT of an HSI Event and return a dataframe with the individual items that
        are used, collecting these from across the packages and the individual items that are specified.
        A column indicates the package from which the item come from and shows NaN if the item is requested individually
        """

        # Shortcut to the input cons_footprint
        cons = cons_footprint

        # Load the data on consumables
        consumables = self.parameters['Consumables']

        individual_consumables = []
        # Get the individual items in each package:
        for (package_code, quantity_of_packages) in cons['Intervention_Package_Code'].items():
            items = consumables.loc[
                consumables['Intervention_Pkg_Code'] == package_code, ['Item_Code', 'Expected_Units_Per_Case']
            ].to_dict(orient='records')
            for item in items:
                item['Quantity_Of_Item'] = item['Expected_Units_Per_Case'] * quantity_of_packages
                item['Package_Code'] = package_code
                individual_consumables.append(item)

        # Add in any additional items that have been specified seperately:
        for (item_code, quantity_of_item) in cons['Item_Code'].items():
            item = {
                'Item_Code': item_code,
                'Package_Code': np.nan,
                'Quantity_Of_Item': quantity_of_item,
                'Expected_Units_Per_Case': np.nan,
            }
            individual_consumables.append(item)

        consumables_as_individual_items = pd.DataFrame.from_dict(individual_consumables)

        try:
            consumables_as_individual_items = consumables_as_individual_items.drop('Expected_Units_Per_Case', axis=1)
        except KeyError:
            # No data from cons['Intervention_Package_Code'] or cons['Item_Code']
            raise ValueError("No packages or individual items requested")

        # confirm that Item_Code is returned as an int, Package_Code and Expected_Units_Per_Case as float
        # NB. package_code is held as float as may as np.nan's in and known issuse that pandas cannot handle this
        #       Non-null package_code must be coerced to int before use.
        consumables_as_individual_items['Item_Code'] = consumables_as_individual_items['Item_Code'].astype(int)

        return consumables_as_individual_items

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

            appts = actual_appt_footprint
            log_info = dict()
            log_info['TREATMENT_ID'] = hsi_event.TREATMENT_ID
            # key appointment types that are non-zero
            log_info['Number_By_Appt_Type_Code'] = {
                key: val for key, val in appts.items() if val
            }
            log_info['Person_ID'] = hsi_event.target

            if squeeze_factor == np.inf:
                log_info['Squeeze_Factor'] = 100.0  # arbitrarily high value to replace infinity
            else:
                log_info['Squeeze_Factor'] = squeeze_factor

        log_info['did_run'] = did_run

        logger.info('%s|HSI_Event|%s', self.sim.date, log_info)

    def log_current_capabilities(self, current_capabilities, all_calls_today):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type
        NB. To get this per Officer_Type_Code, it would be possible to simply log the entire current_capabilities df.
        :param current_capabilities: the current_capabilities of the health system.
        :param all_calls_today: dataframe of all the HSI events that ran
        """

        # Combine the current_capabiliites and the sum-across-columns of all_calls_today
        comparison = current_capabilities[['Facility_ID', 'Total_Minutes_Per_Day']].merge(
            all_calls_today.sum(axis=1).to_frame(), left_index=True, right_index=True, how='inner'
        )
        comparison = comparison.rename(columns={0: 'Minutes_Used'})
        assert len(comparison) == len(current_capabilities)
        assert (
            abs(comparison['Minutes_Used'].sum() - all_calls_today.sum().sum()) <= 0.0001 * all_calls_today.sum().sum()
        )

        # Sum within each Facility_ID using groupby (Index of 'summary' is Facility_ID)
        summary = comparison.groupby('Facility_ID')[['Total_Minutes_Per_Day', 'Minutes_Used']].sum()

        # Compute Fraction of Time Used Across All Facilities
        fraction_time_used_across_all_facilities = 0.0  # no capabilities or nan arising
        if summary['Total_Minutes_Per_Day'].sum() > 0:
            fraction_time_used_across_all_facilities = (
                summary['Minutes_Used'].sum() / summary['Total_Minutes_Per_Day'].sum()
            )

        # Compute Fraction of Time Used In Each Facility
        summary['Fraction_Time_Used'] = summary['Minutes_Used'] / summary['Total_Minutes_Per_Day']
        summary['Fraction_Time_Used'].replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)

        # Put out to the logger
        logger.debug('-------------------------------------------------')
        logger.debug('Current State of Health Facilities Appts:')
        print_table = summary.to_string().splitlines()
        for line in print_table:
            logger.debug(line)
        logger.debug('-------------------------------------------------')

        log_capacity = dict()
        log_capacity['Frac_Time_Used_Overall'] = fraction_time_used_across_all_facilities
        log_capacity['Frac_Time_Used_By_Facility_ID'] = summary['Fraction_Time_Used'].to_dict()

        logger.info('%s|Capacity|%s', self.sim.date, log_capacity)


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

        logger.debug(
            'HealthSystemScheduler>> I will now determine what calls on resource will be met today: %s', self.sim.date
        )

        # 0) Determine the availability of consumables today based on their probabilities

        # random draws: assume that availability of the same item is independent between different facility levels
        random_draws = self.module.rng.rand(
            len(self.module.prob_unique_item_codes_available), len(self.module.prob_unique_item_codes_available.columns)
        )

        # Determine the availability of the consumables today
        if not self.module.ignore_cons_constraints:
            self.module.cons_item_code_availability_today = self.module.prob_unique_item_codes_available > random_draws
        else:
            # Make all true if ignoring consumables constraints
            self.module.cons_item_code_availability_today = self.module.prob_unique_item_codes_available > 0.0

        logger.debug('----------------------------------------------------------------------')
        logger.debug("This is the entire HSI_EVENT_QUEUE heapq:")
        logger.debug(self.module.HSI_EVENT_QUEUE)
        logger.debug('----------------------------------------------------------------------')

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
                # The event has expired (after tclose), do nothing more
                pass

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
            # empty dataframe for logging
            df_footprints_of_all_individual_level_hsi_event = pd.DataFrame(index=current_capabilities.index)
        else:
            # 4) Examine total call on health officers time from the HSI events that are due today

            # For all events in 'list_of_individual_hsi_event_tuples_due_today',
            # expand the appt-footprint of the event into give the demands on
            # each officer-type in each facility_id. [Name of columns is the position in the list of event_due_today)

            footprints_of_all_individual_level_hsi_event = {
                event_number: self.module.get_appt_footprint_as_time_request(hsi_event=(event_tuple[4]))
                for event_number, event_tuple in enumerate(list_of_individual_hsi_event_tuples_due_today)
            }

            # dataframe to store all the calls to the healthsystem today
            df_footprints_of_all_individual_level_hsi_event = pd.DataFrame(
                footprints_of_all_individual_level_hsi_event, index=current_capabilities.index
            )
            df_footprints_of_all_individual_level_hsi_event.fillna(0, inplace=True)

            assert len(df_footprints_of_all_individual_level_hsi_event.columns) == len(
                list_of_individual_hsi_event_tuples_due_today
            )
            assert df_footprints_of_all_individual_level_hsi_event.index.equals(current_capabilities.index)

            # 5) Estimate Squeeze-Factors for today
            if self.module.mode_appt_constraints == 0:
                # For Mode 0 (no Constraints), the squeeze factors are all zero.
                squeeze_factor_per_hsi_event = [0] * len(df_footprints_of_all_individual_level_hsi_event.columns)
            else:
                # For Other Modes, the squeeze factors must be computed
                squeeze_factor_per_hsi_event = self.module.get_squeeze_factors(
                    all_calls_today=df_footprints_of_all_individual_level_hsi_event,
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

                # Mode 0: All HSI Event run
                # Mode 1: All Run
                # Mode 2: Only if squeeze <1

                if ok_to_run:

                    # Run the HSI event (allowing it to return an updated appt_footprint)
                    actual_appt_footprint = event.run(squeeze_factor=squeeze_factor)

                    # Check if the HSI event returned updated appt_footprint
                    if actual_appt_footprint is not None:
                        # The returned footprint is different to the expected footprint: so must update load factors

                        # check its formatting:
                        self.module.check_appt_footprint_format(actual_appt_footprint)

                        # Update load factors:
                        updated_call = self.module.get_appt_footprint_as_time_request(event, actual_appt_footprint)
                        df_footprints_of_all_individual_level_hsi_event.loc[updated_call.index, ev_num] = updated_call
                        squeeze_factor_per_hsi_event = self.get_squeeze_factors(
                            all_calls_today=df_footprints_of_all_individual_level_hsi_event,
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
            current_capabilities=current_capabilities, all_calls_today=df_footprints_of_all_individual_level_hsi_event
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

    def apply(self, *args, **kwargs):
        """Apply this event to the population.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def did_not_run(self, *args, **kwargs):
        """Called when this event is due but it is not run. Return False to prevent the event being rescheduled.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def post_apply_hook(self):
        """Do any required processing after apply() completes."""
        pass

    def run(self, squeeze_factor):
        """Make the event happen."""
        self.apply(self.target, squeeze_factor)
        self.post_apply_hook()


class HSIEventWrapper(Event):
    # This is wrapper that contains an HSI event.
    # It is used when the healthsystem is 'disabled' and all HSI events sent to the health system scheduler should
    # be passed to the main simulation scheduler.
    # When this event is run (by the simulation scheduler) it runs the HSI event with squeeze_factor=0.0

    def __init__(self, hsi_event):
        self.hsi_event = hsi_event

    def run(self):
        # check that the person is still alive
        # (this check normally happens in the HealthSystemScheduler and silently do not run the HSI event)

        if isinstance(self.hsi_event.target, tlo.population.Population) \
                or (self.hsi_event.module.sim.population.props.at[self.hsi_event.target, 'is_alive']):
            _ = self.hsi_event.run(squeeze_factor=0.0)
