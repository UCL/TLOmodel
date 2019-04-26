"""
This is the Health System Module
"""
import logging
import os

import pandas as pd
import numpy as np
import heapq as hp

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HealthSystem(Module):
    """
    Requests for access to particular services are handled by Disease/Intervention Modules by this Module
    """

    def __init__(self, name=None,
                 resourcefilepath=None,
                 service_availability='all'):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Checks on the service_availability dateframe argument
        assert (service_availability == 'all') or (service_availability == 'none') or (
                type(service_availability) == list)

        self.service_availability = service_availability

        self.registered_disease_modules = {}

        self.registered_interventions = pd.DataFrame(
            columns=['Name', 'Nurse_Time', 'Doctor_Time', 'Electricity', 'Water'])

        self.health_system_resources = None

        self.HEALTH_SYSTEM_CALLS = []  # This will be the heapq for storing all the health system calls
        self.heap_counter = 0  # Counter to help with the sorting in the heapq

        logger.info('----------------------------------------------------------------------')
        logger.info("Setting up the Health System With the Following Service Availabilty:")
        logger.info(service_availability)
        logger.info('----------------------------------------------------------------------')

    PARAMETERS = {

        'Officer_Types':
            Parameter(Types.DATA_FRAME,
                      'The names of the types of health workers ("officers")'),

        'Daily_Capabilities':
            Parameter(Types.DATA_FRAME,
                      'The capabilities by facility and officer type available each day'),

        'Appt_Types_Table':
            Parameter(Types.DATA_FRAME,
                      'The names of the type of appointments with the health system'),

        'Appt_Time_Table':
            Parameter(Types.DATA_FRAME,
                      'The time taken for each appointment, according to officer and facility type.'),

        'Master_Facilities_List':
            Parameter(Types.DATA_FRAME,
                      'Listing of all health facilities.'),

        'Facilities_For_Each_District':
            Parameter(Types.DATA_FRAME,
                      'Mapping between a district and all of the health facilities to which its population have access.'),

        'Consumables':
            Parameter(Types.DATA_FRAME,
                      'List of consumables used in each intervention and their costs.'),

        'Consumables_Cost_List':
            Parameter(Types.DATA_FRAME,
                      'List of each consumable item and it''s cost')

    }

    PROPERTIES = {
        'Distance_To_Nearest_HealthFacility':
            Property(Types.REAL,
                     'The distance for each person to their nearest clinic (of any type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Officer_Types_Table'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Officer_Types_Table.csv')
        )

        self.parameters['Daily_Capabilities'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Daily_Capabilities.csv')
        )

        self.parameters['Appt_Types_Table'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Appt_Types_Table.csv')
        )

        self.parameters['Appt_Time_Table'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Appt_Time_Table.csv')
        )

        self.parameters['Master_Facilities_List'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Master_Facilities_List.csv')
        )

        self.parameters['Facilities_For_Each_District'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Facilities_For_Each_District.csv')
        )

        self.parameters['Consumables'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Consumables.csv')
        )

        self.parameters['Consumables_Cost_List'] = (self.parameters['Consumables'][['Item_Code', 'Unit_Cost']]) \
            .drop_duplicates().reset_index(drop=True)

    def initialise_population(self, population):
        df = population.props

        # Assign Distance_To_Nearest_HealthFacility'
        # For now, let this be a random number, but in future it will be properly informed based on population density distribitions.
        # Note that this characteritic is inherited from mother to child.
        df['Distance_To_Nearest_HealthFacility'] = self.sim.rng.uniform(0.01, 5.00, len(df))

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

        # Launch the healthsystem_scheduler (a regular event occurring each day)
        sim.schedule_event(HealthSystemScheduler(self), sim.date)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'Distance_To_Nearest_HealthFacility'] = \
            df.at[mother_id, 'Distance_To_Nearest_HealthFacility']

    def register_disease_module(self, *new_disease_modules):
        # Register Disease Modules (so that the health system can broadcast triggers to all disease modules)
        for module in new_disease_modules:
            assert module.name not in self.registered_disease_modules, (
                'A module named {} has already been registered'.format(module.name))
            self.registered_disease_modules[module.name] = module

            logger.info('Registering disease module %s', module.name)

            # TODO: Check that the module being registered has the neccessary components.

    def schedule_event(self, treatment_event, priority, topen, tclose=None):

        logger.debug('HealthSystem.schedule_event>>Logging a request for an HSI: %s for person: %s',
                     treatment_event.TREATMENT_ID, treatment_event.target)

        # get population dataframe
        df = self.sim.population.props

        # Check that this is a legitimate request for a treatment

        # 1) Correctly formatted footprint
        assert 'APPT_FOOTPRINT' in dir(treatment_event)
        assert set(treatment_event.APPT_FOOTPRINT.keys()) == set(self.parameters['Appt_Types_Table']['Appt_Type_Code'])
        list(treatment_event.APPT_FOOTPRINT.values())

        # 2) All sensible numbers for the number of appointments requested (no negative and at least one appt req)
        assert all(
            np.asarray([(treatment_event.APPT_FOOTPRINT[k]) for k in treatment_event.APPT_FOOTPRINT.keys()]) >= 0)
        assert not all(value == 0 for value in treatment_event.APPT_FOOTPRINT.values())

        # 3) That it has a dictionary for the consumables needed in the right format
        assert 'CONS_FOOTPRINT' in dir(treatment_event)
        assert type(treatment_event.CONS_FOOTPRINT['Intervention_Package_Code']) == list
        assert type(treatment_event.CONS_FOOTPRINT['Item_Code']) == list
        consumables = self.parameters['Consumables']
        assert (treatment_event.CONS_FOOTPRINT['Intervention_Package_Code'] == []) or (
            set(treatment_event.CONS_FOOTPRINT['Intervention_Package_Code']).issubset(
                consumables['Intervention_Pkg_Code']))
        assert (treatment_event.CONS_FOOTPRINT['Item_Code'] == []) or (
            set(treatment_event.CONS_FOOTPRINT['Item_Code']).issubset(consumables['Item_Code']))

        # 4) That it has a list for the other disease that will be alerted when it is run and that this make sense
        assert 'ALERT_OTHER_DISEASES' in dir(treatment_event)
        assert type(treatment_event.ALERT_OTHER_DISEASES) is list

        if len(treatment_event.ALERT_OTHER_DISEASES) > 0:
            if not (treatment_event.ALERT_OTHER_DISEASES[0] == '*'):
                for d in treatment_event.ALERT_OTHER_DISEASES:
                    assert d in self.sim.modules['HealthSystem'].registered_disease_modules.keys()

        # 5) Check that this request is allowable under current policy (i.e. included in service_availability)
        allowed = False
        if (self.service_availability == 'all') or (self.service_availability[0] == '*'):
            allowed = True
        elif (self.service_availability == 'none') or (self.service_availability[0] == []):
            allowed = False
        elif (treatment_event.TREATMENT_ID in self.service_availability):
            allowed = True
        elif treatment_event.TREATMENT_ID == None:
            allowed = True  # (if no treatment_id it can pass)
        else:
            # check to see if anything provided given any wildcards
            for s in range(len(self.service_availability)):
                if '*' in self.service_availability[s]:
                    stub = self.service_availability[s].split('*')[0]
                    if treatment_event.TREATMENT_ID.startswith(stub):
                        allowed = True
                        break

        # If there is no specified tclose time then set this is after the end of the simulation
        if (tclose == None):
            tclose = self.sim.end_date + DateOffset(days=1)

        # If it is allowed then add this request to the queue of HEALTH_SYSTEM_CALLS
        if allowed:

            # First two parts of tuple used for sorting (smaller values come first), last part is the event
            # (topen, priority []) [ie. sort by order in which needed, and then the priority of the call]

            new_request = (topen, priority, self.heap_counter, treatment_event, tclose)
            self.heap_counter = self.heap_counter + 1

            hp.heappush(self.HEALTH_SYSTEM_CALLS, new_request)

        else:
            logger.debug(
                '%s| A request was made for a service but it was not included in the service_availability list: %s',
                self.sim.date,
                treatment_event.TREATMENT_ID)

    def broadcast_healthsystem_interaction(self, event):

        """
        This will alert disease modules than a treatment_id is occuring to a particular person.
        It receives an HSI event object
        """

        # Get the name of the disease module that this event came from ultimately
        originating_disease_module_name = event.module.name

        if event.ALERT_OTHER_DISEASES == []:
            alert_modules = []

        else:
            if event.ALERT_OTHER_DISEASES[0] == '*':
                alert_modules = list(self.registered_disease_modules.keys())
            else:
                alert_modules = event.ALERT_OTHER_DISEASES

            if originating_disease_module_name in alert_modules:
                alert_modules.remove(originating_disease_module_name)

        for module_name in alert_modules:
            module = self.registered_disease_modules.values[module_name]
            module.on_healthsystem_interactionn(person_id=event.target,
                                                treatment_id=event.TREATMENT_ID)


    def GetCapabilities(self):

        """
        This will return a dataframe of the capabilities that the healthsystem has for today.
        Function can go in here in the future that could expand the time available, simulating increasing efficiency.
        """

        capabilities = self.parameters['Daily_Capabilities']

        return capabilities

    def get_blank_appt_footprint(self):
        """
        This is a helper function so that disease modules can easily create their footprint.
        It returns a dataframe containing the appointment footprint information in the format that the HealthSystemScheduler expects.
        """

        keys = self.parameters['Appt_Types_Table']['Appt_Type_Code']
        values = np.zeros(len(keys))
        blank_footprint = dict(zip(keys, values))
        return (blank_footprint)

    def get_blank_cons_footprint(self):
        """
        This is a helper function so that disease modules can easily create their footprint.
        It returns a dictionary containing the consumables information in the format that the HealthSystemScheduler expects.
        """

        blank_footprint = {
            'Intervention_Package_Code': [],
            'Item_Code': []
        }
        return (blank_footprint)

    def get_prob_seek_care(self, person_id, symptom_code=0):
        """
        This gives the probability that a person who had developed a particular symptom will seek care.
        Disease modules call this when a person has symptoms onset to determine if there will be a health interaction.
        """
        # It currently just returns 1.0

        return 1.0

    def check_if_can_do_appt_footprint(self, event, current_capabilities):
        """
        This will determine if an HSI event can run given the constraints that capabilities available.
        It accepts the argument of the HSI event itself and a dataframe describing the current capabilities of the health
        system.
        It returns a tuple:
            * 1st element: True/False about whether the footprint can be accomodated
            * 2nd element: an updated version of current capabilities (unchanged in the case of the footprint not being able to be accomodated)
        """

        assert not pd.isnull(current_capabilities['Minutes_Remaining_Today']).any()

        # Gather useful information
        df = self.sim.population.props
        mfl = self.parameters['Master_Facilities_List']
        fac_per_district = self.parameters['Facilities_For_Each_District']
        appt_types = self.parameters['Appt_Types_Table']['Appt_Type_Code'].values
        appt_times = self.parameters['Appt_Time_Table']
        officer_type_codes = self.parameters['Officer_Types_Table']['Officer_Type_Code'].values

        # Gather information about the HSI event
        the_person_id = event.target
        the_treatment_event_name = event.TREATMENT_ID
        the_district = df.at[the_person_id, 'district_of_residence']

        # Get th health_facilities available to this person and sort by Facility_Level
        the_health_facilities = fac_per_district.loc[fac_per_district['District'] == the_district]
        the_health_facilities = the_health_facilities.sort_values(['Facility_Level'])

        capabilities_of_the_health_facilities = current_capabilities.loc[
            current_capabilities['Facility_ID'].isin(the_health_facilities['Facility_ID'])]

        the_treatment_footprint = event.APPT_FOOTPRINT

        # ------------
        # Test if capabilities of health system can meet this request
        # This requires there to be at least one facility that can fulfill the entire request (each type of appointment)

        # Loop through facilities to look for facilities
        # (Note that as the health_facilities dataframe was sorted on Facility_Level, this will start
        #  at the lowest levels and work upwards successively).

        can_do_footprint = False  # set outcome to False and see if it is overwritten in the checks

        for try_fac_id in the_health_facilities.Facility_ID.values:

            this_facility_level = mfl.loc[mfl['Facility_ID'] == try_fac_id, 'Facility_Level'].values[0]

            # Establish how much time is available at this facility
            time_available = capabilities_of_the_health_facilities.loc[
                capabilities_of_the_health_facilities['Facility_ID'] == try_fac_id, ['Officer_Type_Code',
                                                                                     'Minutes_Remaining_Today']]

            # Transform the treatment footprint into a demand for time for officers of each type, for this facility type
            time_requested = pd.DataFrame(columns=['Officer_Type_Code', 'Time_Taken'])
            for this_appt in appt_types:
                if the_treatment_footprint[this_appt] > 0:
                    time_req_for_this_appt = appt_times.loc[(appt_times['Appt_Type_Code'] == this_appt) &
                                                            (appt_times['Facility_Level'] == this_facility_level),
                                                            ['Officer_Type_Code',
                                                             'Time_Taken']].copy().reset_index(drop=True)
                    time_requested = pd.concat([time_requested, time_req_for_this_appt])

            if len(time_requested) > 0:
                # (If the data-frame of time-requested is empty, it means that the appointments is not possible
                # at that type of facility. So we check that time_requested is not empty.)
                # -------------------------

                # Collapse down the total_time_requested dataframe to give a sum of Time Taken by each Officer_Type_Code
                time_requested = pd.DataFrame(
                    time_requested.groupby(['Officer_Type_Code'])['Time_Taken'].sum()).reset_index()
                time_requested = time_requested.drop(time_requested[time_requested['Time_Taken'] == 0].index)

                # merge the Minutes_Available at this facility with the minutes required in the footprint
                comparison = time_requested.merge(time_available, on='Officer_Type_Code', how='left',
                                                  indicator=True)

                # check if all the needs are met by this facility
                if (all(comparison['_merge'] == 'both') &
                    all(comparison['Minutes_Remaining_Today'] > comparison['Time_Taken'])
                ):

                    can_do_footprint = True

                    # Now, impose the footprint for each one of the types being requested
                    for this_officer_type in time_requested['Officer_Type_Code'].values.tolist():
                        old_mins_remaining = \
                            current_capabilities.loc[
                                (current_capabilities['Facility_ID'] == try_fac_id) & (current_capabilities[
                                                                                           'Officer_Type_Code'] == this_officer_type), 'Minutes_Remaining_Today'].values[
                                0]

                        time_to_take_away = \
                            time_requested.loc[
                                time_requested['Officer_Type_Code'] == this_officer_type, 'Time_Taken'].values[0]

                        new_mins_remaining = \
                            old_mins_remaining - time_to_take_away

                        assert (new_mins_remaining >= 0)

                        # update
                        current_capabilities.loc[
                            (current_capabilities['Facility_ID'] == try_fac_id) & (current_capabilities[
                                                                                       'Officer_Type_Code'] == this_officer_type), 'Minutes_Remaining_Today'] = \
                            new_mins_remaining

                    break  # cease looking at other facility_types if the need has been met

        assert not pd.isnull(current_capabilities['Minutes_Remaining_Today']).any()

        rtn_tuple = (can_do_footprint, current_capabilities)
        return (rtn_tuple)

    def log_consumables_used(self, event):
        """
        This will write to the log with a record of the consumables that were in the footprint of an HSI event

        """
        # Consumables:
        consumables = self.parameters['Consumables']
        consumables_used = pd.DataFrame(columns=['Item_Code', 'Expected_Units_Per_Case'])
        cons = event.CONS_FOOTPRINT

        # Get the individual items in each package:
        if not cons['Intervention_Package_Code'] == []:
            for p in cons['Intervention_Package_Code']:
                items = consumables.loc[
                    consumables['Intervention_Pkg_Code'] == p, ['Item_Code', 'Expected_Units_Per_Case']]
                consumables_used = consumables_used.append(items, ignore_index=True, sort=False).reset_index(drop=True)

        # Add in any additional items specified:
        if not cons['Item_Code'] == []:
            for i in cons['Item_Code']:
                items = pd.DataFrame(data={'Item_Code': i, 'Expected_Units_Per_Case': 1}, index=[0])
                consumables_used = consumables_used.append(items, ignore_index=True, sort=False).reset_index(
                    drop=True)

        if len(consumables_used) > 0:
            # do a groupby for the different consumables (there could be repeats which need to be summed)
            consumables_used = pd.DataFrame(consumables_used.groupby('Item_Code').sum())
            consumables_used = consumables_used.rename(columns={'Expected_Units_Per_Case': 'Units_By_Item_Code'})

            # Get the the total cost of the consumables
            consumables_used_with_cost = consumables_used.merge(self.parameters['Consumables_Cost_List'],
                                                                how='left',
                                                                on='Item_Code',
                                                                left_index=True
                                                                )
            TotalCost = (
                    consumables_used_with_cost['Units_By_Item_Code'] * consumables_used_with_cost['Unit_Cost']).sum()

            # Enter to the log
            log_consumables = consumables_used.to_dict()
            log_consumables['TREATMENT_ID'] = event.TREATMENT_ID
            log_consumables['Total_Cost'] = TotalCost
            log_consumables['Person_ID'] = event.target

            logger.info('%s|Consumables|%s',
                        self.sim.date,
                        log_consumables)

    def log_appts_used(self, event):
        """
        This will write to the log with a record of the appts that were in the footprint of an HSI event
        """

        # Log the individual footprints being applied
        appts = event.APPT_FOOTPRINT
        log_appts = dict()
        log_appts['TREATMENT_ID'] = event.TREATMENT_ID
        log_appts['Number_By_Appt_Type_Code'] = {k: v for k, v in appts.items() if v}  # remove the zeros
        log_appts['Person_ID'] = event.target

        logger.info('%s|Appt|%s',
                    self.sim.date,
                    log_appts)

    def log_current_capabilities(self, current_capabilities):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type
        NB. To get this per Officer_Type_Code, it would be possible to simply log the entire current_capabilities df.
        """
        # Log the current state of resource use

        X = current_capabilities.groupby('Facility_ID')[['Total_Minutes_Per_Day', 'Minutes_Remaining_Today']].sum()
        overall_useage = 1 - (X['Minutes_Remaining_Today'].sum() / X['Total_Minutes_Per_Day'].sum())
        X = X.merge(self.parameters['Master_Facilities_List'][['Facility_ID', 'Facility_Name']], how='left',
                    left_index=True, right_on='Facility_ID')
        X['Fraction_Time_Used'] = 1 - (X['Minutes_Remaining_Today'] / X['Total_Minutes_Per_Day'])
        X = X[['Facility_Name', 'Fraction_Time_Used']]
        X.index = X['Facility_Name']
        X = X.drop(columns='Facility_Name')

        logger.debug('-------------------------------------------------')
        logger.debug('Current State of Health Facilities Appts:')
        print_table = X.to_string().splitlines()
        for line in print_table:
            logger.debug(line)
        logger.debug('-------------------------------------------------')

        log_capacity = dict()
        log_capacity['Frac_Time_Used_Overall'] = overall_useage
        log_capacity['Frac_Time_Used_By_Facility_Name'] = X.to_dict()

        logger.info('%s|Capacity|%s',
                    self.sim.date,
                    log_capacity)


# --------- SCHEDULING OF ACCESS TO HEALTH CARE -----
class HealthSystemScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This event occurs every day, inspects the calls on the healthsystem and commissions event to occur that
    are consistent with the healthsystem's capabilities for the following day, given assumptions about how this
    decision is made.
    Overall Prioritation algorithm is:
        * ignore is topen is not yet
        * remove and do nothing if tclose has expired
        * if prioirty = 0 then run and log resource use
        * if priority>0 then examine resource use and run/log accordingly
    At this point, we can have multiple types of assumption regarding how these capabilities are modelled.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):

        df = self.sim.population.props

        logger.debug('HealthSystemScheduler>> I will now determine what calls on resource will be met today: %s',
                     self.sim.date)

        logger.debug('----------------------------------------------------------------------')
        logger.debug("This is the entire HealthSystemCalls heapq:")
        logger.debug(self.module.HEALTH_SYSTEM_CALLS)
        logger.debug('----------------------------------------------------------------------')

        # Get the total Capabilities for Appts for today
        current_capabilities = self.module.GetCapabilities()

        # Add column for Minutes Remaining Today (for live-tracking the use of those resources)
        current_capabilities.loc[:, 'Minutes_Remaining_Today'] = current_capabilities['Total_Minutes_Per_Day']

        hold_over = list()  # This heapq will hold events that cannot occur today before they are added back to the heapq

        while len(self.module.HEALTH_SYSTEM_CALLS) > 0:

            # Look at next event and stop if it is not due by today
            # (The first entry will have the oldest topen date and it's not yet due, no events will be due)

            if self.module.HEALTH_SYSTEM_CALLS[0][0] > self.sim.date:
                break

            # Otherwise, pop the event:
            next_event_tuple = hp.heappop(self.module.HEALTH_SYSTEM_CALLS)

            next_event = dict({
                'topen': next_event_tuple[0],
                'priority': next_event_tuple[1],
                'object': next_event_tuple[3],
                'tclose': next_event_tuple[4]})

            event = next_event['object']

            if next_event['tclose'] < self.sim.date:
                # The event has expired, do nothing more
                pass

            else:

                # check if the event can run
                (can_do_appt_footprint, current_capabilities) = \
                    self.module.check_if_can_do_appt_footprint(event, current_capabilities=current_capabilities)

                if can_do_appt_footprint:
                    # The event can be run:

                    if df.at[event.target, 'is_alive']:
                        # Run the event
                        event.run()

                        # Broadcast to other modules that the event is running:
                        self.module.broadcast_healthsystem_interaction(event=event)

                        # Write to the log
                        self.module.log_consumables_used(event)
                        self.module.log_appts_used(event)


                else:
                    # The event will not be run and will be added back to the heapq
                    hp.heappush(hold_over, next_event_tuple)

        # Add back to the HEALTH_SYSTEM_CALLS heapq all those events which are eligible to run but which did not
        while len(hold_over) > 0:
            hp.heappush(self.module.HEALTH_SYSTEM_CALLS, hp.heappop(hold_over))

        # After completing routine for the day, log total usage of the facilities
        self.module.log_current_capabilities(current_capabilities)
