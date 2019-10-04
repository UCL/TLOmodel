import heapq as hp
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HealthSystem(Module):
    """
    This is the Health System Module
    Version: May 2019
    The execution of all health systems interactions are controlled through this module.
    """

    def __init__(self, name=None,
                 resourcefilepath=None,
                 service_availability=None,     # must be a list of treatment_ids to allow
                 ignore_appt_constraints=False,  # remove constraints to do with officer numbers and time
                 ignore_cons_constraints=False,  # remove constraints to do with consumables availability
                 ignore_priority=False,  # do not use the priroity information in HSI event to schedule
                 capabilities_coefficient=1.0):  # multiplier for the capabailities of health officers

        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.ignore_appt_constraints = ignore_appt_constraints
        self.ignore_cons_constraints = ignore_cons_constraints
        self.ignore_priority = ignore_priority

        print('ignore_appt_constraints', ignore_appt_constraints)

        self.Facility_Levels = None

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

        # Define the container for calls for health system interaction events
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0  # Counter to help with the sorting in the heapq

        logger.info('----------------------------------------------------------------------')
        logger.info("Setting up the Health System With the Following Service Availability:")
        logger.info(self.service_availability)
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
                      'Mapping between a district and all of the health facilities to which its \
                      population have access.'),

        'Consumables':
            Parameter(Types.DATA_FRAME,
                      'List of consumables used in each intervention and their costs.'),

        'Consumables_Cost_List':
            Parameter(Types.DATA_FRAME,
                      'List of each consumable item and it''s cost')

    }

    PROPERTIES = {
        'hs_dist_to_facility':
            Property(Types.REAL,
                     'The distance for each person to their nearest clinic (of any type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Officer_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Officer_Types_Table.csv'
        )

        self.parameters['Daily_Capabilities'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Daily_Capabilities.csv'
        )

        self.parameters['Appt_Types_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Types_Table.csv'
        )

        self.parameters['Appt_Time_Table'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Appt_Time_Table.csv'
        )

        self.parameters['Master_Facilities_List'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Master_Facilities_List.csv'
        )

        self.parameters['Facilities_For_Each_District'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Facilities_For_Each_District.csv'
        )

        self.parameters['Consumables'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Consumables.csv'
        )

        self.parameters['Consumables_Cost_List'] = (self.parameters['Consumables'][['Item_Code', 'Unit_Cost']]) \
            .drop_duplicates().reset_index(drop=True)

    def initialise_population(self, population):
        df = population.props

        # Assign hs_dist_to_facility'
        # (For now, let this be a random number, but in future it may be properly informed based on \
        #  population density distribitions)
        # Note that this characteritic is inherited from mother to child.
        df['hs_dist_to_facility'] = self.sim.rng.uniform(0.01, 5.00, len(df))

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

        # Launch the healthsystem scheduler (a regular event occurring each day)
        sim.schedule_event(HealthSystemScheduler(self), sim.date)

    def on_birth(self, mother_id, child_id):

        # New child inherits the hs_dist_to_facility of the mother
        df = self.sim.population.props
        df.at[child_id, 'hs_dist_to_facility'] = \
            df.at[mother_id, 'hs_dist_to_facility']

    def register_disease_module(self, new_disease_module):
        """
        Register Disease Module
        In order for a disease module to use the functionality of the health system it must be registered
        This list is also used to alert other disease modules when a health system interaction occurs

        :param new_disease_module: The pointer to the disease module
        """
        assert new_disease_module.name not in self.registered_disease_modules, (
            'A module named {} has already been registered'.format(new_disease_module.name))
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

        logger.debug('HealthSystem.schedule_event>>Logging a request for an HSI: %s for person: %s',
                     hsi_event.TREATMENT_ID, hsi_event.target)

        # 1) Check that this is a legitimate health system interaction (HSI) event

        # Correctly formatted footprint
        assert 'APPT_FOOTPRINT' in dir(hsi_event)
        assert set(hsi_event.APPT_FOOTPRINT.keys()) == set(self.parameters['Appt_Types_Table']['Appt_Type_Code'])

        # All sensible numbers for the number of appointments requested (no negative and at least one appt required)
        assert all(
            np.asarray([(hsi_event.APPT_FOOTPRINT[k]) for k in hsi_event.APPT_FOOTPRINT.keys()]) >= 0)
        assert not all(value == 0 for value in hsi_event.APPT_FOOTPRINT.values())

        # That it has a dictionary for the consumables needed in the right format
        assert 'CONS_FOOTPRINT' in dir(hsi_event)
        assert type(hsi_event.CONS_FOOTPRINT['Intervention_Package_Code']) == list
        assert type(hsi_event.CONS_FOOTPRINT['Item_Code']) == list
        consumables = self.parameters['Consumables']
        assert (hsi_event.CONS_FOOTPRINT['Intervention_Package_Code'] == []) or (
            set(hsi_event.CONS_FOOTPRINT['Intervention_Package_Code']).issubset(
                consumables['Intervention_Pkg_Code']))
        assert (hsi_event.CONS_FOOTPRINT['Item_Code'] == []) or (
            set(hsi_event.CONS_FOOTPRINT['Item_Code']).issubset(consumables['Item_Code']))

        # That it has an 'ACCEPTED_FACILITY_LEVELS' attribute (a list of Ok facility levels of a '*'
        # If it a list with one element '*' in it, then update that with all the facility levels
        assert 'ACCEPTED_FACILITY_LEVELS' in dir(hsi_event)
        assert type(hsi_event.ACCEPTED_FACILITY_LEVELS) is list
        assert len(hsi_event.ACCEPTED_FACILITY_LEVELS) > 0
        all_fac_levels = list(pd.unique(self.parameters['Facilities_For_Each_District']['Facility_Level']))
        if hsi_event.ACCEPTED_FACILITY_LEVELS[0] == '*':
            # replace the '*' with all the facility_levels being used
            hsi_event.ACCEPTED_FACILITY_LEVELS = all_fac_levels
        assert set(hsi_event.ACCEPTED_FACILITY_LEVELS).issubset(set(all_fac_levels))

        # That it has a list for the other disease that will be alerted when it is run and that this make sense
        assert 'ALERT_OTHER_DISEASES' in dir(hsi_event)
        assert type(hsi_event.ALERT_OTHER_DISEASES) is list

        if len(hsi_event.ALERT_OTHER_DISEASES) > 0:
            if not (hsi_event.ALERT_OTHER_DISEASES[0] == '*'):
                for d in hsi_event.ALERT_OTHER_DISEASES:
                    assert d in self.sim.modules['HealthSystem'].registered_disease_modules.keys()

        # 2) Check topen, tclose and priority

        # If there is no specified tclose time then set this is after the end of the simulation
        if tclose is None:
            tclose = self.sim.end_date + DateOffset(days=1)

        # Check topen is not in the past
        assert topen >= self.sim.date

        # Check that priority is either 0, 1 or 2
        assert priority in {0, 1, 2}

        # 3) Check that this request is allowable under current policy (i.e. included in service_availability)
        allowed = False
        if not self.service_availability:   # it's an empty list
            allowed = False
        elif self.service_availability[0] == '*':   # it's the overall wild-card, do anything
            allowed = True
        elif hsi_event.TREATMENT_ID in self.service_availability:
            allowed = True
        elif hsi_event.TREATMENT_ID is None:
            allowed = True  # (if no treatment_id it can pass)
        else:
            # check to see if anything provided given any wildcards
            for s in self.service_availability:
                if '*' in s:
                    stub = s.split('*')[0]
                    if hsi_event.TREATMENT_ID.startswith(stub):
                        allowed = True
                        break

        # 4) Manipulate the priority level

        # If ignoring the priority in scheduling, then over-write the provided priority information
        if self.ignore_priority:
            priority = 0  # set all event to priority 0
        # This is where could attach a different priority score according to the treatment_id (and other things)
        # in order to examine the influence of the prioritisation score.

        # 5) If all is correct and the hsi event is allowed then add this request to the queue of HSI_EVENT_QUEUE
        if allowed:

            # Create a tuple to go into the heapq
            # (NB. the sorting is done ascending and by the order of the items in the tuple)
            # Pos 0: priority,
            # Pos 1: topen,
            # Pos 2: hsi_event_queue_counter,
            # Pos 3: tclose,
            # Pos 4: the hsi_event itself

            new_request = (priority, topen, self.hsi_event_queue_counter, tclose, hsi_event)
            self.hsi_event_queue_counter = self.hsi_event_queue_counter + 1

            hp.heappush(self.HSI_EVENT_QUEUE, new_request)

            logger.debug('HealthSystem.schedule_event>>HSI has been added to the queue: %s for person: %s',
                         hsi_event.TREATMENT_ID, hsi_event.target)

        else:
            logger.debug(
                '%s| A request was made for a service but it was not included in the service_availability list: %s',
                self.sim.date,
                hsi_event.TREATMENT_ID)

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
                module.on_hsi_alert(person_id=hsi_event.target,
                                    treatment_id=hsi_event.TREATMENT_ID)

    def get_capabilities(self):
        """
        This will return a dataframe of the capabilities that the healthsystem has for today.
        Functions can go in here in the future that could expand the time available, simulating increasing efficiency.
        (The concept of a productivitiy ratio raised by Martin Chalkley). For now just have a single scaling value,
        named capabilities_coefficient.
        """

        capabilities = self.parameters['Daily_Capabilities']

        # apply the coefficient
        capabilities['Total_Minutes_Per_Day'] = capabilities['Total_Minutes_Per_Day'] * self.capabilities_coefficient

        return capabilities

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
        """
        blank_footprint = {
            'Intervention_Package_Code': [],
            'Item_Code': []
        }
        return blank_footprint

    def get_prob_seek_care(self, person_id, symptom_code=0):
        """
        This gives the probability that a person who had developed a particular symptom will seek care.
        Disease modules call this when a person has symptoms onset to determine if there will be a health interaction.
        """
        # It currently just returns 1.0, pending the work of Wingston on the health care seeking behaviour.
        return 1.0

    def check_if_can_do_hsi_event(self, hsi_event, current_capabilities):
        """
        This will determine if an HSI event can run given the constraints that of capabilities available.
        It accepts the argument of the HSI event itself and a dataframe describing the current /
        capabilities of the health system.

        :param hsi_event: The HSI event
        :param current_capabilities: The current capabailities of the health system.

        :return: It returns a tuple:
            * 1st element: True/False about whether the footprint can be accomodated
            * 2nd element: an updated version of current capabilities (unchanged in the case of the footprint /
                not being able to be accomodated)

        """

        # Check that the current_capabilities dataframe does not contain any errors
        assert not pd.isnull(current_capabilities['Minutes_Remaining_Today']).any()

        # Gather useful information
        df = self.sim.population.props
        mfl = self.parameters['Master_Facilities_List']
        fac_per_district = self.parameters['Facilities_For_Each_District']
        appt_types = self.parameters['Appt_Types_Table']['Appt_Type_Code'].values
        appt_times = self.parameters['Appt_Time_Table']
        consumables = self.parameters['Consumables']

        # Gather information about the HSI event
        the_person_id = hsi_event.target
        the_district = df.at[the_person_id, 'district_of_residence']
        the_treatment_footprint = hsi_event.APPT_FOOTPRINT

        # Get the health_facilities available to this person (based on their district), and which are accepted by the
        # hsi_event.ACCEPTED_FACILITY_LEVELS. Then sort by Facility_Level (to determine the order in which to try to
        # place the hsi_event: lowest level first).
        the_acceptable_health_facilities = fac_per_district.loc[(fac_per_district['District'] == the_district) &
                                                                fac_per_district['Facility_Level'].isin(
                                                                    hsi_event.ACCEPTED_FACILITY_LEVELS)]
        the_acceptable_health_facilities = the_acceptable_health_facilities.sort_values(['Facility_Level'])

        # get the capabilities of acceptable health facilities
        capabilities_of_the_health_facilities = current_capabilities.loc[
            current_capabilities['Facility_ID'].isin(the_acceptable_health_facilities['Facility_ID'])]

        # ------------
        # Test if capabilities of health system can meet this request
        # This requires there to be at least one facility that can fulfill the entire appt_footprint/
        #  (each type of appointment) AND has the consumables available (on this draw).
        # NB. Even with the ignore_appp_constraints flag in place, the full algorithm needs to be gone through
        # in order that we know at facility to look for consumables.

        # Loop through facilities to look for facilities
        # (Note that as the health_facilities dataframe was sorted on Facility_Level, this will start
        #  at the lowest levels and work upwards successively).

        can_do_appt_footprint = False
        can_do_cons_footprint = False

        for try_fac_id in the_acceptable_health_facilities.Facility_ID.values:
            # Look at each facility to see if it can run the appointment

            this_facility_level = mfl.loc[mfl['Facility_ID'] == try_fac_id, 'Facility_Level'].values[0]

            # Establish how much time is available at this facility
            time_available = capabilities_of_the_health_facilities.loc[
                capabilities_of_the_health_facilities['Facility_ID'] == try_fac_id, ['Officer_Type_Code',
                                                                                     'Minutes_Remaining_Today']]

            # Transform the treatment footprint into a demand for time for officers of each type, for this
            # facility level (it varies by facility level)
            time_requested = pd.DataFrame(columns=['Officer_Type_Code', 'Time_Taken'])
            for this_appt in appt_types:
                if the_treatment_footprint[this_appt] > 0:
                    time_req_for_this_appt = appt_times.loc[(appt_times['Appt_Type_Code'] == this_appt) &
                                                            (appt_times['Facility_Level'] == this_facility_level),
                                                            ['Officer_Type_Code',
                                                             'Time_Taken']].copy().reset_index(drop=True)
                    time_requested = pd.concat([time_requested, time_req_for_this_appt])

            if len(time_requested) > 0 or self.ignore_appt_constraints:
                # (If the data-frame of time-requested is empty, it means that the appointments is not possible
                # at that type of facility. So we check that time_requested is not empty before progressing.)
                # We also also allow the request to progress if we are ignoring appt_constraints
                # -------------------------

                # Collapse down the total_time_requested dataframe to give a sum of Time Taken by each
                # Officer_Type_Code
                time_requested = pd.DataFrame(
                    time_requested.groupby(['Officer_Type_Code'])['Time_Taken'].sum()).reset_index()
                time_requested = time_requested.drop(time_requested[time_requested['Time_Taken'] == 0].index)

                # Merge the Minutes_Available at this facility with the minutes required in the footprint
                comparison = time_requested.merge(time_available, on='Officer_Type_Code', how='left',
                                                  indicator=True)

                # Check if there are sufficient minutes available for each type of officer to satisfy
                # the appt_footprint
                if self.ignore_appt_constraints or (all(comparison['_merge'] == 'both') & all(
                                comparison['Minutes_Remaining_Today'] >= comparison['Time_Taken'])):

                    # the appt_footprint can be accommodated by officers at this facility
                    can_do_appt_footprint = True

                    # Now, check if the consumables are available at this facility
                    consumables_used = self.get_consumable_items(hsi_event)

                    #   Get the probabilities that each item is available
                    prob_item_available = consumables.loc[consumables['Item_Code'].isin(consumables_used['Item_Code']),
                                                          'Available_Facility_Level_' + str(this_facility_level)]

                    #   Detetermine if this facility level ever has these consumables
                    #   (the appt_footprint will be imposed if these items are ever available)
                    all_items_available_ever = bool((prob_item_available > 0).all())

                    #   Get random numbers and see if the the items will be available on this occasion:
                    all_items_available_now = bool(
                        (prob_item_available > self.rng.rand(len(prob_item_available))).all())

                    if self.ignore_cons_constraints or all_items_available_now:
                        can_do_cons_footprint = True

                    # -- Determine if the appt_footprint should be imposed:
                    if (can_do_appt_footprint and all_items_available_ever) \
                            or (can_do_appt_footprint and self.ignore_cons_constraints):
                        # (Impose the appt_footprint if it can be done and the consumables are ever available,
                        #  [represening that the appt happens but this is a temporary stock-out] or if ignoring
                        # consumables constraints)

                        # Impose the footprint for each one of the types being requested
                        for this_officer_type in time_requested['Officer_Type_Code'].values.tolist():
                            # print(hsi_event.TREATMENT_ID)
                            # print('time_requested[Officer_Type_Code].values: ',
                            #       time_requested['Officer_Type_Code'].values)
                            # # print('consumables: ', consumables_used)
                            # print('try_fac_id: ', try_fac_id)
                            # print('this_officer_type: ', this_officer_type)
                            # print('current_capabilities: ', current_capabilities.loc[
                            #     (current_capabilities['Facility_ID'] == try_fac_id) &
                            #     (current_capabilities['Officer_Type_Code'] == this_officer_type),
                            #     'Minutes_Remaining_Today'])

                            old_mins_remaining = \
                                current_capabilities.loc[
                                    (current_capabilities['Facility_ID'] == try_fac_id) &
                                    (current_capabilities['Officer_Type_Code'] == this_officer_type),
                                    'Minutes_Remaining_Today'].values[0]

                            time_to_take_away = \
                                time_requested.loc[
                                    time_requested['Officer_Type_Code'] == this_officer_type, 'Time_Taken'].values[0]

                            new_mins_remaining = \
                                old_mins_remaining - time_to_take_away

                            assert (new_mins_remaining >= 0) or self.ignore_appt_constraints

                            # update current_capabilities
                            current_capabilities.loc[
                                (current_capabilities['Facility_ID'] == try_fac_id) & (current_capabilities[
                                     'Officer_Type_Code'] == this_officer_type), 'Minutes_Remaining_Today'] = \
                                new_mins_remaining

                    break  # cease looking at other facility_types as the need has been met

        assert not pd.isnull(current_capabilities['Minutes_Remaining_Today']).any()

        rtn_tuple = ((can_do_appt_footprint and can_do_cons_footprint), current_capabilities)
        return rtn_tuple

    def log_consumables_used(self, hsi_event):
        """
        This will write to the log with a record of the consumables that were in the footprint of an HSI event

        :param hsi_event: The hsi event

        """
        # get the list of individual items used (from across the packages and individual items specified in the
        #  footprint)

        consumables_used = self.get_consumable_items(hsi_event)

        if len(consumables_used) > 0:  # if any consumables have been recorded

            # Do a groupby for the different consumables (there could be repeats of individual items which need /
            # to be summed)
            consumables_used = pd.DataFrame(consumables_used.groupby('Item_Code').sum())
            consumables_used = consumables_used.rename(columns={'Expected_Units_Per_Case': 'Units_By_Item_Code'})

            # Get the the total cost of the consumables
            consumables_used_with_cost = consumables_used.merge(self.parameters['Consumables_Cost_List'],
                                                                how='left',
                                                                on='Item_Code',
                                                                left_index=True
                                                                )
            total_cost = (
                consumables_used_with_cost['Units_By_Item_Code'] * consumables_used_with_cost['Unit_Cost']).sum()

            # Enter to the log
            log_consumables = consumables_used.to_dict()
            log_consumables['TREATMENT_ID'] = hsi_event.TREATMENT_ID
            log_consumables['Total_Cost'] = total_cost
            log_consumables['Person_ID'] = hsi_event.target

            logger.info('%s|Consumables|%s',
                        self.sim.date,
                        log_consumables)

    def get_consumable_items(self, hsi_event):
        """
        This will look at the CONS_FOOTPRINT of an HSI Event and return a dataframe with the individual items that
        are used, collecting these from across the packages and the individual items that are specified.
        """
        # Load the data on consumables
        consumables = self.parameters['Consumables']

        # Get the consumables in the hsi_event
        cons = hsi_event.CONS_FOOTPRINT

        # Create empty dataframe for storing the items used in the cons_footprint
        consumables_used = pd.DataFrame(columns=['Item_Code', 'Expected_Units_Per_Case'])

        # Get the individual items in each package:
        if not cons['Intervention_Package_Code'] == []:
            for p in cons['Intervention_Package_Code']:
                items = consumables.loc[
                    consumables['Intervention_Pkg_Code'] == p, ['Item_Code', 'Expected_Units_Per_Case']]
                consumables_used = consumables_used.append(items, ignore_index=True, sort=False).reset_index(drop=True)

        # Add in any additional items that have been specified seperately:
        if not cons['Item_Code'] == []:
            for i in cons['Item_Code']:
                items = pd.DataFrame(data={'Item_Code': i, 'Expected_Units_Per_Case': 1}, index=[0])
                consumables_used = consumables_used.append(items, ignore_index=True, sort=False).reset_index(
                    drop=True)

        return consumables_used

    def log_appts_used(self, hsi_event):
        """
        This will write to the log with a record of the appts that were in the footprint of an HSI event
        :param hsi_event: The hsi event
        """

        # Log the individual footprints being applied
        appts = hsi_event.APPT_FOOTPRINT
        log_appts = dict()
        log_appts['TREATMENT_ID'] = hsi_event.TREATMENT_ID
        log_appts['Number_By_Appt_Type_Code'] = {k: v for k, v in appts.items() if
                                                 v}  # remove the appt-types with zeros
        log_appts['Person_ID'] = hsi_event.target

        logger.info('%s|Appt|%s',
                    self.sim.date,
                    log_appts)

    def log_current_capabilities(self, current_capabilities):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type
        NB. To get this per Officer_Type_Code, it would be possible to simply log the entire current_capabilities df.
        :param current_capabilities: the current_capabilities of the health system.
        """

        # Log the current state of resource use

        # groupby Facility_ID: index of X is Facility_ID
        summary = current_capabilities.groupby('Facility_ID')[['Total_Minutes_Per_Day',
                                                               'Minutes_Remaining_Today']].sum()

        if summary['Total_Minutes_Per_Day'].sum() > 0:
            overall_useage = 1 - (summary['Minutes_Remaining_Today'].sum() / summary['Total_Minutes_Per_Day'].sum())
        else:
            overall_useage = 0.0  # in the case of there being no capabilities and nan arising

        summary['Fraction_Time_Used'] = 1 - (summary['Minutes_Remaining_Today'] / summary['Total_Minutes_Per_Day'])
        summary['Fraction_Time_Used'] = summary['Fraction_Time_Used'].replace([np.inf, -np.inf], 0.0)
        summary['Fraction_Time_Used'] = summary['Fraction_Time_Used'].fillna(0.0)

        # To change so that log includes the facility names:
        # Merge in Facilty_Name and drop everything else
        # X = X.merge(self.parameters['Master_Facilities_List'][['Facility_ID', 'Facility_Name']], how='left',
        #              left_index=True, right_on='Facility_ID')
        # X.index = X['Facility_Name']
        # X = X.drop(columns='Facility_Name')
        # X = X['Fraction_Time_Used']

        # Get just the series of Fraction_Time_Used in each facility, indexed by the Facility_ID
        summary = summary['Fraction_Time_Used']

        logger.debug('-------------------------------------------------')
        logger.debug('Current State of Health Facilities Appts:')
        print_table = summary.to_string().splitlines()
        for line in print_table:
            logger.debug(line)
        logger.debug('-------------------------------------------------')

        log_capacity = dict()
        log_capacity['Frac_Time_Used_Overall'] = overall_useage
        log_capacity['Frac_Time_Used_By_Facility_Name'] = summary.to_dict()

        logger.info('%s|Capacity|%s',
                    self.sim.date,
                    log_capacity)


class HealthSystemScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This is the HealthSystemScheduler. It is an event that occurs every day, inspects the calls on the healthsystem
    and commissions event to occur that are consistent with the healthsystem's capabilities for the following day, given
    assumptions about how this decision is made.
    The overall Prioritation algorithm is:
        * Look at events in order (the order is set by the heapq: see schedule_event
        * Ignore is the current data is before topen
        * Remove and do nothing if tclose has expired
        * Otherwise, test to see if there are sufficient health system capabilities to run the event

    If the event is to be run, then the following events occur:
        * The resources used are logged
        * The resources used are 'occupied'
        * The HSI event itself is run.
        * Other disease modules are alerted of the occurence of the HSI event

    At this point, we can have multiple types of assumption regarding how these capabilities are modelled.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):

        df = self.sim.population.props

        logger.debug('HealthSystemScheduler>> I will now determine what calls on resource will be met today: %s',
                     self.sim.date)

        logger.debug('----------------------------------------------------------------------')
        logger.debug("This is the entire HSI_EVENT_QUEUE heapq:")
        logger.debug(self.module.HSI_EVENT_QUEUE)
        logger.debug('----------------------------------------------------------------------')

        # Get the total Capabilities for Appts for today
        current_capabilities = self.module.get_capabilities()

        # Add column for Minutes Remaining Today (for live-tracking the use of those resources)
        current_capabilities.loc[:, 'Minutes_Remaining_Today'] = current_capabilities['Total_Minutes_Per_Day']

        # Create hold-over list (will be a heapq).
        # This will hold events that cannot occur today before they are added back to the heapq
        hold_over = list()

        while len(self.module.HSI_EVENT_QUEUE) > 0:

            # Pop the next event off the heapq
            next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)

            # Read the tuple and assemble into a dict 'next_event'
            # Pos 0: priority,
            # Pos 1: topen,
            # Pos 2: hsi_event_queue_counter,
            # Pos 3: tclose,
            # Pos 4: the hsi_event itself

            next_event = {
                'priority': next_event_tuple[0],
                'topen': next_event_tuple[1],
                'tclose': next_event_tuple[3],
                'object': next_event_tuple[4]}

            event = next_event['object']

            if self.sim.date > next_event['tclose']:
                # The event has expired, do nothing more
                pass

            elif self.sim.date < next_event['topen']:
                # The event is not yet due, add to the hold-over list
                hp.heappush(hold_over, next_event_tuple)

                if next_event['priority'] == 2:
                    # If the next event is not due and has low priorty, then stop looking through the heapq
                    # as all other events will also not be due.
                    break

            else:

                # Check if the event can run
                (can_do_appt_footprint, current_capabilities) = \
                    self.module.check_if_can_do_hsi_event(event, current_capabilities=current_capabilities)

                if can_do_appt_footprint:
                    # The event can be run:

                    if df.at[event.target, 'is_alive']:
                        # Run the event
                        event.run()

                        # Broadcast to other modules that the event is running:
                        self.module.broadcast_healthsystem_interaction(hsi_event=event)

                        # Write to the log
                        self.module.log_consumables_used(event)
                        self.module.log_appts_used(event)

                else:
                    # Add to hold-over list
                    hp.heappush(hold_over, next_event_tuple)

        # Add back to the HSI_EVENT_QUEUE heapq all those events which are eligible to run but which did not
        while len(hold_over) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(hold_over))

        # After completing routine for the day, log total usage of the facilities
        self.module.log_current_capabilities(current_capabilities)

        """
        Possible alternative is to empty the current HSI_EVENT_QUEUE, either processing the event or adding to the list
         hold_over. Then at the end of the loop, set self.module.HSI_EVENT_QUEUE = hp.heapify(hold_over)
        """
