"""
This module stands in for the "Health System" in the current implementation of the module
It is used to control access to interventions
It will be replaced by the Health Care Seeking Behaviour Module and the
"""
import logging

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HealthSystem(Module):
    """
    Requests for access to particular services are lodged here.
    """

    def __init__(self, name=None,
                 resourcefilepath=None,
                 service_availability=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        if service_availability is None:
            service_availability = pd.DataFrame(data=[], columns=['Service', 'Available'], dtype=['object', bool])

        self.store_ServiceUse = {
            'Skilled Birth Attendance': []
        }
        self.service_availability = service_availability

        self.registered_disease_modules = {}

        self.registered_interventions = pd.DataFrame()

        self.health_system_resources = None

        logger.info('----------------------------------------------------------------------')
        logger.info("Setting up the Health System With the Following Service Availabilty:")
        print_table = service_availability.to_string().splitlines()
        for line in print_table:
            logger.info(line)
        logger.info('----------------------------------------------------------------------')

    PARAMETERS = {
        'Probability_Skilled_Birth_Attendance':
            Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'Master_Facility_List':
            Parameter(Types.DATA_FRAME, 'Imported Master Facility List workbook')
    }

    PROPERTIES = {
        'Distance_To_Nearest_HealthFacility':
            Property(Types.REAL,
                     'The distance for each person to their nearest clinic (of any type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Master_Facility_List'] = pd.read_csv(
            self.resourcefilepath+'ResourceFile_MasterFacilitiesList.csv')

        # Establish the MasterCapacitiesList
        # (Maybe this will become imported, or maybe it will stay being generated here)

        hf = self.parameters['Master_Facility_List']

        # Fill in some simple patterns for now

        # Minutes of work time per month
        nurse_time = pd.DataFrame(index=[hf.Facility_ID], columns=['Capacity', 'CurrentUse'])
        nurse_time.append({'CurrentUse': 0, 'Capacity': 1000}, ignore_index=True)

        # Minutes of work time per month
        doctor_time = pd.DataFrame(index=[hf.Facility_ID], columns=['Capacity', 'CurrentUse'])
        doctor_time.append({'CurrentUse': 0, 'Capacity': 500}, ignore_index=True)

        # Water: False in outreach and Health post, True otherwise
        # available: yes/no
        electricity = pd.DataFrame(index=[hf.Facility_ID], columns=['Capacity', 'CurrentUse'])
        electricity.append({'CurrentUse': False, 'Capacity': True}, ignore_index=True)

        # available: yes/no
        water = pd.DataFrame(index=[hf.Facility_ID], columns=['Capacity', 'CurrentUse'])
        water.append({'CurrentUse': False, 'Capacity': True}, ignore_index=True)

        self.health_system_resources = dict()
        self.health_system_resources['Nurse_Time'] = nurse_time
        self.health_system_resources['Doctor_Time'] = doctor_time
        self.health_system_resources['Electricity'] = electricity
        self.health_system_resources['Water'] = water

    def initialise_population(self, population):
        df = population.props

        # Assign Distance_To_Nearest_HealthFacility'
        # For now, let this be a random number, but in future it will be properly informed
        # Note that this characteritic is inherited from mother to child.
        df['Distance_To_Nearest_HealthFacility'] = self.sim.rng.randn(len(df))

    def initialise_simulation(self, sim):
        # Launch the healthcare seeking poll
        sim.schedule_event(HealthCareSeekingPoll(self), sim.date)

        # Check that people can find their health facilities:
        pop = self.sim.population.props
        hf = self.parameters['Master_Facility_List']

        for person_id in pop.index:
            my_village = pop.at[person_id, 'village_of_residence']
            my_health_facilities = hf.loc[hf['Village'] == my_village]

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'Distance_To_Nearest_HealthFacility'] = \
            df.at[mother_id, 'Distance_To_Nearest_HealthFacility']

    def register_disease_module(self, *new_disease_modules):
        # Register Disease Modules (in order that the health system can trigger things
        # in each module)...
        for module in new_disease_modules:
            assert module.name not in self.registered_disease_modules, (
                'A module named {} has already been registered'.format(module.name))
            self.registered_disease_modules[module.name] = module

    def register_interventions(self, footprint_df):
        # Register the interventions that each disease module can offer and will ask for
        # permission to use.
        logger.info('Registering intervention %s', footprint_df.at[0, 'Name'])
        self.registered_interventions = self.registered_interventions.append(footprint_df)

    def query_access_to_service(self, person, service):
        logger.info('Query person %d has access to service %s', person, service)

        sa = self.service_availability
        hsr = self.health_system_resources

        # Default to fault (this is the variable that is returned to
        # the disease module that does the request)
        gets_service = False

        # 1) Check if policy allows the offering of this treatment
        policy_allows = False  # default to False

        if service in sa.Service.values:
            policy_allows = sa.loc[sa['Service'] == service, 'Available'].values[0]

        # 2) Check capacitiy
        enough_capacity = False  # Default to False unless it can be proved there is capacity

        # Look-up resources for the requested service:
        needed = self.registered_interventions.loc[self.registered_interventions['Name'] == service]

        # Look-up what health facilities this person has access to:
        village = self.sim.population.props.at[person, 'village_of_residence']
        hf = self.parameters['Master_Facility_List']
        local_facilities = hf.loc[hf['Village'] == village]
        local_facilities_idx = local_facilities['Facility_ID'].values

        # Sum capacity across the facilities to which persons in this village have access
        available_nurse_time = 0
        available_doctor_time = 0
        for lf_id in local_facilities_idx:
            available_nurse_time += (hsr['Nurse_Time'].loc[lf_id, 'Capacity'].values[0] -
                                     hsr['Nurse_Time'].loc[lf_id, 'CurrentUse'].values[0])
            available_doctor_time += (hsr['Doctor_Time'].loc[lf_id, 'Capacity'].values[0] -
                                      hsr['Doctor_Time'].loc[lf_id, 'CurrentUse'].values[0])

        # See if there is enough capacity
        if (needed.Nurse_Time.values < available_nurse_time) and (needed.Doctor_Time.values <
                                                                  available_doctor_time):
            enough_capacity = True
            # ... and impose the "footprint"
            # TODO: need to know how footprint is defined to be able to impose it here.

        if policy_allows and enough_capacity:
            gets_service = True

        # Log the occurance of this request for services
        logger.info('%s|Query_Access_To_Service|%s', self.sim.date,
                    {
                        'person_id': person,
                        'service': service,
                        'policy_allows': policy_allows,
                        'enough_capacity': enough_capacity,
                        'gets_service': gets_service
                    })

        return gets_service


# --------- FORMS OF HEALTH-CARE SEEKING -----


class HealthCareSeekingPoll(RegularEvent, PopulationScopeEventMixin):
        # This event is occuring regularly at 3-monthly intervals
        # It asseess who has symptoms that are sufficient to bring them into care

        def __init__(self, module: HealthSystem):
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):

            logger.debug('Health Care Seeking Poll')

            # 1) Work out the overall unified symptom code for all the differet diseases
            # (and taking maxmium of them)

            unified_symptoms_code = pd.DataFrame()

            # Ask each module to update and report-out the symptoms it is currently causing on the
            # unified symptomology scale:
            registered_disease_modules = self.module.registered_disease_modules
            for module in registered_disease_modules.values():
                out = module.query_symptoms_now()
                # each column of this dataframe gives the reports from each module of the
                # unified symptom code
                unified_symptoms_code = pd.concat([unified_symptoms_code, out], axis=1)
            pass

            # Look across the columns of the unified symptoms code reports to determine an overall
            # symptom level
            # Maximum Value of reported Symptom is taken as overall level of symptoms
            overall_symptom_code = unified_symptoms_code.max(axis=1)

            # 2) For each individual, examine symptoms and other circumstances,
            # and trigger a Health System Interaction if required
            df = population.props
            indicies_of_alive_person = df.index[df.is_alive]

            for person_index in indicies_of_alive_person:

                # Collect up characteristics that will inform whether this person will seek care
                # at thie moment...
                age = df.at[person_index, 'age_years']

                # TODO: check this inherits the correct index (pertainng to populaiton.props)
                healthlevel = overall_symptom_code.at[person_index]
                education = df.at[person_index, 'li_ed_lev']

                # Fill-in the regression equation about health-care seeking behaviour
                prob_seek_care = min(1.00, 0.02 + age*0.02+education*0.1 + healthlevel*0.2)

                # determine if there will be health-care contact and schedule if so
                if self.sim.rng.rand() < prob_seek_care:
                    event = FirstApptHealthSystemInteraction(self.module, person_index,
                                                             'HealthCareSeekingPoll')
                    self.sim.schedule_event(event, self.sim.date)


class OutreachEvent(Event, PopulationScopeEventMixin):
    # This event can be used to simulate the occurance of a one-off 'outreach event'
    # It does not automatically reschedule.
    # It commissions Interactions with the Health System for persons based location
    # (and other variables) in a different manner to HealthCareSeeking process

    def __init__(self, module, outreach_type, indicies):
        super().__init__(module)

        logger.debug('Outreach event being created. Type: %s, %s', outreach_type, indicies)

        self.outreach_type = outreach_type
        self.indicies = indicies

    def apply(self, population):

        logger.debug('Outreach event running now')

        if self.outreach_type == 'this_disease_only':
            # Schedule a first appointment for each person for this disease only
            for person_index in self.indicies:

                if self.sim.population.props.at[person_index, 'is_alive']:
                    self.module.on_first_healthsystem_interaction(person_index,
                                                                  'OutreachEvent_ThisDiseaseOnly')

        else:
            # Schedule a first appointment for each person for all disease
            for person_index in self.indicies:
                if self.sim.population.props.at[person_index, 'is_alive']:
                    registered_disease_modules = (
                        self.sim.modules['HealthSystem'].registered_disease_modules
                    )
                    for module in registered_disease_modules.values():
                        module.on_first_healthsystem_interaction(person_index,
                                                                 'OutreachEvent_AllDiseases')

        # Log the occurance of the outreach event
        logger.info('%s|outreach_event|%s', self.sim.date,
                    {
                        'type': self.outreach_type
                    })


class EmergencyHealthSystemInteraction(Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            # This is an event call by a disease module and care
            # The on_healthsystem_interaction function is called only for the module that called
            # for the EmergencyCare

            logger.debug('EMERGENCY: I have been called by %s for person %d',
                         self.module.name,
                         person_id)
            self.module.on_first_healthsystem_interaction(person_id, 'Emergency')

            # Log the occurance of this interaction with the health system

            logger.info('%s|InteractionWithHealthSystem_Followups|%s', self.sim.date,
                        {
                            'person_id': person_id
                        })


# --------- TRIGGERING INTERACTIONS WITH THE HEALTH SYSTEM -----


class FirstApptHealthSystemInteraction(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id, cue_type):
        super().__init__(module, person_id=person_id)
        self.cue_type = cue_type

    def apply(self, person_id):
        # This is a FIRST meeting between the person and the health system
        # Symptoms (across all diseases) will be assessed and the disease-specific
        # on-health-system function is called

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            logger.debug("Health appointment with individual %d", person_id)

            # For each disease module, trigger the on_healthsystem() event
            registered_disease_modules = self.module.registered_disease_modules
            for module in registered_disease_modules.values():
                module.on_first_healthsystem_interaction(person_id, self.cue_type)

            # Log the occurance of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem_FirstAppt|%s',
                        self.sim.date,
                        {
                            'person_id': person_id,
                            'cue_type': self.cue_type
                        })


class FollowupHealthSystemInteraction(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            # Use this interaction type for persons once they are in care for monitoring and
            # follow-up for a specific disease
            logger.debug("in a follow-up appoinntment")

            # Log the occurance of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem_Followups|%s', self.sim.date,
                        {
                            'person_id': person_id
                        })
