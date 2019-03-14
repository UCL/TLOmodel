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
            service_availability = pd.DataFrame(data=[], columns=['Service', 'Available'])

        self.store_ServiceUse={
            'Skilled Birth Attendance': []
        }
        self.Service_Availabilty = service_availability

        self.RegisteredDiseaseModules = {}

        self.RegisteredInterventions = pd.DataFrame()

        self.HEALTH_SYSTEM_RESOURCES = dict()

        print('----------------------------------------------------------------------')
        print("Setting up the Health System With the Following Service Availabilty: ")
        print(service_availability)
        print('----------------------------------------------------------------------')

    PARAMETERS = {
        'Probability_Skilled_Birth_Attendance':
            Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'Master_Facility_List':
            Parameter(Types.DATA_FRAME, 'Imported Master Facility List workbook')
    }

    PROPERTIES = {
        'Distance_To_Nearest_HealthFacility':
            Property(Types.REAL, 'The distance for each person to their nearest clinic (of an type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Master_Facility_List'] = pd.read_csv(self.resourcefilepath+'ResourceFile_MasterFacilitiesList.csv')

        # Establish the MasterCapacitiesList
        # (Maybe this will become imported, or maybe it will stay being generated here)

        hf = self.parameters['Master_Facility_List']

        # Fill in some simple patterns for now
        # Water: False in outreach and Health post, True otherwise

        # Minutes of work time per month
        self.HEALTH_SYSTEM_RESOURCES['Nurse_Time'] = pd.DataFrame(index=[hf['Facility_ID']], columns=['Capacity', 'CurrentUse'])
        self.HEALTH_SYSTEM_RESOURCES['Nurse_Time']['CurrentUse'] = 0
        self.HEALTH_SYSTEM_RESOURCES['Nurse_Time']['Capacity'] = 1000

        # Minutes of work time per month
        self.HEALTH_SYSTEM_RESOURCES['Doctor_Time'] = pd.DataFrame(index=[hf['Facility_ID']], columns=['Capacity', 'CurrentUse']),
        self.HEALTH_SYSTEM_RESOURCES['Doctor_Time']['CurrentUse'] = 0
        self.HEALTH_SYSTEM_RESOURCES['Doctor_Time']['Capacity'] = 500

        # available: yes/no
        self.HEALTH_SYSTEM_RESOURCES['Electricity'] = pd.DataFrame(index=[hf['Facility_ID']], columns=['Capacity', 'CurrentUse']),
        self.HEALTH_SYSTEM_RESOURCES['Electricity']['CurrentUse'] = False
        self.HEALTH_SYSTEM_RESOURCES['Electricity']['Capacity'] = True

        # available: yes/no
        self.HEALTH_SYSTEM_RESOURCES['Water'] = pd.DataFrame(index=[hf['Facility_ID']], columns=['Capacity', 'CurrentUse'])
        self.HEALTH_SYSTEM_RESOURCES['Water']['CurrentUse'] = False
        self.HEALTH_SYSTEM_RESOURCES['Water']['Capacity'] = True

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
        df.at[child_id, 'Distance_To_Nearest_HealthFacility'] = df.at[mother_id,'Distance_To_Nearest_HealthFacility']

    def Register_Disease_Module(self, *new_disease_modules):
        # Register Disease Modules (in order that the health system can trigger things in each module)...
        for module in new_disease_modules:
            assert module.name not in self.RegisteredDiseaseModules, (
                'A module named {} has already been registered'.format(module.name))
            self.RegisteredDiseaseModules[module.name] = module

    def Register_Interventions(self, footprint_df):
        # Register the interventions that each disease module can offer and will ask for permission to use.
        print('Now registering a new intervention')
        self.RegisteredInterventions = self.RegisteredInterventions.append(footprint_df)

    def Query_Access_To_Service(self, person, service):
        print("Querying whether this person,", person, "will have access to this service:", service, ' ...')

        gets_service = False  # Default to fault (this is the variable that is returned to the disease module that does the request)

        # 1) Check if policy allows the offering of this treatment
        policy_allows = False  # default to False

        try:
            # Overwrite with the boolean value in the look-up table provided by the user, if a match can be found in the table for the service that is requested
            policy_allows = self.Service_Availabilty.loc[self.Service_Availabilty['Service'] == service, 'Available'].values[0]
        except:
            pass

        # 2) Check capacitiy
        enough_capacity = False  # Default to False unless it can be proved there is capacity

        # Look-up resources for the requested service:
        needed = self.RegisteredInterventions.loc[self.RegisteredInterventions['Name'] == service]

        # Look-up what health facilities this person has access to:
        village = self.sim.population.props.at[person,'village_of_residence']
        hf = self.parameters['Master_Facility_List']
        local_facilities = hf.loc[hf['Village']==village]
        local_facilities_idx = local_facilities['Facility_ID'].values

        # Sum capacity across the facilities to which persons in this village have access
        available_nurse_time = 0
        available_doctor_time = 0
        for lf_id in local_facilities_idx:
            available_nurse_time += self.HEALTH_SYSTEM_RESOURCES['Nurse_Time'].loc[lf_id, 'Capacity'].values[0] - self.HEALTH_SYSTEM_RESOURCES['Nurse_Time'].loc[lf_id,'CurrentUse'].values[0]
            available_doctor_time += self.HEALTH_SYSTEM_RESOURCES['Doctor_Time'].loc[lf_id, 'Capacity'].values[0] - self.HEALTH_SYSTEM_RESOURCES['Doctor_Time'].loc[lf_id, 'CurrentUse'].values[0]

        # See if there is enough capacity
        if (needed.Nurse_Time.values < available_nurse_time) and (needed.Doctor_Time.values < available_doctor_time):
            enough_capacity = True

            # ... and impose the "footprint"
            # TODO: We need to know how the footprint is defined in order to be able to impose it here.

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

        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):

            print('@@@@@@@@ Health Care Seeking Poll:::::')

            # 1) Work out the overall unified symptom code for all the differet diseases (and taking maxmium of them)

            unified_symptoms_code = pd.DataFrame()

            # Ask each module to update and report-out the symptoms it is currently causing on the unified symptomology scale:
            registered_disease_modules = self.sim.modules['HealthSystem'].RegisteredDiseaseModules
            for module in registered_disease_modules.values():
                out = module.query_symptoms_now()
                unified_symptoms_code = pd.concat([unified_symptoms_code, out], axis=1)  # each column of this dataframe gives the reports from each module of the unified symptom code
            pass

            # Look across the columns of the unified symptoms code reports to determine an overall symmtom level
            overall_symptom_code = unified_symptoms_code.max(axis=1)  # Maximum Value of reported Symptom is taken as overall level of symptoms

            # 2) For each individual, examine symptoms and other circumstances, and trigger a Health System Interaction if required
            df = population.props
            indicies_of_alive_person = df[df['is_alive'] == True].index

            for person_index in indicies_of_alive_person:

                # Collect up characteristics that will inform whether this person will seek care at thie moment...
                age = df.at[person_index,'age_years']
                healthlevel = overall_symptom_code.at[person_index]  # TODO: check that this is inheriting the correct index (pertainng to populaiton.props)
                education = df.at[person_index,'li_ed_lev']

                # Fill-in the regression equation about health-care seeking behaviour
                prob_seek_care = min(1.00, 0.02 + age*0.02+education*0.1 + healthlevel*0.2)

                # determine if there will be health-care contact and schedule if so
                if self.sim.rng.rand() < prob_seek_care:
                    event = InteractionWithHealthSystem_FirstAppt(self, person_index, 'HealthCareSeekingPoll')
                    self.sim.schedule_event(event, self.sim.date)


class OutreachEvent(Event, PopulationScopeEventMixin):
    # This event can be used to simulate the occurance of a one-off 'outreach event'
    # It does not automatically reschedule.
    # It commissions Interactions with the Health System for persons based location (and other variables)
    # in a different manner to HealthCareSeeking process

    def __init__(self, module, type, indicies):
        super().__init__(module)

        print("@@@@@ Outreach event being created!!!! @@@@@@")
        print("@@@ type: ", type, indicies)

        self.type = type
        self.indicies = indicies

    def apply(self, population):

        print("@@@@@ Outreach event running now @@@@@@")

        if self.type=='this_disease_only':
            # Schedule a first appointment for each person for this disease only
            for person_index in self.indicies:

                if self.sim.population.props.at[person_index,'is_alive']:
                    self.module.on_first_healthsystem_interaction(person_index,'OutreachEvent_ThisDiseaseOnly')

        else:
            # Schedule a first appointment for each person for all disease
            for person_index in self.indicies:
                if self.sim.population.props.at[person_index, 'is_alive']:
                    RegisteredDiseaseModules = self.sim.modules['HealthSystem'].RegisteredDiseaseModules
                    for module in RegisteredDiseaseModules.values():
                        module.on_first_healthsystem_interaction(person_index,'OutreachEvent_AllDiseases')

        # Log the occurance of the outreach event
        logger.info('%s|outreach_event|%s', self.sim.date,
                    {
                        'type': self.type
                    })


class InteractionWithHealthSystem_Emergency(Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            # This is an event call by a disease module and care
            # The on_healthsystem_interaction function is called only for the module that called for the EmergencyCare

            print('@@ EMERGENCY: I have been called by', self.module, 'to act on person', person_id)
            self.module.on_first_healthsystem_interaction(person_id,'Emergency')

            # Log the occurance of this interaction with the health system

            logger.info('%s|InteractionWithHealthSystem_Followups|%s', self.sim.date,
                        {
                            'person_id': person_id
                        })


# --------- TRIGGERING INTERACTIONS WITH THE HEALTH SYSTEM -----


class InteractionWithHealthSystem_FirstAppt(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id, cue_type):
        super().__init__(module, person_id=person_id)
        self.cue_type = cue_type

    def apply(self, person_id):
        # This is a FIRST meeting between the person and the health system
        # Symptoms (across all diseases) will be assessed and the disease-specific on-health-system function is called

        df = self.sim.population.props

        if df.at[person_id,'is_alive']:
            print("@@@@ We are now having an health appointment with individual", person_id)

            # For each disease module, trigger the on_healthsystem() event
            registered_disease_modules = self.sim.modules['HealthSystem'].RegisteredDiseaseModules
            for module in registered_disease_modules.values():
                module.on_first_healthsystem_interaction(person_id, self.cue_type)

            # Log the occurance of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem_FirstAppt|%s', self.sim.date,
                    {
                        'person_id': person_id,
                        'cue_type': self.cue_type
                    })


class InteractionWithHealthSystem_Followups(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        if df.at[person_id,'is_alive']:
            # Use this interaction type for persons once they are in care for monitoring and follow-up for a specific disease
            print("in a follow-up appoinntment")

            # Log the occurance of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem_Followups|%s', self.sim.date,
                        {
                            'person_id': person_id
                        })
