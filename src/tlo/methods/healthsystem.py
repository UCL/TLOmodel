"""
This module stands in for the "Health System" in the current implementation of the module
It is used to control access to interventions
It will be replaced by the Health Care Seeking Behaviour Module and the
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin


class HealthSystem(Module):
    """
    Requests for access to particular services are lodged here.
    """

    def __init__(self, name=None, resourcefilepath=None, Service_Availability=pd.DataFrame(data=[],columns=['Service','Available'])):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.store_ServiceUse={
            'Skilled Birth Attendance': []
        }
        self.Service_Availabilty=Service_Availability

        self.RegisteredDiseaseModules = {}

        self.RegisteredInterventions=pd.DataFrame()


        print('----------------------------------------------------------------------')
        print("Setting up the Health System With the Following Service Availabilty: ")
        print(Service_Availability)
        print('----------------------------------------------------------------------')


    PARAMETERS = {'Probability_Skilled_Birth_Attendance': Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
                  'Master_Facility_List':Parameter(Types.DATA_FRAME,'Imported Master Facility List workbook')}


    def read_parameters(self, data_folder):

        self.parameters['Probability_Skilled_Birth_Attendance']=pd.DataFrame(
            data=[
                (0, 'Facility', 0.00), (0, 'Home', 0.00), (0, 'Roadside', 0.00),         # No availability
                (1,'Facility',0.95),(1,'Home',0.5),(1,'Roadside',0.05),                 # Default levels of availability
                (2, 'Facility', 1.00), (2, 'Home', 1.00), (2, 'Roadside', 1.00)          # Maximal levels of availability
            ],
            columns=['Availability','Location_Of_Births','prob'])

        self.parameters['Master_Facility_List']=pd.read_csv(self.resourcefilepath+'ResourceFile_MasterFacilitiesList.csv')


    def initialise_population(self, population):

        pass

    def initialise_simulation(self, sim):

        sim.schedule_event(HealthCareSeekingPoll(self), sim.date) #Launch the healthcare seeking poll

        # Check that people can find their health facilities:
        pop=self.sim.population.props
        hf=self.parameters['Master_Facility_List']

        for person_id in pop.index:
            my_village=pop.at[person_id, 'village_of_residence']
            my_health_facilities=hf.loc[hf['Village']==my_village]

        # Establish the MasterCapacitiesList
        # (Maybe this will become imported, or maybe it will stay being generated here)
        self.HEALTH_SYSTEM_RESOURCES = {'Nurse_Time': pd.DataFrame(index=[hf['Facility_ID']],columns=['Capacity','CurrentUse']), # Minutes of work time per month
                                    'Doctor_Time':pd.DataFrame(index=[hf['Facility_ID']],columns=['Capacity','CurrentUse']),# Minutes of work time per month
                                    'Electricity':pd.DataFrame(index=[hf['Facility_ID']],columns=['Capacity','CurrentUse']),# available: yes/no
                                    'Water':pd.DataFrame(index=[hf['Facility_ID']],columns=['Capacity','CurrentUse'])}       # available: yes/no

        # Fill in some simple patterns for now

        #Water: False in outreach and Health post, True otherwise
        self.HEALTH_SYSTEM_RESOURCES['Nurse_Time']['CurrentUse']=0
        self.HEALTH_SYSTEM_RESOURCES['Nurse_Time']['Capacity'] = 1000

        self.HEALTH_SYSTEM_RESOURCES['Doctor_Time']['CurrentUse']=0
        self.HEALTH_SYSTEM_RESOURCES['Doctor_Time']['Capacity'] = 500

        self.HEALTH_SYSTEM_RESOURCES['Electricity']['CurrentUse']=False
        self.HEALTH_SYSTEM_RESOURCES['Electricity']['Capacity'] =True

        self.HEALTH_SYSTEM_RESOURCES['Water']['CurrentUse']=False
        self.HEALTH_SYSTEM_RESOURCES['Water']['Capacity'] =True



    def on_birth(self, mother, child):

        pass


    def Register_Disease_Module(self, *NewDiseaseModule):

        # Register Disease Modules (in order that the health system can trigger things in each module)...

        for module in NewDiseaseModule:
            assert module.name not in self.RegisteredDiseaseModules, (
                'A module named {} has already been registered'.format(module.name))
            self.RegisteredDiseaseModules[module.name] = module


    def Register_Interventions(self,footprint_df):

        # Register the interventions that each disease module can offer and will ask for permission to use.

        print('Now registering a new intervention')
        self.RegisteredInterventions=self.RegisteredInterventions.append(footprint_df)


    def Query_Access_To_Service(self,person,service):


        print("Querying whether this person,", person, "will have access to this service:", service, ' ...')

        GetsService=False # Default to fault (this is the variable that is returned to the disease module that does the request)


        # Check if policy allows the offering of this treatment

        # try:
        #     availability=int( self.Service_Availabilty.Available[self.Service_Availabilty['Service']==service] )
        #
        #
        #     if service == "Skilled Birth Attendance":
        #
        #         location_of_birth=person.Location_Of_Births
        #         df=self.parameters['Probability_Skilled_Birth_Attendance']
        #
        #         prob=float(df[( df['Availability'] ==availability ) & ( df['Location_Of_Births']==location_of_birth )].prob)
        #
        #
        #         if self.sim.rng.rand() < prob:
        #             GetsService=True
        #
        #             if self.sim.verboseoutput:
        #                 print('         .... They will! ')
        # except:
        #     print("A request for a service was met with an error")





        # Check capacitiy
        EnoughCapacity=False # Default to False unless it can be proved there is capacity

        # Look-up resources for the requested service:
        needed=self.RegisteredInterventions.loc[self.RegisteredInterventions['Name']==service]

        # Look-up what health facilities this person has access to:
        village=self.sim.population.props.at[person,'village_of_residence']
        hf = self.parameters['Master_Facility_List']
        local_facilities=hf.loc[hf['Village']==village]
        local_facilities_idx = local_facilities['Facility_ID'].values

        # Sum capacity across the facilities to which persons in this village have access
        available_Nurse_Time=0
        available_Doctor_Time=0
        for lf_id in local_facilities_idx:
            available_Nurse_Time+=self.HEALTH_SYSTEM_RESOURCES['Nurse_Time'].loc[lf_id, 'Capacity'] - self.HEALTH_SYSTEM_RESOURCES['Nurse_Time'].loc[lf_id,'CurrentUse']
            available_Doctor_Time+= self.HEALTH_SYSTEM_RESOURCES['Doctor_Time'].loc[lf_id, 'Capacity'] - self.HEALTH_SYSTEM_RESOURCES['Doctor_Time'].loc[lf_id, 'CurrentUse']

        # See if there is enough capacity
        if (needed.Nurse_Time.values < available_Nurse_Time) and (needed.Doctor_Time.values < available_Doctor_Time):
            EnoughCapacity=True


        if PolicyAllows and EnoughCapacity:
            GetsService=True

        return GetsService








# --------- FORMS OF HEALTH-CARE SEEKING -----

class HealthCareSeekingPoll(RegularEvent, PopulationScopeEventMixin):
        # This event is occuring regularly at 3-monthly intervals
        # It asseess who has symptoms that are sufficient to bring them into care

        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):

            print('@@@@@@@@ Health Care Seeking Poll:::::')

            # 1) Work out the overall unified symptom code for all the differet diseases (and taking maxmium of them)

            UnifiedSymptomsCode=pd.DataFrame()

            # Ask each module to update and report-out the symptoms it is currently causing on the unified symptomology scale:
            RegisteredDiseaseModules=self.sim.modules['HealthSystem'].RegisteredDiseaseModules
            for module in RegisteredDiseaseModules.values():
                out=module.query_symptoms_now()
                UnifiedSymptomsCode=pd.concat([UnifiedSymptomsCode, out], axis=1) # each column of this dataframe gives the reports from each module of the unified symptom code
            pass

            # Look across the columns of the unified symptoms code reports to determine an overall symmtom level
            OverallSymptomCode = UnifiedSymptomsCode.max(axis=1) # Maximum Value of reported Symptom is taken as overall level of symptoms


            # 2) For each individual, examine symptoms and other circumstances, and trigger a Health System Interaction if required
            df=population.props
            indicies_of_alive_person = df[df['is_alive']==True].index

            for person_index in indicies_of_alive_person:

                # Collect up characteristics that will inform whether this person will seek care at thie moment...
                age=df.at[person_index,'age_years']
                healthlevel=OverallSymptomCode.at[person_index]  # TODO: check that this is inheriting the correct index (pertainng to populaiton.props)
                education=df.at[person_index,'li_ed_lev']

                # Fill-in the regression equation about health-care seeking behaviour
                prob_seek_care = min(1.00, 0.02 + age*0.02+education*0.1 + healthlevel*0.2)

                # determine if there will be health-care contact and schedule if so
                if (self.sim.rng.rand() < prob_seek_care) :
                    event=InteractionWithHealthSystem_FirstAppt(self,person_index)
                    self.sim.schedule_event(event, self.sim.date)


class OutreachEvent(Event, PopulationScopeEventMixin):
    # This event can be used to simulate the occurance of a one-off 'outreach event'
    # It does not automatically reschedule.
    # It commissions Interactions with the Health System for persons based location (and other variables)
    # in a different manner to HealthCareSeeking process

    def __init__(self, module,type, indicies):
        super().__init__(module)

        print("@@@@@ Outreach event being created!!!! @@@@@@")
        print("@@@ type: ", type, indicies)

        self.type=type
        self.indicies=indicies

    def apply(self, population):

        print("@@@@@ Outreach event running now @@@@@@")

        if self.type=='this_disease_only':
            # Schedule a first appointment for each person for this disease only
            for person_index in self.indicies:
                self.module.on_first_healthsystem_interaction(person_index)

        else:
            # Schedule a first appointment for each person for all disease
            for person_index in self.indicies:
                RegisteredDiseaseModules = self.sim.modules['HealthSystem'].RegisteredDiseaseModules
                for module in RegisteredDiseaseModules.values():
                    module.on_first_healthsystem_interaction(person_index)



# --------- TRIGGERING INTERACTIONS WITH THE HEALTH SYSTEM -----

class InteractionWithHealthSystem_Emergency(Event,IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        # This is an event call by a disease module and care
        # The on_healthsystem_interaction function is called only for the module that called for the EmergencyCare

        print('@@ EMERGENCY: I have been called by', self.module, 'to act on person', person_id)
        self.module.on_first_healthsystem_interaction(person_id)

        pass



class InteractionWithHealthSystem_FirstAppt(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        # This is a FIRST meeting between the person and the health system
        # Symptoms (across all diseases) will be assessed and the disease-specific on-health-system function is called

        df = self.sim.population.props
        print("@@@@ We are now having an health appointment with individual", person_id)

        # For each disease module, trigger the on_healthsystem() event
        RegisteredDiseaseModules = self.sim.modules['HealthSystem'].RegisteredDiseaseModules
        for module in RegisteredDiseaseModules.values():
            module.on_first_healthsystem_interaction(person_id)



class InteractionWithHealthSystem_Followups(Event,IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        # Use this interaction type for persons once they are in care for monitoring and follow-up for a specific disease
        print("in a follow-up appoinntment")




        pass
