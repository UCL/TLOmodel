"""
This module stands in for the "Health System" in the current implementation of the module
It is used to control access to interventions
It will be replaced by the Health Care Seeking Behaviour Module and the
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class HealthSystem(Module):
    """
    Requests for access to particular services are lodged here.
    """

    def __init__(self, name=None, Service_Availability=pd.DataFrame(data=[],columns=['Service','Available'])):
        super().__init__(name)
        self.store_ServiceUse={
            'Skilled Birth Attendance': []
        }
        self.Service_Availabilty=Service_Availability

        print('----------------------------------------------------------------------')
        print("Setting up the Health System With the Following Service Availabilty: ")
        print(Service_Availability)
        print('----------------------------------------------------------------------')


    PARAMETERS = {'Probability_Skilled_Birth_Attendance': Parameter(Types.DATA_FRAME, 'Interpolated population structure')}


    def read_parameters(self, data_folder):

        self.parameters['Probability_Skilled_Birth_Attendance']=pd.DataFrame(
            data=[
                (0, 'Facility', 0.00), (0, 'Home', 0.00), (0, 'Roadside', 0.00),         # No availability
                (1,'Facility',0.95),(1,'Home',0.5),(1,'Roadside',0.05),                 # Default levels of availability
                (2, 'Facility', 1.00), (2, 'Home', 1.00), (2, 'Roadside', 1.00)          # Maximal levels of availability
            ],
            columns=['Availability','Location_Of_Births','prob'])



    def initialise_population(self, population):

        pass

    def initialise_simulation(self, sim):

        pass

    def on_birth(self, mother, child):

        pass



    def Query_Access_To_Service(self,person,service):

        if self.sim.verboseoutput:
            print("Querying whether this person,", person, "will have access to this service:", service, ' ...')

        GetsService=False


        try:
            availability=int( self.Service_Availabilty.Available[self.Service_Availabilty['Service']==service] )


            if service == "Skilled Birth Attendance":

                location_of_birth=person.Location_Of_Births
                df=self.parameters['Probability_Skilled_Birth_Attendance']

                prob=float(df[( df['Availability'] ==availability ) & ( df['Location_Of_Births']==location_of_birth )].prob)


                if self.sim.rng.rand() < prob:
                    GetsService=True

                    if self.sim.verboseoutput:
                        print('         .... They will! ')
        except:
            print("A request for a service was met with an error")

        return GetsService



    class HealthCareSeekingPoll(RegularEvent, PopulationScopeEventMixin):
        # This event is occuring regularly at 3-monthly intervals
        # It asseess who has symptoms that are sufficient to bring them into care

        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):
            pass


