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
    Requests for access to particular services are handled by Disease/Intervention Modules by this Module
    """

    def __init__(self, name=None,
                 resourcefilepath=None,
                 service_availability=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        if service_availability is None:
            service_availability = pd.DataFrame(data=[], columns=['Service', 'Available'], dtype=['object', bool])

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
        'Master_Facility_List':
            Parameter(Types.DATA_FRAME, 'Imported Master Facility List workbook: one row per each facility'),
        'Village_To_Facility_Mapping':
            Parameter(Types.DATA_FRAME, 'Imported long-list of links between villages and health facilities: ' \
                                            'one row per each link between a village and a facility')
    }

    PROPERTIES = {
        'Distance_To_Nearest_HealthFacility':
            Property(Types.REAL,
                     'The distance for each person to their nearest clinic (of any type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Master_Facility_List'] = pd.read_csv(
            self.resourcefilepath+'ResourceFile_MasterFacilitiesList.csv')

        self.parameters['Village_To_Facility_Mapping']=pd.read_csv(
            self.resourcefilepath+'ResourceFile_Village_To_Facility_Mapping.csv'
        )


        # Establish the MasterCapacitiesList
        # (This is where the data on all health capabilities will be stored. For now, nothing happens)


    def initialise_population(self, population):
        df = population.props

        # Assign Distance_To_Nearest_HealthFacility'
        # For now, let this be a random number, but in future it will be properly informed based on population density distribitions.
        # Note that this characteritic is inherited from mother to child.
        df['Distance_To_Nearest_HealthFacility'] = self.sim.rng.randn(len(df))

    def initialise_simulation(self, sim):
        # Launch the healthcare seeking poll
        sim.schedule_event(HealthCareSeekingPoll(self), sim.date)

        # Check that each person is atttached to a village and a set of attached health facilities
        pop = self.sim.population.props
        mapping = self.parameters['Village_To_Facility_Mapping']
        for person_id in pop.index[pop.is_alive]:
            my_village = pop.at[person_id, 'village_of_residence']
            my_health_facilities = mapping.loc[mapping['Village'] == my_village]
            assert len(my_health_facilities)>0


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

    def register_interventions(self, footprint_df):
        # Register the interventions that can be requested
        logger.info('Registering intervention %s', footprint_df.at[0, 'Name'])
        self.registered_interventions = self.registered_interventions.append(footprint_df)

    def query_access_to_service(self, person_id, service):
        logger.info('Query whether person %d has access to service %s', person_id, service)

        # Health system allows all requests for services
        # (This is where the constraint will be imposed and the footprint recorded)

        enough_capacity=True        # No constraint imposed
        policy_allows=True          # No constraint imposed
        gets_service = True         # No constraint imposed

        # Log the occurance of this request for services
        logger.info('%s|Query_Access_To_Service|%s', self.sim.date,
                    {
                        'person_id': person_id,
                        'service': service,
                        'policy_allows': policy_allows,
                        'enough_capacity': enough_capacity,
                        'gets_service': gets_service
                    })

        return gets_service


# --------- FORMS OF HEALTH-CARE SEEKING -----

class HealthCareSeekingPollEvent(RegularEvent, PopulationScopeEventMixin):
        """
        This event is occurring regularly at 3-monthly intervals.
        It determines who has symptoms that are sufficient to bring them into care.
        It uses the "general health care seeking" equation to do this.
        """


        def __init__(self, module: HealthSystem):
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):

            logger.debug('Health Care Seeking Poll is running')


            # ----------
            # 1) Work out the overall unified symptom code

            unified_symptoms_code = pd.DataFrame()

            # Ask each module to update and report-out the symptoms it is currently causing on the
            # unified symptomology scale:
            registered_disease_modules = self.module.registered_disease_modules
            for module in registered_disease_modules.values():
                out = module.query_symptoms_now()

                # check that the data received is in correct format
                assert len(out)==self.sim.population.props.is_alive.sum()
                assert all(out.astype(int)>=0)
                assert all(out.astype(int)<5)

                unified_symptoms_code = pd.concat([unified_symptoms_code, out], axis=1)


            # Look across the columns of the unified symptoms code reports to determine an overall
            # symptom level.
            # The Maximum Value of reported Symptom is taken as overall level of symptoms
            overall_symptom_code = unified_symptoms_code.max(axis=1)

            # ----------
            # 2) For each individual, examine symptoms and other circumstances,
            # and trigger a Health System Interaction if required
            df = population.props
            indicies_of_alive_person = df.index[df.is_alive]

            for person_index in indicies_of_alive_person:

                # Collect up characteristics that will inform whether this person will seek care
                # at this moment...
                age = df.at[person_index, 'age_years']
                healthlevel = overall_symptom_code.at[person_index]
                education = df.at[person_index, 'li_ed_lev']

                # Fill-in the regression equation about health-care seeking behaviour (Dummy values used here)
                prob_seek_care = min(1.00, 0.02 + age*0.02+education*0.1 + healthlevel*0.2)

                # determine if there will be health-care contact and schedule FirstAppt if so
                if self.sim.rng.rand() < prob_seek_care:
                    event = HealthSystemInteractionEvent(self.module, person_index,
                                                             'HealthCareSeekingPoll')
                    self.sim.schedule_event(event, self.sim.date)

            # ----------

class OutreachEvent(Event, PopulationScopeEventMixin):
    """
    This event can be used to simulate the occurance of an 'outreach event'
    It does not automatically reschedule.
    It is limited to the person_id that are supplied to it.
    It commissions FirstAppt's with the Health System for those persons.
    The type argument determines whether only one particular disease is notified or all diseases.
    """

    def __init__(self, module, outreach_type, person_indicies):
        super().__init__(module)

        logger.debug('Outreach event being created. Type: %s, %s', outreach_type, person_indicies)

        self.outreach_type = outreach_type
        self.indicies = person_indicies

    def apply(self, population):

        logger.debug('Outreach event running now')

        if self.outreach_type == 'this_disease_only':

            # Schedule a first appointment for each person for this disease only
            for person_index in self.person_indicies:

                if self.sim.population.props.at[person_index, 'is_alive']:

                    event = HealthSystemInteractionEvent(self.module, person_index,
                                                             'OutreachEvent_ThisDiseaseOnly'+self.outreach_type)
                    self.sim.schedule_event(event, self.sim.date)


        # Log the occurrence of the outreach event
        logger.info('%s|outreach_event|%s', self.sim.date,
                    {
                        'type': self.outreach_type
                    })






class HealthSystemInteractionEvent(Event, IndividualScopeEventMixin):
    """
    This is a generic interaction between the person and the health system.
    All actual interactions between a person and the health system happen here.
    It can be called by: HealthCareSeekingPoll, OutreachEvent or a DiseaseModule.
    It broadcasts details of the interaction to all disease modules with
    information about the type of interaction. It logs the interaction and calls
    for resources.
    """

    def __init__(self, module, person_id, cue_type):
        super().__init__(module, person_id=person_id)
        self.cue_type = cue_type

    def apply(self, person_id):

        df = self.sim.population.props

        assert self.cue_type # TODO: Check that cue_type is the right format.


        if df.at[person_id, 'is_alive']:

            # Confirm availability of health system resources for this interaction

            # Impose the footprint of this health system resource use

            # For each disease module, trigger the on_healthsystem_interaction() event
            registered_disease_modules = self.module.registered_disease_modules
            for module in registered_disease_modules.values():
                module.on_first_healthsystem_interaction(person_id, cue_type=self.cue_type)

            # Log the occurrence of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem_FirstAppt|%s',
                        self.sim.date,
                        {
                            'person_id': person_id,
                            'cue_type': self.cue_type,
                        })

