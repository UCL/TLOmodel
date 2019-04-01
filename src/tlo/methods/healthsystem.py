"""
This module stands in for the "Health System" in the current implementation of the module
It is used to control access to interventions
It will be replaced by the Health Care Seeking Behaviour Module and the
"""
import logging
import os

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HealthSystem(Module):
    """
    Requests for access to particular services are handled by Disease/Intervention Modules by this Module
    """

    def __init__(self, name=None, resourcefilepath=None, service_availability=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        if service_availability is None:
            service_availability = pd.DataFrame(columns=['Service', 'Available']).astype(
                {'Service': object, 'Available': bool}
            )

        # Checks on the service_availability dateframe argument
        assert type(service_availability) is pd.DataFrame
        assert len(service_availability.columns) == 2
        assert 'Service' in service_availability.columns
        assert 'Available' in service_availability.columns
        assert service_availability['Service'].dtype == object
        assert service_availability['Available'].dtype == bool

        self.service_availability = service_availability

        self.registered_disease_modules = {}

        self.registered_interventions = pd.DataFrame(
            columns=['Name', 'Nurse_Time', 'Doctor_Time', 'Electricity', 'Water'])

        self.health_system_resources = None

        logger.info('----------------------------------------------------------------------')
        logger.info("Setting up the Health System With the Following Service Availabilty:")
        print_table = service_availability.to_string().splitlines()
        for line in print_table:
            logger.info(line)
        logger.info('----------------------------------------------------------------------')

    PARAMETERS = {
        'Master_Facility_List':
            Parameter(Types.DATA_FRAME,
                      'Imported Master Facility List workbook: one row per each facility'),
        'Village_To_Facility_Mapping':
            Parameter(Types.DATA_FRAME, 'Imported long-list of links between villages and health '
                                        'facilities: one row per each link between a village and a facility')
    }

    PROPERTIES = {
        'hs_dist_to_facility':
            Property(Types.REAL,
                     'The distance for each person to their nearest clinic (of any type)')
    }

    def read_parameters(self, data_folder):

        self.parameters['Master_Facility_List'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_MasterFacilitiesList.csv')
        )

        self.parameters['Village_To_Facility_Mapping'] = pd.read_csv(
            os.path.join(self.resourcefilepath, 'ResourceFile_Village_To_Facility_Mapping.csv')
        )

        # Establish the MasterCapacitiesList
        # (This is where the data on all health capabilities will be stored. For now, nothing happens)

    def initialise_population(self, population):
        df = population.props

        # Assign hs_dist_to_facility'
        # For now, let this be a random number, but in future it will be properly informed based on
        # population density distribitions.

        # Note that this characteritic is inherited from mother to child.
        df['hs_dist_to_facility'] = self.sim.rng.uniform(0.01, 5.00, len(df))

    def initialise_simulation(self, sim):
        # Launch the healthcare seeking poll
        sim.schedule_event(HealthCareSeekingPollEvent(self), sim.date)

        # Check that each person is attached to a village and a set of attached health facilities
        pop = self.sim.population.props
        mapping = self.parameters['Village_To_Facility_Mapping']
        for person_id in pop.index[pop.is_alive]:
            my_village = pop.at[person_id, 'village_of_residence']
            my_health_facilities = mapping.loc[mapping['Village'] == my_village]
            assert len(my_health_facilities) > 0

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'hs_dist_to_facility'] = df.at[mother_id, 'hs_dist_to_facility']

    def register_disease_module(self, *new_disease_modules):
        # Register Disease Modules (so that the health system can broadcast triggers to all disease modules)
        for module in new_disease_modules:
            assert module.name not in self.registered_disease_modules, (
                'A module named {} has already been registered'.format(module.name))
            self.registered_disease_modules[module.name] = module

            logger.info('Registering disease module %s', module.name)

    def register_interventions(self, footprint_df):
        # Register the interventions that can be requested

        # Check that this footprint can be added to the registered interventions
        assert type(footprint_df) == pd.DataFrame
        assert not (footprint_df.Name.values in self.registered_interventions.Name.values)
        assert 'Name' in footprint_df.columns
        assert 'Nurse_Time' in footprint_df.columns
        assert 'Doctor_Time' in footprint_df.columns
        assert 'Electricity' in footprint_df.columns
        assert 'Water' in footprint_df.columns
        assert len(footprint_df.columns) == 5

        self.registered_interventions = self.registered_interventions.append(footprint_df)
        logger.info('Registering intervention %s', footprint_df.at[0, 'Name'])

    def query_access_to_service(self, person_id, service):
        logger.info('Query whether person %d has access to service %s', person_id, service)

        df = self.sim.population.props

        # Check that this is a legitimate request:
        assert df.at[person_id, 'is_alive']  # All requests should come from alive persons
        assert service in self.registered_interventions.Name.values

        # Health system allows all requests for services
        # (This is where the constraint will be imposed and the footprint recorded)

        enough_capacity = True  # No constraint imposed
        policy_allows = True  # No constraint imposed
        gets_service = True  # No constraint imposed

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
    This event determines who has symptoms that are sufficient to bring them into care.
    It occurs regularly at 3-monthly intervals.
    It uses the "general health care seeking equation" to do this.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):

        logger.debug('Health Care Seeking Poll is running')

        # ----------
        # 1) Work out the overall unified symptom code

        #   Fill in value of zeros (in case that no disease modules are registerd)
        overall_symptom_code = pd.Series(data=0, index=self.sim.population.props.index)

        registered_disease_modules = self.module.registered_disease_modules

        if len(registered_disease_modules.values()):
            # Ask each module to update and report-out the symptoms it is currently causing on the
            # unified symptomology scale:

            unified_symptoms_code = pd.DataFrame()
            for module in registered_disease_modules.values():
                out = module.query_symptoms_now()

                # check that the data received is in correct format
                assert type(out) is pd.Series
                assert len(out) == self.sim.population.props.is_alive.sum()
                assert self.sim.population.props.index.name == out.index.name
                assert self.sim.population.props.is_alive[out.index].all()
                assert (~pd.isnull(out)).all()
                assert all(out.dtype.categories == [0, 1, 2, 3, 4])

                # Add this to the dataframe
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

            # *************************************************************************
            # The "general health care seeking equation" ***
            # (This is a dummy)
            prob_seek_care = max(0.00,
                                 min(1.00,
                                     0.02 + age * 0.02 + education * 0.1 + healthlevel * 0.2))
            # *************************************************************************

            # determine if there will be health-care contact and schedule FirstAppt if so
            if self.sim.rng.rand() < prob_seek_care:
                event = HealthSystemInteractionEvent(self.module, person_index,
                                                     cue_type='HealthCareSeekingPoll',
                                                     disease_specific=None)
                self.sim.schedule_event(event, self.sim.date)

        # ----------


class OutreachEvent(Event, PopulationScopeEventMixin):
    """
    This event can be used to simulate the occurrence of an 'outreach' intervention such as screening.
    It commissions new interactions with the Health System for those persons reach. It receives an arguement 'target'
    which is a pd.Series (of length alive persons and with the index of the population dataframe) that shows who is
    reached in the outreach intervention. It does not automatically reschedule. The disease_specific argument determines
    the type of interaction that is triggered: if disease_specific = None, thee resulting HealthSystemInteractionEvents
    will have disease_specific=None; if disease_specific is set to a registered disease module name, this will be passed
    to the resulting HealthSystemInteractionEvents.
    NB. A known issue is that if this event is scheduled into the future, then persons that are born into the population
    after the event is defined will not benefit from the outreach intervention.
    """

    # TODO: Would like to create event and give rules for persons to be included/exlcuded that are
    # evaluated when the event is run.

    def __init__(self, module, disease_specific=None, target=pd.Series([], dtype=bool)):
        super().__init__(module)

        logger.debug('Outreach event being created. Type: %s, %s', disease_specific)

        # Check the arguments that have been passed:
        assert (disease_specific is None) or (disease_specific in self.sim.modules[
            'HealthSystem'].registered_disease_modules.keys())
        assert type(target) is pd.Series
        assert len(target) == self.sim.population.props.is_alive.sum()
        assert self.sim.population.props.index.name == target.index.name
        assert self.sim.population.props.is_alive[target.index].all()
        assert (~pd.isnull(target)).all()

        self.disease_specific = disease_specific
        self.target = target

    def apply(self, population):

        logger.debug('Outreach event running now')

        target_indicies = self.target.index

        # Schedule a first appointment for each person for this disease only
        for person_id in target_indicies:

            if self.sim.population.props.at[person_id, 'is_alive']:
                event = HealthSystemInteractionEvent(self.module, person_id,
                                                     cue_type='OutreachEvent',
                                                     disease_specific=self.disease_specific)
                self.sim.schedule_event(event, self.sim.date)

        # Log the occurrence of the outreach event
        logger.info('%s|outreach_event|%s', self.sim.date,
                    {
                        'disease_specific': self.disease_specific
                    })


class HealthSystemInteractionEvent(Event, IndividualScopeEventMixin):
    """
    This is the generic interaction between the person and the health system.
    All actual interactions between a person and the health system happen here.
    It can be called by: HealthCareSeekingPoll, OutreachEvent or a DiseaseModule itself.
    It broadcasts details of the interaction to all registered disease modules by calling
    the 'on_healthsystem_interaction' in each. It passes, the information about the type of interaction
    (cue_type and disease type) that have been received.
    * cue_type is the type of event that has caused this interaction.
    * disease_specific is the name of a disease module (or None) that is linked to this interaction.
    It logs the interaction and imposes any health system constraint that may exist.
    the calls for resources.
    """

    def __init__(self, module, person_id, cue_type=None, disease_specific=None):
        super().__init__(module, person_id=person_id)
        self.cue_type = cue_type
        self.disease_specific = disease_specific

        # Check that this is correctly specificed interaction
        assert person_id in self.sim.population.props.index
        assert self.cue_type in ['HealthCareSeekingPoll', 'OutreachEvent', 'InitialDiseaseCall',
                                 'FollowUp', None]
        assert (self.disease_specific is None) or (self.disease_specific in self.sim.modules[
            'HealthSystem'].registered_disease_modules.keys())

    def apply(self, person_id):
        logger.debug('@@@ An interaction with the health system')

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:

            # 1. Confirm availability of health system resources for this interaction
            # Skip TODO: Re-Insert constraint

            # 2. Impose the footprint of this health system resource use
            # Skip TODO: Re-Insert constraint

            # 3. For each disease module, trigger the on_healthsystem_interaction() event

            registered_disease_modules = self.sim.modules['HealthSystem'].registered_disease_modules

            for module in registered_disease_modules.values():
                module.on_healthsystem_interaction(person_id,
                                                   cue_type=self.cue_type,
                                                   disease_specific=self.disease_specific)

            # 4. Log the occurrence of this interaction with the health system
            logger.info('%s|InteractionWithHealthSystem|%s',
                        self.sim.date,
                        {
                            'person_id': person_id,
                            'cue_type': self.cue_type,
                            'disease_specific': self.disease_specific
                        })
