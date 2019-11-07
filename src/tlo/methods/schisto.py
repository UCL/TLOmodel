import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Schisto(Module):
    """
    Schistosomiasis module
    It demonstrates the following behaviours in respect of the healthsystem module:
        - Registration of the disease module
        - Reading DALY weights and reporting daly values related to this disease
        - Health care seeking
        - Usual HSI behaviour
        - Restrictive requirements on the facility_level for the HSI_event

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)` [If this is disease module]
    * `on_hsi_alert(person_id, treatment_id)` [If this is disease module]
    *  `report_daly_values()` [If this is disease module]

    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {

    # natural history
        'prevalence_2010_haem': Parameter(Types.REAL, 'Initial prevalence in 2010 of s.haematobium infection'),
        'prob_infection_haem': Parameter(Types.REAL, 'Probability that a susceptible individual becomes infected with S. Haematobium'),
        'prevalence_2010_mansoni': Parameter(Types.REAL, 'Initial prevalence in 2010 of s.Mansoni infection'),
        'prob_infection_mansoni': Parameter(Types.REAL, 'Probability that a susceptible individual becomes infected with S. Mansoni'),
        'rr_PSAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age under 5 yo'),
        'rr_SAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age 5 - 14 yo'),
        'rr_adults': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age above 14 yo'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'death_schisto_mansoni': Parameter(Types.REAL, 'Rate at which a death from S.Mansoni complications occure'),
        'death_schisto_haematobium': Parameter(Types.REAL, 'Rate at which a death from S.Haematobium complications occure'),

    # health system interaction
        'prob_seeking_healthcare': Parameter(Types.REAL, 'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test': Parameter(Types.REAL, 'Probability that an infected individual gets sent to urine or stool lab test'),
        'PZQ_efficacy': Parameter(Types.REAL, 'Efficacy of prazinquantel'),

    # MDA
        'prob_PZQ_in_MDA': Parameter(Types.REAL, 'Probability of being administered PZQ in the MDA programme'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        'ss_is_infected': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories = ['Non-infected', 'Latent_Haem', 'Latent_Mans', 'Heamatobium', 'Mansoni']),
        'ss_haematobium_specific_symptoms': Property(
            Types.CATEGORICAL, 'Symptoms for S. Haematobium infection',
            categories=['none', 'fever', 'stomach_ache', 'skin', 'other']),
        'ss_mansoni_specific_symptoms': Property(
            Types.CATEGORICAL, 'Symptoms for S. Mansoni infection',
            categories=['none', 'fever', 'stomach_ache', 'diarrhoea', 'vomit', 'skin', 'other']),
        'ss_schedule_infectiousness_start': Property(
            Types.DATE, 'Date of start of infectious period')
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        params = self.parameters
        params['prevalence_2010_haem'] = 0.5
        params['prob_infection'] = 0.5
        params['delay_a'] = 25
        params['delay_b'] = 30
        params['initial_prevalence'] = 0.5
        params['death_schisto_haematobium'] = 0.0005
        params['death_schisto_mansoni'] = 0.0005

        params['prob_seeking_healthcare'] = 0.3
        params['prob_sent_to_lab_test'] = 0.9
        params['PZQ_efficacy'] = 1.0

        params['symptoms_haematobium'] = pd.DataFrame(
            data={
                'symptoms': ['none', 'fever', 'stomach_ache', 'skin', 'other'],
                'probability': [0.2, 0.2, 0.2, 0.2, 0.2]
            })

        params['symptoms_mansoni'] = pd.DataFrame(
            data={
                'symptoms': ['none', 'fever', 'stomach_ache', 'diarrhoea', 'vomit', 'skin', 'other'],
                'probability': [0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
            })


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individiuals
        params = self.parameters

        df['ss_is_infected'] = 'Non-infected'
        df['ss_scheduled_date_death'] = pd.NaT # not a time
        df['ss_haematobium_specific_symptoms'] = np.nan # NaN value
        df['ss_mansoni_specific_symptoms'] = np.nan
        df['ss_schedule_infectiousness_start'] = pd.NaT

        ### initial infected population - assuming no one is in the latent period
        # first for simplicity let's assume every infected person has S. Haematobium and no one has S.Mansoni
        eligible = df.index()
        infected_idx = self.rng.choice(eligible, size = int(params.prevalence_2010_haem * (len(eligible))), replace=False)
        df.loc[infected_idx, 'ss_is_infected'] = 'Haematobium'

        # assign scheduled date of death (from a demography module?)

        # assign s. heamatobium symptoms
        inf_haem_idx = df[df['ss_is_infected'] == 'Haematobium'].index
        eligible = len(inf_haem_idx)
        symptoms_haematobium = params.symptoms_haematobium.symptoms.values # from the params
        symptoms_haem_prob = params.symptoms_haematobium.probability.values
        symptoms = np.random.choice(symptoms_haematobium, size = int((len(eligible))), replace = True, p = symptoms_haem_prob)
        df.loc[inf_haem_idx, 'symptoms_haematobium'] = symptoms

        # assign s. mansoni symptoms
        inf_mans_idx = df[df['ss_is_infected'] == 'Mansoni'].index
        eligible = len(inf_mans_idx)
        symptoms_mansoni = params.symptoms_mansoni.symptoms.values # from the params
        symptoms_mans_prob = params.symptoms_mansoni.probability.values
        symptoms = np.random.choice(symptoms_mansoni, size = int((len(eligible))), replace = True, p = symptoms_mans_prob)
        df.loc[inf_mans_idx, 'symptoms_mansoni'] = symptoms

        # set the start of infectiousness to the start date of the simulation for simplicity
        latent_period_ahead_haem = self.module.rng.uniform(params.delay_a, params.delay_b,
                                                      size=len(inf_haem_idx))
        latent_period_ahead_mans = self.module.rng.uniform(params.delay_a, params.delay_b,
                                                      size=len(inf_mans_idx))
        latent_period_td_ahead_haem = pd.to_timedelta(latent_period_ahead_haem, unit='D')
        latent_period_td_ahead_mans = pd.to_timedelta(latent_period_ahead_mans, unit='D')

        df.loc[df.inf_haem_idx, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_td_ahead_haem
        df.loc[df.inf_mans_idx, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_td_ahead_mans

        # Schedule the event that will launch the Outreach event
        MDA_event = SchistoMDAEvent(self)
        self.sim.schedule_event(MDA_event, self.sim.date + DateOffset(months=12)) # MDA once per year



    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        If this is a disease module, register this disease module with the healthsystem:
        self.sim.modules['HealthSystem'].register_disease_module(self)
        """
        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        raise NotImplementedError

    def on_birth(self, child_id):
        """Initialise our properties for a newborn individual.

        All children are born without an infection, even if the mother is infected.

        :param child_id: the new child
        """
        # Assign the default for a newly born child
        df.at[child_id, 'ss_is_infected'] = 'Non-infected'
        df.at[child_id, 'ss_scheduled_date_death'] = pd.NaT
        df.at[child_id, 'ss_haematobium_specific_symptoms'] = pd.NaT
        df.at[child_id, 'ss_mansoni_specific_symptoms'] = pd.NaT
        df.at[child_id, 'ss_schedule_infectiousness_start'] = pd.NaT


    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

        raise NotImplementedError

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class SchistoEventInfections(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=0.5))
        assert isinstance(module, Schisto)

    def apply(self, population):

        logger.debug('This is SchistoEvent, tracking the disease progression of the population.')

        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected_latent_haem = df.index[df['ss_is_infected'] == 'Latent_Haem']
        currently_infected_latent_mans = df.index[df['ss_is_infected'] == 'Latent_Mans']
        currently_infected_latent_any = currently_infected_latent_haem + currently_infected_latent_mans
        currently_infected_haematobium = df.index[df['ss_is_infected'] == 'Haematobium']
        currently_infected_mansoni = df.index[df['ss_is_infected'] == 'Mansoni']
        currently_infected_infectious_any = currently_infected_haematobium + currently_infected_mansoni
        currently_infected_any = currently_infected_haematobium + currently_infected_mansoni + currently_infected_latent_any

        currently_uninfected = df.index[df['ss_is_infected'] == 'Non-Infected']
        total_population = currently_uninfected + currently_infected_any

        # calculates prevalence of infectious people only to calculate the new infections, not the actual prevalence
        if df.is_alive.sum():
            prevalence_haematobium = len(currently_infected_haematobium) / len(total_population)
            prevalence_mansoni = len(currently_infected_mansoni) / len(total_population)
        else:
            prevalence_haematobium = 0
            prevalence_mansoni = 0


        # 2. handle new infections - for now no co-infections
        # now_infected_haematobium = self.module.rng.choice([True, False],
        #                                       size = len(currently_uninfected),
        #                                       p = prevalence_haematobium)
        # now_infected_mansoni = self.module.rng.choice([True, False],
        #                                           size = len(currently_uninfected), # here we will get co-infections!!!!
        #                                           p = prevalence_mansoni)
        # now_infected_haematobium = self.module.rng.choice(currently_uninfected, size = len)

        new_infections = self.module.rng.choice(['Latent_Haem', 'Latent_Mans', 'Non-Infected'], len(currently_uninfected),
                                                p = [prevalence_haematobium, prevalence_mansoni, 1-prevalence_haematobium-prevalence_mansoni])
        df.loc[currently_uninfected, 'ss_is_infected'] = new_infections

        # new infections are those with un-scheduled time of the end of latency period
        new_infections_haem = df.index[(df['ss_is_infected'] == 'Latent_Haem') & (df['ss_schedule_infectiousness_start'].isnan())]
        new_infections_mans = df.index[(df['ss_is_infected'] == 'Latent_Mans') & (df['ss_schedule_infectiousness_start'].isnan())]
        new_infections_all = new_infections_haem + new_infections_mans

        # if any are infected
        if len(new_infections_all) > 0:

            # schedule start of infectiousness
            latent_period_ahead = self.module.rng.uniform(population.parameters.delay_a,population.parameters.delay_b,
                                                              size = len(new_infections_all))
            # this is continuous, do we need that? probably discrete number of days would be ok
            latent_period_ahead = pd.to_timedelta(latent_period_ahead, unit = 'D')
            df.loc[new_infections_all, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_ahead

            for person_index in new_infections_all:
                end_latent_period_event = SchistoLatentPeriodEndEvent(self.module, person_index)
                self.sim.schedule_event(end_latent_period_event, df.at[person_index, 'ss_schedule_infectiousness_start'])

            # assign symptoms - when should they be triggered????????
            symptoms_haematobium = population.parameters.symptoms_haematobium.symptoms.values  # from the params
            symptoms_haem_prob = population.parameters.symptoms_haematobium.probability.values
            df.loc[new_infections_haem, 'ss_haematobium_specific_symptoms'] =\
                np.random.choice(symptoms_haematobium, size=int((len(new_infections_haem))), replace=True,
                                        p=symptoms_haem_prob)

            symptoms_mansoni = population.parameters.symptoms_haematobium.symptoms.values  # from the params
            symptoms_mans_prob = population.parameters.symptoms_haematobium.probability.values
            df.loc[new_infections_haem, 'ss_mansoni_specific_symptoms'] =\
                np.random.choice(symptoms_mansoni, size=int((len(new_infections_mans))), replace=True,
                                        p=symptoms_mans_prob)

        #     # Determine if anyone with severe symptoms will seek care
        #     serious_symptoms = (df['is_alive']) & ((df['mi_specific_symptoms'] == 'extreme emergency') | (
        #         df['mi_specific_symptoms'] == 'coughing and irritiable'))
        #
        #     seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
        #     for i in df.index[serious_symptoms]:
        #         prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
        #         seeks_care[i] = self.module.rng.rand() < prob
        #
        #     if seeks_care.sum() > 0:
        #         for person_index in seeks_care.index[seeks_care is True]:
        #             logger.debug(
        #                 'This is MockitisEvent, scheduling Mockitis_PresentsForCareWithSevereSymptoms for person %d',
        #                 person_index)
        #             event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
        #             self.sim.modules['HealthSystem'].schedule_hsi_event(event,
        #                                                                 priority=2,
        #                                                                 topen=self.sim.date,
        #                                                                 tclose=self.sim.date + DateOffset(weeks=2)
        #                                                                 )
        #     else:
        #         logger.debug(
        #             'This is SchistoEvent, There is  no one with new severe symptoms so no new healthcare seeking')
        # else:
        #     logger.debug('This is SchistoEvent, no one is newly infected.')

class SchistoLatentPeriodEndEvent(RegularEvent, IndividualScopeEventMixin):
    """End of the latency period (Asymptomatic -> Infectious transition)
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props
        if df.at[person_id, 'ss_is_infected'] == 'Latent_Haem':
            df.at[person_id, 'ss_is_infected'] = 'Haematobium'
        elif df.at[person_id, 'ss_is_infected'] == 'Latent_Mans':
            df.at[person_id, 'ss_is_infected'] = 'Mansoni'

class SchistoMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    should be scheduled district-wise
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Shisto)

    def apply(self, population):
        df = self.sim.population.props
        coverage = 0.7 # for now we assign the same coverage for all ages and districts
        # choose coverage-fraction of the population
        eligible = df.index()
        treated_idx = self.rng.choice(eligible, size = int(coverage * (len(eligible))), replace=False)
        # change their infection status to Non-Infected
        df.loc[treated_idx, 'ss_is_infected'] = 'Non-Infected' # PZQ efficay 100%
        # count how many PZQ tablets were distributed
        PZQ_tablets_used = len(treated_idx) # just in this round of MDA
        print("PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a logging event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class Skeleton_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        # Make some summary statitics

        dict_to_output = {
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info('%s|summary_12m|%s', self.sim.date, dict_to_output)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_Skeleton_Example_Interaction(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
    like this.
    It must begin HSI_<Module_Name>_Description
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Skeleton)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 0

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Skeleton_Example_Interaction'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """
        Do the action that take place in this health system interaction, in light of squeeze_factor
        Can reutrn an updated APPT_FOOTPRINT if this differs from the declaration in self.EXPECTED_APPT_FOOTPRINT
        """
        pass

    def did_not_run(self):
        """
        Do any action that is neccessary when the health system interaction is not run.
        This is called each day that the HSI is 'due' but not run due to insufficient health system capabilities

        """
        pass
