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
        'death_schisto_mansoni': Parameter(Types.REAL, 'Rate at which a death from S.Mansoni complications occurs'),
        'death_schisto_haematobium': Parameter(Types.REAL, 'Rate at which a death from S.Haematobium complications occurs'),
        'daly_wt_fever': Parameter(Types.REAL, 'DALY weight for fever'),
        'daly_wt_stomach_ache': Parameter(Types.REAL, 'DALY weight for stomach_ache'),
        'daly_wt_skin': Parameter(Types.REAL, 'DALY weight for skin rash'),
        'daly_wt_diarrhoea': Parameter(Types.REAL, 'DALY weight for diarrhoea'),
        'daly_wt_other': Parameter(Types.REAL, 'DALY weight for other symptoms'),

        # health system interaction
        'prob_seeking_healthcare': Parameter(Types.REAL, 'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test': Parameter(Types.REAL, 'Probability that an infected individual gets sent to urine or stool lab test'),
        'PZQ_efficacy': Parameter(Types.REAL, 'Efficacy of prazinquantel'),

        # MDA
        'prob_PZQ_in_MDA': Parameter(Types.REAL, 'Probability of being administered PZQ in the MDA programme'),
    }

    PROPERTIES = {
        'ss_is_infected': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Latent_Haem', 'Latent_Mans', 'Heamatobium', 'Mansoni']),
        'ss_haematobium_specific_symptoms': Property(
            Types.CATEGORICAL, 'Symptoms for S. Haematobium infection',
            categories=['none', 'fever', 'stomach_ache', 'skin', 'other']),
        'ss_mansoni_specific_symptoms': Property(
            Types.CATEGORICAL, 'Symptoms for S. Mansoni infection',
            categories=['none', 'fever', 'stomach_ache', 'diarrhoea', 'vomit', 'skin', 'other']),
        'ss_schedule_infectiousness_start': Property(
            Types.DATE, 'Date of start of infectious period')
    }

    # def __init__(self, name=None, resourcefilepath=None):
    #     # NB. Parameters passed to the module can be inserted in the __init__ definition.
    #
    #     super().__init__(name)
    #     self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        params = self.parameters

        params['prevalence_2010_haem'] = 0.5
        params['prevalence_2010_mans'] = 0.5
        params['prob_infection'] = 0.5
        params['delay_a'] = 25
        params['delay_b'] = 30
        params['death_schisto_haematobium'] = 0.0005
        params['death_schisto_mansoni'] = 0.0005

        params['prob_seeking_healthcare'] = 0.3
        params['prob_sent_to_lab_test'] = 0.95
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
        params['prob_PZQ_in_MDA'] = 0.7

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_fever'] = self.sim.modules['HealthBurden'].get_daly_weight(262)
            params['daly_wt_stomach_ache'] = self.sim.modules['HealthBurden'].get_daly_weight(263)
            params['daly_wt_skin'] = self.sim.modules['HealthBurden'].get_daly_weight(261)
            params['daly_wt_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(254)
            params['daly_wt_other'] = self.sim.modules['HealthBurden'].get_daly_weight(259)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individiuals
        params = self.parameters

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['ss_is_infected'] = 'Non-infected'
        df['ss_scheduled_date_death'] = pd.NaT # not a time
        df['ss_haematobium_specific_symptoms'] = 'none'
        df['ss_mansoni_specific_symptoms'] = 'none'
        df['ss_schedule_infectiousness_start'] = pd.NaT

        # initial infected population - assuming no one is in the latent period
        # first for simplicity let's assume every infected person has S. Haematobium and no one has S.Mansoni
        print("Assign initially infected with S.Haematobium")
        eligible = df.index.tolist()
        infected_idx = self.rng.choice(eligible, size=int(params['prevalence_2010_haem'] *
                                                            (len(eligible))), replace=False)
        df.loc[infected_idx, 'ss_is_infected'] = 'Haematobium'
        print("Assigned - S.Haematobium")

        # assign s. heamatobium symptoms
        print("Assing S.Haematobium symptoms to the infected")
        inf_haem_idx = df[df['ss_is_infected'] == 'Haematobium'].index
        eligible_count = len(inf_haem_idx.tolist())
        symptoms_haematobium = params['symptoms_haematobium'].symptoms.values
        symptoms_haem_prob = params['symptoms_haematobium'].probability.values
        symptoms = np.random.choice(symptoms_haematobium, size=eligible_count, replace = True, p = symptoms_haem_prob)
        df.loc[inf_haem_idx, 'ss_haematobium_specific_symptoms'] = symptoms

        # assign s. mansoni symptoms
        print("Assing S.Mansoni symptoms to the infected")
        inf_mans_idx = df[df['ss_is_infected'] == 'Mansoni'].index
        eligible_count = len(inf_mans_idx.tolist())
        symptoms_mansoni = params['symptoms_mansoni'].symptoms.values # from the params
        symptoms_mans_prob = params['symptoms_mansoni'].probability.values
        symptoms = np.random.choice(symptoms_mansoni, size=eligible_count, replace = True, p = symptoms_mans_prob)
        df.loc[inf_mans_idx, 'ss_mansoni_specific_symptoms'] = symptoms

        print("Fill in start of infectiousness")
        # set the start of infectiousness to the start date of the simulation for simplicity
        df.loc[inf_haem_idx, 'ss_schedule_infectiousness_start'] = self.sim.date
        df.loc[inf_mans_idx, 'ss_schedule_infectiousness_start'] = self.sim.date

        # # maybe use this instead later??
        # latent_period_ahead_haem = self.module.rng.uniform(params['delay_a'], params['delay_b'],
        #                                               size=len(inf_haem_idx))
        # latent_period_ahead_mans = self.module.rng.uniform(params['delay_a'], params['delay_b'],
        #                                               size=len(inf_mans_idx))
        # latent_period_td_ahead_haem = pd.to_timedelta(latent_period_ahead_haem, unit='D')
        # latent_period_td_ahead_mans = pd.to_timedelta(latent_period_ahead_mans, unit='D')
        #
        # df.loc[df.inf_haem_idx, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_td_ahead_haem
        # df.loc[df.inf_mans_idx, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_td_ahead_mans

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

        # add the basic events
        event_infection = SchistoInfectionsEvent(self)
        sim.schedule_event(event_infection, sim.date + DateOffset(months=1))
        event_treatment = SchistoHealthCareSeekEvent(self)
        sim.schedule_event(event_treatment, sim.date + DateOffset(months=1))

        # add and event of MDA
        MDA_event = SchistoMDAEvent(self)
        self.sim.schedule_event(MDA_event, self.sim.date + DateOffset(months=3))

        # add an event to log to screen
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + DateOffset(months=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props

        # Assign the default for a newly born child
        df.at[child_id, 'ss_is_infected'] = 'Non-infected'
        df.at[child_id, 'ss_scheduled_date_death'] = pd.NaT
        df.at[child_id, 'ss_haematobium_specific_symptoms'] = 'none'
        df.at[child_id, 'ss_mansoni_specific_symptoms'] = 'none'
        df.at[child_id, 'ss_schedule_infectiousness_start'] = pd.NaT

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Schisto reporting my health values')

        df = self.sim.population.props
        params = self.parameters

        # for now we only have haematobium infections anyway
        health_values = df.loc[df.is_alive, 'ss_haematobium_specific_symptoms'].map({
            'none': 0,
            'fever': params['daly_wt_fever'],
            'stomach_ache': params['daly_wt_stomach_ache'],
            'skin': params['daly_wt_skin'],
            'diarrhoea': params['daly_wt_diarrhoea'],
            'other': params['daly_wt_other']
        })

        health_values.name = 'Schisto_Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Schisto, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class SchistoInfectionsEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    """

    def __init__(self, module):
        """
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months = 1))
        assert isinstance(module, Schisto)

    def apply(self, population):

        logger.debug('This is SchistoEvent, tracking the disease progression of the population.')

        df = population.props
        params = self.module.parameters

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected_latent_haem = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Haem')].tolist()
        currently_infected_latent_mans = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Mans')].tolist()
        currently_infected_latent_any = currently_infected_latent_haem + currently_infected_latent_mans
        currently_infected_haematobium = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Haematobium')].tolist()
        currently_infected_mansoni = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Mansoni')].tolist()
        currently_infected_infectious_any = currently_infected_haematobium + currently_infected_mansoni
        currently_infected_any = currently_infected_haematobium + currently_infected_mansoni + currently_infected_latent_any

        currently_uninfected = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Non-infected')].tolist()
        total_population_alive = currently_uninfected + currently_infected_any

        # calculates prevalence of infectious people only to calculate the new infections, not the actual prevalence
        if df.is_alive.sum():
            prevalence_haematobium = len(currently_infected_haematobium) / len(total_population_alive)
            prevalence_mansoni = len(currently_infected_mansoni) / len(total_population_alive)
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

        susceptibles_next_state = self.module.rng.choice(['Latent_Haem', 'Latent_Mans', 'Non-infected'],
                                                         len(currently_uninfected),
                                                         p=[prevalence_haematobium,
                                                            prevalence_mansoni,
                                                            1-prevalence_haematobium-prevalence_mansoni])
        df.loc[currently_uninfected, 'ss_is_infected'] = susceptibles_next_state
        # print(susceptibles_next_state)

        # new infections are those with un-scheduled time of the end of latency period
        new_infections_haem = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Haem')
                                       & (df['ss_schedule_infectiousness_start'].isna())].tolist()
        new_infections_mans = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Mans')
                                       & (df['ss_schedule_infectiousness_start'].isna())].tolist()
        new_infections_all = new_infections_haem + new_infections_mans

        # if any are infected
        if len(new_infections_all) > 0:

            # schedule start of infectiousness
            latent_period_ahead = self.module.rng.uniform(params['delay_a'],
                                                          params['delay_b'],
                                                          size=len(new_infections_all))
            # this is continuous, do we need that? probably discrete number of days would be ok
            latent_period_ahead = pd.to_timedelta(latent_period_ahead, unit='D')
            df.loc[new_infections_all, 'ss_schedule_infectiousness_start'] = self.sim.date + latent_period_ahead

            for person_index in new_infections_all:
                end_latent_period_event = SchistoLatentPeriodEndEvent(self.module, person_id=person_index)
                self.sim.schedule_event(end_latent_period_event, df.at[person_index, 'ss_schedule_infectiousness_start'])

            # assign symptoms - when should they be triggered???????? Also best to make it another event???
            symptoms_haematobium = params['symptoms_haematobium'].symptoms.values  # from the params
            symptoms_haem_prob = params['symptoms_haematobium'].probability.values
            df.loc[new_infections_haem, 'ss_haematobium_specific_symptoms'] = \
                self.module.rng.choice(symptoms_haematobium, size=int((len(new_infections_haem))),
                                       replace=True, p=symptoms_haem_prob)

            symptoms_mansoni = params['symptoms_mansoni'].symptoms.values  # from the params
            symptoms_mans_prob = params['symptoms_mansoni'].probability.values
            df.loc[new_infections_mans, 'ss_mansoni_specific_symptoms'] =\
                self.module.rng.choice(symptoms_mansoni, size=int((len(new_infections_mans))),
                                       replace=True, p=symptoms_mans_prob)
        else:
            print("No newly infected")
            logger.debug('This is SchistoInfectionEvent, no one is newly infected.')


class SchistoHealthCareSeekEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    """

    def __init__(self, module):
        """
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto)

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        eligible = df.index[(df.is_alive) & (df.ss_is_infected.isin(['Haematobium', 'Mansoni'])) &
                           ~((df['ss_haematobium_specific_symptoms'] == 'none') &
                           (df['ss_mansoni_specific_symptoms'] == 'none'))].tolist()  # these are all infectious & symptomatic

        # determine who will seek healthcare
        seeking_healthcare = self.module.rng.choice(eligible,
                                                    size=int(params['prob_seeking_healthcare'] * (len(eligible))),
                                                    replace=False)
        # determine which of those who seek healthcare are sent to the schisto diagnostics (and hence getting treated)
        treated_idx = self.module.rng.choice(seeking_healthcare,
                                                    size=int(params['prob_sent_to_lab_test'] * (len(seeking_healthcare))),
                                                    replace=False)
        # for those who seeks the healthcare initiate treatment
        df.loc[treated_idx, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%, effective immediately
        df.loc[treated_idx, 'ss_haematobium_specific_symptoms'] = 'none'
        df.loc[treated_idx, 'ss_mansoni_specific_symptoms'] = 'none'

        if len(treated_idx) > 0:
            print("Number of treated due to HSI: " + str(len(treated_idx)))
        else:
            print("No one got treatment")
            logger.debug('This is SchistoInfectionEvent, no one got treated with PZQ.')


class SchistoLatentPeriodEndEvent(Event, IndividualScopeEventMixin):
    """End of the latency period (Asymptomatic -> Infectious transition)
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props
        if df.at[person_id, 'ss_is_infected'] == 'Latent_Haem':
            df.at[person_id, 'ss_is_infected'] = 'Haematobium'
        elif df.at[person_id, 'ss_is_infected'] == 'Latent_Mans':
            df.at[person_id, 'ss_is_infected'] = 'Mansoni'


# class SchistoTreatment(Event, IndividualScopeEventMixin):
#     """Treatment upon Heathcare interaction - simple version
#     """
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Schisto)
#
#     def apply(self, person_id):
#         df = self.sim.population.props
#         df.loc[treated_idx, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%, effective immediately


class SchistoMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    should be scheduled district-wise
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, Schisto)

    def apply(self, population):
        df = self.sim.population.props
        params = self.module.parameters
        coverage = params['prob_PZQ_in_MDA']  # for now we assign the same coverage for all ages and districts

        # choose coverage-fraction of the population
        eligible = df.index[df.is_alive].tolist()
        treated_idx = self.module.rng.choice(eligible, size=int(coverage * (len(eligible))), replace=False)

        # change their infection status to Non-infected
        df.loc[treated_idx, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%, effective immediately
        df.loc[treated_idx, 'ss_haematobium_specific_symptoms'] = 'none'
        df.loc[treated_idx, 'ss_mansoni_specific_symptoms'] = 'none'

        # count how many PZQ tablets were distributed
        PZQ_tablets_used = len(treated_idx)  # just in this round of MDA
        print("PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a logging event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------


class SchistoLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """
        # run this event every year
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto)

    def apply(self, population):
        df = population.props

        currently_infected_latent_haem = len(df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Haem')].tolist())
        currently_infected_latent_mans = len(df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Mans')].tolist())
        currently_infected_latent_any = currently_infected_latent_haem + currently_infected_latent_mans
        currently_infected_haematobium = len(df.index[(df.is_alive) & (df['ss_is_infected'] == 'Haematobium')].tolist())
        currently_infected_mansoni = len(df.index[(df.is_alive) & (df['ss_is_infected'] == 'Mansoni')].tolist())
        currently_infected_infectious_any = currently_infected_haematobium + currently_infected_mansoni
        currently_infected_any = currently_infected_haematobium + currently_infected_mansoni + currently_infected_latent_any

        currently_uninfected = len(df.index[(df.is_alive) & (df['ss_is_infected'] == 'Non-infected')].tolist())
        total_population_alive = currently_uninfected + currently_infected_any

        print("currently haem infectious: " + str(currently_infected_haematobium))
        print("currently haem latent: " + str(currently_infected_latent_haem))
        print("currently susceptible: " + str(currently_uninfected))
        print("total population alive: " + str(total_population_alive))

        # counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        # counts.update(df.loc[df.is_alive, 'mi_status'].value_counts().to_dict())
        #
        # # logger.info('%s|summary|%s', self.sim.date,
        # #             {
        # #                 'TotalInf': infected_total,
        # #                 'PropInf': proportion_infected,
        # #                 'PrevMonth': infected_in_last_month,
        # #                 'Cured': cured_in_last_month,
        # #             })
        #
        # logger.info('%s|status_counts|%s', self.sim.date, counts)

# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------
#
# class HSI_Skeleton_Example_Interaction(HSI_Event, IndividualScopeEventMixin):
#     """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
#     like this.
#     It must begin HSI_<Module_Name>_Description
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Skeleton)
#
#         # Define the call on resources of this treatment event: Time of Officers (Appointments)
#         #   - get an 'empty' footprint:
#         the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#         #   - update to reflect the appointments that are required
#         the_appt_footprint['Over5OPD'] = 1  # This requires one out patient
#
#         # Define the facilities at which this event can occur (only one is allowed)
#         # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
#         #                            ['Facility_Level']))
#         the_accepted_facility_level = 0
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = 'Skeleton_Example_Interaction'  # This must begin with the module name
#         self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
#         self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         """
#         Do the action that take place in this health system interaction, in light of squeeze_factor
#         Can reutrn an updated APPT_FOOTPRINT if this differs from the declaration in self.EXPECTED_APPT_FOOTPRINT
#         """
#         pass
#
#     def did_not_run(self):
#         """
#         Do any action that is neccessary when the health system interaction is not run.
#         This is called each day that the HSI is 'due' but not run due to insufficient health system capabilities
#
#         """
#         pass
