import logging

import numpy as np
import pandas as pd
import os
from pathlib import Path


from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   help functions
# ---------------------------------------------------------------------------------------------------------


def map_age_groups(age_group):
    """Helper function for obtaining the age range for each age_group
    It returns a tuple of two integers (a,b) such that the given age group is in range a <= group <= b,i.e.:
        0 <= PSAC <= 4
        5 <= SAC <= 14
        15 <= Adults
        0 <= All

    this will cover all ages because we look at the int variable of age
    :param age_group: 'SAC', 'PSAC', 'Adults', 'All'
    """
    assert age_group in ['SAC', 'PSAC', 'Adults', 'All'], "Incorrect age group"

    if age_group == 'PSAC':
        return (0, 4)
    elif age_group == 'SAC':
        return (5, 14)
    elif age_group == 'Adults':
        return (15, 150)
    else:
        return (0,150)


def prob_seeking_healthcare(probabilities):
    """Helper function to get a probability of seeking healthcare due to multiple symptoms. Governed by the maths below:
    s_i - symptoms
    p_i = prob of seeking healthcare due to having symptom s_i
    q_i = 1 - p_i = prob of NOT seeking healthacre due to having symptom s_i
    P(seek healthcare) = 1 - P(not seek healthcare) = 1 - q_1 * q__2 * .... * q_n

    :param probabilities: list of probabilities to seek treatment due to every symptom
    :return: total probability that an individual will seek treatment
    """
    probabilities = [1 - p for p in probabilities]
    total_prob = 1 - np.prod(probabilities)

    return total_prob


def add_elements(el1, el2):
    """Helper function for multiple symptoms assignments

    :param el1: np.nan or a list
    :param el2: list containing a single string with a symptom
    :return: either a sum of two lists or the list el2
    """
    if isinstance(el1, list):
        return el1 + el2
    else:
        return el2

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
    """

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        # natural history
        'prevalence_2010_haem_PSAC': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 of s.haematobium infection among PSAC'),
        'prevalence_2010_haem_SAC': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 of s.haematobium infection among SAC'),
        'prevalence_2010_haem_Adults': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 of s.haematobium infection among Adults'),
        'prevalence_2010_mans_PSAC': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 of s.mansoni infection among PSAC'),
        'prevalence_2010_mans_SAC': Parameter(Types.REAL,
                                              'Initial prevalence in 2010 of s.mansoni infection among SAC'),
        'prevalence_2010_mans_Adults': Parameter(Types.REAL,
                                                'Initial prevalence in 2010 of s.mansoni infection among Adults'),
        'prob_infection': Parameter(Types.REAL, 'Probability that a susceptible individual becomes infected'),  # unused
        'rr_PSAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age under 5 yo'),
        'rr_SAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age 5 - 14 yo'),
        'rr_adults': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age above 14 yo'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'death_schisto_mansoni': Parameter(Types.REAL, 'Rate at which a death from S.Mansoni complications occurs'),
        'death_schisto_haematobium': Parameter(Types.REAL, 'Rate at which a death from S.Haematobium complications occurs'),
        'symptoms_haematobium': Parameter(Types.DATA_FRAME, 'Symptoms for S. Haematobium infection'),
        'symptoms_mansoni': Parameter(Types.DATA_FRAME, 'Symptoms for S. Mansoni infection'),

        # healthburden
        'daly_wt_anemia': Parameter(Types.REAL, 'DALY weight for anemia'),
        'daly_wt_fever': Parameter(Types.REAL, 'DALY weight for fever'),
        'daly_wt_hydronephrosis': Parameter(Types.REAL, 'DALY weight for hydronephrosis'),
        'daly_wt_dysuria': Parameter(Types.REAL, 'DALY weight for dysuria'),
        'daly_wt_bladder_pathology': Parameter(Types.REAL, 'DALY weight for bladder pathology'),
        'daly_wt_diarrhoea': Parameter(Types.REAL, 'DALY weight for diarrhoea'),
        'daly_wt_vomit': Parameter(Types.REAL, 'DALY weight for vomitting'),
        'daly_wt_ascites': Parameter(Types.REAL, 'DALY weight for ascites'),
        'daly_wt_hepatomegaly': Parameter(Types.REAL, 'DALY weight for hepatomegaly'),

        # health system interaction
        'prob_seeking_healthcare': Parameter(Types.REAL,
                                             'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test': Parameter(Types.REAL,
                                           'Probability that an infected individual gets sent to urine or stool lab test'),
        'PZQ_efficacy': Parameter(Types.REAL, 'Efficacy of prazinquantel'),  # unused

        # MDA
        'MDA_prognosed_PSAC': Parameter(Types.REAL, 'Prognosed coverage of MDA in PSAC'),
        'MDA_prognosed_SAC': Parameter(Types.REAL, 'Prognosed coverage of MDA in SAC'),
        'MDA_prognosed_Adults': Parameter(Types.REAL, 'Prognosed coverage of MDA in Adults'),

        'MDA_coverage_PSAC': Parameter(Types.REAL, 'Probability of being administered PZQ in the MDA for PSAC'),
        'MDA_coverage_SAC': Parameter(Types.REAL, 'Probability of being administered PZQ in the MDA for SAC'),
        'MDA_coverage_Adults': Parameter(Types.REAL, 'Probability of being administered PZQ in the MDA for Adults')
    }

    PROPERTIES = {
        'ss_is_infected': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Latent_Haem', 'Latent_Mans', 'Heamatobium', 'Mansoni']),
        'ss_haematobium_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Haematobium infection'),  # actually might also be np.nan
        'ss_mansoni_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Mansoni infection'),  # actually might also be np.nan
        'ss_schedule_infectiousness_start': Property(
            Types.DATE, 'Date of start of infectious period')
    }

    def read_parameters(self, data_folder):

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_Schisto.xlsx'), sheet_name=None)

        # workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx',
        #                          sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['Parameters']
        self.param_list.set_index("Parameter", inplace=True)

        # natural history params
        params['prob_infection'] = self.param_list.loc['prob_infection', 'Value']
        params['delay_a'] = self.param_list.loc['delay_a', 'Value']
        params['delay_b'] = self.param_list.loc['delay_b', 'Value']
        params['death_schisto_haematobium'] = self.param_list.loc['death_schisto_haematobium', 'Value']
        params['death_schisto_mansoni'] = self.param_list.loc['death_schisto_mansoni', 'Value']

        # HSI and treatment params
        params['prob_seeking_healthcare'] = self.param_list.loc['prob_seeking_healthcare', 'Value']
        params['prob_sent_to_lab_test'] = self.param_list.loc['prob_sent_to_lab_test', 'Value']
        params['PZQ_efficacy'] = self.param_list.loc['PZQ_efficacy', 'Value']

        # MDA prognosed
        params['MDA_prognosed_PSAC'] = self.param_list.loc['MDA_prognosed_PSAC', 'Value']
        params['MDA_prognosed_SAC'] = self.param_list.loc['MDA_prognosed_SAC', 'Value']
        params['MDA_prognosed_Adults'] = self.param_list.loc['MDA_prognosed_Adults', 'Value']

        # baseline prevalence
        params['schisto_haem_initial_prev'] = workbook['Prevalence_Haem_2010']
        self.schisto_haem_initial_prev.set_index("District", inplace=True)
        params['prevalence_2010_haem_PSAC'] = self.schisto_haem_initial_prev.loc[:, 'Prevalence PSAC']
        params['prevalence_2010_haem_SAC'] = self.schisto_haem_initial_prev.loc[:, 'Prevalence SAC']
        params['prevalence_2010_haem_Adults'] = self.schisto_haem_initial_prev.loc[:, 'Prevalence Adults']

        params['schisto_mans_initial_prev'] = workbook['Prevalence_Mansoni_2010']
        self.schisto_mans_initial_prev.set_index("District", inplace=True)
        params['prevalence_2010_mans_PSAC'] = self.schisto_mans_initial_prev.loc[:, 'Prevalence PSAC']
        params['prevalence_2010_mans_SAC'] = self.schisto_mans_initial_prev.loc[:, 'Prevalence SAC']
        params['prevalence_2010_mans_Adults'] = self.schisto_mans_initial_prev.loc[:, 'Prevalence Adults']

        params['symptoms_haematobium'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'hydronephrosis', 'dysuria', 'bladder_pathology'],
                'prevalence': [0.6, 0.3, 0.083, 0.2857, 0.7857]

            })
        params['symptoms_mansoni'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'ascites', 'diarrhoea', 'vomit', 'hepatomegaly'],
                'prevalence': [0.6, 0.3, 0.0054, 0.0144, 0.0172, 0.1574]
            })

        # MDA coverage historical
        params['MDA_coverage'] = workbook['MDA_Coverage']
        self.MDA_coverage.set_index(['District', 'Year'], inplace=True)
        params['MDA_coverage_PSAC'] = self.MDA_coverage.loc[:, 'Coverage PSAC']
        params['MDA_coverage_SAC'] = self.MDA_coverage.loc[:, 'Coverage SAC']
        params['MDA_coverage_Adults'] = self.MDA_coverage.loc[:, 'Coverage Adults']

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_anemia'] = self.sim.modules['HealthBurden'].get_daly_weight(258) # moderate anemia
            params['daly_wt_fever'] = self.sim.modules['HealthBurden'].get_daly_weight(262)
            params['daly_wt_hydronephrosis'] = self.sim.modules['HealthBurden'].get_daly_weight(260)
            params['daly_wt_dysuria'] = self.sim.modules['HealthBurden'].get_daly_weight(263)
            params['daly_wt_bladder_pathology'] = self.sim.modules['HealthBurden'].get_daly_weight(264)
            params['daly_wt_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(259)
            params['daly_wt_vomit'] = self.sim.modules['HealthBurden'].get_daly_weight(254)
            params['daly_wt_ascites'] = self.sim.modules['HealthBurden'].get_daly_weight(261)
            params['daly_wt_hepatomegaly'] = self.sim.modules['HealthBurden'].get_daly_weight(257)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individiuals
        n = len(df.index)
        params = self.parameters

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['ss_is_infected'] = 'Non-infected'
        df['ss_scheduled_date_death'] = pd.NaT  # not a time
        df['ss_haematobium_specific_symptoms'] = np.nan
        df['ss_mansoni_specific_symptoms'] = np.nan
        df['ss_schedule_infectiousness_start'] = pd.NaT

        # initial infected population - assuming no one is in the latent period
        # first for simplicity let's assume every infected person has S. Haematobium and no one has S.Mansoni
        self.assign_initial_prevalence(population, 'SAC', 'Haematobium')
        self.assign_initial_prevalence(population, 'PSAC', 'Haematobium')
        self.assign_initial_prevalence(population, 'Adults', 'Haematobium')

        # assign s. heamatobium symptoms
        inf_haem_idx = df[df['ss_is_infected'] == 'Haematobium'].index.tolist()
        self.assign_symptoms(population, inf_haem_idx, 'Haematobium')

        # assign s. mansoni symptoms
        inf_mans_idx = df[df['ss_is_infected'] == 'Mansoni'].index.tolist()
        self.assign_symptoms(population, inf_mans_idx, 'Mansoni')

    def assign_initial_prevalence(self, population, age_group, inf_type):
        """Assign initial 2010 prevalence of S.Haematobium or S.Mansoni.
        This will depend on a district and age group.

        :param population: population
        :param age_group: 'SAC', 'PSAC', 'Adults'
        :param inf_type: 'Mansoni', 'Haematobium'
        """
        assert age_group in ['SAC', 'PSAC', 'Adults'], "Incorrect age group"
        assert inf_type in ['Mansoni', 'Haematobium'], "Incorrect infection type."

        df = population.props
        params = self.parameters
        districts = df.district_of_residence.unique().tolist()

        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b

        if inf_type == 'Haematobium':
            inf_string = 'haem'
        else:
            inf_string = 'mans'
        prev_string = "prevalence_2010_" + inf_string + "_" + age_group

        prevalence = params[prev_string]  # this is a pd.Series not a single value
        # pd.Series.between is by default inclusive of the edges
        for distr in districts:
            prevalence_distr = prevalence[distr]  # get a correct value from the pd.Series
            eligible = df.index[(df['district_of_residence'] == distr) &
                                (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            if len(eligible):
                infected_idx = self.rng.choice(eligible, size=int(prevalence_distr * (len(eligible))), replace=False)
                df.loc[infected_idx, 'ss_is_infected'] = inf_type
                # set the infectiousness period to start now
                df.loc[infected_idx, 'ss_schedule_infectiousness_start'] = self.sim.date

    def assign_symptoms(self, population, eligible_idx, inf_type):
        """
        Assigns multiple symptoms to the initial population.

        :param eligible_idx: indices of infected individuals
        :param population:
        :param inf_type: type of infection, Haematobium or Mansoni
        """
        assert inf_type in ['Mansoni', 'Haematobium'], "Incorrect infection type. Can't assign symptoms."

        if len(eligible_idx):
            if inf_type == 'Haematobium':
                inf_type = 'haematobium'
            else:
                inf_type = 'mansoni'

            df = population.props
            params = self.parameters
            symptoms_dict = params['symptoms_' + inf_type].set_index('symptoms').to_dict()['prevalence']  # create a dictionary form a df
            symptoms_column = 'ss_' + inf_type + '_specific_symptoms'

            for symptom in symptoms_dict.keys():
                p = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
                # find who should get this symptom assigned - get p indices
                s_idx = self.rng.choice(eligible_idx, size=int(p * len(eligible_idx)), replace=False)
                df.loc[s_idx, symptoms_column] = df.loc[s_idx, symptoms_column].apply(lambda x: add_elements(x, [symptom]))

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add the basic events (infection, treatment, MDA)
        sim.schedule_event(SchistoInfectionsEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(SchistoHealthCareSeekEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(SchistoMDAEvent(self), self.sim.date + DateOffset(months=3))
        # self.sim.schedule_event(SchistoMDAEvent(self), self.sim.date + DateOffset(months=3))

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
        df.at[child_id, 'ss_haematobium_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_mansoni_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_schedule_infectiousness_start'] = pd.NaT

    def report_daly_values(self):
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Schisto reporting my health values')

        df = self.sim.population.props

        # for now we only have haematobium infections anyway
        health_values = df.loc[df.is_alive, 'ss_haematobium_specific_symptoms'].apply(lambda x: self.add_DALYs_from_symptoms(x))

        # the mapping above included counting DALYs for people with 'scheduled' symptoms. i.e. in Latent period
        # we want to calculate it only for people who are infectious
        health_values[~df['ss_is_infected'].isin(['Haematobium', 'Mansoni'])] = 0
        health_values.name = 'Schisto_Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series

    def add_DALYs_from_symptoms(self, symptoms):
        params = self.parameters

        dalys_map = {
            'anemia': params['daly_wt_anemia'],
            'fever': params['daly_wt_fever'],
            'hydronephrosis': params['daly_wt_hydronephrosis'],
            'dysuria': params['daly_wt_dysuria'],
            'bladder_pathology': params['daly_wt_bladder_pathology'],
            'diarrhoea': params['daly_wt_diarrhoea'],
            'vomit': params['daly_wt_vomit'],
            'ascites': params['daly_wt_ascites'],
            'hepatomegaly': params['daly_wt_hepatomegaly']
        }
        if isinstance(symptoms, list):
            symptoms = [dalys_map[s] for s in symptoms]
            return sum(symptoms)
        else:
            return 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Schisto, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class SchistoInfectionsEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    In each Infection event a susceptible individual is infected with prob P
    P = #infectious / #total_population
    """

    def __init__(self, module):
        """
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto)

    def apply(self, population):

        logger.debug('This is SchistoEvent, tracking the disease progression of the population.')

        df = population.props
        params = self.module.parameters

        districts = df.district_of_residence.unique().tolist()

        ######################## assign new infections in each district ################################################
        for distr in districts:
            self.new_infections_distr(population, distr)

        ######################## schedule end of the latent period #####################################################
        # new infections are those with un-scheduled time of the end of latency period
        new_infections_haem = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Haem')
                                       & (df['ss_schedule_infectiousness_start'].isna())].tolist()
        new_infections_mans = df.index[(df.is_alive) & (df['ss_is_infected'] == 'Latent_Mans')
                                       & (df['ss_schedule_infectiousness_start'].isna())].tolist()
        new_infections_all = new_infections_haem + new_infections_mans
        print("Number of new infections: " + str(len(new_infections_all)))

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

            ######################## assign symptoms to newly infected #################################################
            self.module.assign_symptoms(population, new_infections_haem, 'Haematobium')
            self.module.assign_symptoms(population, new_infections_mans, 'Mansoni')

        else:
            print("No newly infected")
            logger.debug('This is SchistoInfectionEvent, no one is newly infected.')

    def new_infections_distr(self, population, distr):
        """Assigns new infections of S.Haematobium and S.Mansoni in one district distr

        :param population: population
        : param distr: one of the 32 Malawi districts
        """

        df = population.props
        df_distr = df[(df["district_of_residence"] == distr) & (df.is_alive)].copy()  # get a copy of the main df with only one district and alive ind

        if df_distr.shape[0]:  # if there are any rows in the dataframe

            # 1. get a count of infected to calculate the prevalence
            count_states = {'Non-infected': 0, 'Latent_Haem': 0, 'Latent_Mans': 0, 'Haematobium': 0, 'Mansoni': 0}

            count_states.update(df_distr.ss_is_infected.value_counts().to_dict())  # this will get counts of non-infected, latent and infectious individuals

            count_states.update({'infected_latent_any': count_states['Latent_Haem']
                                                        + count_states['Latent_Mans']})
            count_states.update({'infected_infectious_any': count_states['Haematobium']
                                                            + count_states['Mansoni']})
            count_states.update({'infected_any': count_states['infected_latent_any']
                                                 + count_states['infected_infectious_any']})
            count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})

            # 2 get the indices of susceptibles
            currently_uninfected = df_distr.index[df_distr['ss_is_infected'] == 'Non-infected'].tolist()

            # 3. calculate prevalence of infectious people only, not the actual prevalence
            prevalence_haematobium = count_states['Haematobium'] / count_states['total_pop_alive']
            prevalence_mansoni = count_states['Mansoni'] / count_states['total_pop_alive']

        else:
            currently_uninfected = []
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

        # What is the new state of the susceptibles: are they infected or not?
        susceptibles_next_state = self.module.rng.choice(['Latent_Haem', 'Latent_Mans', 'Non-infected'],
                                                         len(currently_uninfected),
                                                         p=[prevalence_haematobium,
                                                            prevalence_mansoni,
                                                            1-prevalence_haematobium-prevalence_mansoni])
        # 3. update the state of susceptibles in the main df
        df.loc[currently_uninfected, 'ss_is_infected'] = susceptibles_next_state


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

        eligible = df.index[(df.is_alive) & (df.ss_is_infected.isin(['Haematobium', 'Mansoni']))
                            & ~((df['ss_haematobium_specific_symptoms'].isna())
                                & (df['ss_mansoni_specific_symptoms'].isna()))].tolist()  # empty lists are bool False so we get those with at least one symptom
        # these are all infectious & symptomatic

        if len(eligible):  # there are infectious symptomatic people
            # determine who will seek healthcare
            seeking_healthcare = self.module.rng.choice(eligible,
                                                        size=int(params['prob_seeking_healthcare'] * (len(eligible))),
                                                        replace=False)
            # determine which of those who seek healthcare are sent to the schisto diagnostics (hence getting treated)
            treated_idx = self.module.rng.choice(seeking_healthcare,
                                                 size=int(params['prob_sent_to_lab_test'] * (len(seeking_healthcare))),
                                                 replace=False)
            if len(treated_idx) > 0:
                # for those who seek the healthcare initiate treatment
                df.loc[treated_idx, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%, effective immediately
                df.loc[treated_idx, 'ss_schedule_infectiousness_start'] = pd.NaT

                df.loc[treated_idx, 'ss_haematobium_specific_symptoms'] = np.nan  # has to be nan bc not possible to insert []
                df.loc[treated_idx, 'ss_mansoni_specific_symptoms'] = np.nan

                print("Number of treated due to HSI: " + str(len(treated_idx)))
            else:
                print("No one seeked treatment")
                logger.debug('This is SchistoInfectionEvent, no one got treated with PZQ.')
        else:
            print("No one got treatment - no one was infectious and symptomatic")
            logger.debug('This is SchistoInfectionEvent, no one got treated with PZQ.')


class SchistoLatentPeriodEndEvent(Event, IndividualScopeEventMixin):
    """End of the latency period (Asymptomatic -> Infectious transition)
        Also assign a symptom to an infection
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        # should we also change scheduled infectiousness start back to pd.NaT???
        # change infection status from Latent to Infectious
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
#         df.loc[treated_idx, 'ss_haematobium_specific_symptoms'] = 'none'
#         df.loc[treated_idx, 'ss_mansoni_specific_symptoms'] = 'none'
#         df.loc[treated_idx, 'ss_schedule_infectiousness_start'] = pd.NaT


class SchistoMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))
        assert isinstance(module, Schisto)

    def apply(self, population):
        df = self.sim.population.props
        year = self.sim.date.year

        if year in [2015, 2016, 2017, 2018]:
            treated_idx_PSAC = self.assign_historical_MDA_coverage(population, year, 'PSAC')
            treated_idx_SAC = self.assign_historical_MDA_coverage(population, year, 'SAC')
            treated_idx_Adults = self.assign_historical_MDA_coverage(population, year, 'Adults')

        else:  # for now no district-specific MDA coverage
            treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, 'PSAC')
            treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, 'SAC')
            treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, 'Adults')

        print("PSAC treated in MDA: " + str(len(treated_idx_PSAC)))
        print("SAC treated in MDA: " + str(len(treated_idx_SAC)))
        print("Adults treated in MDA: " + str(len(treated_idx_Adults)))

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults

        # people administered PZQ in MDA but in the Latent period will get the pill but won't be cured
        # similarly susceptibles will get the pill but nothing will happen
        # The infected will get cured immediately
        infected_idx = df.index[(df.is_alive) & (df.ss_is_infected.isin(['Haematobium', 'Mansoni']))]
        MDA_treated = list(set(treated_idx) & set(infected_idx))  # intersection of infected & given a PZQ, so effectively treated

        df.loc[MDA_treated, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%
        df.loc[MDA_treated, 'ss_haematobium_specific_symptoms'] = np.nan
        df.loc[MDA_treated, 'ss_mansoni_specific_symptoms'] = np.nan
        df.loc[MDA_treated, 'ss_schedule_infectiousness_start'] = pd.NaT

        # count how many PZQ tablets were distributed
        PZQ_tablets_used = len(treated_idx)  # just in this round of MDA
        print("Year " + str(year) + ", PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))

    def assign_historical_MDA_coverage(self, population, year, age_group):
        """Assign coverage of MDA program to chosen age_group.

          :param population: population
          :param year: current year. used to find the coverage
          :param age_group: 'SAC', 'PSAC', 'Adults'
          """
        assert year in [2015, 2016, 2017, 2018], "No data for requested MDA coverage"

        df = population.props
        params = self.module.parameters
        districts = df.district_of_residence.unique().tolist()

        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b

        param_str = 'MDA_coverage_' + age_group
        coverage = params[param_str]  # this is a pd.Series not a single value
        coverage = coverage[:, year]
        MDA_idx = []  # store indices of treated individuals

        print("MDA coverage in district:")
        for distr in districts:
            coverage_distr = coverage[distr]  # get a correct value from the pd.Series
            eligible = df.index[(df['district_of_residence'] == distr) &
                                (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            if len(eligible):
                MDA_idx_distr = self.module.rng.choice(eligible,
                                                       size=int(coverage_distr * (len(eligible))), replace=False)
                MDA_idx = MDA_idx + MDA_idx_distr.tolist()

        return MDA_idx

    def assign_prognosed_MDA_coverage(self, population, age_group):
        """Assign coverage of MDA program to chosen age_group.

          :param population: population
          :param age_group: 'SAC', 'PSAC', 'Adults'
          """

        df = population.props
        params = self.module.parameters
        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
        param_str = 'MDA_prognosed_' + age_group

        coverage = params[param_str]

        eligible = df.index[(df.is_alive) & (df['age_years'].between(age_range[0], age_range[1]))].tolist()
        MDA_idx = []
        if len(eligible):
            MDA_idx = self.module.rng.choice(eligible, size=int(coverage * (len(eligible))), replace=False).tolist()

        return MDA_idx
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
        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto)

    def create_logger(self, population, age_group):
        count_states = self.count_age_group_states(population, age_group)
        tot_prevalence = count_states['infected_any'] / count_states['total_pop_alive']
        log_string = '%s|' + age_group + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'Susc': count_states['Non-infected'],
                        'LatentHaem': count_states['Latent_Haem'],
                        'InfectiousHaem': count_states['Haematobium'],
                        'InfectedHaem': count_states['infected_any'], # not this in fact but for now when only Haem infections
                        'Prevalence': tot_prevalence,
                        'TotalTreated:': 9999999999999999999999999  # how to get this? this would be treatments + MDA coverage
                    })

    def count_age_group_states(self, population, age_group):
        """
        :param population:
        :param age_group:
        :return: count_states: a dictionary of counts of individuals in age_group in different states on infection
        """
        df = population.props
        age_range = map_age_groups(age_group)  # returns a tuple

        # this is not ideal bc we're making a copy but it's for clearer code below
        df_age = df[((df.is_alive) & (df['age_years'].between(age_range[0], age_range[1])))].copy()  # get a copy of the main df with only specified age group only
        count_states = {}

        count_states = {'Non-infected': 0, 'Latent_Haem': 0, 'Latent_Mans': 0, 'Haematobium': 0, 'Mansoni': 0,}
        count_states.update(df_age.ss_is_infected.value_counts().to_dict())  # this will get counts of non-infected, latent and infectious individuals

        count_states.update({'infected_latent_any': count_states['Latent_Haem']
                                                              + count_states['Latent_Mans']})
        count_states.update({'infected_infectious_any': count_states['Haematobium']
                                                                  + count_states['Mansoni']})
        count_states.update({'infected_any': count_states['infected_latent_any']
                                                       + count_states['infected_infectious_any']})
        count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})

        return count_states

    def apply(self, population):

        self.create_logger(population, 'PSAC')
        self.create_logger(population, 'SAC')
        self.create_logger(population, 'Adults')
        self.create_logger(population, 'All')


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

# class HSI_Schisto_Seeking Healthcare(HSI_Event, IndividualScopeEventMixin):
#     """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
#     like this.
#     It must begin HSI_<Module_Name>_Description
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#         assert isinstance(module, Schisto)
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
