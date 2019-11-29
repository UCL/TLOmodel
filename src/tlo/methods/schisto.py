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
        return (0,4)
    elif age_group == 'SAC':
        return (5,14)
    elif age_group == 'Adults':
        return (15,150)
    else:
        return (0,150)


def prob_seeking_healthcare(probabilities):
    """Helper function to get a probability of seeking healthcare due to multiple symptoms. Governed by the maths below:
    s_i - symptoms
    p_i = prob of seeking healthcare due to having symptom s_i
    q_i = 1 - p_i = prob of NOT seeking healthacare due to having symptom s_i
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
        'prevalence_2010_PSAC': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 among PSAC'),
        'prevalence_2010_SAC': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 among SAC'),
        'prevalence_2010_Adults': Parameter(Types.REAL,
                                               'Initial prevalence in 2010 among Adults'),
        'prob_infection': Parameter(Types.REAL, 'Probability that a susceptible individual becomes infected'),
        'rr_PSAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age under 5 yo'),
        'rr_SAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age 5 - 14 yo'),
        'rr_Adults': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age above 14 yo'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'symptoms_haematobium': Parameter(Types.DATA_FRAME, 'Symptoms for S. Haematobium infection'),
        'symptoms_mansoni': Parameter(Types.DATA_FRAME, 'Symptoms for S. Mansoni infection'),

        # healthburden
        'daly_wt_anemia': Parameter(Types.REAL, 'DALY weight for anemia'),
        'daly_wt_fever': Parameter(Types.REAL, 'DALY weight for fever'),
        'daly_wt_haematuria': Parameter(Types.REAL, 'DALY weight for haematuria'),
        'daly_wt_hydronephrosis': Parameter(Types.REAL, 'DALY weight for hydronephrosis'),
        'daly_wt_dysuria': Parameter(Types.REAL, 'DALY weight for dysuria'),
        'daly_wt_bladder_pathology': Parameter(Types.REAL, 'DALY weight for bladder pathology'),
        'daly_wt_diarrhoea': Parameter(Types.REAL, 'DALY weight for diarrhoea'),
        'daly_wt_vomit': Parameter(Types.REAL, 'DALY weight for vomitting'),
        'daly_wt_ascites': Parameter(Types.REAL, 'DALY weight for ascites'),
        'daly_wt_hepatomegaly': Parameter(Types.REAL, 'DALY weight for hepatomegaly'),

        # health system interaction
        'delay_till_hsi_a': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, start'),
        'delay_till_hsi_b': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, end'),
        'prob_seeking_healthcare': Parameter(Types.REAL,
                                             'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test_children': Parameter(Types.REAL,
                                           'Probability that an infected child gets sent to urine or stool lab test'),
        'prob_sent_to_lab_test_adults': Parameter(Types.REAL,
                             'Probability that an infected adults gets sent to urine or stool lab test'),
        'PZQ_efficacy': Parameter(Types.REAL, 'Efficacy of prazinquantel'),  # unused
        'symptoms_mapped_for_hsb': Parameter(Types.REAL,
                                             'Symptoms to which the symptoms assigned in the module are mapped for the HSB module'),

        # MDA
        # 'MDA_prognosed_freq': Parameter(Types.REAL, 'Prognosed MDA frequency in months'),
        # 'MDA_prognosed_PSAC': Parameter(Types.REAL, 'Prognosed coverage of MDA in PSAC'),
        # 'MDA_prognosed_SAC': Parameter(Types.REAL, 'Prognosed coverage of MDA in SAC'),
        # 'MDA_prognosed_Adults': Parameter(Types.REAL, 'Prognosed coverage of MDA in Adults'),

        'MDA_coverage_PSAC': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for PSAC'),
        'MDA_coverage_SAC': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for SAC'),
        'MDA_coverage_Adults': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for Adults'),

        'MDA_prognosed_freq': Parameter(Types.DATA_FRAME, 'Prognosed MDA frequency in months'),
        'MDA_prognosed_PSAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in PSAC'),
        'MDA_prognosed_SAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in SAC'),
        'MDA_prognosed_Adults': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in Adults'),
    }

    PROPERTIES = {
        'ss_is_infected': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Latent', 'Infected']),
        'ss_haematobium_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Haematobium infection'),  # actually might also be np.nan
        'ss_mansoni_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Mansoni infection'),  # actually might also be np.nan
        'ss_infection_date': Property(
            Types.DATE, 'Date of the most recent infection event'),
        'ss_schedule_infectiousness_start': Property(
            Types.DATE, 'Date of start of infectious period'),
        'ss_scheduled_hsi_date': Property(
            Types.DATE, 'Date of scheduled seeking healthcare'),
        'ss_cumulative_infection_time': Property(
            Types.REAL, 'Cumulative time of being infected in days'),
        'ss_cumulative_DALYs': Property(
            Types.REAL, 'Cumulative DALYs due to schistosomiasis symptoms')
    }

    def read_parameters(self, data_folder):

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_Schisto.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['Parameters']
        self.param_list.set_index("Parameter", inplace=True)

        # natural history params
        params['prob_infection'] = self.param_list.loc['prob_infection', 'Value']
        params['delay_a'] = self.param_list.loc['delay_a', 'Value']
        params['delay_b'] = self.param_list.loc['delay_b', 'Value']
        params['rr_PSAC'] = self.param_list.loc['rr_PSAC', 'Value']
        params['rr_SAC'] = self.param_list.loc['rr_SAC', 'Value']
        params['rr_Adults'] = self.param_list.loc['rr_Adults', 'Value']

        # HSI and treatment params
        params['delay_till_hsi_a'] = self.param_list.loc['delay_till_hsi_a', 'Value']
        params['delay_till_hsi_b'] = self.param_list.loc['delay_till_hsi_b', 'Value']
        params['prob_seeking_healthcare'] = self.param_list.loc['prob_seeking_healthcare', 'Value']
        params['prob_sent_to_lab_test_children'] = self.param_list.loc['prob_sent_to_lab_test_children', 'Value']
        params['prob_sent_to_lab_test_adults'] = self.param_list.loc['prob_sent_to_lab_test_adults', 'Value']
        params['PZQ_efficacy'] = self.param_list.loc['PZQ_efficacy', 'Value']

        # MDA prognosed
        # params['MDA_prognosed_freq'] = self.param_list.loc['MDA_prognosed_freq', 'Value']
        # params['MDA_prognosed_PSAC'] = self.param_list.loc['MDA_prognosed_PSAC', 'Value']
        # params['MDA_prognosed_SAC'] = self.param_list.loc['MDA_prognosed_SAC', 'Value']
        # params['MDA_prognosed_Adults'] = self.param_list.loc['MDA_prognosed_Adults', 'Value']

        # baseline prevalence
        params['schisto_initial_prev'] = workbook['Prevalence_2010']
        self.schisto_initial_prev.set_index("District", inplace=True)
        params['prevalence_2010_PSAC'] = self.schisto_initial_prev.loc[:, 'Prevalence PSAC']
        params['prevalence_2010_SAC'] = self.schisto_initial_prev.loc[:, 'Prevalence SAC']
        params['prevalence_2010_Adults'] = self.schisto_initial_prev.loc[:, 'Prevalence Adults']

        params['symptoms_haematobium'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'haematuria', 'hydronephrosis', 'dysuria', 'bladder_pathology'],
                'prevalence': [0.6, 0.3, 0.625, 0.083, 0.2857, 0.7857],
                'hsb_symptoms': ['other', 'fever', 'other', 'stomach_ache', 'stomach_ache', 'stomach_ache'],
            })
        params['symptoms_mansoni'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'ascites', 'diarrhoea', 'vomit', 'hepatomegaly'],
                'prevalence': [0.6, 0.3, 0.0054, 0.0144, 0.0172, 0.1574],
                'hsb_symptoms': ['other', 'fever', 'stomach_ache', 'diarrhoea', 'vomit', 'stomach_ache'],
            })

        # MDA coverage historical
        params['MDA_coverage'] = workbook['MDA_historical_Coverage']
        self.MDA_coverage.set_index(['District', 'Year'], inplace=True)
        params['MDA_coverage_PSAC'] = self.MDA_coverage.loc[:, 'Coverage PSAC']
        params['MDA_coverage_SAC'] = self.MDA_coverage.loc[:, 'Coverage SAC']
        params['MDA_coverage_Adults'] = self.MDA_coverage.loc[:, 'Coverage Adults']

        # MDA coverage prognosed
        params['MDA_coverage_prognosed'] = workbook['MDA_prognosed_Coverage']
        self.MDA_coverage_prognosed.set_index(['District'], inplace=True)
        params['MDA_frequency_prognosed'] = self.MDA_coverage_prognosed.loc[:, 'Frequency']
        params['MDA_prognosed_PSAC'] = self.MDA_coverage_prognosed.loc[:, 'Coverage PSAC']
        params['MDA_prognosed_SAC'] = self.MDA_coverage_prognosed.loc[:, 'Coverage SAC']
        params['MDA_prognosed_Adults'] = self.MDA_coverage_prognosed.loc[:, 'Coverage Adults']

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_anemia'] = self.sim.modules['HealthBurden'].get_daly_weight(258)  # moderate anemia
            params['daly_wt_fever'] = self.sim.modules['HealthBurden'].get_daly_weight(262)
            params['daly_wt_haematuria'] = 0
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

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['ss_is_infected'] = 'Non-infected'
        df['ss_haematobium_specific_symptoms'] = np.nan
        df['ss_mansoni_specific_symptoms'] = np.nan
        df['ss_infection_date'] = pd.NaT
        df['ss_schedule_infectiousness_start'] = pd.NaT
        df['ss_scheduled_hsi_date'] = pd.NaT
        df['ss_cumulative_infection_time'] = 0
        df['ss_cumulative_DALYs'] = 0

        # initial infected population - assuming no one is in the latent period
        self.assign_initial_prevalence(population, 'SAC')
        self.assign_initial_prevalence(population, 'PSAC')
        self.assign_initial_prevalence(population, 'Adults')

        # assign s. heamatobium symptoms
        inf_haem_idx = df[df['ss_is_infected'] == 'Infected'].index.tolist()
        self.assign_symptoms(population, inf_haem_idx, 'haematobium')

    def assign_initial_prevalence(self, population, age_group):
        """Assign initial 2010 prevalence of schistosomiasis infections
        This will depend on a district and age group.

        :param population: population
        :param age_group: 'SAC', 'PSAC', 'Adults'
        """
        assert age_group in ['SAC', 'PSAC', 'Adults'], "Incorrect age group"

        df = population.props
        params = self.parameters
        districts = df.district_of_residence.unique().tolist()

        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b

        prev_string = "prevalence_2010_" + age_group
        prevalence = params[prev_string]  # this is a pd.Series not a single value

        # pd.Series.between is by default inclusive of the edges
        for distr in districts:
            prevalence_distr = prevalence[distr]  # get a correct value from the pd.Series
            eligible = df.index[(df['district_of_residence'] == distr) &
                                (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            if len(eligible):
                infected_idx = self.rng.choice(eligible, size=int(prevalence_distr * (len(eligible))), replace=False)
                df.loc[infected_idx, 'ss_is_infected'] = 'Infected'
                # set the date of infection to start now
                df.loc[infected_idx, 'ss_infection_date'] = self.sim.date
                # set the infectiousness period to start now
                df.loc[infected_idx, 'ss_schedule_infectiousness_start'] = self.sim.date

    def assign_symptoms(self, population, eligible_idx, inf_type):
        """
        Assigns multiple symptoms to the initial population.

        :param eligible_idx: indices of infected individuals
        :param population:
        :param inf_type: type of infection, haematobium or mansoni
        """
        assert inf_type in ['mansoni', 'haematobium'], "Incorrect infection type. Can't assign symptoms."

        if len(eligible_idx):
            df = population.props
            params = self.parameters
            symptoms_dict = params['symptoms_' + inf_type].set_index('symptoms').to_dict()['prevalence']  # create a dictionary from a df
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

        # add the basic events of infection
        sim.schedule_event(SchistoInfectionsEvent(self), sim.date + DateOffset(months=1))

        # schedule historical MDA to happen once per year in July (4 events)
        for historical_mda_year in [2015, 2016, 2017, 2018]:
            if historical_mda_year >= sim.date.year:
                sim.schedule_event(SchistoHistoricalMDAEvent(self),
                                   pd.Timestamp(year=historical_mda_year, month=7, day=1, hour=23))
        # schedule prognosed MDA programmes for every district
        for district in sim.population.props.district_of_residence.unique().tolist():
            freq = self.parameters['MDA_frequency_prognosed'][district]
            sim.schedule_event(SchistoPrognosedMDAEvent(self, freq, district),
                               pd.Timestamp(year=2019, month=6, day=1, hour=12) + DateOffset(months=0))

        # sim.schedule_event(SchistoHealthCareSeekEvent(self), sim.date + DateOffset(days=14))

        # add an event to log to screen
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + DateOffset(months=0))
        sim.schedule_event(SchistoDALYsLoggingEvent(self), sim.date + DateOffset(months=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props

        # Assign the default for a newly born child
        df.at[child_id, 'ss_is_infected'] = 'Non-infected'
        df.at[child_id, 'ss_haematobium_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_mansoni_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_infection_date'] = pd.NaT
        df.at[child_id, 'ss_schedule_infectiousness_start'] = pd.NaT
        df.at[child_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.at[child_id, 'ss_cumulative_infection_time'] = 0
        df.at[child_id, 'ss_cumulative_DALYs'] = 0

    def report_daly_values(self):
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Schisto reporting my health values')

        df = self.sim.population.props

        # for now we only have haematobium infections anyway
        health_values = df.loc[df.is_alive, 'ss_haematobium_specific_symptoms'].apply(lambda x: self.add_DALYs_from_symptoms(x))

        # the mapping above included counting DALYs for people with 'scheduled' symptoms. i.e. in Latent period
        # we want to calculate it only for people who are infectious
        health_values[df['ss_is_infected'] == 'Infected'] = 0
        health_values.name = 'Schisto_Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series

    def add_DALYs_from_symptoms(self, symptoms):
        params = self.parameters

        dalys_map = {
            'anemia': params['daly_wt_anemia'],
            'fever': params['daly_wt_fever'],
            'haematuria': params['daly_wt_haematuria'],
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
    P = transm_prob * relative_risk_age * #infectious_district / #total_population_district
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
        new_infections = []
        for distr in districts:
            new_infections = new_infections + self.new_infections_distr(population, distr)

        if len(new_infections):
            ############## schedule the time of infection in the following month #######################################
            days_till_infection = self.module.rng.uniform(0, 30, size=len(new_infections))
            days_till_infection = pd.to_timedelta(days_till_infection, unit='D')
            df.loc[new_infections, 'ss_infection_date'] = self.sim.date + days_till_infection

            ######################## assign symptoms to newly infected #################################################
            self.module.assign_symptoms(population, new_infections, 'haematobium')

            ############## schedule the time of end of latent period ###################################################
            latent_period_ahead = self.module.rng.uniform(params['delay_a'],
                                                          params['delay_b'],
                                                          size=len(new_infections))
            latent_period_ahead = pd.to_timedelta(latent_period_ahead, unit='D')
            df.loc[new_infections, 'ss_schedule_infectiousness_start'] = df.loc[
                                                                             new_infections, 'ss_infection_date'] + latent_period_ahead

            ############## schedule the time of seeking healthcare #####################################################
            seeking_treatment__ahead = self.module.rng.uniform(params['delay_till_hsi_a'],
                                                          params['delay_till_hsi_b'],
                                                          size=len(new_infections))
            seeking_treatment__ahead = pd.to_timedelta(seeking_treatment__ahead, unit='D')
            df.loc[new_infections, 'ss_scheduled_hsi_date'] = df.loc[new_infections, 'ss_schedule_infectiousness_start'] \
                                                              + seeking_treatment__ahead

            ############ schedule events of infection and end of latent period and seeking healthcare
            for person_index in new_infections:
                infect_event = SchistoInfection(self.module, person_id=person_index)
                self.sim.schedule_event(infect_event, df.at[person_index, 'ss_infection_date'])
                end_latent_period_event = SchistoLatentPeriodEndEvent(self.module, person_id=person_index)
                self.sim.schedule_event(end_latent_period_event,
                                        df.at[person_index, 'ss_schedule_infectiousness_start'])
                seek_treatment_event = HSI_SchistoSeekTreatment(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
                                                                    priority=1,
                                                                    topen=df.at[person_index, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.at[person_index, 'ss_scheduled_hsi_date'] + DateOffset(weeks = 4))

        print("Number of new infections: " + str(len(new_infections)))
        if len(new_infections) == 0:
            logger.debug('This is SchistoInfectionEvent, no one is newly infected.')

    def new_infections_distr(self, population, distr):
        """Randomly samples the indices of newly infected indices in given district

        :param population: population
        :param distr: one of the 32 Malawi districts
        :returns new_infections: indices of newly infected people in district distr
        """

        df = population.props
        params = self.module.parameters
        df_distr = df[(df["district_of_residence"] == distr) & (df.is_alive)].copy()  # get a copy of the main df with only one district and alive ind

        if df_distr.shape[0]:  # if there are any rows in the dataframe, so there are alive poeple in the district
            ############## get a count of infected to calculate the prevalence #######################################
            count_states = {'Non-infected': 0, 'Latent': 0, 'Infected': 0}
            count_states.update(df_distr.ss_is_infected.value_counts().to_dict())  # this will get counts of non-infected, latent and infectious individuals
            count_states.update({'infected_any': count_states['Latent']
                                                 + count_states['Infected']})
            count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})

            ############# get the indices of susceptibles ##########################################################
            currently_uninfected = df_distr.index[df_distr['ss_is_infected'] == 'Non-infected'].tolist()

            ############# calculate prevalence of infectious people only, not the actual prevalence #################
            if count_states['Non-infected']:
                prevalence = count_states['Infected'] / count_states['total_pop_alive']
            else:
                prevalence = 0
        else:
            currently_uninfected = []
            prevalence = 0

        ############# get relative risks for the susceptibles #########################################################
        # ss_risk = pd.Series(1, index=currently_uninfected)  # this will be holding the relative risks for uninfected in the district alive
        ss_risk = pd.Series(1, index=df_distr.index.tolist())  # this will be holding the relative risks for everyone in the district alive
        # ss_risk.loc[df_distr.ss_is_infected == 'Non-infected'] = 1

        for age_group in ['PSAC', 'SAC', 'Adults']:
            params_str = 'rr_' + age_group
            age_range = map_age_groups(age_group)
            ss_risk.loc[df.age_years.between(age_range[0], age_range[1])] *= params[params_str]
        # norm_p = pd.Series(ss_risk) / pd.Series(ss_risk).sum()

        ############## find the new infections indices ###############################################################
        trans_prob = prevalence * params['prob_infection']

        # check whether any of the trans_prob are higher than 1 - in that case they will for sure get the infection, which is not correct
        are_above_1 = trans_prob >= 1
        assert ~are_above_1.any(), "Hazard of infection is higher than 1"

        newly_infected_index = df_distr.index[(self.module.rng.random_sample(size=len(df_distr.index)) < (trans_prob * ss_risk))]
        new_infections = list(set(newly_infected_index) & set(currently_uninfected))

        return new_infections


class SchistoInfection(Event, IndividualScopeEventMixin):
    """Changes the status of a person to Infected (Non-infected -> Latent)
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        if df.loc[person_id, 'ss_is_infected'] == 'Non-infected':
            df.loc[person_id, 'ss_is_infected'] = 'Latent'


class SchistoLatentPeriodEndEvent(Event, IndividualScopeEventMixin):
    """End of the latency period (Latend -> Infected (infectious))
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        if df.loc[person_id, 'ss_is_infected'] == 'Latent':
            df.loc[person_id, 'ss_is_infected'] = 'Infected'


class SchistoTreatmentEvent(Event, IndividualScopeEventMixin):
    """Cured upon PZQ treatment through HSI or MDA (Infected -> Non-infected)
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        # calculate the duration of this infection
        inf_duration = self.sim.date - df.loc[person_id, 'ss_infection_date']
        if np.isnan(inf_duration / np.timedelta64(1, 'D')):
            print(inf_duration, self.sim.date, df.loc[person_id, 'ss_infection_date'])  # this is an error that happens sometimes
        inf_duration = int(inf_duration / np.timedelta64(1, 'D'))
        df.loc[person_id, 'ss_cumulative_infection_time'] += inf_duration
        symptoms = df.loc[person_id, 'ss_haematobium_specific_symptoms']
        df.loc[person_id, 'ss_cumulative_DALYs'] += self.calculate_DALY_per_infection(inf_duration, symptoms)

        df.loc[person_id, 'ss_is_infected'] = 'Non-infected'  # PZQ efficacy 100%, effective immediately
        df.loc[person_id, 'ss_haematobium_specific_symptoms'] = np.nan
        df.loc[person_id, 'ss_mansoni_specific_symptoms'] = np.nan
        df.loc[person_id, 'ss_infection_date'] = pd.NaT
        df.loc[person_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.loc[person_id, 'ss_schedule_infectiousness_start'] = pd.NaT

    def calculate_DALY_per_infection(self, inf_duration, symptoms):
        dalys_weight = self.module.add_DALYs_from_symptoms(symptoms)
        DALY = (inf_duration / 30.0) * (dalys_weight / 12.0)  # inf_duration in days and weight given per year
        return DALY

class HSI_SchistoSeekTreatment(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event of seeking treatment for a person with symptoms
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        the_accepted_facility_level = 0

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Schisto_Treatment_seeking'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level

        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        params = self.module.parameters

        # some ppl will have been cured in the MDA before the app or will have symptomless infections,
        if df.loc[person_id, 'ss_is_infected'] == 'Infected':
            # check if a person is a child or an adult and assign prob of being sent to schisto test (and hence being cured)
            if df.loc[person_id, 'age_years'] <= 15:
                prob_test = params['prob_sent_to_lab_test_children']
            else:
                prob_test = params['prob_sent_to_lab_test_adults']

            sent_to_test = self.sim.rng.choice([True, False], p=[prob_test, 1-prob_test])

            if sent_to_test:
                # request the consumable
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
                items_code1 = \
                    pd.unique(
                        consumables.loc[
                            consumables['Items'] == "Praziquantel, 600 mg (donated)", 'Item_Code'])[0]
                the_cons_footprint = {'Intervention_Package_Code': [], 'Item_Code': [{items_code1: 1}]}
                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=False
                )
                # give the PZQ to the patient
                if outcome_of_request_for_consumables['Item_Code'][items_code1]:
                    self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True
                    )
                    logger.debug('ItemsCode1 is available, so use it.')
                else:
                    logger.debug('ItemsCode1 is not available, so can' 't use it.')

                # patient is cured
                self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
            else:
                print('Person', person_id, 'seeked treatment but was not sent to test')
        else:
            print('Person', person_id, 'had the appt scheduled but was cured in the MDA')

    def did_not_run(self):
        return True


# class SchistoHealthCareSeekEvent(RegularEvent, PopulationScopeEventMixin):
#     """An event of infecting people with Schistosomiasis
#     This is a population level event, the prob of seeking healthcare does not depend on a symptom experienced
#     """
#
#     def __init__(self, module):
#         """
#         :param module: the module that created this event
#         """
#         super().__init__(module, frequency=DateOffset(months=1))
#         assert isinstance(module, Schisto)
#
#     def apply(self, population):
#         df = population.props
#         params = self.module.parameters
#
#         eligible_children = df.index[
#             (df.is_alive) & (df.ss_is_infected == 'Infected') & (df['age_years'].between(0, 14))
#             & ~((df['ss_haematobium_specific_symptoms'].isna())
#                 & (df[
#                        'ss_mansoni_specific_symptoms'].isna()))].tolist()  # empty lists are bool False so we get those with at least one symptom
#
#         eligible_adults = df.index[
#                 (df.is_alive) & (df.ss_is_infected == 'Infected') & (df['age_years'].between(15, 120))
#                 & ~((df['ss_haematobium_specific_symptoms'].isna())
#                     & (df[
#                            'ss_mansoni_specific_symptoms'].isna()))].tolist()  # empty lists are bool False so we get those with at least one symptom
#
#         # these are all infectious & symptomatic
#
#         if len(eligible_children):  # there are infectious symptomatic children
#             # determine who will seek healthcare
#             seeking_healthcare_children = self.module.rng.choice(eligible_children,
#                                                         size=int(params['prob_seeking_healthcare'] * (len(eligible_children))),
#                                                         replace=False)
#             # determine which of those who seek healthcare are sent to the schisto diagnostics (hence getting treated)
#             treated_children_idx = self.module.rng.choice(seeking_healthcare_children,
#                                                  size=int(params['prob_sent_to_lab_test_children'] * (len(seeking_healthcare_children))),
#                                                  replace=False).tolist()
#         else:
#             treated_children_idx = []
#
#         if len(eligible_adults):  # there are infectious symptomatic adults
#             # determine which of those who seek healthcare are sent to the schisto diagnostics (hence getting treated)
#             seeking_healthcare_adults = self.module.rng.choice(eligible_adults,
#                                                         size=int(params['prob_seeking_healthcare'] * (len(eligible_adults))),
#                                                         replace=False)
#             treated_adults_idx = self.module.rng.choice(seeking_healthcare_adults,
#                                                  size=int(params['prob_sent_to_lab_test_adults'] * (len(seeking_healthcare_adults))),
#                                                  replace=False).tolist()
#         else:
#             treated_adults_idx = []
#
#         treated_idx = treated_children_idx + treated_adults_idx
#
#         if len(treated_idx) > 0:  # treat those with confirmed diagnosis
#             for person_id in treated_idx:
#                 self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
#
#             print("Number of treated due to HSI: " + str(len(treated_idx)))
#         else:
#             print("No one seeked treatment")
#             logger.debug('This is SchistoInfectionEvent, no one got treated with PZQ.')


class SchistoHistoricalMDAEvent(Event, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the historical MDA coverage
    """
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto)

    def apply(self, population):
        print("Historical MDA is happening now!")
        df = self.sim.population.props
        year = self.sim.date.year

        assert year in [2015, 2016, 2017, 2018], "No historical coverage data for this year"

        treated_idx_PSAC = self.assign_historical_MDA_coverage(population, year, 'PSAC')
        treated_idx_SAC = self.assign_historical_MDA_coverage(population, year, 'SAC')
        treated_idx_Adults = self.assign_historical_MDA_coverage(population, year, 'Adults')

        print("PSAC treated in MDA: " + str(len(treated_idx_PSAC)))
        print("SAC treated in MDA: " + str(len(treated_idx_SAC)))
        print("Adults treated in MDA: " + str(len(treated_idx_Adults)))

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults

        # people administered PZQ in MDA but in the Latent period will get the pill but won't be cured
        # similarly susceptibles will get the pill but nothing will happen
        # The infected will get cured immediately
        infected_idx = df.index[(df.is_alive) & (df.ss_is_infected == 'Infected')]
        MDA_treated = list(set(treated_idx) & set(infected_idx))  # intersection of infected & given a PZQ, so effectively cured

        for person_id in MDA_treated:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
        # count how many PZQ tablets were distributed
        PZQ_tablets_used = len(treated_idx)  # just in this round of MDA
        print("Year " + str(year) + ", PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))
        print("All cured in MDA: " + str(len(MDA_treated)))

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

        for distr in districts:
            coverage_distr = coverage[distr]  # get a correct value from the pd.Series
            eligible = df.index[(df['district_of_residence'] == distr) &
                                (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            if len(eligible):
                MDA_idx_distr = self.module.rng.choice(eligible,
                                                       size=int(coverage_distr * (len(eligible))), replace=False)
                MDA_idx = MDA_idx + MDA_idx_distr.tolist()

        return MDA_idx


class SchistoPrognosedMDAEvent(RegularEvent, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the proposed MDA coverage
    """
    def __init__(self, module, freq, district):
        super().__init__(module, frequency=DateOffset(months=freq))
        self.district = district
        assert isinstance(module, Schisto)

    def apply(self, population):
        district = self.district
        print("Prognosed MDA is happening now for district", district, "!")
        df = self.sim.population.props

        treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, district, 'PSAC')
        treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, district, 'SAC')
        treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, district, 'Adults')

        print("PSAC treated in MDA: " + str(len(treated_idx_PSAC)))
        print("SAC treated in MDA: " + str(len(treated_idx_SAC)))
        print("Adults treated in MDA: " + str(len(treated_idx_Adults)))

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults

        # people administered PZQ in MDA but in the Latent period will get the pill but won't be cured
        # similarly susceptibles will get the pill but nothing will happen
        # The infected will get cured immediately
        infected_idx = df.index[(df.is_alive) & (df.ss_is_infected == 'Infected')]
        MDA_treated = list(
            set(treated_idx) & set(infected_idx))  # intersection of infected & given a PZQ, so effectively cured

        for person_id in MDA_treated:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
        # count how many PZQ tablets were distributed
        PZQ_tablets_used = len(treated_idx)  # just in this round of MDA
        print("Year " + str(self.sim.date.year) + ", PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))
        print("All cured in MDA: " + str(len(MDA_treated)))

    def assign_prognosed_MDA_coverage(self, population, district, age_group):
        """Assign coverage of MDA program to chosen age_group. The same coverage for every district.

          :param district: district for which the MDA coverage is required
          :param population: population
          :param age_group: 'SAC', 'PSAC', 'Adults'
          :returns MDA_idx: indices of people that will be administered PZQ in the MDA program
          """

        df = population.props
        params = self.module.parameters
        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
        param_str = 'MDA_prognosed_' + age_group

        coverage = params[param_str]
        coverage_distr = coverage[district]

        eligible = df.index[(df.is_alive) & (df['district_of_residence'] == district) & (df['age_years'].between(age_range[0], age_range[1]))].tolist()
        MDA_idx = []
        if len(eligible):
            MDA_idx = self.module.rng.choice(eligible, size=int(coverage_distr * (len(eligible))), replace=False).tolist()

        return MDA_idx


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a logging event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------
class SchistoDALYsLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the DALYs per year
        """
        # run this event every year
        super().__init__(module, frequency=DateOffset(months=12))
        assert isinstance(module, Schisto)

    def create_logger(self, population, age_group):
        df = population.props

        age_range = map_age_groups(age_group)  # returns a tuple
        df_age = df[((df.is_alive) & (df['age_years'].between(age_range[0], age_range[1])))].copy()
        DALY_so_far = df_age['ss_cumulative_DALYs'].values.sum()
        log_string = '%s|' + 'DALY_' + age_group + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'DALY_cumulative': DALY_so_far,
                    })

        # TODO: substract the previous years' DALYs to get just the new ones
        # TODO: the unfinished infections won't be logged this year

    def apply(self, population):
        self.create_logger(population, 'PSAC')
        self.create_logger(population, 'SAC')
        self.create_logger(population, 'Adults')
        self.create_logger(population, 'All')


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
                        'Latent': count_states['Latent'],
                        'Infectious': count_states['Infected'],
                        'Infected': count_states['infected_any'],
                        'Prevalence': tot_prevalence,
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

        count_states = {'Non-infected': 0, 'Latent': 0, 'Infected': 0}
        count_states.update(df_age.ss_is_infected.value_counts().to_dict())  # this will get counts of non-infected, latent and infectious individuals
        count_states.update({'infected_any': count_states['Latent']
                                                       + count_states['Infected']})
        count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})

        return count_states

    def apply(self, population):

        self.create_logger(population, 'PSAC')
        self.create_logger(population, 'SAC')
        self.create_logger(population, 'Adults')
        self.create_logger(population, 'All')
