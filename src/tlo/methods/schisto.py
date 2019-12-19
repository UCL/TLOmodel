import logging

import numpy as np
import pandas as pd
import os
from pathlib import Path


from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   utility functions
# ---------------------------------------------------------------------------------------------------------


def get_params_for_neg_bin(dispersion, mean):
    """This functions return params to be used by the numpy negative binomial distribution,
    as it does not directly accept parametrisation by mean and dispersion"""
    r = dispersion
    var = mean + 1 / r * mean ** 2
    p = (var - mean) / var
    return r, 1-p


def draw_worms(worms_total, rates):
    """ This function splits up worms_total into people bins according to neg binomial distribution with
    dispersion parameter k. It splits only 25% of all worms, as a lot of the carcearie will stay in the water

    :param rates: harbouring rates used for Poisson distribution, drawn from Gamma, multiplied by contact rate per age group
    :param worms_total: total size of the reservoir of infectious material
    :return wormsies: array of numbers of new worms for each of people-persons
    """
    n = len(rates)
    if worms_total == 0:
        return np.zeros(n)
    # worms_total *= 0.05  # THIS IS PLUCKED OUT OF AIR
    worms_total = worms_total / len(rates)  # get a mean worm burden
    worms_total *= (1 + (1 - np.exp(-0.005)) * worms_total / 0.2) ** (-1.2)
    rates = list(rates)
    wormsies = np.asarray([np.random.poisson(x * worms_total, 1)[0] for x in rates]).astype(int)
    return wormsies


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
        'prevalence_2010': Parameter(Types.REAL, 'Initial prevalence in 2010'),
        'reservoir_2010': Parameter(Types.REAL, 'Initial reservoir of infectious material per district in 2010'),
        'prob_infection': Parameter(Types.REAL, 'Probability that a susceptible individual becomes infected'),
        'rr_PSAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age under 5 yo'),
        'rr_SAC': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age 5 - 14 yo'),
        'rr_Adults': Parameter(Types.REAL, 'Relative risk of aquiring infections due to age above 14 yo'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'symptoms_haematobium': Parameter(Types.DATA_FRAME, 'Symptoms for S. Haematobium infection'),
        'symptoms_mansoni': Parameter(Types.DATA_FRAME, 'Symptoms for S. Mansoni infection'),

        # new transmission model
        'eggs_per_worm_extracted': Parameter(Types.REAL, 'Number of eggs per mature female worm extracted to the env'),
        'neg_bin_k': Parameter(Types.REAL, 'Parameter k of negative binomial distribution of new worms acquired'),
        'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
        'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
        'high_intensity_threshold': Parameter(Types.REAL,
                                              'Threshold of worm burden indicating high intensity infection'),
        'low_intensity_threshold': Parameter(Types.REAL,
                                              'Threshold of worm burden indicating low intensity infection'),

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
        'delay_till_hsi_a_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to schisto test, start'),
        'delay_till_hsi_b_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again after not being sent to schisto test, end'),

        'prob_seeking_healthcare': Parameter(Types.REAL,
                                             'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test_children': Parameter(Types.REAL,
                                           'Probability that an infected child gets sent to urine or stool lab test'),
        'prob_sent_to_lab_test_adults': Parameter(Types.REAL,
                             'Probability that an infected adults gets sent to urine or stool lab test'),
        'PZQ_efficacy': Parameter(Types.REAL, 'Efficacy of prazinquantel'),
        'symptoms_mapped_for_hsb': Parameter(Types.REAL,
                                             'Symptoms to which the symptoms assigned in the module are mapped for the HSB module'),

        'MDA_coverage_PSAC': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for PSAC'),
        'MDA_coverage_SAC': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for SAC'),
        'MDA_coverage_Adults': Parameter(Types.DATA_FRAME, 'Probability of being administered PZQ in the MDA for Adults'),

        'MDA_prognosed_freq': Parameter(Types.DATA_FRAME, 'Prognosed MDA frequency in months'),
        'MDA_prognosed_PSAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in PSAC'),
        'MDA_prognosed_SAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in SAC'),
        'MDA_prognosed_Adults': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in Adults'),
    }

    PROPERTIES = {
        'ss_infection_status': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Low-infection', 'High-infection']),
        'ss_aggregate_worm_burden': Property(
            Types.REAL, 'Number of mature worms in the individual'),
        'ss_scheduled_increase_in_worm_burden': Property(
            Types.REAL, 'Number of mature worms that will mature when the latency period ends'),
        'ss_schedule_matured_worms': Property(
            Types.DATE, 'Scheduled date of maturation of the new worms'),
        'ss_haematobium_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Haematobium infection'),  # actually might also be np.nan
        'ss_mansoni_specific_symptoms': Property(
            Types.LIST, 'Symptoms for S. Mansoni infection'),  # actually might also be np.nan
        'ss_scheduled_hsi_date': Property(
            Types.DATE, 'Date of scheduled seeking healthcare'),
        'ss_onset_of_symptoms_date': Property(Types.DATE, 'Date of onset of symptoms'),
        'ss_clinical_days_in_last_year': Property(Types.REAL, 'Cumulative number of symptomatic days in last year'),
        'ss_cumulative_DALYs': Property(
            Types.REAL, 'Cumulative DALYs due to schistosomiasis symptoms'),
        'ss_harbouring_rate': Property(Types.REAL, 'Rate of harbouring new worms (Poisson), drawn from gamma distribution')
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

        # new transmission model
        params['eggs_per_worm_extracted'] = self.param_list.loc['eggs_per_worm_extracted', 'Value']
        params['neg_bin_k'] = self.param_list.loc['neg_bin_k', 'Value']
        params['beta_PSAC'] = self.param_list.loc['beta_PSAC', 'Value']
        params['beta_SAC'] = self.param_list.loc['beta_SAC', 'Value']
        params['beta_Adults'] = self.param_list.loc['beta_Adults', 'Value']
        params['worms_fecundity'] = self.param_list.loc['worms_fecundity', 'Value']
        params['high_intensity_threshold'] = self.param_list.loc['high_intensity_threshold', 'Value']
        params['low_intensity_threshold'] = self.param_list.loc['low_intensity_threshold', 'Value']

        # HSI and treatment params
        params['delay_till_hsi_a'] = self.param_list.loc['delay_till_hsi_a', 'Value']
        params['delay_till_hsi_b'] = self.param_list.loc['delay_till_hsi_b', 'Value']
        params['delay_till_hsi_a_repeated'] = self.param_list.loc['delay_till_hsi_a_repeated', 'Value']
        params['delay_till_hsi_b_repeated'] = self.param_list.loc['delay_till_hsi_b_repeated', 'Value']
        params['prob_seeking_healthcare'] = self.param_list.loc['prob_seeking_healthcare', 'Value']
        params['prob_sent_to_lab_test_children'] = self.param_list.loc['prob_sent_to_lab_test_children', 'Value']
        params['prob_sent_to_lab_test_adults'] = self.param_list.loc['prob_sent_to_lab_test_adults', 'Value']
        params['PZQ_efficacy'] = self.param_list.loc['PZQ_efficacy', 'Value']

        # baseline prevalence
        params['schisto_initial_prev'] = workbook['Prevalence_2010']
        self.schisto_initial_prev.set_index("District", inplace=True)
        params['prevalence_2010'] = self.schisto_initial_prev.loc[:, 'Prevalence']

        # baseline reservoir size
        params['schisto_initial_reservoir'] = workbook['Reservoir_2010']
        self.schisto_initial_reservoir.set_index("District", inplace=True)
        params['reservoir_2010'] = self.schisto_initial_reservoir.loc[:, 'Reservoir']

        # symptoms prevalence
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

        # DALY weights
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

    def change_paramater_value(self, parameter_name, new_value):
        self.parameters[parameter_name] = new_value

    def initialise_population(self, population):
        """Set our property values for the initial population.

        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individiuals
        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['ss_infection_status'] = 'Non-infected'
        df['ss_aggregate_worm_burden'] = 0
        df['ss_scheduled_increase_in_worm_burden'] = 0
        df['ss_schedule_matured_worms'] = pd.NaT
        df['ss_haematobium_specific_symptoms'] = np.nan
        df['ss_mansoni_specific_symptoms'] = np.nan
        df['ss_scheduled_hsi_date'] = pd.NaT
        df['ss_cumulative_DALYs'] = 0
        df['ss_onset_of_symptoms_date'] = pd.NaT
        df['ss_clinical_days_in_last_year'] = 0
        df['ss_haematobium_specific_symptoms'] = df['ss_haematobium_specific_symptoms'].astype(object)
        df['ss_harbouring_rate'] = self.rng.gamma(0.2, size=len(df.index))

        # # draw the harbouring rates
        # self.draw_harbouring_rates(population, 'PSAC')
        # self.draw_harbouring_rates(population, 'SAC')
        # self.draw_harbouring_rates(population, 'Adults')

        # assign initial worm burden
        self.assign_initial_worm_burden(population, 'PSAC')
        self.assign_initial_worm_burden(population, 'SAC')
        self.assign_initial_worm_burden(population, 'Adults')

        # assing infection statuses
        df['ss_infection_status'] = df['ss_aggregate_worm_burden'].apply(lambda x: self.intensity_of_infection(x))

        # # assign s. heamatobium symptoms
        # inf_haem_idx = df[df['ss_infection_status'] == 'Infected'].index.tolist()
        # self.assign_symptoms_intial(population, inf_haem_idx, 'haematobium')

    def draw_harbouring_rates(self, population, age_group):
        df = population.props
        age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
        alpha = 0.2
        indices = df.index[(df['age_years'].between(age_range[0], age_range[1]))].tolist()
        alphas = self.rng.gamma(alpha, size=len(indices))
        df.loc[indices, 'ss_harbouring_rate'] = alphas

    def intensity_of_infection(self, agg_wb):
        params = self.parameters
        if agg_wb > params['high_intensity_threshold']:
            return 'High-infection'
        if agg_wb > params['low_intensity_threshold']:
            return 'Low-infection'
        return 'Non-infected'

    def assign_initial_worm_burden(self, population, age_group):
        """Assign initial 2010 prevalence of schistosomiasis infections
        This will depend on a district and age group.

        :param population: population
        :param age_group: 'SAC', 'PSAC', 'Adults'
        """
        assert age_group in ['SAC', 'PSAC', 'Adults'], "Incorrect age group"

        df = population.props
        params = self.parameters
        districts = df.district_of_residence.unique().tolist()

        # age_range = map_age_groups(age_group)  # returns a tuple (a,b) a <= age_group <= b
        # prevalence = params["prevalence_2010"]  # this is a pd.Series not a single value
        reservoir = params['reservoir_2010'] * len(df.index)

        # rr = params['rr_' + age_group]

        # pd.Series.between is by default inclusive of the edges
        for distr in districts:
            # prevalence_distr = prevalence[distr]  # get a correct value from the pd.Series
            # prevalence_distr = prevalence_distr * rr
            # eligible = df.index[(df['district_of_residence'] == distr) &
            #                     (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            eligible = df.index[df['district_of_residence'] == distr].tolist()
            reservoir_distr = reservoir[distr] * len(eligible)
            contact_rates = pd.Series(1, index=eligible)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                params_str = 'beta_' + age_group
                age_range = map_age_groups(age_group)
                in_the_age_group = df.index[(df['district_of_residence'] == distr) &
                                    (df['age_years'].between(age_range[0], age_range[1]))].tolist()
                multiplier = params[params_str]  # Beta(age_group)
                contact_rates.loc[in_the_age_group] *= multiplier

            if len(eligible):
                harbouring_rates = df.loc[eligible, 'ss_harbouring_rate'].values
                rates = np.multiply(harbouring_rates, contact_rates)
                # infected_idx = self.rng.choice(eligible, size=int(prevalence_distr * (len(eligible))), replace=False)
                # assign a worm burden
                df.loc[eligible, 'ss_aggregate_worm_burden'] = draw_worms(reservoir_distr, rates)

    def assign_symptoms_intial(self, population, eligible_idx, inf_type):
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

    def get_prevalence(self, population):
        df = population.props
        total_infected = len(df[(df['is_alive']) & (df['ss_infection_status'] == 'Non-infected')].index)
        total_pop = len(df[df['is_alive']].index)
        prevalence = total_infected / total_pop
        return prevalence

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add the basic events of infection
        sim.schedule_event(SchistoInfectionWormBurdenEvent(self), sim.date + DateOffset(months=1))

        # # schedule historical MDA to happen once per year in July (4 events)
        # for historical_mda_year in [2015, 2016, 2017, 2018]:
        #     if historical_mda_year >= sim.date.year:
        #         sim.modules['HealthSystem'].schedule_hsi_event(HSI_SchistoHistoricalMDAEvent(self),
        #                                                         priority=0,
        #                                                         topen=pd.Timestamp(year=historical_mda_year, month=7, day=1, hour=23),
        #                                                         tclose=pd.Timestamp(year=historical_mda_year, month=7, day=1, hour=23) + DateOffset(
        #                                                             weeks=4))
        #
        # # schedule prognosed MDA programmes for every district
        # for district in sim.population.props.district_of_residence.unique().tolist():
        #     freq = self.parameters['MDA_frequency_prognosed'][district]
        #     if freq > 0:  # frequency 0 means no need for MDA, because prevalence there is always 0
        #         sim.schedule_event(SchistoPrognosedMDAEvent(self, freq, district),
        #                            pd.Timestamp(year=2019, month=6, day=1, hour=12) + DateOffset(months=0))
        #
        # # schedule a change in a parameter
        # sim.schedule_event(SchistoChangeParameterEvent(self, 'prob_sent_to_lab_test_adults', 0.6),
        #                    pd.Timestamp(year=2019, month=1, day=1))

        # add an event to log to screen
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + DateOffset(months=0))
        sim.schedule_event(SchistoDALYsLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props
        params = self.parameters
        # Assign the default for a newly born child
        df.at[child_id, 'ss_infection_status'] = 'Non-infected'
        df.at[child_id, 'ss_haematobium_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_mansoni_specific_symptoms'] = np.nan
        df.at[child_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.at[child_id, 'ss_cumulative_DALYs'] = 0
        df.at[child_id, 'ss_aggregate_worm_burden'] = 0
        df.at[child_id, 'ss_scheduled_increase_in_worm_burden'] = 0
        df.at[child_id, 'ss_schedule_matured_worms'] = pd.NaT
        df.at[child_id, 'ss_onset_of_symptoms_date'] = pd.NaT
        df.at[child_id, 'ss_clinical_days_in_last_year'] = 0
        df.at[child_id, 'ss_harbouring_rate'] = self.rng.gamma(0.24, size = 1)

    def report_daly_values(self):
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Schisto reporting my health values')

        df = self.sim.population.props

        # for now we only have haematobium infections anyway
        health_values = df.loc[df.is_alive, 'ss_haematobium_specific_symptoms'].apply(lambda x: self.add_DALYs_from_symptoms(x))

        # the mapping above included counting DALYs for people with 'scheduled' symptoms. i.e. in Latent period
        # we want to calculate it only for people who are infectious
        health_values[df['ss_infection_status'] == 'Non-infected'] = 0
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


class SchistoChangeParameterEvent(Event, PopulationScopeEventMixin):
    def __init__(self, module, param_name, new_value):
        """
        This event can be used to update a chosen parameter value at a specific time. The new value will be used in
        the simulation until the simulation is finished or it is updated again.
        :param module: the module that created this event
        :param param_name: name of the parameter we want to update
        :param new_value: new value of the chosen parameter
        """
        super().__init__(module)
        assert isinstance(module, Schisto)
        self.param_name = param_name
        self.new_value = new_value

    def apply(self, population):
        print("Changing the parameter", self.param_name, "from value", self.module.parameters[self.param_name], "to", self.new_value)
        self.module.change_paramater_value(self.param_name, self.new_value)


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    Using Worm Burden and Reservoir of Infectious Material - see write up
    This does not use the prevalence of schistosomiasis infections per district
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

        ######################## increase worm burden of people in each district  ######################################
        for distr in districts:
            df_distr_indx = df.index[(df['is_alive']) & (df['district_of_residence'] == distr)].tolist()
            # df_distr_indx = df[(df["district_of_residence"] == distr) & (df.is_alive)].index.to_list()
            new_worms = self.increase_worm_burden_distr(population, distr)
            df.loc[df_distr_indx, 'ss_scheduled_increase_in_worm_burden'] = new_worms

        worm_burden_increased = df[df['ss_scheduled_increase_in_worm_burden'] > 0].index  # these are the people that acquire new worms

        ############## schedule the time of new worms becoming mature #######################################
        days_till_maturation = self.module.rng.uniform(30, 55, size=len(worm_burden_increased)).astype(int)
        days_till_maturation = pd.to_timedelta(days_till_maturation, unit='D')
        df.loc[worm_burden_increased, 'ss_schedule_matured_worms'] = self.sim.date + days_till_maturation

        for person_index in worm_burden_increased:
            new_worms = df.loc[person_index, 'ss_scheduled_increase_in_worm_burden']
            maturation_event = SchistoMatureWorms(self.module, person_id=person_index, new_worms=new_worms)
            self.sim.schedule_event(maturation_event,
                                    df.at[person_index, 'ss_schedule_matured_worms'])

    def get_expected_mean_worm_burden(self, population):
        k = self.module.parameters['neg_bin_k']
        prevalence = self.module.get_prevalence(population)
        expMWB = (1 - prevalence) ** (-1/k) * k - k  # this formula is explained in the write up
        return expMWB

    def get_new_worm_burden_to_distribute(self, population):
        df = population.props
        expected_WB = self.get_expected_mean_worm_burden(population)
        previous_WB = df['ss_aggregate_worm_burden'].values.mean()
        increase_MWB = expected_WB - previous_WB
        return increase_MWB

    def increase_worm_burden_distr(self, population, district):
        """Calculates the new Total Reservoir of infectious material in district and randomly assigns
        newly acquired worms to each individual in the district, sampled from a negative binomial distribution

         :param population: population
         :param district: one of the 32 Malawi districts
         :returns allocated_new_worms: new worms acquired by the people in the given district
         """

        df = population.props
        params = self.module.parameters
        df_distr = df[(df["district_of_residence"] == district) & (df.is_alive)].copy()

        if df_distr.shape[0]:  # if there are any rows in the dataframe, so there are alive people in the district
            ############## calculate the size of infectious material reservoir ###################
            total_worms = df_distr['ss_aggregate_worm_burden'].values
            # eggs = (total_worms / 2) * params['eggs_per_worm_extracted']  # assuming 1/2 of all worms is female
            # dens_dep_multiplier = 1 + (1 - np.exp(-params['worms_fecundity'])) * total_worms
            # dens_dep_multiplier = dens_dep_multiplier ** (-1 * (params['neg_bin_k'] + 1))
            # eggs *= dens_dep_multiplier
            # reservoir = eggs.sum()
            reservoir = total_worms.sum()
            # reservoir = df_distr['ss_aggregate_worm_burden'].values.sum()
            ###### assign appropriate Beta parameters ######################
            ss_new_eggs = pd.Series(1, index=df_distr.index.tolist())

            for age_group in ['PSAC', 'SAC', 'Adults']:
                params_str = 'beta_' + age_group
                age_range = map_age_groups(age_group)
                multiplier = params[params_str]  # Beta(age_group)
                ss_new_eggs.loc[df.age_years.between(age_range[0], age_range[1])] *= multiplier

            #  draw new worms from a negative binomial distribution
            harbouring_rates = df_distr['ss_harbouring_rate'].values
            rates = np.multiply(harbouring_rates, ss_new_eggs)
            allocated_new_worms = draw_worms(reservoir, rates)
            prob_of_establishment = list(np.exp(-params['worms_fecundity'] * df_distr['ss_aggregate_worm_burden'].values))
            prob_of_establishment = [self.module.rng.choice([1, 0], p=[x, 1-x], size=1)[0] for x in prob_of_establishment]
            prob_of_establishment = np.asarray(prob_of_establishment)
            allocated_new_worms = np.multiply(allocated_new_worms, prob_of_establishment)
            # allocated_new_worms *= ss_new_eggs

            if allocated_new_worms.sum() > reservoir:
                print("More worms drawn then should have been..")
            return list(allocated_new_worms)


class SchistoMatureWorms(Event, IndividualScopeEventMixin):
    """Increases the aggregate worm burden of an individual upon maturation of the worms
    """
    def __init__(self, module, person_id, new_worms):
        super().__init__(module, person_id=person_id)
        self.new_worms = new_worms
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        df.loc[person_id, 'ss_aggregate_worm_burden'] += self.new_worms * params['eggs_per_worm_extracted']

        if df.loc[person_id, 'is_alive']:
            if df.loc[person_id, 'ss_infection_status'] != 'High-infection':
                if df.loc[person_id, 'ss_aggregate_worm_burden'] >= params['high_intensity_threshold']:
                        df.loc[person_id, 'ss_infection_status'] = 'High-infection'
                        develop_symptoms = SchistoDevelopSymptomsEvent(self.module, person_id=person_id)
                        self.sim.schedule_event(develop_symptoms, self.sim.date)  # happens immediately

                elif df.loc[person_id, 'ss_aggregate_worm_burden'] >= params['low_intensity_threshold']:
                    df.loc[person_id, 'ss_infection_status'] = 'Low-infection'


class SchistoDevelopSymptomsEvent(Event, IndividualScopeEventMixin):
    """Development of symptoms upon high intensity infection
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        # assign symptoms
        assert np.isnan(df.loc[person_id, 'ss_haematobium_specific_symptoms']), "This person already was symptomatic"
        symptoms = self.assign_symptoms('haematobium')
        df.at[person_id, 'ss_haematobium_specific_symptoms'] = symptoms
        df.loc[person_id, 'ss_onset_of_symptoms_date'] = self.sim.date
        # schedule Healthcare Seeking
        seeking_treatment_ahead = self.module.rng.uniform(params['delay_till_hsi_a'],
                                                          params['delay_till_hsi_b'],
                                                          size=1).astype(int)
        seeking_treatment_ahead = pd.to_timedelta(seeking_treatment_ahead, unit='D')
        df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead
        seek_treatment_event = HSI_SchistoSeekTreatment(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
                                                            priority=1,
                                                            topen=df.at[person_id, 'ss_scheduled_hsi_date'],
                                                            tclose=df.at[
                                                                       person_id, 'ss_scheduled_hsi_date'] + DateOffset(
                                                                weeks=4))

    def assign_symptoms(self, inf_type):
        """
        Assign symptoms to the person with high intensity infection.

        :param person_id: index of the individual
        :param inf_type: type of infection, haematobium or mansoni
        :return symptoms: np.nan if no symptom or a list of symptoms
        """
        assert inf_type in ['mansoni', 'haematobium'], "Incorrect infection type. Can't assign symptoms."

        params = self.module.parameters
        symptoms_dict = params['symptoms_' + inf_type].set_index('symptoms').to_dict()[
            'prevalence']  # create a dictionary from a df
        symptoms_exp = []
        for symptom in symptoms_dict.keys():
            prev = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
            is_experienced = self.module.rng.choice([True, False], 1, p=[prev, 1-prev])
            if is_experienced:
                symptoms_exp.append(symptom)
        if len(symptoms_exp):
            return symptoms_exp
        return np.nan


class SchistoTreatmentEvent(Event, IndividualScopeEventMixin):
    """Cured upon PZQ treatment through HSI or MDA (Infected -> Non-infected)
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters

        if df.loc[person_id, 'is_alive']:
            if isinstance(df.loc[person_id, 'ss_haematobium_specific_symptoms'], list):
                symptoms_duration = self.sim.date - df.loc[person_id, 'ss_onset_of_symptoms_date']
                symptoms_duration = int(symptoms_duration / np.timedelta64(1, 'D'))
                assert symptoms_duration >= 0, "Duration of the symptoms was negative!!!!"
                df.loc[person_id, 'ss_clinical_days_in_last_year'] += symptoms_duration
                symptoms = df.loc[person_id, 'ss_haematobium_specific_symptoms']
                df.loc[person_id, 'ss_cumulative_DALYs'] += self.calculate_DALY_per_infection(symptoms_duration, symptoms)
            df.loc[person_id, 'ss_haematobium_specific_symptoms'] = np.nan
            df.loc[person_id, 'ss_mansoni_specific_symptoms'] = np.nan
            df.loc[person_id, 'ss_scheduled_hsi_date'] = pd.NaT
            df.loc[person_id, 'ss_aggregate_worm_burden'] *= params['PZQ_efficacy']  # decreasing number of worms
            df.loc[person_id, 'ss_onset_of_symptoms_date'] = pd.NaT
            if df.loc[person_id, 'ss_aggregate_worm_burden'] < params['low_intensity_threshold']:
                df.loc[person_id, 'ss_infection_status'] = 'Non-infected'
            elif df.loc[person_id, 'ss_aggregate_worm_burden'] < params['high_intensity_threshold']:
                df.loc[person_id, 'ss_infection_status'] = 'Low-infection'


    def calculate_DALY_per_infection(self, inf_duration, symptoms):
        dalys_weight = self.module.add_DALYs_from_symptoms(symptoms)
        DALY = (inf_duration / 30.0) * (dalys_weight / 12.0)  # inf_duration in days and weight given per year
        return DALY

# ---------------------------------------------------------------------------------------------------------
#   HSI EVENTS
# ---------------------------------------------------------------------------------------------------------


class HSI_SchistoSeekTreatment(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event of seeking treatment for a person with symptoms
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        df = self.sim.population.props

        if df.loc[person_id, 'age_years'] <= 15:
            the_appt_footprint['Under5OPD'] = 1
        else:
            the_appt_footprint['Over5OPD'] = 1

        the_accepted_facility_level = 0

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Schisto_Treatment_seeking'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        params = self.module.parameters

        # appt are scheduled and cannot be cancelled in the following situations:
        #   a) person has died
        #   b) the infection has been treated in MDA before the appt happened
        if ((df.loc[person_id, 'is_alive']) &
            (df.loc[person_id, 'ss_infection_status'] != 'Non-infected') &
            (df.loc[person_id, 'ss_scheduled_hsi_date'] <= self.sim.date)):

            # check if a person is a child or an adult and assign prob of being sent to schisto test (hence being cured)
            if df.loc[person_id, 'age_years'] <= 15:
                prob_test = params['prob_sent_to_lab_test_children']
            else:
                prob_test = params['prob_sent_to_lab_test_adults']

            sent_to_test = self.module.rng.choice([True, False], p=[prob_test, 1-prob_test])

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
                    # patient is cured
                    self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
                else:
                    logger.debug('ItemsCode1 is not available, so can' 't use it.')
            else:  # person seeked treatment but was not sent to test
                # schedule another Seeking Treatment event for that person
                seeking_treatment_ahead_repeated = int(self.module.rng.uniform(params['delay_till_hsi_a_repeated'],
                                                                   params['delay_till_hsi_b_repeated']))
                seeking_treatment_ahead_repeated = pd.to_timedelta(seeking_treatment_ahead_repeated, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead_repeated

                seek_treatment_repeated = HSI_SchistoSeekTreatment(self.module, person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_repeated,
                                                                    priority=1,
                                                                    topen=df.at[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.at[person_id, 'ss_scheduled_hsi_date'] + DateOffset(weeks=4))

    def did_not_run(self):
        return True


# ---------------------------------------------------------------------------------------------------------
#   MASS-DRUG ADMINISTRATION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_SchistoHistoricalMDAEvent(HSI_Event, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the historical MDA coverage
    """
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto)

        self.TREATMENT_ID = 'Schisto_MDA_historical_data'

    def apply(self, population, squeeze_factor):
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
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        MDA_treated = treated_idx
        # # people administered PZQ in MDA but in the Latent period will get the pill but won't be cured
        # # similarly susceptibles will get the pill but nothing will happen
        # # The infected get cured immediately
        # infected_idx = df.index[(df.is_alive) & (df.ss_infection_status != 'Non-infected')]
        # MDA_treated = list(set(treated_idx) & set(infected_idx))  # intersection of infected & given a PZQ, so effectively cured

        for person_id in MDA_treated:
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            items_code1 = \
                pd.unique(
                    consumables.loc[
                        consumables['Items'] == "Praziquantel 600mg_1000_CMST", 'Item_Code'])[0]
            the_cons_footprint = {'Intervention_Package_Code': [], 'Item_Code': [{items_code1: 1}]}
            # self.sim.modules['HealthSystem'].request_consumables(
            #             hsi_event=self, cons_req_as_footprint=the_cons_footprint, to_log=True)
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
            eligible = df.index[(df['is_alive']) & (df['district_of_residence'] == distr) &
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
        # print("Prognosed MDA is happening now for district", district, "!")
        df = self.sim.population.props

        treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, district, 'PSAC')
        treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, district, 'SAC')
        treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, district, 'Adults')

        # print("PSAC treated in MDA: " + str(len(treated_idx_PSAC)))
        # print("SAC treated in MDA: " + str(len(treated_idx_SAC)))
        # print("Adults treated in MDA: " + str(len(treated_idx_Adults)))

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        MDA_treated = treated_idx
        # # people administered PZQ in MDA but in the Latent period will get the pill but won't be cured
        # # similarly susceptibles will get the pill but nothing will happen
        # # The infected will get cured immediately
        # infected_idx = df.index[(df.is_alive) & (df.ss_infection_status == 'Infected')]
        # MDA_treated = list(
        #     set(treated_idx) & set(infected_idx))  # intersection of infected & given a PZQ, so effectively cured

        for person_id in MDA_treated:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
        # # count how many PZQ tablets were distributed
        # PZQ_tablets_used = len(treated_idx)  # just in this round of MDA
        # print("Year " + str(self.sim.date.year) + ", PZQ tablets used in this MDA round: " + str(PZQ_tablets_used))
        # print("All cured in MDA: " + str(len(MDA_treated)))

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

        eligible = df.index[(df.is_alive) & (df['district_of_residence'] == district)
                            & (df['age_years'].between(age_range[0], age_range[1]))].tolist()
        MDA_idx = []
        if len(eligible):
            MDA_idx = self.module.rng.choice(eligible, size=int(coverage_distr * (len(eligible))), replace=False).tolist()

        return MDA_idx


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class SchistoDALYsLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the DALYs per year
        Note: it only makes sense to look at the ALL population, otherwise the cumulative DALYs might be decreasing due
        to people moving to the higher class age group, i.e. PSAC -> SAC, SAC -> Adults
        """
        # run this event every year
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, Schisto)

    def create_logger(self, population):
        df = population.props
        DALY_so_far = df['ss_cumulative_DALYs'].values.sum()
        log_string = '%s|DALY_All|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'DALY_cumulative': DALY_so_far,
                    })

        # TODO: the unfinished infections won't be logged this year

    def apply(self, population):
        self.create_logger(population)


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
                        'Non_infected': count_states['Non-infected'],
                        'Low_infections': count_states['Low-infection'],
                        'High-infections': count_states['High-infection'],
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

        count_states = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
        count_states.update(df_age.ss_infection_status.value_counts().to_dict())  # this will get counts of non-infected, latent and infectious individuals
        count_states.update({'infected_any': count_states['Low-infection']
                                                       + count_states['High-infection']})
        count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})

        return count_states

    def apply(self, population):

        self.create_logger(population, 'PSAC')
        self.create_logger(population, 'SAC')
        self.create_logger(population, 'Adults')
        self.create_logger(population, 'All')
