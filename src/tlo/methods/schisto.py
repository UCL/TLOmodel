import numpy as np
import pandas as pd
from pathlib import Path

from tlo import logging
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   utility functions
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
    age_map = {'PSAC': (0, 4), 'SAC': (5, 14), 'Adults': (15, 120), 'All': (0, 120)}
    return age_map[age_group]


def count_days_this_year(date_end, date_start):
    """Used for calculating PrevalentYears this year (that is the year of date_end)
    If the start_date is in the previous years only gives the number of days from
     the beginning of the year till date_end, if it start_date is the same year then just gives the time elapsed"""
    year = date_end.year
    if date_start.year < year:
        date_start = pd.Timestamp(year=year, month=1, day=1)
    duration = date_end - date_start
    duration = int(duration / np.timedelta64(1, 'D'))
    assert duration >= 0, "Duration is negative!"
    return duration


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
    Required for the Schisto_Haematobium and Schisto_Mansoni modules.
    Governs the properties that are common to both of the Schisto modules.
    Also schedules HSI events, Treatment and MDA event.
    attribute mda_execute: TRUE (default) /FALSE determines whether MDA events should be scheduled or not
    """

    def __init__(self, name=None, resourcefilepath=None, mda_execute=True):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.prefix = 'ss'
        self.mda_execute = mda_execute

    PARAMETERS = {
        # health system interaction
        'delay_till_hsi_a': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, start'),
        'delay_till_hsi_b': Parameter(Types.REAL, 'Time till seeking healthcare since the onset of symptoms, end'),
        'delay_till_hsi_a_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again '
                                               'after not being sent to schisto test, start'),
        'delay_till_hsi_b_repeated': Parameter(Types.REAL,
                                               'Time till seeking healthcare again '
                                               'after not being sent to schisto test, end'),

        'prob_seeking_healthcare': Parameter(Types.REAL,
                                             'Probability that an infected individual visits a healthcare facility'),
        'prob_sent_to_lab_test_children': Parameter(Types.REAL,
                                                    'Probability that infected child gets sent to lab test'),
        'prob_sent_to_lab_test_adults': Parameter(Types.REAL,
                                                  'Probability that an infected adults gets sent to lab test'),

        # MDA parameters
        'MDA_coverage_PSAC': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for PSAC'),
        'MDA_coverage_SAC': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for SAC'),
        'MDA_coverage_Adults': Parameter(Types.DATA_FRAME, 'Probability of getting PZQ in the MDA for Adults'),

        'MDA_prognosed_freq': Parameter(Types.DATA_FRAME, 'Prognosed MDA frequency in months'),
        'MDA_prognosed_PSAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in PSAC'),
        'MDA_prognosed_SAC': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in SAC'),
        'MDA_prognosed_Adults': Parameter(Types.DATA_FRAME, 'Prognosed coverage of MDA in Adults'),
    }

    PROPERTIES = {
        'ss_scheduled_hsi_date': Property(
            Types.DATE, 'Date of scheduled seeking healthcare'),
        'ss_last_PZQ_date': Property(Types.DATE, 'Day of the most recent treatment with PZQ')
    }

    def read_parameters(self, data_folder):
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx', sheet_name=None)
        # workbook = pd.read_excel(os.path.join(self.resourcefilepath,
        #                                       'ResourceFile_Schisto.xlsx'), sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['Parameters']
        self.param_list.set_index("Parameter", inplace=True)

        # HSI and treatment params
        params['delay_till_hsi_a'] = self.param_list.loc['delay_till_hsi_a', 'Value']
        params['delay_till_hsi_b'] = self.param_list.loc['delay_till_hsi_b', 'Value']
        params['delay_till_hsi_a_repeated'] = self.param_list.loc['delay_till_hsi_a_repeated', 'Value']
        params['delay_till_hsi_b_repeated'] = self.param_list.loc['delay_till_hsi_b_repeated', 'Value']
        params['prob_seeking_healthcare'] = self.param_list.loc['prob_seeking_healthcare', 'Value']
        params['prob_sent_to_lab_test_children'] = self.param_list.loc['prob_sent_to_lab_test_children', 'Value']
        params['prob_sent_to_lab_test_adults'] = self.param_list.loc['prob_sent_to_lab_test_adults', 'Value']
        params['PZQ_efficacy'] = self.param_list.loc['PZQ_efficacy', 'Value']

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
        symptoms_dict = {'anemia': 258, 'fever': 262, 'hydronephrosis': 260, 'dysuria': 263,
                         'bladder_pathology': 264, 'diarrhoea': 259, 'vomit': 254, 'ascites': 261,
                         'hepatomegaly': 257}
        for symptom, weight in symptoms_dict.items():
            str = 'daly_wt_' + symptom
            params[str] = self.sim.modules['HealthBurden'].get_daly_weight(weight)
        params['daly_wt_haematuria'] = 0  # that's a very common symptom but no official DALY weight yet defined

    def change_paramater_value(self, parameter_name, new_value):
        """This function allows updating a parameter change at some point due to e.g. a behavioural change intervention,
        such as increasing awareness or e.g. improving the sensitivity of laboratory tests.
        Used by a SchistoChangeParameterEvent"""
        self.parameters[parameter_name] = new_value

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['ss_scheduled_hsi_date'] = pd.NaT
        df['ss_last_PZQ_date'] = pd.Timestamp(year=1900, month=1, day=1)  # for simplicity to avoid NaT later

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # sim.schedule_event(SchistoLoggingTotalEvent(self), sim.date + DateOffset(months=0))
        sim.schedule_event(SchistoPrevalentDaysLoggingEvent(self),
                           pd.Timestamp(year=sim.date.year, month=12, day=31, hour=23))

        if self.mda_execute:
            # schedule historical MDA to happen once per year in July (4 events)
            years_of_historical_mda = [25, 26, 27, 28]  # 25 years are required for the model to equilibrate
            years_of_historical_mda = [sim.date.year + el for el in years_of_historical_mda]
            for historical_mda_year in years_of_historical_mda:
                sim.schedule_event(SchistoHistoricalMDAEvent(self),
                                   pd.Timestamp(year=historical_mda_year, month=7, day=1, hour=23) +
                                   DateOffset(months=0))
            year_first_simulated_mda = years_of_historical_mda[-1] + 1
            # schedule prognosed MDA programmes for every district
            for district in sim.population.props.district_of_residence.unique().tolist():
                freq = self.parameters['MDA_frequency_prognosed'][district]
                if freq > 0:  # frequency 0 means no need for MDA, because prevalence there is always 0
                    sim.schedule_event(SchistoPrognosedMDAEvent(self, freq, district),
                                       pd.Timestamp(year=year_first_simulated_mda, month=6, day=1, hour=12) +
                                       DateOffset(months=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.at[child_id, 'ss_last_PZQ_date'] = pd.Timestamp(year=1900, month=1, day=1)

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


class Schisto_Haematobium(Module):
    """
    Schistosomiasis Haematobium module
    Models the transmission of the S.Haematobium infection & development of symptoms.
    Requires Schisto() module to be registered to work.
    """

    def __init__(self, name=None, resourcefilepath=None, symptoms_and_HSI=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.prefix = 'sh'
        assert symptoms_and_HSI in [True, False]
        self.symptoms_and_HSI = symptoms_and_HSI

    PARAMETERS = {
        'reservoir_2010': Parameter(Types.REAL, 'Initial reservoir of infectious material per district in 2010'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'symptoms': Parameter(Types.DATA_FRAME, 'Symptoms for S. Haematobium infection'),
        'gamma_alpha': Parameter(Types.REAL, 'Parameter alpha for Gamma distribution for harbouring rates'),
        'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
        'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
        'R0': Parameter(Types.REAL, 'Effective reproduction number, for the FOI'),
        'worm_lifespan': Parameter(Types.REAL, 'Lifespan of the worm in human host given in years'),
        'high_intensity_threshold': Parameter(Types.REAL,
                                              'Threshold of worm burden indicating high intensity infection'),
        'low_intensity_threshold': Parameter(Types.REAL, 'Threshold of worm burden indicating low intensity infection'),
        'high_intensity_threshold_PSAC': Parameter(Types.REAL,
                                                   'Threshold of worm burden indicating'
                                                   ' high intensity infection for PSAC'),
        'list_of_districts': Parameter(Types.LIST, 'List of the districts with the infection present')
    }

    PROPERTIES = {
        'sh_infection_status': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Low-infection', 'High-infection']),
        'sh_aggregate_worm_burden': Property(
            Types.INT, 'Number of mature worms in the individual'),
        'sh_symptoms': Property(
            Types.LIST, 'Symptoms for S. Haematobium infection'),  # actually might also be np.nan
        'sh_start_of_prevalent_period': Property(Types.DATE, 'Date of going from Non-infected to Infected'),
        'sh_start_of_high_infection': Property(Types.DATE, 'Date of going from entering state High-inf'),
        'sh_harbouring_rate': Property(Types.REAL,
                                       'Rate of harbouring new worms (Poisson), drawn from gamma distribution'),
        'sh_prevalent_days_this_year': Property(Types.INT, 'Cumulative days with infection in current year'),
        'sh_high_inf_days_this_year': Property(Types.INT,
                                               'Cumulative days with high-intensity infection in current year')
    }

    def read_parameters(self, data_folder):
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx', sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['Parameters']
        self.param_list.set_index("Parameter", inplace=True)

        params['delay_a'] = self.param_list.loc['delay_a', 'Value']
        params['delay_b'] = self.param_list.loc['delay_b', 'Value']
        params['beta_PSAC'] = self.param_list.loc['beta_PSAC', 'Value']
        params['beta_SAC'] = self.param_list.loc['beta_SAC', 'Value']
        params['beta_Adults'] = self.param_list.loc['beta_Adults', 'Value']
        params['worms_fecundity'] = self.param_list.loc['worms_fecundity_haematobium', 'Value']
        params['worm_lifespan'] = self.param_list.loc['lifespan_haematobium', 'Value']
        params['high_intensity_threshold'] = self.param_list.loc['high_intensity_threshold_haematobium', 'Value']
        params['low_intensity_threshold'] = self.param_list.loc['low_intensity_threshold_haematobium', 'Value']
        params['high_intensity_threshold_PSAC'] = \
            self.param_list.loc['high_intensity_threshold_haematobium_PSAC', 'Value']

        # baseline reservoir size and other district-related params (alpha and R0)
        params['schisto_initial_reservoir'] = workbook['District_Params_haematobium']
        self.schisto_initial_reservoir.set_index("District", inplace=True)
        params['reservoir_2010'] = self.schisto_initial_reservoir.loc[:, 'Reservoir']
        params['gamma_alpha'] = self.schisto_initial_reservoir.loc[:, 'alpha_value']
        params['R0'] = self.schisto_initial_reservoir.loc[:, 'R0_value']

        # symptoms prevalence
        params['symptoms'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'haematuria', 'hydronephrosis', 'dysuria', 'bladder_pathology'],
                'prevalence': [0.9, 0.3, 0.625, 0.083, 0.2857, 0.7857],
                'hsb_symptoms': ['other', 'fever', 'other', 'stomach_ache', 'stomach_ache', 'stomach_ache'],
            })

        # this is to be used if we want to model every district
        params['list_of_districts'] = \
            pd.read_csv(Path(self.resourcefilepath) /
                        'ResourceFile_District_Population_Data.csv').District.unique().tolist()
        # this was used for the analysis done in my thesis
        # params['list_of_districts'] = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

    def initialise_population(self, population):
        """Set our property values for the initial population.

        :param population: the population of individuals
        """
        assert 'Schisto' in self.sim.modules.keys(), "Module Schisto must be registered for Schisto_Haematobium to work"
        df = population.props

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        # df['sh_infection_status'] = 'Non-infected'
        df['sh_aggregate_worm_burden'] = 0
        df['sh_symptoms'] = np.nan
        df['sh_prevalent_days_this_year'] = 0
        df['sh_start_of_prevalent_period'] = pd.NaT
        df['sh_start_of_high_infection'] = pd.NaT
        df['sh_high_inf_days_this_year'] = 0
        df['sh_symptoms'] = df['sh_symptoms'].astype(object)

        # assign a harbouring rate
        self.assign_harbouring_rate(population)

        # assign initial worm burden
        self.assign_initial_worm_burden(population)

        # assign infection statuses
        df['sh_infection_status'] = \
            df.apply(lambda x: self.intensity_of_infection(x['age_years'], x['sh_aggregate_worm_burden']), axis=1)
        df['sh_infection_status'] = df['sh_infection_status'].astype('category')
        #  start of the prevalent period & start of high-intensity infections
        infected_idx = df.index[df['sh_infection_status'] != 'Non-infected']
        df.loc[infected_idx, 'sh_start_of_prevalent_period'] = self.sim.date
        high_infected_idx = df.index[df['sh_infection_status'] == 'High-infection']
        df.loc[high_infected_idx, 'sh_start_of_high_infection'] = self.sim.date

        if self.symptoms_and_HSI:
            # assign initial symptoms
            self.assign_symptoms_initial(population, high_infected_idx)
            # assign initial dates of seeking healthcare for people with symptoms
            symptomatic_idx = df.index[~df['sh_symptoms'].isna()]
            self.assign_hsi_dates_initial(population, symptomatic_idx)

    def assign_harbouring_rate(self, population):
        """Assign a harbouring rate to every individual, this happens with a district-related param"""
        df = population.props
        params = self.parameters
        for district in params['list_of_districts']:
            eligible = df.index[df['district_of_residence'] == district].tolist()
            hr = params['gamma_alpha'][district]
            df.loc[eligible, 'sh_harbouring_rate'] = self.rng.gamma(hr, size=len(eligible))

    def assign_initial_worm_burden(self, population):
        """Assign initial 2010 prevalence of schistosomiasis infections
        This will depend on a district and age group.
        """
        df = population.props
        params = self.parameters
        districts = params['list_of_districts']
        reservoir = params['reservoir_2010']

        for distr in districts:
            eligible = df.index[df['district_of_residence'] == distr].tolist()
            contact_rates = pd.Series(1, index=eligible)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = map_age_groups(age_group)
                in_the_age_group = \
                    df.index[(df['district_of_residence'] == distr) &
                             (df['age_years'].between(age_range[0], age_range[1]))].tolist()
                contact_rates.loc[in_the_age_group] *= params['beta_' + age_group]  # Beta(age_group)

            if len(eligible):
                harbouring_rates = df.loc[eligible, 'sh_harbouring_rate'].values
                rates = np.multiply(harbouring_rates, contact_rates)
                reservoir_distr = int(reservoir[distr] * len(eligible))
                # dsitribute a worm burden
                chosen = self.rng.choice(eligible, reservoir_distr, p=rates / rates.sum())
                unique, counts = np.unique(chosen, return_counts=True)
                worms_per_idx = dict(zip(unique, counts))
                df['sh_aggregate_worm_burden'].update(pd.Series(worms_per_idx))

        # schedule death of worms
        people_with_worms = df.index[df['sh_aggregate_worm_burden'] > 0]
        print('Initial people with worms:', len(people_with_worms))
        for person_id in people_with_worms:
            worms = df.loc[person_id, 'sh_aggregate_worm_burden']
            natural_death_of_worms = SchistoWormsNatDeath(self, person_id=person_id,
                                                          number_of_worms=worms)
            months_till_death = int(self.rng.uniform(1, params['worm_lifespan'] * 12 / 2))
            self.sim.schedule_event(natural_death_of_worms, self.sim.date + DateOffset(months=months_till_death))

    def draw_worms(self, worms_total, rates, district):
        """ This function generates random number of new worms drawn from Poisson distribution multiplied by
        a product of harbouring rate and exposure rate
        :param district: district name
        :param rates: harbouring rates used for Poisson distribution, drawn from Gamma,
        multiplied by contact rate per age group
        :param worms_total: total size of the reservoir of infectious material
        :return harboured_worms: array of numbers of new worms for each of the persons (len = len(rates))
        """
        params = self.parameters
        if worms_total == 0:
            return np.zeros(len(rates))
        rates = list(rates)
        R0 = params['R0'][district]
        worms_total *= R0
        harboured_worms = np.asarray([self.rng.poisson(x * worms_total, 1)[0] for x in rates]).astype(int)
        return harboured_worms

    def intensity_of_infection(self, age, agg_wb):
        params = self.parameters
        if age < 5:
            if agg_wb >= params['high_intensity_threshold_PSAC']:
                return 'High-infection'
        if agg_wb >= params['high_intensity_threshold']:
            return 'High-infection'
        if agg_wb >= params['low_intensity_threshold']:
            return 'Low-infection'
        return 'Non-infected'

    def assign_symptoms_initial(self, population, eligible_idx):
        """
        Assigns multiple symptoms to the initial population.

        :param eligible_idx: indices of infected individuals
        :param population:
        """
        if len(eligible_idx):
            df = population.props
            params = self.parameters
            # create a dictionary from a df
            symptoms_dict = params['symptoms'].set_index('symptoms').to_dict()['prevalence']
            symptoms_column = 'sh_symptoms'

            for symptom in symptoms_dict.keys():
                prev = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
                # find who should get this symptom assigned - get p indices
                s_idx = self.rng.choice(eligible_idx, size=int(prev * len(eligible_idx)), replace=False)
                df.loc[s_idx, symptoms_column] = \
                    df.loc[s_idx, symptoms_column].apply(lambda x: add_elements(x, [symptom]))

    def assign_hsi_dates_initial(self, population, symptomatic_idx):
        df = population.props
        params = self.sim.modules['Schisto'].parameters
        for person_id in symptomatic_idx:
            p = params['prob_seeking_healthcare']
            will_seek_treatment = self.rng.choice(['True', 'False'], size=1, p=[p, 1 - p])
            if will_seek_treatment:
                seeking_treatment_ahead = \
                    self.rng.uniform(params['delay_till_hsi_a'], params['delay_till_hsi_b'], size=1).astype(int)
                seeking_treatment_ahead = pd.to_timedelta(seeking_treatment_ahead, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead
                seek_treatment_event = HSI_SchistoSeekTreatment(self.sim.modules['Schisto'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
                                                                    priority=1,
                                                                    topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
                                                                    + DateOffset(weeks=502))

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add the basic events of infection
        sim.schedule_event(SchistoInfectionWormBurdenEvent(self), sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + DateOffset(months=0))
        sim.schedule_event(SchistoLoggingPrevDistrictEvent(self), sim.date + DateOffset(years=70))
        # sim.schedule_event(SchistoParamFittingLogging(self), sim.date + DateOffset(years=15))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props
        params = self.parameters
        # Assign the default for a newly born child
        df.at[child_id, 'sh_infection_status'] = 'Non-infected'
        df.at[child_id, 'sh_symptoms'] = np.nan
        df.at[child_id, 'sh_aggregate_worm_burden'] = 0
        df.at[child_id, 'sh_prevalent_days_this_year'] = 0
        df.at[child_id, 'sh_start_of_prevalent_period'] = pd.NaT
        df.at[child_id, 'sh_start_of_high_infection'] = pd.NaT
        df.at[child_id, 'sh_high_inf_days_this_year'] = 0
        district = df.loc[mother_id, 'district_of_residence']
        # generate the harbouring rate depending on a district
        alpha = params['gamma_alpha'][district]
        df.at[child_id, 'sh_harbouring_rate'] = self.rng.gamma(alpha, size=1)

    def report_daly_values(self):
        logger.debug('This is Schisto Haematobium reporting my health values')
        df = self.sim.population.props
        health_values = df.loc[df.is_alive,
                               'sh_symptoms'].apply(lambda x: self.sim.modules['Schisto'].add_DALYs_from_symptoms(x))
        health_values.name = 'Schisto_Haematobium_Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug('This is Schisto Haematobium, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class Schisto_Mansoni(Module):
    """
    Schistosomiasis module
    It demonstrates the following behaviours in respect of the healthsystem module:
    Models the transmission of the S.Mansoni infection & development of symptoms.
    Requires Schisto() module to be registered to work.
    """

    def __init__(self, name=None, resourcefilepath=None, symptoms_and_HSI=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.prefix = 'sm'
        assert symptoms_and_HSI in [True, False]
        self.symptoms_and_HSI = symptoms_and_HSI

    PARAMETERS = {
        'reservoir_2010': Parameter(Types.REAL, 'Initial reservoir of infectious material per district in 2010'),
        'delay_a': Parameter(Types.REAL, 'End of the latent period in days, start'),
        'delay_b': Parameter(Types.REAL, 'End of the latent period in days, end'),
        'symptoms': Parameter(Types.DATA_FRAME, 'Symptoms for S. Mansoni infection'),
        'eggs_per_worm_extracted': Parameter(Types.REAL, 'Number of eggs per mature female worm extracted to the env'),
        'gamma_alpha': Parameter(Types.REAL, 'Parameter alpha for Gamma distribution for harbouring rates'),
        'beta_PSAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_SAC': Parameter(Types.REAL, 'Contact/exposure rate of PSAC'),
        'beta_Adults': Parameter(Types.REAL, 'Contact/exposure rate of Adults'),
        'worms_fecundity': Parameter(Types.REAL, 'Fecundity parameter, driving density-dependent reproduction'),
        'R0': Parameter(Types.REAL, 'Effective reproduction number, for the FOI'),
        'worm_lifespan': Parameter(Types.REAL, 'Lifespan of the worm in human host given in years'),
        'high_intensity_threshold': Parameter(Types.REAL,
                                              'Threshold of worm burden indicating high intensity infection'),
        'low_intensity_threshold': Parameter(Types.REAL, 'Threshold of worm burden indicating low intensity infection'),
        'high_intensity_threshold_PSAC': Parameter(Types.REAL, 'Threshold of worm burden '
                                                               'indicating high intensity infection for PSAC'),
        'list_of_districts': Parameter(Types.LIST, 'List of the districts with the infection present')
    }

    PROPERTIES = {
        'sm_infection_status': Property(
            Types.CATEGORICAL, 'Current status of schistosomiasis infection',
            categories=['Non-infected', 'Low-infection', 'High-infection']),
        'sm_aggregate_worm_burden': Property(
            Types.INT, 'Number of mature worms in the individual'),
        'sm_symptoms': Property(
            Types.LIST, 'Symptoms for S. Mansoni infection'),  # actually might also be np.nan
        'sm_start_of_prevalent_period': Property(Types.DATE, 'Date of going from Non-infected to Infected'),
        'sm_start_of_high_infection': Property(Types.DATE, 'Date of going from entering state High-inf'),
        'sm_harbouring_rate': Property(Types.REAL,
                                       'Rate of harbouring new worms (Poisson), drawn from gamma distribution'),
        'sm_prevalent_days_this_year': Property(Types.INT, 'Cumulative days with infection in current year'),
        'sm_high_inf_days_this_year': Property(Types.INT,
                                               'Cumulative days with high-intensity infection in current year')
    }

    def read_parameters(self, data_folder):
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Schisto.xlsx', sheet_name=None)

        params = self.parameters
        params['param_list'] = workbook['Parameters']
        self.param_list.set_index("Parameter", inplace=True)

        # natural history params
        params['delay_a'] = self.param_list.loc['delay_a', 'Value']
        params['delay_b'] = self.param_list.loc['delay_b', 'Value']
        params['beta_PSAC'] = self.param_list.loc['beta_PSAC', 'Value']
        params['beta_SAC'] = self.param_list.loc['beta_SAC', 'Value']
        params['beta_Adults'] = self.param_list.loc['beta_Adults', 'Value']
        params['worms_fecundity'] = self.param_list.loc['worms_fecundity_mansoni', 'Value']
        params['worm_lifespan'] = self.param_list.loc['lifespan_mansoni', 'Value']
        params['high_intensity_threshold'] = self.param_list.loc['high_intensity_threshold_mansoni', 'Value']
        params['low_intensity_threshold'] = self.param_list.loc['low_intensity_threshold_mansoni', 'Value']
        params['high_intensity_threshold_PSAC'] = self.param_list.loc['high_intensity_threshold_mansoni_PSAC', 'Value']

        # baseline reservoir size and other district-related params (alpha and R0)
        params['schisto_initial_reservoir'] = workbook['District_Params_mansoni']
        self.schisto_initial_reservoir.set_index("District", inplace=True)
        params['reservoir_2010'] = self.schisto_initial_reservoir.loc[:, 'Reservoir']
        params['gamma_alpha'] = self.schisto_initial_reservoir.loc[:, 'alpha_value']
        params['R0'] = self.schisto_initial_reservoir.loc[:, 'R0_value']

        # symptoms prevalence
        params['symptoms'] = pd.DataFrame(
            data={
                'symptoms': ['anemia', 'fever', 'ascites', 'diarrhoea', 'vomit', 'hepatomegaly'],
                'prevalence': [0.9, 0.3, 0.0054, 0.0144, 0.0172, 0.1574],
                'hsb_symptoms': ['other', 'fever', 'stomach_ache', 'diarrhoea', 'vomit', 'stomach_ache'],
            })

        params['list_of_districts'] = \
            pd.read_csv(Path(self.resourcefilepath) /
                        'ResourceFile_District_Population_Data.csv').District.unique().tolist()

    def initialise_population(self, population):
        """Set our property values for the initial population.

        :param population: the population of individuals
        """
        assert 'Schisto' in self.sim.modules.keys(), "Module Schisto must be registered for Schisto_Mansoni to work"

        df = population.props

        assert len(df.index[df.is_alive].tolist()) == len(df.index.tolist()), "Dead subjects in the initial population"

        df['sm_infection_status'].values[:] = 'Non-infected'
        df['sm_aggregate_worm_burden'] = 0
        df['sm_symptoms'] = np.nan
        df['sm_prevalent_days_this_year'] = 0
        df['sm_start_of_prevalent_period'] = pd.NaT
        df['sm_start_of_high_infection'] = pd.NaT
        df['sm_high_inf_days_this_year'] = 0
        df['sm_symptoms'] = df['sm_symptoms'].astype(object)

        # assign a harbouring rate
        self.assign_harbouring_rate(population)

        # assign initial worm burden
        self.assign_initial_worm_burden(population)

        # assign infection statuses
        df['sm_infection_status'].values[:] = \
            df.apply(lambda x: self.intensity_of_infection(x['age_years'], x['sm_aggregate_worm_burden']), axis=1)

        # df['sm_infection_status'] = \
        #     df.apply(lambda x: self.intensity_of_infection(x['age_years'], x['sm_aggregate_worm_burden']), axis=1)
        # df['sm_infection_status'] = df['sm_infection_status'].astype('category')

        #  start of the prevalent period & start of high-intensity infections
        infected_idx = df.index[df['sm_infection_status'] != 'Non-infected']
        df.loc[infected_idx, 'sm_start_of_prevalent_period'] = self.sim.date
        high_infected_idx = df.index[df['sm_infection_status'] == 'High-infection']
        df.loc[high_infected_idx, 'sm_start_of_high_infection'] = self.sim.date

        if self.symptoms_and_HSI:
            # assign initial symptoms
            self.assign_symptoms_initial(population, high_infected_idx)
            # assign initial dates of seeking healthcare for people with symptoms
            symptomatic_idx = df.index[~df['sm_symptoms'].isna()]
            self.assign_hsi_dates_initial(population, symptomatic_idx)

    def assign_harbouring_rate(self, population):
        """Assign a harbouring rate to every individual, this happens with a district-related param"""
        df = population.props
        params = self.parameters
        for district in params['list_of_districts']:
            eligible = df.index[df['district_of_residence'] == district].tolist()
            hr = params['gamma_alpha'][district]
            df.loc[eligible, 'sm_harbouring_rate'] = self.rng.gamma(hr, size=len(eligible))

    def assign_initial_worm_burden(self, population):
        """Assign initial 2010 prevalence of schistosomiasis infections
        This will depend on a district and age group.

        :param population: population
        """
        df = population.props
        params = self.parameters
        districts = df.district_of_residence.unique().tolist()
        reservoir = params['reservoir_2010']

        for distr in districts:
            eligible = df.index[df['district_of_residence'] == distr].tolist()
            reservoir_distr = int(reservoir[distr] * len(eligible))
            contact_rates = pd.Series(1, index=eligible)
            for age_group in ['PSAC', 'SAC', 'Adults']:
                age_range = map_age_groups(age_group)
                in_the_age_group = \
                    df.index[(df['district_of_residence'] == distr) &
                             (df['age_years'].between(age_range[0], age_range[1]))].tolist()
                contact_rates.loc[in_the_age_group] *= params['beta_' + age_group]  # Beta(age_group)

            if len(eligible):
                harbouring_rates = df.loc[eligible, 'sm_harbouring_rate'].values
                rates = np.multiply(harbouring_rates, contact_rates)
                reservoir_distr = int(reservoir[distr] * len(eligible))
                # assign a worm burden
                chosen = self.rng.choice(eligible, reservoir_distr, p=rates / rates.sum())
                unique, counts = np.unique(chosen, return_counts=True)
                worms_per_idx = dict(zip(unique, counts))
                df['sm_aggregate_worm_burden'].update(pd.Series(worms_per_idx))

        # schedule death of worms
        people_with_worms = df.index[df['sm_aggregate_worm_burden'] > 0]
        print('Initial people with worms:', len(people_with_worms))
        for person_id in people_with_worms:
            worms = df.loc[person_id, 'sm_aggregate_worm_burden']
            natural_death_of_worms = SchistoWormsNatDeath(self, person_id=person_id,
                                                          number_of_worms=worms)
            months_till_death = int(self.rng.uniform(1, params['worm_lifespan'] * 12 / 2))
            self.sim.schedule_event(natural_death_of_worms, self.sim.date + DateOffset(months=months_till_death))

    def draw_worms(self, worms_total, rates, district):
        """ This function generates random number of new worms drawn from Poisson distribution multiplied by
        a product of harbouring rate and exposure rate
        :param district: district name
        :param rates: harbouring rates used for Poisson distribution, drawn from Gamma,
         multiplied by contact rate per age group
        :param worms_total: total size of the reservoir of infectious material
        :return harboured_worms: array of numbers of new worms for each of the persons (len = len(rates))
        """
        params = self.parameters
        if worms_total == 0:
            return np.zeros(len(rates))
        rates = list(rates)
        R0 = params['R0'][district]
        worms_total *= R0
        harboured_worms = np.asarray([self.rng.poisson(x * worms_total, 1)[0] for x in rates]).astype(int)
        return harboured_worms

    def intensity_of_infection(self, age, agg_wb):
        params = self.parameters
        if age < 5:
            if agg_wb >= params['high_intensity_threshold_PSAC']:
                return 'High-infection'
        if agg_wb >= params['high_intensity_threshold']:
            return 'High-infection'
        if agg_wb >= params['low_intensity_threshold']:
            return 'Low-infection'
        return 'Non-infected'

    def assign_symptoms_initial(self, population, eligible_idx):
        """
        Assigns multiple symptoms to the initial population.

        :param eligible_idx: indices of infected individuals
        :param population:
        """
        if len(eligible_idx):
            df = population.props
            params = self.parameters
            # create a dictionary from a df
            symptoms_dict = params['symptoms'].set_index('symptoms').to_dict()['prevalence']
            symptoms_column = 'sm_symptoms'

            for symptom in symptoms_dict.keys():
                prev = symptoms_dict[symptom]  # get the prevalence of the symptom among the infected population
                # find who should get this symptom assigned - get p indices
                s_idx = self.rng.choice(eligible_idx, size=int(prev * len(eligible_idx)), replace=False)
                df.loc[s_idx, symptoms_column] = \
                    df.loc[s_idx, symptoms_column].apply(lambda x: add_elements(x, [symptom]))

    def assign_hsi_dates_initial(self, population, symptomatic_idx):
        df = population.props
        params = self.sim.modules['Schisto'].parameters
        for person_id in symptomatic_idx:
            p = params['prob_seeking_healthcare']
            will_seek_treatment = self.rng.choice(['True', 'False'], size=1, p=[p, 1 - p])
            if will_seek_treatment:
                seeking_treatment_ahead = \
                    self.rng.uniform(params['delay_till_hsi_a'], params['delay_till_hsi_b'], size=1).astype(int)
                seeking_treatment_ahead = pd.to_timedelta(seeking_treatment_ahead, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead
                seek_treatment_event = HSI_SchistoSeekTreatment(self.sim.modules['Schisto'], person_id=person_id)
                # self.sim.schedule_event(seek_treatment_event, self.sim.date)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
                                                                    priority=1,
                                                                    topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
                                                                    + DateOffset(weeks=502))

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # add the basic events of infection
        sim.schedule_event(SchistoInfectionWormBurdenEvent(self), sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(SchistoLoggingEvent(self), sim.date + DateOffset(months=0))
        sim.schedule_event(SchistoLoggingPrevDistrictEvent(self), sim.date + DateOffset(years=69))
        # sim.schedule_event(SchistoParamFittingLogging(self), sim.date + DateOffset(years=15))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        All children are born without an infection, even if the mother is infected.

        :param mother_id: the ID for the mother for this child (redundant)
        :param child_id: the new child
        """
        df = self.sim.population.props
        params = self.parameters
        # Assign the default for a newly born child
        df.at[child_id, 'sm_infection_status'] = 'Non-infected'
        df.at[child_id, 'sm_symptoms'] = np.nan
        df.at[child_id, 'sm_aggregate_worm_burden'] = 0
        df.at[child_id, 'sm_prevalent_days_this_year'] = 0
        df.at[child_id, 'sm_start_of_prevalent_period'] = pd.NaT
        df.at[child_id, 'sm_start_of_high_infection'] = pd.NaT
        df.at[child_id, 'sm_high_inf_days_this_year'] = 0
        district = df.loc[mother_id, 'district_of_residence']
        # generate the harbouring rate depending on a district
        alpha = params['gamma_alpha'][district]
        df.at[child_id, 'sm_harbouring_rate'] = self.rng.gamma(alpha, size=1)

    def report_daly_values(self):
        logger.debug('This is Schisto Mansoni reporting my health values')
        df = self.sim.population.props
        health_values = \
            df.loc[df.is_alive, 'sm_symptoms'].apply(lambda x: self.sim.modules['Schisto'].add_DALYs_from_symptoms(x))
        health_values.name = 'Schisto_Mansoni_Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug('This is Schisto_Mansoni, being alerted about a health system interaction '
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
        print("Changing the parameter", self.param_name, "from value",
              self.module.parameters[self.param_name], "to", self.new_value)
        self.module.change_paramater_value(self.param_name, self.new_value)


class SchistoInfectionWormBurdenEvent(RegularEvent, PopulationScopeEventMixin):
    """An event of infecting people with Schistosomiasis
    Using Worm Burden and Reservoir of Infectious Material - see write up
    """

    def __init__(self, module):
        """
        :param module: the module that created this event,
        must be either Schisto_Haematobium or Schisto_Mansoni
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def apply(self, population):
        logger.debug('This is SchistoEvent, tracking the disease progression of the population.')

        df = population.props
        districts = self.module.parameters['list_of_districts']

        worms_dict = {}
        # increase worm burden of people in each district
        for distr in districts:
            df_distr_indx = df.index[(df['is_alive']) & (df['district_of_residence'] == distr)].tolist()
            if len(df_distr_indx):
                new_worms = self.increase_worm_burden_distr(population, distr)  # this is a list
                worms_dict.update(dict(zip(df_distr_indx, new_worms)))

        # get indices of people who harboured new worms
        # remove all indices for which there is no increment
        worms_dict = {key: val for key, val in worms_dict.items() if val != 0}
        worm_burden_increased = list(worms_dict.keys())

        # schedule the time of new worms becoming mature
        days_till_maturation = self.module.rng.uniform(30, 55, size=len(worm_burden_increased)).astype(int)
        days_till_maturation = pd.to_timedelta(days_till_maturation, unit='D')

        date_counter = 0
        for person_index in worm_burden_increased:
            new_worms = worms_dict[person_index]
            maturation_event = SchistoMatureWorms(self.module, person_id=person_index, new_worms=new_worms)
            event_date = self.sim.date + days_till_maturation[date_counter]
            self.sim.schedule_event(maturation_event, event_date)
            date_counter += 1

    def calculate_reservoir_size(self, population, district):
        """Calculates the size of the reservoir of infectious material in a given district
        by multiplying the exposure rates beta and the worm burdens in each age group and then summing it up
        :param population: population
        :param district: one of the 32 Malawi districts
        :returns district_reservoir: size of the reservoir for the given district
        """
        df = population.props
        params = self.module.parameters
        district_reservoir = 0
        ppl_in_district = len(df.index[df['district_of_residence'] == district])
        for age_group in ['PSAC', 'SAC', 'Adults']:
            beta = params['beta_' + age_group]  # the input to the reservoir of the age group = contact rate
            age_range = map_age_groups(age_group)
            in_the_age_group = df.index[(df['district_of_residence'] == district) &
                                        (df['age_years'].between(age_range[0], age_range[1]))].tolist()
            if len(in_the_age_group):
                age_worm_burden = df.loc[in_the_age_group, self.module.prefix + '_aggregate_worm_burden'].values.mean()
                age_worm_burden *= beta  # to get contribution to the reservoir
                age_worm_burden *= len(in_the_age_group) / ppl_in_district  # to get weighted mean
            else:
                age_worm_burden = 0
            district_reservoir += age_worm_burden
        return district_reservoir

    def increase_worm_burden_distr(self, population, district):
        """Randomly assigns newly acquired worms to each individual in the district from Poisson,
        based on the calculated reservoir size and their harbouring and contact rates,
        then draws from Bernoulli to determine if the worms will establish or not

         :param population: population
         :param district: one of the 32 Malawi districts
         :returns allocated_new_worms: new worms acquired by the people in the given district
         """

        df = population.props
        params = self.module.parameters
        df_distr_idx = df.index[(df["district_of_residence"] == district) & (df.is_alive)]

        # calculate the size of infectious material reservoir
        reservoir = self.calculate_reservoir_size(population, district)
        if reservoir == 0:
            return [0] * len(df_distr_idx)
        # assign appropriate Beta parameters
        contact_rates = pd.Series(1, index=df_distr_idx)
        for age_group in ['PSAC', 'SAC', 'Adults']:
            age_range = map_age_groups(age_group)
            multiplier = params['beta_' + age_group]  # Beta(age_group)
            contact_rates.loc[df.age_years.between(age_range[0], age_range[1])] *= multiplier

        #  draw new worms
        harbouring_rates = df.loc[df_distr_idx, self.module.prefix + '_harbouring_rate'].values
        rates = np.multiply(harbouring_rates, contact_rates)
        allocated_new_worms = self.module.draw_worms(reservoir, rates, district)
        prob_of_establishment = list(np.exp(-params['worms_fecundity'] * df.loc[df_distr_idx, self.module.prefix +
                                                                                '_aggregate_worm_burden'].values))
        prob_of_establishment = [self.module.rng.choice([1, 0], p=[x, 1-x], size=1)[0] for x in prob_of_establishment]
        prob_of_establishment = np.asarray(prob_of_establishment)  # zeroes and ones
        allocated_new_worms = np.multiply(allocated_new_worms, prob_of_establishment)  # becomes 0 if 0 was drawn
        return list(allocated_new_worms)


class SchistoMatureWorms(Event, IndividualScopeEventMixin):
    """Increases the aggregate worm burden of an individual upon maturation of the worms
    Changes the infection status accordingly
    Schedules the natural death of worms and symptoms development if High-infection
    """
    def __init__(self, module, person_id, new_worms):
        super().__init__(module, person_id=person_id)
        self.new_worms = new_worms
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        if df.loc[person_id, 'is_alive']:
            # increase worm burden
            df.loc[person_id, self.module.prefix + '_aggregate_worm_burden'] += self.new_worms

            # schedule the natural death of the worms
            natural_death_of_worms = SchistoWormsNatDeath(self.module, person_id=person_id,
                                                          number_of_worms=self.new_worms)
            self.sim.schedule_event(natural_death_of_worms, self.sim.date + DateOffset(years=params['worm_lifespan']))

            if df.loc[person_id, self.module.prefix + '_infection_status'] != 'High-infection':
                if df.loc[person_id, 'age_years'] < 5:
                    threshold = params['high_intensity_threshold_PSAC']
                else:
                    threshold = params['high_intensity_threshold']
                if df.loc[person_id, self.module.prefix + '_aggregate_worm_burden'] >= threshold:
                    df.loc[person_id, self.module.prefix + '_infection_status'] = 'High-infection'
                    df.loc[person_id, self.module.prefix + '_start_of_high_infection'] = self.sim.date
                    if self.module.symptoms_and_HSI:
                        develop_symptoms = SchistoDevelopSymptomsEvent(self.module, person_id=person_id)
                        self.sim.schedule_event(develop_symptoms, self.sim.date)  # develop symptoms immediately

                elif df.loc[person_id, self.module.prefix +
                            '_aggregate_worm_burden'] >= params['low_intensity_threshold']:
                    if df.loc[person_id, self.module.prefix + '_infection_status'] == 'Non-infected':
                        df.loc[person_id, self.module.prefix + '_infection_status'] = 'Low-infection'

            if \
                (df.loc[person_id, self.module.prefix + '_infection_status'] != 'Non-infected') &\
                    (pd.isna(df.loc[person_id, self.module.prefix + '_start_of_prevalent_period'])):
                df.loc[person_id, self.module.prefix + '_start_of_prevalent_period'] = self.sim.date


class SchistoWormsNatDeath(Event, IndividualScopeEventMixin):
    """Decreases the aggregate worm burden of an individual upon natural death of the adult worms
    This event checks the last day of PZQ treatment and if has been less than the lifespan of the worm
    it doesn't do anything - worms have been killed by now
    Otherwise, the worms die naturally and the worm burden is decreased
    """
    def __init__(self, module, person_id, number_of_worms):
        super().__init__(module, person_id=person_id)
        self.number_of_worms = number_of_worms
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def apply(self, person_id):
        params = self.module.parameters
        df = self.sim.population.props
        worms_now = df.loc[person_id, self.module.prefix + '_aggregate_worm_burden']
        days_since_last_treatment = self.sim.date - df.loc[person_id, 'ss_last_PZQ_date']
        days_since_last_treatment = int(days_since_last_treatment / np.timedelta64(1, 'Y'))
        if days_since_last_treatment > params['worm_lifespan']:
            df.loc[person_id, self.module.prefix + '_aggregate_worm_burden'] = worms_now - self.number_of_worms
            # clearance of the worms
            if df.loc[person_id, self.module.prefix + '_aggregate_worm_burden'] < params['low_intensity_threshold']:
                df.loc[person_id, self.module.prefix + '_infection_status'] = 'Non-infected'
                # does not matter if low or high int infection
                if df.loc[person_id, self.module.prefix + '_infection_status'] != 'Non-infected':
                    # calculate prevalent period
                    prevalent_duration = self.sim.date - df.loc[person_id, self.module.prefix +
                                                                '_start_of_prevalent_period']
                    prevalent_duration = int(prevalent_duration / np.timedelta64(1, 'D')) % 365
                    df.loc[person_id, self.module.prefix + '_prevalent_days_this_year'] += prevalent_duration
                    df.loc[person_id, self.module.prefix + '_start_of_prevalent_period'] = pd.NaT
                    df.loc[person_id, self.module.prefix + '_start_of_high_infection'] = pd.NaT
            else:
                if df.loc[person_id, self.module.prefix + '_infection_status'] == 'High-infection':
                    if df.loc[person_id, 'age_years'] < 5:
                        threshold = params['high_intensity_threshold_PSAC']
                    else:
                        threshold = params['high_intensity_threshold']
                    if df.loc[person_id, self.module.prefix + '_aggregate_worm_burden'] < threshold:
                        df.loc[person_id, self.module.prefix + '_infection_status'] = 'Low-infection'
                        high_inf_duration = self.sim.date - df.loc[person_id, self.module.prefix +
                                                                   '_start_of_high_infection']
                        high_inf_duration = int(high_inf_duration / np.timedelta64(1, 'D')) % 365
                        df.loc[person_id, self.module.prefix + '_high_inf_days_this_year'] += high_inf_duration
                        df.loc[person_id, self.module.prefix + '_start_of_high_infection'] = pd.NaT


class SchistoDevelopSymptomsEvent(Event, IndividualScopeEventMixin):
    """Development of symptoms upon high intensity infection
    Schedules the HSI_seek_treatment event, provided a True value is drawn from Bernoulli(prob_seek_healthcare)
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def apply(self, person_id):
        df = self.sim.population.props
        params_hsi = self.sim.modules['Schisto'].parameters
        # assign symptoms
        symptoms = self.assign_symptoms(self.module.prefix)
        if isinstance(symptoms, list):
            df.at[person_id, self.module.prefix + '_symptoms'] = symptoms
            # schedule Healthcare Seeking
            p = params_hsi['prob_seeking_healthcare']
            will_seek_treatment = self.module.rng.choice(['True', 'False'], size=1, p=[p, 1-p])
            if will_seek_treatment:
                seeking_treatment_ahead = self.module.rng.uniform(params_hsi['delay_till_hsi_a'],
                                                                  params_hsi['delay_till_hsi_b'],
                                                                  size=1).astype(int)
                seeking_treatment_ahead = pd.to_timedelta(seeking_treatment_ahead, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead
                seek_treatment_event = HSI_SchistoSeekTreatment(self.sim.modules['Schisto'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_event,
                                                                    priority=1,
                                                                    topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
                                                                    + DateOffset(weeks=502))

    def assign_symptoms(self, module_prefix):
        """
        Assign symptoms to the person with high intensity infection.

        :param module_prefix: indicates type of infection, haematobium or mansoni
        :return symptoms: np.nan if no symptom or a list of symptoms
        """
        assert module_prefix in ['sm', 'sh'], "Incorrect infection type. Can't assign symptoms."
        params = self.module.parameters
        symptoms_dict = params['symptoms'].set_index('symptoms').to_dict()[
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
    PZQ treats both types of infections, so affect symptoms and worm burden of any infection type registered
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Schisto)

    def apply(self, person_id):
        df = self.sim.population.props

        prefixes = []
        if 'Schisto_Haematobium' in self.sim.modules.keys():
            prefixes.append('sh')
        if 'Schisto_Mansoni' in self.sim.modules.keys():
            prefixes.append('sm')

        if df.loc[person_id, 'is_alive']:
            for prefix in prefixes:
                if df.loc[person_id,  prefix + '_infection_status'] != 'Non-infected':

                    # check if they experienced symptoms, and if yes, treat them
                    df.loc[person_id, prefix + '_symptoms'] = np.nan
                    # if isinstance(df.loc[person_id, prefix + '_symptoms'], list):
                    #     df.loc[person_id, prefix + '_symptoms'] = np.nan

                    # calculate the duration of the prevalent period
                    prevalent_duration = count_days_this_year(self.sim.date, df.loc[
                        person_id, prefix + '_start_of_prevalent_period'])
                    df.loc[person_id, prefix + '_prevalent_days_this_year'] += prevalent_duration
                    df.loc[person_id, prefix + '_start_of_prevalent_period'] = pd.NaT

                    # calculate the duration of the high-intensity infection
                    if df.loc[person_id, prefix + '_infection_status'] == 'High-infection':
                        high_infection_duration = count_days_this_year(self.sim.date, df.loc[
                            person_id, prefix + '_start_of_high_infection'])
                        df.loc[person_id, prefix + '_high_inf_days_this_year'] += high_infection_duration
                        df.loc[person_id, prefix + '_start_of_high_infection'] = pd.NaT

                    df.loc[person_id, prefix + '_aggregate_worm_burden'] = 0  # PZQ_efficacy = 100% for now
                    df.loc[person_id, prefix + '_start_of_prevalent_period'] = pd.NaT
                    df.loc[person_id, prefix + '_start_of_high_infection'] = pd.NaT
                    df.loc[person_id, prefix + '_infection_status'] = 'Non-infected'

        # the general Schisto module properties
        df.loc[person_id, 'ss_scheduled_hsi_date'] = pd.NaT
        df.loc[person_id, 'ss_last_PZQ_date'] = self.sim.date

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

        prefixes = []
        if 'Schisto_Haematobium' in self.sim.modules.keys():
            prefixes.append('sh')
        if 'Schisto_Mansoni' in self.sim.modules.keys():
            prefixes.append('sm')
        is_infected = False
        for pref in prefixes:
            if df.loc[person_id, pref + '_infection_status'] != 'Non-infected':
                is_infected = True

        # appt are scheduled and cannot be cancelled in the following situations:
        #   a) person has died
        #   b) the infection has been treated in MDA or by treating symptoms from
        #   other schisto infection before the appt happened
        if (df.loc[person_id, 'is_alive'] & is_infected):  # &
            # (df.loc[person_id, 'ss_scheduled_hsi_date'] <= self.sim.date)):
            # check if a person is a child or an adult and assign prob of being sent to schisto test (hence being cured)
            if df.loc[person_id, 'age_years'] <= 15:
                prob_test = params['prob_sent_to_lab_test_children']
            else:
                prob_test = params['prob_sent_to_lab_test_adults']

            sent_to_test = self.module.rng.choice([True, False], p=[prob_test, 1-prob_test])
            # use this is you don't care about whether PZQ is available or not
            # if sent_to_test:
            #     self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)
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
                    # print('Patients is cured with PZQ')
                    self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

            #     else:
            #         # print('There was no PZQ available')
            #         logger.debug('ItemsCode1 is not available, so can' 't use it.')
            else:  # person seeked treatment but was not sent to test; visit is reschedulled
                # schedule another Seeking Treatment event for that person
                seeking_treatment_ahead_repeated = \
                    int(self.module.rng.uniform(params['delay_till_hsi_a_repeated'],
                                                params['delay_till_hsi_b_repeated']))
                seeking_treatment_ahead_repeated = pd.to_timedelta(seeking_treatment_ahead_repeated, unit='D')
                df.loc[person_id, 'ss_scheduled_hsi_date'] = self.sim.date + seeking_treatment_ahead_repeated

                seek_treatment_repeated = HSI_SchistoSeekTreatment(self.module, person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(seek_treatment_repeated,
                                                                    priority=1,
                                                                    topen=df.loc[person_id, 'ss_scheduled_hsi_date'],
                                                                    tclose=df.loc[person_id, 'ss_scheduled_hsi_date']
                                                                    + DateOffset(weeks=500))

    def did_not_run(self):
        print('HSI event did not run')
        return True


# ---------------------------------------------------------------------------------------------------------
#   MASS-DRUG ADMINISTRATION EVENTS
# ---------------------------------------------------------------------------------------------------------

class SchistoHistoricalMDAEvent(Event, PopulationScopeEventMixin):
    """Mass-Drug administration scheduled for the population
    Using the historical MDA coverage
    """
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto)

    def apply(self, population):
        print("Historical MDA is happening now!")
        year = self.sim.date.year
        year = year - 2035 + 2015
        assert year in [2015, 2016, 2017, 2018], "No historical coverage data for this year"

        treated_idx_PSAC = self.assign_historical_MDA_coverage(population, year, 'PSAC')
        treated_idx_SAC = self.assign_historical_MDA_coverage(population, year, 'SAC')
        treated_idx_Adults = self.assign_historical_MDA_coverage(population, year, 'Adults')

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        for person_id in treated_idx:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

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
        print("Prognosed MDA is happening now!")
        district = self.district

        treated_idx_PSAC = self.assign_prognosed_MDA_coverage(population, district, 'PSAC')
        treated_idx_SAC = self.assign_prognosed_MDA_coverage(population, district, 'SAC')
        treated_idx_Adults = self.assign_prognosed_MDA_coverage(population, district, 'Adults')

        treated_idx = treated_idx_PSAC + treated_idx_SAC + treated_idx_Adults
        # all treated people will have worm burden decreased, and we already have chosen only alive people
        for person_id in treated_idx:
            self.sim.schedule_event(SchistoTreatmentEvent(self.module, person_id), self.sim.date)

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
            MDA_idx = self.module.rng.choice(eligible, size=int(coverage_distr * (len(eligible))),
                                             replace=False).tolist()

        return MDA_idx


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------


class SchistoParamFittingLogging(Event, PopulationScopeEventMixin):
    """This Logging event should only be used for creating a lookup table for parameter fitting
    Otherwise should not be scheduled"""
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def create_logger(self, population):

        if isinstance(self.module, Schisto_Haematobium):
            inf_type = 'Haematobium'
            prefix = 'sh'
        else:
            inf_type = 'Mansoni'
            prefix = 'sm'

        df = population.props
        df_alive = df[df.is_alive].copy()
        all_infected = len(df_alive.index[df_alive[prefix + '_aggregate_worm_burden'] > 1])
        total_pop = len(df_alive.index)
        prevalence = all_infected / total_pop
        mwb = df_alive[prefix + '_aggregate_worm_burden'].mean()

        log_string = '%s|' + inf_type + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'Prevalence': prevalence,
                        'MWB': mwb,
                        'alpha': self.module.alpha,
                        'R0': self.module.r0
                    })

    def apply(self, population):
        self.create_logger(population)


class SchistoPrevalentDaysLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the prevalent days & high infection days per year
        It is best to use it only with one type of infection, as the logger does not contain info on the infection type
        """
        # run this event every year
        super().__init__(module, frequency=DateOffset(months=12))
        assert isinstance(module, Schisto)

    def create_logger(self, population):
        df = population.props
        # adding unfinished infections
        prefixes = []
        if 'Schisto_Haematobium' in self.sim.modules.keys():
            prefixes.append('sh')
        if 'Schisto_Mansoni' in self.sim.modules.keys():
            prefixes.append('sm')

        df_alive = df[df.is_alive].copy()

        for prefix in prefixes:
            still_infected = df_alive.index[~df_alive[prefix + '_start_of_prevalent_period'].isna()]
            for person_id in still_infected:
                prevalent_duration = count_days_this_year(self.sim.date,
                                                          df_alive.loc[person_id,
                                                                       prefix + '_start_of_prevalent_period'])
                df_alive.loc[person_id, prefix + '_prevalent_days_this_year'] += prevalent_duration
            still_high_infected = df_alive.index[~df_alive[prefix + '_start_of_high_infection'].isna()]
            for person_id in still_high_infected:
                high_inf_duration = count_days_this_year(self.sim.date,
                                                         df_alive.loc[person_id,
                                                                      prefix + '_start_of_high_infection'])
                df_alive.loc[person_id, prefix + '_high_inf_days_this_year'] += high_inf_duration

            for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
                count_states = self.count_prev_years_age(prefix, df_alive, age_group)
                log_string = '%s|' + age_group + '_PrevalentYears' + '|%s'
                logger.info(log_string, self.sim.date.date(),
                            {
                                'Prevalent_years_this_year_total': count_states['Prevalent_years_this_year_total'],
                                'Prevalent_years_per_100': count_states['Prevalent_years_per_100'],
                                'High_infection_years_this_year_total':
                                    count_states['High_infection_years_this_year_total'],
                                'High_infection_years_per_100': count_states['High_infection_years_per_100'],
                                'Total_pop_alive': count_states['Tot_pop_alive']
                            })

            # clear so that it's ready for next year
            df[prefix + '_prevalent_days_this_year'] = 0
            df[prefix + '_high_inf_days_this_year'] = 0

    def count_prev_years_age(self, prefix, df_age, age):
        # sum up for each age_group
        age_range = map_age_groups(age)  # returns a tuple
        idx = df_age.index[df_age['age_years'].between(age_range[0], age_range[1])]
        Tot_pop_alive = len(idx)
        Prevalent_years_this_year_total = \
            df_age.loc[idx, prefix + '_prevalent_days_this_year'].values.sum() / 365  # get years only
        High_infection_years_this_year_total = \
            df_age.loc[idx, prefix + '_high_inf_days_this_year'].values.sum() / 365
        Prevalent_years_per_100 = Prevalent_years_this_year_total * 100 / Tot_pop_alive
        High_infection_years_per_100 = High_infection_years_this_year_total * 100 / Tot_pop_alive
        count_states = {'Prevalent_years_this_year_total': Prevalent_years_this_year_total,
                        'Prevalent_years_per_100': Prevalent_years_per_100,
                        'High_infection_years_this_year_total': High_infection_years_this_year_total,
                        'High_infection_years_per_100': High_infection_years_per_100,
                        'Tot_pop_alive': Tot_pop_alive
                        }
        return count_states

    def apply(self, population):
        self.create_logger(population)


class SchistoLoggingPrevDistrictEvent(Event, PopulationScopeEventMixin):
    """Produces a log of prevalence and MWB in every district;
    used for validating the parameters fit"""
    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def create_logger(self, population, district):
        if isinstance(self.module, Schisto_Haematobium):
            inf_type = 'Haematobium'
        else:
            inf_type = 'Mansoni'
        count_states = self.count_district_states(population, district)
        log_string = '%s|' + district + '_' + inf_type + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'Non_infected': count_states['Non-infected'],
                        'Low_infections': count_states['Low-infection'],
                        'High_infections': count_states['High-infection'],
                        'Infected': count_states['infected_any'],
                        'Prevalence': count_states['Prevalence'],
                        'MeanWormBurden': count_states['MeanWormBurden']
                    })

    def count_district_states(self, population, district):
        """
        :param population:
        :param district:
        :return: count_states: a dictionary of counts of individuals in district in different states on infection
        """
        df = population.props

        # this is not ideal bc we're making a copy but it's for clearer code below
        # and it happens only once per simulation, if we are interested in the district prevalence
        df_distr = df[((df.is_alive) & (df['district_of_residence'] == district))].copy()

        count_states = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
        count_states.update(df_distr[self.module.prefix + '_infection_status'].value_counts().to_dict())
        count_states.update({'infected_any': count_states['Low-infection'] + count_states['High-infection']})
        count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})
        if count_states['total_pop_alive'] != 0.0:
            count_states.update({'Prevalence': count_states['infected_any'] / count_states['total_pop_alive']})
        else:
            count_states.update({'Prevalence': 0.0})
        mean = df_distr[self.module.prefix + '_aggregate_worm_burden'].values.mean()
        count_states.update({'MeanWormBurden': mean})
        return count_states

    def apply(self, population):
        # districts = population.props.district_of_residence.unique()
        districts = self.module.parameters['list_of_districts']
        for distr in districts:
            self.create_logger(population, distr)


class SchistoLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people
        """
        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto_Haematobium) or isinstance(module, Schisto_Mansoni)

    def create_logger(self, population, age_group):
        if isinstance(self.module, Schisto_Haematobium):
            inf_type = 'Haematobium'
        else:
            inf_type = 'Mansoni'
        count_states = self.count_age_group_states(population, age_group)
        log_string = '%s|' + age_group + '_' + inf_type + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'Non_infected': count_states['Non-infected'],
                        'Low_infections': count_states['Low-infection'],
                        'High_infections': count_states['High-infection'],
                        'Infected': count_states['infected_any'],
                        'Prevalence': count_states['Prevalence'],
                        'High-inf_Prevalence': count_states['High-inf_Prevalence'],
                        'MeanWormBurden': count_states['MeanWormBurden']
                    })

    def count_age_group_states(self, population, age_group):
        """
        :param population:
        :param age_group:
        :return: count_states: a dictionary of counts of individuals in age_group in different states on infection
        """

        districts = self.module.parameters['list_of_districts']
        df = population.props
        age_range = map_age_groups(age_group)  # returns a tuple

        idx = df.index[((df.is_alive) & (df['age_years'].between(age_range[0], age_range[1]))
                        & (df['district_of_residence'].isin(districts)))]

        count_states = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
        count_states.update(df.loc[idx, self.module.prefix + '_infection_status'].value_counts().to_dict())
        count_states.update({'infected_any': count_states['Low-infection'] + count_states['High-infection']})
        count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})
        count_states.update({'Prevalence': count_states['infected_any'] / count_states['total_pop_alive']})
        count_states.update({'High-inf_Prevalence': count_states['High-infection'] / count_states['total_pop_alive']})
        mean = df.loc[idx, self.module.prefix + '_aggregate_worm_burden'].values.mean()
        count_states.update({'MeanWormBurden': mean})
        return count_states

    def apply(self, population):
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            self.create_logger(population, age)


class SchistoLoggingTotalEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This logger logs the prevalence of ANY type of schistosomiasis
        If both schisto modules are registered it will count people with s.haematobium OR s.mansoni
        If only one of the schisto modules registered, it will work as the normal logger, i.e.
        count people with the specific infection type
        """
        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Schisto)

    def create_logger(self, population, age_group):
        count_states = self.count_age_group_states(population, age_group)
        log_string = '%s|' + age_group + '_Total' + '|%s'
        logger.info(log_string, self.sim.date.date(),
                    {
                        'Prevalence': count_states['Prevalence'],
                        'High_infections_Prevalence': count_states['High-inf_Prevalence'],
                    })

    def count_age_group_states(self, population, age_group):
        """
        :param population:
        :param age_group:
        :return: count_states: a dictionary of counts of individuals in age_group in different states on infection
        """
        df = population.props
        age_range = map_age_groups(age_group)  # returns a tuple

        idx = df.index[((df.is_alive) & (df['age_years'].between(age_range[0], age_range[1])))]

        inf = self.find_count_infected(population, idx)
        high_inf = self.find_count_high_infected(population, idx)
        total_pop_size = len(idx)

        count_states = {}
        count_states.update({'Prevalence': inf/total_pop_size})
        count_states.update({'High-inf_Prevalence': high_inf/total_pop_size})

        return count_states

    def find_count_infected(self, population, idx):
        df = population.props
        if ('Schisto_Haematobium' in self.sim.modules.keys()) & ('Schisto_Mansoni' in self.sim.modules.keys()):
            inf = len(df[(df.loc[idx, 'sh_infection_status'] != 'Non-infected') |
                         (df.loc[idx, 'sm_infection_status'] != 'Non-infected')])
        elif 'Schisto_Haematobium' in self.sim.modules.keys():
            inf = len(df[(df.loc[idx, 'sh_infection_status'] != 'Non-infected')])
        elif 'Schisto_Mansoni' in self.sim.modules.keys():
            inf = len(df[(df.loc[idx, 'sm_infection_status'] != 'Non-infected')])
        return inf

    def find_count_high_infected(self, population, idx):
        df = population.props
        if ('Schisto_Haematobium' in self.sim.modules.keys()) & ('Schisto_Mansoni' in self.sim.modules.keys()):
            high_inf = len(df[(df.loc[idx, 'sh_infection_status'] == 'High-infection') | (
                    df.loc[idx, 'sm_infection_status'] == 'High-infection')])
        elif 'Schisto_Haematobium' in self.sim.modules.keys():
            high_inf = len(df[(df.loc[idx, 'sh_infection_status'] == 'High-infection')])
        elif 'Schisto_Mansoni' in self.sim.modules.keys():
            high_inf = len(df[(df.loc[idx, 'sm_infection_status'] == 'High-infection')])
        return high_inf

    def apply(self, population):
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            self.create_logger(population, age)
