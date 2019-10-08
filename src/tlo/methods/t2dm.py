"""
This is the method for type 2 diabetes
Developed by Mikaela Smit, October 2018

"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: Asif/Stef to check if file path is mac/window flexible
# TODO: circular declaration with diabetes to be addressed
# TODO: Update weight to BMI
# ToDO: Update complication, symptoms and QALYs in line with DALY file
# TODO: how to handle symptoms and how specific to be - develop
# TODO: What about Qaly for mild symptoms (urination, tiredness etc), nephrology, amputation, and NOTE: severe vision impairmenet and blindness same qaly so only one


def make_t2dm_age_range_lookup(min_age, max_age, range_size):
    """Generates and returns a dictionary mapping age (in years) to age range
    as per data for validation (i.e. 25-35, 35-45, etc until 65)
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit (100 years) into 5 year ranges
    parts = chunks(range(min_age, max_age), range_size)

    # any ages >= 100 are in the '100+' category
    # TODO: would be good to have those younger than 25 in 25- instead or just other
    default_category = '%d+' % max_age
    lookup = defaultdict(lambda: default_category)

    # collect the possible ranges
    ranges = []

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = '%s-%s' % (start, end)
        ranges.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    ranges.append(default_category)
    return ranges, lookup


class Type2DiabetesMellitus(Module):
    """
    This is type 2 diabetes mellitus.
    Version October 2019
    The execution of all health system related interaction for type 2 diabetes mellitus are controlled through
    this module.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        logger.info('----------------------------------------------------------------------')
        logger.info("Running type 2 diabetes mellitus.  ")
        logger.info('----------------------------------------------------------------------')

    d2_age_range_categories, d2_age_range_lookup = make_t2dm_age_range_lookup(25, 65, 10)

    # We should have 4 age range categories
    assert len(d2_age_range_categories) == 5

    PARAMETERS = {

        # Here we declare parameters for this module. Each parameter has a name, data type,
        # and longer description.

        # Define epidemiological parameters
        'D2_prevalence': Parameter(Types.REAL, 'T2DM prevalence'),
        'D2_incidence': Parameter(Types.REAL, 'T2DM incidence'),
        'prob_t2dm_basic': Parameter(Types.REAL, 'Basic T2DM probability'),
        'prob_t2dmgivenbmi': Parameter(Types.REAL, 'T2DM probability given being BMI'),
        'prob_t2dmgivenht': Parameter(Types.REAL, 'T2DM probability given pre-existing hypertension'),
        'prob_t2dmgivengd': Parameter(Types.REAL, 'T2DM probability given pre-existing gestational diabetes'),
        'initial_prevalence_d2': Parameter(Types.REAL, 'Prevalence of T2DM as per data'),

        # Define disease progression parameters
        'prob_RetinoComp': Parameter(Types.REAL, 'Probability of developing retinopathy'),
        'prob_NephroComp': Parameter(Types.REAL, 'Probability of developing diabetic nephropathy'),
        'prob_NeuroComp': Parameter(Types.REAL, 'Probability of developing peripheral neuropathy'),
        'prob_death': Parameter(Types.REAL, 'Probability of dying'),

        # Define health care parameters
        # TODO: update after HSI events
        # TODO: update with DALY weights
        'level_of_symptoms': Parameter(Types.CATEGORICAL,      'Severity of symptoms that the individual will have'),
        'dalywt_mild_retino': Parameter(Types.REAL,            'DALY weighting for mild retinopathy'),
        'dalywt_severe_retino': Parameter(Types.REAL,          'DALY weighting for severe retinopathy'),
        'dalywt_uncomplicated': Parameter(Types.REAL,          'DALY weighting for uncomplicated diabetic'),
        'dalywt_neuropathy': Parameter(Types.REAL,             'DALY weighting for neuropathy'),
    }

    PROPERTIES = {

        # Next we declare the properties of individuals that this module provides.
        # Again each has a name, type and description.

        # Note that all properties must have a two letter prefix that identifies them to this module.

        # ToDo: Update symptoms name

        # Define disease properties
        'd2_age_range': Property(Types.CATEGORICAL, 'The age range categories for hypertension validation use',
                                 categories=d2_age_range_categories),
        'd2_risk': Property(Types.REAL,                       'Risk of T2DM given pre-existing condition and risk'),
        'd2_date': Property(Types.DATE,                       'Date of latest T2DM'),
        'd2_current_status': Property(Types.BOOL,             'Current T2DM status'),
        'd2_historic_status': Property(Types.CATEGORICAL,     'Historical status: N=never; C=Current, P=Previous',
                                                                 categories=['N', 'C', 'P']),
        'd2_death_date': Property(Types.DATE,                 'Date of scheduled death of infected individual'),

        # Define health care properties  #ToDO: to add more if needed once HSi coded
        'd2_diag_date': Property(Types.DATE,                  'Date of latest T2DM diagnosis'),
        'd2_diag_status': Property(Types.CATEGORICAL,         'Status: N=No; Y=Yes',
                                                                 categories=['N', 'Y']),
        'd2_treat_date': Property(Types.DATE,                 'Date of latest T2DM treatment'),
        'd2_treat_status': Property(Types.CATEGORICAL,        'Status: N=never; C=Current, P=Previous',
                                                                 categories=['N', 'C', 'P']),
        'd2_contr_date': Property(Types.DATE,                 'Date of latest T2DM control'),
        'd2_contr_status': Property(Types.CATEGORICAL,        'Status: N=No; Y=Yes',
                                                                 categories=['N', 'Y']),
        # TODO: update symptoms
        'd2_specific_symptoms': Property(Types.CATEGORICAL,   'Level of symptoms for T2DM specifically',
                                                                 categories=['none', 'mild sneezing', 'coughing and irritable', 'extreme emergency']),
        'd2_unified_symptom_code': Property(Types.CATEGORICAL,'Level of symptoms on the standardised scale (governing health-care seeking): '
                                                              '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
                                                                 categories=[0, 1, 2, 3, 4]),
    }

    def read_parameters(self, data_folder):
        """
        Reads in parameter values from file.

        Loads the 'Method_T2DM' file containing all parameter values for diabetes type 2 method

        :param data_folder: path of a folder supplied to the Simulation containing data files.
        """

        logger.debug('Type 2 Diabetes Mellitus method: reading in parameters.  ')

        # Reads in the parameters for hypertension from  file
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Method_T2DM.xlsx',
                                 sheet_name=None)

        p = self.parameters
        p['D2_prevalence'] = workbook['prevalence2018']  # reads in prevalence parameters
        p['D2_incidence'] = workbook['incidence2018_plus']  # reads in incidence parameters

        self.load_parameters_from_dataframe(workbook['parameters'])  # reads in parameters

        D2_data = workbook['data']  # reads in data
        df = D2_data.set_index('index')  # sets index

        # TODO: Asif/Stef to see if we can do the below shorter.

        # Read in prevalence data from file and put into model df
        p['initial_prevalence_d2'] = pd.DataFrame({'prevalence': [df.at['b_all', 'value'], df.at['b_25_35', 'value'],
                                                                  df.at['b_35_45', 'value'], df.at['b_45_55', 'value'],
                                                                  df.at['b_55_65', 'value']],
                                                   'min95ci': [df.at['b_all', 'min'], df.at['b_25_35', 'min'],
                                                               df.at['b_35_45', 'min'], df.at['b_45_55', 'min'],
                                                               df.at['b_55_65', 'min']],
                                                   'max95ci': [df.at['b_all', 'max'], df.at['b_25_35', 'max'],
                                                               df.at['b_35_45', 'max'], df.at['b_45_55', 'max'],
                                                               df.at['b_55_65', 'max']]
                                                   },
                                                  index=['total', '25_to_35', '35_to_45', '45_to_55', '55_to_65'])

        p['initial_prevalence_d2'].loc[:, 'prevalence'] *= 100  # Convert data to percentage
        p['initial_prevalence_d2'].loc[:, 'min95ci'] *= 100  # Convert data to percentage
        p['initial_prevalence_d2'].loc[:, 'max95ci'] *= 100  # Convert data to percentage

        # Get the DALY weight that diabetes type 2 will us from the weight database
        # TODO: check mapping of DALY to states of T2DM
        if 'HealthBurden' in self.sim.modules.keys():
            p['qalywt_mild_retino']   = self.sim.modules['HealthBurden'].get_daly_weight(967)
            p['qalywt_severe_retino'] = self.sim.modules['HealthBurden'].get_daly_weight(977)
            p['qalywt_uncomplicated'] = self.sim.modules['HealthBurden'].get_daly_weight(971)
            p['qalywt_neuropathy']    = self.sim.modules['HealthBurden'].get_daly_weight(970)

        logger.debug("Type 2 Diabetes Mellitus method: finished reading in parameters.  ")


    def initialise_population(self, population):
        """
        Assigns default values and prevalent cases of type 2 diabetes mellitus for the initial population.

        This method is called by the simulation and is responsible for assigning default values for every individuals of
        those properties 'owned' by the module, i.e. those declared in the PROPERTIES dictionary above.
        It also assigns prevalent values of type 2 diabetes mellitus at the start of the model.

        :param population: the population of individuals
        """

        logger.debug('Type 2 Diabetes Mellitus method: initialising population.  ')

        # Define key variables
        df = population.props                               # Shortcut to the data frame storing individual data
        m = self

        # Define default properties
        df.loc[df.is_alive,'d2_risk'] = 1.0                 # Default setting: no risk given pre-existing conditions
        df.loc[df.is_alive,'d2_current_status'] = False     # Default setting: no one has T2DM
        df.loc[df.is_alive,'d2_historic_status'] = 'N'      # Default setting: no one has T2DM
        df.loc[df.is_alive,'d2_date'] = pd.NaT              # Default setting: no one has T2DM
        df.loc[df.is_alive,'d2_death_date'] = pd.NaT        # Default setting: no one has T2DM
        df.loc[df.is_alive,'d2_diag_date'] = pd.NaT         # Default setting: no one is diagnosed
        df.loc[df.is_alive,'d2_diag_status'] = 'N'          # Default setting: no one is diagnosed
        df.loc[df.is_alive,'d2_treat_date'] = pd.NaT        # Default setting: no one is treated
        df.loc[df.is_alive,'d2_treat_status'] = 'N'         # Default setting: no one is treated
        df.loc[df.is_alive,'d2_contr_date'] = pd.NaT        # Default setting: no one is controlled
        df.loc[df.is_alive,'d2_contr_status'] = 'N'         # Default setting: no one is controlled
        df.loc[df.is_alive,'d2_specific_symptoms'] = 'none' # Default setting: no one has symptoms
        df.loc[df.is_alive,'d2_unified_symptom_code'] = 0   # Default setting: no one has symptoms
        # TODO: note that d2_age_range has not been given default

        # Assign prevalence as per data
        # Get corresponding risk
        d2_probability = df.loc[df.is_alive, ['d2_risk', 'age_years']].merge(self.parameters['D2_prevalence'],
                                                                             left_on=['age_years'], right_on=['age'],
                                                                             how='left')['probability']
        # TODO: update with BMI once merged to master
        df.loc[df.is_alive & df.li_overwt, 'd2_risk'] *= m.prob_htgivenbmi
        # TODO: add hypertension risk after circular declaration has been fixed with Asif/Stef
        # df.loc[df.is_alive & df.ht_current_status, 'd2_risk'] *= self.prob_d2givenht
        # TODO complete with other risk factors

        # Define key variables
        alive_count = df.is_alive.sum()
        assert alive_count == len(d2_probability)
        df.loc[df.is_alive, 'd2_age_range'] = df.loc[df.is_alive, 'age_years'].map(self.d2_age_range_lookup)

        # Assign prevalent cases of hypertension according to risk
        d2_probability = d2_probability * df.loc[df.is_alive, 'd2_risk']

        # TODO: remove duplication later. Asif and Stef: when the model runs with 1st random_number, numbers are not
        # TODO: [cont] random (younger and older age group is always very high. Even if seed is changed. When executing
        # TODO: [cont] a second time this is fine. Once fixed remove excess code up to below note.
        random_numbers = self.rng.random_sample(size=alive_count)
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'd2_current_status'] = (random_numbers < d2_probability)

        # Check random numbers      # TODO: Remove from here ...
        logger.debug('Lets generate the random number check for type 2 diabetes mellitus')

        pop_over_18 = df.index[(df.age_years >= 18) & (df.age_years < 25)]
        mean_randnum_18_25 = random_numbers[pop_over_18].mean()
        pop_over_25_35 = df.index[(df.age_years >= 25) & (df.age_years < 35)]
        mean_randnum_25_35 = random_numbers[pop_over_25_35].mean()
        pop_over_35_45 = df.index[(df.age_years >= 35) & (df.age_years < 45)]
        mean_randnum_35_45 = random_numbers[pop_over_35_45].mean()
        pop_over_45_55 = df.index[(df.age_years >= 45) & (df.age_years < 55)]
        mean_randnum_45_55 = random_numbers[pop_over_45_55].mean()
        pop_over_55 = df.index[df['age_years'] >= 55]
        mean_randnum_over55 = random_numbers[pop_over_55].mean()

        logger.debug('Lets generate the random number check for type 2 diabetes mellitus. ')
        logger.debug({
            'Mean rand num in 18-25 yo': mean_randnum_18_25,
            'Mean rand num_in 25-35 yo': mean_randnum_25_35,
            'Mean rand num_in 35-45 yo': mean_randnum_35_45,
            'Mean rand num_in 45-55 yo': mean_randnum_45_55,
            'Mean_rand num_in over 55 yo': mean_randnum_over55
        })

        assert 0.4 < mean_randnum_18_25 < 0.6
        assert 0.4 < mean_randnum_25_35 < 0.6
        assert 0.4 < mean_randnum_35_45 < 0.6
        assert 0.4 < mean_randnum_45_55 < 0.6
        assert 0.4 < mean_randnum_over55 < 0.6

        # TODO: ... to here

        # Assign relevant properties amongst those hypertensive
        count_d2_all = len(df[df.is_alive & df.d2_current_status])  # Count all people with type 2 diabetes mellitus
        d2_years_ago = np.array([1] * count_d2_all)
        infected_td_ago = pd.to_timedelta(d2_years_ago, unit='y')
        df.loc[df.is_alive & df.d2_current_status, 'd2_date'] = self.sim.date - infected_td_ago
        df.loc[df.is_alive & df.d2_current_status, 'd2_historic_status'] = 'C'

        # Assign level of symptoms # TODO: complete this section
        # level_of_symptoms = self.parameters['level_of_symptoms']
        # symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
        #                           size=T2DM_count,
        #                           p=level_of_symptoms.probability)
        # df.loc[df.d2_current_status, 'd2_specific_symptoms'] = symptoms

        # Set date of death # TODO: complete this section
        death_years_ahead = np.random.exponential(scale=20, size=count_d2_all)
        death_d2_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # Register this disease module with the health system  #TODO: TO DO LATER
        # self.sim.modules['HealthSystem'].register_disease_module(self)

        logger.debug("Type 2 Diabetes Mellitus method: finished initialising population.  ")

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.

        This method is called just before the main simulation begins, and after all
        parameters have been read in and prevalent hypertension has been assigned.
        It is a good place to add initial events to the event queue.

        :param sim: simulation
        """

        logger.debug("Type 2 Diabetes Mellitus method: initialising simulation.  ")

        # Add  basic event
        event = Type2DiabetesMellitusEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(years=1))

        # Launch the repeating event that will store statistics about type 2 diabetes mellitus
        sim.schedule_event(Type2DiabetesMellitusLoggingEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(Type2DiabetesMellitusLoggingValidationEvent(self), sim.date + DateOffset(days=0))


        # add the death event of infected individuals # TODO: could be used for testing later
        # schedule the mockitis death event
        #people_who_will_die = df.index[df.mi_is_infected]
        #for person_id in people_who_will_die:
        #    self.sim.schedule_event(HypDeathEvent(self, person_id),
        #                            df.at[person_id, 'mi_scheduled_date_death'])

        # Register this disease module with the health system
        #self.sim.modules['HealthSystem'].register_disease_module(self)

        # Schedule the outreach event... # ToDo: need to test this with T2DM!
        # event = HypOutreachEvent(self, 'this_module_only')
        # self.sim.schedule_event(event, self.sim.date + DateOffset(months=24))

        logger.debug("Type 2 Diabetes Mellitus method: finished initialising simulation.  ")

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        logger.debug("Type 2 Diabetes Mellitus method: on birth is being defined.  ")

        df = self.sim.population.props
        df.at[child_id, 'd2_risk'] = 1.0                        # Default setting: no risk given pre-existing conditions
        df.at[child_id, 'd2_current_status'] = False            # Default setting: no one has T2DM
        df.at[child_id, 'd2_historic_status'] = 'N'             # Default setting: no one has T2DM
        df.at[child_id, 'd2_date'] = pd.NaT                     # Default setting: no one has T2DM
        df.at[child_id, 'd2_death_date'] = pd.NaT               # Default setting: no one has T2DM
        df.at[child_id, 'd2_diag_date'] = pd.NaT                # Default setting: no one is diagnosed
        df.at[child_id, 'd2_diag_status'] = 'N'                 # Default setting: no one is diagnosed
        df.at[child_id, 'd2_treat_date'] = pd.NaT                # Default setting: no one is treated
        df.at[child_id, 'd2_treat_status'] = 'N'                 # Default setting: no one is treated
        df.at[child_id, 'd2_contr_date'] = pd.NaT               # Default setting: no one is controlled
        df.at[child_id, 'd2_contr_status'] = 'N'                # Default setting: no one is controlled
        df.loc[df.is_alive, 'd2_specific_symptoms'] = 'none'    # Default setting: no one has symptoms
        df.loc[df.is_alive, 'd2_unified_symptom_code'] = 0      # Default setting: no one has symptoms

    def on_healthsystem_interaction(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is T2DM, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


    def on_hsi_alert(self, person_id, treatment_id):  # TODO: update
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass


    def report_daly_values(self):  # TODO: update whole section
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('Type 2 Diabetes Mellitus: reporting  health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'd2_specific_symptoms'].map({
            'none': 0,
            'mild retinopathy': p['qalywt_mild_retino'],
            'severe retinopathy': p['qalywt_severe_retino'],
            'uncomplicated diabetes': p['qalywt_uncomplicated'],
            'neuropathy': p['qalywt_neuropathy']
        })
        return health_values.loc[df.is_alive]


class Type2DiabetesMellitusEvent(RegularEvent, PopulationScopeEventMixin):

    """
    This event is occurring regularly and controls the infection process of type 2 diabetes mellitus.
    It assigns new cases of diabetes and defines all related variables (e.g. date of diabetes)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1)) # TODO: change time scale if needed

        logger.debug('Type 2 Diabetes mellitus method: This is a type 2 diabetes mellitus Event')

    def apply(self, population):

        logger.debug('Type 2 diabetes mellitus method: tracking the disease progression in the population.')

        # 1. Basic variables
        df = population.props
        rng = self.module.rng

        t2dm_total = (df.is_alive & df.d2_current_status).sum()

        # 2. Get (and hold) index of people with and w/o T2DM
        currently_t2dm_yes = df[df.d2_current_status & df.is_alive].index
        currently_t2dm_no = df[~df.d2_current_status & df.is_alive].index
        age_index = df.loc[currently_t2dm_no, ['age_years']]
        alive_count = df.is_alive.sum()

        assert alive_count == len(currently_t2dm_yes) + len(currently_t2dm_no)

        # Calculate population prevalence
        if df.is_alive.sum():
            prevalence = len(currently_t2dm_yes) / (
                len(currently_t2dm_yes) + len(currently_t2dm_no))
        else:
            prevalence = 0

        # Calculate STEP prevalence
        #adults_count_overall = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 65)])
        #count = len(df[(df.is_alive) & (df.t2dm_current_status) & (df.age_years > 24) & (df.age_years < 65)])
        #prevalence_overall = (count / adults_count_overall) * 100

        #print("\n", "We are about to assign new T2DM cases. The time is: ", self.sim.date,
        #      "\n", "Population prevalence of T2DM is: ", prevalence,
        #      "\n", "Adult prevalence in STEP age-brackets is: ", prevalence_overall, "\n")

        # 3. Handle new cases of T2DM
        # 3.1 First get relative risk
        t2dm_prob = df.loc[currently_t2dm_no, ['age_years', 'd2_risk']].reset_index().merge(T2DM_incidence,
                                            left_on=['age_years'], right_on=['age'], how='left').set_index(
                                            'person')['probability']
        # df.loc[df.is_alive & ~df.hc_current_status, 'd2_risk'] = self.prob_T2DM_basic  # Basic risk, no pre-existing conditions
        # df.loc[df.is_alive & df.hc_current_status, 'd2_risk'] = self.prob_T2DMgivenHC  # Risk if pre-existing high cholesterol
        assert len(currently_t2dm_no) == len(t2dm_prob)
        t2dm_prob = t2dm_prob * df.loc[currently_t2dm_no, 'd2_risk']
        random_numbers = rng.random_sample(size=len(t2dm_prob))
        now_T2DM = (t2dm_prob > random_numbers)
        t2dm_idx = currently_t2dm_no[now_T2DM]

        # Check random numbers      #TODO: CHECK WITH TIM and remove later
        #random_numbers_df = random_numbers, index = currently_t2dm_no
        aaa = pd.DataFrame(data=random_numbers, index=currently_t2dm_no, columns=['nr'])

        over_18 = age_index.index[age_index.age_years > 17]
        a = aaa.loc[over_18]
        a = a.mean()

        over_25_35 = age_index.index[age_index.age_years > 24 & (age_index.age_years < 35)]
        b = aaa.loc[over_25_35]
        b = a.mean()

        over_35_45 = age_index.index[age_index.age_years > 34 & (age_index.age_years < 45)]
        c = aaa.loc[over_35_45]
        c = a.mean()

        over_45_55 = age_index.index[age_index.age_years > 44 & (age_index.age_years < 55)]
        d = aaa.loc[over_45_55]
        d = a.mean()

        over_55 = age_index.loc[age_index['age_years'] > 55].index
        e = aaa.loc[over_55]
        e = a.mean()

        print("\n", "Lets generate the random number check for incidence T2DM: ",
              "\n", "A: ", a,
              "\n", "B: ", b,
              "\n", "C: ", c,
              "\n", "D: ", d,
              "\n", "e: ", e, "\n")

        # 3.3 If newly diabetic
        df.loc[t2dm_idx, 'd2_current_status'] = True
        df.loc[t2dm_idx, 'd2_historic_status'] = 'C'
        df.loc[t2dm_idx, 'd2_date'] = self.sim.date

        # 3.2. Calculate prevalence
        # Count adults in different age groups
        adults_count_overall = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[(df.is_alive) & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[(df.is_alive) & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[(df.is_alive) & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_overall == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with T2DM by age group
        count = len(df[(df.is_alive) & (df.d2_current_status) & (df.age_years > 24) & (df.age_years < 65)])
        count_25to35 = len(df[(df.is_alive) & (df.d2_current_status) & (df.age_years > 24) & (df.age_years < 35)])
        count_35to45 = len(df[(df.is_alive) & (df.d2_current_status) & (df.age_years > 34) & (df.age_years < 45)])
        count_45to55 = len(df[(df.is_alive) & (df.d2_current_status) & (df.age_years > 44) & (df.age_years < 55)])
        count_55to65 = len(df[(df.is_alive) & (df.d2_current_status) & (df.age_years > 54) & (df.age_years < 65)])

        assert count == count_25to35 + count_35to45 + count_45to55 + count_55to65

        # Calculate overall and age-specific prevalence
        prevalence_overall = (count / adults_count_overall) * 100
        prevalence_25to35 = (count_25to35 / adults_count_25to35) * 100
        prevalence_35to45 = (count_35to45 / adults_count_35to45) * 100
        prevalence_45to55 = (count_45to55 / adults_count_45to55) * 100
        prevalence_55to65 = (count_55to65 / adults_count_55to65) * 100

        # 3.1 Log prevalence compared to data
        df2 = self.module.parameters['initial_prevalence']
        print("\n", "New cases of T2DM have been assigned. The time is: ", self.sim.date,
              "\n", "Prevalent T2DM has been assigned"
                    "\n", "MODEL: ",
              "\n", "both sexes: ", prevalence_overall, "%",
              "\n", "25 to 35:   ", prevalence_25to35, "%",
              "\n", "35 to 45:   ", prevalence_35to45, "%",
              "\n", "45 to 55:   ", prevalence_45to55, "%",
              "\n", "55 to 65:   ", prevalence_55to65, "%",
              "\n", "DATA: ", df2)

        print("\n", "Pause to check t2DM incidence - REMOVE LATER", "\n")

        # 3.2. Calculate prevalence
        #count = df.d2_current_status.sum()
        #prevalence = (sum(count) / alive_count) * 100

        # 4. Assign level of symptoms
        #level_of_symptoms = self.parameters['level_of_symptoms']
        #symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
        #                           size=t2dm_idx,
        #                           p=level_of_symptoms.probability)
        #df.loc[t2dm_idx, 'd2_specific_symptoms'] = symptoms

        # 5. Set date of death
        #death_years_ahead = np.random.exponential(scale=20, size=t2dm_idx)
        #death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

         # 3.3 If newly T2DM
        df.loc[t2dm_idx, 'd2_current_status'] = True
        df.loc[t2dm_idx, 'd2_historic_status'] = 'C'
        df.loc[t2dm_idx, 'd2_date'] = self.sim.date
       # df.loc[t2dm_idx, 'd2_death_date'] = self.sim.date + death_td_ahead



        print("\n", "Time is: ", self.sim.date, "New T2DM cases have been assigned.  "
              "\n", "Prevalence of T2DM is: ", prevalence, "%")

        # 6. Determine if anyone with severe symptoms will seek care
        serious_symptoms = (df['is_alive']) & ((df['d2_specific_symptoms'] == 'extreme emergency') | (
            df['d2_specific_symptoms'] == 'coughing and irritiable'))

        seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
        for i in df.index[serious_symptoms]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care == True]:
                logger.debug(
                    'This is T2DM, scheduling Hyp_PresentsForCareWithSevereSymptoms for person %d',
                    person_index)
                event = HSI_Hyp_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
            else:
                logger.debug(
                    'This is MockitisEvent, There is  no one with new severe symptoms so no new healthcare seeking')
        else:
            logger.debug('This is MockitisEvent, no one is newly infected.')


class T2DMDeathEvent(Event, IndividualScopeEventMixin):   # TODO: update
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'd2_current_status'] == 'C':
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='T2DM')
            self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_Hyp_PresentsForCareWithSevereSymptoms(Event, IndividualScopeEventMixin):

    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Mockitis.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hyp_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id):

        logger.debug('This is HSI_Hyp_PresentsForCareWithSevereSymptoms, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Hyp_PresentsForCareWithSevereSymptoms: there should now be treatment for person %d',
                person_id)
            event = HSI_Hyp_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

        else:
            logger.debug(
                '...This is HSI_Hyp_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_Hyp_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=date_turns_15,
                                                            tclose=date_turns_15 + DateOffset(months=12))


class HSI_Hyp_StartTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is inititaed.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation


        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
            0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
            0]

        item_code1 = \
        pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_Treatment_Initiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Mockitis_StartTreatment: initiating treatent for person %d', person_id)
        df = self.sim.population.props
        treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks:
            df.at[person_id, 'mi_is_infected'] = False
            df.at[person_id, 'mi_status'] = 'P'

            # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT

            df.at[person_id, 'mi_date_cure'] = self.sim.date
            df.at[person_id, 'mi_specific_symptoms'] = 'none'
            df.at[person_id, 'mi_unified_symptom_code'] = 0
            pass

        # Create a follow-up appointment
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug('....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
                     person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=target_date_for_followup_appt,
                                                        tclose=target_date_for_followup_appt + DateOffset(weeks=2)
        )




class HSI_Mockitis_TreatmentMonitoring(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is monitored.
    (In practise, nothing happens!)

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation


        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
            0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
            0]

        item_code1 = \
        pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_TreatmentMonitoring'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id):

        # There is a follow-up appoint happening now but it has no real effect!

        # Create the next follow-up appointment....
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            '....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=target_date_for_followup_appt,
                                                        tclose=target_date_for_followup_appt + DateOffset(weeks=2))



# ---------------------------------------------------------------------------------



class T2DMLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        # infected_total = df.loc[df.is_alive, 'mi_is_infected'].sum()
        # proportion_infected = infected_total / len(df)
        #
        # mask: pd.Series = (df.loc[df.is_alive, 'mi_date_infected'] >
        #                    self.sim.date - DateOffset(months=self.repeat))
        # infected_in_last_month = mask.sum()
        # mask = (df.loc[df.is_alive, 'mi_date_cure'] > self.sim.date - DateOffset(months=self.repeat))
        # cured_in_last_month = mask.sum()
        #
        # counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        # counts.update(df.loc[df.is_alive, 'mi_status'].value_counts().to_dict())
        #
        # logger.info('%s|summary|%s', self.sim.date,
        #             {
        #                 'TotalInf': infected_total,
        #                 'PropInf': proportion_infected,
        #                 'PrevMonth': infected_in_last_month,
        #                 'Cured': cured_in_last_month,
        #             })
        #
        # logger.info('%s|status_counts|%s', self.sim.date, counts)
