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
# TODO: symptoms - waiting for Tara

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
    Version 2 - October 2019
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
        'd2_prevalence': Parameter(Types.REAL, 'T2DM prevalence'),
        'd2_incidence': Parameter(Types.REAL, 'T2DM incidence'),
        'prob_d2_basic': Parameter(Types.REAL, 'Basic T2DM probability'),
        'prob_d2givenbmi': Parameter(Types.REAL, 'T2DM probability given being BMI'),
        'prob_d2givenht': Parameter(Types.REAL, 'T2DM probability given pre-existing hypertension'),
        'prob_d2givengd': Parameter(Types.REAL, 'T2DM probability given pre-existing gestational diabetes'),
        'initial_prevalence_d2': Parameter(Types.REAL, 'Prevalence of T2DM as per data'),
        'initial_complication_prevalence_d2': Parameter(Types.REAL, 'Prevalence of T2DM complications as per data'),

        # Define disease progression parameters
        # At the start of the model, parameters to assign prevalence of complications
        'prev_nephro_1': Parameter(Types.REAL, 'Probability of having mild nephropathy'),
        'prev_nephro_2': Parameter(Types.REAL, 'Probability of having moderate nephropathy'),
        'prev_neuro_1': Parameter(Types.REAL, 'Probability of having mild peripheral neuropathy'),
        'prev_neuro_2': Parameter(Types.REAL, 'Probability of having moderate peripheral neuropathy'),
        'prev_retino_1': Parameter(Types.REAL, 'Probability of having severe retinopathy'),
        'prev_retino_2': Parameter(Types.REAL, 'Probability of having severe retinopathy'),

        # During model run, parameters to assign incidence of new complication
        'prob_nephrocomp_1': Parameter(Types.REAL, 'Probability of developing mild diabetic nephropathy'),
        'prob_nephrocomp_2': Parameter(Types.REAL, 'Probability of developing severe diabetic nephropathy'),
        'prob_neurocomp_1': Parameter(Types.REAL, 'Probability of developing mild peripheral neuropathy'),
        'prob_neurocomp_2': Parameter(Types.REAL, 'Probability of developing severe peripheral neuropathy'),
        'prob_retinocomp_1': Parameter(Types.REAL, 'Probability of developing mild retinopathy'),
        'prob_retinocomp_2': Parameter(Types.REAL, 'Probability of developing severe retinopathy'),
        't_to_increase_complication_severity': Parameter(Types.REAL, 'Time until severity of complication increases'),
        'prob_death': Parameter(Types.REAL, 'Probability of dying'),

        # Define health care parameters
        # TODO: update after HSI events
        # TODO: update with DALY weights
        #'level_of_symptoms': Parameter(Types.CATEGORICAL,      'Severity of symptoms that the individual will have'),
        'dalywt_uncomplicated': Parameter(Types.REAL,          'DALY weighting for uncomplicated diabetic'),
        'dalywt_neuropathy': Parameter(Types.REAL,             'DALY weighting for neuropathy'),
        'dalywt_one_amputation': Parameter(Types.REAL,         'DALY weighting for amputation of one limp'),
        'dalywt_two_amputations': Parameter(Types.REAL,        'DALY weighting for amputation of two limps'),
        'dalywt_nephropathy_mild': Parameter(Types.REAL,       'DALY weighting for mild nephropathy'),
        'dalywt_nephropathy_severe': Parameter(Types.REAL,     'DALY weighting for severe nephropathy'),
        'dalywt_neuropathy': Parameter(Types.REAL,             'DALY weighting for neuropathy'),
        'dalywt_one_amputation': Parameter(Types.REAL,         'DALY weighting for amputation of one limp'),
        'dalywt_two_amputations': Parameter(Types.REAL,        'DALY weighting for amputation of two limps'),
        'dalywt_mild_retino': Parameter(Types.REAL, 'DALY weighting for mild retinopathy'),
        'dalywt_severe_retino': Parameter(Types.REAL, 'DALY weighting for severe retinopathy'),

    }

    PROPERTIES = {

        # Next we declare the properties of individuals that this module provides.
        # Again each has a name, type and description.

        # Note that all properties must have a two letter prefix that identifies them to this module.

        # ToDo: Update symptoms name

        # Define disease properties
        'd2_age_range': Property(Types.CATEGORICAL, 'The age range categories for t2dm validation use',
                                 categories=d2_age_range_categories),
        'd2_risk': Property(Types.REAL,                       'Risk of T2DM given pre-existing condition and risk'),
        'd2_date': Property(Types.DATE,                       'Date of latest T2DM'),
        'd2_current_status': Property(Types.BOOL,             'Current T2DM status'),
        'd2_historic_status': Property(Types.CATEGORICAL,     'Historical status: N=never; C=Current, P=Previous',
                                                                 categories=['N', 'C', 'P']),
        'd2_nephro_date': Property(Types.DATE,                'Date of latest nephropathy complication'),
        'd2_nephro_status': Property(Types.CATEGORICAL,       'Level of nephropathy: none, mild, severe referring to none, reduced function and end-stage',
                                                                categories=[0, 1, 2]),
        'd2_neuro_date': Property(Types.DATE,                 'Date of latest neuropathy complication'),
        'd2_neuro_status': Property(Types.CATEGORICAL,        'Level of neuropathy: none, mild, moderate, severe referring to none, numbness, 1 and 2 amputations',
                                                                categories=[0, 1, 2, 3]),
        'd2_retino_date': Property(Types.DATE,                'Date of latest retinopathy complication'),
        'd2_retino_status': Property(Types.CATEGORICAL,       'Level of retinopathy: none, mild, severe referring to none, any and severely visually impaired',
                                                                categories=[0, 1, 2]),
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

        # Reads in the parameters for t2dm from  file
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Method_T2DM.xlsx',
                                 sheet_name=None)

        p = self.parameters
        p['d2_prevalence'] = workbook['prevalence2018']  # reads in prevalence parameters
        p['d2_incidence'] = workbook['incidence2018_plus']  # reads in incidence parameters

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

        # Read in prevalence data on complications from file
        p['initial_complication_prevalence_d2'] = pd.DataFrame({'prevalence': [df.at['p_neuro_1', 'value'], df.at['p_neuro_2', 'value'],
                                                                  df.at['p_nephro_1', 'value'], df.at['p_nephro_2', 'value'],
                                                                  df.at['p_retino_1', 'value'], df.at['p_retino_2', 'value']],
                                                   },
                                                  index=['Neuro_mild', 'Neuro_sev', 'Nephro_mild', 'Nephro_sev', 'Retino_mild', 'Retino_sev'])
        p['initial_complication_prevalence_d2'].loc[:, 'prevalence'] *= 100  # Convert data to percentage


        # Get the DALY weight that diabetes type 2 will us from the weight database
        # TODO: check mapping of DALY to states of T2DM
        if 'HealthBurden' in self.sim.modules.keys():
            p['qalywt_uncomplicated'] = self.sim.modules['HealthBurden'].get_daly_weight(971)

            p['dalywt_nephropathy_mild'] = self.sim.modules['HealthBurden'].get_daly_weight(989)
            p['dalywt_nephropathy_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(987)

            p['qalywt_neuropathy'] = self.sim.modules['HealthBurden'].get_daly_weight(970)
            p['dalywt_one_amputation'] = 0.164  # no value in 2016 DALY weight, value taken from 2010 w/o treatment
            p['dalywt_two_amputations'] = 0.494  # no value in 2016 DALY weight, value taken from 2010 w/o treatment

            p['qalywt_mild_retino'] = self.sim.modules['HealthBurden'].get_daly_weight(967)
            p['qalywt_severe_retino'] = self.sim.modules['HealthBurden'].get_daly_weight(974)

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
        df.loc[df.is_alive, 'd2_risk'] = 1.0                 # Default setting: no risk given pre-existing conditions
        df.loc[df.is_alive, 'd2_current_status'] = False     # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_historic_status'] = 'N'      # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_date'] = pd.NaT              # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_nephro_date'] = pd.NaT       # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_nephro_status'] = 0          # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_neuro_date'] = pd.NaT  # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_neuro_status'] = 0  # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_retino_date'] = pd.NaT  # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_retino_status'] = 0  # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_death_date'] = pd.NaT        # Default setting: no one has T2DM
        df.loc[df.is_alive, 'd2_diag_date'] = pd.NaT         # Default setting: no one is diagnosed
        df.loc[df.is_alive, 'd2_diag_status'] = 'N'          # Default setting: no one is diagnosed
        df.loc[df.is_alive, 'd2_treat_date'] = pd.NaT        # Default setting: no one is treated
        df.loc[df.is_alive, 'd2_treat_status'] = 'N'         # Default setting: no one is treated
        df.loc[df.is_alive, 'd2_contr_date'] = pd.NaT        # Default setting: no one is controlled
        df.loc[df.is_alive, 'd2_contr_status'] = 'N'         # Default setting: no one is controlled
        df.loc[df.is_alive, 'd2_specific_symptoms'] = 'none' # Default setting: no one has symptoms
        df.loc[df.is_alive, 'd2_unified_symptom_code'] = 0   # Default setting: no one has symptoms
        # TODO: note that d2_age_range has not been given default

        # Assign prevalence as per data
        # Get corresponding risk
        d2_probability = df.loc[df.is_alive, ['d2_risk', 'age_years']].merge(self.parameters['d2_prevalence'],
                                                                             left_on=['age_years'], right_on=['age'],
                                                                             how='left')['probability']
        # TODO: update with BMI once merged to master
        df.loc[df.is_alive & df.li_overwt, 'd2_risk'] *= m.prob_d2givenbmi
        # TODO: add hypertension risk after circular declaration has been fixed with Asif/Stef
        # df.loc[df.is_alive & df.ht_current_status, 'd2_risk'] *= m.prob_d2givenht
        # TODO complete with other risk factors

        # Define key variables
        alive_count = df.is_alive.sum()
        assert alive_count == len(d2_probability)
        df.loc[df.is_alive, 'd2_age_range'] = df.loc[df.is_alive, 'age_years'].map(self.d2_age_range_lookup)

        # Assign prevalent cases of t2dm according to risk
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

        # Assign relevant properties amongst those t2dm
        count_d2_all = len(df[df.is_alive & df.d2_current_status])  # Count all people with type 2 diabetes mellitus
        d2_years_ago = np.array([1] * count_d2_all)
        infected_td_ago = pd.to_timedelta(d2_years_ago, unit='y')
        df.loc[df.is_alive & df.d2_current_status, 'd2_date'] = self.sim.date - infected_td_ago
        df.loc[df.is_alive & df.d2_current_status, 'd2_historic_status'] = 'C'

        # Assign complication
        # TODO: not sure if there is a more efficient way to code the below?
        # Nephropathy
        currently_d2_yes = df[df.d2_current_status & df.is_alive].index  # hold index of all those who have diabetes
        random_numbers = self.rng.random_sample(size=len(currently_d2_yes))  # Get random numbers
        now_nephro_1 = (random_numbers < m.prev_nephro_1)  # assign nephropathy amongst those with diabetes
        now_nephro_2 = (random_numbers < m.prev_nephro_2)  # amongst those with nephropathy assign moderate to severe nephropathy
        d2_idx_nephro1 = currently_d2_yes[now_nephro_1]  # hold id of those with nephropathy
        d2_idx_nephro2 = currently_d2_yes[now_nephro_2]  # hold id of those with mod-severe nephropathy

        # Now update the df
        df.loc[d2_idx_nephro1, 'd2_nephro_status'] = 1
        df.loc[d2_idx_nephro2, 'd2_nephro_status'] = 2

        # Neuropathy
        random_numbers = self.rng.random_sample(size=len(currently_d2_yes))  # Get random numbers
        now_neuro_1 = (random_numbers < m.prev_neuro_1)  # assign neuropathy amongst those with diabetes
        now_neuro_2 = (
                random_numbers < m.prev_neuro_2)  # amongst those with neuropathy assign moderate to severe neuropathy
        d2_idx_neuro1 = currently_d2_yes[now_neuro_1]  # hold id of those with neuropathy
        d2_idx_neuro2 = currently_d2_yes[now_neuro_2]  # hold id of those with mod-severe neuropathy

        # Now update the df
        df.loc[d2_idx_neuro1, 'd2_neuro_status'] = 1
        df.loc[d2_idx_neuro2, 'd2_neuro_status'] = 2

        # Retinopathy
        random_numbers = self.rng.random_sample(size=len(currently_d2_yes))  # Get random numbers
        now_retino_1 = (random_numbers < m.prev_retino_1)  # assign retinopathy amongst those with diabetes
        now_retino_2 = (random_numbers < m.prev_retino_2)  # amongst those with retinopathy assign severe retinopathy
        d2_idx_retino1 = currently_d2_yes[now_retino_1]  # hold id of those with retinopathy
        d2_idx_retino2 = currently_d2_yes[now_retino_2]  # hold id of those with severe retinopathy

        # Now update the df
        df.loc[d2_idx_retino1, 'd2_retino_status'] = 1
        df.loc[d2_idx_retino2, 'd2_retino_status'] = 2


        # Assign level of symptoms # TODO: complete this section once symptoms haev been finalised
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
        parameters have been read in and prevalent t2dm has been assigned.
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
    This event is occurring regularly and controls the disease process of type 2 diabetes mellitus.
    It assigns new cases of diabetes and defines all related variables (e.g. date of diabetes)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1)) # TODO: change time scale if needed

        logger.debug('Type 2 Diabetes mellitus method: This is a type 2 diabetes mellitus Event')

    def apply(self, population):

        logger.debug('Type 2 diabetes mellitus method: tracking the disease progression in the population.')

        # Basic variables
        df = population.props
        m = self.module
        rng = self.module.rng

        # Get (and hold) index of people with and w/o T2DM
        currently_d2_yes = df[df.d2_current_status & df.is_alive].index  # holds all people with t2dm
        currently_d2_no = df[~df.d2_current_status & df.is_alive].index  # hold all people w/o t2dm
        age_index = df.loc[currently_d2_no, ['age_years']]  # looks up age index of those w/o t2dm
        alive_count = df.is_alive.sum()  # counts all those with t2dm

        assert alive_count == len(currently_d2_yes) + len(currently_d2_no)  # checks that we didn't loose anyone

        # Handle new cases of T2DM
        # First get relative risk
        d2_probability = df.loc[currently_d2_no,
                         ['age_years', 'd2_risk']].reset_index().merge(self.module.parameters['d2_incidence'],
                                                                       left_on=['age_years'],
                                                                       right_on=['age'],
                                                                       how='left').set_index('person')['probability']
        df['d2_risk'] - 1.0 # Reset risk for all people
        # TODO: update with BMI once merged to master
        df.loc[df.is_alive & df.li_overwt, 'd2_risk'] *= m.prob_d2givenbmi  # Adjust risk if overwt
        # TODO: add hypertension and GD once cricular declaration has been fixed with Asif/Stef and GD is coded
        # df.loc[df.is_alive & ~df.hc_current_status, 'd2_risk'] = self.prob_d2_basic  # Adjust risk if hypertension
        # df.loc[df.is_alive & df.gd_current_status, 'd2_risk'] = self.prob_d2givengd  # Adjust risk if gestational diabetes

        d2_probability = d2_probability * df.loc[currently_d2_no, 'd2_risk']  # look up updated risk for t2dm
        random_numbers = rng.random_sample(size=len(d2_probability))  # Get random numbers
        now_d2 = (d2_probability > random_numbers)  # assign new t2dm cases as per incidence and risk
        d2_idx = currently_d2_no[now_d2]  # hold id of those with new t2dm

        # Check random numbers      #TODO: Remove from here .... (not pretty code - just for testing)
        #random_numbers_df = random_numbers, index = currently_d2_no
        randnum_holder = pd.DataFrame(data=random_numbers, index=currently_d2_no, columns=['nr'])

        pop_over_18 = age_index.index[(age_index.age_years >= 18) & (age_index.age_years < 25)]
        rand_18_25 = (randnum_holder.loc[pop_over_18]).mean()
        rand_18_25_mean = rand_18_25.mean()

        pop_over_25_35 = age_index.index[(age_index.age_years >= 25) & (age_index.age_years < 35)]
        rand_25_35 = (randnum_holder.loc[pop_over_25_35]).mean()
        rand_25_35_mean = rand_25_35.mean()

        pop_over_35_45 = age_index.index[(age_index.age_years >= 35) & (age_index.age_years < 45)]
        rand_35_45 = (randnum_holder.loc[pop_over_35_45]).mean()
        rand_35_45_mean = rand_35_45.mean()

        pop_over_45_55 = age_index.index[(age_index.age_years >= 45) & (age_index.age_years < 55)]
        rand_45_55 = (randnum_holder.loc[pop_over_45_55]).mean()
        rand_45_55_mean = rand_45_55.mean()

        pop_over_55 = age_index.index[(age_index.age_years >= 55)]
        rand_over55 = (randnum_holder.loc[pop_over_55]).mean()
        rand_over55_mean = rand_over55.mean()

        logger.debug('Lets generate the random number check for incident t2dm. ')
        logger.debug({
            'Mean rand num in 18-25 yo': rand_18_25,
            'Mean rand num_in 25-35 yo': rand_25_35,
            'Mean rand num_in 35-45 yo': rand_35_45,
            'Mean rand num_in 45-55 yo': rand_45_55,
            'Mean_rand num_in over 55 yo': rand_over55
        })

        assert 0.4 < rand_18_25_mean < 0.6
        assert 0.4 < rand_25_35_mean < 0.6
        assert 0.4 < rand_35_45_mean < 0.6
        assert 0.4 < rand_45_55_mean < 0.6
        assert 0.4 < rand_over55_mean < 0.6

        # TODO: ... to here

        # Assign variables amongst those with new cases of t2dm
        df.loc[d2_idx, 'd2_current_status'] = True
        df.loc[d2_idx, 'd2_historic_status'] = 'C'
        df.loc[d2_idx, 'd2_date'] = self.sim.date


        # Assign complication amongst anyone who has diabetes
        # TODO: not sure if there is a more efficient way to code the below?
        # Nephropathy
        random_numbers = self.rng.random_sample(size=len(d2_idx))  # Get random numbers
        now_nephro_1 = (random_numbers < m.prev_nephro_1)  # assign nephropathy amongst those with diabetes
        now_nephro_2 = (
                random_numbers < m.prev_nephro_2)  # amongst those with nephropathy assign moderate to severe nephropathy
        d2_idx_nephro1 = d2_idx[now_nephro_1]  # hold id of those with nephropathy
        d2_idx_nephro2 = d2_idx[now_nephro_2]  # hold id of those with mod-severe nephropathy

        # Now update the df
        df.loc[d2_idx_nephro1, 'd2_nephro_status'] = 1
        df.loc[d2_idx_nephro2, 'd2_nephro_status'] = 2

        # Neuropathy
        random_numbers = self.rng.random_sample(size=len(d2_idx))  # Get random numbers
        now_neuro_1 = (random_numbers < m.prev_neuro_1)  # assign neuropathy amongst those with diabetes
        now_neuro_2 = (
            random_numbers < m.prev_neuro_2)  # amongst those with neuropathy assign moderate to severe neuropathy
        d2_idx_neuro1 = d2_idx[now_neuro_1]  # hold id of those with neuropathy
        d2_idx_neuro2 = d2_idx[now_neuro_2]  # hold id of those with mod-severe neuropathy

        # Now update the df
        df.loc[d2_idx_neuro1, 'd2_neuro_status'] = 1
        df.loc[d2_idx_neuro2, 'd2_neuro_status'] = 2

        # Retinopathy
        random_numbers = self.rng.random_sample(size=len(d2_idx))  # Get random numbers
        now_retino_1 = (random_numbers < m.prev_retino_1)  # assign retinopathy amongst those with diabetes
        now_retino_2 = (random_numbers < m.prev_retino_2)  # amongst those with retinopathy assign severe retinopathy
        d2_idx_retino1 = d2_idx[now_retino_1]  # hold id of those with retinopathy
        d2_idx_retino2 = d2_idx[now_retino_2]  # hold id of those with severe retinopathy

        # Now update the df
        df.loc[d2_idx_retino1, 'd2_retino_status'] = 1
        df.loc[d2_idx_retino2, 'd2_retino_status'] = 2

        # TODO: add mortality


class T2DMDeathEvent(Event, IndividualScopeEventMixin):   # TODO: update
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        #if df.at[person_id, 'd2_current_status'] == 'C':
            # Fire the centralised death event:
            #death = InstantaneousDeath(self.module, person_id, cause='T2DM')
            #self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Logging events


class Type2DiabetesMellitusLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This Logging event logs the model prevalence for analysis purposed.
        This Logging event is different to the validation  logging event as it logs prevalence by standard age groups
        used throughout the model. The validation logging event logs prevalence generated by the model to the age
        groups used in the data to ensure the model is running well.
        """

        self.repeat = 12  # run this event every year
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # This code will calculate the prevalence and log it

        df = population.props
        p = self.module.parameters

        # Calculate prevalence
        count_by_age = df[df.is_alive].groupby('age_range').size()
        count_d2_by_age = df[df.is_alive & (df.d2_current_status)].groupby('age_range').size()
        prevalence_d2_by_age = (count_d2_by_age / count_by_age) * 100
        prevalence_d2_by_age.fillna(0, inplace=True)

        # Log prevalence
        logger.info('%s|d2_prevalence|%s', self.sim.date, prevalence_d2_by_age.to_dict())

        # Calculate prevalence of complications
        prev_nephro_1 = (df.d2_nephro_status.value_counts()[1] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_nephro_2 = (df.d2_nephro_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_neuro_1 = (df.d2_neuro_status.value_counts()[1] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_neuro_2 = (df.d2_neuro_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_retino_1 = (df.d2_retino_status.value_counts()[1] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_retino_2 = (df.d2_retino_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100

        # Log prevalence of complications for validation
        logger.info('%s|d2_complications|%s', self.sim.date,
                    {
                        'Nephro_mild': prev_nephro_1,
                        'Nephro_sev': prev_nephro_2,
                        'Neuro_mild': prev_neuro_1,
                        'Neuro_sev': prev_neuro_2,
                        'Retino_mild': prev_retino_1,
                        'Retino_sev': prev_retino_2
                    })


class Type2DiabetesMellitusLoggingValidationEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This Logging event logs the data and the model prevalence for validation purposed.
        This Logging event is different than the regular model logging event as it logs prevalence by age groups to
        match those reported in the data. The regular logging event logs prevalence generated by the model to the age
        groups used throughout the model.
        """

        self.repeat = 12  # run this event every year # TODO: only run this for the first 5 years for validation
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # This code logs the prevalence for the first 5 years from the data and the model.
        # Assumption is that T2DM prevalence will remain stable over 5 years and thus equal to 2010 data

        df = population.props
        p = self.module.parameters

        # Log the data every year (for plotting)
        # TODO: Can this codebelow be shorten (i.e. read in the whole frame?
        # TODO: Can this be logged only first year
        logger.info('%s|test|%s',
                    self.sim.date,
                    {
                        'total': p['initial_prevalence_d2'].loc['total', 'prevalence']})

        logger.info('%s|d2_prevalence_data_validation|%s',
                    self.sim.date,
                    {
                        'total': p['initial_prevalence_d2'].loc['total', 'prevalence'],
                        'total_min': p['initial_prevalence_d2'].loc['total', 'min95ci'],
                        'total_max': p['initial_prevalence_d2'].loc['total', 'max95ci'],
                        'age25to35': p['initial_prevalence_d2'].loc['25_to_35', 'prevalence'],
                        'age25to35_min': p['initial_prevalence_d2'].loc['25_to_35', 'min95ci'],
                        'age25to35_max': p['initial_prevalence_d2'].loc['25_to_35', 'max95ci'],
                        'age35to45': p['initial_prevalence_d2'].loc['35_to_45', 'prevalence'],
                        'age35to45_min': p['initial_prevalence_d2'].loc['35_to_45', 'min95ci'],
                        'age35to45_max': p['initial_prevalence_d2'].loc['35_to_45', 'max95ci'],
                        'age45to55': p['initial_prevalence_d2'].loc['45_to_55', 'prevalence'],
                        'age45to55_min': p['initial_prevalence_d2'].loc['45_to_55', 'min95ci'],
                        'age45to55_max': p['initial_prevalence_d2'].loc['45_to_55', 'max95ci'],
                        'age55to65': p['initial_prevalence_d2'].loc['55_to_65', 'prevalence'],
                        'age55to65_min': p['initial_prevalence_d2'].loc['55_to_65', 'min95ci'],
                        'age55to65_max': p['initial_prevalence_d2'].loc['55_to_65', 'max95ci']
                    })

        # Calculate prevalence from the model     #TODO: use groupby (make new cats) -  remove after check
        adults_count_all = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[df.is_alive & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[df.is_alive & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[df.is_alive & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_all == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with t2dm by age group
        count_d2 = len(df[df.is_alive & df.d2_current_status & (df.age_years > 24) & (df.age_years < 65)])
        count_d2_25to35 = len(df[df.is_alive & df.d2_current_status & (df.age_years > 24) & (df.age_years < 35)])
        count_d2_35to45 = len(df[df.is_alive & df.d2_current_status & (df.age_years > 34) & (df.age_years < 45)])
        count_d2_45to55 = len(df[df.is_alive & df.d2_current_status & (df.age_years > 44) & (df.age_years < 55)])
        count_d2_55to65 = len(df[df.is_alive & df.d2_current_status & (df.age_years > 54) & (df.age_years < 65)])

        assert count_d2 == count_d2_25to35 + count_d2_35to45 + count_d2_45to55 + count_d2_55to65

        # Calculate overall and age-specific prevalence
        prevalence_overall = (count_d2 / adults_count_all) * 100
        prevalence_25to35 = (count_d2_25to35 / adults_count_25to35) * 100
        prevalence_35to45 = (count_d2_35to45 / adults_count_35to45) * 100
        prevalence_45to55 = (count_d2_45to55 / adults_count_45to55) * 100
        prevalence_55to65 = (count_d2_55to65 / adults_count_55to65) * 100

        assert prevalence_overall > 0
        assert prevalence_25to35 > 0
        assert prevalence_35to45 > 0
        assert prevalence_45to55 > 0
        assert prevalence_55to65 > 0

        # Log prevalence from the model
        logger.info('%s|d2_prevalence_model_validation|%s',
                    self.sim.date,
                    {
                        'total': prevalence_overall,
                        '25to35': prevalence_25to35,
                        '35to45': prevalence_35to45,
                        '45to55': prevalence_45to55,
                        '55to65': prevalence_55to65
                    })

        # TODO: remove up to here

        # Calculate prevalence from the model using groupby instead
        # TODO: use groupby (make new cats) below instead - check it is working

        # First by age
        count_by_age_val = df[df.is_alive].groupby('d2_age_range').size()
        count_d2_by_age_val = df[df.is_alive & df.d2_current_status].groupby('d2_age_range').size()
        prevalence_d2_by_age_val = (count_d2_by_age_val / count_by_age_val) * 100
        prevalence_d2_by_age_val.fillna(0, inplace=True)

        # Then overall
        prevalence_d2_all_val = count_d2_by_age_val[0:4].sum() / count_by_age_val[0:4].sum() * 100

        # Log prevalence
        logger.info('%s|d2_prevalence_model_validation_2|%s', self.sim.date,
                    {'total': prevalence_d2_all_val,
                     '25to35': prevalence_d2_by_age_val.iloc[0],
                     '35to45': prevalence_d2_by_age_val.iloc[1],
                     '45to55': prevalence_d2_by_age_val.iloc[2],
                     '55to65': prevalence_d2_by_age_val.iloc[3]
                     })

        # Log the data on complication
        logger.info('%s|d2_complication_data_validation|%s',
                    self.sim.date,
                    {
                        'Nephro_mild': p['initial_complication_prevalence_d2'].loc['Nephro_mild', 'prevalence'],
                        'Nephro_sev': p['initial_complication_prevalence_d2'].loc['Nephro_sev', 'prevalence'],
                        'Neuro_mild': p['initial_complication_prevalence_d2'].loc['Neuro_mild', 'prevalence'],
                        'Neuro_sev': p['initial_complication_prevalence_d2'].loc['Neuro_sev', 'prevalence'],
                        'Retino_mild': p['initial_complication_prevalence_d2'].loc['Retino_mild', 'prevalence'],
                        'Retino_sev': p['initial_complication_prevalence_d2'].loc['Retino_sev', 'prevalence']
                    })


        # Calculate prevalence of complications
        prev_nephro_1 = ((df.d2_nephro_status.value_counts()[1] + df.d2_nephro_status.value_counts()[2]) / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_nephro_2 = (df.d2_nephro_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_neuro_1 = ((df.d2_neuro_status.value_counts()[1] + df.d2_neuro_status.value_counts()[2]) / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_neuro_2 = (df.d2_neuro_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_retino_1 = ((df.d2_retino_status.value_counts()[1] + df.d2_retino_status.value_counts()[2]) / len(df[df.d2_current_status & df.is_alive])) * 100
        prev_retino_2 = (df.d2_retino_status.value_counts()[2] / len(df[df.d2_current_status & df.is_alive])) * 100

        # Log prevalence of complications for validation
        logger.info('%s|d2_complications_model_validation|%s', self.sim.date,
                    {
                        'Nephro_mild': prev_nephro_1,
                        'Nephro_sev': prev_nephro_2,
                        'Neuro_mild': prev_neuro_1,
                        'Neuro_sev': prev_neuro_2,
                        'Retino_mild': prev_retino_1,
                        'Retino_sev': prev_retino_2
                    })




