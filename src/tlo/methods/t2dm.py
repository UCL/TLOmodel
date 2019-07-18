"""
This is the method for type 2 diabetes
Developed by Mikaela Smit, October 2018

"""

import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Read in data
file_path = 'resources/ResourceFile_Method_T2DM.xlsx'
method_T2DM_data = pd.read_excel(file_path, sheet_name=None, header=0)
T2DM_prevalence, T2DM_incidence, T2DM_risk, T2DM_data = method_T2DM_data['prevalence2018'], method_T2DM_data['incidence2018_plus'], \
                                                        method_T2DM_data['parameters'], method_T2DM_data['data']

# TODO: Read in 95% CI from file?
# TODO: Update weight to BMI AND ADAPT TO UPDATED CODE FOR WEIGHT!
# TODO: Triple check that DALYs are correct
# ToDO: Update complication, symptoms and QALYs in line with DALY file
# TODO: how to handle symptoms and how specific to be - develop
# TODO: What about Qaly for mild symptoms (urination, tiredness etc), nephrology, amputation, and NOTE: severe vision impairmenet and blindness same qaly so only one


class T2DM(Module):
    """
    This is type 2 diabetes.
    It demonstrates the following behaviours in respect of the healthsystem module:

    - Declaration of TREATMENT_ID
    - Registration of the disease module
    - Reading DALY weights and reporting daly values related to this disease
    - Health care seeking
    - Running an "outreach" event
    """

    print("\n", "Type 2 Diabetes mellitus method is running", "\n")

    PARAMETERS = {

        # 1. Risk parameters
        'prob_T2DM_basic': Parameter(Types.REAL,               'Probability of T2DM given no pre-existing condition'),
        'prob_T2DMgivenBMI': Parameter(Types.REAL,             'Probability of getting T2DM given being BMI'),
        'prob_T2DMgivenHT': Parameter(Types.REAL,              'Probability of getting T2DM given pre-existing hypertension'),
        'prob_T2DMgivenGD': Parameter(Types.REAL,              'Probability of getting T2DM given pre-existing gestational diabetes'),
        #'prob_T2DMgivenFamHis': Parameter(Types.REAL,          'Probability of getting T2DM given family history'),
        'prob_RetinoComp': Parameter(Types.REAL,               'Probability of developing retinopathy'),
        'prob_NephroComp': Parameter(Types.REAL,               'Probability of developing diabetic nephropathy'),
        'prob_NeuroComp': Parameter(Types.REAL,                'Probability of developing peripheral neuropathy'),
        'prob_death': Parameter(Types.REAL,                    'Probability of dying'),

        # 2. Health care parameters
        'prob_diag': Parameter(Types.REAL,                     'Probability of being diagnosed'),
        'prob_treat': Parameter(Types.REAL,                    'Probability of being treated'),
        'prob_contr': Parameter(Types.REAL,                    'Probability achieving normal glucose levels on medication'),
        'level_of_symptoms': Parameter(Types.CATEGORICAL,      'Level of symptoms that the individual will have'),
        'dalywt_mild_retino': Parameter(Types.REAL,            'DALY weighting for mild retinopathy'),
        'dalywt_severe_retino': Parameter(Types.REAL,          'DALY weighting for severe retinopathy'),
        'dalywt_uncomplicated': Parameter(Types.REAL,          'DALY weighting for uncomplicated diabetic'),
        'dalywt_neuropathy': Parameter(Types.REAL,             'DALY weighting for neuropathy'),

    }

    PROPERTIES = {
        # ToDo: Update symptoms name

        # 1. Disease properties
        'd2_risk': Property(Types.REAL,                       'Risk of T2DM given pre-existing condition and risk'),
        'd2_date': Property(Types.DATE,                       'Date of latest T2DM'),
        'd2_current_status': Property(Types.BOOL,             'Current T2DM status'),
        'd2_historic_status': Property(Types.CATEGORICAL,     'Historical status: N=never; C=Current, P=Previous',
                                                                 categories=['N', 'C', 'P']),
        'd2_death_date': Property(Types.DATE,                 'Date of scheduled death of infected individual'),

        # 2. Health care properties
        'd2_diag_date': Property(Types.DATE,                  'Date of latest T2DM diagnosis'),
        'd2_diag_status': Property(Types.CATEGORICAL,         'Status: N=No; Y=Yes',
                                                                 categories=['N', 'Y']),
        'd2_treat_date': Property(Types.DATE,                 'Date of latest T2DM treatment'),
        'd2_treat_status': Property(Types.CATEGORICAL,        'Status: N=never; C=Current, P=Previous',
                                                                 categories=['N', 'C', 'P']),
        'd2_contr_date': Property(Types.DATE,                 'Date of latest T2DM control'),
        'd2_contr_status': Property(Types.CATEGORICAL,        'Status: N=No; Y=Yes',
                                                                 categories=['N', 'Y']),
        'd2_specific_symptoms': Property(Types.CATEGORICAL,   'Level of symptoms for T2DM specifically',
                                                                 categories=['none', 'mild sneezing', 'coughing and irritable', 'extreme emergency']),
        'd2_unified_symptom_code': Property(Types.CATEGORICAL,'Level of symptoms on the standardised scale (governing health-care seeking): '
                                                              '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
                                                                 categories=[0, 1, 2, 3, 4]),
    }

    def read_parameters(self, data_folder):

        p = self.parameters
        df = T2DM_risk.set_index('parameter')
        p['prob_T2DM_basic'] = df.at['prob_basic', 'value']
        p['prob_T2DMgivenBMI'] = pd.DataFrame([[df.at['prob_d2givenbmi', 'value']]],
                                            index=['overweight'],
                                            columns=['risk'])
        p['prob_T2DMgivenHT'] = df.at['prob_d2givenht', 'value']
        #p['prob_T2DMgivenFamHis'] = pd.DataFrame([[df.at['prob_d2givenfamhis', 'value']]],
        #                            index=['family history'],
        #                            columns=['risk'])
        p['prob_diag']          = 0.5
        p['prob_treat']         = 0.5
        p['prob_contr']         = 0.5
        p['level_of_symptoms']  = pd.DataFrame(data={'level_of_symptoms':
                                      ['none',
                                      'mild',
                                      'moderate',
                                      'extreme emergency'],
                                      'probability': [0.25, 0.25, 0.25, 0.25]
                                    })
        df = T2DM_data.set_index('index')
        p['initial_prevalence'] = pd.DataFrame([[df.at['b_all', 'value']], [df.at['b_25_35', 'value']], [df.at['b_35_45', 'value']], [df.at['b_45_55', 'value']], [df.at['b_55_65', 'value']]],
                                                index = ['both sexes', '25 to 35', '35 to 45', '45 to 55', '55 to 65'],
                                                columns = ['prevalence'])
        p['initial_prevalence'].loc[:, 'prevalence'] *= 100  # Convert to %

        #Get the DALY weight that diabetes type 2 will us from the weight database
        if 'HealthBurden' in self.sim.modules.keys():
            p['qalywt_mild_retino']   = self.sim.modules['HealthBurden'].get_daly_weight(967)
            p['qalywt_severe_retino'] = self.sim.modules['HealthBurden'].get_daly_weight(977)
            p['qalywt_uncomplicated'] = self.sim.modules['HealthBurden'].get_daly_weight(971)
            p['qalywt_neuropathy']    = self.sim.modules['HealthBurden'].get_daly_weight(970)


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        # 1. Define key variables
        df = population.props                               # Shortcut to the data frame storing individual data

        # 2. Disease and health care properties
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

        # 3. Assign prevalence as per data
        alive_count = df.is_alive.sum()

        # ToDO: Re-add this later after testing T2DM alone
        # 3.1 First get relative risk for T2DM
        t2dm_prob = df.loc[df.is_alive, ['d2_risk', 'age_years']].merge(T2DM_prevalence, left_on=['age_years'], right_on=['age'],
                                                                      how='left')['probability']
        # df.loc[df.is_alive & ~df.d2_current_status, 'd2_risk'] = self.prob_T2DM_basic  # Basic risk, no pre-existing conditions
        # df.loc[df.is_alive & df.d2_current_status, 'd2_risk'] = self.prob_T2DMgivenHC  # Risk if pre-existing high cholesterol
        assert alive_count == len(t2dm_prob)
        t2dm_prob = t2dm_prob * df.loc[df.is_alive, 'd2_risk']
        random_numbers = self.rng.random_sample(size=alive_count)
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'd2_current_status'] = (random_numbers < t2dm_prob)

        # Check random numbers      #TODO: CHECK WITH TIM and remove later
        over_18 = df.loc[df['age_years'] > 17].index
        a = random_numbers[over_18].mean()

        over_25_35 = df.index[df.age_years > 24 & (df.age_years < 35)]
        b = random_numbers[over_25_35].mean()

        over_35_45 = df.index[df.age_years > 34 & (df.age_years < 45)]
        c = random_numbers[over_35_45].mean()

        over_45_55 = df.index[df.age_years > 44 & (df.age_years < 55)]
        d = random_numbers[over_45_55].mean()

        over_55 = df.loc[df['age_years'] > 55].index
        e = random_numbers[over_55].mean()

        print("\n", "Lets generate the random number check for T2DM: ",
              "\n", "A: ", a,
              "\n", "B: ", b,
              "\n", "C: ", c,
              "\n", "D: ", d,
              "\n", "e: ", e, "\n")

        # 3.2. Calculate prevalence
        # Count adults in different age groups
        adults_count_overall = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[(df.is_alive) & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[(df.is_alive) & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[(df.is_alive) & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_overall == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with T2DM by age group
        count_all = len(df[(df.is_alive) & (df.d2_current_status)])
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
        df2 = self.parameters['initial_prevalence']
        print("\n", "Prevalent T2DM has been assigned"
                    "\n", "MODEL: ",
              "\n", "both sexes: ", prevalence_overall, "%",
              "\n", "25 to 35: ", prevalence_25to35, "%",
              "\n", "35 to 45: ", prevalence_35to45, "%",
              "\n", "45 to 55: ", prevalence_45to55, "%",
              "\n", "55 to 65: ", prevalence_55to65, "%",
              "\n", "DATA: ", df2)

        # 3.2. Calculate prevalence
        #count = df.d2_current_status.sum()
        #prevalence = (count / alive_count) * 100

        # 4. Assign level of symptoms
        #level_of_symptoms = self.parameters['level_of_symptoms']
        #symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
        #                           size=T2DM_count,
        #                           p=level_of_symptoms.probability)
        #df.loc[df.d2_current_status, 'd2_specific_symptoms'] = symptoms

        # 5. Set date of death # TODO: correct this to parameter not distribution and bbelow in event
        death_years_ahead = np.random.exponential(scale=20, size=count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # 6. Set relevant properties of those with prevalent T2DM
        t2dm_years_ago = 1
        infected_td_ago = pd.to_timedelta(t2dm_years_ago * 365.25, unit='d')
        df.loc[df.is_alive & df.d2_current_status, 'd2_date'] = self.sim.date - infected_td_ago # TODO: check with Tim if we should make it more 'realistic'. Con: this still allows us  to check prevalent cases against data, no severity diff with t
        df.loc[df.is_alive & df.d2_current_status, 'd2_historic_status'] = 'C'
        #df.loc[df.is_alive & df.d2_current_status, 'd2_death_date'] = self.sim.date + death_td_ahead

        #print("\n", "Population has been initialised, prevalent T2DM cases have been assigned.  "
        #      "\n", "Prevalence of T2DM is: ", prevalence, "%", "\n")


    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # 1. Add  basic event
        event = T2DMEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(years=1)) # ToDo: looks like this is running every month instead of every year

        # 2. Add an event to log to screen
        sim.schedule_event(T2DMLoggingEvent(self), sim.date + DateOffset(years=5))

        # 3. Add shortcut to the data frame
        df = sim.population.props

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

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props
        df.at[child_id, 'd2_risk'] = 1.0                        # Default setting: no risk given pre-existing conditions
        df.at[child_id, 'd2_current_status'] = False            # Default setting: no one has T2DM
        df.at[child_id, 'd2_historic_status'] = 'N'             # Default setting: no one has T2DM
        df.at[child_id, 'd2_date'] = pd.NaT                     # Default setting: no one has T2DM
        df.at[child_id, 'd2_death_date'] = pd.NaT               # Default setting: no one has T2DM
        df.at[child_id, 'd2_diag_date'] = pd.NaT                # Default setting: no one is diagnosed
        df.at[child_id, 'd2_diag_status'] = 'N'                 # Default setting: no one is diagnosed
        df.at[child_id, 'd2_diag_date'] = pd.NaT                # Default setting: no one is treated
        df.at[child_id, 'd2_diag_status'] = 'N'                 # Default setting: no one is treated
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



    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        # logger.debug('This is T2DM reporting my health values')

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


class T2DMEvent(RegularEvent, PopulationScopeEventMixin):

    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of Mockitis.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1)) # TODO: change time scale if needed
        self.prob_T2DM_basic = module.parameters['prob_T2DM_basic']
        #self.prob_T2DMgivenWeight = module.parameters['prob_T2DMgivenWeight']
        #self.prob_T2DMgivenHT = module.parameters['prob_T2DMgivenHT']
        #self.prob_T2DMgivenGD = module.parameters['prob_T2DMgivenGD']
        #self.prob_T2DMgivenFamHis = module.parameters['prob_T2DMgivenFamHis']
        self.prob_treat = module.parameters['prob_treat']

        # ToDO: need to add code from original if it bugs.

    def apply(self, population):

        logger.debug('This is T2DMEvent, tracking the disease progression of the population.')

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
