"""
This is the method for hypertension
Developed by Mikaela Smit, October 2018

"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: change file path mac/window flex

class HT(Module):
    """
    This is hypertension.
    It demonstrates the following behaviours in respect of the healthsystem module:

    - Declaration of TREATMENT_ID
    - Registration of the disease module
    - Reading DALY weights and reporting daly values related to this disease
    - Health care seeking
    - Running an "outreach" event
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        logger.info('----------------------------------------------------------------------')
        logger.info("Running hypertension.  ")
        logger.info('----------------------------------------------------------------------')

    PARAMETERS = {

        # 1. Risk parameters
        'HT_prevalence' : Parameter(Types.REAL, 'HT prevalence'),
        'HT_incidence' : Parameter(Types.REAL, 'HT incidence'),
        'prob_HT_basic': Parameter(Types.REAL, 'Basic HTN probability'),
        'prob_HTgivenBMI': Parameter(Types.CATEGORICAL, 'HTN probability given BMI'),
        'prob_HTgivenDiab': Parameter(Types.REAL, 'HTN probability given pre-existing diabetes'),

        # 2. Health care parameters                          #ToDO: remove part of this section when HSI activated
        'prob_diag': Parameter(Types.REAL, 'Probability of being diagnosed'),
        'prob_treat': Parameter(Types.REAL, 'Probability of being treated'),
        'prob_contr': Parameter(Types.REAL, 'Probability achieving control on medication'),
        'dalywt_ht': Parameter(Types.REAL, 'DALY weighting for hypertension'),
        'initial_prevalence' : Parameter(Types.REAL, 'Prevalence of hypertension as per data')
    }

    PROPERTIES = {
        # 1. Disease properties
        'ht_risk': Property(Types.REAL, 'HTN risk given pre-existing condition or risk factors'),
        'ht_date': Property(Types.DATE, 'Date of latest hypertensive episode'),
        'ht_current_status': Property(Types.BOOL, 'Current hypertension status'),
        'ht_historic_status': Property(Types.CATEGORICAL, 'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),

        # 2. Health care properties
        'ht_diag_date': Property(Types.DATE, 'Date of latest hypertension diagnosis'),
        'ht_diag_status': Property(Types.CATEGORICAL, 'Status: N=No; Y=Yes',
                                   categories=['N', 'C', 'P']),
        'ht_treat_date': Property(Types.DATE, 'Date of latest hypertension treatment'),
        'ht_treat_status': Property(Types.CATEGORICAL, 'Status: N=never; C=Current, P=Previous',
                                    categories=['N', 'C', 'P']),
        'ht_contr_date': Property(Types.DATE, 'Date of latest hypertension control'),
        'ht_contr_status': Property(Types.CATEGORICAL, 'Status: N=No; Y=Yes',
                                    categories=['N', 'C', 'P']),
    }

    def read_parameters(self, data_folder):
        logger.debug('Hypertension method: reading in parameters.  ')

        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Method_HT.xlsx',
                                 sheet_name=None)

        p = self.parameters
        p['HT_prevalence'] = workbook['prevalence2018']
        p['HT_incidence'] = workbook['incidence2018_plus']

        HT_risk = workbook['parameters']
        df = HT_risk.set_index('parameter')
        p['prob_HT_basic'] = df.at['prob_basic', 'value']
        p['prob_HTgivenBMI'] = pd.DataFrame([[df.at['prob_htgivenbmi', 'value']], [df.at['prob_htgivenbmi', 'value2']],
                                             [df.at['prob_htgivenbmi', 'value3']]],
                                            index=['overweight', 'obese', 'morbidly obese'],
                                            columns=['risk'])
        p['prob_HTgivenDiab'] = df.at['prob_htgivendiabetes', 'value']
        p['prob_diag'] = 0.5  # ToDO: remove once HSi activated
        p['prob_treat'] = 0.5
        p['prob_contr'] = 0.5
        p['dalywt_ht'] = 0.0

        HT_data = workbook['data']
        df = HT_data.set_index('index')
        p['initial_prevalence'] = pd.DataFrame([[df.at['b_all', 'value']], [df.at['b_25_35', 'value']],
                                                [df.at['b_35_45', 'value']], [df.at['b_45_55', 'value']],
                                                [df.at['b_55_65', 'value']]],
                                               index=['total', '25_to_35', '35_to_45', '45_to_55', '55_to_65'],
                                               columns=['prevalence'])  #ToDo: Add 95% CI
        p['initial_prevalence'].loc[:, 'prevalence'] *= 100  # Convert data to percentage

        logger.debug('%s|prevalence_data|%s',
                     self.sim.date,
                     {
                         'total': p['initial_prevalence'].loc['total', 'prevalence'],
                         '25_to_35':  p['initial_prevalence'].loc['25_to_35', 'prevalence'],
                         '35_to_45': p['initial_prevalence'].loc['35_to_45', 'prevalence'],
                         '45_to_55': p['initial_prevalence'].loc['45_to_55', 'prevalence'],
                         '55_to_65': p['initial_prevalence'].loc['55_to_65', 'prevalence']
                     })

        logger.debug("Hypertension method: finished reading in parameters.  ")

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        logger.debug('Hypertension method: initialising population.  ')

        # 1. Define key variables
        df = population.props  # Shortcut to the data frame storing individual data

        # 2. Define default properties
        df.loc[df.is_alive, 'ht_risk'] = 1.0  # Default: no risk given pre-existing conditions
        df.loc[df.is_alive, 'ht_current_status'] = False  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_historic_status'] = 'N'  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_date'] = pd.NaT  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_diag_date'] = pd.NaT  # Default: no one is diagnosed
        df.loc[df.is_alive, 'ht_diag_status'] = 'N'  # Default: no one is diagnosed
        df.loc[df.is_alive, 'ht_treat_date'] = pd.NaT  # Default: no one is treated
        df.loc[df.is_alive, 'ht_treat_status'] = 'N'  # Default: no one is treated
        df.loc[df.is_alive, 'ht_contr_date'] = pd.NaT  # Default: no one is controlled
        df.loc[df.is_alive, 'ht_contr_status'] = 'N'  # Default: no one is controlled

        # 3. Assign prevalence as per data
        # 3.1 Get corresponding risk
        ht_prob = df.loc[df.is_alive, ['ht_risk', 'age_years']].merge(self.parameters['HT_prevalence'], left_on=['age_years'],
                                                                      right_on=['age'], how='left')['probability']
        df.loc[df.is_alive & df.li_overwt, 'ht_risk'] = self.prob_HTgivenBMI.loc['overweight']['risk']
        # df.loc[df.is_alive & df.d2_current_status, 'ht_risk'] = self.prob_HTgivenDiab # TODO: reactivate once circular issue is sorted out
        alive_count = df.is_alive.sum()

        assert alive_count == len(ht_prob)

        ht_prob = ht_prob * df.loc[df.is_alive, 'ht_risk']
        random_numbers = self.rng.random_sample(size=alive_count)  # TODO: remove duplication later, 1st is not random
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'ht_current_status'] = (random_numbers < ht_prob)

        # Check random numbers      #TODO: CHECK WITH TIM and remove later
        logger.debug('Lets generate the random number check for hypertension')

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

        logger.debug('Lets generate the random number check for hypertension. ')
        logger.debug({
                        'A': a,
                        'B': b,
                        'C': c,
                        'D': d,
                        'E': e
                        })

        # 3.2. Calculate prevalence
        adults_count_all = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[df.is_alive & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[df.is_alive & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[df.is_alive & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_all == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with hypertension by age group
        count_all = len(df[df.is_alive & df.ht_current_status])
        count = len(df[df.is_alive & df.ht_current_status & (df.age_years > 24) & (df.age_years < 65)])
        count_25to35 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 24) & (df.age_years < 35)])
        count_35to45 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 34) & (df.age_years < 45)])
        count_45to55 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 44) & (df.age_years < 55)])
        count_55to65 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 54) & (df.age_years < 65)])

        assert count == count_25to35 + count_35to45 + count_45to55 + count_55to65

        # Calculate overall and age-specific prevalence
        prevalence_overall = (count / adults_count_all) * 100
        prevalence_25to35 = (count_25to35 / adults_count_25to35) * 100
        prevalence_35to45 = (count_35to45 / adults_count_35to45) * 100
        prevalence_45to55 = (count_45to55 / adults_count_45to55) * 100
        prevalence_55to65 = (count_55to65 / adults_count_55to65) * 100

        # 3.1 Log prevalence compared to data
        df2 = self.parameters['initial_prevalence']
        logger.info('Prevalent hypertension has been assigned.  ')
        print("\n", df2, "\n")      #TODO: Add this to logger when debug code
        logger.info('%s|prevalence|%s',
                    self.sim.date,
                    {
                        'total': prevalence_overall,
                        '25_to_35': prevalence_25to35,
                        '35_to_45': prevalence_35to45,
                        '45_to_55': prevalence_45to55,
                        '55_to_65': prevalence_55to65
                    })

        # 4. Set relevant properties of those with prevalent hypertension
        ht_years_ago = np.array([1] * count_all)
        infected_td_ago = pd.to_timedelta(ht_years_ago, unit='y')
        df.loc[df.is_alive & df.ht_current_status, 'ht_date'] = self.sim.date - infected_td_ago
        df.loc[df.is_alive & df.ht_current_status, 'ht_historic_status'] = 'C'

        # Register this disease module with the health system       #TODO: CHECK WITH TIM
        # self.sim.modules['HealthSystem'].register_disease_module(self)

        logger.debug("Hypertension method: finished initialising population.  ")

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        logger.debug("Hypertension method: initialising simulation.  ")

        # 1. Add  basic event
        event = HTEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(years=1))

        # 2. Launch the repeating event that will store statistics about hypertension
        sim.schedule_event(HTLoggingEvent(self), sim.date + DateOffset(days=0))

        # 3. Schedule the outreach event... # ToDo: need to test this with HT!
        # outreach_event = HT_LaunchOutreachEvent(self)
        # self.sim.schedule_event(outreach_event, self.sim.date+36)

        logger.debug("Hypertension method: finished initialising simulation.  ")

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        logger.debug("Hypertension method: on birth is being defined.  ")

        df = self.sim.population.props
        df.at[child_id, 'ht_risk'] = 1.0  # Default: no risk given pre-existing conditions
        df.at[child_id, 'ht_current_status'] = False  # Default: no one has hypertension
        df.at[child_id, 'ht_historic_status'] = 'N'  # Default: no one has hypertension
        df.at[child_id, 'ht_date'] = pd.NaT  # Default: no one has hypertension
        df.at[child_id, 'ht_diag_date'] = pd.NaT  # Default: no one is diagnosed
        df.at[child_id, 'ht_diag_status'] = 'N'  # Default: no one is diagnosed
        df.at[child_id, 'ht_diag_date'] = pd.NaT  # Default: no one is treated
        df.at[child_id, 'ht_diag_status'] = 'N'  # Default: no one is treated
        df.at[child_id, 'ht_contr_date'] = pd.NaT  # Default: no one is controlled
        df.at[child_id, 'ht_contr_status'] = 'N'  # Default: no one is controlled

        logger.debug("Hypertension method: finished defining on birth.  ")

    def on_healthsystem_interaction(self, person_id, treatment_id): #TODO: update
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Hypertension, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def on_hsi_alert(self, person_id, treatment_id):    #TODO: update
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):   #TODO: update
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('Hypertension method: reporting health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'ht_specific_symptoms'].map({
            'N': 0,
        })
        health_values.name = 'HT Symptoms'  # label the cause of this disability

        return health_values.loc[df.is_alive]


class HTEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at annual intervals and controls the disease process of Hypertension.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1))
        self.HT_incidence = module.parameters['HT_incidence']
        self.prob_HT_basic = module.parameters['prob_HT_basic']
        self.prob_HTgivenBMI = module.parameters['prob_HTgivenBMI']
        self.prob_HTgivenDiab = module.parameters['prob_HTgivenDiab']

        self.prob_diag = module.parameters['prob_diag']
        self.prob_treat = module.parameters['prob_treat']
        self.prob_contr = module.parameters['prob_contr']
        self.dalywt_ht = module.parameters['dalywt_ht']

        logger.debug('Hypertension method: This is a HTN Event')

        # ToDO: need to add code from original if it bugs.

    def apply(self, population):

        logger.debug('Hypertension method: tracking the disease progression of the population.')

        # 1. Basic variables
        df = population.props
        rng = self.module.rng

        ht_total = (df.is_alive & df.ht_current_status).sum()

        # 2. Get (and hold) index of people with and w/o hypertension
        currently_ht_yes = df[df.ht_current_status & df.is_alive].index
        currently_ht_no = df[~df.ht_current_status & df.is_alive].index
        age_index = df.loc[currently_ht_no, ['age_years']]
        alive_count = df.is_alive.sum()

        assert alive_count == len(currently_ht_yes) + len(currently_ht_no)

        # 3. Handle new cases of hypertension
        # 3.1 First get relative risk
        ht_prob = df.loc[currently_ht_no, ['age_years', 'ht_risk']].reset_index().merge(self.HT_incidence,
                                                                                        left_on=['age_years'],
                                                                                        right_on=['age'],
                                                                                        how='left').set_index(
                                                                                        'person')['probability']
        df.loc[df.is_alive & df.li_overwt, 'ht_risk'] = self.prob_HTgivenBMI.loc['overweight']['risk']
        # df.loc[df.is_alive & df.d2_current_status, 'ht_risk'] = self.prob_HTgivenDiab  # TODO: update once diabetes is active and test it's linking
        ht_prob = ht_prob * df.loc[currently_ht_no, 'ht_risk']
        random_numbers = rng.random_sample(size=len(ht_prob))
        now_hypertensive = (ht_prob > random_numbers)
        ht_idx = currently_ht_no[now_hypertensive]

        # Check random numbers      #TODO: CHECK WITH TIM and remove later
        # random_numbers_df = random_numbers, index = currently_ht_no
        aaa = pd.DataFrame(data=random_numbers, index=currently_ht_no, columns=['nr'])

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

        logger.debug('Lets generate the random number check for incident hypertension. ')
        logger.debug({
            'A': a,
            'B': b,
            'C': c,
            'D': d,
            'E': e
        })


        # 3.3 If newly hypertensive
        df.loc[ht_idx, 'ht_current_status'] = True
        df.loc[ht_idx, 'ht_historic_status'] = 'C'
        df.loc[ht_idx, 'ht_date'] = self.sim.date

        # TODO: check it we want to log evert new event also?

        # 3.2. Calculate prevalence
        # Count adults in different age groups
        adults_count_overall = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[(df.is_alive) & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[(df.is_alive) & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[(df.is_alive) & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[(df.is_alive) & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_overall == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with hypertension by age group
        count = len(df[(df.is_alive) & (df.ht_current_status) & (df.age_years > 24) & (df.age_years < 65)])
        count_25to35 = len(df[(df.is_alive) & (df.ht_current_status) & (df.age_years > 24) & (df.age_years < 35)])
        count_35to45 = len(df[(df.is_alive) & (df.ht_current_status) & (df.age_years > 34) & (df.age_years < 45)])
        count_45to55 = len(df[(df.is_alive) & (df.ht_current_status) & (df.age_years > 44) & (df.age_years < 55)])
        count_55to65 = len(df[(df.is_alive) & (df.ht_current_status) & (df.age_years > 54) & (df.age_years < 65)])

        assert count == count_25to35 + count_35to45 + count_45to55 + count_55to65

        # Calculate overall and age-specific prevalence
        prevalence_overall = (count / adults_count_overall) * 100
        prevalence_25to35 = (count_25to35 / adults_count_25to35) * 100
        prevalence_35to45 = (count_35to45 / adults_count_35to45) * 100
        prevalence_45to55 = (count_45to55 / adults_count_45to55) * 100
        prevalence_55to65 = (count_55to65 / adults_count_55to65) * 100

        # 3.1 Log prevalence compared to data
        df2 = self.module.parameters['initial_prevalence']
        logger.info('Prevalent hypertension has been assigned.  ')
        print("\n", df2, "\n")  # TODO: Add this to logger when debug code
        logger.info('%s|prevalence|%s',
                    self.sim.date,
                    {
                        'total': prevalence_overall,
                        '25_to_35': prevalence_25to35,
                        '35_to_45': prevalence_35to45,
                        '45_to_55': prevalence_45to55,
                        '55_to_65': prevalence_55to65
                    })

class HT_LaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    This is the event that is run by Hypertension and it is the Outreach Event.
    It will now submit the individual HSI events that occur when each individual is met.
    (i.e. Any large campaign is composed of many individual outreach events).
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        gets_outreach = df.index[(df['is_alive']) & (df['sex'] == 'F')]
        # self.sim.rng.choice(gets_outreach,2)

        for person_id in gets_outreach:
            # make the outreach event (let this disease module be alerted about it, and also Mockitis)
            outreach_event_for_individual = HSI_HT_Outreach_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(outreach_event_for_individual,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_HT_Outreach_Individual(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    This event can be used to simulate the occurrence of an 'outreach' intervention.

    NB. This needs to be created and run for each individual that benefits from the outreach campaign.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        logger.debug('Outreach event being created.')

        # Define the necessary information for an HSI
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HT_Outreach_Individual'
        self.APPT_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        self.APPT_FOOTPRINT['ConWithDCSA'] = 0.5  # outreach event takes small amount of time for DCSA
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # Can occur at any facility level
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id):
        logger.debug('Outreach event running now for person: %s', person_id)

        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        if df.at[person_id, 'ht_current_status']:
            if df.at[person_id, 'is_alive']:
                referral_event_for_individual = HSI_HT_Refer_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(referral_event_for_individual,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        pass


class HSI_HT_Refer_Individual(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Hypertension.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'HSI_HT_Refer_Individual'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_HT_PresentsForCareWithSevereSymptoms, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_HT_PresentsForCareWithSevereSymptoms: there should now be treatment for person %d',
                person_id)
            event = HSI_HT_Refer_Individual(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

        else:
            logger.debug(
                '...This is HSI_HT_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_HT_Refer_Individual(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=date_turns_15,
                                                            tclose=date_turns_15 + DateOffset(months=12))


# #class HSI_HT_StartTreatment(Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event.
#
#     It is appointment at which treatment for hypertension is inititaed.
#
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#         the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
#         the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation
#
#
#         # Get the consumables required
#         consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
#         pkg_code1 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
#             0]
#         pkg_code2 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
#             0]
#
#         item_code1 = \
#         pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
#         item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]
#
#         the_cons_footprint = {
#             'Intervention_Package_Code': [pkg_code1, pkg_code2],
#             'Item_Code': [item_code1, item_code2]
#         }
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = 'HT_Treatment_Initiation'
#         self.APPT_FOOTPRINT = the_appt_footprint
#         self.CONS_FOOTPRINT = the_cons_footprint
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id):
#         logger.debug('This is HSI_HT_StartTreatment: initiating treatent for person %d', person_id)
#         df = self.sim.population.props
#         treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']
#
#         if treatmentworks:
#             df.at[person_id, 'mi_is_infected'] = False
#             df.at[person_id, 'mi_status'] = 'P'
#
#             # (in this we nullify the death event that has been scheduled.)
#             df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT
#
#             df.at[person_id, 'mi_date_cure'] = self.sim.date
#             df.at[person_id, 'mi_specific_symptoms'] = 'none'
#             df.at[person_id, 'mi_unified_symptom_code'] = 0
#             pass
#
#         # Create a follow-up appointment
#         target_date_for_followup_appt = self.sim.date + DateOffset(months=6)
#
#         logger.debug('....This is HSI_HT_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
#                      person_id, target_date_for_followup_appt)
#
#         followup_appt = HSI_HT_TreatmentMonitoring(self.module, person_id=person_id)
#
#         # Request the heathsystem to have this follow-up appointment
#         self.sim.modules['HealthSystem'].schedule_event(followup_appt,
#                                                         priority=2,
#                                                         topen=target_date_for_followup_appt,
#                                                         tclose=target_date_for_followup_appt + DateOffset(weeks=2)
#         )
#
#
#
#
# class HSI_HT_TreatmentMonitoring(Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event.
#
#     It is appointment at which treatment for hypertension is monitored.
#     (In practise, nothing happens!)
#
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#         the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
#         the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation
#
#
#         # Get the consumables required
#         consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
#         pkg_code1 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
#             0]
#         pkg_code2 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
#             0]
#
#         item_code1 = \
#         pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
#         item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]
#
#         the_cons_footprint = {
#             'Intervention_Package_Code': [pkg_code1, pkg_code2],
#             'Item_Code': [item_code1, item_code2]
#         }
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = 'HT_TreatmentMonitoring'
#         self.APPT_FOOTPRINT = the_appt_footprint
#         self.CONS_FOOTPRINT = the_cons_footprint
#         self.ALERT_OTHER_DISEASES = ['*']
#
#     def apply(self, person_id):
#
#         # There is a follow-up appoint happening now but it has no real effect!
#
#         # Create the next follow-up appointment....
#         target_date_for_followup_appt = self.sim.date + DateOffset(months=6)
#
#         logger.debug(
#             '....This is HSI_HT_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
#             person_id, target_date_for_followup_appt)
#
#         followup_appt = HSI_HT_TreatmentMonitoring(self.module, person_id=person_id)
#
#         # Request the heathsystem to have this follow-up appointment
#         self.sim.modules['HealthSystem'].schedule_event(followup_appt,
#                                                         priority=2,
#                                                         topen=target_date_for_followup_appt,
#                                                         tclose=target_date_for_followup_appt + DateOffset(weeks=2))
#
#

# ---------------------------------------------------------------------------------


class HTLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'hypertension status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        # TODO: update to always log the prevalence, every month?

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

        # logger.info('%s|status_counts|%s', self.sim.date, counts)
