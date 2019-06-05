"""
This is the method for hypertension
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
file_path = '/Users/mc1405/Dropbox/Projects - ongoing/Malawi Project/Thanzi la Onse/04 - Methods Repository/Method_HT.xlsx'
method_ht_data = pd.read_excel(file_path, sheet_name=None, header=0)
HT_prevalence, HT_incidence, HT_treatment, HT_risk = method_ht_data['prevalence2018'], method_ht_data['incidence2018_plus'], \
                                            method_ht_data['treatment_parameters'], method_ht_data['parameters']

class HT(Module):
    """
    This is hypertension.
    It demonstrates the following behaviours in respect of the healthsystem module:

    - Declaration of TREATMENT_ID
    - Registration of the disease module
    - Reading QALY weights and reporting qaly values related to this disease
    - Health care seeking
    - Running an "outreach" event
    """

    PARAMETERS = {
        'prob_HT_basic': Parameter(Types.REAL,
                                    'Probability of getting hypertension given no pre-existing condition'),
        #'prob_HTgivenHC': Parameter(Types.REAL, 'Probability of getting hypertension given pre-existing high cholesterol'),
        #'prob_HTgivenDiab': Parameter(Types.REAL,
        #                            'Probability of getting hypertension given pre-existing diabetes'),
        #'prob_HTgivenHIV': Parameter(Types.REAL,
        #                            'Probability of getting hypertension given pre-existing HIV'),
        'prob_success_treat': Parameter(Types.REAL,
                                    'Probability of intervention for hypertension reduced blood pressure to normal levels'),

        # TODO: Merge/Delete underneath after testing above
        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL, 'Level of symptoms that the individual will have'),
        'p_cure': Parameter(
            Types.REAL, 'Probability that a treatment is successful in curing the individual'),
        'initial_prevalence': Parameter(
            Types.REAL, 'Prevalence of the disease in the initial population'),
        'qalywt_mild_sneezing': Parameter(
            Types.REAL, 'QALY weighting for mild sneezing'),
        'qalywt_coughing': Parameter(
            Types.REAL, 'QALY weighting for coughing'),
        'qalywt_advanced': Parameter(
            Types.REAL, 'QALY weighting for extreme emergency')
    }

    PROPERTIES = {
        'ht_risk': Property(Types.REAL, 'Risk of hypertension given pre-existing condition'),
        'ht_current_status': Property(Types.BOOL, 'Current hypertension status'),
        'ht_historic_status': Property(Types.CATEGORICAL,
                                       'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),
        'ht_case_date': Property(Types.DATE, 'Date of latest hypertension'),

        'ht_diag_status': Property(Types.CATEGORICAL,
                                        'Status: N=No; Y=Yes',
                                        categories=['N', 'C', 'P']),
        'ht_diag_date': Property(Types.DATE, 'Date of latest hypertension diagnosis'),

        'ht_treat_status': Property(Types.CATEGORICAL,
                                        'Status: N=never; C=Current, P=Previous',
                                        categories=['N', 'C', 'P']),
        'ht_treat_date': Property(Types.DATE, 'Date of latest hypertension treatment'),

        'ht_contr_status': Property(Types.CATEGORICAL,
                                        'Status: N=No; Y=Yes',
                                        categories=['N', 'C', 'P']),
        'ht_contr_date': Property(Types.DATE, 'Date of latest hypertension control'),

        # TODO: Merge/Delete underneath after testing above
        'mi_is_infected': Property(
            Types.BOOL, 'Current status of mockitis'),
        'mi_status': Property(
            Types.CATEGORICAL, 'Historical status: N=never; C=currently; P=previously',
            categories=['N', 'C', 'P']),
        'mi_date_infected': Property(
            Types.DATE, 'Date of latest infection'),
        'mi_scheduled_date_death': Property(
            Types.DATE, 'Date of scheduled death of infected individual'),
        'mi_date_cure': Property(
            Types.DATE, 'Date an infected individual was cured'),
        'mi_specific_symptoms': Property(
            Types.CATEGORICAL, 'Level of symptoms for mockitiis specifically',
            categories=['none', 'mild sneezing', 'coughing and irritable', 'extreme emergency']),
        'mi_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4])
    }

    def read_parameters(self, data_folder):
        p = self.parameters

        p = self.parameters
        p['prob_HT_basic'] = 1.0
        p['prob_HTgivenHC'] = 2.0
        # p['prob_HTgivenDiab'] = 1.4
        # p['prob_HTgivenHIV'] = 1.49
        p['prob_success_treat'] = 0.5

        # TODO: Merge/Delete underneath after testing above
        p['p_infection'] = 0.001
        p['p_cure'] = 0.50
        p['initial_prevalence'] = 0.5
        p['level_of_symptoms'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none',
                                      'mild sneezing',
                                      'coughing and irritable',
                                      'extreme emergency'],
                'probability': [0.25, 0.25, 0.25, 0.25]
            })

        # get the QALY values that this module will use from the weight database
        # (these codes are just random!)
        p['qalywt_mild_sneezing'] = self.sim.modules['QALY'].get_qaly_weight(50)
        p['qalywt_coughing'] = self.sim.modules['QALY'].get_qaly_weight(50)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        # 1. Define key variables
        df = population.props  # a shortcut to the dataframe storing data for individiuals

        # 2. Set default for properties
        df.loc[df.is_alive,'ht_risk'] = 1.0                 # Default setting: no risk given pre-existing conditions
        df.loc[df.is_alive,'ht_current_status'] = False     # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_historic_status'] = 'N'      # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_case_date'] = pd.NaT         # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_diag_date'] = pd.NaT         # Default setting: no one is diagnosed
        df.loc[df.is_alive,'ht_diag_status'] = 'N'          # Default setting: no one is diagnosed
        df.loc[df.is_alive,'ht_treat_date'] = pd.NaT        # Default setting: no one is treated
        df.loc[df.is_alive,'ht_treat_status'] = 'N'         # Default setting: no one is treated
        df.loc[df.is_alive,'ht_case_date'] = pd.NaT         # Default setting: no one is controlled
        df.loc[df.is_alive,'ht_case_status'] = 'N'          # Default setting: no one is controlled

        # TODO: Merge/Delete underneath after testing above
        df.loc[df.is_alive, 'mi_is_infected'] = False  # default: no individuals infected
        df.loc[df.is_alive, 'mi_status'] = 'N'  # default: never infected
        df.loc[df.is_alive, 'mi_date_infected'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_scheduled_date_death'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_date_cure'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_specific_symptoms'] = 'none'
        df.loc[df.is_alive, 'mi_unified_symptom_code'] = 0

        # 3. Assign prevalence as per data
        alive_count = df.is_alive.sum()
        ht_prob = df.loc[df.is_alive, ['ht_risk', 'age_years']].merge(HT_prevalence,
                                                                      left_on=['age_years'],
                                                                      right_on=['age'],
                                                                      how='left')['probability']

        assert alive_count == len(ht_prob) #ToDO: what does this line do?

        # 3.1 Depending on pre-existing conditions, get associated risk and update prevalence and assign hypertension
        #df.loc[df.is_alive & ~df.hc_current_status, 'ht_risk'] = self.prob_HT_basic  # Basic risk, no pre-existing conditions
        #df.loc[df.is_alive & df.hc_current_status, 'ht_risk'] = self.prob_HTgivenHC  # Risk if pre-existing high cholesterol

        # 3.2. Finish assigning prevalene
        ht_prob = ht_prob * df.loc[df.is_alive, 'ht_risk']
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'ht_current_status'] = (random_numbers < ht_prob)  # Assign prevalence at t0

        # 4. Count all individuals by status at the start
        hypertension_count = (df.is_alive & df.ht_current_status).sum()

        # 5. Set date of hypertension amongst those with prevalent cases
        ht_years_ago = self.rng.exponential(scale=5, size=hypertension_count)
        infected_td_ago = pd.to_timedelta(ht_years_ago * 365.25, unit='d') #TODO: set current date

        # 5.1 Set the properties of those with prevalent hypertension
        df.loc[df.is_alive & df.ht_current_status, 'ht_date_case'] = self.sim.date - infected_td_ago
        df.loc[df.is_alive & df.ht_current_status, 'ht_historic_status'] = 'C'

        print("\n", "Population has been initialised, prevalent cases have been assigned.  ")

        # ToDo: Merge/Delete below code

        # randomly selected some individuals as infected
        initial_infected = self.parameters['initial_prevalence']
        df.loc[df.is_alive, 'mi_is_infected'] = self.rng.random_sample(size=alive_count) < initial_infected
        df.loc[df.mi_is_infected, 'mi_status'] = 'C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        infected_count = df.mi_is_infected.sum()

        # Assign level of symptoms
        level_of_symptoms = self.parameters['level_of_symptoms']
        symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                   size=infected_count,
                                   p=level_of_symptoms.probability)
        df.loc[df.mi_is_infected, 'mi_specific_symptoms'] = symptoms

        # date of infection of infected individuals
        # sample years in the past
        infected_years_ago = self.rng.exponential(scale=5, size=infected_count)
        # pandas requires 'timedelta' type for date calculations
        # TODO: timedelta calculations should always be in days
        infected_td_ago = pd.to_timedelta(infected_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=20, size=infected_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.mi_is_infected, 'mi_date_infected'] = self.sim.date - infected_td_ago
        df.loc[df.mi_is_infected, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = HTEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1)) # ToDo: need to update this to adjust to time used for this method

        # add an event to log to screen
        sim.schedule_event(HTLoggingEvent(self), sim.date + DateOffset(months=6))

        # ToDo: understand the below, remove if not necessary
        # a shortcut to the dataframe storing data for individiuals
        df = sim.population.props

        # add the death event of infected individuals
        # schedule the mockitis death event
        people_who_will_die = df.index[df.mi_is_infected]
        for person_id in people_who_will_die:
            self.sim.schedule_event(MockitisDeathEvent(self, person_id),
                                    df.at[person_id, 'mi_scheduled_date_death'])

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Schedule the outreach event... # ToDo: need to test this with HT!
        # event = MockitisOutreachEvent(self, 'this_module_only')
        # self.sim.schedule_event(event, self.sim.date + DateOffset(months=24))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props
        df.at[child_id, 'ht_risk'] = 1.0                # Default setting: no risk given pre-existing conditions
        df.at[child_id, 'ht_current_status'] = False    # Default setting: no one has hypertension
        df.at[child_id, 'ht_historic_status'] = 'N'     # Default setting: no one has hypertension
        df.at[child_id, 'ht_case_date'] = pd.NaT        # Default setting: no one has hypertension
        df.at[child_id, 'ht_diag_date'] = pd.NaT        # Default setting: no one is diagnosed
        df.at[child_id, 'ht_diag_status'] = 'N'         # Default setting: no one is diagnosed
        df.at[child_id, 'ht_diag_date'] = pd.NaT        # Default setting: no one is treated
        df.at[child_id, 'ht_diag_status'] = 'N'         # Default setting: no one is treated
        df.at[child_id, 'ht_contr_date'] = pd.NaT       # Default setting: no one is controlled
        df.at[child_id, 'ht_contr_status'] = 'N'        # Default setting: no one is controlled

        # TODO: REMOVE THIS!
        df.at[child, 'hc_current_status'] = False

        # ToDo: Merge/Delete this later
        df = self.sim.population.props  # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:

        child_is_infected = df.at[mother_id, 'mi_is_infected']  # is infected if mother is infected

        if child_is_infected:

            # Scheduled death date
            death_years_ahead = np.random.exponential(scale=20)
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

            # Level of symptoms
            symptoms = self.rng.choice(self.parameters['level_of_symptoms']['level_of_symptoms'],
                                       p=self.parameters['level_of_symptoms']['probability'])

            # Assign properties
            df.at[child_id, 'mi_is_infected'] = True
            df.at[child_id, 'mi_status'] = 'C'
            df.at[child_id, 'mi_date_infected'] = self.sim.date
            df.at[child_id, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.at[child_id, 'mi_date_cure'] = pd.NaT
            df.at[child_id, 'mi_specific_symptoms'] = symptoms
            df.at[child_id, 'mi_unified_symptom_code'] = 0

            # Schedule death event:
            death_event = MockitisDeathEvent(self, child_id)
            self.sim.schedule_event(death_event, df.at[child_id, 'mi_scheduled_date_death'])

        else:
            # Assign the default for a child who is not infected
            df.at[child_id, 'mi_is_infected'] = False
            df.at[child_id, 'mi_status'] = 'N'
            df.at[child_id, 'mi_date_infected'] = pd.NaT
            df.at[child_id, 'mi_scheduled_date_death'] = pd.NaT
            df.at[child_id, 'mi_date_cure'] = pd.NaT
            df.at[child_id, 'mi_specific_symptoms'] = 'none'
            df.at[child_id, 'mi_unified_symptom_code'] = 0

    def on_healthsystem_interaction(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Mockitis, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)



    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        # logger.debug('This is mockitis reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'mi_specific_symptoms'].map({
            'none': 0,
            'mild sneezing': p['qalywt_mild_sneezing'],
            'coughing and irritable': p['qalywt_coughing'],
            'extreme emergency': p['qalywt_advanced']
        })
        return health_values.loc[df.is_alive]


class HTEvent(RegularEvent, PopulationScopeEventMixin):

    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of Mockitis.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1)) # TODO: change time scale if needed

        # ToDO: need to add code from original if it bugs.

    def apply(self, population):

        logger.debug('This is MockitisEvent, tracking the disease progression of the population.')

        # 1. Basic variables
        df = population.props
        rng = self.module.rng

        ht_total = (df.is_alive & df.ht_current_status).sum()

        # 2. Get (and hold) index of people with and w/o hypertension
        currently_ht_yes = df[df.ht_current_status & df.is_alive].index
        currently_ht_no = df[~df.ht_current_status & df.is_alive].index

        # 3. Handle new cases of hypertension
        ht_prob = df.loc[currently_ht_no, ['age_years', 'ht_risk']].reset_index().merge(HT_incidence,
                                                                                        left_on=['age_years'],
                                                                                        right_on=['age'],
                                                                                        how='inner').set_index(
            'person')['probability']

        assert len(currently_ht_no) == len(ht_prob)

        # 3.1 Depending on pre-existing conditions, get associated risk and update prevalence and assign hypertension
        #df.loc[df.is_alive & ~df.hc_current_status, 'ht_risk'] = self.prob_HT_basic  # Basic risk, no pre-existing conditions
        #df.loc[df.is_alive & df.hc_current_status, 'ht_risk'] = self.prob_HTgivenHC  # Risk if pre-existing high cholesterol

        ht_prob = ht_prob * df.loc[currently_ht_no, 'ht_risk']
        random_numbers = rng.random_sample(size=len(ht_prob))
        now_hypertensive = (ht_prob > random_numbers)  # Assign incidence

        # 3.2 Ways to check what's happening
        # temp = pd.merge(population.age, df, left_index=True, right_index=True, how='inner')
        # temp_2 = pd.DataFrame([population.age.years, joined.probability, random_numbers, df['ht_current_status']])

        # 3.3 If newly hypertensive
        ht_idx = currently_ht_no[now_hypertensive]

        df.loc[ht_idx, 'ht_current_status'] = True
        df.loc[ht_idx, 'ht_historic_status'] = 'C'
        df.loc[ht_idx, 'ht_date_case'] = self.sim.date

        print("\n", "Time is: ", self.sim.date, "New cases have been assigned.  ")





        # TODO: remove/merge below after test

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected = df.index[df.mi_is_infected & df.is_alive]
        currently_uninfected = df.index[~df.mi_is_infected & df.is_alive]

        if df.is_alive.sum():
            prevalence = len(currently_infected) / (
                len(currently_infected) + len(currently_uninfected))
        else:
            prevalence = 0

        # 2. handle new infections
        now_infected = np.random.choice([True, False], size=len(currently_uninfected),
                                        p=[prevalence, 1 - prevalence])

        # if any are infected
        if now_infected.sum():
            infected_idx = currently_uninfected[now_infected]

            death_years_ahead = np.random.exponential(scale=2, size=now_infected.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')
            symptoms = self.module.rng.choice(
                self.module.parameters['level_of_symptoms']['level_of_symptoms'],
                size=now_infected.sum(),
                p=self.module.parameters['level_of_symptoms']['probability'])

            df.loc[infected_idx, 'mi_is_infected'] = True
            df.loc[infected_idx, 'mi_status'] = 'C'
            df.loc[infected_idx, 'mi_date_infected'] = self.sim.date
            df.loc[infected_idx, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[infected_idx, 'mi_date_cure'] = pd.NaT
            df.loc[infected_idx, 'mi_specific_symptoms'] = symptoms
            df.loc[infected_idx, 'mi_unified_symptom_code'] = 0

            # schedule death events for newly infected individuals
            for person_index in infected_idx:
                death_event = MockitisDeathEvent(self, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'mi_scheduled_date_death'])

            # Determine if anyone with severe symptoms will seek care
            serious_symptoms = (df['is_alive']) & ((df['mi_specific_symptoms'] == 'extreme emergency') | (
                df['mi_specific_symptoms'] == 'coughing and irritiable'))

            seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
            for i in df.index[serious_symptoms]:
                prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = self.module.rng.rand() < prob

            if seeks_care.sum() > 0:
                for person_index in seeks_care.index[seeks_care == True]:
                    logger.debug(
                        'This is MockitisEvent, scheduling Mockitis_PresentsForCareWithSevereSymptoms for person %d',
                        person_index)
                    event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
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

# TODO: remove afterwards - no HT deaths
class HTDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'ht_current_status'] == 'C':
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='HT')
            self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_Mockitis_PresentsForCareWithSevereSymptoms(Event, IndividualScopeEventMixin):

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
        self.TREATMENT_ID = 'Mockitis_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id):

        logger.debug('This is HSI_Mockitis_PresentsForCareWithSevereSymptoms, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: there should now be treatment for person %d',
                person_id)
            event = HSI_Mockitis_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

        else:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=date_turns_15,
                                                            tclose=date_turns_15 + DateOffset(months=12))


class HSI_Mockitis_StartTreatment(Event, IndividualScopeEventMixin):
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



class MockitisLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        infected_total = df.loc[df.is_alive, 'mi_is_infected'].sum()
        proportion_infected = infected_total / len(df)

        mask: pd.Series = (df.loc[df.is_alive, 'mi_date_infected'] >
                           self.sim.date - DateOffset(months=self.repeat))
        infected_in_last_month = mask.sum()
        mask = (df.loc[df.is_alive, 'mi_date_cure'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        counts.update(df.loc[df.is_alive, 'mi_status'].value_counts().to_dict())

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalInf': infected_total,
                        'PropInf': proportion_infected,
                        'PrevMonth': infected_in_last_month,
                        'Cured': cured_in_last_month,
                    })

        logger.info('%s|status_counts|%s', self.sim.date, counts)
