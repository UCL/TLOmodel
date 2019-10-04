import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mockitis(Module):
    """
    This is a dummy infectious disease.
    It demonstrates the following behaviours in respect of the healthsystem module:
        - Registration of the disease module
        - Reading DALY weights and reporting daly values related to this disease
        - Health care seeking
        - Usual HSI behaviour
        - Restrictive requirements on the facility_level for the HSI_event
    """

    PARAMETERS = {
        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL,
            'Level of symptoms that the individual will have',
            categories=['low', 'high']),
        'p_cure': Parameter(
            Types.REAL, 'Probability that a treatment is successful in curing the individual'),
        'initial_prevalence': Parameter(
            Types.REAL, 'Prevalence of the disease in the initial population'),
        'daly_wt_mild_sneezing': Parameter(
            Types.REAL, 'DALY weight for mild sneezing'),
        'daly_wt_coughing': Parameter(
            Types.REAL, 'DALY weight for coughing'),
        'daly_wt_advanced': Parameter(
            Types.REAL, 'DALY weight for extreme emergency')
    }

    PROPERTIES = {
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

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wt_mild_sneezing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_coughing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_advanced'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals

        # Set default for properties
        df.loc[df.is_alive, 'mi_is_infected'] = False  # default: no individuals infected
        df.loc[df.is_alive, 'mi_status'].values[:] = 'N'  # default: never infected
        df.loc[df.is_alive, 'mi_date_infected'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_scheduled_date_death'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_date_cure'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_specific_symptoms'] = 'none'
        df.loc[df.is_alive, 'mi_unified_symptom_code'] = 0

        alive_count = df.is_alive.sum()

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
        infected_td_ago = pd.to_timedelta(infected_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=20, size=infected_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.mi_is_infected, 'mi_date_infected'] = self.sim.date - infected_td_ago
        df.loc[df.mi_is_infected, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = MockitisEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(MockitisLoggingEvent(self), sim.date + DateOffset(months=6))

        # a shortcut to the dataframe storing data for individiuals
        df = sim.population.props

        # add the death event of infected individuals
        # schedule the mockitis death event
        people_who_will_die = df.index[df.mi_is_infected]
        for person_id in people_who_will_die:
            self.sim.schedule_event(MockitisDeathEvent(self, person_id),
                                    df.at[person_id, 'mi_scheduled_date_death'])

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Mockitis, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is mockitis reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'mi_specific_symptoms'].map({
            'none': 0,
            'mild sneezing': p['daly_wt_mild_sneezing'],
            'coughing and irritable': p['daly_wt_coughing'],
            'extreme emergency': p['daly_wt_advanced']
        })
        health_values.name = 'Mockitis Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series


class MockitisEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of Mockitis.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('This is MockitisEvent, tracking the disease progression of the population.')

        df = population.props

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
                death_event = MockitisDeathEvent(self.module, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'mi_scheduled_date_death'])

            # Determine if anyone with severe symptoms will seek care
            serious_symptoms = (df['is_alive']) & ((df['mi_specific_symptoms'] == 'extreme emergency') | (
                df['mi_specific_symptoms'] == 'coughing and irritiable'))

            seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
            for i in df.index[serious_symptoms]:
                prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = self.module.rng.rand() < prob

            if seeks_care.sum() > 0:
                for person_index in seeks_care.index[seeks_care is True]:
                    logger.debug(
                        'This is MockitisEvent, scheduling Mockitis_PresentsForCareWithSevereSymptoms for person %d',
                        person_index)
                    event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )
            else:
                logger.debug(
                    'This is MockitisEvent, There is  no one with new severe symptoms so no new healthcare seeking')
        else:
            logger.debug('This is MockitisEvent, no one is newly infected.')


class MockitisDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'mi_status'] == 'C':

            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='Mockitis')
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
        self.ACCEPTED_FACILITY_LEVELS = [0]     # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_Mockitis_PresentsForCareWithSevereSymptoms, a first appointment for person %d',
                     person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: \
                there should now be treatment for person %d',
                person_id)
            event = HSI_Mockitis_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=None)

        else:
            logger.debug(
                '...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
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
                                                  'Intervention_Pkg'] ==
                                              'First line treatment for new TB cases for adults',
                                              'Intervention_Pkg_Code'])[0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'MDR notification among previously treated patients',
                                              'Intervention_Pkg_Code'])[0]

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
        self.ACCEPTED_FACILITY_LEVELS = [1, 2]  # Enforces that this apppointment must happen at those facility-levels
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

        logger.debug(
            '....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2)
                                                            )


class HSI_Mockitis_TreatmentMonitoring(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitis is monitored.
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
                                                  'Intervention_Pkg'] ==
                                              'First line treatment for new TB cases for adults',
                                              'Intervention_Pkg_Code'])[0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'MDR notification among previously treated patients',
                                              'Intervention_Pkg_Code'])[0]

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
        self.ACCEPTED_FACILITY_LEVELS = ['*']   # Allows this HSI to occur at any facility-level
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
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
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
