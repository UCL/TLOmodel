
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Mockitis(Module):
    """This is a dummy infectious disease.

    It demonstrates the following behaviours in respect of the healthsystem module:

        - Registration of the disease module with healthsystem
        - Reading DALY weights and reporting daly values related to this disease
        - Health care seeking
        - Usual HSI behaviour
        - Restrictive requirements on the facility_level for the HSI_event
        - Use of the SymptomManager
    """

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Mockitis': Cause(label='Mockitis_Disability_And_Death'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Mockitis': Cause(label='Mockitis_Disability_And_Death')
    }

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
        'daly_wts': Parameter(
            Types.DICT, 'DALY weights for conditions'),
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
            Types.DATE, 'Date an infected individual was cured')
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read in parameters and do the registration of this module and its symptoms"""

        p = self.parameters

        p['p_infection'] = 0.001
        p['p_cure'] = 0.99
        p['initial_prevalence'] = 0.5

        # The distribution of symptoms that may be caused at onset by mockitis
        p['level_of_symptoms'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none',
                                      'weird_sense_of_deja_vu',
                                      'coughing_and_irritable',
                                      'extreme_pain_in_the_nose'],
                'probability': [0.25, 0.25, 0.25, 0.25]
            })

        # Get the DALY weight that this module will use from the weights database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wts'] = {
                'weird_sense_of_deja_vu': self.sim.modules['HealthBurden'].get_daly_weight(48),
                'coughing_and_irritable': self.sim.modules['HealthBurden'].get_daly_weight(49),
                'extreme_pain_in_the_nose': self.sim.modules['HealthBurden'].get_daly_weight(50)
            }

        # ---- Register the Symptoms ----
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name='weird_sense_of_deja_vu'),  # will not trigger any health seeking behaviour
            Symptom(name='coughing_and_irritable'),  # will not trigger any health seeking behaviour
            Symptom.emergency('extreme_pain_in_the_nose')
        )

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
        df.loc[df.is_alive, 'mi_status'] = 'N'  # default: never infected
        df.loc[df.is_alive, 'mi_date_infected'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_scheduled_date_death'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'mi_date_cure'] = pd.NaT  # default: not a time

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

        for person_id_infected in df.index[df.mi_is_infected]:
            symptom_string_for_this_person = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                                             p=level_of_symptoms.probability)

            if symptom_string_for_this_person != 'none':
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=person_id_infected,
                    symptom_string=symptom_string_for_this_person,
                    add_or_remove='+',
                    disease_module=self,
                    duration_in_days=20
                )

        # date of infection of infected individuals: sample years in the past
        infected_years_ago = self.rng.exponential(scale=5, size=infected_count)

        # pandas requires 'timedelta' type for date calculations
        infected_td_ago = pd.to_timedelta(infected_years_ago * DAYS_IN_YEAR, unit='D')

        # date of death of the infected individuals (in the future)
        death_years_ahead = self.rng.exponential(scale=20, size=infected_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

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

        # Store an item code for a consumable
        self.cons_code = 0

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
            death_years_ahead = self.rng.exponential(scale=10)
            death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

            # Level of symptoms
            level_of_symptoms = self.parameters['level_of_symptoms']
            symptom_string_for_this_person = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                                             p=level_of_symptoms.probability)

            if symptom_string_for_this_person != 'none':
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=child_id,
                    symptom_string=symptom_string_for_this_person,
                    add_or_remove='+',
                    disease_module=self
                )

            # Assign properties
            df.at[child_id, 'mi_is_infected'] = True
            df.at[child_id, 'mi_status'] = 'C'
            df.at[child_id, 'mi_date_infected'] = self.sim.date
            df.at[child_id, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.at[child_id, 'mi_date_cure'] = pd.NaT

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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug(key='debug',
                     data='This is Mockitis, being alerted about a health system interaction '
                          'person %d for: %s' % (person_id, treatment_id))

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key='debug', data='This is mockitis reporting my daly values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        for symptom, daly_wt in self.parameters['daly_wts'].items():
            health_values.loc[
                self.sim.modules['SymptomManager'].who_has(symptom)
            ] += daly_wt

        return health_values  # returns the series


class MockitisEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of Mockitis.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Mockitis)

    def apply(self, population):

        logger.debug(key='debug', data='This is MockitisEvent, tracking the disease progression of the population.')

        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected = df.index[df.mi_is_infected & df.is_alive]
        currently_susc = df.index[(df.is_alive) & (df['mi_status'] == 'N')]

        if df.is_alive.sum():
            prevalence = len(currently_infected) / (
                len(currently_infected) + len(currently_susc))
        else:
            prevalence = 0

        # 2. handle new infections
        now_infected = self.module.rng.choice([True, False],
                                              size=len(currently_susc),
                                              p=[prevalence, 1 - prevalence])

        # if any are newly infected...
        if now_infected.sum():
            infected_idx = currently_susc[now_infected]

            death_years_ahead = 5  # self.module.rng.exponential(scale=30, size=now_infected.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

            df.loc[infected_idx, 'mi_is_infected'] = True
            df.loc[infected_idx, 'mi_status'] = 'C'
            df.loc[infected_idx, 'mi_date_infected'] = self.sim.date
            df.loc[infected_idx, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[infected_idx, 'mi_date_cure'] = pd.NaT

            # schedule death events for newly infected individuals
            for person_index in infected_idx:
                death_event = MockitisDeathEvent(self.module, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'mi_scheduled_date_death'])

            # assign symptoms
            level_of_symptoms = self.module.parameters['level_of_symptoms']
            for person_id_infected in infected_idx:
                symptom_string_for_this_person = self.module.rng.choice(level_of_symptoms.level_of_symptoms,
                                                                        p=level_of_symptoms.probability)

                if symptom_string_for_this_person != 'none':
                    self.sim.modules['SymptomManager'].change_symptom(
                        person_id=person_id_infected,
                        symptom_string=symptom_string_for_this_person,
                        add_or_remove='+',
                        disease_module=self.module,
                        date_of_onset=self.sim.date + DateOffset(days=1 + int(self.module.rng.rand() * 5)),
                        duration_in_days=30
                    )

        else:
            logger.debug(key='debug', data='This is MockitisEvent, no one is newly infected.')


class MockitisDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Mockitis)

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

class HSI_Mockitis_PresentsForCareWithSevereSymptoms(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Mockitis.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Mockitis)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_PresentsForCareWithSevereSymptoms'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'  # This enforces that the appointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            key='debug',
            data=f'This is HSI_Mockitis_PresentsForCareWithSevereSymptoms, a first appointment for person {person_id}',
        )

        logger.debug(
            key='debug',
            data=('...This is HSI_Mockitis_PresentsForCareWithSevereSymptoms: '
                  f'there should now be treatment for person {person_id}'),
        )

        event = HSI_Mockitis_StartTreatment(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_Mockitis_PresentsForCareWithSevereSymptoms: did not run')
        # return False to prevent this event from being rescheduled if it did not run.
        return False


class HSI_Mockitis_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is inititaed.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Mockitis)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_Treatment_Initiation'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug',
                     data=f'This is HSI_Mockitis_StartTreatment: initiating treatent for person {person_id}')
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks and self.get_consumables(self.module.cons_code):
            df.at[person_id, 'mi_is_infected'] = False
            df.at[person_id, 'mi_status'] = 'P'

            # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT

            df.at[person_id, 'mi_date_cure'] = self.sim.date

            # remove symptoms instantaneously
            self.module.sim.modules['SymptomManager'].clear_symptoms(
                person_id=person_id,
                disease_module=self.module)

        # Create a follow-up appointment
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            key='debug',
            data=('....This is HSI_Mockitis_StartTreatment: '
                  f'scheduling a follow-up appointment for person {person_id} on date {target_date_for_followup_appt}'),
        )

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2)
                                                            )

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_Mockitis_StartTreatment: did not run')
        pass


class HSI_Mockitis_TreatmentMonitoring(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitis is monitored.
    (In practise, nothing happens!)

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Mockitis)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_TreatmentMonitoring'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1, 'NewAdult': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        # There is a follow-up appoint happening now but it has no real effect!

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            # The person is not alive, the event did not happen: so return a blank footprint
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Create the next follow-up appointment....
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            key='debug',
            data=('....This is HSI_Mockitis_StartTreatment: '
                  f'scheduling a follow-up appointment for person {person_id} on date {target_date_for_followup_appt}'),
        )

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_hsi_event(followup_appt,
                                                            priority=2,
                                                            topen=target_date_for_followup_appt,
                                                            tclose=target_date_for_followup_appt + DateOffset(weeks=2))

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_Mockitis_TreatmentMonitoring: did not run')
        pass


# ---------------------------------------------------------------------------------


class MockitisLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Mockitis)

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

        logger.info(key='summary',
                    data={'TotalInf': infected_total,
                          'PropInf': proportion_infected,
                          'PrevMonth': infected_in_last_month,
                          'Cured': cured_in_last_month,
                          })

        logger.info(key='status_counts', data=counts)
