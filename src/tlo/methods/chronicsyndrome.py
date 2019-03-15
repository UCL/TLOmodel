"""
A skeleton template for disease methods.
"""
import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import EmergencyHealthSystemInteraction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChronicSyndrome(Module):
    """This is a dummy chronic disease

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'p_acquisition': Parameter(
            Types.REAL,
            'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL,
            'Level of symptoms that the individual will have'),
        'p_cure': Parameter(
            Types.REAL,
            'Probability that a treatment is succesful in curing the individual'),
        'initial_prevalence': Parameter(
            Types.REAL,
            'Prevalence of the disease in the initial population'),
        'prob_dev_severe_symptoms_per_year': Parameter(
            Types.REAL,
            'Probability per year of severe symptoms developing'),
        'prob_severe_symptoms_seek_emergency_care': Parameter(
            Types.REAL,
            'Probability that an individual will seak emergency care on developing extreme illneess'),
        'qalywt_ill': Parameter(
            Types.REAL, 'QALY weighting')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'cs_has_cs': Property(
            Types.BOOL, 'Current status of mockitis'),
        'cs_status': Property(
            Types.CATEGORICAL,
            'Historical status: N=never; C=currently 2; P=previously',
            categories=['N', 'C', 'P']),
        'cs_date_acquired': Property(
            Types.DATE,
            'Date of latest infection'),
        'cs_scheduled_date_death': Property(
            Types.DATE,
            'Date of scheduled death of infected individual'),
        'cs_date_cure': Property(
            Types.DATE,
            'Date an infected individual was cured'),
        'cs_specific_symptoms': Property(
            Types.CATEGORICAL,
            'Level of symptoms for chronic syndrome specifically',
            categories=['none', 'extreme illness']),
        'cs_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4])
    }

    TREATMENT_ID = 'ChronicSyndrome_Treatment'

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        For now, we are going to hard code them explicity
        """
        self.parameters['p_acquisition_per_year'] = 0.10
        self.parameters['p_cure'] = 0.10
        self.parameters['initial_prevalence'] = 0.30
        self.parameters['level_of_symptoms'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none', 'extreme illness'],
                'probability': [0.95, 0.05]
            })
        self.parameters['prob_dev_severe_symptoms_per_year'] = 0.50
        self.parameters['prob_severe_symptoms_seek_emergency_care'] = 0.95

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the dataframe storing data for individiuals
        p = self.parameters

        # Set default for properties
        df.loc[df.is_alive, 'cs_has_cs'] = False  # default: no individuals infected
        df.loc[df.is_alive, 'cs_status'].values[:] = 'N'  # default: never infected
        df.loc[df.is_alive, 'cs_date_acquired'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'cs_scheduled_date_death'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'cs_date_cure'] = pd.NaT  # default: not a time
        df.loc[df.is_alive, 'cs_specific_symptoms'] = 'none'
        df.loc[df.is_alive, 'cs_unified_symptom_code'] = 0

        # randomly selected some individuals as infected
        num_alive = df.is_alive.sum()
        df.loc[df.is_alive, 'cs_has_cs'] = self.rng.random_sample(size=num_alive) < p['initial_prevalence']
        df.loc[df.cs_has_cs, 'cs_status'] = 'C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        acquired_count = df.cs_has_cs.sum()

        # Assign level of symptoms
        symptoms = self.rng.choice(p['level_of_symptoms']['level_of_symptoms'],
                                   size=acquired_count,
                                   p=p['level_of_symptoms']['probability'])
        df.loc[df.cs_has_cs, 'cs_specific_symptoms'] = symptoms

        # date acquired cs
        # sample years in the past
        acquired_years_ago = np.random.exponential(scale=10, size=acquired_count)

        # pandas requires 'timedelta' type for date calculations
        # TODO: timedelta arithmetic requires days
        acquired_td_ago = pd.to_timedelta(acquired_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=20, size=acquired_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.cs_has_cs, 'cs_date_infected'] = self.sim.date - acquired_td_ago
        df.loc[df.cs_has_cs, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead

        # get the QALY values that this module will use from the weight database (these codes are just random!)
        p['qalyw_ill'] = self.sim.modules['QALY'].get_qaly_weight(87)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event (we will implement below)
        event = ChronicSyndromeEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(ChronicSyndromeLoggingEvent(self), sim.date + DateOffset(months=6))

        # # add the death event of individuals with ChronicSyndrome
        df = sim.population.props  # a shortcut to the dataframe storing data for individiuals
        indicies_of_persons_who_will_die = df.index[df.cs_has_cs]
        for person_index in indicies_of_persons_who_will_die:
            death_event = ChronicSyndromeDeathEvent(self, person_index)
            self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Register with the HealthSystem the treatment interventions that this module runs
        # and define the footprint that each intervention has on the common resources
        # Define the footprint for the intervention on the common resources
        footprint_for_treatment = pd.DataFrame(index=np.arange(1),
                                               data={
                                                   'Name': ChronicSyndrome.TREATMENT_ID,
                                                   'Nurse_Time': 30,
                                                   'Doctor_Time': 200,
                                                   'Electricity': True,
                                                   'Water': True
                                               })

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """
        df = self.sim.population.props  # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:
        df.at[child_id, 'cs_has_cs'] = False
        df.at[child_id, 'cs_status'] = 'N'
        df.at[child_id, 'cs_date_acquired'] = pd.NaT
        df.at[child_id, 'cs_scheduled_date_death'] = pd.NaT
        df.at[child_id, 'cs_date_cure'] = pd.NaT
        df.at[child_id, 'cs_specific_symptoms'] = 'none'
        df.at[child_id, 'cs_unified_symptom_code'] = 0

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale

        logger.debug("This is chronicsyndome, being asked to report unified symptomology")
        # print('Now being asked to update symptoms')
        # df['cs_unified_symptom_code']
        # df['cs_specific_symptoms']

        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

        df.loc[df.is_alive, 'cs_unified_symptom_code'] = df.loc[df.is_alive, 'cs_specific_symptoms'].map(
            {
                'none': 0,
                'extreme illness': 4
            })

        return df.loc[df.is_alive, 'cs_unified_symptom_code']

    def on_first_healthsystem_interaction(self, person_id, cue_type):
        logging.debug('being asked what to do at a health system appointment for person', person_id)

        # Queries whether treatment is allowable under global policy
        allowable = True

        # Queries whether treatment is available locally
        available = True

        if allowable and available:
            # # Commission treatment for this individual
            event = ChronicSyndromeTreatmentEvent(self, person_id)
            self.sim.schedule_event(event, self.sim.date)

    def on_followup_healthsystem_interaction(self, person_id):
        logging.debug('This is a follow-up appointment. Nothing to do')

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over the past year

        logging.debug('This is chronicsyndrome reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = df.loc[df.is_alive, 'cs_specific_symptoms'].map(
            {
                'none': 0,
                'extreme illness': self.parameters['qalyw_ill']
            })

        return health_values.loc[df.is_alive]


class ChronicSyndromeEvent(RegularEvent, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        p = self.module.parameters
        rng: np.random.RandomState = self.module.rng

        # 1. get (and hold) index of currently infected and uninfected individuals
        # currently_cs = df.index[df.cs_has_cs & df.is_alive]
        currently_not_cs = df.index[~df.cs_has_cs & df.is_alive]

        # 2. handle new cases
        p_aq = p['p_acquisition_per_year'] / 12.0
        now_acquired = rng.random_sample(size=len(currently_not_cs)) < p_aq

        # if any are new cases
        if now_acquired.sum():
            newcases_idx = currently_not_cs[now_acquired]

            death_years_ahead = rng.exponential(scale=20, size=now_acquired.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')
            symptoms = rng.choice(p['level_of_symptoms']['level_of_symptoms'],
                                  size=now_acquired.sum(),
                                  p=p['level_of_symptoms']['probability'])

            df.loc[newcases_idx, 'cs_has_cs'] = True
            df.loc[newcases_idx, 'cs_status'] = 'C'
            df.loc[newcases_idx, 'cs_date_acquired'] = self.sim.date
            df.loc[newcases_idx, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[newcases_idx, 'cs_date_cure'] = pd.NaT
            df.loc[newcases_idx, 'cs_specific_symptoms'] = symptoms
            df.loc[newcases_idx, 'cs_unified_symptom_code'] = 0

            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])

        # 3) Handle progression to severe symptoms
        curr_cs_not_severe = df.index[df.cs_has_cs &
                                      df.is_alive &
                                      (df.cs_specific_symptoms != 'extreme illness')]

        become_severe = rng.random_sample(size=len(curr_cs_not_severe)) < p['prob_dev_severe_symptoms_per_year'] / 12
        become_severe_idx = curr_cs_not_severe[become_severe]
        df.loc[become_severe_idx, 'cs_specific_symptoms'] = 'extreme illness'

        # 4) With some probability, the new severe cases seek "Emergency care"...
        if len(become_severe_idx) > 0:
            random_sample = rng.random_sample(size=len(become_severe_idx))
            seeks_emergency_care = random_sample < p['prob_severe_symptoms_seek_emergency_care']
            seeks_emergency_care_idx = become_severe_idx[seeks_emergency_care]

            for person_index in seeks_emergency_care_idx:
                event = EmergencyHealthSystemInteraction(self.module, person_index)
                self.sim.schedule_event(event, self.sim.date)


class ChronicSyndromeDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'mi_status'] == 'C':
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='ChronicSyndrome')
            self.sim.schedule_event(death, self.sim.date)


class ChronicSyndromeTreatmentEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("We are now ready to treat this person %d", person_id)

        df = self.sim.population.props
        treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks:
            df.at[person_id, 'cs_has_cs'] = False
            df.at[person_id, 'cs_status'] = 'P'
            # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'cs_scheduled_date_death'] = pd.NaT
            df.at[person_id, 'cs_date_cure'] = self.sim.date
            df.at[person_id, 'cs_specific_symptoms'] = 'none'
            df.at[person_id, 'cs_unified_symptom_code'] = 0


class ChronicSyndromeLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        hascs_total = (df.is_alive & df.cs_has_cs).sum()
        proportion_infected = hascs_total / df.is_alive.sum()
