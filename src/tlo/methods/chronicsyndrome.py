import numpy as np
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


class ChronicSyndrome(Module):
    """
    This is a dummy chronic disease
    It demonstrates the following behaviours in respect of the healthsystem module:

        - Registration of the disease module with health system
        - Internal symptom tracking and health care seeking
        - Outreach campaigns
        - Piggy-backing appointments
        - Reporting two sets of DALY weights with specific labels
        - Usual HSI behaviour
        - Population-wide HSI event
        - On-the-fly consumables access
        - Returning an update footprint
        - Receiving a 'squeeze factor'
        - Use of the SymptomManager
    """

    # Declare modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

    # Declare optional modules that need to be registered in simulation and initialised
    # before this module if present
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
        'ChronicSyndrome': Cause(label='ChronicSyndrome_Disability_And_Death'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'ChronicSyndrome': Cause(label='ChronicSyndrome_Disability_And_Death'),
    }

    PARAMETERS = {
        'p_acquisition_per_year': Parameter(Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'prob_of_symptoms': Parameter(
            Types.DICT, 'Probabilities of developing each type of symptom at onset'),
        'p_cure': Parameter(Types.REAL, 'Probability that a treatment is succesful in curing the individual'),
        'initial_prevalence': Parameter(Types.REAL, 'Prevalence of the disease in the initial population'),
        'prob_dev_symptom_craving_sandwiches': Parameter(
            Types.REAL, 'Probability per year of developing severe symptoms of craving sandwiches'
        ),
        'prob_seek_emergency_care_if_craving_sandwiches': Parameter(
            Types.REAL, 'Probability that an individual will seak emergency care following onset of craving sandwiches'
        ),
        'daly_wts': Parameter(Types.DICT, 'DALY weights for conditions'),
    }

    PROPERTIES = {
        'cs_has_cs': Property(Types.BOOL, 'Current status of mockitis'),
        'cs_status': Property(
            Types.CATEGORICAL, 'Historical status: N=never; C=currently 2; P=previously', categories=['N', 'C', 'P']
        ),
        'cs_date_acquired': Property(Types.DATE, 'Date of latest infection'),
        'cs_scheduled_date_death': Property(Types.DATE, 'Date of scheduled death of infected individual'),
        'cs_date_cure': Property(Types.DATE, 'Date an infected individual was cured'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        For now, we are going to hard code them explicity.
        Register the module with the health system and register the symptoms
        """
        self.parameters['p_acquisition_per_year'] = 0.10
        self.parameters['p_cure'] = 0.10
        self.parameters['initial_prevalence'] = 0.30
        self.parameters['prob_of_symptoms'] = {
            'inappropriate_jokes': 0.95,
            'craving_sandwiches': 0.5
        }
        self.parameters['prob_dev_severe_symptoms_per_year'] = 0.50
        self.parameters['prob_severe_symptoms_seek_emergency_care'] = 0.95

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight that this module will use from the weight database (these codes are just random!)
            self.parameters['daly_wts'] = {
                'inappropriate_jokes': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=86),
                'craving_sandwiches': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=87)
            }

        # Register symptoms that this module will use:
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='inappropriate_jokes',
                odds_ratio_health_seeking_in_adults=3.0
            ),
            Symptom.emergency('craving_sandwiches', which='adults')
        )

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

        # randomly selected some individuals as infected
        num_alive = df.is_alive.sum()
        df.loc[df.is_alive, 'cs_has_cs'] = self.rng.random_sample(size=num_alive) < p['initial_prevalence']
        df.loc[df.cs_has_cs, 'cs_status'].values[:] = 'C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        acquired_count = df.cs_has_cs.sum()

        # Assign level of symptoms to each person with cd:
        person_id_all_with_cs = list(df[df.cs_has_cs].index)

        for symp in self.parameters['prob_of_symptoms']:
            # persons who will have symptoms (each can occur independently)
            persons_id_with_symp = np.array(person_id_all_with_cs)[
                self.rng.rand(len(person_id_all_with_cs)) < self.parameters['prob_of_symptoms'][symp]
                ]

            self.sim.modules['SymptomManager'].change_symptom(
                person_id=list(persons_id_with_symp),
                symptom_string=symp,
                add_or_remove='+',
                disease_module=self
            )

        # date acquired cs
        # sample years in the past
        acquired_years_ago = self.rng.exponential(scale=10, size=acquired_count)

        # pandas requires 'timedelta' type for date calculations
        acquired_td_ago = pd.to_timedelta(acquired_years_ago * DAYS_IN_YEAR, unit='D')

        # date of death of the infected individuals (in the future)
        death_years_ahead = self.rng.exponential(scale=20, size=acquired_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

        # set the properties of infected individuals
        df.loc[df.cs_has_cs, 'cs_date_acquired'] = self.sim.date - acquired_td_ago
        df.loc[df.cs_has_cs, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead

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

        # Schedule the event that will launch the Outreach event
        outreach_event = ChronicSyndrome_LaunchOutreachEvent(self)
        self.sim.schedule_event(outreach_event, self.sim.date + DateOffset(months=6))

        # Schedule the occurance of a population wide change in risk that goes through the health system:
        popwide_hsi_event = HSI_ChronicSyndrome_PopulationWideBehaviourChange(self)
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            popwide_hsi_event, priority=1, topen=self.sim.date, tclose=None
        )
        logger.debug(key='debug', data='The population wide HSI event has been scheduled successfully!')

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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug(
            key='debug',
            data=('This is ChronicSyndrome, being alerted about a health system interaction '
                  f'person {person_id} for: {treatment_id}'),
        )

        # To simulate a 'piggy-backing' appointment, whereby additional treatment and test are done
        # for another disease, schedule another appointment (with smaller resources than a full appointmnet)
        # and set it to priority 0 (to give it highest possible priority).

        if treatment_id == 'Mockitis_TreatmentMonitoring':
            piggy_back_dx_at_appt = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(self, person_id)
            piggy_back_dx_at_appt.TREATMENT_ID = 'ChronicSyndrome_PiggybackAppt'

            # Arbitrarily reduce the size of appt footprint to reflect that this is a piggy back appt
            for key in piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT:
                piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[key] = piggy_back_dx_at_appt.EXPECTED_APPT_FOOTPRINT[
                                                                         key] * 0.25

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                piggy_back_dx_at_appt, priority=0, topen=self.sim.date, tclose=None
            )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key='debug', data='This is chronicsyndrome reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        for symptom, daly_wt in self.parameters['daly_wts'].items():
            health_values.loc[
                health_values.index.isin(self.sim.modules['SymptomManager'].who_has(symptom))
            ] += daly_wt

        return health_values


class ChronicSyndromeEvent(RegularEvent, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, ChronicSyndrome)

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
            death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

            df.loc[newcases_idx, 'cs_has_cs'] = True
            df.loc[newcases_idx, 'cs_status'].values[:] = 'C'
            df.loc[newcases_idx, 'cs_date_acquired'] = self.sim.date
            df.loc[newcases_idx, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[newcases_idx, 'cs_date_cure'] = pd.NaT

            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self.module, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])

            # Assign symptoms:
            for symp in self.module.parameters['prob_of_symptoms']:
                # persons who will have symptoms (each can occur independently)
                persons_id_with_symp = np.array(newcases_idx)[
                    self.module.rng.rand(len(newcases_idx)) < self.module.parameters['prob_of_symptoms'][symp]
                    ]

                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=list(persons_id_with_symp),
                    symptom_string=symp,
                    add_or_remove='+',
                    disease_module=self.module
                )

        # 3) Handle progression to severe symptoms
        curr_cs_but_not_craving_sandwiches = list(set(df.index[df.cs_has_cs & df.is_alive])
                                                  - set(
            self.sim.modules['SymptomManager'].who_has('craving_sandwiches')))

        become_severe = (
            self.module.rng.random_sample(size=len(curr_cs_but_not_craving_sandwiches))
            < p['prob_dev_severe_symptoms_per_year'] / 12
        )
        become_severe_idx = np.array(curr_cs_but_not_craving_sandwiches)[become_severe]

        self.sim.modules['SymptomManager'].change_symptom(
            person_id=list(become_severe_idx),
            symptom_string='craving_sandwiches',
            add_or_remove='+',
            disease_module=self.module
        )


class ChronicSyndromeDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'cs_status'] == 'C':
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='ChronicSyndrome')
            self.sim.schedule_event(death, self.sim.date)


class ChronicSyndrome_LaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    This is the event that is run by ChronicSyndrome and it is the Outreach Event.
    It will now submit the individual HSI events that occur when each individual is met.
    (i.e. Any large campaign that involves contct with individual is composed of many individual outreach events).
    """

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ChronicSyndrome)

    def apply(self, population):
        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        gets_outreach = df.index[(df['is_alive']) & (df['sex'] == 'F')]
        for person_id in gets_outreach:
            # make the outreach event (let this disease module be alerted about it, and also Mockitis)
            outreach_event_for_individual = HSI_ChronicSyndrome_Outreach_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(
                outreach_event_for_individual, priority=1, topen=self.sim.date, tclose=None
            )


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events


class HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is the event when a person with the severe symptoms of chronic syndrome presents for emergency care
    and is immediately provided with treatment.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, ChronicSyndrome)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'  # Can occur at this facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            key='debug',
            data=('This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: '
                  f'We are now ready to treat this person {person_id}.'),

        )
        logger.debug(
            key='debug',
            data=('This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: '
                  f'The squeeze-factor is {squeeze_factor}.'),
        )

        if squeeze_factor < 0.5:
            # If squeeze factor is not too large:
            logger.debug(key='debug', data='Treatment will be provided.')
            df = self.sim.population.props
            treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

            if treatmentworks:
                df.at[person_id, 'cs_has_cs'] = False
                df.at[person_id, 'cs_status'] = 'P'

                # (in this we nullify the death event that has been scheduled.)
                df.at[person_id, 'cs_scheduled_date_death'] = pd.NaT
                df.at[person_id, 'cs_date_cure'] = self.sim.date

                # remove all symptoms instantly
                self.sim.modules['SymptomManager'].clear_symptoms(
                    person_id=person_id,
                    disease_module=self.module)
        else:
            # Squeeze factor is too large
            logger.debug(key='debug', data='Treatment will not be provided due to squeeze factor.')

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: did not run')
        pass


class HSI_ChronicSyndrome_Outreach_Individual(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    This event can be used to simulate the occurrence of an 'outreach' intervention.

    NB. This needs to be created and run for each individual that benefits from the outreach campaign.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, ChronicSyndrome)

        logger.debug(key='debug', data='Outreach event being created.')

        # Define the necessary information for an HSI
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'ChronicSyndrome_Outreach_Individual'

        # APPP_FOOTPRINT: outreach event takes small amount of time for DCSA
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['ConWithDCSA'] = 0.5
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'ConWithDCSA': 0.5})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 0.5})
        self.ACCEPTED_FACILITY_LEVEL = '0'  # Can occur at facility-level 0
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug', data=f'Outreach event running now for person: {person_id}', )

        # Do here whatever happens during an outreach event with an individual
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_ChronicSyndrome_Outreach_Individual: did not run')
        pass


class HSI_ChronicSyndrome_PopulationWideBehaviourChange(HSI_Event, PopulationScopeEventMixin):
    """
    This is a Population-Wide Health System Interaction Event - will change the variables to do with risk for
    ChronicSyndrome
    """

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ChronicSyndrome)

        # Define the necessary information for a Population level HSI
        self.TREATMENT_ID = 'ChronicSyndrome_PopulationWideBehaviourChange'

    def apply(self, population, squeeze_factor):
        logger.debug(key='debug', data='This is HSI_ChronicSyndrome_PopulationWideBehaviourChange')

        # As an example, we will reduce the chance of acquisition per year (due to behaviour change)
        self.module.parameters['p_acquisition_per_year'] = self.module.parameters['p_acquisition_per_year'] * 0.5


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


class ChronicSyndromeLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ There is no logging done here.
        """
        # run this event every month
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, ChronicSyndrome)

    def apply(self, population):
        pass
