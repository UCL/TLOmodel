import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, HSI_Event
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChronicSyndrome(Module):
    """
    This is a dummy chronic disease
    It demonstrates the following behaviours in respect of the healthsystem module:

        - Registration of the disease module
        - Internal symptom tracking and health care seeking
        - Outreach campaigns
        - Piggy-backing appointments
        - Reporting two sets of DALY weights with specific labels
        - Usual HSI behaviour
        - Population-wide HSI event
    """

    PARAMETERS = {
        'p_acquisition': Parameter(
            Types.REAL,
            'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL,
            'Level of symptoms that the individual will have',
            categories=['low', 'high']),
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
        'daly_wt_ill': Parameter(
            Types.REAL, 'DALY weight for being ill caused by Chronic Syndrome')
    }

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

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight that this module will use from the weight database (these codes are just random!)
            seq_code = 87  # the sequale code that is related to this disease (notionally!)
            self.parameters['daly_wt_ill'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=seq_code)

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
        acquired_td_ago = pd.to_timedelta(acquired_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=20, size=acquired_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.cs_has_cs, 'cs_date_acquired'] = self.sim.date - acquired_td_ago
        df.loc[df.cs_has_cs, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

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
        self.sim.modules['HealthSystem'].schedule_hsi_event(popwide_hsi_event,
                                                            priority=1,
                                                            topen=self.sim.date,
                                                            tclose=None)
        logger.debug('The population wide HSI event has been scheduled succesfully!')

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

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is ChronicSyndrome, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

        # To simulate a "piggy-backing" appointment, whereby additional treatment and test are done
        # for another disease, schedule another appointment (with smaller resources than a full appointmnet)
        # and set it to priority 0 (to give it highest possible priority).

        if treatment_id == 'Mockitis_TreatmentMonitoring':
            piggy_back_dx_at_appt = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(self, person_id)
            piggy_back_dx_at_appt.TREATMENT_ID = 'ChronicSyndrome_PiggybackAppt'

            # Arbitrarily reduce the size of appt footprint to reflect that this is a piggy back appt
            for key in piggy_back_dx_at_appt.APPT_FOOTPRINT:
                piggy_back_dx_at_appt.APPT_FOOTPRINT[key] = piggy_back_dx_at_appt.APPT_FOOTPRINT[key] * 0.25

            self.sim.modules['HealthSystem'].schedule_hsi_event(piggy_back_dx_at_appt,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=None)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logging.debug('This is chronicsyndrome reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        # ChronicSyndrome will produce two sets of DALYS as it want to be able to count up disability according to
        # different types of disability.

        health_values_1 = df.loc[df.is_alive, 'cs_specific_symptoms'].map(
            {
                'none': 0,
                'extreme illness': self.parameters['daly_wt_ill']
            })
        health_values_1.name = 'Extreme Illness'

        health_values_2 = df.loc[df.is_alive, 'cs_specific_symptoms'].map(
            {
                'none': 0,
                'extreme illness': 0.05
            })
        health_values_2.name = 'Extra Terrible'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive]], axis=1)

        return health_values_df  # return the dataframe


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
                death_event = ChronicSyndromeDeathEvent(self.module, person_index)
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
            for person_index in become_severe_idx:
                prob = self.sim.modules['HealthSystem'].get_prob_seek_care(person_index)
                seeks_care = self.module.rng.rand() < prob
                if seeks_care:
                    event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(self.module, person_index)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=1,
                                                                        topen=self.sim.date,
                                                                        tclose=None)


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


class ChronicSyndrome_LaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    This is the event that is run by ChronicSyndrome and it is the Outreach Event.
    It will now submit the individual HSI events that occur when each individual is met.
    (i.e. Any large campaign that involves contct with individual is composed of many individual outreach events).
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        gets_outreach = df.index[(df['is_alive']) & (df['sex'] == 'F')]
        for person_id in gets_outreach:
            # make the outreach event (let this disease module be alerted about it, and also Mockitis)
            outreach_event_for_individual = HSI_ChronicSyndrome_Outreach_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(outreach_event_for_individual,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)


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

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['AccidentsandEmerg'] = 1  # Plus, an amount of resources similar to an A&E

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at any facility level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(
            "This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: We are now ready to treat this person %d.",
            person_id)
        logger.debug(
            "This is HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: The squeeze-factor is %d.",
            squeeze_factor)


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

    def did_not_run(self):
        logger.debug('HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment: did not run')
        pass

class HSI_ChronicSyndrome_Outreach_Individual(HSI_Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = 'ChronicSyndrome_Outreach_Individual'

        # APPP_FOOTPRINT: outreach event takes small amount of time for DCSA
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['ConWithDCSA'] = 0.5
        self.APPT_FOOTPRINT = appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 0 # Can occur at facility-level 0
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        logger.debug('Outreach event running now for person: %s', person_id)

        # Do here whatever happens during an outreach event with an individual
        # ~~~~~~~~~~~~~~~~~~~~~~


        # Make request for some consumables
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

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code1: 1}, {pkg_code2:4}],
            'Item_Code': [{item_code1:1}, {item_code2:10}]
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                                                                            hsi_event = self,
                                                                            cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code1]:
            logger.debug('PkgCode1 is available, so use it.')
        else:
            logger.debug('PkgCode1 is not available, so can''t use it.')




        # Return the actual footprints
        actual_appt_footprint = self.APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_ChronicSyndrome_Outreach_Individual: did not run')
        pass


class HSI_ChronicSyndrome_PopulationWideBehaviourChange(HSI_Event, PopulationScopeEventMixin):
    """
    This is a Population-Wide Health System Interaction Event - will change the variables to do with risk for
    ChronicSyndrome
    """

    def __init__(self, module):
        super().__init__(module)

        # Define the necessary information for a Population level HSI
        self.TREATMENT_ID = 'ChronicSyndrome_PopulationWideBehaviourChange'

    def apply(self, population, squeeze_factor):
        logger.debug('This is HSI_ChronicSyndrome_PopulationWideBehaviourChange')

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

    def apply(self, population):
        pass
