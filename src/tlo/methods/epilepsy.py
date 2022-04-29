from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

# todo: note this code is becoming very depracated and does not include health interactions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Epilepsy(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.item_codes = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthBurden', 'HealthSystem', 'SymptomManager'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Epilepsy': Cause(gbd_causes='Idiopathic epilepsy', label='Epilepsy'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Epilepsy': Cause(gbd_causes='Idiopathic epilepsy', label='Epilepsy'),
    }

    # Module parameters
    PARAMETERS = {
        'init_epil_seiz_status': Parameter(Types.LIST, 'Proportions in each seizure status category at baseline'),
        'init_prop_antiepileptic_seiz_stat_1': Parameter(
            Types.REAL, 'initial proportions on antiepileptic by if seizure status = 1'
        ),
        'init_prop_antiepileptic_seiz_stat_2': Parameter(
            Types.REAL, 'initial proportions on antiepileptic by if seizure status = 2'
        ),
        'init_prop_antiepileptic_seiz_stat_3': Parameter(
            Types.REAL, 'initial proportions on antiepileptic by if seizure status = 3'
        ),
        'base_3m_prob_epilepsy': Parameter(Types.REAL, 'base probability of epilepsy per 3 month period if age < 20'),
        'rr_epilepsy_age_ge20': Parameter(Types.REAL, 'relative rate of epilepsy if age over 20'),
        'prop_inc_epilepsy_seiz_freq': Parameter(
            Types.REAL, 'proportion of incident epilepsy cases with frequent seizures'
        ),
        'base_prob_3m_seiz_stat_freq_infreq': Parameter(
            Types.REAL, 'base probability per 3 months of seizure status frequent if current infrequent'
        ),
        'rr_effectiveness_antiepileptics': Parameter(
            Types.REAL, 'relative rate of seizure status frequent if current infrequent if on antiepileptic'
        ),
        'base_prob_3m_seiz_stat_infreq_freq': Parameter(
            Types.REAL, 'base probability per 3 months of seizure status infrequent if current frequent'
        ),
        'base_prob_3m_seiz_stat_infreq_none': Parameter(
            Types.REAL, 'base probability per 3 months of seizure status infrequent if current nonenow'
        ),
        'base_prob_3m_seiz_stat_none_freq': Parameter(
            Types.REAL, 'base probability per 3 months of seizure status nonenow if current frequent'
        ),
        'base_prob_3m_seiz_stat_none_infreq': Parameter(
            Types.REAL, 'base probability per 3 months of seizure status nonenow if current infrequent'
        ),
        'base_prob_3m_stop_antiepileptic': Parameter(
            Types.REAL, 'base probability per 3 months of stopping antiepileptic, if nonenow seizures'
        ),
        'rr_stop_antiepileptic_seiz_infreq_or_freq': Parameter(
            Types.REAL, 'relative rate of stopping antiepileptic if infrequent or frequent seizures'
        ),
        'base_prob_3m_epi_death': Parameter(Types.REAL, 'base probability per 3 months of epilepsy death'),
        # these definitions for disability weights are the ones in the global burden of disease list (Salomon)
        'daly_wt_epilepsy_severe': Parameter(
            Types.REAL, 'disability weight for severe epilepsy' 'controlled phase - code 860'
        ),
        'daly_wt_epilepsy_less_severe': Parameter(
            Types.REAL, 'disability weight for less severe epilepsy' 'controlled phase - code 861'
        ),
        'daly_wt_epilepsy_seizure_free': Parameter(
            Types.REAL, 'disability weight for less severe epilepsy' 'controlled phase - code 862'
        ),
    }

    """
    860,Severe epilepsy,'Epilepsy, seizures >= once a month','has sudden seizures one or more times
    each month, with violent muscle contractions and stiffness, loss of consciousness, and loss of
    urine or bowel control. Between seizures the person has memory loss and difficulty concentrating.
    ',0.552,0.375,0.71

    861,Less severe epilepsy,'Epilepsy, seizures 1-11 per year','has sudden seizures two to five times
    a year, with violent muscle contractions and stiffness, loss of consciousness, and loss of urine
    or bowel control.',0.263,0.173,0.367

    862,'Seizure-free, treated epilepsy',Generic uncomplicated disease: worry and daily medication,
    has a chronic disease that requires medication every day and causes some worry but minimal
    interference with daily activities.,0.049,0.031,0.072
    """

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'ep_seiz_stat': Property(
            Types.CATEGORICAL,
            '(0 = never epilepsy, 1 = previous seizures none now, 2 = infrequent seizures, 3 = frequent seizures)',
            categories=['0', '1', '2', '3'],
        ),
        'ep_antiep': Property(Types.BOOL, 'on antiepileptic'),
        'ep_epi_death': Property(Types.BOOL, 'epilepsy death this 3 month period'),
        'ep_unified_symptom_code': Property(Types.CATEGORICAL, '', categories=['0', '1', '2', '3']),
        'ep_disability': Property(Types.REAL, 'disability weight for current 3 month period'),
    }

    # Declaration of how we will refer to any treatments that are related to this disease.
    TREATMENT_ID = 'antiepileptic'

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Epilepsy.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight - 860-862 are the sequale codes for epilepsy
            p['daly_wt_epilepsy_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=860)
            p['daly_wt_epilepsy_less_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861)
            p['daly_wt_epilepsy_seizure_free'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=862)

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom("seizures", emergency_in_children=True, emergency_in_adults=True))

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        p = self.parameters
        rng = self.rng

        df.loc[df.is_alive, 'ep_seiz_stat'] = '0'
        df.loc[df.is_alive, 'ep_antiep'] = False
        df.loc[df.is_alive, 'ep_epi_death'] = False
        df.loc[df.is_alive, 'ep_disability'] = 0

        # allocate initial ep_seiz_stat
        alive_idx = df.index[df.is_alive]
        df.loc[alive_idx, 'ep_seiz_stat'] = self.rng.choice(['0', '1', '2', '3'], size=len(alive_idx),
                                                            p=self.parameters['init_epil_seiz_status'])
        # Assign those with epilepsy seizure status 2 and 3 the seizure symptom at the start of the simulation
        dfg = df.index[df.is_alive & ((df.ep_seiz_stat == '2') | (df.ep_seiz_stat == '3'))]
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=dfg.to_list(),
            symptom_string='seizures',
            add_or_remove='+',
            disease_module=self
        )

        def allocate_antiepileptic(status, probability):
            mask = (df.is_alive & (df.ep_seiz_stat == status))
            random_draw = rng.random_sample(size=mask.sum())
            df.loc[mask, 'ep_antiep'] = random_draw < probability

        # allocate initial on antiepileptic seiz status 1, 2 and 3
        allocate_antiepileptic('1', p['init_prop_antiepileptic_seiz_stat_1'])
        allocate_antiepileptic('2', p['init_prop_antiepileptic_seiz_stat_2'])
        allocate_antiepileptic('3', p['init_prop_antiepileptic_seiz_stat_3'])

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our three-monthly event to poll the population for epilepsy starting
        or stopping.
        """
        epilepsy_poll = EpilepsyEvent(self)
        sim.schedule_event(epilepsy_poll, sim.date + DateOffset(months=3))

        event = EpilepsyLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        # Get item_codes for the consumables used in the HSI
        hs = self.sim.modules['HealthSystem']
        self.item_codes = dict()
        self.item_codes['phenobarbitone'] = hs.get_item_code_from_item_name("Phenobarbital, 100 mg")
        self.item_codes['carbamazepine'] = hs.get_item_code_from_item_name('Carbamazepine 200mg_1000_CMST')
        self.item_codes['phenytoin'] = hs.get_item_code_from_item_name('Phenytoin sodium 100mg_1000_CMST')

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ep_seiz_stat'] = '0'
        df.at[child_id, 'ep_antiep'] = False
        df.at[child_id, 'ep_epi_death'] = False
        df.at[child_id, 'ep_disability'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(key='debug',
                     data=f'This is Epilepsy, being alerted about a health system interaction '
                          f'by person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug(key='debug', data='This is Epilepsy reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        disability_series_for_alive_persons = df.loc[df.is_alive, 'ep_disability']
        return disability_series_for_alive_persons

    def transition_seizure_stat(self):
        """
        This function handles all transitions in epilepsy seizure status, for those on and off anti epileptics. The
        function determines the current seizure status of those with epilepsy and based on their original status,
        and whether they are on anti epileptics determines a new seizure status
        :return:
        """
        # Get the parameters used to determine transitions between seizure states
        p = self.sim.modules['Epilepsy'].parameters
        prop_transition_1_2 = p['base_prob_3m_seiz_stat_infreq_none']
        prop_transition_2_1 = p['base_prob_3m_seiz_stat_none_infreq']
        prop_transition_2_3 = p['base_prob_3m_seiz_stat_freq_infreq']
        prop_transition_3_1 = p['base_prob_3m_seiz_stat_none_freq']
        prop_transition_3_2 = p['base_prob_3m_seiz_stat_infreq_freq']
        rr_effectiveness_antiepileptics = p['rr_effectiveness_antiepileptics']
        # get the population and the current seizure status of those with epilepsy and those who are on anti epileptics
        df = self.sim.population.props
        population_with_seizure_status_1 = df.index[df.is_alive & (df.ep_seiz_stat == '1')]
        seizure_status_1_on_anti_epileptics = df.index[df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep]
        population_with_seizure_status_2 = df.index[df.is_alive & (df.ep_seiz_stat == '2')]
        seizure_status_2_on_anti_epileptics = df.index[df.is_alive & (df.ep_seiz_stat == '2') & df.ep_antiep]
        population_with_seizure_status_3 = df.index[df.is_alive & (df.ep_seiz_stat == '3')]
        seizure_status_3_on_anti_epileptics = df.index[df.is_alive & (df.ep_seiz_stat == '3') & df.ep_antiep]
        # Determine who will transition from seizure status 1 to 2, first create a random number to determine likelihood
        # of transition
        random_draw_1_2 = self.rng.random_sample(size=len(population_with_seizure_status_1))
        # Get the base probability of transitioning from state 1 to state 2
        probability_of_transition_1_2 = pd.DataFrame([prop_transition_1_2] * len(random_draw_1_2), columns=['prob'],
                                                     index=population_with_seizure_status_1)
        # Reduce the risk of worsening seizures if on anti epileptics
        probability_of_transition_1_2.loc[seizure_status_1_on_anti_epileptics, 'prob'] /= \
            rr_effectiveness_antiepileptics
        # determine who will transition between seizure state 1 to 2
        changing_1_2 = population_with_seizure_status_1[probability_of_transition_1_2['prob'] > random_draw_1_2]
        # update seizure status
        df.loc[changing_1_2, 'ep_seiz_stat'] = '2'
        # Determine if those with seizure status 2 increase or decrease in severity, first get random draws for each
        # transition state
        random_draw_2_1 = self.rng.random_sample(size=len(population_with_seizure_status_2))
        random_draw_2_3 = self.rng.random_sample(size=len(population_with_seizure_status_2))
        # create a dataframe for the transition probabilities
        probability_of_transition_2 = pd.DataFrame({'prob_down': [prop_transition_2_1] * len(random_draw_2_1),
                                                    'prob_up': [prop_transition_2_3] * len(random_draw_2_1)},
                                                   index=population_with_seizure_status_2)
        # Increase the likelihood of reducing seizure status for those on anti epileptics
        probability_of_transition_2.loc[seizure_status_2_on_anti_epileptics, 'prob_down'] *= \
            rr_effectiveness_antiepileptics
        # Decrease the likelihood of reducing seizure status for those on anti epileptics
        probability_of_transition_2.loc[seizure_status_2_on_anti_epileptics, 'prob_up'] /= \
            rr_effectiveness_antiepileptics
        # Establish who has a changing seizure status
        changing_2_1 = population_with_seizure_status_2[probability_of_transition_2['prob_down'] > random_draw_2_1]
        changing_2_3 = population_with_seizure_status_2[probability_of_transition_2['prob_up'] > random_draw_2_3]
        # If someone with seizure status 2 has been selected to both increase and decrease in severity, choose a
        # transition direction based on the likelihood of transitioning states
        both_up_down_seiz_stat_2 = changing_2_1.intersection(changing_2_3)
        if len(both_up_down_seiz_stat_2) > 0:
            for person in both_up_down_seiz_stat_2:
                chosen_direction = self.rng.choice(
                    ['1', '3'],
                    p=np.divide([prop_transition_2_1, prop_transition_2_3],
                                sum([prop_transition_2_1, prop_transition_2_3]))
                )
                df.loc[person, 'ep_seiz_stat'] = chosen_direction
        # Drop those who have already had their seizure status changed from the indexs changing_2_1 and changing_2_3
        changing_2_1 = changing_2_1.drop(both_up_down_seiz_stat_2)
        changing_2_3 = changing_2_3.drop(both_up_down_seiz_stat_2)
        df.loc[changing_2_1, 'ep_seiz_stat'] = '1'
        df.loc[changing_2_3, 'ep_seiz_stat'] = '3'
        # Determine if those with seizure status 3 decrease in severity and by how much, first get random draws for each
        # transition state
        random_draw_3_1 = self.rng.random_sample(size=len(population_with_seizure_status_3))
        random_draw_3_2 = self.rng.random_sample(size=len(population_with_seizure_status_3))
        # create a dataframe for the transition probabilities
        probability_of_transition_3 = pd.DataFrame({'prob_down_1': [prop_transition_3_2] * len(random_draw_3_2),
                                                    'prob_down_2': [prop_transition_3_1] * len(random_draw_3_1)},
                                                   index=population_with_seizure_status_3)
        # Increase the likelihood of reducing seizure status for those on anti epileptics
        probability_of_transition_3.loc[seizure_status_3_on_anti_epileptics, 'prob_down_1'] *= \
            rr_effectiveness_antiepileptics
        # Decrease the likelihood of reducing seizure status for those on anti epileptics
        probability_of_transition_3.loc[seizure_status_3_on_anti_epileptics, 'prob_down_2'] /= \
            rr_effectiveness_antiepileptics
        # Establish who has a changing seizure status
        changing_3_2 = population_with_seizure_status_3[probability_of_transition_3['prob_down_1'] > random_draw_3_2]
        changing_3_1 = population_with_seizure_status_3[probability_of_transition_3['prob_down_2'] > random_draw_3_1]
        # If someone with seizure status 2 has been selected to both increase and decrease in severity, choose a
        # transition direction based on the likelihood of transitioning states
        both_down_seiz_stat_3 = changing_3_1.intersection(changing_3_2)
        if len(both_down_seiz_stat_3) > 0:
            for person in both_down_seiz_stat_3:
                chosen_direction = self.rng.choice(
                    ['1', '2'],
                    p=np.divide([prop_transition_3_1, prop_transition_3_2],
                                sum([prop_transition_3_1, prop_transition_3_2]))
                )
                df.loc[person, 'ep_seiz_stat'] = chosen_direction
        # Drop those who have already had their seizure status changed from the indexs changing_2_1 and changing_2_3
        changing_3_1 = changing_3_1.drop(both_down_seiz_stat_3)
        changing_3_2 = changing_3_2.drop(both_down_seiz_stat_3)
        df.loc[changing_3_1, 'ep_seiz_stat'] = '1'
        df.loc[changing_3_2, 'ep_seiz_stat'] = '2'

    def stop_antiep(self, indices, probability):
        """stop individuals on antiep with given probability"""
        df = self.sim.population.props
        df.loc[indices, 'ep_antiep'] = probability < self.rng.random_sample(size=len(indices))


class EpilepsyEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually changes individuals' epilepsy status

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """Create a new depr event.

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))
        p = module.parameters

        self.base_3m_prob_epilepsy = p['base_3m_prob_epilepsy']
        self.rr_epilepsy_age_ge20 = p['rr_epilepsy_age_ge20']
        self.prop_inc_epilepsy_seiz_freq = p['prop_inc_epilepsy_seiz_freq']
        self.base_prob_3m_stop_antiepileptic = p['base_prob_3m_stop_antiepileptic']
        self.rr_stop_antiepileptic_seiz_infreq_or_freq = p['rr_stop_antiepileptic_seiz_infreq_or_freq']
        self.base_prob_3m_epi_death = p['base_prob_3m_epi_death']
        self.daly_wt_epilepsy_severe = p['daly_wt_epilepsy_severe']
        self.daly_wt_epilepsy_less_severe = p['daly_wt_epilepsy_less_severe']
        self.daly_wt_epilepsy_seizure_free = p['daly_wt_epilepsy_seizure_free']

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """
        ep = self.module
        df = population.props

        # Declaration of how we will refer to any treatments that are related to this disease.
        # TREATMENT_ID = 'antiepileptic'

        # set ep_epi_death back to False after death
        df.loc[~df.is_alive & df.ep_epi_death, 'ep_epi_death'] = False
        df.loc[df.is_alive, 'ep_disability'] = 0

        # update ep_seiz_stat for people ep_seiz_stat = 0
        # Find who does not have epilepsy
        alive_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0')]
        # Find who does not have epilepsy and is 20 & over
        ge20_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0') & (df.age_years >= 20)]
        # Create a pandas series of the length of people who are alive, this is the basic probability of people
        # developing epilepsy
        eff_prob_epilepsy = pd.Series(self.base_3m_prob_epilepsy, index=alive_seiz_stat_0_idx)
        # Find the indexes of people who are aged 20 and above and increase their risk of developing epilepsy
        eff_prob_epilepsy.loc[ge20_seiz_stat_0_idx] *= self.rr_epilepsy_age_ge20
        # Create a list of random numbers, one per person, to determine who will develop epilepsy
        random_draw_01 = self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx))
        # If a person's number is less than their likelihood of developing epilepsy, then they will now have epilepsy
        epi_now = eff_prob_epilepsy > random_draw_01
        # For those who have developed epilepsy, we need to work out what their seizure status will be, create another
        # list of random numbers to determine their seizure status
        random_draw_02 = self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx))
        # if a person's random number is less than the probability of frequent seizures, then update they will have a
        # seizure status of 2 assigned to them
        seiz_stat_3_idx = alive_seiz_stat_0_idx[epi_now & (self.prop_inc_epilepsy_seiz_freq > random_draw_02)]
        # if a person's random number is greater than the probability of frequent seizures, then update they will have a
        # seizure status of 1 assigned to them
        seiz_stat_2_idx = alive_seiz_stat_0_idx[epi_now & (self.prop_inc_epilepsy_seiz_freq <= random_draw_02)]
        # based on the above predictions, update each person's seizure status
        df.loc[seiz_stat_3_idx, 'ep_seiz_stat'] = '3'
        df.loc[seiz_stat_2_idx, 'ep_seiz_stat'] = '2'
        # Calculate & log the incidence of epilepsy
        n_incident_epilepsy = epi_now.sum()
        n_alive = df.is_alive.sum()

        incidence_epilepsy = (n_incident_epilepsy * 4 * 100000) / n_alive if n_alive > 0 else 0

        logger.info(
            key='inc_epilepsy',
            data={
                'incidence_epilepsy': incidence_epilepsy,
                'n_incident_epilepsy': n_incident_epilepsy,
                'n_alive': n_alive
            }
        )
        # For those who are not on anti epileptics, determine whether their seizure status changes in severity
        ep.transition_seizure_stat()
        # save all individuals that are currently on anti-epileptics (seizure status: 2 & 3 or 1)
        alive_seiz_stat_1_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep]
        alive_seiz_stat_2_or_3_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3'])) & df.ep_antiep]

        # rate of stop ep_antiep if ep_seiz_stat = 1
        ep.stop_antiep(alive_seiz_stat_1_antiep_idx, self.base_prob_3m_stop_antiepileptic)

        # rate of stop ep_antiep if ep_seiz_stat = 2 or 3
        ep.stop_antiep(alive_seiz_stat_2_or_3_antiep_idx,
                       self.base_prob_3m_stop_antiepileptic * self.rr_stop_antiepileptic_seiz_infreq_or_freq)

        # disability

        # note disability weights from gbd do not map fully onto epilepsy states in model - could re-visit
        # this proposed mapping below
        df.loc[df.is_alive & (df.ep_seiz_stat == '1'), 'ep_disability'] = self.daly_wt_epilepsy_seizure_free
        df.loc[df.is_alive & (df.ep_seiz_stat == '2'), 'ep_disability'] = self.daly_wt_epilepsy_less_severe
        df.loc[df.is_alive & (df.ep_seiz_stat == '3'), 'ep_disability'] = self.daly_wt_epilepsy_severe

        # Work out who is having seizures which may lead to death
        alive_seiz_stat_2_or_3_idx = df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3']))]
        # determine if those having seizures will die as a result of epilepsy
        chosen = self.base_prob_3m_epi_death > self.module.rng.random_sample(size=len(alive_seiz_stat_2_or_3_idx))
        # update the ep_epi_death property
        df.loc[alive_seiz_stat_2_or_3_idx, 'ep_epi_death'] = chosen
        # schedule any deaths
        for individual_id in alive_seiz_stat_2_or_3_idx[chosen]:
            self.sim.schedule_event(
                InstantaneousDeath(self.module, individual_id, 'Epilepsy'), self.sim.date
            )

        # -------------------- UPDATING OF SYMPTOM OF seizures OVER TIME --------------------------------

        dfg = df.index[df.is_alive & ((df.ep_seiz_stat == '2') | (df.ep_seiz_stat == '3'))]
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=dfg.to_list(),
            symptom_string='seizures',
            add_or_remove='+',
            disease_module=self.module
        )

        dfh = df.index[df.is_alive & ((df.ep_seiz_stat == '0') | (df.ep_seiz_stat == '1'))]
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=dfh.to_list(),
            symptom_string='seizures',
            add_or_remove='-',
            disease_module=self.module
        )

# todo: seizures while not on on anti-epileptic leads to call HSI_Epilepsy_Start_Anti_Epileptic


class EpilepsyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        status_groups = df.groupby('ep_seiz_stat').sum()

        n_seiz_stat_1_3 = sum(status_groups.iloc[1:].is_alive)
        n_seiz_stat_2_3 = sum(status_groups.iloc[2:].is_alive)

        n_antiep = (df.is_alive & df.ep_antiep).sum()

        n_epi_death = df.ep_epi_death.sum()

        status_groups['prop_seiz_stats'] = status_groups.is_alive / sum(status_groups.is_alive)

        status_groups['prop_seiz_stat_on_anti_ep'] = status_groups['ep_antiep'] / status_groups.is_alive
        status_groups['prop_seiz_stat_on_anti_ep'] = status_groups['prop_seiz_stat_on_anti_ep'].fillna(0)
        epi_death_rate = \
            (n_epi_death * 4 * 1000) / n_seiz_stat_2_3 if n_seiz_stat_2_3 > 0 else 0

        cum_deaths = (~df.is_alive).sum()

        logger.info(key='epilepsy_logging',
                    data={
                        'prop_seiz_stat_0': status_groups['prop_seiz_stats'].iloc[0],
                        'prop_seiz_stat_1': status_groups['prop_seiz_stats'].iloc[1],
                        'prop_seiz_stat_2': status_groups['prop_seiz_stats'].iloc[2],
                        'prop_seiz_stat_3': status_groups['prop_seiz_stats'].iloc[3],
                        'prop_antiepilep_seiz_stat_0': status_groups['prop_seiz_stat_on_anti_ep'].iloc[0],
                        'prop_antiepilep_seiz_stat_1': status_groups['prop_seiz_stat_on_anti_ep'].iloc[1],
                        'prop_antiepilep_seiz_stat_2': status_groups['prop_seiz_stat_on_anti_ep'].iloc[2],
                        'prop_antiepilep_seiz_stat_3': status_groups['prop_seiz_stat_on_anti_ep'].iloc[3],
                        'n_epi_death': n_epi_death,
                        'cum_deaths': cum_deaths,
                        'epi_death_rate': epi_death_rate,
                        'n_seiz_stat_1_3': n_seiz_stat_1_3,
                        'n_seiz_stat_2_3': n_seiz_stat_2_3,
                        'n_antiep': n_antiep
                    })

        individual = df.loc[[5]]

        logger.info(key='individual_check', data=individual, description='following an individual through simulation')


class HSI_Epilepsy_Start_Anti_Epileptic(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Epilepsy_Start_Anti-Epileptics'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'  # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        # Create a shorthand reference to the function used to request consumables
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        # Get item code for first choice treatment, Phenobarbitone
        self.module.item_codes_for_consumables_required['anti_epileptic_first_choice'] = {
            get_item_code("Phenobarbitone  30mg_1000_CMST"): 1,
        }
        # Get item code for second choice treatment Carbamazepine
        self.module.item_codes_for_consumables_required['anti_epileptic_second_choice'] = {
            get_item_code("Carbamazepine 200mg_1000_CMST"): 1,
        }
        # Get item code for third choice treatment Phenytoin sodium
        self.module.item_codes_for_consumables_required['anti_epileptic_third_choice'] = {
            get_item_code("Phenytoin sodium 100mg_1000_CMST"): 1,
        }
        # Check what drugs are available, to_log is set to false in order to not actually request the consumables from
        # the health system before we know what is available
        available_treatments = {
            'anti_epileptic_first_choice': self.get_consumables(
                self.module.item_codes_for_consumables_required['anti_epileptic_first_choice'],
                to_log=False
            ),
            'anti_epileptic_second_choice': self.get_consumables(
                self.module.item_codes_for_consumables_required['anti_epileptic_second_choice'],
                to_log=False
            ),
            'anti_epileptic_third_choice': self.get_consumables(
                self.module.item_codes_for_consumables_required['anti_epileptic_third_choice'],
                to_log=False
            ),
        }
        # Now we know which treatments are available, we will request the best treatment from the health system.
        # Determine the most preferable treatment by finding the index in available_treatments of the first True value:
        best_option_index = next((i for i, j in enumerate(available_treatments.values()) if j), None)
        # If at least one of the medicines is available, request the medicine from the health system
        if best_option_index is not None:
            # Get the name of the most preferable medicine currently available
            best_available_medicine = list(available_treatments.keys())[best_option_index]
            # Request the medicine from the health system, with to_log set to True to actually consume the medicine
            # from the health system
            self.get_consumables(
                self.module.item_codes_for_consumables_required[best_available_medicine],
                to_log=True
            )
            # Update this person's properties to show that they are currently on medication
            df.at[person_id, 'ep_antiep'] = True
            # Log the successful treatment
            logger.debug(key='debug', data='@@@@@@@@@@ STARTING TREATMENT FOR SOMEONE!!!!!!!')
        # Schedule a follow-up for 3 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_Epilepsy_Follow_Up(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )


class HSI_Epilepsy_Follow_Up(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "Epilepsy_Treatment"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # Schedule a follow-up for 3 months:
        hs.schedule_hsi_event(
            hsi_event=HSI_Epilepsy_Follow_Up(
                module=self.module,
                person_id=person_id,
            ),
            topen=self.sim.date + DateOffset(months=3),
            tclose=None,
            priority=0
        )

    def did_not_run(self):
        pass
