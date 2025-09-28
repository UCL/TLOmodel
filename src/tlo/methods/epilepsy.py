from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Epilepsy(Module, GenericFirstAppointmentsMixin):
    def __init__(self, name=None):
        super().__init__(name)
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
        'base_3m_prob_epilepsy': Parameter(
            Types.REAL, 'base probability of epilepsy per 3 month period if age below threshold'
        ),
        'rr_epilepsy_above_threshold_age': Parameter(
            Types.REAL, 'relative rate of epilepsy if age above threshold age'
        ),
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
        'prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt': Parameter(
            Types.REAL, 'probability that someone who has had a seizure is started on anti-epileptics. This is '
                        'calibrated to induce the correct proportion of persons with epilepsy currently receiving '
                        'anti-epileptics.'
        ),
        'max_num_of_failed_attempts_before_defaulting': Parameter(
            Types.INT, 'maximum number of time an HSI can be repeated if the relevant essential consumables are not '
                       'available.'
        ),
        'main_polling_event_frequency_months': Parameter(
            Types.INT, 'frequency in months for the main polling event that checks epilepsy status changes'
        ),
        'age_threshold_epilepsy_transition': Parameter(
            Types.REAL, 'age threshold in years at which probability of developing epilepsy changes'
        ),
        'incidence_calculation_annualization_factor': Parameter(
            Types.REAL, 'factor to annualize incidence calculations (quarters to year conversion)'
        ),
        'incidence_calculation_per_population_factor': Parameter(
            Types.REAL, 'factor for case incidence calculation per population'
        ),
        'death_calculation_per_population_factor': Parameter(
            Types.REAL, 'factor for death incidence calculation per population'
        ),
        'pediatric_age_threshold': Parameter(
            Types.REAL, 'age threshold in years below which patients are considered pediatric'
        ),
        'medicine_follow_up_frequency_months': Parameter(
            Types.REAL, 'frequency in months for follow-up appointments when on anti-epileptic medication'
        ),
        'unavailable_medicine_retry_months': Parameter(
            Types.REAL, 'frequency in months to retry when medicine is unavailable'
        ),
        'unavailable_appt_retry_months': Parameter(
            Types.REAL, 'frequency in months to retry when appt is unavailable'
        ),
        'severe_follow_up_frequency_months': Parameter(
            Types.REAL, 'frequency in months for follow-up appointments for severe epilepsy cases'
        ),
        'standard_follow_up_frequency_months': Parameter(
            Types.REAL, 'frequency in months for follow-up appointments for standard epilepsy cases'
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

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        # Update parameters from the resource dataframe
        dfd = read_csv_files(resourcefilepath / 'epilepsy' / 'ResourceFile_Epilepsy',
                            files='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight - 860-862 are the sequale codes for epilepsy
            p['daly_wt_epilepsy_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=860)
            p['daly_wt_epilepsy_less_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861)
            p['daly_wt_epilepsy_seizure_free'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=862)

        # Register Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom.emergency("seizures"))

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
        p = self.parameters
        epilepsy_poll = EpilepsyEvent(self)
        sim.schedule_event(epilepsy_poll, sim.date + DateOffset(months=int(p['main_polling_event_frequency_months'])))

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

    def report_daly_values(self):
        df = self.sim.population.props  # shortcut to population properties dataframe
        return df.loc[df.is_alive, 'ep_disability']

    def transition_seizure_stat(self):
        """
        This function handles all transitions in epilepsy seizure status, for those on and off anti epileptics. The
        function determines the current seizure status of those with epilepsy and based on their original status,
        and whether they are on anti epileptics determines a new seizure status
        :return:
        """
        # Get the parameters used to determine transitions between seizure states
        p = self.parameters
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

    def get_best_available_medicine(self, hsi_event) -> Union[None, str]:
        """Returns the best available medicine (as string), or None if none are available"""
        # Check what drugs are available. (`to_log` is set to false in order to not actually request the consumables
        # from the health system before we know what is available)
        possible_treatments = {
            'phenobarbitone': hsi_event.get_consumables(
                self.item_codes['phenobarbitone'],
                to_log=False
            ),
            'carbamazepine': hsi_event.get_consumables(
                self.item_codes['carbamazepine'],
                to_log=False
            ),
            'phenytoin': hsi_event.get_consumables(
                self.item_codes['phenytoin'],
                to_log=False
            ),
        }
        # Now we know which treatments are available, we will determine the most preferable treatment by finding the
        # index in available_treatments of the first True value:
        best_option_index = next((i for i, j in enumerate(possible_treatments.values()) if j), None)

        if best_option_index is not None:
            # At least one of the treatment is available: Return the name of the most preferable medicine
            return list(possible_treatments.keys())[best_option_index]
        else:
            # None of the treatment is available: return None
            return None

    def do_at_generic_first_appt_emergency(
        self,
        person_id: int,
        symptoms: List[str],
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        if "seizures" in symptoms:
            # Determine if treatment will start - depends on probability of prescribing, which is calibrated to
            # induce the right proportion of persons with epilepsy receiving treatment.

            prob_start = self.parameters['prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt']

            if self.rng.random_sample() < prob_start:
                event = HSI_Epilepsy_Start_Anti_Epileptic(person_id=person_id, module=self)
                schedule_hsi_event(event, priority=0, topen=self.sim.date)


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
        p = module.parameters
        super().__init__(module, frequency=DateOffset(months=int(p['main_polling_event_frequency_months'])))

        self.base_3m_prob_epilepsy = p['base_3m_prob_epilepsy']
        self.rr_epilepsy_above_threshold_age = p['rr_epilepsy_above_threshold_age']
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
        p = self.module.parameters
        df = population.props

        # set ep_epi_death back to False after death
        df.loc[~df.is_alive & df.ep_epi_death, 'ep_epi_death'] = False
        df.loc[df.is_alive, 'ep_disability'] = 0

        # update ep_seiz_stat for people ep_seiz_stat = 0
        # Find who does not have epilepsy
        alive_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0')]
        # Find who does not have epilepsy and is at/above age threshold
        ge_age_threshold_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0') &
                                                    (df.age_years >= p['age_threshold_epilepsy_transition'])]
        # Create a pandas series of the length of people who are alive, this is the basic probability of people
        # developing epilepsy
        eff_prob_epilepsy = pd.Series(self.base_3m_prob_epilepsy, index=alive_seiz_stat_0_idx)
        # Find the indexes of people who are at/above age threshold and increase their risk of developing epilepsy
        eff_prob_epilepsy.loc[ge_age_threshold_seiz_stat_0_idx] *= self.rr_epilepsy_above_threshold_age
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

        incidence_epilepsy = (n_incident_epilepsy * p['incidence_calculation_annualization_factor'] *
                              p['incidence_calculation_per_population_factor']) / n_alive if n_alive > 0 else 0

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


class EpilepsyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Get some summary statistics and log them"""
        df = population.props
        p = self.module.parameters

        status_groups = df[
            ["ep_seiz_stat", "is_alive", "ep_antiep"]
        ].groupby('ep_seiz_stat').sum()

        n_seiz_stat_1_3 = sum(status_groups.iloc[1:].is_alive)
        n_seiz_stat_2_3 = sum(status_groups.iloc[2:].is_alive)

        n_antiep = int((df.is_alive & df.ep_antiep).sum())

        n_epi_death = int(df.ep_epi_death.sum())

        status_groups['prop_seiz_stats'] = status_groups.is_alive / sum(status_groups.is_alive)

        status_groups['prop_seiz_stat_on_anti_ep'] = status_groups['ep_antiep'] / status_groups.is_alive
        status_groups['prop_seiz_stat_on_anti_ep'] = status_groups['prop_seiz_stat_on_anti_ep'].fillna(0)
        epi_death_rate = \
            (n_epi_death * p['incidence_calculation_annualization_factor'] *
             p['death_calculation_per_population_factor']) / n_seiz_stat_2_3 if n_seiz_stat_2_3 > 0 else 0.0

        cum_deaths = (~df.is_alive).sum()

        # Proportion of those with infrequent or frequent seizures currently on anti-epileptics
        prop_freq_or_infreq_seiz_on_antiep = status_groups[2:].ep_antiep.sum() / status_groups[2:].is_alive.sum() \
            if status_groups[2:].is_alive.sum() > 0 else 0

        logger.info(key='epilepsy_logging',
                    data={
                        'prop_seiz_stat_0': status_groups['prop_seiz_stats'].iloc[0],
                        'prop_seiz_stat_1': status_groups['prop_seiz_stats'].iloc[1],
                        'prop_seiz_stat_2': status_groups['prop_seiz_stats'].iloc[2],
                        'prop_seiz_stat_3': status_groups['prop_seiz_stats'].iloc[3],
                        'prop_freq_or_infreq_seiz_on_antiep': prop_freq_or_infreq_seiz_on_antiep,
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


class HSI_Epilepsy_Start_Anti_Epileptic(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Epilepsy_Treatment_Start'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        self._MAX_NUMBER_OF_FAILED_ATTEMPTS_BEFORE_DEFAULTING = module.parameters[
            "max_num_of_failed_attempts_before_defaulting"
        ]
        self._counter_of_failed_attempts_due_to_unavailable_medicines = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters

        # Add equipment for severe epilepsy cases
        if df.at[person_id, 'ep_seiz_stat'] == '3':  # Frequent seizures
            # Assessment and mobility equipment for severe epilepsy
            self.add_equipment({'Wheelchair'})

            # Pediatric equipment for children with severe epilepsy
            if df.at[person_id, 'age_years'] < p['pediatric_age_threshold']:
                self.add_equipment({
                    'Paediatric Corner sit',
                    'Paediatric CP Chair',
                    'Paediatric mat',
                    'Paediatric rollator',
                    'Paediatric Standing frame'
                })

        # Check what drugs are available
        best_available_medicine = self.module.get_best_available_medicine(self)

        if best_available_medicine is not None:
            # Request the medicine from the health system

            dose = {'phenobarbitone': 9131,  # 100mg per day - 3 months
                    'carbamazepine': 91_311,  # 1000mg per day - 3 months
                    'phenytoin': 27_393}  # 300mg per day - 3 months

            self.get_consumables({self.module.item_codes[best_available_medicine]:
                                  dose[best_available_medicine]})

            # Update this person's properties to show that they are currently on medication
            df.at[person_id, 'ep_antiep'] = True

            # Schedule a follow-up:
            hs.schedule_hsi_event(
                hsi_event=HSI_Epilepsy_Follow_Up(
                    module=self.module,
                    person_id=person_id,
                ),
                topen=self.sim.date + DateOffset(months=int(p['medicine_follow_up_frequency_months'])),
                tclose=None,
                priority=0
            )

        elif (
            self._counter_of_failed_attempts_due_to_unavailable_medicines
            < self._MAX_NUMBER_OF_FAILED_ATTEMPTS_BEFORE_DEFAULTING
        ):
            # If no medicine is available, run this HSI again after retry period
            self._counter_of_failed_attempts_due_to_unavailable_medicines += 1
            self.module.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=self,
                topen=self.sim.date + pd.DateOffset(months=int(p['unavailable_medicine_retry_months'])),
                tclose=None,
                priority=0)


class HSI_Epilepsy_Follow_Up(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self._MAX_NUMBER_OF_FAILED_ATTEMPTS_BEFORE_DEFAULTING = module.parameters[
            "max_num_of_failed_attempts_before_defaulting"
        ]
        self._DEFAULT_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self._REPEATED_APPT_FOOTPRINT = self.make_appt_footprint({'PharmDispensing': 1})

        self.TREATMENT_ID = "Epilepsy_Treatment_Followup"
        self.EXPECTED_APPT_FOOTPRINT = self._DEFAULT_APPT_FOOTPRINT
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self._counter_of_failed_attempts_due_to_unavailable_medicines = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        p = self.module.parameters

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()

        # If the person does not remain on anti-epileptics, do nothing:
        if not df.at[person_id, 'ep_antiep']:
            return hs.get_blank_appt_footprint()

        # Add equipment for severe epilepsy cases at follow-up visits
        if df.at[person_id, 'ep_seiz_stat'] == '3':  # Frequent seizures
            # Equipment for severe epilepsy rehabilitation and support
            self.add_equipment({
                'Adaptive communication switches (Infrared switches)',
                'Speech therapy kit',
                'Bobath bed',
                'Box and Block Test',
                'Built Up (adapted) Utensils',
                'Goniometer',
                'Grasp Dynamometer',
                'Hand function kits',
                'Hot Pack Therapy Units',
                'Interferential therapy machine',
                'Muscle stimulator',
                'TENs Unit',
                'Transcutaneous electrical neuromuscular stimulation',
                'Exercise Mats',
                'Gym mat',
                'Parallel Bars',
                'Pulley System',
                'Suspension Slings',
                'Rehabilitation wall bars',
                'Rollators',
                'Walking Frame',
                'Walking Cane',
                'Overhead pulley',
                'Training stairs',
                'Stress Ball',
                'Wheelchair',
                'Ordinary balls',
                'Medicinal Balls (Sets of 0.5kgs, 1kg, 2kgs, 3kgs, 4kgs, 5kgs)',
                'Exercise ball'
            })

            # Pediatric equipment for children with severe epilepsy
            if df.at[person_id, 'age_years'] < p['pediatric_age_threshold']:
                self.add_equipment({
                    'Paediatric Corner sit',
                    'Paediatric CP Chair',
                    'Paediatric mat',
                    'Paediatric rollator',
                    'Paediatric Standing frame'
                })

        # Request the medicine
        best_available_medicine = self.module.get_best_available_medicine(self)
        if best_available_medicine is not None:

            # Schedule a reoccurrence of this follow-up based on seizure severity
            # Schedule a reoccurrence of this follow-up in 3 months if ep_seiz_stat == '3',
            # else, schedule this reoccurrence of it in 1 year (i.e., if ep_seiz_stat == '2'
            if df.at[person_id, 'ep_seiz_stat'] == '3':
                fu_mnths = p['severe_follow_up_frequency_months']
            else:
                fu_mnths = p['standard_follow_up_frequency_months']

            # The medicine is available, so request it
            dose = {'phenobarbitone_3_mnths': 9131, 'phenobarbitone_12_mnths': 36_525,  # 100mg per day - 3/12 months
                    'carbamazepine_3_mnths': 91_311, 'carbamazepine_12_mnths': 365_250,  # 1000mg per day - 3/12 months
                    'phenytoin_3_mnths': 27_393,  'phenytoin_12_mnths': 109_575}  # 300mg per day - 3/12 months

            self.get_consumables({self.module.item_codes[best_available_medicine]:
                                  dose[f'{best_available_medicine}_{int(fu_mnths)}_mnths']})

            # Reset counter of "failed attempts" and put the appointment for the next occurrence to the usual
            self._counter_of_failed_attempts_due_to_unavailable_medicines = 0
            self.EXPECTED_APPT_FOOTPRINT = self._DEFAULT_APPT_FOOTPRINT

            # Schedule follow-up
            hs.schedule_hsi_event(
                hsi_event=self,
                topen=self.sim.date + DateOffset(months=int(fu_mnths)),
                tclose=None,
                priority=0
            )
        elif (
            self._counter_of_failed_attempts_due_to_unavailable_medicines
            < self._MAX_NUMBER_OF_FAILED_ATTEMPTS_BEFORE_DEFAULTING
        ):
            # Nothing is available currently: schedule a recurrence of this appointment in one month, with a modified
            # footprint. (This will repeat a certain number of time before the person defaults to being off
            # anti-epileptics: see next clause.)
            self._counter_of_failed_attempts_due_to_unavailable_medicines += 1
            self.EXPECTED_APPT_FOOTPRINT = self._REPEATED_APPT_FOOTPRINT

            hs.schedule_hsi_event(
                hsi_event=self,
                topen=self.sim.date + DateOffset(months=int(p['unavailable_appt_retry_months'])),
                tclose=None,
                priority=0
            )
        else:
            # No medicine is available and the maximum number of repeats has been reached: The person will default
            # to being off the anti-epileptics and no further follow-ups are scheduled.
            df.at[person_id, 'ep_antiep'] = False
