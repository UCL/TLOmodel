
import logging
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
import numpy as np
import pandas as pd
import random
from pathlib import Path

# todo: code specific clinic visits

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#l ogger.setLevel(logging.CRITICAL)

class Epilepsy(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Module parameters
    PARAMETERS = {
        'init_epil_seiz_status': Parameter(
            Types.LIST,
            'Proportions in each seizure status category at baseline'),
        'init_prop_antiepileptic_seiz_stat_1': Parameter(
            Types.REAL,
            'initial proportions on antiepileptic by if seizure status = 1'),
        'init_prop_antiepileptic_seiz_stat_2': Parameter(
            Types.REAL,
            'initial proportions on antiepileptic by if seizure status = 2'),
        'init_prop_antiepileptic_seiz_stat_3': Parameter(
            Types.REAL,
            'initial proportions on antiepileptic by if seizure status = 3'),
        'base_3m_prob_epilepsy': Parameter(
            Types.REAL,
            'base probability of epilepsy per 3 month period if age < 20'),
        'rr_epilepsy_age_ge20': Parameter(
            Types.REAL,
            'relative rate of epilepsy if age over 20'),
        'prop_inc_epilepsy_seiz_freq': Parameter(
            Types.REAL,
            'proportion of incident epilepsy cases with frequent seizures'),
        'base_prob_3m_seiz_stat_freq_infreq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status frequent if current infrequent'),
        'rr_effectiveness_antiepileptics': Parameter(
            Types.REAL,
            'relative rate of seizure status frequent if current infrequent if on antiepileptic'),
        'base_prob_3m_seiz_stat_infreq_freq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status infrequent if current frequent'),
        'base_prob_3m_seiz_stat_infreq_none': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status infrequent if current nonenow'),
        'base_prob_3m_seiz_stat_none_freq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status nonenow if current frequent'),
        'base_prob_3m_seiz_stat_none_infreq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status nonenow if current infrequent'),
        'base_prob_3m_antiepileptic': Parameter(
            Types.REAL,
            'base probability per 3 months of starting antiepileptic, if frequent seizures'),
        'rr_antiepileptic_seiz_infreq': Parameter(
            Types.REAL,
            'relative rate of starting antiepileptic if infrequent seizures'),
        'base_prob_3m_stop_antiepileptic': Parameter(
            Types.REAL,
            'base probability per 3 months of stopping antiepileptic, if nonenow seizures'),
        'rr_stop_antiepileptic_seiz_infreq_or_freq': Parameter(
            Types.REAL,
            'relative rate of stopping antiepileptic if infrequent or frequent seizures'),
        'base_prob_3m_epi_death': Parameter(
            Types.REAL,
            'base probability per 3 months of epilepsy death'),
     }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'ep_seiz_stat': Property(Types.CATEGORICAL, '(0 = never epilepsy, 1 = previous seizures none now, '
                                                    '2 = infrequent seizures, 3 = frequent seizures)',
                                 categories=['0', '1', '2', '3']),
        'ep_antiep': Property(Types.BOOL, 'on antiepileptic'),
        'ep_epi_death': Property(Types.BOOL, 'epilepsy death this 3 month period'),
        'ep_unified_symptom_code': Property(Types.CATEGORICAL, '',
                                            categories=['0', '1', '2', '3']),
        'ep_disability': Property(
            Types.REAL, 'disability weight for current 3 month period')
    }



    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """



        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Epilepsy.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)

        self.parameters['init_epil_seiz_status'] = \
            [dfd.loc['init_epil_seiz_status', 'value'], dfd.loc['init_epil_seiz_status', 'value2'],
             dfd.loc['init_epil_seiz_status', 'value3'], dfd.loc['init_epil_seiz_status', 'value4']]
        self.parameters['init_prop_antiepileptic_seiz_stat_1'] = \
            dfd.loc['init_prop_antiepileptic_seiz_stat_1', 'value']
        self.parameters['init_prop_antiepileptic_seiz_stat_2'] = \
            dfd.loc['init_prop_antiepileptic_seiz_stat_2', 'value']
        self.parameters['init_prop_antiepileptic_seiz_stat_3'] = \
            dfd.loc['init_prop_antiepileptic_seiz_stat_3', 'value']
        self.parameters['base_3m_prob_epilepsy'] = dfd.loc['base_3m_prob_epilepsy', 'value']
        self.parameters['rr_epilepsy_age_ge20'] = dfd.loc['rr_epilepsy_age_ge20', 'value']
        self.parameters['prop_inc_epilepsy_seiz_freq'] = dfd.loc['prop_inc_epilepsy_seiz_freq', 'value']
        self.parameters['rr_effectiveness_antiepileptics'] = dfd.loc['rr_effectiveness_antiepileptics', 'value']
        self.parameters['base_prob_3m_seiz_stat_freq_infreq'] = dfd.loc['base_prob_3m_seiz_stat_freq_infreq', 'value']
        self.parameters['base_prob_3m_seiz_stat_infreq_freq'] = dfd.loc['base_prob_3m_seiz_stat_infreq_freq', 'value']
        self.parameters['base_prob_3m_seiz_stat_none_freq'] = dfd.loc['base_prob_3m_seiz_stat_none_freq', 'value']
        self.parameters['base_prob_3m_seiz_stat_none_infreq'] = dfd.loc['base_prob_3m_seiz_stat_none_infreq', 'value']
        self.parameters['base_prob_3m_seiz_stat_infreq_none'] = dfd.loc['base_prob_3m_seiz_stat_infreq_none', 'value']
        self.parameters['base_prob_3m_antiepileptic'] = dfd.loc['base_prob_3m_antiepileptic', 'value']
        self.parameters['rr_antiepileptic_seiz_infreq'] = dfd.loc['rr_antiepileptic_seiz_infreq', 'value']
        self.parameters['base_prob_3m_stop_antiepileptic'] = dfd.loc['base_prob_3m_stop_antiepileptic', 'value']
        self.parameters['rr_stop_antiepileptic_seiz_infreq_or_freq'] = dfd.loc['rr_stop_antiepileptic_seiz_infreq_or_freq', 'value']
        self.parameters['base_prob_3m_epi_death'] = dfd.loc['base_prob_3m_epi_death', 'value']

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        """

        df = population.props  # a shortcut to the data-frame storing data for individuals

        df['ep_seiz_stat'] = '0'
        df['ep_antiep'] = False
        df['ep_epi_death'] = False
        df['ep_disability'] = 0

        # allocate initial ep_seiz_stat
        alive_idx = df.index[df.is_alive]
        df.loc[alive_idx, 'ep_seiz_stat'] = self.rng.choice(['0', '1', '2', '3'], size=len(alive_idx),
                                                                   p=self.parameters['init_epil_seiz_status'])

        # allocate initial on antiepileptic seiz status 1
        seiz_stat_1_idx = df.index[df.is_alive & (df.ep_seiz_stat == '1')]
        random_draw = self.rng.random_sample(size=len(seiz_stat_1_idx))
        df.loc[seiz_stat_1_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_1'])

        # allocate initial on antiepileptic seiz status 2
        seiz_stat_2_idx = df.index[df.is_alive & (df.ep_seiz_stat == '2')]
        random_draw = self.rng.random_sample(size=len(seiz_stat_2_idx))
        df.loc[seiz_stat_2_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_2'])

        # allocate initial on antiepileptic seiz status 3
        seiz_stat_3_idx = df.index[df.is_alive & (df.ep_seiz_stat == '3')]
        random_draw = self.rng.random_sample(size=len(seiz_stat_3_idx))
        df.loc[seiz_stat_3_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_3'])

        # disability

        df.loc[seiz_stat_1_idx, 'ep_disability'] = 0.049
        df.loc[seiz_stat_2_idx, 'ep_disability'] = 0.263
        df.loc[seiz_stat_3_idx, 'ep_disability'] = 0.552

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

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

        # todo: amend this below when identifid data
        # Define the footprint for the intervention on the common resources
        footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
            'Name': Epilepsy.TREATMENT_ID,
            'Nurse_Time': 15,
            'Doctor_Time': 15,
            'Electricity': False,
            'Water': False})


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


    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale

#       logger.debug("This is Epilepsy being asked to report unified symptomology")

        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe

#       df.loc[df.is_alive, 'ep_unified_symptom_code'] \
#           = df.loc[df.is_alive, 'ep_seiz_stat'].map({ 0 : 1,  1 : 1,  2 : 1,  3 : 1})

#       return df.loc[df.is_alive, 'ep_unified_symptom_code']

        return pd.Series('1', index=df.index[df.is_alive])


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Epilepsy, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Epilepsy reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        # todo update this below
        # todo get daly weight direct from document
        # todo add more comments

        # df.loc[seiz_stat_1_idx, 'ep_disability'] = 0.049
        # df.loc[seiz_stat_2_idx, 'ep_disability'] = 0.263
        # df.loc[seiz_stat_3_idx, 'ep_disability'] = 0.552

        disability_weights = df.loc[df['is_alive'], 'ep_disability']

        return disability_weights


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

        self.base_3m_prob_epilepsy = module.parameters['base_3m_prob_epilepsy']
        self.rr_epilepsy_age_ge20 = module.parameters['rr_epilepsy_age_ge20']
        self.prop_inc_epilepsy_seiz_freq = module.parameters['prop_inc_epilepsy_seiz_freq']
        self.base_prob_3m_seiz_stat_freq_infreq = module.parameters['base_prob_3m_seiz_stat_freq_infreq']
        self.rr_effectiveness_antiepileptics = module.parameters['rr_effectiveness_antiepileptics']
        self.base_prob_3m_seiz_stat_infreq_freq = module.parameters['base_prob_3m_seiz_stat_infreq_freq']
        self.base_prob_3m_seiz_stat_none_freq = module.parameters['base_prob_3m_seiz_stat_none_freq']
        self.base_prob_3m_seiz_stat_none_infreq = module.parameters['base_prob_3m_seiz_stat_none_infreq']
        self.base_prob_3m_seiz_stat_infreq_none = module.parameters['base_prob_3m_seiz_stat_infreq_none']
        self.base_prob_3m_antiepileptic = module.parameters['base_prob_3m_antiepileptic']
        self.rr_antiepileptic_seiz_infreq = module.parameters['rr_antiepileptic_seiz_infreq']
        self.base_prob_3m_stop_antiepileptic = module.parameters['base_prob_3m_stop_antiepileptic']
        self.rr_stop_antiepileptic_seiz_infreq_or_freq = module.parameters['rr_stop_antiepileptic_seiz_infreq_or_freq']
        self.base_prob_3m_epi_death = module.parameters['base_prob_3m_epi_death']


    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """

        df = population.props

        # set ep_epi_death back to False after death
        df.loc[~df.is_alive, 'ep_epi_death'] = False
        df['ep_disability'] = 0

        # update ep_seiz_stat for people ep_seiz_stat = 0

        alive_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0')]
        ge20_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == '0') & (df.age_years >= 20)]

        eff_prob_epilepsy = pd.Series(self.base_3m_prob_epilepsy,
                                      index=df.index[df.is_alive & (df.ep_seiz_stat == '0')])
        eff_prob_epilepsy.loc[ge20_seiz_stat_0_idx] *= self.rr_epilepsy_age_ge20

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '0')])
        random_draw_02 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '0')])

        series_prop_inc_epilepsy_seiz_freq = pd.Series(self.prop_inc_epilepsy_seiz_freq,
                                      index=df.index[df.is_alive & (df.ep_seiz_stat == '0')])

        df_incident_epilepsy = pd.concat([eff_prob_epilepsy, random_draw_01, random_draw_02,
                                          series_prop_inc_epilepsy_seiz_freq], axis=1)
        df_incident_epilepsy.columns = ['eff_prob_epilepsy', 'random_draw_01', 'random_draw_02',
                                        'prop_inc_epilepsy_seiz_freq']

        df_incident_epilepsy['x_ep_seiz_stat'] = '0'
        df_incident_epilepsy['incident_epi_this_period'] = False
        df_incident_epilepsy.loc[(df_incident_epilepsy.eff_prob_epilepsy > random_draw_01),
                                 'incident_epi_this_period'] = True
        df_incident_epilepsy.loc[(df_incident_epilepsy.eff_prob_epilepsy > random_draw_01) &
                                 (df_incident_epilepsy.prop_inc_epilepsy_seiz_freq > random_draw_02),
                                 'x_ep_seiz_stat'] = '3'
        df_incident_epilepsy.loc[(df_incident_epilepsy.eff_prob_epilepsy > random_draw_01) &
                                 (df_incident_epilepsy.prop_inc_epilepsy_seiz_freq < random_draw_02),
                                 'x_ep_seiz_stat'] = '2'

        df.loc[alive_seiz_stat_0_idx, 'ep_seiz_stat'] = df_incident_epilepsy['x_ep_seiz_stat']

        n_incident_epilepsy = df_incident_epilepsy.incident_epi_this_period.sum()
        n_alive = df.is_alive.sum()

        incidence_epilepsy = (n_incident_epilepsy * 4 * 100000) / n_alive

        logger.info('%s|incidence_epilepsy|%s|n_incident_epilepsy|%s|n_alive|%s',
                    self.sim.date, incidence_epilepsy, n_incident_epilepsy, n_alive
                    )

        # transition from ep_seiz_stat 1 to 2

        alive_seiz_stat_1_idx = df.index[df.is_alive & (df.ep_seiz_stat == '1')]

        eff_prob_seiz_stat_2 = pd.Series(self.base_prob_3m_seiz_stat_infreq_none,
                                      index=df.index[df.is_alive & (df.ep_seiz_stat == '1')])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_1_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '1')])

        dfx = pd.concat([eff_prob_seiz_stat_2, random_draw_01],
                        axis=1)
        dfx.columns = ['eff_prob_seiz_stat_2', 'random_draw_01']

        dfx['x_ep_seiz_stat'] = '1'
        dfx.loc[(dfx.eff_prob_seiz_stat_2 > random_draw_01), 'x_ep_seiz_stat'] = '2'

        df.loc[alive_seiz_stat_1_idx, 'ep_seiz_stat'] = dfx['x_ep_seiz_stat']

        # transition from ep_seiz_stat 2 to 1

        alive_seiz_stat_2_idx = df.index[df.is_alive & (df.ep_seiz_stat == '2')]

        eff_prob_seiz_stat_1 = pd.Series(self.base_prob_3m_seiz_stat_infreq_none,
                                         index=df.index[df.is_alive & (df.ep_seiz_stat == '2')])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_2_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '2')])

        dfx = pd.concat([eff_prob_seiz_stat_1, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_seiz_stat_1', 'random_draw_01']

        dfx['x_ep_seiz_stat'] = '2'
        dfx.loc[(dfx.eff_prob_seiz_stat_1 > random_draw_01), 'x_ep_seiz_stat'] = '1'

        df.loc[alive_seiz_stat_2_idx, 'ep_seiz_stat'] = dfx['x_ep_seiz_stat']

        # transition from ep_seiz_stat 2 to 3

        alive_seiz_stat_2_idx = df.index[df.is_alive & (df.ep_seiz_stat == '2')]

        eff_prob_seiz_stat_3 = pd.Series(self.base_prob_3m_seiz_stat_freq_infreq,
                                         index=df.index[df.is_alive & (df.ep_seiz_stat == '2')])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_2_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '2')])

        dfx = pd.concat([eff_prob_seiz_stat_3, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_seiz_stat_3', 'random_draw_01']

        dfx['x_ep_seiz_stat'] = '2'
        dfx.loc[(dfx.eff_prob_seiz_stat_3 > random_draw_01), 'x_ep_seiz_stat'] = '3'

        df.loc[alive_seiz_stat_2_idx, 'ep_seiz_stat'] = dfx['x_ep_seiz_stat']

        # transition from ep_seiz_stat 3 to 1

        alive_seiz_stat_3_idx = df.index[df.is_alive & (df.ep_seiz_stat == '3')]

        eff_prob_seiz_stat_1 = pd.Series(self.base_prob_3m_seiz_stat_none_freq,
                                         index=df.index[df.is_alive & (df.ep_seiz_stat == '3')])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_3_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '3')])

        dfx = pd.concat([eff_prob_seiz_stat_1, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_seiz_stat_1', 'random_draw_01']

        dfx['x_ep_seiz_stat'] = '3'
        dfx.loc[(dfx.eff_prob_seiz_stat_1 > random_draw_01), 'x_ep_seiz_stat'] = '1'

        df.loc[alive_seiz_stat_3_idx, 'ep_seiz_stat'] = dfx['x_ep_seiz_stat']

        # transition from ep_seiz_stat 3 to 2

        alive_seiz_stat_3_idx = df.index[df.is_alive & (df.ep_seiz_stat == '3')]

        eff_prob_seiz_stat_2 = pd.Series(self.base_prob_3m_seiz_stat_infreq_freq,
                                         index=df.index[df.is_alive & (df.ep_seiz_stat == '3')])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_3_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '3')])

        dfx = pd.concat([eff_prob_seiz_stat_2, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_seiz_stat_2', 'random_draw_01']

        # x_ep_antiep is whether requests health system for treatment to start
        dfx['x_ep_seiz_stat'] = '3'
        dfx.loc[(dfx.eff_prob_seiz_stat_2 > random_draw_01), 'x_ep_seiz_stat'] = '2'

        df.loc[alive_seiz_stat_3_idx, 'ep_seiz_stat'] = dfx['x_ep_seiz_stat']

        # update ep_antiep if ep_seiz_stat = 2

        alive_seiz_stat_2_not_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat == '2') & ~df.ep_antiep]

        eff_prob_antiep = pd.Series(self.base_prob_3m_antiepileptic,
                                    index=df.index[df.is_alive & (df.ep_seiz_stat == '2') & ~df.ep_antiep])
        eff_prob_antiep *= self.rr_antiepileptic_seiz_infreq

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_2_not_antiep_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '2') & ~df.ep_antiep])

        dfx = pd.concat([eff_prob_antiep, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_antiep', 'random_draw_01']

        # x_ep_antiep is whether requests health system for treatment to start
        dfx['x_ep_antiep'] = False
        dfx.loc[(dfx.eff_prob_antiep > random_draw_01), 'x_ep_antiep'] = True

        # update ep_antiep if ep_seiz_stat = 3

        alive_seiz_stat_3_not_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat == '3') & ~df.ep_antiep]

        eff_prob_antiep = pd.Series(self.base_prob_3m_antiepileptic,
                                    index=df.index[df.is_alive & (df.ep_seiz_stat == '3') & ~df.ep_antiep])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_3_not_antiep_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '3') & ~df.ep_antiep])

        dfx = pd.concat([eff_prob_antiep, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_antiep', 'random_draw_01']

        # x_ep_antiep is whether requests health system for treatment to start
        dfx['x_ep_antiep'] = False
        dfx.loc[(dfx.eff_prob_antiep > random_draw_01), 'x_ep_antiep'] = True

        # start on treatment if health system has capacity

        start_antiep_this_period_idx = dfx.index[dfx.x_ep_antiep]

        # create a df with one row per person needing to start treatment - this is only way I have
        # managed to get query access to service code to work properly here (should be possible to remove
        # relevant rows from dfx rather than create dfxx
        e1 = pd.Series(True, index=dfx.index[dfx.x_ep_antiep])

        for person_id_to_start_treatment in e1.index:
            event = HSI_Epilepsy_Start_Anti_Epilpetic(self.module,
                                                      person_id=person_id_to_start_treatment)
            target_date = self.sim.date + DateOffset(days=int(self.module.rng.rand()*30))
            self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=2,
                                                                topen=target_date,
                                                                tclose=None)

        # note that this line seems to apply to all in dfxx so had to restrict it to those needing to be treated

#       df.loc[start_antiep_this_period_idx, 'ep_on_antiep'] = dfxx['gets_trt']

        # rate of stop ep_antiep if ep_seiz_stat = 2 or 3

        alive_seiz_stat_2_or_3_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3'])) & df.ep_antiep]

        eff_prob_stop_antiep = pd.Series(self.base_prob_3m_antiepileptic,
                                    index=df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3'])) & df.ep_antiep])
        eff_prob_stop_antiep *= self.rr_stop_antiepileptic_seiz_infreq_or_freq

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_2_or_3_antiep_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3'])) & df.ep_antiep])

        dfx = pd.concat([eff_prob_stop_antiep, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_stop_antiep', 'random_draw_01']

        dfx['x_ep_antiep'] = True
        dfx.loc[(dfx.eff_prob_stop_antiep > random_draw_01), 'x_ep_antiep'] = False

        df.loc[alive_seiz_stat_2_or_3_antiep_idx, 'ep_antiep'] = dfx['x_ep_antiep']

        # rate of stop ep_antiep if ep_seiz_stat = 1

        alive_seiz_stat_1_antiep_idx = df.index[df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep]

        eff_prob_stop_antiep = pd.Series(self.base_prob_3m_antiepileptic,
                                         index=df.index[df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep])
        eff_prob_stop_antiep *= self.rr_stop_antiepileptic_seiz_infreq_or_freq

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_1_antiep_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep])

        dfx = pd.concat([eff_prob_stop_antiep, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_stop_antiep', 'random_draw_01']

        dfx['x_ep_antiep'] = True
        dfx.loc[(dfx.eff_prob_stop_antiep > random_draw_01), 'x_ep_antiep'] = False

        df.loc[alive_seiz_stat_1_antiep_idx, 'ep_antiep'] = dfx['x_ep_antiep']

        # disability

        seiz_stat_1_idx = df.index[df.is_alive & (df.ep_seiz_stat == '1')]
        seiz_stat_2_idx = df.index[df.is_alive & (df.ep_seiz_stat == '2')]
        seiz_stat_3_idx = df.index[df.is_alive & (df.ep_seiz_stat == '3')]

        df.loc[seiz_stat_1_idx, 'ep_disability'] = 0.049
        df.loc[seiz_stat_2_idx, 'ep_disability'] = 0.263
        df.loc[seiz_stat_3_idx, 'ep_disability'] = 0.552

        # update ep_epi_death

        alive_seiz_stat_2_or_3_idx = df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3']))]

        eff_prob_epi_death = pd.Series(self.base_prob_3m_epi_death,
                                       index=df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3']))])

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_2_or_3_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat.isin(['2', '3']))])

        dfx = pd.concat([eff_prob_epi_death, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_epi_death', 'random_draw_01']

        dfx['x_epi_death'] = False
        dfx.loc[(dfx.eff_prob_epi_death > random_draw_01), 'x_epi_death'] = True

        df.loc[alive_seiz_stat_2_or_3_idx, 'ep_epi_death'] = dfx['x_epi_death']

        death_this_period = df.index[df.ep_epi_death]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'Epilepsy'),
                                    self.sim.date)






class HSI_Epilepsy_Start_Anti_Epilpetic(Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = 'Epilepsy_Start_Anti-Epilpetics'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]     # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.info('This is EPIlepsy %s, a first appointment for person %d',
                     person_id)
        print('@@@@@@@@@@ STARTING TREATMENT FOR SOMEONE!!!!!!!')






class EpilepsyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):

        # get some summary statistics
        df = population.props

        n_alive = df.is_alive.sum()
        n_antiepilep_seiz_stat_0 = (df.is_alive & (df.ep_seiz_stat == '0') & df.ep_antiep).sum()
        n_antiepilep_seiz_stat_1 = (df.is_alive & (df.ep_seiz_stat == '1') & df.ep_antiep).sum()
        n_antiepilep_seiz_stat_2 = (df.is_alive & (df.ep_seiz_stat == '2') & df.ep_antiep).sum()
        n_antiepilep_seiz_stat_3 = (df.is_alive & (df.ep_seiz_stat == '3') & df.ep_antiep).sum()

        n_seiz_stat_0 = (df.is_alive & (df.ep_seiz_stat == '0')).sum()
        n_seiz_stat_1 = (df.is_alive & (df.ep_seiz_stat == '1')).sum()
        n_seiz_stat_2 = (df.is_alive & (df.ep_seiz_stat == '2')).sum()
        n_seiz_stat_3 = (df.is_alive & (df.ep_seiz_stat == '3')).sum()

        n_seiz_stat_1_3 = n_seiz_stat_1 + n_seiz_stat_2 + n_seiz_stat_3
        n_seiz_stat_2_3 = n_seiz_stat_2 + n_seiz_stat_3

        n_antiep = (df.is_alive & df.ep_antiep).sum()

        n_epi_death = df.ep_epi_death.sum()

        prop_seiz_stat_0 = n_seiz_stat_0 / n_alive
        prop_seiz_stat_1 = n_seiz_stat_1 / n_alive
        prop_seiz_stat_2 = n_seiz_stat_2 / n_alive
        prop_seiz_stat_3 = n_seiz_stat_3 / n_alive

        prop_antiepilep_seiz_stat_0 = n_antiepilep_seiz_stat_0 / n_seiz_stat_0
        prop_antiepilep_seiz_stat_1 = n_antiepilep_seiz_stat_1 / n_seiz_stat_1
        prop_antiepilep_seiz_stat_2 = n_antiepilep_seiz_stat_2 / n_seiz_stat_2
        prop_antiepilep_seiz_stat_3 = n_antiepilep_seiz_stat_3 / n_seiz_stat_3

        epi_death_rate = (n_epi_death * 4 * 1000) / (n_seiz_stat_2 +n_seiz_stat_3)

        cum_deaths = (~df.is_alive).sum()


        # TODO: Can you define what these ouputs are in the comments [in the future, I think we need a systematic way of storing this beyond comments.]

        dict_for_output = {
            'prop_seiz_stat_0': prop_seiz_stat_0,
            'prop_seiz_stat_1': prop_seiz_stat_1,
            'prop_seiz_stat_2': prop_seiz_stat_2,
            'prop_seiz_stat_3': prop_seiz_stat_3,
            'prop_antiepilep_seiz_stat_0': prop_antiepilep_seiz_stat_0,
            'prop_antiepilep_seiz_stat_1': prop_antiepilep_seiz_stat_1,
            'prop_antiepilep_seiz_stat_2': prop_antiepilep_seiz_stat_2,
            'prop_antiepilep_seiz_stat_3': prop_antiepilep_seiz_stat_3,
            'n_epi_death': n_epi_death,
            'cum_deaths': cum_deaths,
            'epi_death_rate': epi_death_rate,
            'n_seiz_stat_1_3': n_seiz_stat_1_3,
            'n_seiz_stat_2_3': n_seiz_stat_2_3,
            'n_antiep': n_antiep,
            'n_alive': n_alive
        }

        logger.info('%s|summary_stats_per_3m|%s',
                    self.sim.date,
                    dict_for_output)



#
#         #
#
#         #       logger.info('%s,%s,', self.sim.date, n_epi_death)
#         """
#         logger.info('%s|prop_seiz_stat_0|%s|prop_seiz_stat_1|%s|prop_seiz_stat_2|%s|'
#                     'prop_seiz_stat_3|%s|prop_antiepilep_seiz_stat_0|%s|prop_antiepilep_seiz_stat_1|%s|'
#                     'prop_antiepilep_seiz_stat_2|%s|prop_antiepilep_seiz_stat_3|%s|n_epi_death|%s|'
#                     'cum_deaths|%s|epi_death_rate |%s|n_seiz_stat_1_3|%s|n_seiz_stat_2_3|%s|n_antiep|%s'
#                     '|n_alive|%s',
#                     self.sim.date, prop_seiz_stat_0, prop_seiz_stat_1, prop_seiz_stat_2, prop_seiz_stat_3,
#                     prop_antiepilep_seiz_stat_0, prop_antiepilep_seiz_stat_1, prop_antiepilep_seiz_stat_2,
#                     prop_antiepilep_seiz_stat_3, n_epi_death, cum_deaths, epi_death_rate, n_seiz_stat_1_3,
#                     n_seiz_stat_2_3, n_antiep, n_alive
#                     )
#         """
# #       logger.info('%s|person_one|%s',
# #                    self.sim.date,
# #                    df.loc[0].to_dict())
#


