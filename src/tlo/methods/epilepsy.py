
import logging
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

class Epilepsy(Module):

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
        'base_prob_3m_epi_death_seiz_infreq': Parameter(
            Types.REAL,
            'base probability per 3 months of epilepsy death'),
     }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'ep_seiz_stat': Property(Types.CATEGORICAL, 'seizure status', categories=[0, 1, 2, 3]),
        'ep_antiep': Property(Types.BOOL, 'on antiepileptic'),
        'ep_epi_death': Property(Types.BOOL, 'epilepsy death this 3 month period'),
    }

    def __init__(self):
        super().__init__()

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        self.parameters['init_epil_seiz_status'] = [0.98, 0.002, 0.008, 0.01]
        self.parameters['init_prop_antiepileptic_seiz_stat_1'] = 0.1
        self.parameters['init_prop_antiepileptic_seiz_stat_2'] = 0.2
        self.parameters['init_prop_antiepileptic_seiz_stat_3'] = 0.3
        self.parameters['base_3m_prob_epilepsy'] = 0.001
        self.parameters['rr_epilepsy_age_ge20'] = 0.5
        self.parameters['prop_inc_epilepsy_seiz_freq'] = 0.1
        self.parameters['base_prob_3m_seiz_stat_freq_infreq'] = 0.001
        self.parameters['rr_effectiveness_antiepileptics'] = 5
        self.parameters['base_prob_3m_seiz_stat_infreq_freq'] = 0.01
        self.parameters['base_prob_3m_seiz_stat_none_freq'] = 0.05
        self.parameters['base_prob_3m_seiz_stat_none_infreq'] = 0.10
        self.parameters['base_prob_3m_antiepileptic'] = 0.05
        self.parameters['rr_antiepileptic_seiz_infreq'] = 0.3
        self.parameters['base_prob_3m_stop_antiepileptic'] = 0.05
        self.parameters['rr_stop_antiepileptic_seiz_infreq_or_freq'] = 0.1
        self.parameters['base_prob_3m_epi_death_seiz_infreq'] = 0.1

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        """

        df = population.props  # a shortcut to the data-frame storing data for individuals

        df['ep_seiz_stat'] = 0
        df['ep_antiep'] = False
        df['ep_epi_death'] = False

        # allocate initial ep_seiz_stat
        alive_idx = df.index[df.is_alive]
        df.loc[alive_idx, 'ep_seiz_stat'] = self.rng.choice([0, 1, 2, 3], size=len(alive_idx),
                                                                   p=self.parameters['init_epil_seiz_status'])

        # allocate initial on antiepileptic seiz status 1
        seiz_stat_1_idx = df.index[df.is_alive & (df.ep_seiz_stat == 1)]
        random_draw = self.rng.random_sample(size=len(seiz_stat_1_idx))
        df.loc[seiz_stat_1_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_1'])

        # allocate initial on antiepileptic seiz status 2
        seiz_stat_2_idx = df.index[df.is_alive & (df.ep_seiz_stat == 2)]
        random_draw = self.rng.random_sample(size=len(seiz_stat_2_idx))
        df.loc[seiz_stat_2_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_2'])

        # allocate initial on antiepileptic seiz status 3
        seiz_stat_3_idx = df.index[df.is_alive & (df.ep_seiz_stat == 3)]
        random_draw = self.rng.random_sample(size=len(seiz_stat_3_idx))
        df.loc[seiz_stat_3_idx, 'ep_antiep'] = (random_draw < self.parameters['init_prop_antiepileptic_seiz_stat_3'])

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our three-monthly event to poll the population for depr starting
        or stopping.
        """
        epilepsy_poll = EpilepsyEvent(self)
        sim.schedule_event(epilepsy_poll, sim.date + DateOffset(months=3))

        event = EpilepsyLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ep_seiz_stat'] = 0
        df.at[child_id, 'ep_antiep'] = False
        df.at[child_id, 'ep_epi_death'] = False

class EpilepsyEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually changes individuals' depr status.

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
        self.base_prob_3m_antiepileptic = module.parameters['base_prob_3m_antiepileptic']
        self.rr_antiepileptic_seiz_infreq = module.parameters['rr_antiepileptic_seiz_infreq']
        self.base_prob_3m_stop_antiepileptic = module.parameters['base_prob_3m_stop_antiepileptic']
        self.rr_stop_antiepileptic_seiz_infreq_or_freq = module.parameters['rr_stop_antiepileptic_seiz_infreq_or_freq']
        self.base_prob_3m_epi_death_seiz_infreq = module.parameters['base_prob_3m_epi_death_seiz_infreq']

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """

        df = population.props

        # update ep_seiz_stat for people ep_seiz_stat = 0

        alive_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == 0)]
        ge20_seiz_stat_0_idx = df.index[df.is_alive & (df.ep_seiz_stat == 0) & (df.age_years >= 20)]

        eff_prob_epilepsy = pd.Series(self.base_3m_prob_epilepsy,
                                      index=df.index[df.is_alive & (df.ep_seiz_stat == 0)])
        eff_prob_epilepsy.loc[ge20_seiz_stat_0_idx] *= self.rr_epilepsy_age_ge20

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == 0)])
        random_draw_02 = pd.Series(self.module.rng.random_sample(size=len(alive_seiz_stat_0_idx)),
                                   index=df.index[df.is_alive & (df.ep_seiz_stat == 0)])

        series_prop_inc_epilepsy_seiz_freq = pd.Series(self.prop_inc_epilepsy_seiz_freq,
                                      index=df.index[df.is_alive & (df.ep_seiz_stat == 0)])

        dfx = pd.concat([eff_prob_epilepsy, random_draw_01, random_draw_02, series_prop_inc_epilepsy_seiz_freq],
                        axis=1)
        dfx.columns = ['eff_prob_epilepsy', 'random_draw_01', 'random_draw_02', 'prop_inc_epilepsy_seiz_freq']
        dfx['ep_seiz_stat'] = 0

        dfx.loc[(dfx['eff_prob_epilepsy'] > random_draw_01) & (dfx['prop_inc_epilepsy_seiz_freq'] > random_draw_02),
                dfx['ep_seiz_stat']] = 3
        dfx.loc[(dfx['eff_prob_epilepsy'] > random_draw_01) & (dfx['prop_inc_epilepsy_seiz_freq'] < random_draw_02),
                dfx['ep_seiz_stat']] = 2

        df.loc[alive_seiz_stat_0_idx, 'ep_seiz_stat'] = dfx['ep_seiz_stat']

        # todo
        # update ep_seiz_stat for people ep_seiz_stat = 1
        # update ep_seiz_stat for people ep_seiz_stat = 2
        # update ep_seiz_stat for people ep_seiz_stat = 3
        # update ep_antiep if ep_seiz_stat = 1
        # update ep_antiep if ep_seiz_stat = 2
        # update ep_antiep if ep_seiz_stat = 3
        # update ep_epi_death


class EpilepsyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):

        # todo  checking for consistency of initial condittions and transitions....

        # get some summary statistics
        df = population.props
        alive = df.is_alive.sum()

        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()


        """        
        logger.info('%s|de_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('de_depr').size().to_dict())
        
        logger.info('%s|p_depr|%s',
                    self.sim.date,
                    prop_depr)

        logger.info('%s|de_ever_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'de_ever_depr']).size().to_dict())
        
        """

        logger.debug('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())




