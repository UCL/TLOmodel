
import logging
from collections import defaultdict
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
        'init_prop_antiepileptic': Parameter(
            Types.LIST,
            'initial proportions on antiepileptic by seizure status'),
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
        'rr_seiz_stat_freq_infreq_antiepileptic': Parameter(
            Types.REAL,
            'relative rate of seizure status frequent if current infrequent if on antiepileptic'),
        'base_prob_3m_seiz_stat_infreq_freq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status infrequent if current frequent'),
        'rr_seiz_stat_infreq_freq_antiepileptic': Parameter(
            Types.REAL,
            'relative rate of seizure status infrequent if current frequent if on antiepileptic'),
        'base_prob_3m_seiz_stat_none_freq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status nonenow if current frequent'),
        'rr_seiz_stat_none_freq_antiepileptic': Parameter(
            Types.REAL,
            'relative rate of seizure status none if current frequent if on antiepileptic'),
        'base_prob_3m_seiz_stat_none_infreq': Parameter(
            Types.REAL,
            'base probability per 3 months of seizure status nonenow if current infrequent'),
        'rr_seiz_stat_none_infreq_antiepileptic': Parameter(
            Types.REAL,
            'relative rate of seizure status none if current infrequent if on antiepileptic'),
        'base_prob_3m_antiepileptic': Parameter(
            Types.REAL,
            'base probability per 3 months of starting antiepileptic, if frequent seizures'),
        'rr_antiepileptic_seiz_infreq': Parameter(
            Types.REAL,
            'relative rate of starting antiepileptic if infrequent seizures'),
        'base_prob_3m_stop_antiepileptic': Parameter(
            Types.REAL,
            'base probability per 3 months of stopping antiepileptic, if nonenow seizures'),
        'rr_stop_antiepileptic_seiz_infreq': Parameter(
            Types.REAL,
            'relative rate of stopping antiepileptic if infrequent seizures'),
        'rr_stop_antiepileptic_seiz_freq': Parameter(
            Types.REAL,
            'relative rate of stopping antiepileptic if frequent seizures'),
        'base_prob_3m_epi_death_seiz_infreq': Parameter(
            Types.REAL,
            'base probability per 3 months of epilepsy death'),
     }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'ep_seiz_stat': Property(Types.CATEGORICAL, 'seizure status'),
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
        self.parameters['init_prop_antiepileptic'] = [0, 0.1, 0.2, 0.3]
        self.parameters['base_3m_prob_epilepsy':] = 0.001
        self.parameters['rr_epilepsy_age_ge20'] = 0.0005
        self.parameters['prop_inc_epilepsy_seiz_freq'] = 0.5
        self.parameters['base_prob_3m_seiz_stat_freq_infreq'] = 0.01
        self.parameters['rr_seiz_stat_freq_infreq_antiepileptic'] = 0.1
        self.parameters['base_prob_3m_seiz_stat_infreq_freq'] = 0.01
        self.parameters['rr_seiz_stat_infreq_freq_antiepileptic'] = 0.2
        self.parameters['base_prob_3m_seiz_stat_none_freq'] = 0.05
        self.parameters['rr_seiz_stat_none_freq_antiepileptic'] = 5
        self.parameters['base_prob_3m_seiz_stat_none_infreq'] = 0.10
        self.parameters['rr_seiz_stat_none_infreq_antiepileptic'] = 5
        self.parameters['base_prob_3m_antiepileptic'] = 0.05
        self.parameters['rr_antiepileptic_seiz_infreq'] = 0.3
        self.parameters['base_prob_3m_stop_antiepileptic'] = 0.05
        self.parameters['rr_stop_antiepileptic_seiz_infreq'] = 0.1
        self.parameters['rr_stop_antiepileptic_seiz_freq'] = 0.1
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

        # allocate initial on antiepileptic





        eff_prob_epilepsy = pd.Series(self.init_pr_depr_m_age1519_no_cc_wealth123, index=df.index[df.age_years >= 15])
        eff_prob_depression.loc[cc_idx] *= self.init_rp_depr_cc
        eff_prob_depression.loc[age_2059_idx] *= self.init_rp_depr_age2059
        eff_prob_depression.loc[age_ge60_idx] *= self.init_rp_depr_agege60
        eff_prob_depression.loc[wealth45_ge15_idx] *= self.init_rp_depr_wealth45
        eff_prob_depression.loc[f_not_rec_preg_idx] *= self.init_rp_depr_f_not_rec_preg
        eff_prob_depression.loc[f_rec_preg_idx] *= self.init_rp_depr_f_rec_preg

        random_draw1 = self.rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'de_depr'] = (random_draw1 < eff_prob_depression)

        f_index = df.index[(df.sex == 'F') & df.is_alive & (df.age_years >= 15)]
        m_index = df.index[(df.sex == 'M') & df.is_alive & (df.age_years >= 15)]

        eff_prob_ever_depr = pd.Series(1, index=df.index[df.age_years >= 15])

        eff_prob_ever_depr.loc[f_index] = df.age_years * self.init_rp_ever_depr_per_year_older_f
        eff_prob_ever_depr.loc[m_index] = df.age_years * self.init_rp_ever_depr_per_year_older_m

        random_draw = self.rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'de_ever_depr'] = (eff_prob_ever_depr > random_draw)

        curr_depr_index = df.index[df.de_depr & df.is_alive]

        p_antidepr_curr_depr = self.init_pr_antidepr_curr_depr
        p_antidepr_ever_depr_not_curr = self.init_pr_antidepr_curr_depr * self.init_rp_antidepr_ever_depr_not_curr

        antidepr_curr_de = np.random.choice([True, False], size=len(curr_depr_index),
                                   p=[p_antidepr_curr_depr,
                                      1 - p_antidepr_curr_depr])

        if antidepr_curr_de.sum():
            df.loc[curr_depr_index, 'de_on_antidepr'] = antidepr_curr_de
        # for people depressed at baseline give de_date_init_most_rec_depr of sim.date
            df.loc[curr_depr_index, 'de_date_init_most_rec_depr'] = self.sim.date

        ever_depr_not_curr_index = df.index[df.de_ever_depr & ~df.de_depr & df.is_alive]

        antidepr_ev_de_not_curr = np.random.choice([True, False], size=len(ever_depr_not_curr_index),
                                   p=[p_antidepr_ever_depr_not_curr,
                                      1 - p_antidepr_ever_depr_not_curr])

        if antidepr_ev_de_not_curr.sum():
            df.loc[ever_depr_not_curr_index, 'de_on_antidepr'] = antidepr_ev_de_not_curr

        curr_depr_index = df.index[df.de_depr & df.is_alive]
        df.loc[curr_depr_index, 'de_ever_depr'] = True

# todo: find a way to use depr_resolution_rates parameter list for resol rates rather than list 0.2, 0.3, 0.5, 0.7, 0.95]

        df.loc[curr_depr_index, 'de_prob_3m_resol_depression'] = np.random.choice \
            ([0.2, 0.3, 0.5, 0.7, 0.95], size=len(curr_depr_index), p=[0.2, 0.2, 0.2, 0.2, 0.2])

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our three-monthly event to poll the population for depr starting
        or stopping.
        """
        depr_poll = DeprEvent(self)
        sim.schedule_event(depr_poll, sim.date + DateOffset(months=3))

        event = DepressionLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'de_depr'] = False
        df.at[child_id, 'de_date_init_most_rec_depr'] = pd.NaT
        df.at[child_id, 'de_date_depr_resolved'] = pd.NaT
        df.at[child_id, 'de_non_fatal_self_harm_event'] = False
        df.at[child_id, 'de_suicide'] = False
        df.at[child_id, 'de_on_antidepr'] = False
        df.at[child_id, 'de_ever_depr'] = False
        df.at[child_id, 'de_prob_3m_resol_depression'] = 0


class DeprEvent(RegularEvent, PopulationScopeEventMixin):
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

        self.base_3m_prob_depr = module.parameters['base_3m_prob_depr']
        self.rr_depr_wealth45 = module.parameters['rr_depr_wealth45']
        self.rr_depr_cc = module.parameters['rr_depr_cc']
        self.rr_depr_pregnancy = module.parameters['rr_depr_pregnancy']
        self.rr_depr_female = module.parameters['rr_depr_female']
        self.rr_depr_prev_epis = module.parameters['rr_depr_prev_epis']
        self.rr_depr_on_antidepr = module.parameters['rr_depr_on_antidepr']
        self.rr_depr_age1519 = module.parameters['rr_depr_age1519']
        self.rr_depr_agege60 = module.parameters['rr_depr_agege60']
        self.depr_resolution_rates = module.parameters['depr_resolution_rates']
        self.rr_resol_depr_cc = module.parameters['rr_resol_depr_cc']
        self.rr_resol_depr_on_antidepr = module.parameters['rr_resol_depr_on_antidepr']
        self.rate_init_antidep = module.parameters['rate_init_antidep']
        self.rate_stop_antidepr = module.parameters['rate_stop_antidepr']
        self.rate_default_antidepr = module.parameters['rate_default_antidepr']
        self.prob_3m_suicide_depr_m = module.parameters['prob_3m_suicide_depr_m']
        self.rr_suicide_depr_f = module.parameters['rr_suicide_depr_f']
        self.prob_3m_selfharm_depr = module.parameters['prob_3m_selfharm_depr']

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """

        df = population.props

        df['de_non_fatal_self_harm_event'] = False
        df['de_suicide'] = False

        ge15_not_depr_idx = df.index[(df.age_years >= 15) & ~df.de_depr & df.is_alive]
        cc_ge15_idx = df.index[df.de_cc & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        age_1519_idx = df.index[(df.age_years >= 15) & (df.age_years < 20) & df.is_alive & ~df.de_depr]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive & ~df.de_depr]
        wealth45_ge15_idx = df.index[df.de_wealth.isin([4, 5]) & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        f_not_rec_preg_idx = df.index[(df.sex == 'F') & ~df.is_pregnant & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.is_pregnant & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        ever_depr_idx = df.index[df.de_ever_depr & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        on_antidepr_idx = df.index[df.de_on_antidepr & (df.age_years >= 15) & df.is_alive & ~df.de_depr]

        eff_prob_newly_depr = pd.Series(self.base_3m_prob_depr,
                                        index=df.index[(df.age_years >= 15) & ~df.de_depr & df.is_alive])
        eff_prob_newly_depr.loc[cc_ge15_idx] *= self.rr_depr_cc
        eff_prob_newly_depr.loc[age_1519_idx] *= self.rr_depr_age1519
        eff_prob_newly_depr.loc[age_ge60_idx] *= self.rr_depr_agege60
        eff_prob_newly_depr.loc[wealth45_ge15_idx] *= self.rr_depr_wealth45
        eff_prob_newly_depr.loc[f_not_rec_preg_idx] *= self.rr_depr_female
        eff_prob_newly_depr.loc[f_rec_preg_idx] *= self.rr_depr_female*self.rr_depr_pregnancy
        eff_prob_newly_depr.loc[ever_depr_idx] *= self.rr_depr_prev_epis
        eff_prob_newly_depr.loc[on_antidepr_idx] *= self.rr_depr_on_antidepr

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(ge15_not_depr_idx)),
                                   index=df.index[(df.age_years >= 15) & ~df.de_depr & df.is_alive])

        dfx = pd.concat([eff_prob_newly_depr, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_newly_depr', 'random_draw_01']

        dfx['x_depr'] = False
        dfx['x_date_init_most_rec_depr'] = pd.NaT

        dfx.loc[dfx['eff_prob_newly_depr'] > random_draw_01, 'x_depr'] = True
        dfx.loc[dfx['eff_prob_newly_depr'] > random_draw_01, 'x_date_init_most_rec_depr'] = self.sim.date

        df.loc[ge15_not_depr_idx, 'de_depr'] = dfx['x_depr']
        df.loc[ge15_not_depr_idx, 'de_date_init_most_rec_depr'] = dfx['x_date_init_most_rec_depr']

        newly_depr_idx = df.index[df.de_date_init_most_rec_depr == self.sim.date]

        df.loc[newly_depr_idx, 'de_prob_3m_resol_depression'] = np.random.choice \
            ([0.2, 0.3, 0.5, 0.7, 0.95], size=len(newly_depr_idx), p=[0.2, 0.2, 0.2, 0.2, 0.2])

        # resolution of depression

        depr_idx = df.index[df.de_depr & df.is_alive]
        cc_depr_idx = df.index[(df.age_years >= 15) & df.de_depr & df.is_alive & df.de_cc]
        on_antidepr_idx = df.index[(df.age_years >= 15) & df.de_depr & df.is_alive & df.de_on_antidepr]

        eff_prob_depr_resolved = pd.Series(df.de_prob_3m_resol_depression,
                                               index=df.index[(df.age_years >= 15) & df.de_depr & df.is_alive])
        eff_prob_depr_resolved.loc[cc_depr_idx] *= self.rr_resol_depr_cc
        eff_prob_depr_resolved.loc[on_antidepr_idx] *= self.rr_resol_depr_on_antidepr

        random_draw_01 = pd.Series(self.module.rng.random_sample(size=len(depr_idx)),
                                   index=df.index[df.de_depr & df.is_alive])

        dfx = pd.concat([eff_prob_depr_resolved, random_draw_01], axis=1)
        dfx.columns = ['eff_prob_depr_resolved', 'random_draw_01']

        dfx['x_depr'] = True
        dfx['x_date_depr_resolved'] = pd.NaT

        dfx.loc[dfx['eff_prob_depr_resolved'] > random_draw_01, 'x_depr'] = False
        dfx.loc[dfx['eff_prob_depr_resolved'] > random_draw_01, 'x_date_depr_resolved'] = self.sim.date

        df.loc[depr_idx, 'de_depr'] = dfx['x_depr']
        df.loc[depr_idx, 'de_date_depr_resolved'] = dfx['x_date_depr_resolved']

        depr_resolved_now_idx = df.index[df.de_date_depr_resolved == self.sim.date]
        df.loc[depr_resolved_now_idx, 'de_prob_3m_resol_depression'] = 0

        curr_depr_idx = df.index[df.de_depr & df.is_alive & (df.age_years >= 15)]
        df.loc[curr_depr_idx, 'de_ever_depr'] = True

        eff_prob_self_harm = pd.Series(self.prob_3m_selfharm_depr, index=df.index[(df.age_years >= 15)
                                                                                  & df.de_depr & df.is_alive])

        random_draw = self.module.rng.random_sample(size=len(curr_depr_idx))
        df.loc[curr_depr_idx, 'de_non_fatal_self_harm_event'] = (eff_prob_self_harm > random_draw)

        curr_depr_f_idx = df.index[df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        eff_prob_suicide = pd.Series(self.prob_3m_suicide_depr_m, index=df.index[(df.age_years >= 15)
                                                                                  & df.de_depr & df.is_alive])
        eff_prob_suicide.loc[curr_depr_f_idx] *= self.rr_suicide_depr_f

        random_draw = self.module.rng.random_sample(size=len(curr_depr_idx))
        df.loc[curr_depr_idx, 'de_suicide'] = (eff_prob_suicide > random_draw)

        suicide_idx = df.index[df.de_suicide]
        df.loc[suicide_idx, 'is_alive'] = False

# todo: schedule the death event for the suicide
#       self.sim.schedule_event(InstantaneousDeath(self.module, person, cause='Other'),self.sim.date)


class DepressionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        n_depr = (df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_depr_m = (df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'M')).sum()
        n_depr_f = (df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()

        prop_depr = n_depr / alive

        """
        logger.info('%s|de_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('de_depr').size().to_dict())
        """

        logger.info('%s|p_depr|%s',
                    self.sim.date,
                    prop_depr)

        """
        logger.info('%s|de_ever_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'de_ever_depr']).size().to_dict())
        
        logger.debug('%s|person_one|%s',
                     self.sim.date,
                     df.loc[0].to_dict())
        """



