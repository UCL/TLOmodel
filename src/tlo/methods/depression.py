"""
First draft of depression module based on Andrew's document.
"""

import logging
from collections import defaultdict
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Depression(Module):
    """Models incidence and recovery from moderate/severe depression.

    Mild depression is excluded.
    Treatment for depression is in the EHP.
    """

    # Module parameters
    PARAMETERS = {
        'init_pr_depr_m_age1519_no_cc_wealth123': Parameter(
            Types.REAL,
            'initial probability of being depressed in male age1519 with no chronic condition with wealth level 123'),
        'init_rp_depr_f_not_rec_preg': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in females not recently pregnant'),
        'init_rp_depr_f_rec_preg': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in females recently pregnant'),
        'init_rp_depr_age2059': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in 20-59 year olds'),
        'init_rp_depr_agege60': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in 60 + year olds'),
        'init_rp_depr_cc': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in people with chronic condition'),
        'init_rp_depr_wealth45': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in people with wealth level 4 or 5'),
        'init_rp_ever_depr_per_year_older_m': Parameter(
            Types.REAL,
            'initial relative prevalence ever depression per year older in men if not currently depressed'),
        'init_rp_ever_depr_per_year_older_f': Parameter(
            Types.REAL,
            'initial relative prevalence ever depression per year older in women if not currently depressed'),
        'init_pr_on_antidepr_curr_depressed': Parameter(
            Types.REAL,
            'initial prob of being on antidepressants if currently depressed'),
        'init_rp_never_depr': Parameter(
            Types.REAL,
            'initial relative prevalence of having never been depressed'),
        'init_rp_ever_depr_not_current': Parameter(
            Types.REAL,
            'initial relative prevalence of being ever depressed but not currently depressed'),
        'base_3m_prob_depr': Parameter(
            Types.REAL,
            'base probability of depression over a 3 month period if male, wealth123, no chronic condition, never previously depressed'),
        'rr_depr_wealth45': Parameter(
            Types.REAL,
            'Relative rate of depression when in wealth level 4 or 5'),
        'rr_depr_cc': Parameter(
            Types.REAL,
            'Relative rate of depression associated with chronic disease'),
        'rr_depr_pregnancy': Parameter(
            Types.REAL,
            'Relative rate of depression when pregnant or recently pregnant'),
        'rr_depr_female': Parameter(
            Types.REAL,
            'Relative rate of depression for females'),
        'rr_depr_prev_epis': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression'),
        'rr_depr_on_antidepr': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression if on antidepressants'),
        'rr_depr_age1519': Parameter(
            Types.REAL,
            'Relative rate of depression associated with 15-20 year olds'),
        'rr_depr_agege60': Parameter(
            Types.REAL,
            'Relative rate of depression associated with age > 60'),
        'depr_resolution_rates': Parameter(
            Types.LIST,
            'Probabilities that depression will resolve in a 3 month window. '
            'Each individual is equally likely to fall into one of the listed'
            ' categories.'),
        'rr_resol_depr_cc': Parameter(
            Types.REAL,
            'Relative rate of resolving depression associated with chronic disease symptoms'),
        'rr_resol_depr_on_antidepr': Parameter(
            Types.REAL,
            'Relative rate of resolving depression if on antidepressants'),
        'rate_stop_antidepr': Parameter(
            Types.REAL,
            'rate of stopping antidepressants when not currently depressed'),
        'rate_default_antidepr': Parameter(
            Types.REAL,
            'rate of stopping antidepressants when still depressed'),
        'rate_init_antidepr': Parameter(
            Types.REAL,
            'rate of initiation of antidepressants'),
        'prob_3m_suicide_depr_m': Parameter(
            Types.REAL,
            'rate of suicide in (currently depressed) men'),
        'prob_3m_suicide_depr_f': Parameter(
            Types.REAL,
            'rate of suicide in (currently depressed) women'),
        'prob_3m_selfharm_depr': Parameter(
            Types.REAL,
            'rate of non-fatal self harm in (currently depressed)'),
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'de_depr': Property(Types.BOOL, 'currently depr'),
        'de_newly_depr': Property(Types.BOOL, 'newly depressed this period'),
        'de_resol_depr': Property(Types.BOOL, 'resolved depression this period'),
        'de_base_p_depr': Property(Types.REAL, 'baseline probability of depression'),
        'de_p_new_depr': Property(Types.REAL, 'current probability of new depression'),
        'de_p_resol_depr': Property(Types.REAL, 'current probability of resolution of depression'),
        'de_non_fatal_self_harm_event': Property(Types.BOOL, 'non fatal self harm event this 3 month period'),
        'de_suicide': Property(Types.BOOL, 'suicide this 3 month period'),
        'de_on_antidepr': Property(Types.BOOL, 'on anti-depressants'),
        'de_date_init_most_rec_depr': Property(
            Types.DATE, 'When this individual last initiated a depr episode'),
        'de_date_depr_resolved': Property(
            Types.DATE, 'When the last episode of depr was resolved'),
        'de_ever_depr': Property(
            Types.BOOL, 'Whether this person has ever experienced depr'),
        'de_prob_3m_resol_depression': Property(
            Types.REAL, 'probability per 3 months of resolution of depresssion'),

        # todo - this to be removed when defined in other modules

        'de_is_pregnant': Property(
            Types.BOOL, 'recently pregnant'),
        'de_wealth': Property(
            Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'de_cc': Property(
            Types.BOOL, 'whether has chronic condition'),

    }

    def __init__(self):
        super().__init__()
# todo update this below
        self.store = {'alive': []}
        self.o_prop_depr = {'prop_depr': []}

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        # todo 2059 is base group for rr while 1519 is for prevalence at baseline

        self.parameters['init_pr_depr_m_age1519_no_cc_wealth123'] = 0.1
        self.parameters['init_rp_depr_f_not_rec_preg'] = 1.5
        self.parameters['init_rp_depr_f_rec_preg'] = 3
        self.parameters['init_rp_depr_age2059'] = 1
        self.parameters['init_rp_depr_agege60'] = 3
        self.parameters['init_rp_depr_cc'] = 1
        self.parameters['init_rp_depr_wealth45'] = 1
        self.parameters['init_rp_ever_depr_per_year_older_m'] = 0.007
        self.parameters['init_rp_ever_depr_per_year_older_f'] = 0.009
        self.parameters['init_pr_antidepr_curr_depr'] = 0.15
        self.parameters['init_rp_never_depr'] = 0
        self.parameters['init_rp_antidepr_ever_depr_not_curr'] = 1.5
        self.parameters['base_3m_prob_depr'] = 0.0007
        self.parameters['rr_depr_wealth45'] = 3
        self.parameters['rr_depr_cc'] = 1.25
        self.parameters['rr_depr_pregnancy'] = 3
        self.parameters['rr_depr_female'] = 1.5
        self.parameters['rr_depr_prev_epis'] = 50
        self.parameters['rr_depr_on_antidepr'] = 30
        self.parameters['rr_depr_age1519'] = 1
        self.parameters['rr_depr_agege60'] = 3
        self.parameters['depr_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]
        self.parameters['rr_resol_depr_cc'] = 0.5
        self.parameters['rr_resol_depr_on_antidepr'] = 1.5
        self.parameters['rate_init_antidep'] = 0.03
        self.parameters['rate_stop_antidepr'] = 0.70
        self.parameters['rate_default_antidepr'] = 0.20
        self.parameters['prob_3m_suicide_depr_m'] = 0.001
        self.parameters['prob_3m_suicide_depr_f'] = 0.0005
        self.parameters['prob_3m_selfharm_depr'] = 0.002

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['de_depr'] = False
        df['de_date_init_most_rec_depr'] = pd.NaT
        df['de_date_depr_resolved'] = pd.NaT
        df['de_non_fatal_self_harm_event'] = False
        df['de_suicide'] = False
        df['de_on_antidepr'] = False
        df['de_ever_depr'] = False
        df['de_prob_3m_resol_depression'] = 0

        # todo these below to be removed as properties
        df['de_p_new_depr'] = 0
        df['de_newly_depr'] = False
        df['de_resol_depr'] = False
        df['de_p_resol_depr'] = 0


        # todo - this to be removed when defined in other modules
        df['de_cc'] = False
        df['de_wealth'] = 3
        df['de_is_pregnant'] = False

        #  this below calls the age dataframe / call age.years to get age in years

        age_ge15_idx = df.index[(df.age_years >= 15) & df.is_alive]
        cc_idx = df.index[df.de_cc & (df.age_years >= 15) & df.is_alive]
        age_2059_idx = df.index[(df.age_years >= 20) & (df.age_years < 60) & df.is_alive]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive]
        wealth45_ge15_idx = df.index[df.de_wealth.isin([4, 5]) & (df.age_years >= 15) & df.is_alive]
        f_not_rec_preg_idx = df.index[(df.sex == 'F') & ~df.de_is_pregnant & (df.age_years >= 15) & df.is_alive]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.de_is_pregnant & (df.age_years >= 15) & df.is_alive]

        # todo: build new code which does not create extra properties for probabilities from this below

        eff_prob_depression = pd.Series(self.init_pr_depr_m_age1519_no_cc_wealth123, index = df.index[df.age_years >= 15])
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

        eff_prob_ever_depr = pd.Series(1, index = df.index[df.age_years >= 15])

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

        ever_depr_not_curr_index = df.index[df.de_ever_depr & ~df.de_depr & df.is_alive]

        antidepr_ev_de_not_curr = np.random.choice([True, False], size=len(ever_depr_not_curr_index),
                                   p=[p_antidepr_ever_depr_not_curr,
                                      1 - p_antidepr_ever_depr_not_curr])

        if antidepr_ev_de.sum():
            df.loc[ever_depr_not_curr_index, 'de_on_antidepr'] = antidepr_ev_de_not_curr

        curr_depr_index = df.index[df.de_depr & df.is_alive]
        df.loc[curr_depr_index, 'de_ever_depr'] = True

        # todo
        # - find a way to use depr_resolution_rates parameter list for resol
        # rates rather than list 0.2, 0.3, 0.5, 0.7, 0.95]

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

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        child.de_depr = False
        child.de_ever_depr = False


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
        self.prob_3m_suicide_depr_f = module.parameters['prob_3m_suicide_depr_f']
        self.prob_3m_selfharm_depr = module.parameters['prob_3m_selfharm_depr']

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """

        df['de_newly_depr'] = False

        df = population.props
        age = population.age

        now = self.sim.date

        ago_15yr = now - DateOffset(years=15)
        ago_20yr = now - DateOffset(years=20)
        ago_60yr = now - DateOffset(years=60)

        age_lt15_idx = df.index[(df.age_years < 15) & df.is_alive]
        cc_idx = df.index[df.de_cc & df.is_alive]
        age_1519_idx = df.index[(df.age_years >= 15) & (df.age_years < 20) & df.is_alive]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive]
        wealth45_idx = df.index[df.de_wealth.isin([4, 5]) & df.is_alive]
        f_not_rec_preg_idx = df.index[(df.sex == 'F') & ~df.de_is_pregnant & df.is_alive]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.de_is_pregnant & df.is_alive]
        ever_depr_idx = df.index[df.de_ever_depr & df.is_alive]
        onantidepr_idx = df.index[df.de_on_antidepr & df.is_alive]

        df['de_p_new_depr'] = self.base_3m_prob_depr

        df.loc[age_lt15_idx, 'de_p_new_depr'] *= 0
        df.loc[cc_idx, 'de_p_new_depr'] *= self.rr_depr_cc
        df.loc[age_1519_idx, 'de_p_new_depr'] *= self.rr_depr_age1519
        df.loc[age_ge60_idx, 'de_p_new_depr'] *= self.rr_depr_agege60
        df.loc[wealth45_idx, 'de_p_new_depr'] *= self.rr_depr_wealth45
        df.loc[f_rec_preg_idx, 'de_p_new_depr'] *= self.rr_depr_female*self.rr_depr_pregnancy
        df.loc[f_not_rec_preg_idx, 'de_p_new_depr'] *= self.rr_depr_female
        df.loc[ever_depr_idx, 'de_p_new_depr'] *= self.rr_prev_epis
        df.loc[ever_depr_idx, 'de_p_new_depr'] *= self.rr_depr_on_antidepr

        df['de_p_new_depr'] = min(1, df['de_p_new_depr'])

        curr_not_depr_idx = df.index[~df.de_depr & df.is_alive]

        random_draw2 = self.rng.random_sample(size=len(curr_not_depr_idx))
        df.loc[curr_not_depr_idx, 'de_newly_depr'] = (random_draw2 < df['de_p_new_depr'])

        newly_depr_idx = df.index[df.de_newly_depr]
        df.loc[newly_depr_idx, 'de_depr'] = True
        df.loc[newly_depr_idx, 'de_ever_depr'] = True
        df.loc[newly_depr_idx, 'de_date_init_most_rec_depr'] = self.sim.date
        df.loc[newly_depr_idx, 'de_prob_3m_resol_depression'] = np.random.choice \
            ([0.2, 0.3, 0.5, 0.7, 0.95], size=len(newly_depr_idx), p=[0.2, 0.2, 0.2, 0.2, 0.2])

        curr_depr_idx = df.index[df.de_depr & df.is_alive]
        df.loc[curr_depr_idx, 'de_ever_depr'] = True

        # resolution of depression

        df.loc[cc_idx, 'de_p_resol_depr'] *= self.rr_resol_depr_cc
        df.loc[onantidepr_idx, 'de_p_resol_depr'] *= self.rr_resol_depr_on_antidepr

        depr_idx = df.index[df.de_depr & df.is_alive]

        random_draw3 = self.rng.random_sample(size=len(depr_idx))
        df.loc[depr_idx, 'de_resol_depr'] = (random_draw3 > df['de_p_resol_depr'])

        depr_resol_idx = df.index[df.de_resol_depr & df.is_alive]
        df.loc[depr_resol_idx, 'de_depr'] = False
        df.loc[depr_resol_idx, 'de_date_depr_resolved'] = self.sim.date

# todo  suicide, self-harm + de-bugging + checking of graphs for consistency of initial condittions and transitions....

#       for person in alive.index[will_die]:
#           # schedule the death for "now"
#           self.sim.schedule_event(InstantaneousDeath(self.module, person, cause='Other'),
#                                   self.sim.date)


class DepressionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        alive = df.is_alive.sum()
        age = population.age

        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()

        n_depr = (df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_depr_m = (df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'M')).sum()
        n_depr_f = (df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()

        prop_depr = n_depr / alive

        self.module.store['alive'].append(alive)

        self.module.o_prop_depr['prop_depr'].append(prop_depr)

#       self.module.o_prop_m_age1519_w1_tob['prop_m_age1519_w1_tob'].append(prop_m_age1519_w1_tob)

#       wealth_count_alive = df.loc[df.is_alive, 'de_wealth'].value_counts()

        print('%s ,  n_depr_m:%d ,n_depr_f:%d , alive: %d' %
              (self.sim.date, n_depr_m, n_depr_f, alive),
              flush=True)



