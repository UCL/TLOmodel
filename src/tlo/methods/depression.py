"""
First draft of depression module based on Andrew's document.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd


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
            'base probability of depression over a 3 month period if male, wealth123, no chronic condition '),
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
        'rr_depr_prev_epis_on_antidepr': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression if on antidepressants'),
        'rr_depr_age_15_20': Parameter(
            Types.REAL,
            'Relative rate of depression associated with 15-20 year olds'),
        'rr_depr_age_60plus': Parameter(
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
        'de_non_fatal_self_harm_event': Property(Types.BOOL, 'non fatal self harm event this 3 month period'),
        'de_suicide': Property(Types.BOOL, 'suicide this 3 month period'),
        'de_on_antidepr': Property(Types.BOOL, 'on anti-depressants'),
        'de_date_init_last_depr': Property(
            Types.DATE, 'When this individual last initiated a depr episode'),
        'de_date_depr_resolved': Property(
            Types.DATE, 'When the last episode of depr was resolved'),
        'de_ever_depr': Property(
            Types.BOOL, 'Whether this person has ever experienced depr'),
        'de_prob_3m_resol_depr': Property(
            Types.REAL, 'Base probability for recovering from this bout of depr (if relevant)'),
        'li_wealth': Property(
            Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'de_cc': Property(
            Types.BOOL, 'whether has chronic condition')
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
        self.parameters['init_pr_depr_m_age1519_no_cc_wealth123'] = 0.1
        self.parameters['init_rp_depr_f_not_rec_preg'] = 1
        self.parameters['init_rp_depr_f_rec_preg'] = 1
        self.parameters['init_rp_depr_age2059'] = 1
        self.parameters['init_rp_depr_agege60'] = 1
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
        self.parameters['rr_depr_prev_epis_on_antidepr'] = 30
        self.parameters['rr_depr_age_15_20'] = 1
        self.parameters['rr_depr_age_60plus'] = 3
        self.parameters['depr_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]
        self.parameters['rr_resol_depress_cc'] = 0.5
        self.parameters['rr_resol_depress_on_antidepr'] = 1.5
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
        df['de_date_init_last_depr'] = pd.NaT
        df['de_date_depr_resolved'] = pd.NaT
        df['de_non_fatal_self_harm_event'] = False
        df['de_suicide'] = False
        df['de_on_antidepr'] = False
        df['de_ever_depr'] = False
        df['de_prob_3m_resol_depr'] = False

        # todo - this to be removed when defined in other modules
        df['de_cc'] = False
        df['li_wealth'] = 3

        #  this below calls the age dataframe / call age.years to get age in years
        age = population.age

        p_depr_m_age1519_no_cc_wealth123 = self.init_pr_depr_m_age1519_no_cc_wealth123
#       p_depr_f_not_rec_preg_age1519_no_cc_wealth123 = self.prob_depr_m_age1519_no_cc_wealth123 * self.init_rp_depr_f_not_rec_preg
#       p_depr_f_rec_preg_age1519_no_cc_wealth123 = self.prob_depr_m_age1519_no_cc_wealth123 * self.init_rp_depr_f_rec_preg
#       p_depr_m_age2059_no_cc_wealth123 = self.prob_depr_m_age1519_no_cc_wealth123 * self.init_rp_depr_age2059
#       p_depr_f_not_rec_preg_age2059_no_cc_wealth123 = self.prob_depr_m_age2059_no_cc_wealth123 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_age2059
#       p_depr_f_rec_preg_age2059_no_cc_wealth123 = self.prob_depr_m_age2059_no_cc_wealth123 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_age2059
#       p_depr_m_agege60_no_cc_wealth123 = self.prob_depr_m_age1519_no_cc_wealth123 * self.init_rp_depr_agege60
#       p_depr_f_not_rec_preg_agege60_no_cc_wealth123 = self.prob_depr_m_agege60_no_cc_wealth123 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_agege60
#       p_depr_f_rec_preg_agege60_no_cc_wealth123 = self.prob_depr_m_agege60_no_cc_wealth123 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_agege60
#       p_depr_m_age1519_cc_wealth123 = self.prob_depr_m_age1519_cc_wealth123 * self.init_rp_depr_cc
#       p_depr_f_not_rec_preg_age1519_cc_wealth123 = self.prob_depr_m_age1519_cc_wealth123 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_cc
#       p_depr_f_rec_preg_age1519_cc_wealth123 = self.prob_depr_m_age1519_cc_wealth123 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_cc
#       p_depr_m_age2059_cc_wealth123 = self.prob_depr_m_age1519_cc_wealth123 * self.init_rp_depr_age2059 * self.init_rp_depr_cc
#       p_depr_f_not_rec_preg_age2059_cc_wealth123 = self.prob_depr_m_age2059_cc_wealth123 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_cc
#       p_depr_f_rec_preg_age2059_cc_wealth123 = self.prob_depr_m_age2059_cc_wealth123 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_cc
#       p_depr_m_agege60_cc_wealth123 = self.prob_depr_m_age1519_cc_wealth123 * self.init_rp_depr_agege60 * self.init_rp_depr_cc
#       p_depr_f_not_rec_preg_agege60_cc_wealth123 = self.prob_depr_m_agege60_cc_wealth123 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_cc
#       p_depr_f_rec_preg_agege60_cc_wealth123 = self.prob_depr_m_agege60_cc_wealth123 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_cc
#       p_depr_m_age1519_no_cc_wealth45 = self.prob_depr_m_age1519_no_cc_wealth45 * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_age1519_no_cc_wealth45 = self.prob_depr_m_age1519_no_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_age1519_no_cc_wealth45 = self.prob_depr_m_age1519_no_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_wealth45
#       p_depr_m_age2059_no_cc_wealth45 = self.prob_depr_m_age1519_no_cc_wealth45 * self.init_rp_depr_age2059 * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_age2059_no_cc_wealth45 = self.prob_depr_m_age2059_no_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_age2059_no_cc_wealth45 = self.prob_depr_m_age2059_no_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_wealth45
#       p_depr_m_agege60_no_cc_wealth45 = self.prob_depr_m_age1519_no_cc_wealth45 * self.init_rp_depr_agege60 * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_agege60_no_cc_wealth45 = self.prob_depr_m_agege60_no_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_agege60_no_cc_wealth45 = self.prob_depr_m_agege60_no_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_wealth45
#       p_depr_m_age1519_cc_wealth45 = self.prob_depr_m_age1519_cc_wealth45 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_age1519_cc_wealth45 = self.prob_depr_m_age1519_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_age1519_cc_wealth45 = self.prob_depr_m_age1519_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_m_age2059_cc_wealth45 = self.prob_depr_m_age1519_cc_wealth45 * self.init_rp_depr_age2059 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_age2059_cc_wealth45 = self.prob_depr_m_age2059_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_age2059_cc_wealth45 = self.prob_depr_m_age2059_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_age2059 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_m_agege60_cc_wealth45 = self.prob_depr_m_age1519_cc_wealth45 * self.init_rp_depr_agege60 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_not_rec_preg_agege60_cc_wealth45 = self.prob_depr_m_agege60_cc_wealth45 * self.init_rp_depr_f_not_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_depr_f_rec_preg_agege60_cc_wealth45 = self.prob_depr_m_agege60_cc_wealth45 * self.init_rp_depr_f_rec_preg * self.init_rp_depr_agege60 * self.init_rp_depr_cc * self.init_rp_depr_wealth45
#       p_ever_depr_not_curr_m = age.years * self.init_rp_ever_depr_per_year_older_m
#       p_ever_depr_not_curr_f = age.years * self.init_rp_ever_depr_per_year_older_f
#       p_antidepr_curr_depr = self.init_pr_antidepr_curr_depr
#       p_antidepr_ever_depr_not_curr = self.init_rp_antidepr_ever_depr_not_curr

        depr_m_age1519_no_cc_wealth123_index = df.index[(age.years >= 15) & (age.years < 20) & ~df.de_cc &
                                                        (df.li_wealth.values[:] == [1, 2, 3])
                                                        & (df.sex == 'M') & df.is_alive]

        depr_m_age1519_no_cc_wealth123 = np.random.choice([True, False], size=len(depr_m_age1519_no_cc_wealth123_index),
                                                          p=[p_depr_m_age1519_no_cc_wealth123,
                                                             1 - p_depr_m_age1519_no_cc_wealth123])

        if depr_m_age1519_no_cc_wealth123.sum():
            df.loc[depr_m_age1519_no_cc_wealth123_index, 'de_depr'] = depr_m_age1519_no_cc_wealth123

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
        self.base_3m_prob_depr = module.parameters['base_3m_prob_depr']
        self.rr_depr_wealth45 = module.parameters['rr_depr_wealth45']
        self.rr_depr_cc = module.parameters['rr_depr_cc']
        self.rr_depr_pregnancy = module.parameters['rr_depr_pregnancy']
        self.rr_depr_pregnancy = module.parameters['rr_depr_pregnancy']
        self.rr_depr_female = module.parameters['rr_depr_female']
        self.rr_depr_prev_epis = module.parameters['rr_depr_prev_epis']
        self.rr_depr_prev_epis_on_antidepr = module.parameters['rr_depr_prev_epis_on_antidepr']
        self.rr_depr_prev_epis_on_antidepr = module.parameters['rr_depr_prev_epis_on_antidepr']
        self.rr_depr_age_15_20 = module.parameters['rr_depr_age_15_20']
        self.rr_depr_age_60plus = module.parameters['rr_depr_age_60plus']
        self.depr_resolution_rates = module.parameters['depr_resolution_rates']
        self.rr_resol_depress_cc = module.parameters['rr_resol_depress_cc']
        self.rr_resol_depress_on_antidepr = module.parameters['rr_resol_depress_on_antidepr']
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
#       params = self.module.parameters

        df = population.props
        age = population.age

#       now = self.sim.date
#       rng = self.module.rng


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

#       urban_alive = (df.is_alive & df.li_urban).sum()
#       ex_alc = (df.is_alive & (age.years >= 15) & df.li_ex_alc).sum()
#       prop_urban = urban_alive / alive
#       tob = df.index[df.li_tob & df.is_alive & (age.years >= 15)]
#       m_age1519_w1_tob = df.index[df.li_tob & df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
#                                   & (df.li_wealth == 1)]
#
#       m_age1519_w1 = df.index[df.is_alive & (age.years >= 15) & (age.years < 20) & (df.sex == 'M')
#                               & (df.li_wealth == 1)]
#
#
#       f_ex_alc = (df.is_alive & (age.years >= 15) & (df.sex == 'F') & df.li_ex_alc).sum()
#
#       n_m_ge15 = (df.is_alive & (age.years >= 15) & (df.sex == 'M')).sum()
#       n_f_ge15 = (df.is_alive & (age.years >= 15) & (df.sex == 'F')).sum()

#       n_depr = (df.de_depr & df.is_alive & age.year >= 15).sum()

        n_depr = df.de_depr.sum()

        self.module.store['alive'].append(alive)

#       proportion_urban = urban_alive / (df.is_alive.sum())
#       rural_alive = (df.is_alive & (~df.li_urban)).sum()

#       mask = (df['li_date_trans_to_urban'] > self.sim.date - DateOffset(months=self.repeat))
#       newly_urban_in_last_3mths = mask.sum()

#       prop_m_urban_overwt = len(m_urban_ge15_overwt) / len(m_urban_ge15)
#       prop_f_rural_low_ex = len(f_rural_ge15_low_ex) / len(f_rural_ge15)

#       prop_wealth1 = len(wealth1) / alive
#       prop_f_ex_alc = f_ex_alc / n_f_ge15

#       prop_m_age1519_w1_tob = len(m_age1519_w1_tob) / len(m_age1519_w1)

#       self.module.o_prop_f_rural_low_ex['prop_f_rural_low_ex'].append(prop_f_rural_low_ex)
#       self.module.o_prop_m_age1519_w1_tob['prop_m_age1519_w1_tob'].append(prop_m_age1519_w1_tob)

#       wealth_count_alive = df.loc[df.is_alive, 'li_wealth'].value_counts()

        print('%s ,  n_depr:%d , alive: %d' %
              (self.sim.date, n_depr, alive),
              flush=True)


"""

code that may be referred to:

    # current depr ;
    # dataframe sex  age group  de_cc  recent_preg  wealth   pr_depr
    depr_lookup = pd.DataFrame([('M', '15-19', 0.01),
                               ('M', '20-24', 0.04),
                               ('M', '25-29', 0.04),
                               ('M', '30-34', 0.04),
                               ('M', '35-39', 0.04),
                               ('M', '40-44', 0.06),
                               ('M', '45-49', 0.06),
                               ('M', '50-54', 0.06),
                               ('M', '55-59', 0.06),
                               ('M', '60-64', 0.06),
                               ('M', '65-69', 0.06),
                               ('M', '70-74', 0.06),
                               ('M', '75-79', 0.06),
                               ('M', '80-84', 0.06),
                               ('M', '85-89', 0.06),
                               ('M', '90-94', 0.06),
                               ('M', '95-99', 0.06),
                               ('M', '100+', 0.06),

                               ('F', '15-19', 0.002),
                               ('F', '20-24', 0.002),
                               ('F', '25-29', 0.002),
                               ('F', '30-34', 0.002),
                               ('F', '35-39', 0.002),
                               ('F', '40-44', 0.002),
                               ('F', '45-49', 0.002),
                               ('F', '50-54', 0.002),
                               ('F', '55-59', 0.002),
                               ('F', '60-64', 0.002),
                               ('F', '65-69', 0.002),
                               ('F', '70-74', 0.002),
                               ('F', '75-79', 0.002),
                               ('F', '80-84', 0.002),
                               ('F', '85-89', 0.002),
                               ('F', '90-94', 0.002),
                               ('F', '95-99', 0.002),
                               ('F', '100+', 0.002)],
                              columns=['sex', 'age_range', 'p_tob'])

    # join the population dataframe with age information (we need them both together)
    df_with_age = df.loc[gte_15, ['sex', 'li_wealth']].merge(age, left_index=True, right_index=True, how='inner')
    assert len(df_with_age) == len(gte_15)  # check we have the same number of individuals after the merge

    # join the population-with-age dataframe with the tobacco use lookup table (join on sex and age_range)
    tob_probs = df_with_age.merge(tob_lookup, left_on=['sex', 'age_range'], right_on=['sex', 'age_range'],
                                  how='left')

    assert np.array_equal(tob_probs.years_exact, df_with_age.years_exact)
    # check the order of individuals is the same by comparing exact ages
    assert tob_probs.p_tob.isna().sum() == 0  # ensure we found a p_tob for every individual

    # each individual has a baseline probability
    # multiply this probability by the wealth level. wealth is a category, so convert to integer
    tob_probs = tob_probs['li_wealth'].astype(int) * tob_probs['p_tob']

    # we now have the probability of tobacco use for each individual where age >= 15
    # draw a random number between 0 and 1 for all of them
    random_draw = self.rng.random_sample(size=len(gte_15))

    # decide on tobacco use based on the individual probability is greater than random draw
    # this is a list of True/False. assign to li_tob
    df.loc[gte_15, 'li_tob'] = (random_draw < tob_probs.values)

"""






"""

possibly useful code

	        # as above - transition between overwt and not overwt
	        # transition to ovrwt depends on sex


	        currently_not_overwt_f_urban = df.index[~df.li_overwt & df.is_alive & (df.sex == 'F') & df.li_urban
	                                                & (age.years >= 15)]
	        currently_not_overwt_m_urban = df.index[~df.li_overwt & df.is_alive & (df.sex == 'M') & df.li_urban
	                                                & (age.years >= 15)]
	        currently_not_overwt_f_rural = df.index[~df.li_overwt & df.is_alive & (df.sex == 'F') & ~df.li_urban
	                                                & (age.years >= 15)]
	        currently_not_overwt_m_rural = df.index[~df.li_overwt & df.is_alive & (df.sex == 'M') & ~df.li_urban
	                                                & (age.years >= 15)]
	        currently_overwt = df.index[df.li_overwt & df.is_alive]


	        ri_overwt_f_urban = self.r_overwt * self.rr_overwt_f * self.rr_overwt_urban
	        ri_overwt_f_rural = self.r_overwt * self.rr_overwt_f
	        ri_overwt_m_urban = self.r_overwt * self.rr_overwt_urban
	        ri_overwt_m_rural = self.r_overwt


	        now_overwt_f_urban = np.random.choice([True, False],
	                                              size=len(currently_not_overwt_f_urban),
	                                              p=[ri_overwt_f_urban, 1 - ri_overwt_f_urban])


	        if now_overwt_f_urban.sum():
	            overwt_f_urban_idx = currently_not_overwt_f_urban[now_overwt_f_urban]
	            df.loc[overwt_f_urban_idx, 'li_overwt'] = True


	        now_overwt_m_urban = np.random.choice([True, False],
	                                              size=len(currently_not_overwt_m_urban),
	                                              p=[ri_overwt_m_urban, 1 - ri_overwt_m_urban])


	        if now_overwt_m_urban.sum():
	            overwt_m_urban_idx = currently_not_overwt_m_urban[now_overwt_m_urban]
	            df.loc[overwt_m_urban_idx, 'li_overwt'] = True


	        now_not_overwt = np.random.choice([True, False], size=len(currently_overwt),
	                                          p=[self.r_not_overwt, 1 - self.r_not_overwt])


	        now_overwt_f_rural = np.random.choice([True, False],
	                                              size=len(currently_not_overwt_f_rural),
	                                              p=[ri_overwt_f_rural, 1 - ri_overwt_f_rural])
	        if now_overwt_f_rural.sum():
	            overwt_f_rural_idx = currently_not_overwt_f_rural[now_overwt_f_rural]
	            df.loc[overwt_f_rural_idx, 'li_overwt'] = True


	        now_overwt_m_rural = np.random.choice([True, False],
	                                              size=len(currently_not_overwt_m_rural),
	                                              p=[ri_overwt_m_rural, 1 - ri_overwt_m_rural])
	        if now_overwt_m_rural.sum():
	            overwt_m_rural_idx = currently_not_overwt_m_rural[now_overwt_m_rural]
	            df.loc[overwt_m_rural_idx, 'li_overwt'] = True


	        if now_not_overwt.sum():
	            not_overwt_idx = currently_overwt[now_not_overwt]
	            df.loc[not_overwt_idx, 'li_overwt'] = False

"""






"""

possibly useful code

        ago_15yr = now - DateOffset(years=15)
        ago_20yr = now - DateOffset(years=20)
        ago_60yr = now - DateOffset(years=60)
        p = population

        depr = p.is_depr.copy()

        # calculate the effective probability of depr for not-depr persons
        eff_prob_depr = pd.Series(params['base_3m_prob_depr'], index=p[~p.is_depr].index)
        eff_prob_depr.loc[p.is_pregnant] *= params['rr_depr_pregnancy']
        eff_prob_depr.loc[~p.ever_depr] *= params['rr_depr_prev_episode']
        eff_prob_depr.loc[p.date_of_birth.between(ago_20yr, ago_15yr)] *= params['rr_depr_age_15_20']
        eff_prob_depr.loc[p.date_of_birth > ago_60yr] *= params['rr_depr_age_60plus']
        eff_prob_depr.loc[p.female] *= params['rr_depr_female']
        eff_prob_depr.loc[p.has_hyptension & p.has_chronic_back_pain] *= params['rr_depr_cc']

        is_newly_depr = eff_prob_depr > rng.rand(len(eff_prob_depr))
        newly_depr = is_newly_depr[is_newly_depr].index
        p[newly_depr, 'is_depr'] = True
        p[newly_depr, 'ever_depr'] = True
        p[newly_depr, 'date_init_depr'] = now
        p[newly_depr, 'date_depr_resolved'] = None
        p[newly_depr, 'prob_3m_resol_depr'] = rng.choice(
            params['depr_resolution_rates'], size=len(newly_depr))

        # continuation or resolution of depr
        eff_prob_recover = pd.Series(p.prob_3m_resol_depr, index=p[depr].index)
        eff_prob_recover[p.has_hyptension & p.has_chronic_back_pain] *= params['rr_resol_depress_cc']
        is_resol_depr = eff_prob_recover > rng.rand(len(eff_prob_recover))
        resolved_depress = is_resol_depr[is_resol_depr].index
        p[resolved_depress, 'is_depr'] = False
        p[resolved_depress, 'date_depr_resolved'] = now
        p[resolved_depress, 'date_init_depr'] = None

"""
