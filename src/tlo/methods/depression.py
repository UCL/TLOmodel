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
        'init_pr_depressed_m_age1519_no_chron_cond_wealth123': Parameter(
            Types.REAL,
            'initial probability of being depressed in male age1519 with no chronic condition with wealth level 123'),
        'init_rp_depressed_f_not_rec_preg': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in females not recently pregnant'),
        'init_rp_depressed_f_rec_preg': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in females recently pregnant'),
        'init_rp_depressed_age2059': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in 20-59 year olds'),
        'init_rp_depressed_agege60': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in 60 + year olds'),
        'init_rp_depressed_chron_cond': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in people with chroonic condition'),
        'init_rp_depressed_wealth45': Parameter(
            Types.REAL,
            'initial relative prevalence of being depressed in people with wealth level 4 or 5'),
        'init_rp_ever_depressed_per_year_older_m': Parameter(
            Types.REAL,
            'init_rp_ever_depressed_per_year_older_m if not currently depressed'),
        'init_rp_ever_depressed_per_year_older_f': Parameter(
            Types.REAL,
            'init_rp_ever_depressed_per_year_older_f if nor currently depressed'),
        'init_pr_on_antidepr_curr_depressed': Parameter(
            Types.REAL,
            'init_pr_on_antidepr_curr_depressed'),
        'init_rp_never_depressed': Parameter(
            Types.REAL,
            'init_rp_never_depressed'),
        'init_rp_ever_depressed_not_current': Parameter(
            Types.REAL,
            'init_rp_ever_depressed_not_current'),
        'base_3m_prob_depression': Parameter(
            Types.REAL,
            'Base incidence of depression over a 3 month period'),
        'rr_depression_wealth45': Parameter(
            Types.REAL,
            'Relative rate of depression when in wealth level 4 or 5'),
        'rr_depression_chron_cond': Parameter(
            Types.REAL,
            'Relative rate of depression associated with chronic disease'),
        'rr_depression_pregnancy': Parameter(
            Types.REAL,
            'Relative rate of depression when pregnant or recently pregnant'),
        'rr_depression_female': Parameter(
            Types.REAL,
            'Relative rate of depression for females'),
        'rr_depression_prev_epis': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression'),
        'rr_depression_prev_epis_on_antidepr': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression if on antidepressants'),
        'rr_depression_age_15_20': Parameter(
            Types.REAL,
            'Relative rate of depression associated with 15-20 year olds'),
        'rr_depression_age_60plus': Parameter(
            Types.REAL,
            'Relative rate of depression associated with age > 60'),
        'depression_resolution_rates': Parameter(
            Types.LIST,
            'Probabilities that depression will resolve in a 3 month window. '
            'Each individual is equally likely to fall into one of the listed'
            ' categories.'),
        'rr_resol_depress_chron_cond': Parameter(
            Types.REAL,
            'Relative rate of resolving depression associated with chronic disease symptoms'),
        'rr_resol_depress_on_antidepr': Parameter(
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
        'prob_3m_suicide_depression_m': Parameter(
            Types.REAL,
            'rate of suicide in (currently depressed) men'),
        'prob_3m_suicide_depression_f': Parameter(
            Types.REAL,
            'rate of suicide in (currently depressed) women'),
        'prob_3m_selfharm_depression': Parameter(
            Types.REAL,
            'rate of non-fatal self harm in (currently depressed)'),
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'de_depressed': Property(Types.BOOL, 'currently depressed'),
        'de_non_fatal_self_harm_event': Property(Types.BOOL, 'non fatal self harm event this 3 month period'),
        'de_suicide': Property(Types.BOOL, 'suicide this 3 month period'),
        'de_on_antidepr': Property(Types.BOOL, 'on anti-depressants'),
        'de_date_init_last_depression': Property(
            Types.DATE, 'When this individual last initiated a depression episode'),
        'de_date_depression_resolved': Property(
            Types.DATE, 'When the last episode of depression was resolved'),
        'de_ever_depressed': Property(
            Types.BOOL, 'Whether this person has ever experienced depression'),
        'de_prob_3m_resol_depression': Property(
            Types.REAL, 'Base probability for recovering from this bout of depression (if relevant)'),
    }

    def __init__(self):
        super().__init__()
# todo update this below
        self.store = {'alive': []}
        self.o_prop_urban = {'prop_urban': []}


    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        self.parameters['init_pr_depressed_m_age1519_no_chron_cond_wealth123'] = 0.1
        self.parameters['init_rp_depressed_f_not_rec_preg'] = 1
        self.parameters['init_rp_depressed_f_rec_preg'] = 1
        self.parameters['init_rp_depressed_age2059'] = 1
        self.parameters['init_rp_depressed_agege60'] = 1
        self.parameters['init_rp_depressed_chron_cond'] = 1
        self.parameters['init_rp_depressed_wealth45'] = 1
        self.parameters['init_rp_ever_depressed_per_year_older_m'] = 0.007
        self.parameters['init_rp_ever_depressed_per_year_older_f'] = 0.009
        self.parameters['init_pr_on_antidepr_curr_depressed'] = 0.15
        self.parameters['init_rp_never_depressed'] = 0
        self.parameters['init_rp_ever_depressed_not_current'] = 1.5
        self.parameters['base_3m_prob_depression'] = 0.0007
        self.parameters['rr_depression_wealth45'] = 3
        self.parameters['rr_depression_chron_cond'] = 1.25
        self.parameters['rr_depression_pregnancy'] = 3
        self.parameters['rr_depression_female'] = 1.5
        self.parameters['rr_depression_prev_epis'] = 50
        self.parameters['rr_depression_prev_epis_on_antidepr'] = 30
        self.parameters['rr_depression_age_15_20'] = 1
        self.parameters['rr_depression_age_60plus'] = 3
        self.parameters['depression_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]
        self.parameters['rr_resol_depress_chron_cond'] = 0.5
        self.parameters['rr_resol_depress_on_antidepr'] = 1.5
        self.parameters['rate_init_antidep'] = 0.03
        self.parameters['rate_stop_antidepr'] = 0.70
        self.parameters['rate_default_antidepr'] = 0.20
        self.parameters['prob_3m_suicide_depression_m'] = 0.001
        self.parameters['prob_3m_suicide_depression_f'] = 0.0005
        self.parameters['prob_3m_selfharm_depression'] = 0.002

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['de_depressed'] = False
        df['de_date_init_last_depression'] = pd.NaT
        df['de_date_depression_resolved'] = pd.NaT
        df['de_non_fatal_self_harm_event'] = False
        df['de_suicide'] = False
        df['de_on_antidepr'] = False
        df['de_ever_depressed'] = False
        df['de_prob_3m_resol_depression'] = False

        #  this below calls the age dataframe / call age.years to get age in years
        age = population.age

    # current depression ;
    # dataframe sex  age group  de_chron_cond  recent_preg  wealth   pr_depressed
    depressed_lookup = pd.DataFrame([('M', '15-19', 0.01),
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

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our three-monthly event to poll the population for depression starting
        or stopping.
        """
        depression_poll = DepressionEvent(self)
        sim.schedule_event(depression_poll, sim.date + DateOffset(months=3))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        child.is_depressed = False
        child.ever_depressed = False


class DepressionEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually changes individuals' depressed status.

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """Create a new depression event.

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """
        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        ago_15yr = now - DateOffset(years=15)
        ago_20yr = now - DateOffset(years=20)
        ago_60yr = now - DateOffset(years=60)
        p = population

        depressed = p.is_depressed.copy()

        # calculate the effective probability of depression for not-depressed persons
        eff_prob_depression = pd.Series(params['base_3m_prob_depression'], index=p[~p.is_depressed].index)
        eff_prob_depression.loc[p.is_pregnant] *= params['rr_depression_pregnancy']
        eff_prob_depression.loc[~p.ever_depressed] *= params['rr_depression_prev_episode']
        eff_prob_depression.loc[p.date_of_birth.between(ago_20yr, ago_15yr)] *= params['rr_depression_age_15_20']
        eff_prob_depression.loc[p.date_of_birth > ago_60yr] *= params['rr_depression_age_60plus']
        eff_prob_depression.loc[p.female] *= params['rr_depression_female']
        eff_prob_depression.loc[p.has_hyptension & p.has_chronic_back_pain] *= params['rr_depression_chron_cond']

        is_newly_depressed = eff_prob_depression > rng.rand(len(eff_prob_depression))
        newly_depressed = is_newly_depressed[is_newly_depressed].index
        p[newly_depressed, 'is_depressed'] = True
        p[newly_depressed, 'ever_depressed'] = True
        p[newly_depressed, 'date_init_depression'] = now
        p[newly_depressed, 'date_depression_resolved'] = None
        p[newly_depressed, 'prob_3m_resol_depression'] = rng.choice(
            params['depression_resolution_rates'], size=len(newly_depressed))

        # continuation or resolution of depression
        eff_prob_recover = pd.Series(p.prob_3m_resol_depression, index=p[depressed].index)
        eff_prob_recover[p.has_hyptension & p.has_chronic_back_pain] *= params['rr_resol_depress_chron_cond']
        is_resol_depression = eff_prob_recover > rng.rand(len(eff_prob_recover))
        resolved_depress = is_resol_depression[is_resol_depression].index
        p[resolved_depress, 'is_depressed'] = False
        p[resolved_depress, 'date_depression_resolved'] = now
        p[resolved_depress, 'date_init_depression'] = None
