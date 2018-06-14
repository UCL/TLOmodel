"""
First draft of depression module based on Andrew's document.
"""

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class Depression(Module):
    """Models incidence and recovery from moderate/severe depression.

    Mild depression is excluded.
    Treatment for depression is in the EHP.
    """

    # Module parameters
    PARAMETERS = {
        'base_3m_prob_depression': Parameter(
            Types.REAL,
            'Base incidence of depression over a 3 month period'),
        'rr_depression_low_ses': Parameter(
            Types.REAL,
            'Relative rate of depression when low socio-economic status'),
        'rr_depression_chron_cond': Parameter(
            Types.REAL,
            'Relative rate of depression associated with chronic disease'),
        'rr_depression_pregnancy': Parameter(
            Types.REAL,
            'Relative rate of depression when pregnant or recently pregnant'),
        'rr_depression_female': Parameter(
            Types.REAL,
            'Relative rate of depression for females'),
        'rr_depression_prev_episode': Parameter(
            Types.REAL,
            'Relative rate of depression associated with previous depression'),
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
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'is_depressed': Property(Types.BOOL, 'Whether this individual is currently depressed'),
        'date_init_depression': Property(
            Types.DATE, 'When this individual became depressed (or null)'),
        'date_depression_resolved': Property(
            Types.DATE, 'When the last bout of depression was resolved (or NULL)'),
        'ever_depressed': Property(
            Types.BOOL, 'Whether this person has ever experienced depression'),
        'prob_3m_resol_depression': Property(
            Types.REAL,
            'Base probability for recovering from this bout of depression (if relevant)'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters  # To save typing!
        params['base_3m_prob_depression'] = 0.001
        params['rr_depression_low_ses'] = 3
        params['rr_depression_chron_cond'] = 1.25
        params['rr_depression_pregnancy'] = 3
        params['rr_depression_female'] = 1.5
        params['rr_depression_prev_episode'] = 50
        params['rr_depression_age_15_20'] = 1
        params['rr_depression_age_60plus'] = 3
        params['depression_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]
        params['rr_resol_depress_chron_cond'] = 0.75

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        # No-one is initially depressed (this is wrong, but we don't have data otherwise)
        population.is_depressed = False
        population.ever_depressed = False
        # Other properties are left uninitialised, i.e. null/NaN/NaT as appropriate

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
