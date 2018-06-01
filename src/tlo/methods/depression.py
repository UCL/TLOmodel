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
        population.props.loc[:, 'is_depressed'] = False
        population.props.loc[:, 'ever_depressed'] = False
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
        ago_15yr = now - DateOffset(years=15)
        ago_20yr = now - DateOffset(years=20)
        ago_60yr = now - DateOffset(years=60)
        p = population
        already_depressed = population.props[p.is_depressed]
        not_depressed = population.props[~p.is_depressed]
        # First calculate the incidence of depression
        effective_prob_depression = (
            # Not yet depressed
            params['base_3m_prob_depression'] *
            # Adjust for pregnancy
            not_depressed.loc[:, 'is_pregnant'].map(
                {True: params['rr_depression_pregnancy'], False: 1.0}) *
            # Adjust for previous depression
            p.ever_depressed.map({True: params['rr_depression_prev_episode'], False: 1.0}) *
            # Adjust for age bands
            (ago_15yr > p.date_of_birth > ago_20yr).map(
                {True: params['rr_depression_age_15_20'], False: 1.0}) *
            (ago_60yr > p.date_of_birth).map(
                {True: params['rr_depression_age_60plus'], False: 1.0}) *
            # Adjust for sex
            p.sex.map({'male': 1.0, 'female': params['rr_depression_female']})
        )
        # Adjust for chronic condition
        chronic_cond = p.has_hyptension & p.has_chronic_back_pain
        effective_prob_depression *= chronic_cond.map(
            {True: params['rr_depression_chron_cond'], False: 1.0})
        # Now make people newly depressed
        random_probs = self.module.rng.rand(len(not_depressed))
        affected = not_depressed[random_probs < effective_prob_depression]
        population.props[affected, 'is_depressed'] = True
        affected['is_depressed'] = True
        affected['ever_depressed'] = True
        affected['date_init_depression'] = now
        affected['date_depression_resolved'] = None
        affected.loc[:, 'prob_3m_resol_depression'] = self.module.rng.choice(
            params['depression_resolution_rates'], size=len(affected))

        # Continuation or resolution of depression
        effective_prob_recover = (
            p.prob_3m_resol_depression *
            chronic_cond.map({True: params['rr_resol_depress_chron_cond'], False: 1.0})
        )
        random_probs = self.module.rng.rand(len(already_depressed))
        resolved = population.props[random_probs < effective_prob_recover]
        resolved['is_depressed'] = False
        resolved['date_depression_resolved'] = now
        resolved['date_init_depression'] = None
