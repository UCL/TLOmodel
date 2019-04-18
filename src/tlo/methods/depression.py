
import logging
from collections import defaultdict
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
import numpy as np
import pandas as pd
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)


class Depression(Module):

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
        'init_rp_antidepr_ever_depr_not_curr': Parameter(
            Types.REAL,
            'initial relative prevalence of being on antidepressants if ever depressed but not currently'),
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
        'rr_suicide_depr_f': Parameter(
            Types.REAL,
            'relative rate of suicide in women compared with me'),
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
        'de_date_init_most_rec_depr': Property(
            Types.DATE, 'When this individual last initiated a depr episode'),
        'de_date_depr_resolved': Property(
              Types.DATE, 'When the last episode of depr was resolved'),
        'de_ever_depr': Property(
              Types.BOOL, 'Whether this person has ever experienced depr'),
        'de_prob_3m_resol_depression': Property(
              Types.REAL, 'probability per 3 months of resolution of depresssion'),

# todo - this to be removed when defined in other modules

        'de_wealth': Property(
              Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'de_cc': Property(
              Types.BOOL, 'whether has chronic condition')
    }

    # Declaration of how we will refer to any treatments that are related to this disease.
    TREATMENT_ID = 'antidepressant'

    def __init__(self):
        super().__init__()

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we just assign parameter values explicitly.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

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
#       self.parameters['rr_depr_prev_epis'] = 50
        self.parameters['rr_depr_prev_epis'] = 10
#       self.parameters['rr_depr_on_antidepr'] = 30
        self.parameters['rr_depr_on_antidepr'] = 3
        self.parameters['rr_depr_age1519'] = 1
        self.parameters['rr_depr_agege60'] = 3
        self.parameters['depr_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]
        self.parameters['rr_resol_depr_cc'] = 0.5
        self.parameters['rr_resol_depr_on_antidepr'] = 1.5
        self.parameters['rate_init_antidep'] = 0.00
        self.parameters['rate_init_antidep'] = 0.03
        self.parameters['rate_stop_antidepr'] = 0.70
        self.parameters['rate_default_antidepr'] = 0.20
        self.parameters['prob_3m_suicide_depr_m'] = 0.001
        self.parameters['rr_suicide_depr_f'] = 0.5
        self.parameters['prob_3m_selfharm_depr'] = 0.002

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

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

# todo - this to be removed when defined in other modules
        df['de_cc'] = False
        df['de_wealth'] = 4

        #  this below calls the age dataframe / call age.years to get age in years

        age_ge15_idx = df.index[(df.age_years >= 15) & df.is_alive]
        cc_idx = df.index[df.de_cc & (df.age_years >= 15) & df.is_alive]
        age_2059_idx = df.index[(df.age_years >= 20) & (df.age_years < 60) & df.is_alive]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive]
        wealth45_ge15_idx = df.index[df.de_wealth.isin([4, 5]) & (df.age_years >= 15) & df.is_alive]
        f_not_rec_preg_idx = df.index[(df.sex == 'F') & ~df.is_pregnant & (df.age_years >= 15) & df.is_alive]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.is_pregnant & (df.age_years >= 15) & df.is_alive]

        eff_prob_depression = pd.Series(self.init_pr_depr_m_age1519_no_cc_wealth123, index=df.index[df.age_years >= 15])
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

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Define the footprint for the intervention on the common resources
        footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
            'Name': Depression.TREATMENT_ID,
            'Nurse_Time': 15,
            'Doctor_Time': 15,
            'Electricity': False,
            'Water': False})

        self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

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


    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
#       logger.debug("This is Epilepsy being asked to report unified symptomology")

        # Map the specific symptoms for this disease onto the unified coding scheme
         df = self.sim.population.props  # shortcut to population properties dataframe

#        df.loc[df.is_alive, 'ep_unified_symptom_code'] \
#           = df.loc[df.is_alive, 'ep_seiz_stat'].map({ 0 : 1,  1 : 1,  2 : 1,  3 : 1})

#        return df.loc[df.is_alive, 'ep_unified_symptom_code']

         return pd.Series('1', index = df.index[df.is_alive])

    def on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None):

        #       logger.debug('This is epilepsy, being alerted about a health system interaction '
        #                    'person %d triggered by %s : %s', person_id, cue_type, disease_specific)

        pass

    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        #       logger.debug('This is epilepsy reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

#       health_values = df.loc[df.is_alive, 'ep_unified_symptom_code'].map({
#           '0': 0,
#           '1': 0.9,  # todo create parameter for this value - ask qaly module
#           '2': 0.3,
#           '3': 0.2
#       })
#       return health_values.loc[df.is_alive]

        return pd.Series(0.5, index=df.index[df.is_alive])


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

        # Declaration of how we will refer to any treatments that are related to this disease.
        TREATMENT_ID = 'antidepressant'

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

        # initiation of antidepressants

        depr_not_on_antidepr_idx = df.index[df.is_alive & df.de_depr & ~df.de_on_antidepr]

        eff_prob_antidepressants = pd.Series(self.rate_init_antidep,
                                             index=df.index[df.is_alive & df.de_depr & ~df.de_on_antidepr])

        random_draw = pd.Series(self.module.rng.random_sample(size=len(depr_not_on_antidepr_idx)),
                                index=df.index[df.is_alive & df.de_depr & ~df.de_on_antidepr])

        dfx = pd.concat([eff_prob_antidepressants, random_draw], axis=1)
        dfx.columns = ['eff_prob_antidepressants', 'random_draw']

        dfx['x_antidepr'] = False
        dfx.loc[dfx['eff_prob_antidepressants'] > random_draw, 'x_antidepr'] = True

        # todo: need / should have this line below ?
        df.loc[depr_not_on_antidepr_idx, 'de_on_antidepr'] = dfx['x_antidepr']

        # x_antidepr is whether requests health system for treatment to start
        for person_id in dfx.index[dfx.x_antidepr]:
            df.de_on_antidepr = self.sim.modules['HealthSystem'].query_access_to_service(person_id, TREATMENT_ID)

        # defaulting from antidepressant use

        on_antidepr_currently_depr_idx = df.index[df.is_alive & df.de_depr & df.de_on_antidepr]

        eff_prob_default_antidepr = pd.Series(self.rate_default_antidepr,
                                             index=df.index[df.is_alive & df.de_depr & df.de_on_antidepr])

        random_draw = pd.Series(self.module.rng.random_sample(size=len(on_antidepr_currently_depr_idx)),
                                index=df.index[df.is_alive & df.de_depr & df.de_on_antidepr])

        dfx = pd.concat([eff_prob_default_antidepr, random_draw], axis=1)
        dfx.columns = ['eff_prob_default_antidepr', 'random_draw']

        dfx['x_antidepr'] = True
        dfx.loc[dfx['eff_prob_default_antidepr'] > random_draw, 'x_antidepr'] = False

        df.loc[on_antidepr_currently_depr_idx, 'de_on_antidepr'] = dfx['x_antidepr']

        # stopping of antidepressants when no longer depressed

        on_antidepr_not_depr_idx = df.index[df.is_alive & ~df.de_depr & df.de_on_antidepr]

        eff_prob_stop_antidepr = pd.Series(self.rate_stop_antidepr,
                                             index=df.index[df.is_alive & ~df.de_depr & df.de_on_antidepr])

        random_draw = pd.Series(self.module.rng.random_sample(size=len(on_antidepr_not_depr_idx)),
                                index=df.index[df.is_alive & ~df.de_depr & df.de_on_antidepr])

        dfx = pd.concat([eff_prob_stop_antidepr, random_draw], axis=1)
        dfx.columns = ['eff_prob_stop_antidepr', 'random_draw']

        dfx['x_antidepr'] = True
        dfx.loc[dfx['eff_prob_stop_antidepr'] > random_draw, 'x_antidepr'] = False

        df.loc[on_antidepr_not_depr_idx, 'de_on_antidepr'] = dfx['x_antidepr']

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

        # self harm and suicide

        random_draw = self.module.rng.random_sample(size=len(curr_depr_idx))
        df.loc[curr_depr_idx, 'de_non_fatal_self_harm_event'] = (eff_prob_self_harm > random_draw)

        curr_depr_f_idx = df.index[df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        eff_prob_suicide = pd.Series(self.prob_3m_suicide_depr_m, index=df.index[(df.age_years >= 15)
                                                                                  & df.de_depr & df.is_alive])
        eff_prob_suicide.loc[curr_depr_f_idx] *= self.rr_suicide_depr_f

        random_draw = self.module.rng.random_sample(size=len(curr_depr_idx))
        df.loc[curr_depr_idx, 'de_suicide'] = (eff_prob_suicide > random_draw)

        suicide_idx = df.index[df.de_suicide]

        death_this_period = df.index[df.de_suicide]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'Suicide'),
                                    self.sim.date)


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

        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()

        n_depr = (df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_ever_depr = (df.de_ever_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_not_depr = (~df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_antidepr = (df.is_alive & df.de_on_antidepr & (df.age_years >= 15)).sum()
        n_antidepr_depr = (df.is_alive & df.de_on_antidepr & df.de_depr & (df.age_years >= 15)).sum()
        n_antidepr_not_depr = (df.is_alive & df.de_on_antidepr & ~df.de_depr & (df.age_years >= 15)).sum()

        prop_depr = n_depr / n_ge15
        prop_ever_depr = n_ever_depr / n_ge15
        prop_antidepr_depr = n_antidepr_depr / n_depr
        prop_antidepr_not_depr = n_antidepr_not_depr / n_not_depr
        prop_antidepr = n_antidepr / n_ge15

        """
        logger.info('%s|de_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby('de_depr').size().to_dict())
        """

        logger.info('%s|p_depr|%s|prop_ever_depr|%s|prop_antidepr|%s|prop_antidepr_depr|%s|prop_antidepr_not_depr|%s',
                    self.sim.date,
                    prop_depr, prop_ever_depr, prop_antidepr, prop_antidepr_depr, prop_antidepr_not_depr)

        """
        logger.info('%s|de_ever_depr|%s',
                    self.sim.date,
                    df[df.is_alive].groupby(['sex', 'de_ever_depr']).size().to_dict())
        """

#       logger.debug('%s|person_one|%s',
#                    self.sim.date,
#                    df.loc[0].to_dict())




