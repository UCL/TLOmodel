import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Depression(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Module parameters
    PARAMETERS = {
        'init_pr_depr_m_age1519_no_cc_wealth123': Parameter(
            Types.REAL,
            'initial probability of being depressed in male age1519 with no chronic condition with wealth level 123',
        ),
        'init_rp_depr_f_not_rec_preg': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in females not recently pregnant'
        ),
        'init_rp_depr_f_rec_preg': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in females recently pregnant'
        ),
        'init_rp_depr_age2059': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in 20-59 year olds'
        ),
        'init_rp_depr_agege60': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in 60 + year olds'
        ),
        'init_rp_depr_cc': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in people with chronic condition'
        ),
        'init_rp_depr_wealth45': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in people with wealth level 4 or 5'
        ),
        'init_rp_ever_depr_per_year_older_m': Parameter(
            Types.REAL, 'initial relative prevalence ever depression per year older in men if not currently depressed'
        ),
        'init_rp_ever_depr_per_year_older_f': Parameter(
            Types.REAL, 'initial relative prevalence ever depression per year older in women if not currently depressed'
        ),
        'init_pr_antidepr_curr_depr': Parameter(
            Types.REAL, 'initial prob of being on antidepressants if currently depressed'
        ),
        'init_rp_antidepr_ever_depr_not_curr': Parameter(
            Types.REAL, 'initial relative prevalence of being on antidepressants if ever depressed but not currently'
        ),
        'init_rp_never_depr': Parameter(Types.REAL, 'initial relative prevalence of having never been depressed'),
        'init_rp_ever_depr_not_current': Parameter(
            Types.REAL, 'initial relative prevalence of being ever depressed but not currently depressed'
        ),
        'base_3m_prob_depr': Parameter(
            Types.REAL,
            'base probability of depression over a 3 month period if male, wealth123, '
            'no chronic condition, never previously depressed',
        ),
        'rr_depr_wealth45': Parameter(Types.REAL, 'Relative rate of depression when in wealth level 4 or 5'),
        'rr_depr_cc': Parameter(Types.REAL, 'Relative rate of depression associated with chronic disease'),
        'rr_depr_pregnancy': Parameter(Types.REAL, 'Relative rate of depression when pregnant or recently pregnant'),
        'rr_depr_female': Parameter(Types.REAL, 'Relative rate of depression for females'),
        'rr_depr_prev_epis': Parameter(Types.REAL, 'Relative rate of depression associated with previous depression'),
        'rr_depr_on_antidepr': Parameter(
            Types.REAL, 'Relative rate of depression associated with previous depression if on antidepressants'
        ),
        'rr_depr_age1519': Parameter(Types.REAL, 'Relative rate of depression associated with 15-20 year olds'),
        'rr_depr_agege60': Parameter(Types.REAL, 'Relative rate of depression associated with age > 60'),
        'depr_resolution_rates': Parameter(
            Types.LIST,
            'Probabilities that depression will resolve in a 3 month window. '
            'Each individual is equally likely to fall into one of the listed'
            ' categories.',
        ),
        'rr_resol_depr_cc': Parameter(
            Types.REAL, 'Relative rate of resolving depression associated with chronic disease symptoms'
        ),
        'rr_resol_depr_on_antidepr': Parameter(
            Types.REAL, 'Relative rate of resolving depression if on antidepressants'
        ),
        'rate_stop_antidepr': Parameter(Types.REAL, 'rate of stopping antidepressants when not currently depressed'),
        'rate_default_antidepr': Parameter(Types.REAL, 'rate of stopping antidepressants when still depressed'),
        'rate_init_antidepr': Parameter(Types.REAL, 'rate of initiation of antidepressants'),
        'prob_3m_suicide_depr_m': Parameter(Types.REAL, 'rate of suicide in (currently depressed) men'),
        'rr_suicide_depr_f': Parameter(Types.REAL, 'relative rate of suicide in women compared with me'),
        'prob_3m_selfharm_depr': Parameter(Types.REAL, 'rate of non-fatal self harm in (currently depressed)'),
        # these definitions for disability weights are the ones in the global burden of disease list (Salomon)
        'daly_wt_severe_episode_major_depressive_disorder': Parameter(
            Types.REAL, 'daly_wt_severe_major_depressive_disorder' ' - code 932'
        ),
        'daly_wt_moderate_episode_major_depressive_disorder': Parameter(
            Types.REAL, 'daly_wt_moderate_episode_major_depressive_disorder ' '- code 933'
        ),
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'de_depr': Property(Types.BOOL, 'currently depr'),
        'de_non_fatal_self_harm_event': Property(Types.BOOL, 'non fatal self harm event this 3 month period'),
        'de_suicide': Property(Types.BOOL, 'suicide this 3 month period'),
        'de_on_antidepr': Property(Types.BOOL, 'on anti-depressants'),
        'de_date_init_most_rec_depr': Property(Types.DATE, 'When this individual last initiated a depr episode'),
        'de_date_depr_resolved': Property(Types.DATE, 'When the last episode of depr was resolved'),
        'de_ever_depr': Property(Types.BOOL, 'Whether this person has ever experienced depr'),
        'de_prob_3m_resol_depression': Property(Types.REAL, 'probability per 3 months of resolution of depresssion'),
        'de_disability': Property(Types.REAL, 'disability weight for current 3 month period'),
        # todo - this to be removed when defined in other modules
        'de_wealth': Property(Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'de_cc': Property(Types.BOOL, 'whether has chronic condition'),
    }

    def read_parameters(self, data_folder):
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Depression.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters
        p['depr_resolution_rates'] = [0.2, 0.3, 0.5, 0.7, 0.95]

        if 'HealthBurden' in self.sim.modules.keys():
            # get the DALY weight - 932 and 933 are the sequale codes for epilepsy
            p['daly_wt_severe_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=932)
            p['daly_wt_moderate_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=933)

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['de_depr'] = False
        df['de_disability'] = 0
        df['de_date_init_most_rec_depr'] = pd.NaT
        df['de_date_depr_resolved'] = pd.NaT
        df['de_non_fatal_self_harm_event'] = False
        df['de_suicide'] = False
        df['de_on_antidepr'] = False
        df['de_ever_depr'] = False
        df['de_prob_3m_resol_depression'] = 0

        # todo - this to be removed when defined in other modules
        df['de_cc'] = False
        df['de_wealth'].values[:] = 4

        #  this below calls the age dataframe / call age.years to get age in years

        # TODO: More comments on each of these steps would be very helpful

        age_ge15_idx = df.index[(df.age_years >= 15) & df.is_alive]
        cc_idx = df.index[df.de_cc & (df.age_years >= 15) & df.is_alive]
        age_2059_idx = df.index[(df.age_years >= 20) & (df.age_years < 60) & df.is_alive]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive]
        wealth45_ge15_idx = df.index[df.de_wealth.isin([4, 5]) & (df.age_years >= 15) & df.is_alive]
        f_not_rec_preg_idx = df.index[(df.sex == 'F') & ~df.is_pregnant & (df.age_years >= 15) & df.is_alive]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.is_pregnant & (df.age_years >= 15) & df.is_alive]

        eff_prob_depression = pd.Series(self.init_pr_depr_m_age1519_no_cc_wealth123, index=age_ge15_idx)
        eff_prob_depression.loc[cc_idx] *= self.init_rp_depr_cc
        eff_prob_depression.loc[age_2059_idx] *= self.init_rp_depr_age2059
        eff_prob_depression.loc[age_ge60_idx] *= self.init_rp_depr_agege60
        eff_prob_depression.loc[wealth45_ge15_idx] *= self.init_rp_depr_wealth45
        eff_prob_depression.loc[f_not_rec_preg_idx] *= self.init_rp_depr_f_not_rec_preg
        eff_prob_depression.loc[f_rec_preg_idx] *= self.init_rp_depr_f_rec_preg

        random_draw1 = self.rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'de_depr'] = random_draw1 < eff_prob_depression

        f_index = df.index[(df.sex == 'F') & df.is_alive & (df.age_years >= 15)]
        m_index = df.index[(df.sex == 'M') & df.is_alive & (df.age_years >= 15)]

        eff_prob_ever_depr = pd.Series(1, index=age_ge15_idx)

        eff_prob_ever_depr.loc[f_index] = df.loc[f_index, 'age_years'] * self.init_rp_ever_depr_per_year_older_f
        eff_prob_ever_depr.loc[m_index] = df.loc[m_index, 'age_years'] * self.init_rp_ever_depr_per_year_older_m

        random_draw = self.rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'de_ever_depr'] = eff_prob_ever_depr > random_draw

        curr_depr_index = df.index[df.de_depr & df.is_alive]

        p_antidepr_curr_depr = self.init_pr_antidepr_curr_depr
        p_antidepr_ever_depr_not_curr = self.init_pr_antidepr_curr_depr * self.init_rp_antidepr_ever_depr_not_curr

        antidepr_curr_de = np.random.choice(
            [True, False], size=len(curr_depr_index), p=[p_antidepr_curr_depr, 1 - p_antidepr_curr_depr]
        )

        if antidepr_curr_de.sum():
            df.loc[curr_depr_index, 'de_on_antidepr'] = antidepr_curr_de
            # for people depressed at baseline give de_date_init_most_rec_depr of sim.date
            df.loc[curr_depr_index, 'de_date_init_most_rec_depr'] = self.sim.date

        ever_depr_not_curr_index = df.index[df.de_ever_depr & ~df.de_depr & df.is_alive]

        antidepr_ev_de_not_curr = np.random.choice(
            [True, False],
            size=len(ever_depr_not_curr_index),
            p=[p_antidepr_ever_depr_not_curr, 1 - p_antidepr_ever_depr_not_curr],
        )

        if antidepr_ev_de_not_curr.sum():
            df.loc[ever_depr_not_curr_index, 'de_on_antidepr'] = antidepr_ev_de_not_curr

        curr_depr_index = df.index[df.de_depr & df.is_alive]
        df.loc[curr_depr_index, 'de_ever_depr'] = True

        # todo: find a way to use depr_resolution_rates parameter list for resol rates rather than
        # list 0.2, 0.3, 0.5, 0.7, 0.95]

        df.loc[curr_depr_index, 'de_prob_3m_resol_depression'] = np.random.choice(
            [0.2, 0.3, 0.5, 0.7, 0.95], size=len(curr_depr_index), p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        # disability

        depr_idx = df.index[df.is_alive & df.de_depr]

        df.loc[depr_idx, 'de_disability'] = 0.49

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our three-monthly event to poll the population for depr starting
        or stopping.
        """

        depr_poll = DeprEvent(self)
        sim.schedule_event(depr_poll, sim.date + DateOffset(months=0))

        event = DepressionLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

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
        df.at[child_id, 'de_disability'] = 0

        # todo - this to be removed when defined in other modules
        df.at[child_id, 'de_cc'] = False
        df.at[child_id, 'de_wealth'] = 4

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
        #       logger.debug('This is Epilepsy being asked to report unified symptomology')

        # Map the specific symptoms for this disease onto the unified coding scheme
        df = self.sim.population.props  # shortcut to population properties dataframe
        return pd.Series('1', index=df.index[df.is_alive])

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(
            'This is Depression, being alerted about a health system interaction ' 'person %d for: %s',
            person_id,
            treatment_id,
        )

    def report_daly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past month
        #       logger.debug('This is Depression reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        disability_series_for_alive_persons = df.loc[df.is_alive, 'de_disability']
        return disability_series_for_alive_persons


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
        p = module.parameters

        self.base_3m_prob_depr = p['base_3m_prob_depr']
        self.rr_depr_wealth45 = p['rr_depr_wealth45']
        self.rr_depr_cc = p['rr_depr_cc']
        self.rr_depr_pregnancy = p['rr_depr_pregnancy']
        self.rr_depr_female = p['rr_depr_female']
        self.rr_depr_prev_epis = p['rr_depr_prev_epis']
        self.rr_depr_on_antidepr = p['rr_depr_on_antidepr']
        self.rr_depr_age1519 = p['rr_depr_age1519']
        self.rr_depr_agege60 = p['rr_depr_agege60']
        self.depr_resolution_rates = p['depr_resolution_rates']
        self.rr_resol_depr_cc = p['rr_resol_depr_cc']
        self.rr_resol_depr_on_antidepr = p['rr_resol_depr_on_antidepr']
        self.rate_init_antidepr = p['rate_init_antidepr']
        self.rate_stop_antidepr = p['rate_stop_antidepr']
        self.rate_default_antidepr = p['rate_default_antidepr']
        self.prob_3m_suicide_depr_m = p['prob_3m_suicide_depr_m']
        self.rr_suicide_depr_f = p['rr_suicide_depr_f']
        self.prob_3m_selfharm_depr = p['prob_3m_selfharm_depr']
        self.daly_wt_moderate_episode_major_depressive_disorder = p[
            'daly_wt_moderate_episode_major_depressive_disorder']
        self.daly_wt_severe_episode_major_depressive_disorder = p[
            'daly_wt_severe_episode_major_depressive_disorder']

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population in bulk.

        :param population: the current population
        """

        # TODO: more comments on every step of this would be helpful
        df = population.props

        df.loc[df.is_alive, 'de_non_fatal_self_harm_event'] = False
        df.loc[df.is_alive, 'de_suicide'] = False
        df.loc[df.is_alive, 'de_disability'] = 0

        ge15_not_depr_idx = df.index[(df.age_years >= 15) & ~df.de_depr & df.is_alive]
        cc_ge15_idx = df.index[df.de_cc & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        age_1519_idx = df.index[(df.age_years >= 15) & (df.age_years < 20) & df.is_alive & ~df.de_depr]
        age_ge60_idx = df.index[(df.age_years >= 60) & df.is_alive & ~df.de_depr]
        wealth45_ge15_idx = df.index[df.de_wealth.isin([4, 5]) & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        f_not_rec_preg_idx = df.index[
            (df.sex == 'F') & ~df.is_pregnant & (df.age_years >= 15) & df.is_alive & ~df.de_depr
        ]
        f_rec_preg_idx = df.index[(df.sex == 'F') & df.is_pregnant & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        ever_depr_idx = df.index[df.de_ever_depr & (df.age_years >= 15) & df.is_alive & ~df.de_depr]
        on_antidepr_idx = df.index[df.de_on_antidepr & (df.age_years >= 15) & df.is_alive & ~df.de_depr]

        eff_prob_newly_depr = pd.Series(
            self.base_3m_prob_depr, index=ge15_not_depr_idx
        )
        eff_prob_newly_depr.loc[cc_ge15_idx] *= self.rr_depr_cc
        eff_prob_newly_depr.loc[age_1519_idx] *= self.rr_depr_age1519
        eff_prob_newly_depr.loc[age_ge60_idx] *= self.rr_depr_agege60
        eff_prob_newly_depr.loc[wealth45_ge15_idx] *= self.rr_depr_wealth45
        eff_prob_newly_depr.loc[f_not_rec_preg_idx] *= self.rr_depr_female
        eff_prob_newly_depr.loc[f_rec_preg_idx] *= self.rr_depr_female * self.rr_depr_pregnancy
        eff_prob_newly_depr.loc[ever_depr_idx] *= self.rr_depr_prev_epis
        eff_prob_newly_depr.loc[on_antidepr_idx] *= self.rr_depr_on_antidepr

        newly_depr = eff_prob_newly_depr > self.module.rng.random_sample(size=len(ge15_not_depr_idx))
        newly_depr_idx = newly_depr[newly_depr].index  # index where no_depressed is True

        df.loc[ge15_not_depr_idx, 'de_depr'] = newly_depr
        df.loc[ge15_not_depr_idx, 'de_date_init_most_rec_depr'] = pd.NaT
        df.loc[newly_depr_idx, 'de_date_init_most_rec_depr'] = self.sim.date
        df.loc[newly_depr_idx, 'de_prob_3m_resol_depression'] = np.random.choice(
            [0.2, 0.3, 0.5, 0.7, 0.95], size=len(newly_depr_idx), p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        # initiation of antidepressants
        depr_not_on_antidepr_idx = df.index[df.is_alive & df.de_depr & ~df.de_on_antidepr]

        eff_prob_antidepressants = pd.Series(self.rate_init_antidepr, index=depr_not_on_antidepr_idx)
        antidepr = eff_prob_antidepressants > self.module.rng.random_sample(size=len(depr_not_on_antidepr_idx))

        # get the indicies of persons who are going to present for care at somepoint in the next 3 months

        start_antidepr_this_period_idx = antidepr[antidepr].index

        # generate the HSI Events whereby persons present for care and get antidepressants
        for person_id in start_antidepr_this_period_idx:
            # For this person, determine when they will seek care (uniform distribition [0,30] days from now)
            date_seeking_care = self.sim.date + pd.DateOffset(days=int(self.module.rng.uniform(0, 30)))

            # For this person, create the HSI Event for their presentation for care
            hsi_present_for_care = HSI_Depression_Present_For_Care_And_Start_Antidepressant(self.module, person_id)

            # Enter this event to the HealthSystem
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_present_for_care, priority=0, topen=date_seeking_care, tclose=None
            )

        # defaulting from antidepressant use

        on_antidepr_currently_depr_idx = df.index[df.is_alive & df.de_depr & df.de_on_antidepr]

        eff_prob_default_antidepr = pd.Series(self.rate_default_antidepr, index=on_antidepr_currently_depr_idx)
        df.loc[on_antidepr_currently_depr_idx, 'de_on_antidepr'] = (
            # note comparison is reversed here (less than)
            eff_prob_default_antidepr < self.module.rng.random_sample(size=len(on_antidepr_currently_depr_idx))
        )

        # stopping of antidepressants when no longer depressed

        on_antidepr_not_depr_idx = df.index[df.is_alive & ~df.de_depr & df.de_on_antidepr]

        eff_prob_stop_antidepr = pd.Series(
            self.rate_stop_antidepr, index=on_antidepr_not_depr_idx
        )

        df.loc[on_antidepr_not_depr_idx, 'de_on_antidepr'] = (
            # note comparison is reversed here (less than)
            eff_prob_stop_antidepr < self.module.rng.random_sample(size=len(on_antidepr_not_depr_idx))
        )

        # resolution of depression

        depr_idx = df.index[(df.age_years >= 15) & df.de_depr & df.is_alive]
        cc_depr_idx = df.index[(df.age_years >= 15) & df.de_depr & df.is_alive & df.de_cc]
        on_antidepr_idx = df.index[(df.age_years >= 15) & df.de_depr & df.is_alive & df.de_on_antidepr]

        eff_prob_depr_resolved = pd.Series(
            df.de_prob_3m_resol_depression, index=depr_idx
        )
        eff_prob_depr_resolved.loc[cc_depr_idx] *= self.rr_resol_depr_cc
        eff_prob_depr_resolved.loc[on_antidepr_idx] *= self.rr_resol_depr_on_antidepr

        depr_resolved = eff_prob_depr_resolved < self.module.rng.random_sample(size=len(depr_idx))
        depr_resolved_idx = depr_resolved[depr_resolved].index
        df.loc[depr_idx, 'de_depr'] = depr_resolved
        df.loc[depr_idx, 'de_date_depr_resolved'] = pd.NaT
        df.loc[depr_resolved_idx, 'de_date_depr_resolved'] = self.sim.date
        df.loc[depr_resolved_idx, 'de_prob_3m_resol_depression'] = 0

        curr_depr_idx = df.index[df.de_depr & df.is_alive & (df.age_years >= 15)]
        df.loc[curr_depr_idx, 'de_ever_depr'] = True

        eff_prob_self_harm = pd.Series(
            self.prob_3m_selfharm_depr, index=df.index[(df.age_years >= 15) & df.de_depr & df.is_alive]
        )

        # disability
        depr_idx = df.index[df.is_alive & df.de_depr]

        # note this a call made about which disability weight to map to the 'moderate/severe'
        # depression defined in the model
        df.loc[depr_idx, 'de_disability'] = self.daly_wt_moderate_episode_major_depressive_disorder

        # self harm and suicide

        random_draw = self.module.rng.random_sample(size=len(curr_depr_idx))
        df.loc[curr_depr_idx, 'de_non_fatal_self_harm_event'] = eff_prob_self_harm > random_draw

        curr_depr_f_idx = df.index[df.de_depr & df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        eff_prob_suicide = pd.Series(
            self.prob_3m_suicide_depr_m, index=df.index[(df.age_years >= 15) & df.de_depr & df.is_alive]
        )
        eff_prob_suicide.loc[curr_depr_f_idx] *= self.rr_suicide_depr_f

        df.loc[curr_depr_idx, 'de_suicide'] = eff_prob_suicide > self.module.rng.random_sample(size=len(curr_depr_idx))

        # suicide_idx = df.index[df.de_suicide]

        death_this_period = df.index[df.de_suicide]
        for individual_id in death_this_period:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, 'Suicide'), self.sim.date)


#     # Declaration of how we will refer to any treatments that are related to this disease.
#     TREATMENT_ID = ''
#     Declare the HSI event


class HSI_Depression_Present_For_Care_And_Start_Antidepressant(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which someone with depression presents for care at level 0 and is provided with
    anti-depressants.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Depression_Present_For_Care_And_Start_Antidepressant'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1  # Enforces that this apppointment must happen at level 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # This is the property that repesents currently using antidepresants: de_on_antidepr

        # Check that the person is currently not on antidepressants
        # (not always true so commented out for now)

        #       assert df.at[person_id, 'de_on_antidepr'] is False

        # Change the flag for this person
        df.at[person_id, 'de_on_antidepr'] = True

        # TODO: Here adjust the cons footprint so that it incldues antidepressant medication


class DepressionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):

        # todo  checking for consistency of initial conditions and transitions....

        # get some summary statistics
        df = population.props

        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()
        n_ge45 = (df.is_alive & (df.age_years >= 45)).sum()
        n_ge15_m = (df.is_alive & (df.age_years >= 15) & (df.sex == 'M')).sum()
        n_ge15_f = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()
        n_age_50 = (df.is_alive & (df.age_years >= 49) & (df.age_years < 52)).sum()
        n_depr = (df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_depr_ge45 = (df.de_depr & df.is_alive & (df.age_years >= 45)).sum()
        n_ge15_m_depr = (df.is_alive & (df.age_years >= 15) & (df.sex == 'M') & df.de_depr).sum()
        n_ge15_f_depr = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F') & df.de_depr).sum()

        n_ever_depr = (df.de_ever_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_not_depr = (~df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_antidepr = (df.is_alive & df.de_on_antidepr & (df.age_years >= 15)).sum()
        n_antidepr_depr = (df.is_alive & df.de_on_antidepr & df.de_depr & (df.age_years >= 15)).sum()
        n_antidepr_not_depr = (df.is_alive & df.de_on_antidepr & ~df.de_depr & (df.age_years >= 15)).sum()
        n_antidepr_ever_depr = (df.is_alive & df.de_on_antidepr & df.de_ever_depr & (df.age_years >= 15)).sum()
        n_age_50_ever_depr = (df.is_alive & (df.age_years >= 49) & (df.age_years < 52) & df.de_ever_depr).sum()
        suicides_this_3m = (df.de_suicide).sum()
        self_harm_events_this_3m = (df.de_non_fatal_self_harm_event).sum()

        # prop_depr = n_depr / n_ge15
        prop_ge15_m_depr = n_ge15_m_depr / n_ge15_m
        prop_ge15_f_depr = n_ge15_f_depr / n_ge15_f
        prop_depr_ge45 = n_depr_ge45 / n_ge45
        prop_ever_depr = n_ever_depr / n_ge15
        prop_antidepr_depr = n_antidepr_depr / n_depr
        prop_antidepr_not_depr = n_antidepr_not_depr / n_not_depr
        prop_antidepr = n_antidepr / n_ge15
        prop_antidepr_ever_depr = n_antidepr_ever_depr / n_ever_depr
        prop_age_50_ever_depr = n_age_50_ever_depr / n_age_50

        # TODO: Andrew - I've re-organsied this, check that it's behaving as you wanted
        dict_for_output = {
            'prop_ever_depr': prop_ever_depr,
            'prop_antidepr': prop_antidepr,
            'prop_antidepr_depr': prop_antidepr_depr,
            'prop_antidepr_not_depr': prop_antidepr_not_depr,
            'prop_antidepr_ever_depr': prop_antidepr_ever_depr,
            'prop_ge15_m_depr': prop_ge15_m_depr,
            'prop_ge15_f_depr': prop_ge15_f_depr,
            'prop_age_50_ever_depr': prop_age_50_ever_depr,
            'prop_depr_ge45': prop_depr_ge45,
            'suicides_this_3m': suicides_this_3m,
            'self_harm_events_this_3m': self_harm_events_this_3m,
        }

        logger.info('%s|summary_stats_per_3m|%s', self.sim.date, dict_for_output)

        #       logger.info('%s|person_one|%s',
        #                    self.sim.date,
        #                    df.loc[0].to_dict())
