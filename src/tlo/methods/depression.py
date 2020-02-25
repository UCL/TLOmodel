"""
This is the Depression Module. Documentation at:
https://www.dropbox.com/s/8q9etj23owwlubx/Depression%20and%20Antidepressants%20-%20Description%20-%20Feb%2020.docx?dl=0
"""

# TODO: Use the new logging

from pathlib import Path

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

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
        # TODO: consider renaming -- what does "level 123" mean?

        'init_rp_depr_f_not_rec_preg': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in females not recently pregnant'
        ),
        'init_rp_depr_f_rec_preg': Parameter(
            Types.REAL, 'initial relative prevalence of being depressed in females recently pregnant'
        ),
        # TODO: Consider renaming these two are they are used to determine risk based on *Current* pregancny status

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
        'init_pr_ever_diagnosed_depression': Parameter(
            Types.REAL, 'initial prob of having ever been diagnosed with depression, amongst people with ever depr and '
                        'not on antidepr'
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
            Types.REAL, 'Relative rate of depression episode if on antidepressants'
        ),
        'rr_depr_age1519': Parameter(Types.REAL, 'Relative rate of depression associated with 15-20 year olds'),
        'rr_depr_agege60': Parameter(Types.REAL, 'Relative rate of depression associated with age > 60'),
        'depr_resolution_rates': Parameter(
            Types.LIST,
            'Probabilities that depression will resolve in a 3 month window. '
            'Each individual is equally likely to fall into one of the listed'
            ' categories. Must sum to one.',
        ),
        'rr_resol_depr_cc': Parameter(
            Types.REAL, 'Relative rate of resolving depression associated with chronic disease symptoms'
        ),
        'rr_resol_depr_on_antidepr': Parameter(
            Types.REAL, 'Relative rate of resolving depression if on antidepressants'
        ),
        'rr_resol_depr_current_talk_ther': Parameter(
            Types.REAL, 'Relative rate of resolving depression if current talking therapy'
        ),
        'rate_stop_antidepr': Parameter(Types.REAL, 'rate of stopping antidepressants when not currently depressed'),
        'rate_default_antidepr': Parameter(Types.REAL, 'rate of stopping antidepressants when still depressed'),
        'rate_init_antidepr': Parameter(Types.REAL, 'rate of initiation of antidepressants'),
        'pr_talk_ther_in_3_mth_period': Parameter(Types.REAL, 'pr_talk_ther_in_3_mth_period'),
        'prob_3m_suicide_depr_m': Parameter(Types.REAL, 'rate of suicide in (currently depressed) men'),
        'rr_suicide_depr_f': Parameter(Types.REAL, 'relative rate of suicide in women compared with me'),
        'prob_3m_selfharm_depr': Parameter(Types.REAL, 'rate of non-fatal self harm in (currently depressed)'),
        'rate_diagnosis_depression': Parameter(Types.REAL, 'rate of diagnosis of depression in a person never '
                                                           'previously diagnosed with depression'),
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'de_depr': Property(Types.BOOL, 'currently depr'),
        'de_ever_depr': Property(Types.BOOL, 'Whether this person has ever experienced depr'),
        'de_date_init_most_rec_depr': Property(Types.DATE, 'When this individual last initiated a depr episode'),
        'de_date_depr_resolved': Property(Types.DATE, 'When the last episode of depr was resolved'),

        'de_ever_diagnosed_depression': Property(Types.BOOL, 'Whether ever previously diagnosed with depression'),
        # TODO is the word 'diagnosed' used correctly here

        'de_on_antidepr': Property(Types.BOOL, 'on anti-depressants'),
        'de_current_talk_ther': Property(Types.BOOL, 'Whether having current talking therapy (in this 3 mnth period)'),

        'de_ever_non_fatal_self_harm_event': Property(Types.BOOL, 'ever had a non fatal self harm event'),

        # Temporary property
        'de_cc': Property(Types.BOOL, 'whether has chronic condition')
    }

    # Symptom that this module will use
    SYMPTOMS = {'em_SelfHarm'}  # The 'em_' prefix means that the onset of this symptom leads to seeking emergency care.

    def read_parameters(self, data_folder):
        self.load_parameters_from_dataframe(pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Depression.xlsx',
                                                          sheet_name='parameter_values'))
        p = self.parameters

        # Build the Linear Models:
        self.LinearModels = dict()

        self.LinearModels['Depression_At_Population_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['init_pr_depr_m_age1519_no_cc_wealth123'],
            Predictor('de_cc').when(True, p['init_rp_depr_cc']),
            Predictor('li_wealth').when('isin([4,5])', p['init_rp_depr_wealth45']),
            Predictor().when('(sex=="F") & (is_pregnant==True)', p['init_rp_depr_f_rec_preg']),
            Predictor().when('(sex=="F") & (is_pregnant==False)', p['init_rp_depr_f_not_rec_preg']),
            Predictor('age_years').when('.between(0, 14)', 0)
                .when('.between(15, 19)', 1.0)
                .when('.between(20, 59)', p['init_rp_depr_age2059'])
                .otherwise(p['init_rp_depr_agege60'])
        )

        self.LinearModels['Depression_Ever_At_Population_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('age_years').apply(lambda x: (x if x > 15 else 0) * p['init_rp_ever_depr_per_year_older_f']),
        )
        # TODO: make this depend on sex and age_years and use self.parameters['onit_rp_ever_depr_per_year_older_m'] for sex=='M'

        self.LinearModels['Depression_Ever_Diagnosed_At_Population_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('de_ever_depr').when(True, p['init_pr_ever_diagnosed_depression']).otherwise(0.0)
        )

        self.LinearModels['Using_AntiDepressants_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('de_depr').when(True, p['init_pr_antidepr_curr_depr']),
            Predictor().when('de_depr==False & de_ever_diagnosed_depression==True',
                             p['init_rp_antidepr_ever_depr_not_curr'])
        )

        self.LinearModels['Risk_of_Depression_Onset_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['base_3m_prob_depr'],
            Predictor('de_cc').when(True, p['rr_depr_cc']),
            Predictor('age_years').when('.between(0, 14)', 0)
                .when('.between(15, 19)', p['rr_depr_age1519'])
                .when('>=60', p['init_rp_depr_agege60']),
            Predictor('li_wealth').when('isin([4,5])', p['rr_depr_wealth45']),
            Predictor().when('(sex=="F") & (is_pregnant==True)', p['rr_depr_female'] * p['rr_depr_pregnancy']),
            Predictor().when('(sex=="F") & (is_pregnant==False)', p['rr_depr_female']),
            Predictor('de_ever_depr').when(True, p['rr_depr_prev_epis']),
            Predictor('de_on_antidepr').when(True, p['rr_depr_on_antidepr'])
        )
        # TODO; in this equation there is no RR given for those 20-60 years of age.

        self.LinearModels['Risk_of_Depression_Resolution_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('days_since_onset_of_depression', external=True)
                .when(0, 0)
                .when('<= 90', p['depr_resolution_rates'][0])
                .when('<= 180', p['depr_resolution_rates'][1])
                .when('<= 270', p['depr_resolution_rates'][2])
                .otherwise(p['depr_resolution_rates'][3]),
            Predictor('de_on_antidepr').when(True, 2),  # TODO <--- fill in these values
            Predictor('de_current_talk_ther').when(True, 1.5)  # TODO <--- fill in these values
        )

        self.LinearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_3m_selfharm_depr']
        )

        self.LinearModels['Risk_of_Suicide_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_3m_suicide_depr_m'],
            Predictor('sex').when('F', p['rr_suicide_depr_f'])
        )

        # Get DALY weight values:
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts = dict()
            # get the DALY weight - 932 and 933 are the sequale codes for depression
            # TODO: check are these for the status or for an episode?
            # TODO: if we are only modelling severe should it not just be that weight?
            self.daly_wts['severe_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=932)

            self.daly_wts['moderate_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=933)

            self.daly_wts['average_per_day_during_any_episode'] = 0.5 * ( \
                    self.daly_wts['severe_episode_major_depressive_disorder']
                    + self.daly_wts['moderate_episode_major_depressive_disorder']
            )

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def apply_linear_model(self, lm, df):
        """
        Helper function will apply the linear model (lm) on the dataframe (df) to get a probability of some event
        happening to each individual. It returns a series with same index with bools indicating the outcome
        :param lm: The linear model
        :param df: The dataframe
        :return: Series with same index containing outcomes (bool)
        """
        return self.rng.rand(len(df)) < lm.predict(df)

    def initialise_population(self, population):
        # TODO; reduce these properties when final list is decided.
        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['de_depr'] = False
        df['de_ever_depr'] = False
        df['de_date_init_most_rec_depr'] = pd.NaT
        df['de_date_depr_resolved'] = pd.NaT
        df['de_ever_diagnosed_depression'] = False
        df['de_on_antidepr'] = False
        df['de_current_talk_ther'] = False
        df['de_ever_non_fatal_self_harm_event'] = False
        df['de_cc'] = False

        # Assign initial 'current depression' status
        df.loc[df['is_alive'], 'de_depr'] = self.apply_linear_model(
            self.LinearModels['Depression_At_Population_Initialisation'],
            df.loc[df['is_alive']]
        )
        # If currently depressed, set the date on which this episode began to the start of the simulation
        df.loc[df['is_alive'] & df['de_depr'], 'de_date_init_most_rec_depr'] = self.sim.date

        # Assign initial 'ever depression' status
        df.loc[df['is_alive'], 'de_ever_depr'] = self.apply_linear_model(
            self.LinearModels['Depression_Ever_At_Population_Initialisation'],
            df.loc[df['is_alive']]
        )

        # Assign initial 'ever diagnosed' status
        df.loc[df['is_alive'], 'de_ever_diagnosed_depression'] = self.apply_linear_model(
            self.LinearModels['Depression_Ever_Diagnosed_At_Population_Initialisation'],
            df.loc[df['is_alive']]
        )

        # Assign initial 'using anti-depressants' status
        df.loc[df['is_alive'], 'de_on_antidepr'] = self.apply_linear_model(
            self.LinearModels['Using_AntiDepressants_Initialisation'],
            df.loc[df['is_alive']]
        )
        # TODO: need to get refill prescriptions through an HSI

    def initialise_simulation(self, sim):
        """
        Launch the main polling event and the logging event
        """
        sim.schedule_event(DepressionPollingEvent(self), sim.date)
        sim.schedule_event(DepressionLoggingEvent(self), sim.date)

        # Create Tracker for the number of SelfHarm and Suicide events
        self.EventsTracker = {'SelfHarmEvents': 0, 'SuicideEvents': 0}

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, 'de_depr'] = False
        df.at[child_id, 'de_ever_depr'] = False
        df.at[child_id, 'de_date_init_most_rec_depr'] = pd.NaT
        df.at[child_id, 'de_date_depr_resolved'] = pd.NaT
        df.at[child_id, 'de_ever_diagnosed_depression'] = False
        df.at[child_id, 'de_on_antidepr'] = False
        df.at[child_id, 'de_current_talk_ther'] = False
        df.at[child_id, 'de_ever_non_fatal_self_harm_event'] = False
        df.at[child_id, 'de_cc'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        Nothing happens if this module is alerted to a person attending an HSI
        """
        logger.debug(
            'This is Depression, being alerted about a health system interaction ' 'person %d for: %s',
            person_id,
            treatment_id,
        )
        pass

    def report_daly_values(self):
        """
        Report Daly Values based on current status.
        A daly weight is attracted to a status of depression for as long as the depression lasts.
        """

        def left_censor(obs, window_open):
            return obs.apply(lambda x: max(x, window_open) if ~pd.isnull(x) else pd.NaT)

        def right_censor(obs, window_close):
            return obs.apply(lambda x: window_close if pd.isnull(x) else min(x, window_close))

        df = self.sim.population.props

        # Calculate fraction of the last month that was spent depressed
        any_depr_in_the_last_month = (df['is_alive']) \
                                     & (
                                         ~pd.isnull(df['de_date_init_most_rec_depr']) &
                                         (df['de_date_init_most_rec_depr'] <= self.sim.date)
                                     ) \
                                     & (
                                         pd.isnull(df['de_date_depr_resolved']) |
                                         (df['de_date_depr_resolved'] >= (self.sim.date - DateOffset(months=1)))
                                     )

        start_depr = left_censor(df.loc[any_depr_in_the_last_month, 'de_date_init_most_rec_depr'],
                                 self.sim.date - DateOffset(months=1))
        end_depr = right_censor(df.loc[any_depr_in_the_last_month, 'de_date_depr_resolved'], self.sim.date)
        dur_depr_in_days = (end_depr - start_depr).dt.days.clip(0).fillna(0)
        days_in_last_month = (self.sim.date - (self.sim.date - DateOffset(months=1))).days
        fraction_of_month_depr = dur_depr_in_days / days_in_last_month

        # Apply the daly_wt to give a an average daly_wt for the previous month
        av_daly_wt_last_month = pd.Series(index=df.loc[df.is_alive].index, name='', data=0.0).add(
            fraction_of_month_depr * self.daly_wts['average_per_day_during_any_episode'], fill_value=0.0)

        return av_daly_wt_last_month

# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class DepressionPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    The regular event that actually changes individuals' depression status.
    It occurs every 3 months and this cannot be changed.
    To be efficient, the onset and resolution of depression events occurs at the polling event and synchonrously for
    all persons. Individual level events (HSI, self-harm/suicide events) may occur at other times and lead to earlier
    resolution.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # Create some shortcuts
        df = population.props
        p = self.module.parameters
        apply_linear_model = self.module.apply_linear_model

        # Determine who will be onset with depression among those who are not currently depressed
        # Assign initial 'current depression' status
        onset_depression = apply_linear_model(
            self.module.LinearModels['Risk_of_Depression_Onset_per3mo'],
            df.loc[df['is_alive'] & ~df['de_depr']]
        )

        df.loc[onset_depression.loc[onset_depression].index, 'de_depr'] = True
        df.loc[onset_depression.loc[onset_depression].index, 'de_date_init_most_rec_depr'] = self.sim.date

        # Determine resolution of depression for those with depression
        days_since_onset_of_depression = (self.sim.date - df.loc[df['de_depr'], 'de_date_init_most_rec_depr']).dt.days
        p_resolved_depression = self.module.LinearModels['Risk_of_Depression_Resolution_per3mo'].predict(
            df.loc[df['de_depr']], days_since_onset_of_depression=days_since_onset_of_depression)
        resolved_depression = self.module.rng.rand(len(p_resolved_depression)) < p_resolved_depression

        df.loc[resolved_depression.loc[resolved_depression].index, 'de_depr'] = False
        df.loc[resolved_depression.loc[resolved_depression].index, 'de_date_init_most_rec_depr'] = self.sim.date

        # Self-harm events (individual level events)
        will_self_harm_in_next_3mo = apply_linear_model(
            self.module.LinearModels['Risk_of_SelfHarm_per3mo'],
            df.loc[df['is_alive'] & df['de_depr']]
        )
        for person_id in will_self_harm_in_next_3mo.loc[will_self_harm_in_next_3mo].index:
            self.sim.schedule_event(DepressionSelfHarmEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))

        # Suicide events (individual level events)
        will_suicide_in_next_3mo = apply_linear_model(
            self.module.LinearModels['Risk_of_Suicide_per3mo'],
            df.loc[df['is_alive'] & df['de_depr']]
        )
        for person_id in will_suicide_in_next_3mo.loc[will_suicide_in_next_3mo].index:
            self.sim.schedule_event(DepressionSuicideEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))


class DepressionSelfHarmEvent(Event, IndividualScopeEventMixin):
    """
    This is a Self-Harm event. It has been scheduled to occur by the DepressionPollingEvent.
    It imposes the em_SelfHarm symptom, which will lead to emergency care being sought
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        logger.debug('SelfHarm event')
        self.module.EventsTracker['SelfHarmEvents'] += 1
        self.sim.population.props.at[person_id, 'de_ever_non_fatal_self_harm_event'] = True

        # Add the outward symptom to the SymptomManager. This will result in emergency care being sought
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='em_SelfHarm'
        )


class DepressionSuicideEvent(Event, IndividualScopeEventMixin):
    """
    This is a Suicide event. It has been scheduled to occur by the DepressionPollingEvent.
    It causes the immediate death of the person.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        logger.debug('Suicide event')

        self.module.EventsTracker['SuicideEvents'] += 1
        self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id, 'Suicide'), self.sim.date)



# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class DepressionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This is the LoggingEvent for Depression. It runs every 3 months and give population summaries for statuses for
    Depression.
    """

    def __init__(self, module):
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props

        # 1) Produce summary statistics for the current states
        # Popualation totals
        n_ge15 = (df.is_alive & (df.age_years >= 15)).sum()
        n_ge15_m = (df.is_alive & (df.age_years >= 15) & (df.sex == 'M')).sum()
        n_ge15_f = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F')).sum()
        n_age_50 = (df.is_alive & (df.age_years == 50)).sum()

        # Depression totals
        n_ge15_depr = (df.de_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_ge15_m_depr = (df.is_alive & (df.age_years >= 15) & (df.sex == 'M') & df.de_depr).sum()
        n_ge15_f_depr = (df.is_alive & (df.age_years >= 15) & (df.sex == 'F') & df.de_depr).sum()
        n_ever_depr = (df.de_ever_depr & df.is_alive & (df.age_years >= 15)).sum()
        n_age_50_ever_depr = (df.is_alive & (df.age_years == 50) & df.de_ever_depr).sum()

        # Numbers benefiting from interventions
        n_ever_diagnosed_depression = (df.is_alive & df.de_ever_diagnosed_depression & (df.age_years >= 15)).sum()
        n_antidepr_depr = (df.is_alive & df.de_on_antidepr & df.de_depr & (df.age_years >= 15)).sum()
        n_antidepr_ever_depr = (df.is_alive & df.de_on_antidepr & df.de_ever_depr & (df.age_years >= 15)).sum()
        n_current_talk_ther = (df.de_current_talk_ther & df.is_alive & df.de_depr).sum()

        dict_for_output = {
            'prop_ge15_depr': n_ge15_depr / n_ge15,
            'prop_ge15_m_depr': n_ge15_m_depr / n_ge15_m,
            'prop_ge15_f_depr': n_ge15_f_depr / n_ge15_f,
            'prop_ever_depr': n_ever_depr / n_ge15,
            'prop_age_50_ever_depr': n_age_50_ever_depr / n_age_50,
            'p_ever_diagnosed_depression': n_ever_diagnosed_depression / n_ge15,
            'prop_antidepr_if_curr_depr': n_antidepr_depr / n_ge15_depr,
            'prop_antidepr_if_ever_depr': n_antidepr_ever_depr / n_ever_depr,
            'prop_current_talk_ther_if_depr': n_current_talk_ther / n_ge15_depr,
        }

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        # 2) Log number of Self-Harm and Suicide Events since the last logging event
        logger.info('%s|event_counts|%s', self.sim.date, {
            'SelfHarmEvents': self.module.EventsTracker['SelfHarmEvents'],
            'SuicideEvents': self.module.EventsTracker['SuicideEvents'],
        })

        # reset the tracker
        self.module.EventsTracker = {'SelfHarmEvents': 0, 'SuicideEvents': 0}


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

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
        self.ACCEPTED_FACILITY_LEVEL = 0  # Enforces that this appointment must happen at level 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # This is the property that represents currently using antidepressants: de_on_antidepr

        # Check that the person is currently not on antidepressants
        # (not always true so commented out for now)

        #       assert df.at[person_id, 'de_on_antidepr'] is False

        # Change the flag for this person
        df.at[person_id, 'de_on_antidepr'] = True

        # TODO: Here adjust the cons footprint so that it incldues antidepressant medication

#TODO: HSI Talk Therapy

#TODO: HSI Start Antidepressant

#TODO: means of diagnosis other than self-harm

