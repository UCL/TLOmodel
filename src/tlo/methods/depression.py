"""
This is the Depression Module.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Depression(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    INIT_DEPENDENCIES = {
        'Demography', 'Contraception', 'HealthSystem', 'Lifestyle', 'SymptomManager'
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Suicide': Cause(gbd_causes='Self-harm', label='Depression / Self-harm'),
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'SevereDepression': Cause(gbd_causes='Self-harm', label='Depression / Self-harm')
    }

    # Module parameters
    PARAMETERS = {
        'init_pr_depr_m_age1519_no_cc_wealth123': Parameter(
            Types.REAL,
            'Initial probability of being depressed in male age1519 with no chronic condition with wealth level 1 or '
            '2 or 3',
        ),

        'init_rp_depr_f_not_rec_preg': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in females not recently pregnant'
        ),

        'init_rp_depr_f_rec_preg': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in females recently pregnant'
        ),

        'init_rp_depr_age2059': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in 20-59 year olds vs 15-19'
        ),

        'init_rp_depr_agege60': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in 60+ year olds vs 15-19'
        ),

        'init_rp_depr_cc': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in people with chronic condition'
        ),

        'init_rp_depr_wealth45': Parameter(
            Types.REAL, 'Initial relative prevalence of being depressed in people with wealth level 4 or 5 vs 1 or 2 '
                        'or 3 '
        ),

        'init_rp_ever_depr_per_year_older_m': Parameter(
            Types.REAL, 'Initial relative prevalence ever depression per year older in men'
        ),

        'init_rp_ever_depr_per_year_older_f': Parameter(
            Types.REAL, 'Initial relative prevalence ever depression per year older in women'
        ),
        'init_pr_ever_talking_therapy_if_diagnosed': Parameter(
            Types.REAL, 'Initial probability of ever having had talking therapy if ever diagnosed with depression'
        ),

        'init_pr_antidepr_curr_depr': Parameter(
            Types.REAL, 'Initial probability of being on antidepressants if currently depressed'
        ),

        'init_rp_antidepr_ever_depr_not_curr': Parameter(
            Types.REAL, 'Initial relative prevalence of being on antidepressants if ever depressed but not currently'
        ),

        'init_pr_ever_diagnosed_depression': Parameter(
            Types.REAL, 'Initial probability of having ever been diagnosed with depression, amongst people with ever '
                        'depression and not on antidepressants'
        ),

        'init_pr_ever_self_harmed_if_ever_depr': Parameter(
            Types.REAL, 'Initial probability of having ever self harmed if ever depressed'
        ),

        'base_3m_prob_depr': Parameter(
            Types.REAL, 'Probability of onset of depression in a 3 month period if male, wealth 1 2 or 3, no chronic '
                        'condition and never previously depressed',
        ),

        'rr_depr_wealth45': Parameter(Types.REAL, 'Relative rate of depression when in wealth level 4 or 5 vs 1 or 2 '
                                                  'or 3'),

        'rr_depr_cc': Parameter(Types.REAL, 'Relative rate of depression if has any chronic disease'),

        'rr_depr_pregnancy': Parameter(Types.REAL, 'Relative rate of depression when pregnant or recently pregnant'),

        'rr_depr_female': Parameter(Types.REAL, 'Relative rate of depression for females vs males'),

        'rr_depr_prev_epis': Parameter(Types.REAL, 'Relative rate of depression associated with previous depression '
                                                   'vs never previously depressed'),

        'rr_depr_on_antidepr': Parameter(
            Types.REAL, 'Relative rate of depression episode if on antidepressants'
        ),

        'rr_depr_age1519': Parameter(Types.REAL, 'Relative rate of depression associated with 15-20 year olds'),

        'rr_depr_agege60': Parameter(Types.REAL, 'Relative rate of depression associated with age > 60'),

        'depr_resolution_rates': Parameter(
            Types.LIST,
            'Risk of depression resolving in 3 months if no chronic conditions and no treatments.'
            'Each individual is equally likely to be assigned each of these risks'
        ),

        'rr_resol_depr_cc': Parameter(
            Types.REAL, 'Relative rate of resolving depression if has any chronic disease'
        ),

        'rr_resol_depr_on_antidepr': Parameter(
            Types.REAL, 'Relative rate of resolving depression if on antidepressants'
        ),

        'rr_resol_depr_current_talk_ther': Parameter(
            Types.REAL, 'Relative rate of resolving depression if has ever had talking therapy vs has never had '
                        'talking therapy '
        ),

        'prob_3m_stop_antidepr': Parameter(Types.REAL,
                                           'Probability per 3 months of stopping antidepressants when not currently '
                                           'depressed.'),

        'prob_3m_default_antidepr': Parameter(Types.REAL, 'Probability per 3 months of stopping antidepressants when '
                                                          'still depressed.'),

        'prob_3m_suicide_depr_m': Parameter(Types.REAL, 'Probability per 3 months of suicide in currently depressed '
                                                        'men'),

        'rr_suicide_depr_f': Parameter(Types.REAL, 'Relative risk of suicide in women compared with men'),

        'prob_3m_selfharm_depr': Parameter(Types.REAL, 'Probability per 3 months of non-fatal self harm in those '
                                                       'currently depressed'),

        'sensitivity_of_assessment_of_depression': Parameter(Types.REAL, 'The sensitivity of the clinical assessment '
                                                                         'in detecting the true current status of '
                                                                         'depression'),

        'pr_assessed_for_depression_in_generic_appt_level1': Parameter(
            Types.REAL,
            'Probability that a person is assessed for depression during a non-emergency generic appointment'
            'level 1'),

        'anti_depressant_medication_item_code': Parameter(Types.INT,
                                                          'The item code used for one month of anti-depressant '
                                                          'treatment')
    }

    # Properties of individuals 'owned' by this module
    PROPERTIES = {
        'de_depr': Property(Types.BOOL, 'whether this person is currently depressed'),
        'de_ever_depr': Property(Types.BOOL, 'whether this person has ever experienced depression'),
        'de_date_init_most_rec_depr': Property(Types.DATE, 'date this person last initiated a depression episode'),
        'de_date_depr_resolved': Property(Types.DATE, 'date this person resolved last episode of depression'),
        'de_intrinsic_3mo_risk_of_depr_resolution': Property(Types.REAL,
                                                             'the risk per 3 mo of an episode of depression being '
                                                             'resolved in absence of any treatment'),
        'de_ever_diagnosed_depression': Property(Types.BOOL, 'whether ever diagnosed with depression'),

        'de_on_antidepr': Property(Types.BOOL, 'is currently on anti-depressants'),
        'de_ever_talk_ther': Property(Types.BOOL,
                                      'whether this person has ever had a session of talking therapy'),

        'de_ever_non_fatal_self_harm_event': Property(Types.BOOL, 'ever had a non-fatal self harm event'),
        'de_cc': Property(Types.BOOL, 'whether this person has chronic condition'),
        # TODO: <--- define and update at poll
        'de_recently_pregnant': Property(Types.BOOL, 'whether this person is female and is either currently pregnant '
                                                     'or had a last pregnancy less than one year ago')
    }

    def read_parameters(self, data_folder):
        "read parameters, register disease module with healthsystem and register symptoms"
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Depression.xlsx',
                          sheet_name='parameter_values')
        )
        p = self.parameters

        # Build the Linear Models:
        self.linearModels = dict()
        self.linearModels['Depression_At_Population_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.parameters['init_pr_depr_m_age1519_no_cc_wealth123'],
            Predictor('de_cc').when(True, p['init_rp_depr_cc']),
            Predictor('li_wealth').when('.isin([4,5])', p['init_rp_depr_wealth45']),
            Predictor().when('(sex=="F") & de_recently_pregnant', p['init_rp_depr_f_rec_preg']),
            Predictor().when('(sex=="F") & ~de_recently_pregnant', p['init_rp_depr_f_not_rec_preg']),
            Predictor(
                'age_years',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('.between(0, 14)', 0)
            .when('.between(15, 19)', 1.0)
            .when('.between(20, 59)', p['init_rp_depr_age2059'])
            .when('>= 60', p['init_rp_depr_agege60'])
        )

        self.linearModels['Depression_Ever_At_Population_Initialisation_Males'] = LinearModel.multiplicative(
            Predictor('age_years').apply(
                lambda x: (x if x > 15 else 0) * self.parameters['init_rp_ever_depr_per_year_older_m']
            )
        )

        self.linearModels['Depression_Ever_At_Population_Initialisation_Females'] = LinearModel.multiplicative(
            Predictor('age_years').apply(lambda x: (x if x > 15 else 0) * p['init_rp_ever_depr_per_year_older_f'])
        )

        self.linearModels['Depression_Ever_Diagnosed_At_Population_Initialisation'] = LinearModel.multiplicative(
            Predictor('de_ever_depr').when(True, p['init_pr_ever_diagnosed_depression'])
                                     .otherwise(0.0)
        )

        self.linearModels['Using_AntiDepressants_Initialisation'] = LinearModel.multiplicative(
            Predictor('de_depr').when(True, p['init_pr_antidepr_curr_depr']),
            Predictor().when('~de_depr & de_ever_diagnosed_depression', p['init_rp_antidepr_ever_depr_not_curr'])
        )

        self.linearModels['Ever_Talking_Therapy_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['init_pr_ever_talking_therapy_if_diagnosed'],
            Predictor('de_ever_diagnosed_depression').when(False, 0)
        )

        self.linearModels['Ever_Self_Harmed_Initialisation'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['init_pr_ever_self_harmed_if_ever_depr'],
            Predictor('de_ever_depr').when(False, 0)
        )

        self.linearModels['Risk_of_Depression_Onset_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['base_3m_prob_depr'],
            Predictor('de_cc').when(True, p['rr_depr_cc']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(0, 14)', 0)
            .when('.between(15, 19)', p['rr_depr_age1519'])
            .when('>=60', p['rr_depr_agege60']),
            Predictor('li_wealth').when('.isin([4,5])', p['rr_depr_wealth45']),
            Predictor('sex').when('F', p['rr_depr_female']),
            Predictor('de_recently_pregnant').when(True, p['rr_depr_pregnancy']),
            Predictor('de_ever_depr').when(True, p['rr_depr_prev_epis']),
            Predictor('de_on_antidepr').when(True, p['rr_depr_on_antidepr'])
        )

        self.linearModels['Risk_of_Depression_Resolution_per3mo'] = LinearModel.multiplicative(
            Predictor('de_intrinsic_3mo_risk_of_depr_resolution').apply(lambda x: x),
            Predictor('de_cc').when(True, p['rr_resol_depr_cc']),
            Predictor('de_on_antidepr').when(True, p['rr_resol_depr_on_antidepr']),
            Predictor('de_ever_talk_ther').when(True, p['rr_resol_depr_current_talk_ther'])
        )

        self.linearModels['Risk_of_Stopping_Antidepressants_per3mo'] = LinearModel.multiplicative(
            Predictor('de_depr').when(True, p['prob_3m_default_antidepr'])
                                .when(False, p['prob_3m_stop_antidepr'])
        )

        self.linearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_3m_selfharm_depr']
        )

        self.linearModels['Risk_of_Suicide_per3mo'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['prob_3m_suicide_depr_m'],
            Predictor('sex').when('F', p['rr_suicide_depr_f'])
        )

        # Get DALY weight values:
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts = dict()
            self.daly_wts['severe_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=932)

            self.daly_wts['moderate_episode_major_depressive_disorder'] = self.sim.modules[
                'HealthBurden'
            ].get_daly_weight(sequlae_code=933)

            # The average of these is what is used for the weight for any episode of depression.
            self.daly_wts['average_per_day_during_any_episode'] = (
                0.33 * self.daly_wts['severe_episode_major_depressive_disorder']
                + 0.66 * self.daly_wts['moderate_episode_major_depressive_disorder']
            )

        # Symptom that this module will use
        self.sim.modules['SymptomManager'].register_symptom(Symptom.emergency(name='Injuries_From_Self_Harm',
                                                                              which='adults'))

    def apply_linear_model(self, lm, df):
        """
        Helper function will apply the linear model (lm) on the dataframe (df) to get a probability of some event
        happening to each individual. It then returns a series with same index with bools indicating the outcome based
        on the toss of the biased coin.
        :param lm: The linear model
        :param df: The dataframe
        :return: Series with same index containing outcomes (bool)
        """
        return self.rng.random_sample(len(df)) < lm.predict(df)

    def initialise_population(self, population):
        df = population.props
        df['de_depr'] = False
        df['de_ever_depr'] = False
        df['de_date_init_most_rec_depr'] = pd.NaT
        df['de_date_depr_resolved'] = pd.NaT
        df['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
        df['de_ever_diagnosed_depression'] = False
        df['de_on_antidepr'] = False
        df['de_ever_talk_ther'] = False
        df['de_ever_non_fatal_self_harm_event'] = False
        df['de_recently_pregnant'] = df['is_pregnant'] | (
            df['date_of_last_pregnancy'] > (self.sim.date - DateOffset(years=1))
        )
        df['de_cc'] = False

        # Assign initial 'current depression' status
        df.loc[df['is_alive'], 'de_depr'] = self.apply_linear_model(
            self.linearModels['Depression_At_Population_Initialisation'],
            df.loc[df['is_alive']]
        )
        # If currently depressed, set the date on which this episode began to the start of the simulation
        # and draw the intrinsic risk of resolution
        df.loc[df['is_alive'] & df['de_depr'], 'de_date_init_most_rec_depr'] = self.sim.date
        df.loc[df['is_alive'] & df['de_depr'], 'de_intrinsic_3mo_risk_of_depr_resolution'] = \
            self.rng.choice(
                self.parameters['depr_resolution_rates'],
                (df['is_alive'] & df['de_depr']).sum()
            )

        # Assign initial 'ever depression' status (uses separate LinearModels for Males and Females due to the nature
        # of the model that is specified)
        df.loc[(df['is_alive'] & (df['sex'] == 'M')), 'de_ever_depr'] = self.apply_linear_model(
            self.linearModels['Depression_Ever_At_Population_Initialisation_Males'],
            df.loc[(df['is_alive'] & (df['sex'] == 'M'))]
        )
        df.loc[(df['is_alive'] & (df['sex'] == 'F')), 'de_ever_depr'] = self.apply_linear_model(
            self.linearModels['Depression_Ever_At_Population_Initialisation_Females'],
            df.loc[(df['is_alive'] & (df['sex'] == 'F'))]
        )

        df.loc[(df['is_alive'] & df['de_depr']), 'de_ever_depr'] = True  # For logical consistency
        df.loc[(df['is_alive'] & ~df['de_depr'] & df['de_ever_depr']), 'de_date_depr_resolved'] = \
            self.sim.date - DateOffset(days=1)  # If ever had depression, needs a resolution date in the past

        # Assign initial 'ever diagnosed' status
        df.loc[df['is_alive'], 'de_ever_diagnosed_depression'] = self.apply_linear_model(
            self.linearModels['Depression_Ever_Diagnosed_At_Population_Initialisation'],
            df.loc[df['is_alive']]
        )

        # Assign initial 'de_ever_talk_ther' status
        df.loc[df['is_alive'], 'de_ever_talk_ther'] = self.apply_linear_model(
            self.linearModels['Ever_Talking_Therapy_Initialisation'],
            df.loc[df['is_alive']]
        )

        # Assign initial 'de_ever_non_fatal_self_harm_event' status
        df.loc[df['is_alive'], 'de_ever_non_fatal_self_harm_event'] = self.apply_linear_model(
            self.linearModels['Ever_Self_Harmed_Initialisation'],
            df.loc[df['is_alive']]
        )

        # Assign initial 'using anti-depressants' status to those who are currently depressed and diagnosed
        df.loc[df['is_alive'] & df['de_depr'] & df['de_ever_diagnosed_depression'], 'de_on_antidepr'] = \
            self.apply_linear_model(
                self.linearModels['Using_AntiDepressants_Initialisation'],
                df.loc[df['is_alive'] & df['de_depr'] & df['de_ever_diagnosed_depression']]
            )

    def initialise_simulation(self, sim):
        """
        Launch the main polling event and the logging event.
        Schedule the refill prescriptions for those on antidepressants.
        Register the assessment of depression with the DxManager.

        """
        sim.schedule_event(DepressionPollingEvent(self), sim.date)
        sim.schedule_event(DepressionLoggingEvent(self), sim.date)

        # Create Tracker for the number of SelfHarm and Suicide events
        self.eventsTracker = {'SelfHarmEvents': 0, 'SuicideEvents': 0}

        # Create the diagnostic representing the assessment for whether a person is diagnosed with depression
        # NB. Specificity is assumed to be 100%
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assess_depression=DxTest(
                property='de_depr',
                sensitivity=self.parameters['sensitivity_of_assessment_of_depression'],
            )
        )

        # For those that are taking anti-depressants at initiation, schedule their refill HSI appointments
        # Scatter these refill appointments over approx the first month of the simulation (these refills are assumed
        # to occur monthly).
        df = sim.population.props
        if df['de_on_antidepr'].sum():
            for person_id in df.loc[df['de_on_antidepr']].index:
                date_of_next_appt_scheduled = self.sim.date + DateOffset(days=self.rng.randint(0, 30))
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Depression_Refill_Antidepressant(person_id=person_id, module=self),
                    priority=1,
                    topen=date_of_next_appt_scheduled,
                    tclose=date_of_next_appt_scheduled + DateOffset(days=7)
                )

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual -- they will not have depression or any history of it.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, 'de_depr'] = False
        df.at[child_id, 'de_ever_depr'] = False
        df.at[child_id, 'de_date_init_most_rec_depr'] = pd.NaT
        df.at[child_id, 'de_date_depr_resolved'] = pd.NaT
        df.at[child_id, 'de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
        df.at[child_id, 'de_ever_diagnosed_depression'] = False
        df.at[child_id, 'de_on_antidepr'] = False
        df.at[child_id, 'de_ever_talk_ther'] = False
        df.at[child_id, 'de_ever_non_fatal_self_harm_event'] = False
        df.at[child_id, 'de_recently_pregnant'] = False
        df.at[child_id, 'de_cc'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        Nothing happens if this module is alerted to a person attending an HSI
        """
        pass

    def report_daly_values(self):
        """
        Report DALYs based status in the previous month.
        A DALY weight is attached to a status of depression for as long as the depression lasted in the previous month.
        """

        def left_censor(obs, window_open):
            return obs.apply(lambda x: max(x, window_open) if ~pd.isnull(x) else pd.NaT)

        def right_censor(obs, window_close):
            return obs.apply(lambda x: window_close if pd.isnull(x) else min(x, window_close))

        df = self.sim.population.props

        # Calculate fraction of the last month that was spent depressed
        any_depr_in_the_last_month = (df['is_alive']) & (
            ~pd.isnull(df['de_date_init_most_rec_depr']) & (df['de_date_init_most_rec_depr'] <= self.sim.date)
        ) & (
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
        av_daly_wt_last_month = pd.Series(index=df.loc[df.is_alive].index, name='SevereDepression', data=0.0).add(
            fraction_of_month_depr * self.daly_wts['average_per_day_during_any_episode'], fill_value=0.0)

        return av_daly_wt_last_month

    def do_when_suspected_depression(self, person_id, hsi_event):
        """
        This is called by the a generic HSI event when depression is suspected or otherwise investigated.
        :param person_id:
        :param hsi_event: The HSI event that has called this event
        :return:
        """

        # Assess for depression and initiate treatments for depression if positive diagnosis
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='assess_depression',
                                                                   hsi_event=hsi_event
                                                                   ):
            # If depressed: diagnose the person with depression
            self.sim.population.props.at[person_id, 'de_ever_diagnosed_depression'] = True

            # Provide talking therapy (This can occur even if the person has already had talking therapy before)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Depression_TalkingTherapy(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date
            )

            # Initiate person on anti-depressants (at the same facility level as the HSI event that is calling)
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Depression_Start_Antidepressant(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date
            )


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class DepressionPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    The regular event that actually changes individuals' depression status.
    It occurs every 3 months and this cannot be changed.
    The onset and resolution of depression events occurs at the polling event and synchronously for
    all persons. Individual level events (HSI, self-harm/suicide events) may occur at other times.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # Create some shortcuts
        df = population.props
        p = self.module.parameters
        apply_linear_model = self.module.apply_linear_model

        # -----------------------------------------------------------------------------------------------------
        # Update properties that are used by the module
        df['de_recently_pregnant'] = df['is_pregnant'] | (
            df['date_of_last_pregnancy'] > (self.sim.date - DateOffset(years=1)))

        assert (df.loc[df['is_pregnant'], 'de_recently_pregnant']).all()
        assert not (df['de_recently_pregnant'] & pd.isnull(df['date_of_last_pregnancy'])).any()
        assert (df.loc[((~df['is_pregnant']) & df['de_recently_pregnant']), 'date_of_last_pregnancy'] > (
            self.sim.date - DateOffset(years=1))).all()

        # -----------------------------------------------------------------------------------------------------
        # Determine who will be onset with depression among those who are not currently depressed
        onset_depression = apply_linear_model(
            self.module.linearModels['Risk_of_Depression_Onset_per3mo'],
            df.loc[df['is_alive'] & ~df['de_depr']]
        )

        df.loc[onset_depression.loc[onset_depression].index, 'de_depr'] = True
        df.loc[onset_depression.loc[onset_depression].index, 'de_ever_depr'] = True
        df.loc[onset_depression.loc[onset_depression].index, 'de_date_init_most_rec_depr'] = self.sim.date
        df.loc[onset_depression.loc[onset_depression].index, 'de_date_depr_resolved'] = pd.NaT

        # Set the rate of depression resolution for each person who is onset with depression
        df.loc[onset_depression.loc[onset_depression].index, 'de_intrinsic_3mo_risk_of_depr_resolution'] = \
            self.module.rng.choice(p['depr_resolution_rates'], len(onset_depression.loc[onset_depression]))

        # -----------------------------------------------------------------------------------------------------
        # Determine resolution of depression for those with depression (but not depression that has onset just now)
        resolved_depression = apply_linear_model(
            self.module.linearModels['Risk_of_Depression_Resolution_per3mo'],
            df.loc[df['is_alive'] & df['de_depr'] & ~df.index.isin(onset_depression.loc[onset_depression].index)]
        )
        df.loc[resolved_depression.loc[resolved_depression].index, 'de_depr'] = False
        df.loc[resolved_depression.loc[resolved_depression].index, 'de_date_depr_resolved'] = self.sim.date
        df.loc[resolved_depression.loc[resolved_depression].index, 'de_intrinsic_3mo_risk_of_depr_resolution'] = np.nan

        # -----------------------------------------------------------------------------------------------------
        # Determine cessation of use of antidepressants among those who are currently taking them.
        stop_using_antidepressants = apply_linear_model(
            self.module.linearModels['Risk_of_Stopping_Antidepressants_per3mo'],
            df.loc[df['is_alive'] & df['de_on_antidepr']]
        )
        df.loc[stop_using_antidepressants.loc[stop_using_antidepressants].index, 'de_on_antidepr'] = False

        # -----------------------------------------------------------------------------------------------------
        # Schedule Self-harm events for those with current depression (individual level events)
        will_self_harm_in_next_3mo = apply_linear_model(
            self.module.linearModels['Risk_of_SelfHarm_per3mo'],
            df.loc[df['is_alive'] & df['de_depr']]
        )
        for person_id in will_self_harm_in_next_3mo.loc[will_self_harm_in_next_3mo].index:
            self.sim.schedule_event(DepressionSelfHarmEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))

        # Schedule Suicide events for those with current depression (individual level events)
        will_suicide_in_next_3mo = apply_linear_model(
            self.module.linearModels['Risk_of_Suicide_per3mo'],
            df.loc[df['is_alive'] & df['de_depr']]
        )
        for person_id in will_suicide_in_next_3mo.loc[will_suicide_in_next_3mo].index:
            self.sim.schedule_event(DepressionSuicideEvent(self.module, person_id),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))


class DepressionSelfHarmEvent(Event, IndividualScopeEventMixin):
    """
    This is a Self-Harm event. It has been scheduled to occur by the DepressionPollingEvent.
    It imposes the Injuries_From_Self_Harm symptom, which will lead to emergency care being sought
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        self.module.eventsTracker['SelfHarmEvents'] += 1
        self.sim.population.props.at[person_id, 'de_ever_non_fatal_self_harm_event'] = True

        # Add the outward symptom to the SymptomManager. This will result in emergency care being sought
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='Injuries_From_Self_Harm'
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

        self.module.eventsTracker['SuicideEvents'] += 1
        self.sim.modules['Demography'].do_death(
            individual_id=person_id,
            cause='Suicide',
            originating_module=self.module)


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class DepressionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This is the LoggingEvent for Depression. It runs every 3 months and gives:
    * population summaries for statuses for Depression at that time.
    * counts of events of self-harm and suicide that have occurred in the 3 months prior
    """

    def __init__(self, module):
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props

        # 1) Produce summary statistics for the current states
        # Population totals
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

        # Proportion of ever depressed who have self harmed
        n_ever_self_harmed = (df.is_alive & df.de_ever_non_fatal_self_harm_event).sum()

        # Numbers experiencing interventions
        n_ever_diagnosed_depression = (df.is_alive & df.de_ever_diagnosed_depression & (df.age_years >= 15)).sum()
        n_antidepr_depr = (df.is_alive & df.de_on_antidepr & df.de_depr & (df.age_years >= 15)).sum()
        n_antidepr_ever_depr = (df.is_alive & df.de_on_antidepr & df.de_ever_depr & (df.age_years >= 15)).sum()
        n_ever_talk_ther = (df.de_ever_talk_ther & df.is_alive & df.de_depr).sum()

        def zero_out_nan(x):
            return x if not np.isnan(x) else 0

        dict_for_output = {
            'prop_ge15_depr': zero_out_nan(n_ge15_depr / n_ge15),
            'prop_ge15_m_depr': zero_out_nan(n_ge15_m_depr / n_ge15_m),
            'prop_ge15_f_depr': zero_out_nan(n_ge15_f_depr / n_ge15_f),
            'prop_ever_depr': zero_out_nan(n_ever_depr / n_ge15),
            'prop_age_50_ever_depr': zero_out_nan(n_age_50_ever_depr / n_age_50),
            'p_ever_diagnosed_depression_if_ever_depressed': zero_out_nan(n_ever_diagnosed_depression / n_ever_depr),
            'prop_antidepr_if_curr_depr': zero_out_nan(n_antidepr_depr / n_ge15_depr),
            'prop_antidepr_if_ever_depr': zero_out_nan(n_antidepr_ever_depr / n_ever_depr),
            'prop_ever_talk_ther_if_ever_depr': zero_out_nan(n_ever_talk_ther / n_ever_depr),
            'prop_ever_self_harmed': zero_out_nan(n_ever_self_harmed / n_ever_depr),
        }

        logger.info(key='summary_stats', data=dict_for_output)

        # 2) Log number of Self-Harm and Suicide Events since the last logging event
        logger.info(key='event_counts',
                    data={'SelfHarmEvents': self.module.eventsTracker['SelfHarmEvents'],
                          'SuicideEvents': self.module.eventsTracker['SuicideEvents'],
                          })

        # Reset the EventTracker
        self.module.eventsTracker = {'SelfHarmEvents': 0, 'SuicideEvents': 0}


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_Depression_TalkingTherapy(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event in which a person receives a session of talking therapy. It is one
    of a course of 5 sessions (at months 0, 6, 12, 18, 24). If one of these HSI does not happen
    then no further sessions occur. Sessions after the first have no direct effect, as the only property affected is
    reflects ever having had one session of talking therapy."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Depression_TalkingTherapy'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MentOPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.num_of_sessions_had = 0  # A counter for the number of sessions of talking therapy had

    def apply(self, person_id, squeeze_factor):
        """Set the property `de_ever_talk_ther` to be True and schedule the next session in the course if the person
        has not yet had 5 sessions."""

        self.num_of_sessions_had += 1

        df = self.sim.population.props
        if not df.at[person_id, 'de_ever_talk_ther']:
            df.at[person_id, 'de_ever_talk_ther'] = True

        if self.num_of_sessions_had < 5:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=self,
                topen=self.sim.date + pd.DateOffset(months=6),
                tclose=None,
                priority=1
            )


class HSI_Depression_Start_Antidepressant(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event in which a person is started on anti-depressants.
    The facility_level is modified as a input parameter.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Depression_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MentOPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # If person is already on anti-depressants, do not do anything
        if df.at[person_id, 'de_on_antidepr']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        assert df.at[person_id, 'de_ever_diagnosed_depression'], "The person is not diagnosed and so should not be " \
                                                                 "receiving an HSI. "

        # Check availability of antidepressant medication
        item_code = self.module.parameters['anti_depressant_medication_item_code']

        if self.get_consumables(item_codes=item_code):
            # If medication is available, flag as being on antidepressants
            df.at[person_id, 'de_on_antidepr'] = True

            # Schedule their next HSI for a refill of medication in one month
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Depression_Refill_Antidepressant(person_id=person_id, module=self.module),
                priority=1,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(days=7)
            )


class HSI_Depression_Refill_Antidepressant(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event in which a person seeks a refill prescription of anti-depressants.
    The next refill of anti-depressants is also scheduled.
    If the person is flagged as not being on antidepressants, then the event does nothing and returns a blank footprint.
    If it does not run, then person ceases to be on anti-depressants and no further refill HSI are scheduled.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Depression_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        assert df.at[person_id, 'de_ever_diagnosed_depression'], "The person is not diagnosed and so should not be " \
                                                                 "receiving an HSI. "

        # Check that the person is on anti-depressants
        if not df.at[person_id, 'de_on_antidepr']:
            # This person is not on anti-depressants so will not have this HSI
            # Return the blank_appt_footprint() so that this HSI does not occupy any time resources
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Check availability of antidepressant medication
        if self.get_consumables(self.module.parameters['anti_depressant_medication_item_code']):
            # Schedule their next HSI for a refill of medication, one month from now
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Depression_Refill_Antidepressant(person_id=person_id, module=self.module),
                priority=1,
                topen=self.sim.date + DateOffset(months=1),
                tclose=self.sim.date + DateOffset(months=1) + DateOffset(days=7)
            )
        else:
            # If medication was not available, the persons ceases to be taking antidepressants
            df.at[person_id, 'de_on_antidepr'] = False

    def did_not_run(self):
        # If this HSI event did not run, then the persons ceases to be taking antidepressants
        person_id = self.target
        self.sim.population.props.at[person_id, 'de_on_antidepr'] = False


# ---------------------------------------------------------------------------------------------------------
#   HELPER FUNCTIONS
# ---------------------------------------------------------------------------------------------------------

# %%  Compute Key Outputs
def compute_key_outputs_for_last_3_years(parsed_output):
    """

    Helper function that computes the key outputs for the Depression Module. These are computed as averages for the
    last 3 years of the simulation.

    :param parsed_output: The parsed_output returned from `parse_log_file`
    :return: dict containting the key outputs

    """

    depr = parsed_output['tlo.methods.depression']['summary_stats']
    depr.date = (pd.to_datetime(depr['date']))

    # define the period of interest for averages to be the last 3 years of the simulation
    period = (max(depr.date) - pd.DateOffset(years=3)) < depr['date']

    result = dict()

    # Overall prevalence of current moderate/severe depression in people aged 15+
    # (Note that only severe depressions are modelled)

    result['Current prevalence of depression, aged 15+'] = depr.loc[period, 'prop_ge15_depr'].mean()

    result['Current prevalence of depression, aged 15+ males'] = depr.loc[period, 'prop_ge15_m_depr'].mean()

    result['Current prevalence of depression, aged 15+ females'] = depr.loc[period, 'prop_ge15_f_depr'].mean()

    # Ever depression in people age 50:
    result['Ever depression, aged 50y'] = depr.loc[period, 'prop_age_50_ever_depr'].mean()

    # Prevalence of antidepressant use amongst age 15+ year olds ever depressed
    result['Proportion of 15+ ever depressed using anti-depressants, aged 15+y'] = depr.loc[
        period, 'prop_antidepr_if_ever_depr'].mean()

    # Prevalence of antidepressant use amongst people currently depressed
    result['Proportion of 15+ currently depressed using anti-depressants, aged 15+y'] = depr.loc[
        period, 'prop_antidepr_if_curr_depr'].mean()

    # Proportion of those persons ever depressed that have ever been diagnosed
    result['Proportion of those persons ever depressed that have ever been diagnosed'] = \
        depr.loc[period, 'p_ever_diagnosed_depression_if_ever_depressed'].mean()

    # Process the event outputs from the model
    depr_events = parsed_output['tlo.methods.depression']['event_counts']
    depr_events['year'] = pd.to_datetime(depr_events['date']).dt.year
    depr_events = depr_events.groupby(by='year')[['SelfHarmEvents', 'SuicideEvents']].sum()

    # Get population sizes for the
    def get_15plus_pop_by_year(df):
        df = df.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year
        df.drop(columns='date', inplace=True)
        df.set_index('year', drop=True, inplace=True)
        cols_for_15plus = [int(x[0]) >= 15 for x in df.columns.str.strip('+').str.split('-')]
        return df[df.columns[cols_for_15plus]].sum(axis=1)

    tot_pop = \
        get_15plus_pop_by_year(
            parsed_output['tlo.methods.demography']['age_range_m']
        ) + \
        get_15plus_pop_by_year(
            parsed_output['tlo.methods.demography']['age_range_f']
        )

    depr_event_rate = depr_events.div(tot_pop, axis=0)

    # Rate of serious non fatal self harm incidents per 100,000 adults age 15+ per year
    result['Rate of non-fatal self-harm incidence per 100k persons aged 15+'] = 1e5 * depr_event_rate[
        'SelfHarmEvents'].mean()

    # Rate of suicide per 100,000 adults age 15+ per year
    result['Rate of suicide incidence per 100k persons aged 15+'] = 1e5 * depr_event_rate[
        'SuicideEvents'].mean()

    return result
