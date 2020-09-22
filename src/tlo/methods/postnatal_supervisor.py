from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods import Metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PostnatalSupervisor(Module):
    """"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.PostnatalTracker = dict()

    METADATA = {Metadata.DISEASE_MODULE}  # declare that this is a disease module (leave as empty set otherwise)

    PARAMETERS = {
        'prob_secondary_pph_per_week': Parameter(
            Types.REAL, 'baseline weekly probability of a woman developing a secondary postpartum haemorrhage'),
        'cfr_secondary_pph': Parameter(
            Types.REAL, 'case fatality rate for secondary pph'),
        'prob_postnatal_sepsis_per_week': Parameter(
            Types.REAL, 'baseline weekly probability of a woman developing postnatal sepsis'),
        'cfr_postnatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for postnatal sepsis'),
        'prob_postnatal_gh_per_week': Parameter(
            Types.REAL, 'baseline weekly probability of a woman developing postnatal gestational hypertension'),
        'prob_postnatal_pe_per_week': Parameter(
            Types.REAL, 'baseline weekly probability of a woman developing postnatal mild pre-eclampsia'),
        'prob_secondary_pph_severity': Parameter(
            Types.LIST, 'probability of mild, moderate or severe secondary PPH'),
    }

    PROPERTIES = {
        'pn_postnatal_period_in_weeks': Property(Types.INT, 'The number of weeks a woman is in the postnatal period '
                                                            '(1-6)'),
        'pn_postpartum_haem_secondary': Property(Types.BOOL, 'Whether this woman is experiencing a secondary '
                                                             'postpartum haemorrhage'),
        'pn_postpartum_haem_secondary_severity': Property(Types.CATEGORICAL, 'severity of a womans secondary PPH ',
                                                          categories=['none', 'mild', 'moderate', 'severe']),
        'pn_sepsis_late_postpartum': Property(Types.BOOL, 'Whether this woman is experiencing postnatal (day7+) '
                                                             'sepsis'),
        'pn_postnatal_gest_htn': Property(Types.BOOL, 'Whether this woman is experiencing postnatal gestational '
                                                      'hypertension'),
        'pn_postnatal_mild_pre_eclamp': Property(Types.BOOL, 'Whether this woman is experiencing postnatal mild pre '
                                                             'eclampsia'),
    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    #    if 'HealthBurden' in self.sim.modules.keys():
    #        params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

        self.sim.modules['HealthSystem'].register_disease_module(self)

    # ==================================== LINEAR MODEL EQUATIONS =====================================================

        params['pn_linear_equations'] = {
            'secondary_pph': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_secondary_pph_per_week']),
            #    Predictor('pn_postpartum_haem_secondary_severity').when('mild', params['multiplier_death_mild_pph'])
            #                                              .when('moderate', params['multiplier_death_moderate_pph'])
            #                                              .when('severe', params['multiplier_death_severe_pph'])),

            'secondary_postpartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_secondary_pph']),

            'postnatal_maternal_sepsis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_postnatal_sepsis_per_week']),

            'postnatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_postnatal_sepsis']),

            'postnatal_new_onset_gest_htn': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_postnatal_gh_per_week']),

            'postnatal_new_onset_pre_eclamp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_postnatal_pe_per_week']),

        }

    def initialise_population(self, population):

        df = population.props
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum'] = False
        df.loc[df.is_alive, 'pn_postnatal_gest_htn'] = False
        df.loc[df.is_alive, 'pn_postnatal_mild_pre_eclamp'] = False

    def initialise_simulation(self, sim):
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.PostnatalTracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                 'postnatal_sepsis': 0, 'sepsis_death': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_postpartum_haem_secondary'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.at[child_id, 'pn_sepsis_late_postpartum'] = False
        df.at[child_id, 'pn_postnatal_gest_htn'] = False
        df.at[child_id, 'pn_postnatal_mild_pre_eclamp'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug('This is PostnatalSupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        df = self.sim.population.props

        logger.debug('This is PostnatalSupervisor reporting my health values')

    def set_postnatal_maternal_complications(self, index):
        """"""
        df = self.sim.population.props
        params = self.parameters

    def progression_of_hypertensive_disorders(self, index):
        """"""
        df = self.sim.population.props
        params = self.parameters

        gest_htn = index.ps_htn_disorders == 'gest_htn'
    #    result = params['ps_linear_equations'][f'{complication}'].predict(index)

    #    # And use the result of a random draw to determine which women will experience the complication
   #     random_draw = pd.Series(self.rng.random_sample(size=len(index)), index=index.index)
    #    temp_df = pd.concat([result, random_draw], axis=1)
    #    temp_df.columns = ['result', 'random_draw']

        # Then we use this index to make changes to the data frame and schedule any events required
    #    positive_index = temp_df.index[temp_df.random_draw < temp_df.result]
   #     if df.at[individual_id, 'ps_htn_disorders'] == 'gest_htn':
   #         risk_progression_gh_pe = params['la_labour_equations']['progression_gest_htn'].predict(df.loc[[
   #                 individual_id]])[individual_id]

    #        if risk_progression_gh_pe > self.rng.random_sample():
    #            df.at[individual_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

    #    if df.at[individual_id, 'ps_htn_disorders'] == 'mild_pre_eclamp':
    #        risk_progression_mpe_spe = params['la_labour_equations']['progression_mild_pre_eclamp'].predict(df.loc[[
    #                individual_id]])[individual_id]

    #        if risk_progression_mpe_spe > self.rng.random_sample():
    #            df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    #            self.labour_tracker['severe_pre_eclampsia'] += 1

    #    if df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
    #        risk_progression_spe_ec = params['la_labour_equations']['progression_severe_pre_eclamp'].predict(df.loc[[
     #               individual_id]])[individual_id]

    #        if risk_progression_spe_ec > self.rng.random_sample():
    #            df.at[individual_id, 'ps_htn_disorders'] = 'eclampsia'
    #            df.at[individual_id, 'la_eclampsia_disab'] = True
    #            self.labour_tracker['eclampsia'] += 1


class PostnatalSupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ """

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props

        # ================================ UPDATING LENGTH OF POSTPARTUM PERIOD  IN WEEKS  ============================
        # Here we update how far into the postpartum period each woman who has recently delivered is
        alive_and_recently_delivered = df.is_alive & df.la_is_postpartum
        ppp_in_days = self.sim.date - df.loc[alive_and_recently_delivered, 'la_date_most_recent_delivery']
        ppp_in_weeks = ppp_in_days / np.timedelta64(1, 'W')

        df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] = ppp_in_weeks.astype('int64')
        logger.debug('updating postnatal periods on date %s', self.sim.date)

        # -------------------------------------- WEEK 1 (day 7) -------------------------------------------------------
        week_1_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 1)]
        self.module.progression_of_hypertensive_disorders(week_1_postnatal_women)

        # -------------------------------------- WEEK 2 (day 14) -------------------------------------------------------
        week_2_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none')
                                        & (df.pn_postnatal_period_in_weeks == 2)]
        self.module.progression_of_hypertensive_disorders(week_2_postnatal_women)

        # -------------------------------------- WEEK 3 (day 21) -------------------------------------------------------
        week_3_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                       (df.pn_postnatal_period_in_weeks == 3)]
        self.module.progression_of_hypertensive_disorders(week_3_postnatal_women)

        # -------------------------------------- WEEK 4 (day 28) -------------------------------------------------------
        week_4_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 4)]
        self.module.progression_of_hypertensive_disorders(week_4_postnatal_women)

        # -------------------------------------- WEEK 5 (day 35) -------------------------------------------------------
        week_5_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 5)]
        self.module.progression_of_hypertensive_disorders(week_5_postnatal_women)

        # -------------------------------------- WEEK 6 (day 42) -------------------------------------------------------
        week_6_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none')
                                        & (df.pn_postnatal_period_in_weeks == 6)]
        self.module.progression_of_hypertensive_disorders(week_6_postnatal_women)

        # Here, one week after we stop applying risk of postpartum complications, we reset key postpartum variables

        week_7_postnatal_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 7)
        df.loc[week_7_postnatal_women, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[week_7_postnatal_women, 'la_is_postpartum'] = False

        df.loc[week_7_postnatal_women, 'ps_htn_disorders'] = 'none'
        df.loc[week_7_postnatal_women, 'ps_gestational_htn'] = False
        df.loc[week_7_postnatal_women, 'ps_mild_pre_eclamp'] = False
        df.loc[week_7_postnatal_women, 'ps_severe_pre_eclamp'] = False
        df.loc[week_7_postnatal_women, 'ps_gestational_htn'] = False
        df.loc[week_7_postnatal_women, 'ps_anaemia_in_pregnancy'] = False


class SecondaryPostpartumHaemorrhageOnsetEvent(Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        assert df.at[individual_id, 'la_is_postpartum']

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
            # self.module.PostnatalTracker['secondary_pph'] += 1

            # Set the severity
            severity = ['mild', 'moderate', 'severe']
            probabilities = params['prob_secondary_pph_severity']
            severity_draw = self.module.rng.choice(severity, p=probabilities, size=1)
            df.at[individual_id, 'pn_postpartum_haem_secondary_severity'] = severity_draw

            # TODO: care seeking

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='pph'),
                                    (self.sim.date + pd.Timedelta(days=3)))


class LatePostpartumSepsisOnsetEvent(Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        assert df.at[individual_id, 'la_is_postpartum']

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
            self.module.PostnatalTracker['postnatal_sepsis'] += 1

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='sepsis'),
                                    (self.sim.date + pd.Timedelta(days=3)))

class LateNeonatalSepsisOnsetEvent(Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        pass

class LatePostpartumDeathEvent(Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        if df.at[individual_id, 'is_alive']:

            if self.cause == 'pph':
                assert df.at[individual_id, 'pn_postpartum_haem_secondary']
                assert df.at[individual_id, 'pn_postpartum_haem_secondary_severity'] is not 'none'

                risk_of_death = params['pn_linear_equations']['secondary_postpartum_haem_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    logger.debug(f'person %d has died due to secondary postpartum haemorrhage on date %s', individual_id,
                                 self.sim.date)
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='secondary_pph'), self.sim.date)

                    self.module.PostnatalTracker['postnatal_death'] += 1
                    self.module.PostnatalTracker['secondary_pph_death'] += 1

                else:
                    df.at[individual_id, 'pn_postpartum_haem_secondary'] = False
                    df.at[individual_id, 'pn_postpartum_haem_secondary_severity'] = 'none'

            if self.cause == 'sepsis':
                assert df.at[individual_id, 'pn_sepsis_late_postpartum']

                risk_of_death = params['pn_linear_equations']['postnatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    logger.debug(f'person %d has died due to secondary maternal sepsis on date %s',
                                 individual_id,
                                 self.sim.date)
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='maternal_sepsis'), self.sim.date)

                    self.module.PostnatalTracker['postnatal_death'] += 1
                    self.module.PostnatalTracker['sepsis_death'] += 1

                else:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False


class PostnatalLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """"""

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        # Denominators...
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])
        if total_births_last_year == 0:
            total_births_last_year = 1

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        total_pph = self.module.PostnatalTracker['secondary_pph']
        total_pn_death = self.module.PostnatalTracker['postnatal_death']
        total_pph_death = self.module.PostnatalTracker['secondary_pph_death']
        total_sepsis = self.module.PostnatalTracker['postnatal_sepsis']
        total_sepsis_death = self.module.PostnatalTracker['sepsis_death']

        dict_for_output = {'total_pph': total_pph,
                           'total_deaths': total_pn_death,
                           'total_pph_death': total_pph_death,
                           'total_sepsis': total_sepsis,
                           'total_sepsis_death': total_sepsis_death}

        logger.info('%s|summary_stats|%s', self.sim.date, dict_for_output)

        self.module.PostnatalTracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                        'postnatal_sepsis': 0, 'sepsis_death': 0}
