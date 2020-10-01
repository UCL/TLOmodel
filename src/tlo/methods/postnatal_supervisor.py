from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event

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

        'cfr_secondary_pph': Parameter(
            Types.REAL, 'case fatality rate for secondary pph'),
        'cfr_postnatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for postnatal sepsis'),
        'prob_secondary_pph_severity': Parameter(
            Types.LIST, 'probability of mild, moderate or severe secondary PPH'),
        'prob_obstetric_fistula': Parameter(
            Types.REAL, 'probability of a woman developing an obstetric fistula after birth'),
        'weekly_prob_postpartum_anaemia': Parameter(
            Types.REAL, 'Weekly probability of anaemia in pregnancy'),
        'cfr_late_neonatal_sepsis': Parameter(
            Types.REAL, 'Risk of death from late neonatal sepsis'),
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
        'pn_vesicovaginal_fistula': Property(Types.BOOL, 'Whether this woman has developed an obstetric fistula '
                                                         'following childbirth'),
        'pn_sepsis_late_neonatal': Property(Types.BOOL, 'Whether this neonate has developed late neonatal sepsis '
                                                        'following discharge')

    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    #    if 'HealthBurden' in self.sim.modules.keys():
    #        params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)


    # ==================================== LINEAR MODEL EQUATIONS =====================================================

        params['pn_linear_equations'] = {
            'secondary_postpartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_secondary_pph']),

            'postnatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_postnatal_sepsis']),

            'obstetric_fistula': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_obstetric_fistula']),

            'postpartum_anaemia': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['weekly_prob_postpartum_anaemia']),

            'late_neonatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_late_neonatal_sepsis']),

        }

    def initialise_population(self, population):

        df = population.props
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum'] = False
        df.loc[df.is_alive, 'pn_vesicovaginal_fistula'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal'] = False

    def initialise_simulation(self, sim):
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.PostnatalTracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                 'postnatal_sepsis': 0, 'sepsis_death': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                 'late_neonatal_sepsis': 0, 'neonatal_death': 0, 'neonatal_sepsis_death': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters

        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_postpartum_haem_secondary'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.at[child_id, 'pn_sepsis_late_postpartum'] = False
        df.at[child_id, 'pn_obstetric_fistula'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal'] = False

        # Here we determine if, following childbirth, this woman will develop a fistula
        risk_of_fistula = params['pn_linear_equations'][
            'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

        if self.rng.random_sample() < risk_of_fistula:
            df.at[mother_id, 'pn_obstetric_fistula'] = True
            self.PostnatalTracker['fistula'] += 1
            # todo: should treatment be the only thing that turns this variable off/ or self resolution?

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props

        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')

    def set_postnatal_maternal_complications(self, index):
        """"""
        df = self.sim.population.props
        params = self.parameters

    def progression_of_hypertensive_disorders(self, index):
        """"""
        df = self.sim.population.props
        params = self.parameters

        pass

    def apply_weekly_risk_of_postnatal_anaemia(self, index):
        df = self.sim.population.props
        params = self.parameters

        result = params['pn_linear_equations']['postpartum_anaemia'].predict(index)

        random_draw = pd.Series(self.rng.random_sample(size=len(index)), index=index.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']

        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]
        df.loc[positive_index, 'ps_anaemia_in_pregnancy'] = True
        self.PostnatalTracker['postnatal_anaemia'] += len(positive_index)


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
        logger.debug(key='message', data=f'updating postnatal periods on date {self.sim.date}')

        # -------------------------------------- WEEK 1 (day 7) -------------------------------------------------------
        week_1_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 1)]
        self.module.progression_of_hypertensive_disorders(week_1_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_1_postnatal_women)

        # -------------------------------------- WEEK 2 (day 14) -------------------------------------------------------
        week_2_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none')
                                        & (df.pn_postnatal_period_in_weeks == 2)]
        self.module.progression_of_hypertensive_disorders(week_2_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_2_postnatal_women)

        # -------------------------------------- WEEK 3 (day 21) -------------------------------------------------------
        week_3_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                       (df.pn_postnatal_period_in_weeks == 3)]
        self.module.progression_of_hypertensive_disorders(week_3_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_3_postnatal_women)

        # -------------------------------------- WEEK 4 (day 28) -------------------------------------------------------
        week_4_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 4)]
        self.module.progression_of_hypertensive_disorders(week_4_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_4_postnatal_women)

        # -------------------------------------- WEEK 5 (day 35) -------------------------------------------------------
        week_5_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none') &
                                        (df.pn_postnatal_period_in_weeks == 5)]
        self.module.progression_of_hypertensive_disorders(week_5_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_5_postnatal_women)

        # -------------------------------------- WEEK 6 (day 42) -------------------------------------------------------
        week_6_postnatal_women = df.loc[df.is_alive & df.la_is_postpartum & (df.ps_htn_disorders != 'none')
                                        & (df.pn_postnatal_period_in_weeks == 6)]
        self.module.progression_of_hypertensive_disorders(week_6_postnatal_women)
        self.module.apply_weekly_risk_of_postnatal_anaemia(week_6_postnatal_women)

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
            self.module.PostnatalTracker['secondary_pph'] += 1

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

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_sepsis_late_neonatal'] = True
            self.module.PostnatalTracker['late_neonatal_sepsis'] += 1

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='neonatal_sepsis'),
                                    (self.sim.date + pd.Timedelta(days=3)))


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
                    logger.debug(key='message', data=f'person {individual_id} has died due to secondary postpartum '
                                                     f'haemorrhage on date {self.sim.date}')
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
                    logger.debug(key='message', data=f'person {individual_id} has died due to late maternal sepsis on '
                    f'date {self.sim.date}')

                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='maternal_sepsis'), self.sim.date)

                    self.module.PostnatalTracker['postnatal_death'] += 1
                    self.module.PostnatalTracker['sepsis_death'] += 1

                else:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False

            if self.cause == 'neonatal_sepsis':
                assert df.at[individual_id, 'pn_sepsis_late_neonatal']

                risk_of_death = params['pn_linear_equations']['late_neonatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    logger.debug(key='message', data=f'person {individual_id} has died due to late neonatal sepsis on '
                    f'date {self.sim.date}')

                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='neonatal_sepsis'), self.sim.date)

                    self.module.PostnatalTracker['neonatal_death'] += 1
                    self.module.PostnatalTracker['neonatal_sepsis_death'] += 1

                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False


class HSI_PostnatalSupervisor_PostnatalCareContactOne(HSI_Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactOne'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props


class HSI_PostnatalSupervisor_PostnatalCareContactTwo(HSI_Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactTwo'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props


class HSI_PostnatalSupervisor_PostnatalCareContactThree(HSI_Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactThree'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props


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

        logger.info(key='postnatal_summary_stats', data=dict_for_output, description= 'Yearly summary statistics output '
                                                                                      'from the postnatal supervisor '
                                                                                      'module')

        self.module.PostnatalTracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                        'postnatal_sepsis': 0, 'sepsis_death': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                        'late_neonatal_sepsis': 0, 'neonatal_death': 0, 'neonatal_sepsis_death': 0}
