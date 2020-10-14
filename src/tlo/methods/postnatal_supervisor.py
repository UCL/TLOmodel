from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PostnatalSupervisor(Module):
    """"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.postnatal_tracker = dict()

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM}  # declare that this is a disease module (leave as empty set otherwise)

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
        'prob_attend_pnc2': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC1 care will return for PNC2 care'),
        'prob_attend_pnc3': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC2 care will return for PNC3 care'),
        'postpartum_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for postpartum sepsis'),
        'secondary_pph_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for secondary pph'),
        'neonatal_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for neonatal sepsis'),
    }

    PROPERTIES = {
        'pn_id_most_recent_child': Property(Types.INT, 'person_id of a mothers most recent child'),
        'pn_postnatal_period_in_weeks': Property(Types.INT, 'The number of weeks a woman is in the postnatal period '
                                                            '(1-6)'),
        'pn_pnc_visits_maternal': Property(Types.INT, 'The number of postnatal care visits a woman has undergone '
                                                      'following her most recent delivery'),
        'pn_pnc_visits_neonatal': Property(Types.INT, 'The number of postnatal care visits a neonate has undergone '
                                                      'following delivery'),
        'pn_postpartum_haem_secondary': Property(Types.BOOL, 'Whether this woman is experiencing a secondary '
                                                             'postpartum haemorrhage'),
        'pn_postpartum_haem_secondary_severity': Property(Types.CATEGORICAL, 'severity of a womans secondary PPH ',
                                                          categories=['none', 'mild', 'moderate', 'severe']),
        'pn_postpartum_haem_secondary_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                       'secondary PPH'),
        'pn_sepsis_late_postpartum': Property(Types.BOOL, 'Whether this woman is experiencing postnatal (day7+) '
                                                          'sepsis'),
        'pn_sepsis_late_postpartum_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                    'postpartum sepsis'),
        'pn_vesicovaginal_fistula': Property(Types.BOOL, 'Whether this woman has developed an obstetric fistula '
                                                         'following childbirth'),
        'pn_sepsis_late_neonatal': Property(Types.BOOL, 'Whether this neonate has developed late neonatal sepsis '
                                                        'following discharge'),
        'pn_sepsis_late_neonatal_treatment': Property(Types.BOOL, 'Whether this neonate has received treatment for '
                                                                  'neonatal sepsis'),
        'pn_anaemia_in_postpartum_period': Property(Types.BOOL, 'Whether this woman has developed anaemia following '
                                                                'birth'),
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
                params['cfr_secondary_pph'],
                Predictor('pn_postpartum_haem_secondary_treatment').when(True, params[
                    'secondary_pph_treatment_effect'])),

            'postnatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_postnatal_sepsis'],
                Predictor('pn_sepsis_late_postpartum_treatment').when(True, params[
                    'postpartum_sepsis_treatment_effect'])),

            'obstetric_fistula': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_obstetric_fistula']),

            'postpartum_anaemia': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['weekly_prob_postpartum_anaemia']),

            'late_neonatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_late_neonatal_sepsis'],
                Predictor('pn_sepsis_late_neonatal_treatment').when(True, params[
                    'neonatal_sepsis_treatment_effect'])),

        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'pn_id_most_recent_child'] = -1
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_maternal'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_neonatal'] = 0
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum_treatment'] = False
        df.loc[df.is_alive, 'pn_vesicovaginal_fistula'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal_treatment'] = False
        df.loc[df.is_alive, 'pn_anaemia_in_postpartum_period'] = False
        df.loc[df.is_alive, 'pn_obstetric_fistula'] = False

    def initialise_simulation(self, sim):
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.postnatal_tracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                 'postnatal_sepsis': 0, 'sepsis_death': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                 'late_neonatal_sepsis': 0, 'neonatal_death': 0, 'neonatal_sepsis_death': 0}

        # Register dx_tests used as assessment for postnatal conditions
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assessment_for_postnatal_sepsis=DxTest(
                property='pn_sepsis_late_postpartum',
                sensitivity=0.99),

            assessment_for_secondary_pph=DxTest(
                property='pn_postpartum_haem_secondary',
                sensitivity=0.99),

            assessment_for_hypertension=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=0.99),

            assessment_for_neonatal_sepsis=DxTest(
                property='pn_sepsis_late_neonatal',
                sensitivity=0.99))

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters

        df.at[child_id, 'pn_id_most_recent_child'] = -1
        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_pnc_visits_maternal'] = 0
        df.at[child_id, 'pn_pnc_visits_neonatal'] = 0
        df.at[child_id, 'pn_postpartum_haem_secondary'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.at[child_id, 'pn_sepsis_late_postpartum'] = False
        df.at[child_id, 'pn_sepsis_late_postpartum_treatment'] = False
        df.at[child_id, 'pn_obstetric_fistula'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal_treatment'] = False
        df.at[child_id, 'pn_anaemia_in_postpartum_period'] = False

        # We store the ID number of the child this woman has just delivered
        df.at[mother_id, 'pn_id_most_recent_child'] = child_id

        # Here we determine if, following childbirth, this woman will develop a fistula
        risk_of_fistula = params['pn_linear_equations'][
            'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

        if self.rng.random_sample() < risk_of_fistula:
            df.at[mother_id, 'pn_obstetric_fistula'] = True
            self.postnatal_tracker['fistula'] += 1
            # todo: should treatment be the only thing that turns this variable off/ or self resolution?

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props

        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')


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
        df.loc[positive_index, 'pn_anaemia_in_postpartum_period'] = True
        self.postnatal_tracker['postnatal_anaemia'] += len(positive_index)

    def maternal_postnatal_care_contact_intervention_bundle(self, individual_id, hsi_event):
        """This function is called by each of the postnatal care visits. Currently it the interventions include
        assessment for sepsis, secondary pph and hypertension. If these are detected women are admitted for treatment"""

        # --------------------------MATERNAL ASSESSMENT AND TREATMENT ---------------------------------------------
        # SEPSIS
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_postnatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with postpartum '    
                                             f'sepsis, she will be admitted for treatment')

            sepsis_treatment = HSI_PostnatalSupervisor_InpatientCareForMaternalSepsis(
                                                        self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        # HAEMORRHAGE
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_secondary_pph',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with secondary '
                                             f'postpartum haemorrhage hypertension, she will be admitted for '
                                             f'treatment')
            haemorrhage_treatment = HSI_PostnatalSupervisor_InpatientCareForSecondaryPostpartumHaemorrhage(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(haemorrhage_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        # HYPERTENSION
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_hypertension',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message',
                         data=f'Mother {individual_id} has been assessed and diagnosed with postpartum '
                         f'hypertension, she will be admitted for treatment')

            hypertension_treatment = HSI_PostnatalSupervisor_InpatientCareForPostnatalHypertension(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(hypertension_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def neonatal_postnatal_care_contact_intervention_bundle(self, individual_id, hsi_event):
        """This function is called by each of the postnatal care visits. Currently it the interventions include
        assessment for neonatal sepsis. If detected, neonates are admitted for treatment"""
        # ------------------------------ESSENTIAL NEWBORN CARE (IF HOME BIRTH & VISIT ONE)--------------------------

        # Tetracycline
        # Cord care
        # Vit k
        # starting breast feeding

        # TODO: how will we apply the effect of late essential interventions if we have already applied sepsis risk

        # --------------------------NEONATAL ASSESSMENT AND TREATMENT ---------------------------------------------
        # SEPSIS
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_neonatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with neonatal '
            f'sepsis, they will be admitted for treatment')

            sepsis_treatment = HSI_PostnatalSupervisor_InpatientCareForNeonatalSepsis(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def maternal_postnatal_care_care_seeking(self, individual_id, recommended_day_next_pnc, next_pnc_visit,
                                             maternal_pnc):
        df = self.sim.population.props
        params = self.parameters

        ppp_in_days = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']
        days_calc = pd.to_timedelta(recommended_day_next_pnc, unit='D') - ppp_in_days
        date_next_pnc = self.sim.date + days_calc

        if self.rng.random_sample() < params[f'prob_attend_{next_pnc_visit}']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(maternal_pnc,
                                                                priority=0,
                                                                topen=date_next_pnc,
                                                                tclose=date_next_pnc + DateOffset(days=3))


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
        df.loc[week_7_postnatal_women, 'pn_pnc_visits_maternal'] = 0
        df.loc[week_7_postnatal_women, 'la_is_postpartum'] = False

        df.loc[week_7_postnatal_women, 'ps_htn_disorders'] = 'none'
        df.loc[week_7_postnatal_women, 'ps_gestational_htn'] = False
        df.loc[week_7_postnatal_women, 'ps_mild_pre_eclamp'] = False
        df.loc[week_7_postnatal_women, 'ps_severe_pre_eclamp'] = False
        df.loc[week_7_postnatal_women, 'ps_gestational_htn'] = False
        df.loc[week_7_postnatal_women, 'ps_anaemia_in_pregnancy'] = False


class SecondaryPostpartumHaemorrhageOnsetEvent(Event, IndividualScopeEventMixin):
    """This is SecondaryPostpartumHaemorrhageOnsetEvent. It is scheduled from the Labour module for women who will
    develop a postpartum haemorrhage during the postpartum period, after discharge. Currently this event applies
    severity of bleeding and schedules the LatePostpartumDeathEvent. It is unfinished."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        assert df.at[individual_id, 'la_is_postpartum']

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
            self.module.postnatal_tracker['secondary_pph'] += 1

            # Set the severity
            severity = ['mild', 'moderate', 'severe']
            probabilities = params['prob_secondary_pph_severity']
            severity_draw = self.module.rng.choice(severity, p=probabilities, size=1)
            df.at[individual_id, 'pn_postpartum_haem_secondary_severity'] = severity_draw

            # TODO: care seeking

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='pph'),
                                    (self.sim.date + pd.Timedelta(days=4)))


class LatePostpartumSepsisOnsetEvent(Event, IndividualScopeEventMixin):
    """This is LatePostpartumSepsisOnsetEvent. It is scheduled from the Labour module for women who will
    develop a postpartum sepsis during the postpartum period, after discharge. Currently this event makes changes to the
     dataframe and schedules the LatePostpartumDeathEvent. It is unfinished."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        assert df.at[individual_id, 'la_is_postpartum']

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
            self.module.postnatal_tracker['postnatal_sepsis'] += 1

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='sepsis'),
                                    (self.sim.date + pd.Timedelta(days=4)))


class LateNeonatalSepsisOnsetEvent(Event, IndividualScopeEventMixin):
    """This is LateNeonatalSepsisOnsetEvent. It is scheduled from the Newborn outcomes module for newborns who will
    develop sepsis during the postpartum period, after discharge. Currently this event makes changes to the
     data frame and schedules the LatePostpartumDeathEvent. It is unfinished"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'pn_sepsis_late_neonatal'] = True
            self.module.postnatal_tracker['late_neonatal_sepsis'] += 1

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id, cause='neonatal_sepsis'),
                                    (self.sim.date + pd.Timedelta(days=4)))


class LatePostpartumDeathEvent(Event, IndividualScopeEventMixin):
    """This is LatePostpartumDeathEvent. It is scheduled from SecondaryPostpartumHaemorrhageOnsetEvent,
    LatePostpartumSepsisOnsetEvent or LateNeonatalSepsisOnsetEvent for women and newborns who have developed
    complications in the postpartum period. It uses the linear model to calculte risk of death and schedules death in
    that instance"""

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

                    self.module.postnatal_tracker['postnatal_death'] += 1
                    self.module.postnatal_tracker['secondary_pph_death'] += 1

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

                    self.module.postnatal_tracker['postnatal_death'] += 1
                    self.module.postnatal_tracker['sepsis_death'] += 1

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

                    self.module.postnatal_tracker['neonatal_death'] += 1
                    self.module.postnatal_tracker['neonatal_sepsis_death'] += 1

                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False


class HSI_PostnatalSupervisor_PostnatalCareContactOne(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal. It is scheduled by
    HSI_Labour_ReceivesCareForPostpartumPeriod or PostpartumLabourAtHomeEvent. This event is the first PNC visit women
    are reccomended to undertake. If women deliver at home this should be within 12 hours, if in a facility then before
     48 hours. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactOneMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        # TODO: currently storing treatment and assessment of sepsis/pph/htn within this HSI but should mya

        assert df.at[person_id, 'la_is_postpartum']
        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 0
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 0

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC1 on date'
                                             f' {self.sim.date}')

            maternal_pnc = HSI_PostnatalSupervisor_PostnatalCareContactTwo(
                                                             self.module, person_id=person_id)

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)
            self.module.maternal_postnatal_care_care_seeking(person_id, 7, 'pnc2', maternal_pnc)


class HSI_PostnatalSupervisor_PostnatalCareContactTwo(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_PostnatalSupervisor_PostnatalCareContactTwoMaternal. It is scheduled by
    HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal This event is the second PNC visit women
    are recommended to undertake around week 1. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactTwoMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 1
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 1
        assert df.at[person_id, 'la_is_postpartum']

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC2 on date'
                                             f' {self.sim.date}')

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            maternal_pnc = HSI_PostnatalSupervisor_PostnatalCareContactThree(
                self.module, person_id=person_id)

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)
            self.module.maternal_postnatal_care_care_seeking(person_id, 42, 'pnc3', maternal_pnc)


class HSI_PostnatalSupervisor_PostnatalCareContactThree(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_PostnatalSupervisor_PostnatalCareContactThreeMaternal. It is scheduled by
    HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal This event is the third PNC visit women
    are recommended to undertake around week 6. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactThreeMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 2
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 2
        assert df.at[person_id, 'la_is_postpartum']

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC3 on date'
                                             f' {self.sim.date}')

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)


class HSI_PostnatalSupervisor_InpatientCareForMaternalSepsis(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_InpatientCareForMaternalSepsis. It is scheduled by any of the PNC HSIs for women
    who are assessed as being septic and require treatment as an inpatient"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_InpatientCareForMaternalSepsis'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1}) # TODO: how many days?
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # First check the availability of consumables for treatment
        pkg_code_sepsis = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed_sepsis,
                to_log=False)

        # If available then treatment is delivered and they are logged
        if outcome_of_request_for_consumables_sep:
            logger.debug(key='message', data=f'mother {person_id} has received treatment for sepsis as an inpatient')

            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed_sepsis,
                to_log=True)

            df.at[person_id, 'pn_sepsis_late_postpartum_treatment'] = True

        else:
            logger.debug(key='message', data=f'mother {person_id} was unable to receive treatment for sepsis due to '
                                             f'limited resources')


class HSI_PostnatalSupervisor_InpatientCareForSecondaryPostpartumHaemorrhage(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_InpatientCareForSecondaryPostpartumHaemorrhage. It is scheduled by any of the
    PNC HSIs for women assessed as having experience secondary PPH and require inpatient treatment. """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_InpatientCareForSecondaryPostpartumHaemorrhage'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})  # TODO: how many days
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # First check the availability of consumables for treatment
        pkg_code_pph = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum '
                                                                                    'hemorrhage',
                                                 'Intervention_Pkg_Code'])[0]

        consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_needed_pph,
                    to_log=False)

        # If available then treatment is delivered and they are logged
        if outcome_of_request_for_consumables_pph:
            logger.debug(key='message', data=f'mother {person_id} has received treatment for secondary postpartum '
                                             f'haemorrhage as an inpatient')
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed_pph,
                to_log=True)

            df.at[person_id, 'pn_postpartum_haem_secondary_treatment'] = True

        else:
            logger.debug(key='message', data=f'mother {person_id} was unable to receive treatment for secondary pph due '
                                             f'to limited resources')


class HSI_PostnatalSupervisor_InpatientCareForPostnatalHypertension(HSI_Event, IndividualScopeEventMixin):
    """ """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_InpatientCareForPostnatalHypertension'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # TODO: Is blood pressure monitoring part of PNC? - Maybe assume women only get treatment if they seek care for
        #  emergency hypertensive event - how is death applied?
        # TODO: What if women are already on treatment?


class HSI_PostnatalSupervisor_InpatientCareForNeonatalSepsis(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_InpatientCareForNeonatalSepsis. It is scheduled by any of the PNC HSIs for
    neonates who are assessed as being septic and require treatment as an inpatient"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_InpatientCareForNeonatalSepsis'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})  # TODO: how many days?
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_sep = pd.unique(consumables.loc[
                                     consumables['Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                     'Intervention_Pkg_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_needed,
            to_log=False)

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed,
                to_log=True)

            df.at[person_id, 'pn_sepsis_late_neonatal_treatment'] = True

        else:
            logger.debug(key='message', data=f'neonate {person_id} was unable to receive treatment for sepsis due to '
                                             f'limited resources')

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

        total_pph = self.module.postnatal_tracker['secondary_pph']
        total_pn_death = self.module.postnatal_tracker['postnatal_death']
        total_pph_death = self.module.postnatal_tracker['secondary_pph_death']
        total_sepsis = self.module.postnatal_tracker['postnatal_sepsis']
        total_sepsis_death = self.module.postnatal_tracker['sepsis_death']

        dict_for_output = {'total_pph': total_pph,
                           'total_deaths': total_pn_death,
                           'total_pph_death': total_pph_death,
                           'total_sepsis': total_sepsis,
                           'total_sepsis_death': total_sepsis_death}

        logger.info(key='postnatal_summary_stats', data=dict_for_output, description= 'Yearly summary statistics '
                                                                                      'output from the postnatal '
                                                                                      'supervisor module')

        self.module.postnatal_tracker = {'secondary_pph': 0, 'postnatal_death': 0, 'secondary_pph_death': 0,
                                        'postnatal_sepsis': 0, 'sepsis_death': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                        'late_neonatal_sepsis': 0, 'neonatal_death': 0, 'neonatal_sepsis_death': 0}
