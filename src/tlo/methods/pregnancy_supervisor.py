import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, miscarriage, abortion, and onset of antenatal complications. This module is
    incomplete, currently antenatal death has not been coded. Similarly antenatal care seeking will be house hear, for
    both routine treatment and in emergencies"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_pregnancy_factors': Parameter(
            Types.DATA_FRAME, 'Data frame containing probabilities of key outcomes/complications associated with the '
                              'antenatal period'),
        'base_prev_pe': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'base_prev_gest_htn': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'base_prev_gest_diab': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'rr_miscarriage_prevmiscarriage': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who have previously miscarried'),
        'rr_miscarriage_35': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is over 35 years old'),
        'rr_miscarriage_3134': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who is between 31 and 34 years old'),
        'rr_miscarriage_grav4': Parameter(
            Types.REAL, 'relative risk of miscarriage for women who has a gravidity of greater than 4'),
        'rr_pre_eclamp_nulip': Parameter(
            Types.REAL, 'relative risk of pre-eclampsia in nuliparous women'),
        'rr_pre_eclamp_prev_pe': Parameter(
            Types.REAL, 'relative risk of pre- eclampsia in women who have previous suffered from pre-eclampsia'),
        'rr_gest_diab_overweight': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who are overweight at time of pregnancy'),
        'rr_gest_diab_stillbirth': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who have previously had a still birth'),
        'rr_gest_diab_prevdiab': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who suffered from gestational diabetes in '
                        'previous pregnancy'),
        'rr_gest_diab_chron_htn': Parameter(
            Types.REAL, 'relative risk of gestational diabetes in women who suffer from chronic hypertension'),
        'prob_ectopic_pregnancy': Parameter(
            Types.REAL, 'probability that a womans current pregnancy is ectopic'),
        'level_of_symptoms_ep': Parameter(
            Types.REAL, 'Level of symptoms that the individual will have'),
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_pa_complications': Parameter(
            Types.REAL, 'probability that a woman who has had an induced abortion will experience any complications'),
        'prob_pa_complication_type': Parameter(
            Types.REAL, 'List of probabilities that determine what type of complication a woman who has had an abortion'
                        ' will experience'),
        'r_mild_pe_gest_htn': Parameter(
            Types.REAL, 'probability per month that a woman will progress from gestational hypertension to mild '
                        'pre-eclampsia'),
        'r_severe_pe_mild_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from mild pre-eclampsia to severe '
                        'pre-eclampsia'),
        'r_eclampsia_severe_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from severe pre-eclampsia to eclampsia'),
        'r_hellp_severe_pe': Parameter(
            Types.REAL, 'probability per month that a woman will progress from severe pre-eclampsia to HELLP syndrome'),
    }

    PROPERTIES = {
        'ps_gestational_age_in_weeks': Property(Types.INT, 'current gestational age, in weeks, of this womans '
                                                           'pregnancy'),
        'ps_ectopic_pregnancy': Property(Types.BOOL, 'Whether this womans pregnancy is ectopic'),
        'ps_ectopic_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for ectopic pregnancy',
                                        categories=['none', 'abdominal pain', 'abdominal pain plus bleeding', 'shock']),
        'ps_ep_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4]),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'ps_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'ps_abortion_complication': Property(Types.CATEGORICAL, 'Type of complication following an induced abortion: '
                                                                'None; Sepsis; Haemorrhage; Sepsis and Haemorrhage',
                                             categories=['none', 'haem', 'sepsis', 'haem_sepsis']),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
        'ps_htn_disorder_preg': Property(Types.CATEGORICAL,  'Hypertensive disorders of pregnancy: none, '
                                                             'gestational hypertension, mild pre-eclampsia,'
                                                             'severe pre-eclampsia, eclampsia,'
                                                             ' HELLP syndrome',
                                         categories=['none', 'gest_htn', 'mild_pe', 'severe_pe', 'eclampsia', 'HELLP']),
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy')
    }

    def read_parameters(self, data_folder):
        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # These parameters presently hard coded for ease. Both may be deleted following epi review.
        params['level_of_symptoms_ep'] = pd.DataFrame(
            data={'level_of_symptoms_ep': ['none',
                                           'abdominal pain',
                                           'abdominal pain plus bleeding',
                                           'shock'], 'probability': [0.25, 0.25, 0.25, 0.25]})  # DUMMY

        params['type_pa_complication'] = pd.DataFrame(
            data={'type_pa_complication': ['haem',
                                           'sepsis',
                                           'haem_sepsis'], 'probability': [0.5, 0.3, 0.2]})

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

# ==================================== LINEAR MODEL EQUATIONS ==========================================================
        # All linear equations used in this module are stored within the ps_linear_equations parameter below

        params['ps_linear_equations'] = {
            'ectopic': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_ectopic_pregnancy'],
             Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.0).when('Southern', 1.0)),
            # DUMMY

            'multiples': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_ectopic_pregnancy'],
             Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.0).when('Southern', 1.0)),
            # DUMMY

            'miscarriage':LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('4', 1.0).when('8', 1.1).when('13', 0.8).when('17', 0.8)
                .when('22', 0.8),
             Predictor('ps_total_miscarriages').when(' >1 ', params['rr_miscarriage_prevmiscarriage']),
             Predictor('age_years').when('.between(30,35)', params['rr_miscarriage_3134']).when(' > 34',
                                                                                                params
                                                                                                ['rr_miscarriage_35'])),
            'abortion': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('8', 1.1).when('13', 0.8).when('17', 0.8).when('22', 0.8)),

            'pre_eclampsia': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('22', 1.1).when('27', 0.8).when('31', 0.8).when('35', 0.8)
                .when('40', 0.8).when('46', 0.8),
             Predictor('la_parity').when('0', params['rr_pre_eclamp_nulip']),
             Predictor('ps_prev_pre_eclamp').when(True , params['rr_pre_eclamp_prev_pe'])),

            'gest_htn': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('22', 1.1).when('27', 0.8).when('31', 0.8).when('35', 0.8)
                .when('40', 0.8).when('46', 0.8)),

            'gest_diab': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('22', 1.1).when('27', 0.8).when('31', 0.8).when('35', 0.8)
                .when('40', 0.8).when('46', 0.8),
             Predictor('ps_previous_stillbirth').when(True, params['rr_gest_diab_stillbirth']),
             Predictor('ps_prev_gest_diab').when(True, params['rr_gest_diab_prevdiab'])),

            'stillbirth':LinearModel(
             LinearModelType.MULTIPLICATIVE,
             0.02,  # DUMMY VALUE- doesnt exist as parameter
             Predictor('ps_gestational_age_in_weeks').when('27', 1.0).when('31', 1.1).when('35', 0.8).when('40', 0.8)
                .when('46', 0.8))}

    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.sex == 'F', 'ps_ectopic_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_ectopic_symptoms'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_ep_unified_symptom_code'] = 0
        df.loc[df.sex == 'F', 'ps_multiple_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'ps_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'ps_abortion_complication'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_antepartum_still_birth'] = False
        df.loc[df.sex == 'F', 'ps_previous_stillbirth'] = False
        df.loc[df.sex == 'F', 'ps_htn_disorder_preg'].values[:] = 'none'
        df.loc[df.sex == 'F', 'ps_prev_pre_eclamp'] = False
        df.loc[df.sex == 'F', 'ps_gest_diab'] = False
        df.loc[df.sex == 'F', 'ps_prev_gest_diab'] = False

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancyDiseaseProgressionEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancySupervisorLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

#        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[child_id, 'ps_ectopic_pregnancy'] = False
            df.at[child_id, 'ps_ectopic_symptoms'] = 'none'
            df.at[child_id, 'ps_ep_unified_symptom_code'] = 0
            df.at[child_id, 'ps_multiple_pregnancy'] = False
            df.at[child_id, 'ps_total_miscarriages'] = 0
            df.at[child_id, 'ps_total_induced_abortion'] = 0
            df.at[child_id, 'ps_abortion_complication'] = 'none'
            df.at[child_id, 'ps_antepartum_still_birth'] = False
            df.at[child_id, 'ps_previous_stillbirth'] = False
            df.at[child_id, 'ps_htn_disorder_preg'] = 'none'
            df.at[child_id, 'ps_prev_pre_eclamp'] = False
            df.at[child_id, 'ps_gest_diab'] = False
            df.at[child_id, 'ps_prev_gest_diab'] = False

    def on_hsi_alert(self, person_id, treatment_id):

        logger.debug('This is PregnancySupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):

        df = self.sim.population.props

    # TODO: Antenatal DALYs

    def set_pregnancy_complications(self, params, index, complication):
        """This function is called from within the PregnancySupervisorEvent. It calculates risk of a number of pregnancy
        outcomes/ complications for pregnant women in the data frame using the linear model equations defined above.
        Properties are modified depending on the  complication passed to the function and the result of a random draw"""

        df = self.sim.population.props
        result = params['ps_linear_equations'][f'{complication}'].predict(index)
        random_draw = pd.Series(self.rng.random_sample(size=len(index)), index=index.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']
        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]

        if complication == 'ectopic':
            df.loc[positive_index, 'ps_ectopic_pregnancy'] = True
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_ectopic_symptoms'].values[:] = 'none'
            df.loc[positive_index, 'is_pregnant'] = False
            if not positive_index.empty:
                logger.debug(f'The following women have experience an ectopic pregnancy,{positive_index}')

        if complication == 'multiples':
            df.loc[positive_index, 'ps_multiple_pregnancy'] = True

        if complication == 'miscarriage':
            df.loc[positive_index, 'ps_total_miscarriages'] = +1  # Could this be a function
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            if not positive_index.empty:
                logger.debug(f'The following women have miscarried,{positive_index}')

        if complication == 'abortion':
            df.loc[positive_index, 'ps_total_induced_abortion'] = +1  # Could this be a function
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            if not positive_index.empty:
                logger.debug(f'The following women have had an abortion,{positive_index}')

        if complication == 'pre_eclampsia':
            df.loc[positive_index, 'ps_htn_disorder_preg'].values[:] = 'mild_pe'
            df.loc[positive_index, 'ps_prev_pre_eclamp'] = True
            if not positive_index.empty:
                logger.debug(f'The following women have have developed pre_eclampsia,{positive_index}')

        if complication == 'gest_htn':
            df.loc[positive_index, 'ps_htn_disorder_preg'].values[:] = 'gest_htn'
            if not positive_index.empty:
                logger.debug(f'The following women have have developed gestational hypertension,{positive_index}')

        if complication == 'gest_diab':
            df.loc[positive_index, 'ps_gest_diab'] = True
            df.loc[positive_index, 'ps_prev_gest_diab'] = True
            if not positive_index.empty:
                logger.debug(f'The following women have have developed gestational diabetes,{positive_index}')

        if complication == 'stillbirth':
            df.loc[positive_index, 'ps_antepartum_still_birth'] = True
            df.loc[positive_index, 'ps_previous_stillbirth'] = True
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0
            if not positive_index.empty:
                logger.debug(f'The following women have have experienced an antepartum stillbirth,{positive_index}')

        # TODO: consider using return function to produce negative index which could be used for the chain of functions


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks and uses
    set_pregnancy_complications function to determine if women will experience complication. This event is incomplete
     and will eventually apply risk of antenatal death and handle antenatal care seeking. """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

    # ===================================== UPDATING GESTATIONAL AGE IN WEEKS  ========================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation

        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')
        pregnant_idx = df.index[df.is_alive & df.is_pregnant]
        df.loc[pregnant_idx, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')

    # ===================================== ECTOPIC PREGNANCY & MULTIPLES =============================================
        # Here we use the set_pregnancy_complications function to calculate each womans risk of ectopic pregnancy,
        # conduct a draw and edit relevant properties defined above

        newly_pregnant_idx = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1)]
        self.module.set_pregnancy_complications(params, newly_pregnant_idx, 'ectopic')

        # For women who don't experience and ectopic pregnancy we use the same function to assess risk of multiple
        # pregnancy
        np_no_ectopic = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1) &
                               ~df.ps_ectopic_pregnancy]
        self.module.set_pregnancy_complications(params, np_no_ectopic, 'multiples')

        # TODO: Link with Asif for generation of 2 newborns for multiples mother

    # ========================== MONTH 1 ECTOPIC PREGNANCY SYMPTOMS/CARE SEEKING ======================================
        # We look at all women whose pregnancy is ectopic and determine if they will experience symptoms

        ectopic_month_1 = df.index[df.is_alive & df.ps_ectopic_pregnancy & (df.ps_gestational_age_in_weeks == 4)]
        level_of_symptoms_ep = params['level_of_symptoms_ep']
        symptoms = self.module.rng.choice(level_of_symptoms_ep.level_of_symptoms_ep,
                                          size=len(ectopic_month_1),
                                          p=[0.3, 0.5, 0.2, 0])
        df.loc[ectopic_month_1, 'ps_ectopic_symptoms'].values[:] = symptoms

        # Todo: Re-write in line with symptom manager module and consider care seeking.

    # ==================================== MONTH 1 PREGNANCY RISKS ====================================================
        # Here we look at all the women who have reached one month gestation and apply the risk of early pregnancy loss

        month_1_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 4)]
        self.module.set_pregnancy_complications(params, month_1_idx, 'miscarriage')

    # ============================= MONTH 2 ECTOPIC PREGNANCY SYMPTOMS/CARE SEEKING ===================================
        # We now move on to look at women who are 2 months pregnant, For those whose pregnancy is ectopic we determine
        # if they will experience symptoms

        ectopic_month_2 = df.index[df.is_alive & df.ps_ectopic_pregnancy & (df.ps_gestational_age_in_weeks == 8) &
                                   (df.ps_ectopic_symptoms == 'none')]

        level_of_symptoms_ep = params['level_of_symptoms_ep']
        symptoms = self.module.rng.choice(level_of_symptoms_ep.level_of_symptoms_ep,
                                          size=len(ectopic_month_2),
                                          p=[0.1, 0.2, 0.4, 0.3])
        df.loc[ectopic_month_2, 'ps_ectopic_symptoms'].values[:] = symptoms

    # ========================================== MONTH 2 PREGNANCY RISKS ==============================================
        # Now we use the set_pregnancy_complications function to calculate risk and set properties for women whose
        # pregnancy is not ectopic

        # First MISCARRIAGE:
        month_2_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(params, month_2_idx, 'miscarriage')

        # Here we use the an index of women who will not miscarry to determine who will seek an abortion
        # ABORTION:
        month_2_no_miscarriage = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(params, month_2_no_miscarriage, 'abortion')

        # TODO: need to know the incidence of induced abortion for GA 8 weeks to determine if its worth while

    #    type_pa_complication = params['type_pa_complication']
    #    random_comp_type = self.module.rng.choice(type_pa_complication.type_pa_complication,
    #                           size=len(idx_ia_comps),
    #                           p=type_pa_complication.probability)
    #    df.loc[idx_ia_comps, 'ps_abortion_complication'].values[:] = random_comp_type
        # TODO: Above commented out whilst we consider symptoms/care seeking

    # ========================= MONTH 3 ECTOPIC PREGNANCY SYMPTOMS/CARE SEEKING =======================================

        ectopic_month_3 = df.index[df.is_alive & df.ps_ectopic_pregnancy & (df.ps_gestational_age_in_weeks == 13) &
                                   (df.ps_ectopic_symptoms == 'none')]

        level_of_symptoms_ep = params['level_of_symptoms_ep']
        symptoms = self.module.rng.choice(level_of_symptoms_ep.level_of_symptoms_ep,
                                          size=len(ectopic_month_3),
                                          p=[0, 0.2, 0.4, 0.4])
        df.loc[ectopic_month_3, 'ps_ectopic_symptoms'].values[:] = symptoms

        # Todo: need an end point for EP, point at which rupture WILL occur

    # ======================================= MONTH 3 PREGNANCY RISKS =================================================
        month_3_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 13)]

        # MISCARRIAGE
        self.module.set_pregnancy_complications(params, month_3_idx, 'miscarriage')

        # ABORTION:
        month_3_no_miscarriage = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 13)]
        self.module.set_pregnancy_complications(params, month_3_no_miscarriage, 'abortion')

    #    type_pa_complication = params['type_pa_complication']
    #    random_comp_type = self.module.rng.choice(type_pa_complication.type_pa_complication,
    #                                              size=len(idx_ia_comps),
    #                                              p=type_pa_complication.probability)
    #    df.loc[idx_ia_comps, 'ps_abortion_complication'].values[:] = random_comp_type

    # =========================== MONTH 4 RISK APPLICATION ============================================================
        month_4_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 17)]

        # MISCARRIAGE
        self.module.set_pregnancy_complications(params, month_4_idx, 'miscarriage')

        month_4_no_miscarriage = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 17)]

        # ABORTION:
        self.module.set_pregnancy_complications(params, month_4_no_miscarriage, 'abortion')

    #    type_pa_complication = params['type_pa_complication']
    #    random_comp_type = self.module.rng.choice(type_pa_complication.type_pa_complication,
    #                                              size=len(idx_ia_comps),
    #                                              p=type_pa_complication.probability)
    #    df.loc[idx_ia_comps, 'ps_abortion_complication'].values[:] = random_comp_type

    # =========================== MONTH 5 RISK APPLICATION ===========================================================
        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        month_5_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 22)]

        # MISCARRIAGE
        self.module.set_pregnancy_complications(params, month_5_idx, 'miscarriage')

        # ABORTION:
        month_5_no_miscarriage = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(params, month_5_no_miscarriage, 'abortion')

    #    type_pa_complication = params['type_pa_complication']
    #    random_comp_type = self.module.rng.choice(type_pa_complication.type_pa_complication,
    #                                             size=len(idx_ia_comps),
    #                                              p=type_pa_complication.probability)
    #    df.loc[idx_ia_comps, 'ps_abortion_complication'].values[:] = random_comp_type

        # PRE-ECLAMPSIA
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        month_5_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(params, month_5_preg_continues, 'pre_eclampsia')

        month_5_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 22) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        # Similarly only those women who dont develop pre-eclampsia are able to develop gestational hypertension
        self.module.set_pregnancy_complications(params, month_5_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_5_preg_continues, 'gest_diab')
        # TODO: Exclude current diabetics
        # TODO: review lit in regards to onset date and potentially move this to earlier

    # =========================== MONTH 6 RISK APPLICATION =============================================================
        # TODO: should this be 28 weeks to align with still birth definition

        # From month 6 it is possible women could be in labour at the time of this event so we exclude them
        month_6_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 27) & ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(params, month_6_idx, 'stillbirth')
        # Todo: also consider separating early and late still births (causation is different)
        # TODO: (risk factors) this will require close work, impact of conditions, on top of baseline risk etc etc
        # TODO: still birth should turn off any gestational diseases (if the evidence supports this)
        # TODO: Currently we're just turning off the pregnancy but we will need to allow for delivery of dead fetus?
        #  and associated complications (HSI_PresentsFollowingAntepartumStillbirth?)

        month_6_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 27) & ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(params, month_6_preg_continues, 'pre_eclampsia')

        month_6_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 27) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(params, month_6_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_6_preg_continues, 'gest_diab')

    # =========================== MONTH 7 RISK APPLICATION ===========================================================
        month_7_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 31) & ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(params, month_7_idx, 'stillbirth')

        month_7_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 31) & ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(params, month_7_preg_continues, 'pre_eclampsia')

        month_7_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 31) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(params, month_7_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_7_preg_continues, 'gest_diab')

    # =========================== MONTH 8 RISK APPLICATION =============================================================
        month_8_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 35) & ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(params, month_8_idx, 'stillbirth')

        month_8_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 35) & ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(params, month_8_preg_continues, 'pre_eclampsia')

        month_8_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 35) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(params, month_8_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_8_preg_continues, 'gest_diab')

    # =========================== MONTH 9 RISK APPLICATION ============================================================
        month_9_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 40) & ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(params, month_9_idx, 'stillbirth')

        month_9_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 40) & ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(params, month_9_preg_continues, 'pre_eclampsia')

        month_9_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 40) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(params, month_9_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_9_preg_continues, 'gest_diab')

    # =========================== MONTH 10 RISK APPLICATION ===========================================================
        # TODO: Change to weekly evaluation at the from 41 weeks+
        month_10_idx = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                              (df.ps_gestational_age_in_weeks == 44) & ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        self.module.set_pregnancy_complications(params, month_10_idx, 'stillbirth')
        month_10_preg_continues = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                        (df.ps_gestational_age_in_weeks == 44) & ~df.la_currently_in_labour]

        # PRE ECLAMPSIA
        self.module.set_pregnancy_complications(params, month_10_preg_continues, 'pre_eclampsia')
        month_10_no_pe = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                               (df.ps_gestational_age_in_weeks == 44) & (df.ps_htn_disorder_preg.values[:] == 'none')]

        # GESTATIONAL HYPERTENSION
        self.module.set_pregnancy_complications(params, month_10_no_pe, 'gest_htn')

        # GESTATIONAL DIABETES
        self.module.set_pregnancy_complications(params, month_10_preg_continues, 'gest_diab')


class PregnancyDiseaseProgressionEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancyDiseaseProgressionEvent. It runs every 4 weeks and determines if women who have a disease
    of pregnancy will undergo progression to the next stage. This event will need to be recoded using the
    progression_matrix function """
    # TODO: consider renaming if only dealing with HTN diseases

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=4)) # are we happy with this frequency

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        #  TODO: could we progress ectopic pregnancy here?
        #  TODO: similarly should we progress potential complicated/late abortions, miscarriage, stillbirth here?
        #       or too complicated as we loose the index

        # ============================= PROGRESSION OF HYPERTENSIVE DISEASES ==========================================
        # Here we look at all the women who are suffering from a hypertensive disorder of pregnancy and determine if
        # they will progress to a more severe form of the disease
        current_ghtn = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'gest_htn')
                                & ~df.la_currently_in_labour]
        current_mild_pe = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'mild_pe') &
                                   ~df.la_currently_in_labour]
        current_sev_pe = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'severe_pe') &
                                  ~df.la_currently_in_labour]

        # Now we apply the probability that women will progress from one disease stage/type to another
        def progress_disease(index, next_stage, r_next_stage):
            eff_prob_next_stage = pd.Series(r_next_stage, index=index)
            selected = index[eff_prob_next_stage > self.module.rng.random_sample(size=len(eff_prob_next_stage))]
            df.loc[selected, 'ps_htn_disorder_preg'].values[:] = next_stage
            # TODO: transition states Util function

        progress_disease(current_ghtn, 'mild_pe', params['r_mild_pe_gest_htn'])
        progress_disease(current_mild_pe,  'severe_pe', params['r_severe_pe_mild_pe'])
        progress_disease(current_sev_pe, 'eclampsia', params['r_eclampsia_severe_pe'])
        progress_disease(current_sev_pe, 'HELLP', params['r_hellp_severe_pe'])  # does double counting make sense

        # To determine who has progressed, we index all women by disease stage/type
        post_transition_mpe = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'mild_pe') &
                                       ~df.la_currently_in_labour]
        post_transition_spe = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'severe_pe') &
                                       ~df.la_currently_in_labour]
        post_transition_ec = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'eclampsia') &
                                      ~df.la_currently_in_labour]
        post_transition_hellp = df.index[df.is_alive & df.is_pregnant & (df.ps_htn_disorder_preg == 'help') &
                                         ~df.la_currently_in_labour]

        # Todo: do we have risk factors for progression? Are women less likley to progress if theyre on anti HTNs?

        # and we create a new index for each disease group containing women who have progressed
        # after_transition_mild_pe = current_ghtn[current_ghtn.isin(post_transition_mpe)] # might not need this?
        after_transition_sev_pe = current_mild_pe[current_mild_pe.isin(post_transition_spe)]
        after_transition_eclampsia = current_sev_pe[current_sev_pe.isin(post_transition_ec)]
        after_transition_hellp = current_sev_pe[current_sev_pe.isin(post_transition_hellp)]

        # (Dummy care seeking)
        prob_seek_care_spe = pd.Series(0.6, index=after_transition_sev_pe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(after_transition_sev_pe)),
                                index=after_transition_sev_pe)
        dfx = pd.concat([random_draw, prob_seek_care_spe], axis=1)
        dfx.columns = ['random_draw', 'prob_seek_care_spe']
        idx_care_seeker = dfx.index[dfx.prob_seek_care_spe > dfx.random_draw]

        # For those women who will seek care we schedule the appropriate HSI with a high priority, starting with severe
        # pre-eclampsia
#        for person in idx_care_seeker:  # todo: can we do this without a for loop
#            care_seeking_date = self.sim.date
#            event = antenatal_care.HSI_AntenatalCare_PresentsDuringPregnancyRelatedEmergency(self.module,
        #            person_id=person)
#            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                priority=1,  # ????
#                                                                topen=care_seeking_date,
#                                                                tclose=None)

        # Then we schedule care seeking for women with new onset eclampsia
        prob_seek_care_ec = pd.Series(0.8, index=after_transition_eclampsia)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(after_transition_eclampsia)),
                                index=after_transition_eclampsia)
        dfx = pd.concat([random_draw, prob_seek_care_ec], axis=1)
        dfx.columns = ['random_draw', 'prob_seek_care_ec']
        idx_care_seeker = dfx.index[dfx.prob_seek_care_ec > dfx.random_draw]

#        for person in idx_care_seeker:
#                care_seeking_date = self.sim.date
#                event = antenatal_care.HSI_AntenatalCare_PresentsDuringPregnancyRelatedEmergency(self.module,
        #                person_id=person)
#                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                priority=1,  # ????
#                                                                topen=care_seeking_date,
#                                                                tclose=None)

        # Then we schedule care seeking for women with new onset HELLP
        prob_seek_care_hellp = pd.Series(0.8, index=after_transition_hellp)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(after_transition_hellp)),
                                index=after_transition_hellp)
        dfx = pd.concat([random_draw, prob_seek_care_hellp], axis=1)
        dfx.columns = ['random_draw', 'prob_seek_care_hellp']
        idx_care_seeker = dfx.index[dfx.prob_seek_care_hellp > dfx.random_draw]

#        for person in idx_care_seeker:
#                care_seeking_date = self.sim.date
#                event = antenatal_care.HSI_AntenatalCare_PresentsDuringPregnancyRelatedEmergency(self.module,
#                person_id=person)
#                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                    priority=1,  # ????
#                                                                    topen=care_seeking_date,
#                                                                    tclose=None)

        # Todo: discuss with Tim C if we need to apply symptoms IF we know that severe and > are all symptomatic?
        #  Or do we just apply a code
        # TODO: consider progression to CV event (mainly stroke)


class PregnancyDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # do we schedule all women who dont seek treatment to come to this event and apply CFR
        # AND then schedule women who do seek treatment here in light of treatment failure?

        #HELLP is largely going to be untreated here?
        # Eclampsia is a good one and should be resolved.


class PregnancySupervisorLoggingEvent(RegularEvent, PopulationScopeEventMixin):
        """Handles Pregnancy Supervision  logging"""

        def __init__(self, module):
            """schedule logging to repeat every 3 months
            """
            #    self.repeat = 3
            #    super().__init__(module, frequency=DateOffset(days=self.repeat))
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):
            """Apply this event to the population.
            :param population: the current population
            """
            df = population.props
