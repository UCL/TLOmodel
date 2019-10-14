"""
Module responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
onset of antenatal complications .
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PregnancySupervisor(Module):
    """
    This module is responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
    onset of antenatal complications """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_pregnancy_factors': Parameter(
            Types.DATA_FRAME, 'Data frame containing probabilities of key outcomes/complications associated with the '
                              'antenatal period'),
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
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
    }

    PROPERTIES = {
        'ps_gestational_age': Property(Types.INT, 'current gestational age, in weeks, of this womans pregnancy'),
        'ps_ectopic_pregnancy': Property(Types.BOOL, 'Whether this womans pregnancy is ectopic'),
        'ps_ectopic_symptoms': Property(
            Types.CATEGORICAL, 'Level of symptoms for ectopic pregnancy',
            categories=['none', 'abdominal pain', 'abdominal pain plus bleeding', 'shock']),
        # TODO: review in light of new symptom tracker
        'ps_ep_unified_symptom_code': Property(
            Types.CATEGORICAL,
            'Level of symptoms on the standardised scale (governing health-care seeking): '
            '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
            categories=[0, 1, 2, 3, 4]),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_total_miscarriages': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'ps_total_induced_abortion': Property(Types.INT, 'the number of induced abortions a woman has experienced'),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
        'ps_pre_eclampsia': Property(Types.BOOL, 'whether this woman has pre-eclampsia'),
        'ps_prev_pre_eclamp': Property(Types.BOOL,'whether this woman has experienced pre-eclampsia in a previous '
                                                  'pregnancy'),
        'ps_gest_htn': Property(Types.BOOL, 'whether this woman has gestational hypertension'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy')
    }

    TREATMENT_ID = ''

    def read_parameters(self, data_folder):
        """
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name=None)

        params['prob_pregnancy_factors'] = dfd['prob_pregnancy_factors']
        dfd['prob_pregnancy_factors'].set_index('index', inplace=True)

        dfd['parameter_values'].set_index('parameter_name', inplace=True)
        params['rr_miscarriage_prevmiscarriage'] = dfd['parameter_values'].loc['rr_miscarriage_prevmiscarriage',
                                                                               'value']
        params['rr_miscarriage_35'] = dfd['parameter_values'].loc['rr_miscarriage_35', 'value']
        params['rr_miscarriage_3134'] = dfd['parameter_values'].loc['rr_miscarriage_3134', 'value']
        params['rr_miscarriage_grav4'] = dfd['parameter_values'].loc['rr_miscarriage_grav4', 'value']
        params['rr_pre_eclamp_nulip'] = dfd['parameter_values'].loc['rr_pre_eclamp_nulip', 'value']
        params['rr_pre_eclamp_prev_pe'] = dfd['parameter_values'].loc['rr_pre_eclamp_prev_pe', 'value']
        params['rr_gest_diab_overweight'] = dfd['parameter_values'].loc['rr_gest_diab_overweight', 'value']
        params['rr_gest_diab_stillbirth'] = dfd['parameter_values'].loc['rr_gest_diab_stillbirth', 'value']
        params['rr_gest_diab_prevdiab'] = dfd['parameter_values'].loc['rr_gest_diab_prevdiab', 'value']
        params['rr_gest_diab_chron_htn'] = dfd['parameter_values'].loc['rr_gest_diab_chron_htn', 'value']
        params['prob_ectopic_pregnancy'] = dfd['parameter_values'].loc['prob_ectopic_pregnancy', 'value']
        params['prob_multiples'] = dfd['parameter_values'].loc['prob_multiples', 'value']

    #        if 'HealthBurden' in self.sim.modules.keys():
#            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)


    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ps_gestational_age'] = 0
        df.loc[df.sex == 'F', 'ps_ectopic_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_ectopic_symptoms'] = 'none'
        df.loc[df.sex == 'F', 'ps_ep_unified_symptom_code'] = 0
        df.loc[df.sex == 'F', 'ps_multiple_pregnancy'] = False
        df.loc[df.sex == 'F', 'ps_total_miscarriages'] = 0
        df.loc[df.sex == 'F', 'ps_total_induced_abortion'] = 0
        df.loc[df.sex == 'F', 'ps_antepartum_still_birth'] = False
        df.loc[df.sex == 'F', 'ps_previous_stillbirth'] = False
        df.loc[df.sex == 'F', 'ps_pre_eclampsia'] = False
        df.loc[df.sex == 'F', 'ps_prev_pre_eclamp'] = False
        df.loc[df.sex == 'F', 'ps_gest_htn'] = False
        df.loc[df.sex == 'F', 'ps_gest_diab'] = False
        df.loc[df.sex == 'F', 'ps_prev_gest_diab'] = False

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancySupervisorLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ps_gestational_age'] = 0
            df.at[child_id, 'ps_ectopic_pregnancy'] = False
            df.at[child_id, 'ps_ectopic_symptoms'] = 'none'
            df.at[child_id, 'ps_ep_unified_symptom_code'] = 0
            df.at[child_id, 'ps_multiple_pregnancy'] = False
            df.at[child_id, 'ps_total_miscarriages'] = 0
            df.at[child_id, 'ps_total_induced_abortion'] = 0
            df.at[child_id, 'ps_antepartum_still_birth'] = False
            df.at[child_id, 'ps_previous_stillbirth'] = False
            df.at[child_id, 'ps_pre_eclampsia'] = False
            df.at[child_id, 'ps_prev_pre_eclamp'] = False
            df.at[child_id, 'ps_gest_htn'] = False
            df.at[child_id, 'ps_gest_diab'] = False
            df.at[child_id, 'ps_prev_gest_diab'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is PregnancySupervisor, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

#    def report_daly_values(self):

    #    logger.debug('This is mockitis reporting my health values')
    #    df = self.sim.population.props  # shortcut to population properties dataframe
    #    p = self.parameters


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This event updates the gestational age of each pregnant woman every week and applies the risk of her developing
    any complications during her pregnancy
    """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # todo: talk to programmers to shorten this code section

    # =========================== GESTATIONAL AGE UPDATE FOR ALL PREGNANT WOMEN ========================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation
        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')
        pregnant_idx = df.index[df.is_alive & df.is_pregnant]
        df.loc[pregnant_idx, 'ps_gestational_age'] = gestation_in_weeks.astype(int)

    # ===================================== ECTOPIC PREGNANCY/ MULTIPLES ===============================================
        # Here we look at all the newly pregnant women (1 week gestation) and apply the risk of this pregnancy being
        # ectopic
        newly_pregnant_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 1)]
        eff_prob_ectopic = pd.Series(params['prob_ectopic_pregnancy'], index=newly_pregnant_idx)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(newly_pregnant_idx)),
                                index=newly_pregnant_idx)
        dfx = pd.concat([random_draw, eff_prob_ectopic], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_ectopic']
        idx_ep = dfx.index[dfx.eff_prob_ectopic > dfx.random_draw]
        idx_mp = dfx.index[dfx.eff_prob_ectopic < dfx.random_draw]

        df.loc[idx_ep, 'ps_ectopic_pregnancy'] = True

        # TODO: symptoms (rupture etc), care seeking, (onset), what HSI will this interact with
        # TODO: do we reset pregnancy information (is_preg, due_date)- will these women still present to ANC?

        # If implantation is normal we apply the risk this pregnancy may be multiple gestation
        eff_prob_multiples = pd.Series(params['prob_multiples'], index=idx_mp)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_mp)),
                                index=idx_mp)
        dfx = pd.concat([random_draw, eff_prob_multiples], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_multiples']
        idx_mp_t = dfx.index[dfx.eff_prob_multiples > dfx.random_draw]

        df.loc[idx_mp_t, 'ps_multiple_pregnancy'] = True
        # TODO: simulation code will need to generate 2 children- presumably calculated risks in labour will be the
        #  same for all babies

    # ============================= DF SHORTCUTS =======================================================================

        misc_risk = params['prob_pregnancy_factors']['risk_miscarriage']
        ia_risk = params['prob_pregnancy_factors']['risk_abortion']
        sb_risk = params['prob_pregnancy_factors']['risk_still_birth']
        pe_risk = params['prob_pregnancy_factors']['risk_pre_eclamp']
        gh_risk = params['prob_pregnancy_factors']['risk_g_htn']
        gd_risk = params['prob_pregnancy_factors']['risk_g_diab']

    # =========================== MONTH 1 RISK APPLICATION =============================================================
        # Here we look at all the women who have reached one month gestation and apply the risk of early pregnancy loss

        # Todo: congenital anomalies? - link to miscarraige/stillbirth etc

        month_1_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 4)]

        # MISCARRIAGE
        eff_prob_miscarriage = pd.Series(misc_risk[1], index=month_1_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >=1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31) & (df.age_years < 35)] \
            *= params['rr_miscarriage_3134']
#        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
#            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_1_idx)),
                                index=month_1_idx)
        dfx = pd.concat([random_draw, eff_prob_miscarriage], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0
        df.loc[idx_mc, 'la_due_date_current_pregnancy'] = pd.NaT

        # todo: seems unlikely there will be any complications of pregnancy loss within the first month (but consider)
        # Todo: whats the best way to log this information

    # =========================== MONTH 2 RISK APPLICATION =============================================================
        # Here we apply to risk of adverse pregnancy outcomes for month 2 including miscarriage and induced abortion.
        # Todo: need to know the incidence of induced abortion for GA 8 weeks to determine if its worth while

        # MISCARRIAGE:
        month_2_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 8)]

        eff_prob_miscarriage = pd.Series(misc_risk[2], index=month_2_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >= 1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31) & (df.age_years < 35)] \
            *= params['rr_miscarriage_3134']
        #        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
        #            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_2_idx)),
                                index=month_2_idx)
        dfx = pd.concat([random_draw, eff_prob_miscarriage], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]
        idx_ac = dfx.index[dfx.eff_prob_miscarriage < dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0
        df.loc[idx_mc, 'la_due_date_current_pregnancy'] = pd.NaT

        # ABORTION:
        # Here we use the an index of women who will not miscarry to determine who will seek an abortion
        eff_prob_abortion = pd.Series(ia_risk[2], index=idx_ac)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_ac)),
                                index=idx_ac)
        dfx = pd.concat([random_draw, eff_prob_abortion], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_abortion']
        idx_ia = dfx.index[dfx.eff_prob_abortion > dfx.random_draw]

        df.loc[idx_ia, 'ps_total_induced_abortion'] = +1  # Could this be a function
        df.loc[idx_ia, 'is_pregnant'] = False
        df.loc[idx_ia, 'ps_gestational_age'] = 0
        df.loc[idx_ia, 'la_due_date_current_pregnancy'] = pd.NaT

        # TODO: risk factors for Induced abortion? Or blanket prevelance? Should link to unwanted pregnancy
        # TODO: Incidence of complications, symptoms of complications and care seeking (HSI?)

    # =========================== MONTH 3 RISK APPLICATION =============================================================
        month_3_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 13)]

        # MISCARRIAGE
        eff_prob_miscarriage = pd.Series(misc_risk[3], index=month_3_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >= 1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31) & (df.age_years < 35)] \
            *= params['rr_miscarriage_3134']
        #        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
        #            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_3_idx)),
                                index=month_3_idx)
        dfx = pd.concat([random_draw, eff_prob_miscarriage], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]
        idx_ac = dfx.index[dfx.eff_prob_miscarriage < dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0
        df.loc[idx_mc, 'la_due_date_current_pregnancy'] = pd.NaT

        # ABORTION:
        eff_prob_abortion = pd.Series(ia_risk[3], index=idx_ac)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_ac)),
                                index=idx_ac)
        dfx = pd.concat([random_draw, eff_prob_abortion], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_abortion']
        idx_ia = dfx.index[dfx.eff_prob_abortion > dfx.random_draw]

        df.loc[idx_ia, 'ps_total_induced_abortion'] = +1  # Could this be a function
        df.loc[idx_ia, 'is_pregnant'] = False
        df.loc[idx_ia, 'ps_gestational_age'] = 0
        df.loc[idx_ia, 'la_due_date_current_pregnancy'] = pd.NaT

    # =========================== MONTH 4 RISK APPLICATION =============================================================
        month_4_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 17)]

        # MISCARRIAGE
        eff_prob_miscarriage = pd.Series(misc_risk[4], index=month_4_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >= 1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31) & (df.age_years < 35)] \
            *= params['rr_miscarriage_3134']
        #        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
        #            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_4_idx)),
                                index=month_4_idx)
        dfx = pd.concat([random_draw, eff_prob_miscarriage], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]
        idx_ac = dfx.index[dfx.eff_prob_miscarriage < dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0
        df.loc[idx_mc, 'la_due_date_current_pregnancy'] = pd.NaT

        # ABORTION:
        eff_prob_abortion = pd.Series(ia_risk[4], index=idx_ac)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_ac)),
                                index=idx_ac)
        dfx = pd.concat([random_draw, eff_prob_abortion], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_abortion']
        idx_ia = dfx.index[dfx.eff_prob_abortion > dfx.random_draw]

        df.loc[idx_ia, 'ps_total_induced_abortion'] = +1  # Could this be a function
        df.loc[idx_ia, 'is_pregnant'] = False
        df.loc[idx_ia, 'ps_gestational_age'] = 0
        df.loc[idx_ia, 'la_due_date_current_pregnancy'] = pd.NaT

    # =========================== MONTH 5 RISK APPLICATION =========================================================
        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        month_5_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 22)]

        # MISCARRIAGE
        eff_prob_miscarriage = pd.Series(misc_risk[5], index=month_5_idx)
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.ps_total_miscarriages >= 1)] \
            *= params['rr_miscarriage_prevmiscarriage']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 35)] \
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.age_years >= 31) & (df.age_years < 35)] \
            *= params['rr_miscarriage_3134']
        #        eff_prob_miscarriage.loc[df.is_alive & df.is_pregnant & (df.hp_pre_eclampsia == 'none') &
        #        df.hp_prev_pre_eclamp] \
        #            *= params['rr_miscarriage_grav4']

        # Here add columns to the temporary dataframe to exclude those women already suffering from hypertensive
        # disorders

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_5_idx)),
                                index=month_5_idx)
        dfx = pd.concat([random_draw, eff_prob_miscarriage], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_miscarriage']
        idx_mc = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]
        idx_ac = dfx.index[dfx.eff_prob_miscarriage < dfx.random_draw]

        df.loc[idx_mc, 'ps_total_miscarriages'] = +1  # Could this be a function
        df.loc[idx_mc, 'is_pregnant'] = False
        df.loc[idx_mc, 'ps_gestational_age'] = 0
        df.loc[idx_mc, 'la_due_date_current_pregnancy'] = pd.NaT

        # ABORTION:
        eff_prob_abortion = pd.Series(ia_risk[5], index=idx_ac)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_ac)),
                                index=idx_ac)

        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_5_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_5_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_5_idx)

        dfx = pd.concat([random_draw, eff_prob_abortion, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_abortion', 'eclamp_stat', 'htn_stat', 'gd_stat']
        idx_ia = dfx.index[dfx.eff_prob_abortion > dfx.random_draw]

        df.loc[idx_ia, 'ps_total_induced_abortion'] = +1  # Could this be a function
        df.loc[idx_ia, 'is_pregnant'] = False
        df.loc[idx_ia, 'ps_gestational_age'] = 0
        df.loc[idx_ia, 'la_due_date_current_pregnancy'] = pd.NaT

        at_risk_htn = dfx.index[(dfx.eff_prob_abortion < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_abortion < dfx.random_draw) & ~dfx.gd_stat]

        # PRE-ECLAMPSIA
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        eff_prob_pre_eclamp = pd.Series(pe_risk[5], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)]\
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp]\
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        # Similarly only those women who dont develop pre-eclampsia are able to develop gestational hypertension
        eff_prob_htn = pd.Series(gh_risk[5], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True

        # Todo: review difference in risk factors between GH and PE, there will be overlap. Also need to consider
        #  progression of states from one to another

        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[5], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True

        # Todo: review lit in regards to onset date and potentially move this to earlier

    # =========================== MONTH 6 RISK APPLICATION =============================================================
        # From month 6 it is possible women could be in labour at the time of this event so we exclude them
        month_6_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 27) &
                               ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        eff_prob_stillbirth = pd.Series(sb_risk[6], index=month_6_idx)

        # Todo: also consider separating early and late still births (causation is different)
        # TODO: (risk factors) this will require close work, impact of conditions, on top of baseline risk etc etc

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_6_idx)), index=month_6_idx)
        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_6_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_6_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_6_idx)

        dfx = pd.concat([random_draw, eff_prob_stillbirth, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_stillbirth', 'eclamp_stat', 'htn_stat', 'gd_stat']

        idx_sb = dfx.index[dfx.eff_prob_stillbirth > dfx.random_draw]
        at_risk_htn = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.gd_stat]

        df.loc[idx_sb, 'ps_antepartum_still_birth'] = True
        # TODO: if we're only using boolean we're not going to capture number of antepartum stillbirths here (doesnt
        #  work with tims method of generatng the child then using instant death)
        df.loc[idx_sb, 'ps_previous_stillbirth'] = True
        df.loc[idx_sb, 'is_pregnant'] = False
        df.loc[idx_sb, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[idx_sb, 'ps_gestational_age'] = 0

        # TODO: Currently we're just turning off the pregnancy but we will need to allow for delivery of dead fetus?
        #  and associated complications (HSI_PresentsFollowingAntepartumStillbirth?)

        # PRE ECLAMPSIA
        eff_prob_pre_eclamp = pd.Series(pe_risk[6], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        eff_prob_htn = pd.Series(gh_risk[6], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True

        # Todo: review difference in risk factors between GH and PE, there will be overlap. Also need to consider
        #  progression of states from one to another

        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[6], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True

    # =========================== MONTH 7 RISK APPLICATION =============================================================
        month_7_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 31)&
                               ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        eff_prob_stillbirth = pd.Series(sb_risk[7], index=month_7_idx)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_7_idx)), index=month_7_idx)
        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_7_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_7_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_7_idx)

        dfx = pd.concat([random_draw, eff_prob_stillbirth, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_stillbirth', 'eclamp_stat', 'htn_stat', 'gd_stat']

        idx_sb = dfx.index[dfx.eff_prob_stillbirth > dfx.random_draw]
        at_risk_htn = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.gd_stat]

        df.loc[idx_sb, 'ps_antepartum_still_birth'] = True
        df.loc[idx_sb, 'ps_previous_stillbirth'] = True
        df.loc[idx_sb, 'is_pregnant'] = False
        df.loc[idx_sb, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[idx_sb, 'ps_gestational_age'] = 0


        # PRE ECLAMPSIA
        eff_prob_pre_eclamp = pd.Series(pe_risk[7], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        eff_prob_htn = pd.Series(gh_risk[7], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True


        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[7], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True

    # =========================== MONTH 8 RISK APPLICATION =============================================================
        month_8_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 35)&
                               ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        eff_prob_stillbirth = pd.Series(sb_risk[8], index=month_8_idx)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_8_idx)), index=month_8_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_8_idx)
        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_8_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_8_idx)

        dfx = pd.concat([random_draw, eff_prob_stillbirth, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_stillbirth', 'eclamp_stat', 'htn_stat', 'gd_stat']

        idx_sb = dfx.index[dfx.eff_prob_stillbirth > dfx.random_draw]
        at_risk_htn = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.gd_stat]

        df.loc[idx_sb, 'ps_antepartum_still_birth'] = True
        df.loc[idx_sb, 'ps_previous_stillbirth'] = True
        df.loc[idx_sb, 'is_pregnant'] = False
        df.loc[idx_sb, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[idx_sb, 'ps_gestational_age'] = 0

        # PRE ECLAMPSIA
        eff_prob_pre_eclamp = pd.Series(pe_risk[8], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        eff_prob_htn = pd.Series(gh_risk[8], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True

        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[8], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True

    # =========================== MONTH 9 RISK APPLICATION =============================================================
        month_9_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 40)&
                               ~df.la_currently_in_labour]

        # STILL BIRTH RISK
        eff_prob_stillbirth = pd.Series(sb_risk[9], index=month_9_idx)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_9_idx)), index=month_9_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_9_idx)
        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_9_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_9_idx)

        dfx = pd.concat([random_draw, eff_prob_stillbirth, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_stillbirth', 'eclamp_stat', 'htn_stat', 'gd_stat']

        idx_sb = dfx.index[dfx.eff_prob_stillbirth > dfx.random_draw]
        at_risk_htn = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.gd_stat]

        df.loc[idx_sb, 'ps_antepartum_still_birth'] = True
        df.loc[idx_sb, 'ps_previous_stillbirth'] = True
        df.loc[idx_sb, 'is_pregnant'] = False
        df.loc[idx_sb, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[idx_sb, 'ps_gestational_age'] = 0

        # PRE ECLAMPSIA
        eff_prob_pre_eclamp = pd.Series(pe_risk[9], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        eff_prob_htn = pd.Series(gh_risk[9], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True

        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[9], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True

    # =========================== MONTH 10 RISK APPLICATION ============================================================
        month_10_idx = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age == 44) &
                               ~df.la_currently_in_labour]   # TODO: should we look weekly at post term women?

        # STILL BIRTH RISK
        eff_prob_stillbirth = pd.Series(sb_risk[10], index=month_10_idx)

        random_draw = pd.Series(self.module.rng.random_sample(size=len(month_10_idx)), index=month_10_idx)
        htn_stat = pd.Series(df.ps_gest_htn, index=month_10_idx)
        eclamp_stat = pd.Series(df.ps_pre_eclampsia, index=month_10_idx)
        gd_stat = pd.Series(df.ps_gest_diab, index=month_10_idx)

        dfx = pd.concat([random_draw, eff_prob_stillbirth, eclamp_stat, htn_stat, gd_stat], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_stillbirth', 'eclamp_stat', 'htn_stat', 'gd_stat']

        idx_sb = dfx.index[dfx.eff_prob_stillbirth > dfx.random_draw]
        at_risk_htn = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.eclamp_stat & ~dfx.htn_stat]
        at_risk_gd = dfx.index[(dfx.eff_prob_stillbirth < dfx.random_draw) & ~dfx.gd_stat]

        df.loc[idx_sb, 'ps_antepartum_still_birth'] = True
        df.loc[idx_sb, 'ps_previous_stillbirth'] = True
        df.loc[idx_sb, 'is_pregnant'] = False
        df.loc[idx_sb, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[idx_sb, 'ps_gestational_age'] = 0

        # PRE ECLAMPSIA
        eff_prob_pre_eclamp = pd.Series(pe_risk[10], index=at_risk_htn)
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & (df.la_parity == 0)] \
            *= params['rr_pre_eclamp_nulip']
        eff_prob_pre_eclamp.loc[df.is_alive & df.is_pregnant & df.ps_prev_pre_eclamp] \
            *= params['rr_pre_eclamp_prev_pe']

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_htn)),
                                index=at_risk_htn)
        dfx = pd.concat([random_draw, eff_prob_pre_eclamp], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_eclamp']
        idx_pe = dfx.index[dfx.eff_prob_pre_eclamp > dfx.random_draw]
        idx_npe = dfx.index[dfx.eff_prob_pre_eclamp < dfx.random_draw]

        df.loc[idx_pe, 'ps_pre_eclampsia'] = True
        df.loc[idx_pe, 'ps_prev_pre_eclamp'] = True

        # GESTATIONAL HYPERTENSION
        eff_prob_htn = pd.Series(gh_risk[10], index=idx_npe)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(idx_npe)), index=idx_npe)

        dfx = pd.concat([random_draw, eff_prob_htn], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_htn']
        idx_gh = dfx.index[dfx.eff_prob_htn > dfx.random_draw]
        df.loc[idx_gh, 'ps_gest_htn'] = True

        # GESTATIONAL DIABETES
        eff_prob_pre_gestdiab = pd.Series(gd_risk[10], index=at_risk_gd)

        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_previous_stillbirth] \
            *= params['rr_gest_diab_stillbirth']
        eff_prob_pre_gestdiab.loc[df.is_alive & df.is_pregnant & df.ps_prev_gest_diab] \
            *= params['rr_gest_diab_prevdiab']  # confirm if this is all diabetes or just previous GDM
        # additional risk factors  :overweight, chronic htn

        random_draw = pd.Series(self.module.rng.random_sample(size=len(at_risk_gd)),
                                index=at_risk_gd)
        dfx = pd.concat([random_draw, eff_prob_pre_gestdiab], axis=1)
        dfx.columns = ['random_draw', 'eff_prob_pre_gestdiab']
        idx_gh = dfx.index[dfx.eff_prob_pre_gestdiab > dfx.random_draw]

        df.loc[idx_gh, 'ps_gest_diab'] = True
        df.loc[idx_gh, 'ps_prev_gest_diab'] = True


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
