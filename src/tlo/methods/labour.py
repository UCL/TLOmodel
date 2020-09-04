from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Labour (Module):
    """This module for labour, delivery, the immediate postpartum period and skilled birth attendance."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary will store additional information around delivery and birth
        self.mother_and_newborn_info = dict()

        # This dictionary will track incidence of complications of labour
        self.labour_tracker = dict()

        # This list contains the individual_ids of women in labour, used for testing
        self.women_in_labour = list()

        # These lists will contain possible complications and are used as checks in assert functions
        self.possible_intrapartum_complications = list()
        self.possible_postpartum_complications = list()

    PARAMETERS = {
        #  ===================================  NATURAL HISTORY PARAMETERS =============================================
        'intercept_parity_lr2010': Parameter(
            Types.REAL, 'intercept value for linear regression equation predicating womens parity at 2010 baseline'),
        'effect_age_parity_lr2010': Parameter(
            Types.REAL, 'effect of an increase in age by 1 year in the linear regression equation predicating '
                        'womens parity at 2010 baseline'),
        'effect_mar_stat_2_parity_lr2010': Parameter(
            Types.REAL, 'effect of a change in marriage status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_mar_stat_3_parity_lr2010': Parameter(
            Types.REAL, 'effect of a change in marriage status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_wealth_lev_5_parity_lr2010': Parameter(
            Types.REAL, 'effect of a change in wealth status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_wealth_lev_4_parity_lr2010': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_3_parity_lr2010': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_2_parity_lr2010': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_1_parity_lr2010': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'lower_limit_term_days': Parameter(
            Types.REAL, 'minimum number of days gestation at which a woman can go into labour and be categorised as '
                        'term'),
        'upper_limit_term_days': Parameter(
            Types.REAL, 'maximum number of days gestation at which a woman can go into labour and be categorised as '
                        'term'),
        'lower_limit_early_preterm_days': Parameter(
            Types.REAL, 'minimum number of days gestation at which a woman can go into labour and be categorised as '
                        'early preterm'),
        'upper_limit_early_preterm_days': Parameter(
            Types.REAL, 'maximum number of days gestation at which a woman can go into labour and be categorised as '
                        'early preterm'),
        'lower_limit_late_preterm_days': Parameter(
            Types.REAL, 'minimum number of days gestation at which a woman can go into labour and be categorised as '
                        'early preterm'),
        'upper_limit_late_preterm_days': Parameter(
            Types.REAL, 'maximum number of days gestation at which a woman can go into labour and be categorised as '
                        'late preterm'),
        'lower_limit_postterm_days': Parameter(
            Types.REAL, 'minimum number of days gestation at which a woman can go into labour and be categorised as '
                        'post term '),
        'odds_early_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 28-33 weeks gestation'),
        'or_early_ptb_age<20': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women younger than 20'),
        'or_early_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women who have previously delivered preterm'),
        'odds_late_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 33-36 weeks gestation'),
        'or_late_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of preterm labour for women younger than 20'),
        'odds_post_term_labour': Parameter(
            Types.REAL, 'odds of a woman entering labour at >42 weeks gestation'),
        'rrr_ptl_bmi_more25': Parameter(
            Types.REAL, 'relative risk ratio of post term labour for a woman with a BMI of > 25'),
        'rrr_ptl_not_married': Parameter(
            Types.REAL, 'relative risk ratio of post term labour for a woman who isnt married'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_para_more3': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they have a parity of greater'
                        'than 2'),
        'rr_PL_OL_bmi_less18': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they have a BMI of less than '
                        '18'),
        'rr_PL_OL_bmi_more25': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they have a BMI of > 25'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'odds_ip_eclampsia': Parameter(
            Types.REAL, 'odds of an eclamptic seizure during labour'),
        'or_ip_eclampsia_30_34': Parameter(
            Types.REAL, 'odds ratio of eclampsia for women ages between 30 and 34'),
        'or_ip_eclampsia_35': Parameter(
            Types.REAL, 'odds ratio of eclampsia for women ages older than 35'),
        'or_ip_eclampsia_nullip': Parameter(
            Types.REAL, 'odds ratio of eclampsia for women who have not previously delivered a child'),
        'or_ip_eclampsia_gest_diab': Parameter(
            Types.REAL, 'odds ratio of eclampsia for women who has gestational diabetes'),
        'prob_aph': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage during labour'),
        'rr_aph_pl_ol': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage following obstructed labour'),
        'prob_ip_sepsis': Parameter(
            Types.REAL, 'probability of sepsis in labour'),
        'rr_ip_sepsis_pl_ol': Parameter(
            Types.REAL, 'relative risk of developing sepsis following obstructed labour'),
        'odds_uterine_rupture': Parameter(
            Types.REAL, 'probability of a uterine rupture during labour'),
        'or_ur_grand_multip': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have delivered >4 times previously'),
        'or_ur_prev_cs': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have previously delivered via caesarean '
                        'section'),
        'or_ur_ref_ol': Parameter(
            Types.REAL,
            'relative risk of uterine rupture in women who have been referred in obstructed labour'),
        'severity_maternal_haemorrhage': Parameter(
            Types.LIST, 'probability a maternal hemorrhage is non-severe (<1000mls) or severe (>1000mls)'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_severe_pre_eclamp': Parameter(
            Types.REAL, 'case fatality rate for severe pre eclampsia during labour'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_obstructed_labour': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother survives'),
        'rr_still_birth_ol_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth following a maternal death due to obstructed labour'),
        'prob_still_birth_antepartum_haem': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother survives'),
        'rr_still_birth_aph_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth following a maternal death due to an antepartum haemorrhage'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother survives'),
        'rr_still_birth_sepsis_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth following a maternal death due to sepsis'),
        'prob_still_birth_uterine_rupture': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother survives'),
        'rr_still_birth_ur_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth following a maternal death due to uterine rupture'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother survives'),
        'rr_still_birth_eclampsia_maternal_death': Parameter(
            Types.REAL, 'relative risk of still birth following a maternal death due to eclampsia'),
        'prob_still_birth_severe_pre_eclamp': Parameter(
            Types.REAL, 'probability of still birth following severe pre eclampsia in labour where the mother '
                        'survives'),
        'prob_pp_eclampsia': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spontaneous unobstructed '
                        'labour'),
        'prob_pph': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage following labour'),
        'rr_pph_pl_ol': Parameter(
            Types.REAL, 'relative risk of postpartum haemorrhage following obstructed labour'),
        'prob_pph_source': Parameter(
            Types.LIST, 'probability of uterine atony and retained placenta as the source of postpartum bleeding '),
        'odds_pp_sepsis': Parameter(
            Types.REAL, 'odds of a woman developing sepsis following delivery'),
        'or_pp_sepsis_rural': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following delivery if she is from a rural setting'),
        'or_pp_sepsis_no_edu': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following delivery if she has no formal education'),
        'or_pp_sepsis_primary_edu': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following delivery if she has primary education'),
        'or_pp_sepsis_avd': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following delivery if she delivered with instrumental '
                        'assistance'),
        'or_pp_sepsis_cs': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following delivery if she delivered via caesarean '
                        'section'),
        'or_pp_sepsis_prom': Parameter(
            Types.REAL, 'odds ratio of a woman developing sepsis following premature rupture of membranes '),
        'cfr_pp_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhage'),
        'cfr_pp_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pp_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis following delivery'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_birth_asphyxia': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),

        # ================================= HEALTH CARE SEEKING PARAMETERS ===========================================
        'odds_deliver_in_health_centre': Parameter(
            Types.REAL, 'odds of a woman delivering in a health centre compared to a hospital'),
        'rrr_hc_delivery_age_25_29': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 25-29 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_30_34': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 30-34 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_35_39': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 35-39 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_40_44': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 40-44 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_45_49': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 45-49 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_rural': Parameter(
            Types.REAL, 'relative risk ratio for a woman living in a rural setting delivery in a health centre compared'
                        'to a hospital'),
        'rrr_hc_delivery_parity_3_to_4': Parameter(
            Types.REAL, 'relative risk ratio for a woman with a parity of 3-4 delivering in a health centre compared to'
                        'a hospital'),
        'rrr_hc_delivery_parity_>4': Parameter(
            Types.REAL, 'relative risk ratio of a woman with a parity >4 delivering in health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_married': Parameter(
            Types.REAL, 'relative risk ratio of a married woman delivering in a health centre compared to a hospital'),
        'odds_deliver_at_home': Parameter(
            Types.REAL, 'odds of a woman delivering at home compared to a hospital'),
        'rrr_hb_delivery_age_35_39': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 35-39 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_age_40_44': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 40-44 delivering at home compared to a hospital'),
        'rrr_hb_delivery_age_45_49': Parameter(
            Types.REAL, 'relative risk ratio for a woman aged 45-49 delivering at home compared to a hospital'),

        'rrr_hb_delivery_parity_3_to_4': Parameter(
            Types.REAL, 'relative risk ratio for a woman with a parity of 3-4 delivering at home compared to'
                        'a hospital'),
        'rrr_hb_delivery_parity_>4': Parameter(
            Types.REAL, 'relative risk ratio of a woman with a parity >4 delivering at home compared to a '
                        'hospital'),
        'odds_careseeking_for_complication': Parameter(
            Types.REAL, 'odds of a woman seeking skilled assistance after developing a complication at a home birth'),
        'or_comp_careseeking_wealth_2': Parameter(
            Types.REAL, 'odds ratio of a woman of wealth level 2 seeking assistance after developing a complication at '
                        'a home birth '),

        # ================================= TREATMENT PARAMETERS =====================================================
        'rr_maternal_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following clean birth practices employed in a facility'),
        'rr_newborn_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following clean birth practices employed in a facility'),
        'rr_sepsis_post_abx_prom': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following prophylactic antibiotics for PROM in a facility'),
        'rr_sepsis_post_abx_pprom': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following prophylactic antibiotics for PPROM in a facility'),
        'rr_pph_amtsl': Parameter(
            Types.REAL, 'relative risk of severe post partum haemorrhage following active management of the third '
                        'stage of labour'),
        'prob_cure_uterotonics': Parameter(
            Types.REAL, 'probability of uterotonics stopping a postpartum haemorrhage due to uterine atony '),
        'prob_successful_manual_removal_placenta': Parameter(
            Types.REAL, 'probability of manual removal of retained products arresting a post partum haemorrhage'),
        'success_rate_pph_surgery': Parameter(
            Types.REAL, 'probability of surgery for postpartum haemorrhage being successful'),
        'success_rate_surgical_removal_placenta': Parameter(
            Types.REAL, 'probability of surgery for retained placenta being successful'),
        'success_rate_uterine_repair': Parameter(
            Types.REAL, 'probability repairing a ruptured uterus surgically'),
        'prob_successful_assisted_vaginal_delivery': Parameter(
            Types.REAL, 'probability of successful assisted vaginal delivery'),

        'squeeze_factor_threshold_delivery_attendance': Parameter(
            Types.REAL, 'dummy squeeze factor threshold after which delivery will not be attended '),
        'sensitivity_of_assessment_of_obstructed_labour_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a health '
                        'centre'),
        'sensitivity_of_assessment_of_obstructed_labour_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_sepsis_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for maternal sepsis in a health centre'),
        'sensitivity_of_assessment_of_sepsis_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for maternal sepsis in a level 1'
                        'hospital'),
        'sensitivity_of_assessment_of_severe_pe_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for severe pre-eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_severe_pe_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for severe pre-eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_hypertension_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for hypertension in a level 1 health '
                        'centre'),
        'sensitivity_of_assessment_of_hypertension_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for hypertension in a level 1 hospital'),
        'sensitivity_of_referral_assessment_of_antepartum_haem_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment for referral by birth attendant for antepartum haemorrhage'
                        ' in a health centre'),
        'sensitivity_of_treatment_assessment_of_antepartum_haem_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment for treatment by birth attendant for antepartum haemorrhage'
                        ' in a level 1 hospital'),
        'sensitivity_of_referral_assessment_of_uterine_rupture_hc': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a health centre'),
        'sensitivity_of_treatment_assessment_of_uterine_rupture_hp': Parameter(
            Types.REAL, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a level 1 '
                        'hospital'),
        'obstructed_labour_delayed_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of delayed treatment for obstructed labour on risk of intrapartum stillbirth'),
        'obstructed_labour_prompt_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of prompt treatment for obstructed labour on risk of intrapartum stillbirth'),
        'sepsis_delayed_treatment_effect_md': Parameter(
            Types.REAL, 'effect of delayed treatment for sepsis on risk of maternal death'),
        'sepsis_prompt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of prompt treatment for sepsis on risk of maternal death'),
        'sepsis_delayed_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of delayed treatment for sepsis on risk of intrapartum stillbirth'),
        'sepsis_prompt_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of prompt treatment for sepsis on risk of intrapartum stillbirth'),
        'eclampsia_treatment_effect_severe_pe': Parameter(
            Types.REAL, 'effect of treatment for severe pre eclampsia on risk of eclampsia'),
        'eclampsia_treatment_effect_md': Parameter(
            Types.REAL, 'effect of treatment for eclampsia on risk of maternal death'),
        'eclampsia_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of treatment for eclampsia on risk of intrapartum stillbirth'),
        'aph_prompt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of prompt treatment for antepartum haemorrhage on risk of maternal death'),
        'aph_delayed_treatment_effect_md': Parameter(
            Types.REAL, 'effect of delayed treatment for antepartum haemorrhage on risk of maternal death'),
        'aph_bt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of blood transfusion treatment for antepartum haemorrhage on risk of maternal death'),
        'aph_prompt_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of prompt treatment for antepartum haemorrhage on risk of intrapartum stillbirth'),
        'aph_delayed_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of delayed treatment for antepartum haemorrhage on risk of intrapartum stillbirth'),
        'pph_delayed_treatment_effect_md': Parameter(
            Types.REAL, 'effect of delayed treatment of postpartum haemorrhage on risk of maternal death'),
        'pph_prompt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of prompt treatment of postpartum haemorrhage on risk of maternal death'),
        'pph_bt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of blood transfusion treatment for postpartum haemorrhage on risk of maternal death'),
        'ur_prompt_treatment_effect_md': Parameter(
            Types.REAL, 'effect of prompt treatment for uterine rupture on risk of maternal death '),
        'ur_delayed_treatment_effect_md': Parameter(
            Types.REAL, 'effect of delayed treatment for uterine rupture on risk of maternal death '),
        'ur_prompt_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of prompt treatment for uterine rupture on risk of intrapartum stillbirth'),
        'ur_delayed_treatment_effect_sb': Parameter(
            Types.REAL, 'effect of delayed treatment for uterine rupture on risk of intrapartum stillbirth'),
        'allowed_interventions': Parameter(
            Types.LIST, 'list of interventions allowed to run, used in analysis'),

        # ================================= DALY WEIGHT PARAMETERS =====================================================
        'daly_wt_haemorrhage_moderate': Parameter(
            Types.REAL, 'DALY weight for a moderate maternal haemorrhage (<1 litre)'),
        'daly_wt_haemorrhage_severe': Parameter(
            Types.REAL, 'DALY weight for a severe maternal haemorrhage (>1 litre)'),
        'daly_wt_maternal_sepsis': Parameter(
            Types.REAL, 'DALY weight for maternal sepsis'),
        'daly_wt_eclampsia': Parameter(
            Types.REAL, 'DALY weight for eclampsia'),
        'daly_wt_obstructed_labour': Parameter(
            Types.REAL, 'DALY weight for obstructed labour')
    }

    PROPERTIES = {
        'la_due_date_current_pregnancy': Property(Types.DATE, 'The date on which a newly pregnant woman is scheduled to'
                                                              ' go into labour'),
        'la_currently_in_labour': Property(Types.BOOL, 'whether this woman is currently in labour'),
        'la_intrapartum_still_birth': Property(Types.BOOL, 'whether this womans most recent pregnancy has ended '
                                                           'in a stillbirth'),
        'la_parity': Property(Types.REAL, 'total number of previous deliveries'),
        'la_previous_cs_delivery': Property(Types.BOOL, 'whether this woman has ever delivered via caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour': Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_obstructed_labour_disab': Property(Types.BOOL, 'disability associated with obstructed labour'),
        'la_obstructed_labour_treatment': Property(Types.BOOL, 'If this woman has received treatment for '
                                                               'obstructed labour'),
        'la_antepartum_haem': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this'
                                                   'delivery'),
        'la_antepartum_haem_treatment': Property(Types.BOOL, 'whether this womans antepartum haemorrhage has been '
                                                             'treated'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this '
                                                   'delivery'),
        'la_uterine_rupture_disab': Property(Types.BOOL, 'disability associated with uterine rupture'),
        'la_uterine_rupture_treatment': Property(Types.BOOL, 'whether this womans uterine rupture has been treated'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_sepsis_postpartum': Property(Types.BOOL, 'whether the woman has developed sepsis following delivery'),
        'la_sepsis_disab': Property(Types.BOOL, 'disability associated with maternal sepsis'),
        'la_sepsis_treatment': Property(Types.BOOL, 'If this woman has received treatment for maternal sepsis'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_eclampsia_postpartum': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure '
                                                        'following this delivery'),
        'la_eclampsia_disab': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        'la_eclampsia_treatment': Property(Types.BOOL, 'whether this womans eclampsia has been treated'),
        'la_severe_pre_eclampsia_treatment': Property(Types.BOOL, 'whether this woman has been treated for severe '
                                                                  'pre-eclampsia'),
        'la_maternal_hypertension_treatment': Property(Types.BOOL, 'whether this woman has been treated for maternal '
                                                                   'hypertension'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_treatment': Property(Types.BOOL, 'If this woman has received treatment for '
                                                             'postpartum haemorrhage'),
        'la_maternal_haem_non_severe_disab': Property(Types.BOOL, 'disability associated with non severe maternal '
                                                                  'haemorrhage'),
        'la_maternal_haem_severe_disab': Property(Types.BOOL, 'disability associated with severe maternal haemorrhage'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_maternal_death_in_labour': Property(Types.BOOL, ' whether the woman has died as a result of this '
                                                            'pregnancy'),
        'la_maternal_death_in_labour_date': Property(Types.DATE, 'date of death for a date in pregnancy')
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Here we will include DALY weights if applicable...
        if 'HealthBurden' in self.sim.modules:
            params['la_daly_wts'] = {
                'haemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339),
                'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340),
                'eclampsia': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861),
                'obstructed_labour': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348),
                'uterine_rupture': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
            }

        # ======================================= LINEAR MODEL EQUATIONS ============================
        # Here we define the equations that will be used throughout this module using the linear
        # model and stored them as a parameter
        # TODO: Predictors that may affect incidence are currently commented out for testing,
        #       predictors which denote treatment effect are left in

        params['la_labour_equations'] = {
            'parity': LinearModel(
                LinearModelType.ADDITIVE,
                -3,
                # params['intercept_parity_lr2010'])),
                Predictor('age_years').apply(lambda age_years: (age_years * 0.22)),
                Predictor('li_mar_stat').when('2', 0.91)  # params['effect_mar_stat_2_parity_lr2010'])
                                        .when('3', 0.16),  # params['effect_mar_stat_3_parity_lr2010']),
                Predictor('li_wealth').when('5', -0.13)  # params['effect_wealth_lev_5_parity_lr2010'])
                                      .when('4', -0.13)  # params['effect_wealth_lev_4_parity_lr2010'])
                                      .when('3', -0.26)  # params['effect_wealth_lev_3_parity_lr2010'])
                                      .when('2', -0.37)  # params['effect_wealth_lev_2_parity_lr2010'])
                                      .when('1', -0.9)  # params['effect_wealth_lev_1_parity_lr2010']),
            ),

            # TODO: For some reason using parameters, with the exact same values, is making the result come out as
            # a minus figure and I'm unsure why

            'early_preterm_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_early_ptb'],
                # Predictor('age_years').when('.between(15,19)', params['or_early_ptb_age<20']),
                # Predictor('la_has_previously_delivered_preterm').when(True, params['or_early_ptb_prev_ptb']),
            ),

            'late_preterm_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_late_ptb'],
                # Predictor('la_has_previously_delivered_preterm').when(True, params['or_late_ptb_prev_ptb']),
            ),

            'post_term_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_post_term_labour']
                # Predictor('li_bmi').when('>24', params['rrr_ptl_bmi_more25']),
                # Predictor('li_mar_stat').when('1', params['rrr_ptl_not_married'])
                #                        .when('3', params['rrr_ptl_not_married']),
            ),

            'obstructed_labour_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pl_ol']
                # Predictor('la_parity').when('0', params['rr_PL_OL_nuliparity']),
                # Predictor('la_parity').when('>2', params['rr_PL_OL_para_more3']),
                # Predictor('li_bmi').when('<18', params['rr_PL_OL_bmi_less18']),
                # Predictor('li_bmi').when('>24', params['rr_PL_OL_bmi_more25']),
                # Predictor('age_years').when('<20', params['rr_PL_OL_age_less20']),
            ),

            'obstructed_labour_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_obstructed_labour'],
                Predictor('la_maternal_death_in_labour').when(True,
                                                              params['rr_still_birth_ol_maternal_death']),
                Predictor().when('la_obstructed_labour_treatment & __attended_delivery__',
                                 params['obstructed_labour_prompt_treatment_effect_sb']),
                Predictor().when('la_obstructed_labour_treatment & ~__attended_delivery__',
                                 params['obstructed_labour_delayed_treatment_effect_sb'])
            ),

            'sepsis_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ip_sepsis'],
                # Predictor('la_obstructed_labour').when(True, params['rr_ip_sepsis_pl_ol']),
            ),

            'sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_sepsis'],
                Predictor().when('la_sepsis_treatment & __attended_delivery__',
                                 params['sepsis_prompt_treatment_effect_md']),
                Predictor().when('la_sepsis_treatment & ~__attended_delivery__',
                                 params['sepsis_delayed_treatment_effect_md'])
            ),

            'sepsis_pp': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_pp_sepsis'],
                Predictor('received_clean_delivery', external=True).when(True,
                                                                         params['rr_maternal_sepsis_clean_delivery']),
                Predictor('received_abx_for_prom', external=True).when(True,
                                                                       params['rr_sepsis_post_abx_prom']),
                Predictor('received_abx_for_pprom', external=True).when(True,
                                                                        params['rr_sepsis_post_abx_pprom'])
                # Predictor('li_urban').when(False, params['or_pp_sepsis_rural']),
                # Predictor('li_ed_lev').when('1', params['or_pp_sepsis_no_edu']),
                # Predictor('li_ed_lev').when('2', params['or_pp_sepsis_primary_edu']),
                # Predictor('delivery_type', external=True).when('instrumental', params['or_pp_sepsis_avd'])
                #                                         .when('caesarean_section', params['or_pp_sepsis_cs']),
                # Predictor('ps_premature_rupture_of_membranes').when(True, params['or_pp_sepsis_prom']),
            ),

            'sepsis_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_sepsis'],
                Predictor().when('la_sepsis_treatment & __attended_delivery__',
                                 params['sepsis_prompt_treatment_effect_md']),
                Predictor().when('la_sepsis_treatment & ~__attended_delivery__',
                                 params['sepsis_delayed_treatment_effect_md'])
            ),

            'sepsis_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_sepsis'],
                Predictor('la_maternal_death_in_labour').when(True, params['rr_still_birth_sepsis_maternal_death']),
                Predictor().when('la_sepsis_treatment & __attended_delivery__',
                                 params['sepsis_prompt_treatment_effect_sb']),
                Predictor().when('la_sepsis_treatment & ~__attended_delivery__',
                                 params['sepsis_delayed_treatment_effect_sb'])
            ),

            'eclampsia_ip': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_ip_eclampsia'],
                # Predictor('age_years').when('.between(30,34)', params['or_ip_eclampsia_30_34']),
                # Predictor('age_years').when('>35', params['or_ip_eclampsia_35']),
                # Predictor('la_parity').when('0', params['or_ip_eclampsia_nullip']),
                # Predictor('ps_gest_diab').when(True, params['or_ip_eclampsia_gest_diab']),
                Predictor('la_severe_pre_eclampsia_treatment').when(True,
                                                                    params['eclampsia_treatment_effect_severe_pe'])
            ),

            'eclampsia_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_md'])
            ),

            'eclampsia_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_eclampsia'],
                Predictor('la_maternal_death_in_labour').when(True, params['rr_still_birth_eclampsia_maternal_death']),
                Predictor('la_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_sb'])),

            'eclampsia_pp': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_ip_eclampsia'],
                # Predictor('age_years').when('.between(30,34)', params['or_ip_eclampsia_30_34']),
                # Predictor('age_years').when('>35', params['or_ip_eclampsia_35']),
                # Predictor('la_parity').when('0', params['or_ip_eclampsia_nullip']),
                # Predictor('ps_gest_diab').when(True, params['or_ip_eclampsia_gest_diab']),
                Predictor('la_severe_pre_eclampsia_treatment').when(True,
                                                                    params['eclampsia_treatment_effect_severe_pe'])
            ),

            'eclampsia_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_eclampsia'],
                Predictor('la_eclampsia_treatment').when(True, params['eclampsia_treatment_effect_md'])
            ),

            'severe_pre_eclamp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_severe_pre_eclamp']
            ),

            'severe_pre_eclamp_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_severe_pre_eclamp']
            ),

            'antepartum_haem_ip': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_aph']
                # Predictor('la_obstructed_labour').when(True, params['rr_aph_pl_ol']),
            ),


            'antepartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_aph'],
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "prompt_referral")',
                                 params['aph_prompt_treatment_effect_md']),
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "delayed_referral")',
                                 params['aph_delayed_treatment_effect_md']),
                Predictor('received_blood_transfusion', external=True).when(True, params['aph_bt_treatment_effect_md'])
            ),

            'antepartum_haem_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_antepartum_haem'],
                Predictor('la_maternal_death_in_labour').when(True, params['rr_still_birth_aph_maternal_death']),
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "prompt_referral")',
                                 params['aph_prompt_treatment_effect_sb']),
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "delayed_referral")',
                                 params['aph_delayed_treatment_effect_sb']),
                Predictor('received_blood_transfusion', external=True).when(True,
                                                                            params['aph_bt_treatment_effect_md'])
            ),

            'postpartum_haem_pp': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pph'],
                Predictor('received_amtsl', external=True).when(True, params['rr_pph_amtsl'])
                # Predictor('la_obstructed_labour').when(True, params['rr_pph_pl_ol']),
            ),

            'postpartum_haem_pp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_pp_pph'],
                Predictor().when('la_postpartum_haem_treatment & __attended_delivery__',
                                 params['pph_prompt_treatment_effect_md']),
                Predictor().when('la_postpartum_haem_treatment  & ~__attended_delivery__',
                                 params['pph_delayed_treatment_effect_md']),
                Predictor('received_blood_transfusion', external=True).when(True, params['pph_bt_treatment_effect_md'])
            ),

            'uterine_rupture_ip': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_uterine_rupture']
                #   Predictor('la_parity').when('>4', params['or_ur_grand_multip']),
                #   Predictor('la_previous_cs_delivery').when(True, params['or_ur_prev_cs']),
                #   Predictor('la_obstructed_labour').when(True, params['or_ur_ref_ol']),
                # TODO: to include la_obstructed_labour_treatment to be a predictor that reduces likelihood of uterine
                # rupture?
            ),

            'uterine_rupture_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_uterine_rupture'],
                Predictor().when('la_uterine_rupture_treatment & (__referral_timing_surgery__ == "prompt_referral")',
                                 params['ur_prompt_treatment_effect_md']),
                Predictor().when('la_uterine_rupture_treatment & ( __referral_timing_surgery__== "delayed_referral")',
                                 params['ur_delayed_treatment_effect_md'])
            ),

            'uterine_rupture_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_uterine_rupture'],
                Predictor('la_maternal_death_in_labour').when(True, params['rr_still_birth_ur_maternal_death']),
                Predictor().when('la_uterine_rupture_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__ == "prompt_referral")',
                                 params['ur_prompt_treatment_effect_sb']),
                Predictor().when('la_uterine_rupture_treatment & (__mode_of_delivery__ == "caesarean_section") &'
                                 '( __referral_timing_caesarean__== "delayed_referral")',
                                 params['ur_delayed_treatment_effect_sb'])
            ),

            'probability_delivery_health_centre': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_deliver_in_health_centre']
                # Predictor('age_years').when('.between(24,30)', params['rrr_hc_delivery_age_25_29'])
                #                      .when('.between(29,35)', params['rrr_hc_delivery_age_30_34'])
                #                      .when('.between(34,40)', params['rrr_hc_delivery_age_35_39'])
                #                      .when('.between(39,45)', params['rrr_hc_delivery_age_40_44'])
                #                     .when('.between(44,50)', params['rrr_hc_delivery_age_45_49']),
                # Predictor('li_urban').when(False, params['rrr_hc_delivery_rural']),
                # Predictor('la_parity').when('.between(2,5)', params['rrr_hc_delivery_parity_3_to_4'])
                #                      .when('>4', params['rrr_hc_delivery_parity_>4']),
                # Predictor('li_mar_stat').when('2', params['rrr_hc_delivery_married']),
            ),

            'probability_delivery_at_home': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['odds_deliver_at_home']
                # Predictor('age_years').when('.between(34,40)', params['rrr_hb_delivery_age_35_39'])
                #                      .when('.between(39,45)', params['rrr_hb_delivery_age_40_44'])
                #                      .when('.between(44,50)', params['rrr_hb_delivery_age_45_49']),
                # Predictor('la_parity').when('.between(2,5)', params['rrr_hb_delivery_parity_3_to_4'])
                #                      .when('>4', params['rrr_hb_delivery_parity_>4']),
            ),

            'care_seeking_for_complication': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_careseeking_for_complication'],
                # Predictor('li_wealth').when('2', params['or_comp_careseeking_wealth_2']),
            ),
        }

    def initialise_population(self, population):
        df = population.props
        params = self.parameters

        df.loc[df.is_alive, 'la_currently_in_labour'] = False
        df.loc[df.is_alive, 'la_intrapartum_still_birth'] = False
        df.loc[df.is_alive, 'la_parity'] = 0
        df.loc[df.is_alive, 'la_previous_cs_delivery'] = False
        df.loc[df.is_alive, 'la_has_previously_delivered_preterm'] = False
        df.loc[df.is_alive, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'la_obstructed_labour'] = False
        df.loc[df.is_alive, 'la_obstructed_labour_disab'] = False
        df.loc[df.is_alive, 'la_obstructed_labour_treatment'] = False
        df.loc[df.is_alive, 'la_antepartum_haem'] = False
        df.loc[df.is_alive, 'la_antepartum_haem_treatment'] = False
        df.loc[df.is_alive, 'la_uterine_rupture'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_disab'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_treatment'] = False
        df.loc[df.is_alive, 'la_sepsis'] = False
        df.loc[df.is_alive, 'la_sepsis_postpartum'] = False
        df.loc[df.is_alive, 'la_sepsis_disab'] = False
        df.loc[df.is_alive, 'la_sepsis_treatment'] = False
        df.loc[df.is_alive, 'la_eclampsia'] = False
        df.loc[df.is_alive, 'la_eclampsia_postpartum'] = False
        df.loc[df.is_alive, 'la_eclampsia_disab'] = False
        df.loc[df.is_alive, 'la_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_severe_pre_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_maternal_hypertension_treatment'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_postpartum_haem_treatment'] = False
        df.loc[df.is_alive, 'la_maternal_haem_non_severe_disab'] = False
        df.loc[df.is_alive, 'la_maternal_haem_severe_disab'] = False
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour_date'] = pd.NaT

        #  ----------------------------ASSIGNING PARITY AT BASELINE --------------------------------
        # We assign parity to all women of reproductive age at baseline
        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = np.around(
            params['la_labour_equations']['parity'].predict(df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)])
        )

    def initialise_simulation(self, sim):
        # We set the LoggingEvent to run a the last day of each year to produce statistics for that year
        sim.schedule_event(LabourLoggingEvent(self), sim.date + DateOffset(years=1))

        # This list contains all the women who are currently in labour
        self.women_in_labour = []

        # Create complication tracker
        self.labour_tracker = {'ip_stillbirth': 0, 'maternal_death': 0, 'obstructed_labour': 0,
                               'antepartum_haem': 0, 'antepartum_haem_death': 0, 'sepsis': 0, 'sepsis_death': 0,
                               'eclampsia': 0, 'eclampsia_death': 0, 'uterine_rupture': 0, 'uterine_rupture_death': 0,
                               'postpartum_haem': 0, 'postpartum_haem_death': 0, 'sepsis_postpartum': 0,
                               'eclampsia_postpartum': 0, 'home_birth': 0, 'health_centre_birth': 0,
                               'hospital_birth': 0, 'caesarean_section': 0, 'early_preterm': 0,
                               'late_preterm': 0, 'post_term': 0, 'term': 0}

        self.possible_intrapartum_complications = ['obstructed_labour', 'antepartum_haem', 'sepsis', 'eclampsia',
                                                   'uterine_rupture', 'severe_pre_eclamp']

        self.possible_postpartum_complications = ['sepsis', 'eclampsia', 'postpartum_haem']

        # =======================Register dx_tests for complications during labour/postpartum=======================
        # We register all the dx_tests needed within the labour HSI events. dx_tests in this module represent assessment
        # and correct diagnosis of key complication, leading to treatment or referral for treatment.

        # Sensitivity of testing varies between health centres and hospitals...
        # hp = hospital, hc= health centre
        p = self.parameters

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            # Obstructed Labour diagnosis
            assess_obstructed_labour_hc=DxTest(
                property='la_obstructed_labour',
                sensitivity=p['sensitivity_of_assessment_of_obstructed_labour_hc']),

            assess_obstructed_labour_hp=DxTest(
                property='la_obstructed_labour',
                sensitivity=p['sensitivity_of_assessment_of_obstructed_labour_hp']),

            # Sepsis diagnosis intrapartum...
            # dx_tests for intrapartum and postpartum sepsis only differ in the 'property' variable
            assess_sepsis_hc_ip=DxTest(
                property='la_sepsis',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),

            assess_sepsis_hp_ip=DxTest(
                property='la_sepsis',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),

            # Sepsis diagnosis postpartum
            assess_sepsis_hc_pp=DxTest(
                property='la_sepsis_postpartum',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),

            assess_sepsis_hp_pp=DxTest(
                property='la_sepsis_postpartum',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),

            # Hypertension diagnosis
            assess_hypertension_hc=DxTest(
                property='ps_currently_hypertensive',
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hc']),

            assess_hypertension_hp=DxTest(
                property='ps_currently_hypertensive',
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hp']),

            # severe pre-eclampsia diagnosis
            assess_severe_pe_hc=DxTest(
                property='ps_severe_pre_eclamp',
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hc']),

            assess_severe_pe_hp=DxTest(
                property='ps_severe_pre_eclamp',
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hp']),

            # Antepartum Haemorrhage
            assess_for_referral_aph_hc=DxTest(
                property='la_antepartum_haem',
                sensitivity=p['sensitivity_of_referral_assessment_of_antepartum_haem_hc']),

            assess_for_treatment_aph_hp=DxTest(
                property='la_antepartum_haem',
                sensitivity=p['sensitivity_of_treatment_assessment_of_antepartum_haem_hp']),

            # Uterine Rupture
            assess_for_referral_uterine_rupture_hc=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_referral_assessment_of_uterine_rupture_hc']),

            assess_for_treatment_uterine_rupture_hp=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_treatment_assessment_of_uterine_rupture_hp'])
        )

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        mother = df.loc[mother_id]

        df.at[child_id, 'la_due_date_current_pregnancy'] = pd.NaT
        df.at[child_id, 'la_currently_in_labour'] = False
        df.at[child_id, 'la_intrapartum_still_birth'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs_delivery'] = False
        df.at[child_id, 'la_has_previously_delivered_preterm'] = False
        df.at[child_id, 'la_obstructed_labour'] = False
        df.at[child_id, 'la_obstructed_labour_disab'] = False
        df.at[child_id, 'la_obstructed_labour_treatment'] = False
        df.at[child_id, 'la_antepartum_haem'] = False
        df.at[child_id, 'la_antepartum_haem_treatment'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_uterine_rupture_disab'] = False
        df.at[child_id, 'la_uterine_rupture_treatment'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_sepsis_postpartum'] = False
        df.at[child_id, 'la_sepsis_disab'] = False
        df.at[child_id, 'la_sepsis_treatment'] = False
        df.at[child_id, 'la_eclampsia'] = False
        df.at[child_id, 'la_eclampsia_postpartum'] = False
        df.at[child_id, 'la_eclampsia_disab'] = False
        df.at[child_id, 'la_eclampsia_treatment'] = False
        df.at[child_id, 'la_severe_pre_eclampsia_treatment'] = False
        df.at[child_id, 'la_maternal_hypertension_treatment'] = False
        df.at[child_id, 'la_postpartum_haem'] = False
        df.at[child_id, 'la_postpartum_haem_treatment'] = False
        df.at[child_id, 'la_maternal_haem_non_severe_disab'] = False
        df.at[child_id, 'la_maternal_haem_severe_disab'] = False
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_maternal_death_in_labour'] = False
        df.at[child_id, 'la_maternal_death_in_labour_date'] = pd.NaT

        # If a mothers labour has resulted in an intrapartum still birth her child is still generated by the simulation
        # but the death is recorded through the InstantaneousDeath function

        # Store only live births to a mother parity
        if ~mother.la_intrapartum_still_birth:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity

        if mother.la_intrapartum_still_birth:
            #  N.B this will only record intrapartum stillbirth
            death = demography.InstantaneousDeath(self.sim.modules['Demography'],
                                                  child_id,
                                                  cause='intrapartum stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.at[mother_id, 'la_intrapartum_still_birth'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """ This is called whenever there is an HSI event commissioned by one of the other disease modules."""
        logger.info('This is Labour, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        logger.debug('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties data frame
        p = self.parameters

        health_values_1 = df.loc[df.is_alive, 'la_obstructed_labour_disab'].map(
            {False: 0, True: p['la_daly_wts']['obstructed_labour']})
        health_values_1.name = 'Obstructed Labour'

        health_values_2 = df.loc[df.is_alive, 'la_eclampsia_disab'].map(
            {False: 0, True: p['la_daly_wts']['eclampsia']})
        health_values_2.name = 'Eclampsia'

        health_values_3 = df.loc[df.is_alive, 'la_sepsis_disab'].map(
            {False: 0, True: p['la_daly_wts']['maternal_sepsis']})
        health_values_3.name = 'Maternal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'la_maternal_haem_non_severe_disab'].map(
            {False: 0, True: p['la_daly_wts']['haemorrhage_moderate']})
        health_values_4.name = 'Non Severe Maternal Haemorrhage'

        health_values_5 = df.loc[df.is_alive, 'la_maternal_haem_severe_disab'].map(
            {False: 0, True: p['la_daly_wts']['haemorrhage_severe']})
        health_values_5.name = 'Severe Maternal Haemorrhage'

        health_values_6 = df.loc[df.is_alive, 'la_uterine_rupture_disab'].map(
            {False: 0, True: p['la_daly_wts']['uterine_rupture']})
        health_values_6.name = 'Uterine Rupture'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive], health_values_6.loc[df.is_alive]], axis=1)

        # Must not have one person with more than 1.00 daly weight
        # Hot fix - scale such that sum does not exceed one.
        scaling_factor = (health_values_df.sum(axis=1).clip(lower=0, upper=1) /
                          health_values_df.sum(axis=1)).fillna(1.0)
        health_values_df = health_values_df.multiply(scaling_factor, axis=0)

        return health_values_df
        # TODO: await change of formatting for DALYs by Tim H (?)

    # ===================================== LABOUR SCHEDULER =======================================

    def set_date_of_labour(self, individual_id):
        """This function, called by the contraception module, uses linear equations to determine when a woman will go
        into labour from a future date"""

        df = self.sim.population.props
        eqs = self.parameters['la_labour_equations']
        logger.debug('person %d is having their labour scheduled on date %s', individual_id, self.sim.date)

        # Check only alive newly pregnant women are scheduled to this function
        assert df.at[individual_id, 'is_alive'] and df.at[individual_id, 'is_pregnant']
        assert df.at[individual_id, 'date_of_last_pregnancy'] == self.sim.date

        # Using the linear equations defined above we calculate this womans individual risk of early and late preterm
        # labour
        eptb_prob = eqs['early_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]
        lptb_prob = eqs['late_preterm_birth'].predict(df.loc[[individual_id]])[individual_id]
        ptl_prob = eqs['post_term_birth'].predict(df.loc[[individual_id]])[individual_id]

        denom = 1 + eptb_prob + lptb_prob
        prob_ep = eptb_prob / denom
        prob_lp = lptb_prob / denom
        prob_term = 1 / denom

        labour_type = ['early_preterm', 'late_preterm', 'term']
        probabilities = [prob_ep, prob_lp, prob_term]
        random_draw = self.rng.choice(labour_type, p=probabilities)

        # Depending on the result from the random draw we determine in how many days this woman will go into labour
        if random_draw == 'early_preterm':
            df.at[individual_id, 'la_due_date_current_pregnancy'] = (
                df.at[individual_id, 'date_of_last_pregnancy'] + pd.DateOffset(days=7 * 24 + self.rng.randint(0, 7 * 9))
            )

        elif random_draw == 'late_preterm':
            df.at[individual_id, 'la_due_date_current_pregnancy'] = (
                df.at[individual_id, 'date_of_last_pregnancy'] + pd.DateOffset(days=7 * 34 + self.rng.randint(0, 7 * 3))
            )

        # For women who will deliver after term we apply a risk of post term birth
        else:
            if self.rng.random_sample() < ptl_prob:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = (
                    df.at[individual_id, 'date_of_last_pregnancy'] +
                    pd.DateOffset(days=7 * 42 + self.rng.randint(0, 7 * 4))
                )
            else:
                df.at[individual_id, 'la_due_date_current_pregnancy'] = (
                    df.at[individual_id, 'date_of_last_pregnancy'] +
                    pd.DateOffset(days=7 * 37 + self.rng.randint(0, 7 * 4))
                )

        # Here we check that no one can go into labour before 24 weeks gestation
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(168, unit='d')

        # and then we schedule the labour for that womans due date
        self.sim.schedule_event(LabourOnsetEvent(self, individual_id),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])

    # ===================================== HELPER AND TESTING FUNCTIONS ==============================================
    # The following functions are called throughout the Labour module and HSIs

    def predict(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        person = df.loc[[person_id]]
        # TODO: replace with built in LM evaluate function

        # We define specific external variables used as predictors in the equations defined below
        has_rbt = mni[person_id]['received_blood_transfusion']
        mode_of_delivery = mni[person_id]['mode_of_delivery']
        received_clean_delivery = mni[person_id]['clean_delivery_kit_used']
        received_abx_for_prom = mni[person_id]['abx_for_prom_given']
        received_abx_for_pprom = mni[person_id]['abx_for_pprom_given']
        received_amtsl = mni[person_id]['amtsl_given']
        attended_delivery = mni[person_id]['delivery_attended']
        referral_timing_surgery = mni[person_id]['referred_for_surgery']
        referral_timing_caesarean = mni[person_id]['referred_for_cs']

        return self.rng.random_sample() < eq.predict(person,
                                                     received_clean_delivery=received_clean_delivery,
                                                     received_abx_for_prom=received_abx_for_prom,
                                                     received_abx_for_pprom=received_abx_for_pprom,
                                                     delivery_type=mode_of_delivery,
                                                     mode_of_delivery=mode_of_delivery,
                                                     received_amtsl=received_amtsl,
                                                     received_blood_transfusion=has_rbt,
                                                     referral_timing_surgery=referral_timing_surgery,
                                                     attended_delivery=attended_delivery,
                                                     referral_timing_caesarean=referral_timing_caesarean)[person_id]

    def set_intrapartum_complications(self, individual_id, complication):
        """Uses the result of a linear equation to determine the probability of a complication occuring during
        delivery and makes changes to the appropriate properties (the complication and its associated disability)
        in the data frame dependent on the result."""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info

        assert mni[individual_id]['delivery_setting'] != 'none'
        assert complication in self.possible_intrapartum_complications

        if self.predict(params['la_labour_equations'][f'{complication}_ip'], individual_id):
            # If the woman will experience a complication we make changes to the data frame and store the complication
            # in a tracker
            df.at[individual_id, f'la_{complication}'] = True
            self.labour_tracker[f'{complication}'] += 1

            if complication == 'antepartum_haem':
                # Severity of bleeding is assigned if a woman is experience an antepartum haemorrhage to map to DALY
                # weights
                random_choice = self.rng.choice(['non_severe', 'severe'],
                                                p=params['severity_maternal_haemorrhage'])
                if random_choice == 'non_severe':
                    df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[individual_id, 'la_maternal_haem_severe_disab'] = True
            else:
                df.at[individual_id, f'la_{complication}_disab'] = True

            logger.debug(f'person %d has developed {complication} during birth on date f %s', individual_id,
                         self.sim.date)

    def set_postpartum_complications(self, individual_id, complication):
        """Uses the result of a linear equation to determine the probability of a certain complications occuring
        following delivery at home and makes changes to the appropriate variables in the data frame dependent on the
        result."""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        assert mni[individual_id]['delivery_setting'] != 'none'
        assert complication in self.possible_postpartum_complications

        if self.predict(params['la_labour_equations'][f'{complication}_pp'], individual_id):
            if complication == 'eclampsia' or complication == 'sepsis':
                df.at[individual_id, f'la_{complication}_postpartum'] = True
                self.labour_tracker[f'{complication}_postpartum'] += 1
                df.at[individual_id, f'la_{complication}_disab'] = True

            # TODO: (IMPORTANT) apply distribution of onset of postpartum sepsis, days 0-7, and add code to the
            #  postpartum module to give these women symptoms and seek care

            if complication == 'postpartum_haem':
                # Severity of bleeding is assigned if a woman experiences a postpartum haemorrhage to map to DALY
                # weights
                df.at[individual_id, f'la_{complication}'] = True
                self.labour_tracker[f'{complication}'] += 1

                mni[individual_id]['source_pph'] = self.rng.choice(['uterine_atony', 'retained_placenta'],
                                                                   size=1, p=params['prob_pph_source'])
                random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                p=params['severity_maternal_haemorrhage'])

                if random_choice == 'non_severe':
                    df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = True
                else:
                    df.at[individual_id, 'la_maternal_haem_severe_disab'] = True

            logger.debug(f'person %d has developed {complication} during the postpartum phase of a birth on date '
                         f'%s', individual_id, self.sim.date)

    def set_maternal_death_status_intrapartum(self, individual_id, cause):
        """This function calculates an associated risk of death for a woman who has experience a complication during
        labour and makes appropriate changes to the data frame"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        assert cause in self.possible_intrapartum_complications

        # First we determine if this woman will die of the complication defined in the function
        if self.predict(params['la_labour_equations'][f'{cause}_death'], individual_id):
            logger.debug(f'{cause} has contributed to person %d death during labour', individual_id)

            mni[individual_id]['death_in_labour'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(cause)
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date
            self.labour_tracker[f'{cause}_death'] += 1

        #  And then a risk of stillbirth is calculated, maternal death is a used as a predictor and set very high if
        #  true
        if self.predict(params['la_labour_equations'][f'{cause}_stillbirth'], individual_id):
            logger.debug(f'person %d has experienced a still birth following {cause} in labour')
            df.at[individual_id, 'la_intrapartum_still_birth'] = True
            df.at[individual_id, 'ps_previous_stillbirth'] = True

    def set_maternal_death_status_postpartum(self, individual_id, cause):
        """This function calculates an associated risk of death for a woman who has experience a complication following
        labour and makes appropriate changes to the data frame"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters

        assert cause in self.possible_postpartum_complications

        if self.predict(params['la_labour_equations'][f'{cause}_pp_death'], individual_id):
            logger.debug(f'{cause} has contributed to person %d death following labour', individual_id)

            self.labour_tracker[f'{cause}_death'] += 1
            mni[individual_id]['death_postpartum'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date

    def labour_characteristics_checker(self, individual_id):
        """This function is called at different points in the module to ensure women of the right characteristics are
        in labour. This function doesnt check for a woman being pregnant or alive, as some events will still run despite
         those variables being set to false"""
        df = self.sim.population.props
        mother = df.loc[individual_id]

        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour
        assert mother.ps_gestational_age_in_weeks > 22

    def postpartum_characteristics_checker(self, individual_id):
        """This function is called at different points in the module to ensure women of the right characteristics are
        in labour. This function doesnt check for a woman being pregnant or alive, as some events will still run despite
         those variables being set to false"""
        df = self.sim.population.props
        mother = df.loc[individual_id]

        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour
        assert mother.ps_gestational_age_in_weeks == 0

    def events_queue_checker(self, individual_id):
        """ This function checks a womans event and hsi_event queues to ensure that she has the correct events scheduled
         during the process of labour"""
        mni = self.mother_and_newborn_info
        events = self.sim.find_events_for_person(person_id=individual_id)

        # Here we iterate through each womans event queue to insure she has the correct events scheduled
        # Firstly we check all women have the labour death event and birth event scheduled
        events = [e.__class__ for d, e in events]
        assert LabourDeathEvent in events
        assert BirthEvent in events

        # Then we ensure that women delivering in a facility have the right HSI scheduled
        if mni[individual_id]['delivery_setting'] != 'home_birth':
            health_system = self.sim.modules['HealthSystem']
            # If the health system is disabled, then the HSI event was wrapped in an HSIEventWrapper
            # and put in the simulation event queue. Hence, we only check HSI event if the HSI queue
            # for HSI event if the HSI event is enabled
            if not health_system.disabled:
                hsi_events = health_system.find_events_for_person(person_id=individual_id)
                hsi_events = [e.__class__ for d, e in hsi_events]
                assert HSI_Labour_PresentsForSkilledBirthAttendanceInLabour in hsi_events

    # ============================================== HSI FUNCTIONS ====================================================
    # Management of each complication is housed within its own function, defined here in the module, and all follow a
    # similar pattern ...
    #                   a.) The required consumables for the intervention(s) are defined
    #                   b.) The woman is assessed for a complication using the dx_test function. Specificity of
    #                       assessment varies between facility type (hospital or health centre)
    #                   c.) If she has the complication and it is correctly identified by HCWs, they check
    #                       consumables are available
    #                   d.) If the consumables are available- she will receive treatment

    # If the woman is delivering unattended, we assume any treatment she receives is delayed, delayed treatment is
    # less effective meaning risk of death is increased

    # In the instance that treatment cannot be delivered at the facility type a woman has delivered at (i.e.
    # health centres do not perform caesarean sections) she is assessed for referral and sent to another
    # facility

    def prophylactic_labour_interventions(self, hsi_event):
        """This function houses prophylactic interventions delivered by a Skilled Birth Attendant to women in labour.
        It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'prophylactic_labour_interventions' not in params['allowed_interventions']:
            return
        else:
            #  We define all the possible consumables that could be required
            pkg_code_uncomplicated_delivery = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Vaginal delivery - skilled attendance',
                                                                   'Intervention_Pkg_Code'])[0]

            # A clean delivery kit to reduce probability of maternal and newborn sepsis...
            pkg_code_clean_delivery_kit = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Clean practices and immediate essential newborn '
                                                                   'care (in facility)', 'Intervention_Pkg_Code'])[0]

            # Antibiotics for women whose membranes have ruptured prematurely...
            item_code_abx_prom = pd.unique(
                consumables.loc[consumables['Items'] == 'Benzylpenicillin 1g (1MU), PFR_Each_CMST', 'Item_Code'])[0]
            pkg_code_pprom = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM', 'Intervention_Pkg_Code'])[0]

            # Steroids for women who have gone into preterm labour to improve newborn outcomes...
            item_code_steroids_prem_dexamethasone = pd.unique(
                consumables.loc[consumables['Items'] == 'Dexamethasone 5mg/ml, 5ml_each_CMST', 'Item_Code'])[0]
            item_code_steroids_prem_betamethasone = pd.unique(
                consumables.loc[consumables['Items'] == 'Betamethasone, 12 mg injection', 'Item_Code'])[0]

            # Antibiotics for women delivering preterm to reduce newborn risk of group b strep infection...
            item_code_antibiotics_gbs_proph = pd.unique(
                consumables.loc[consumables['Items'] == 'Benzylpenicillin 3g (5MU), PFR_each_CMST', 'Item_Code'])[0]

            consumables_attended_delivery = {
                'Intervention_Package_Code': {pkg_code_uncomplicated_delivery: 1, pkg_code_clean_delivery_kit: 1,
                                              pkg_code_pprom: 1},
                'Item_Code': {item_code_abx_prom: 3, item_code_steroids_prem_dexamethasone: 5,
                              item_code_steroids_prem_betamethasone: 2, item_code_antibiotics_gbs_proph: 3}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_attended_delivery,
                to_log=True)

            # Availability of consumables determines if the intervention is delivered...
            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_uncomplicated_delivery]:
                mni[person_id]['clean_delivery_kit_used'] = True
                logger.debug('This facility has delivery kits available and have been used for mother %d delivery.',
                             person_id)
            else:
                logger.debug('This facility has no delivery kits.')

            # Prophylactic antibiotics for premature rupture of membranes in term deliveries...
            if mni[person_id]['labour_state'] == 'term_labour' and df.at[person_id,
                                                                         'ps_premature_rupture_of_membranes']:
                if outcome_of_request_for_consumables['Item_Code'][item_code_abx_prom]:
                    mni[person_id]['abx_for_prom_given'] = True
                    logger.debug('This facility has provided antibiotics for mother %d who is a risk of sepsis due '
                                 'to PROM.', person_id)
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')

            # Prophylactic antibiotics for premature rupture of membranes in preterm deliveries...
            if (mni[person_id]['labour_state'] == 'early_preterm_labour' or
                mni[person_id]['labour_state'] == 'late_preterm_labour') and df.at[person_id,
                                                                                   'ps_premature_rupture_of_membranes']:

                if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pprom]:
                    mni[person_id]['abx_for_pprom_given'] = True
                    logger.debug('This facility has provided antibiotics for mother %d who is a risk of sepsis due '
                                 'to PPROM.', person_id)
                else:
                    logger.debug('This facility has no antibiotics for the treatment of PROM.')

            # Prophylactic steroids and antibiotics for women who have gone into preterm labour...
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or \
                    mni[person_id]['labour_state'] == 'late_preterm_labour':

                if outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_betamethasone] and \
                 outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_dexamethasone]:
                    mni[person_id]['corticosteroids_given'] = True
                    logger.debug('This facility has provided corticosteroids for mother %d who is in preterm labour',
                                 person_id)
                else:
                    logger.debug('This facility has no steroids for women in preterm labour.')

                if outcome_of_request_for_consumables['Item_Code'][item_code_antibiotics_gbs_proph]:
                    mni[person_id]['abx_for_preterm_given'] = True
                    logger.debug('This facility has provided antibiotics for mother %d whose baby is a risk of '
                                 'Group B strep', person_id)
                else:
                    logger.debug('This facility has no antibiotics for group B strep prophylaxis.')

    def assessment_and_treatment_of_severe_pre_eclampsia(self, hsi_event, facility_type):
        """This function defines the required consumables, determines correct diagnosis and administers an intervention
        to women suffering from severe pre-eclampsia in labour. It is called by
        HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        person_id = hsi_event.target

        if 'assessment_and_treatment_of_severe_pre_eclampsia' not in params['allowed_interventions']:
            return
        else:

            pkg_code_severe_preeclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_spe = {'Intervention_Package_Code': {pkg_code_severe_preeclampsia: 1},
                                      'Item_Code': {}}

            outcome_of_request_for_consumables_spe = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_spe)

        # Here we run a dx_test function to determine if the birth attendant will correctly identify this womans
        # severe pre-eclampsia, and therefore administer treatment
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run=f'assess_severe_pe_{facility_type}', hsi_event=hsi_event):

                if outcome_of_request_for_consumables_spe:
                    df.at[person_id, 'la_severe_pre_eclampsia_treatment'] = True
                    logger.debug('mother %d has has their severe pre-eclampsia identified during delivery. As '
                                 'consumables are available they will receive treatment', person_id)

                elif df.at[person_id, 'ps_severe_pre_eclamp']:
                    logger.debug('mother %d has not had their severe pre-eclampsia identified during delivery and will '
                                 'not be treated', person_id)

    def assessment_and_treatment_of_obstructed_labour(self, hsi_event, facility_type):
        """This function defines the required consumables, determines correct diagnosis and administers an intervention
        to women suffering from obstructed labour. It is called by
        HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""

        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_obstructed_labour' not in params['allowed_interventions']:
            return
        else:
            pkg_code_obstructed_labour = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of obstructed labour',
                                'Intervention_Pkg_Code'])[0]

            item_code_forceps = pd.unique(consumables.loc[consumables['Items'] == 'Forceps, obstetric', 'Item_Code'])[0]
            item_code_vacuum = pd.unique(consumables.loc[consumables['Items'] == 'Vacuum, obstetric', 'Item_Code'])[0]

            consumables_obstructed_labour = {'Intervention_Package_Code': {pkg_code_obstructed_labour: 1},
                                             'Item_Code': {item_code_forceps: 1, item_code_vacuum: 1}}

        #

            outcome_of_request_for_consumables_ol = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_obstructed_labour, to_log=True)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run=f'assess_obstructed_labour_{facility_type}', hsi_event=hsi_event):

                if outcome_of_request_for_consumables_ol:

                    logger.debug('mother %d has had her obstructed labour identified during delivery. Staff will '
                                 'attempt an assisted vaginal delivery as the equipment is available', person_id)
                    treatment_success = params['prob_successful_assisted_vaginal_delivery'] > self.rng.random_sample()

                    if treatment_success:
                        df.at[person_id, 'la_obstructed_labour_treatment'] = True
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    else:
                        logger.debug('Following a failed assisted vaginal delivery other %d will need additional '
                                     'treatment', person_id)

                        if mni[person_id]['delivery_attended']:
                            mni[person_id]['refer_for_cs'] = 'prompt_referral'
                        else:
                            mni[person_id]['refer_for_cs'] = 'delayed_referral'

            elif df.at[person_id, 'la_obstructed_labour']:
                logger.debug('mother %d has not had their obstructed labour identified during delivery and will not'
                             'be treated', person_id)

    def assessment_and_treatment_of_maternal_sepsis(self, hsi_event, facility_type, labour_stage):
        """This function defines the required consumables, determines correct diagnosis and administers intervention
        to women suffering from maternal sepsis. It is called by either
        HSI_Labour_PresentsForSkilledBirthAttendanceInLabour or HSI_Labour_ReceivesCareForPostpartumPeriod"""
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_maternal_sepsis' not in params['allowed_interventions']:
            return
        else:

            pkg_code_sepsis = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                                                                   'Intervention_Pkg_Code'])[0]
            consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_needed_sepsis)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run=f'assess_sepsis_{facility_type}_{labour_stage}', hsi_event=hsi_event):

                if outcome_of_request_for_consumables_sep:
                    logger.debug('mother %d has has their sepsis identified during delivery. As consumables are '
                                 'available they will receive treatment', person_id)
                    df.at[person_id, 'la_sepsis_treatment'] = True

                elif df.at[person_id, 'la_sepsis']:
                    logger.debug('mother %d has not had their sepsis identified during delivery and will not be '
                                 'treated', person_id)

    def assessment_and_treatment_of_hypertension(self, hsi_event, facility_type):
        """This function defines the required consumables, determines correct diagnosis and administers intervention
        to women suffering from hypertension. It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_hypertension' not in params['allowed_interventions']:
            return
        else:

            pkg_code_severe_hypertension = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_htn = {'Intervention_Package_Code': {pkg_code_severe_hypertension: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_htn = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_needed_htn)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                    dx_tests_to_run=f'assess_hypertension_{facility_type}', hsi_event=hsi_event):

                if outcome_of_request_for_consumables_htn:
                    df.at[person_id, 'la_maternal_hypertension_treatment'] = True
                    logger.debug('mother %d has has their hypertension identified during delivery. As '
                                 'consumables are available they will receive treatment', person_id)

            elif df.at[person_id, 'ps_gestational_htn']:
                logger.debug('mother %d has not had their hypertension identified during delivery and will '
                             'not be treated', person_id)

    def assessment_and_treatment_of_eclampsia(self, hsi_event):
        """This function defines the required consumables, determines correct diagnosis and administers intervention
        to women suffering from eclampsia. It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        or HSI_Labour_ReceivesCareForPostpartumPeriod"""
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_eclampsia' not in params['allowed_interventions']:
            return
        else:

            pkg_code_eclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_eclampsia = {'Intervention_Package_Code': {pkg_code_eclampsia: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_ec = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_eclampsia)

            if outcome_of_request_for_consumables_ec:
                df.at[person_id, 'la_eclampsia_treatment'] = True
                logger.debug('mother %d has has their eclampsia identified during delivery. As '
                             'consumables are available they will receive treatment', person_id)
            elif df.at[person_id, 'la_eclampsia']:
                logger.debug('mother %d has not had their eclampsia identified during delivery and will '
                             'not be treated', person_id)

    def assessment_and_plan_for_referral_antepartum_haemorrhage(self, hsi_event, facility_type, treatment_or_referral):
        """This function determines correct diagnosis and referral for intervention for women suffering from antepartum
        haemorrhage. It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_antepartum_haemorrhage' not in params['allowed_interventions']:
            return
        else:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
               dx_tests_to_run=f'assess_for_{treatment_or_referral}_aph_{facility_type}',
               hsi_event=hsi_event):
                logger.debug('mother %d has has their antepartum haemorrhage identified during delivery. They will now '
                             'be referred for additional treatment', person_id)
                if mni[person_id]['delivery_attended']:
                    mni[person_id]['referred_for_cs'] = 'prompt_referral'
                    mni[person_id]['referred_for_blood'] = 'prompt_referral'
                else:
                    mni[person_id]['referred_for_cs'] = 'delayed_referral'
                    mni[person_id]['referred_for_blood'] = 'delayed_referral'

            elif df.at[person_id, 'la_antepartum_haem']:
                logger.debug('mother %d has not had their antepartum haemorrhage identified during delivery and will'
                             'not be referred for treatment', person_id)

    def assessment_and_plan_for_referral_uterine_rupture(self, hsi_event, facility_type, treatment_or_referral):
        """This function determines correct diagnosis and referral for intervention for women suffering from uterine
        rupture. It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour"""
        df = self.sim.population.props
        params = self.parameters
        mni = self.mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_uterine_rupture' not in params['allowed_interventions']:
            return
        else:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
               dx_tests_to_run=f'assess_for_{treatment_or_referral}_uterine_rupture_{facility_type}',
               hsi_event=hsi_event):
                logger.debug('mother %d has has their uterine rupture identified during delivery. They will now be '
                             'referred for additional treatment', person_id)
                if mni[person_id]['delivery_attended']:
                    mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                    mni[person_id]['referred_for_cs'] = 'prompt_referral'
                else:
                    mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                    mni[person_id]['referred_for_cs'] = 'delayed_referral'

            elif df.at[person_id, 'la_uterine_rupture']:
                logger.debug('mother %d has not had their uterine_rupture identified during delivery and will not be '
                             'referred for treatment', person_id)

    def active_management_of_the_third_stage_of_labour(self, hsi_event):
        """This function define consumables and administration of active management of the third stage of labour for
        women immediately following birth. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod"""
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target

        if 'active_management_of_the_third_stage_of_labour' not in params['allowed_interventions']:
            return
        else:

            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

            pkg_code_am = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Active management of the 3rd stage of labour',
                                                                   'Intervention_Pkg_Code'])[0]

            consumables_needed = {
                    'Intervention_Package_Code': {pkg_code_am: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed)

            # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
            # labour (additional oxytocin, uterine massage and controlled cord traction)
            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_am]:
                logger.debug('pkg_code_am is available, so use it.')
                mni[person_id]['amtsl_given'] = True
            else:
                logger.debug('mother %d did not receive active management of the third stage of labour as she delivered'
                             'without assistance', person_id)

    def assessment_and_treatment_of_pph_retained_placenta(self, hsi_event):
        """This function defines consumables and administration of treatment for women suffering from a postpartum
        haemorrhage attributed to retained placenta. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_pph_retained_placenta' not in params['allowed_interventions']:
            return
        else:
            pkg_code_pph = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum '
                                                                                        'hemorrhage',
                                                     'Intervention_Pkg_Code'])[0]

            consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_pph)

            if outcome_of_request_for_consumables_pph:
                if params['prob_successful_manual_removal_placenta'] > self.rng.random_sample():
                    df.at[person_id, 'la_postpartum_haem_treatment'] = True
                else:
                    if mni[person_id]['delivery_attended']:
                        mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                        mni[person_id]['referred_for_blood'] = 'prompt_referral'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                        mni[person_id]['referred_for_blood'] = 'delayed_referral'

    def assessment_and_treatment_of_pph_uterine_atony(self, hsi_event):
        """This function defines consumables and administration of treatment for women suffering from a postpartum
        haemorrhage attributed to uterine atony. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod"""
        df = self.sim.population.props
        mni = self.mother_and_newborn_info
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_pph_uterine_atony' not in params['allowed_interventions']:
            return
        else:

            pkg_code_pph = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum hemorrhage',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_needed_pph)

            if outcome_of_request_for_consumables_pph:
                if mni[person_id]['delivery_attended']:
                    if params['prob_cure_uterotonics'] > self.rng.random_sample():
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'prompt_treatment'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'prompt_referral'
                        mni[person_id]['referred_for_blood'] = 'prompt_referral'
                else:
                    if params['prob_cure_uterotonics'] > self.rng.random_sample():
                        df.at[person_id, 'la_postpartum_haem_treatment'] = 'delayed_treatment'
                    else:
                        mni[person_id]['referred_for_surgery'] = 'delayed_referral'
                        mni[person_id]['referred_for_blood'] = 'delayed_referral'


class LabourOnsetEvent(Event, IndividualScopeEventMixin):
    """This is the LabourOnsetEvent. It is scheduled by the set_date_of_labour function. It represents the start of a
    womans labour. Here we assign a "type" of labour based on gestation (i.e. early preterm), we create a dictionary to
    store additional variables important to labour and HSIs, and we determine if and where a woman will seek care.
     This event schedules  the LabourAtHome event and the HSI_Labour_PresentsForSkilledAttendance at birth
     (depending on care seeking), the BirthEvent and the LabourDeathEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info
        person = df.loc[individual_id]

        # We exclude women who may have been previously pregnant/had their due date has been moved by an intervention
        # and should no longer be in labour today
        if not (person.is_alive and person.is_pregnant and person.la_due_date_current_pregnancy == self.sim.date):
            logger.debug('person %d has just reached LabourOnsetEvent on %s, however this is event is no longer '
                         'relevant for this individual and will not run', individual_id, self.sim.date)
            return

        else:

            # For women pregnant women who are due today, they move through the event
            logger.debug('person %d has just reached LabourOnsetEvent on %s', individual_id, self.sim.date)

            # We indicate this woman is now in labour using this property, and by adding her individual ID to our
            # labour list (for testing)
            df.at[individual_id, 'la_currently_in_labour'] = True
            self.module.women_in_labour.append(individual_id)

            # We then run the labour_characteristics_checker as a final check that only appropriate women are here
            self.module.labour_characteristics_checker(individual_id)

            # Next we populate the maternal and newborn info dictionary with baseline values before the womans labour
            # begins
            mni[individual_id] = {'labour_state': None,
                                  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or Post Term (POTL)
                                  'delivery_setting': None,  # home_birth, health_centre, hospital
                                  'delivery_attended': False,  # True or False
                                  'corticosteroids_given': False,
                                  'clean_delivery_kit_used': False,
                                  'abx_for_prom_given': False,
                                  'abx_for_pprom_given': False,
                                  'abx_for_preterm_given': False,
                                  'amtsl_given': False,
                                  'source_pph': None,  # Uterine Atony (UA) or Retained Products/Placenta (RPP)
                                  'mode_of_delivery': 'vaginal_delivery',  # vaginal_delivery, instrumental,
                                  # caesarean_section
                                  'squeeze_to_high_for_hsi': False,
                                  'squeeze_to_high_for_hsi_pp': False,
                                  'sought_care_for_complication': False,
                                  'sought_care_labour_phase': 'none',  # none, intrapartum, postpartum
                                  'referred_for_cs': 'none',  # 'none', 'prompt_referral', 'delayed_referral'
                                  'referred_for_blood': 'none',  # 'none', 'prompt_referral', 'late_referral'
                                  'received_blood_transfusion': False,
                                  'referred_for_surgery': 'none',  # 'none', 'prompt_referral', 'late_referral'
                                  'death_in_labour': False,  # True (T) or False (F)
                                  'cause_of_death_in_labour': [],
                                  'stillbirth_in_labour': False,  # True (T) or False (F)
                                  'cause_of_stillbirth_in_labour': [],
                                  'death_postpartum': False}  # True (T) or False (F)

            # ===================================== LABOUR STATE  ==================================

            gestational_age_in_days = (self.sim.date - df.at[individual_id, 'date_of_last_pregnancy']).days

            # Now we use gestational age to categorise the 'labour_state'
            if params['lower_limit_term_days'] <= gestational_age_in_days <= params['upper_limit_term_days']:

                self.module.labour_tracker['term'] += 1
                mni[individual_id]['labour_state'] = 'term_labour'

            # Here we allow a woman to go into early preterm labour with a gestational age of 23 (limit is 24) to
            # account for PregnancySupervisor only updating weekly
            elif params['lower_limit_early_preterm_days'] <= gestational_age_in_days <= params[
                        'upper_limit_early_preterm_days']:

                mni[individual_id]['labour_state'] = 'early_preterm_labour'
                self.module.labour_tracker['early_preterm'] += 1
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif params['lower_limit_late_preterm_days'] <= gestational_age_in_days <= params[
                        'upper_limit_late_preterm_days']:

                mni[individual_id]['labour_state'] = 'late_preterm_labour'
                self.module.labour_tracker['late_preterm'] += 1
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif gestational_age_in_days >= params['lower_limit_postterm_days']:

                mni[individual_id]['labour_state'] = 'postterm_labour'
                self.module.labour_tracker['post_term'] += 1

            # We check all women have had their labour state set
            assert mni[individual_id]['labour_state'] is not None

            labour_state = mni[individual_id]['labour_state']
            logger.debug(f'This is LabourOnsetEvent, person %d has now gone into {labour_state} on date %s',
                         individual_id, self.sim.date)

            # ===================================== CARE SEEKING AND DELIVERY SETTING ==============
            # Here we calculate this womans predicted risk of home birth and health centre birth
            pred_hb_delivery = params['la_labour_equations']['probability_delivery_at_home'].predict(
                df.loc[[individual_id]])[individual_id]
            pred_hc_delivery = params['la_labour_equations']['probability_delivery_health_centre'].predict(
                df.loc[[individual_id]])[individual_id]

            # The denominator is calculated
            denom = 1 + pred_hb_delivery + pred_hc_delivery

            # Followed by the probability of each of the three outcomes - home birth, health centre birth or hospital
            # birth
            prob_hb = pred_hb_delivery / denom
            prob_hc = pred_hc_delivery / denom
            prob_hp = 1 / denom

            # And a probability weighted random draw is used to determine where the woman will deliver
            facility_types = ['home_birth', 'health_centre', 'hospital']
            probabilities = [prob_hb, prob_hc, prob_hp]
            mni[individual_id]['delivery_setting'] = self.module.rng.choice(facility_types, p=probabilities)

            # Check all women's 'delivery setting' is set
            assert mni[individual_id]['delivery_setting'] is not None

            # Women delivering at home move the the LabourAtHomeEvent as they will not receive skilled birth attendance
            if mni[individual_id]['delivery_setting'] == 'home_birth':
                self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)

                logger.info('This is LabourOnsetEvent, person %d as they has chosen not to seek care at a health centre'
                            'for delivery and will give birth at home on date %s', individual_id, self.sim.date)

            # Otherwise the appropriate HSI is scheduled
            elif mni[individual_id]['delivery_setting'] == 'health_centre':
                health_centre_delivery = HSI_Labour_PresentsForSkilledBirthAttendanceInLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=1)
                self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.info('This is LabourOnsetEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on '
                            'date %s for person %d as they have chosen to seek care at a health centre for delivery',
                            self.sim.date, individual_id)

            elif mni[individual_id]['delivery_setting'] == 'hospital':
                # TODO: rng.choice is a placeholder for how to proceed with this aspect of care seeking
                facility_level = int(self.module.rng.choice([1, 2]))
                hospital_delivery = HSI_Labour_PresentsForSkilledBirthAttendanceInLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=facility_level)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hospital_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

            # ======================================== SCHEDULING BIRTH AND DEATH EVENTS ===========
            # We schedule all women to move through the death event where those who have developed a complication
            # that hasn't been treated or treatment has failed will have a case fatality rate applied
            self.sim.schedule_event(LabourDeathEvent(self.module, individual_id), self.sim.date + DateOffset(days=4))

            # Here we schedule the birth event for 4 days after labour- as women who die but still deliver a live child
            # will pass through birth event
            due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
            self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=5))
            logger.debug('This is LabourOnsetEvent scheduling a birth on date %s to mother %d', due_date,
                         individual_id)

            logger.debug('This is LabourOnsetEvent scheduling a potential death on date %s for mother %d',
                         self.sim.date, individual_id)

            # ======================================== RUNNING CHECKS ON EVENT QUEUES ==============
            # Here we run a check to ensure at the end of the preliminary labour event, women have the appropriate
            # future events scheduled
            # For debugging only:
            # self.module.events_queue_checker(individual_id)


class LabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """This is the LabourAtHomeEvent. It is scheduled by the LabourOnsetEvent for women who will not
    seek care. This event applies the probability that women delivering at home will experience
    complications, makes the appropriate changes to the data frame . Women who seek care, but for some
    reason are unable to deliver at a facility will return to this event"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        # Check only women delivering at home pass through this event and that the right characteristics are present
        assert mni[individual_id]['delivery_setting'] == 'home_birth'
        self.module.labour_characteristics_checker(individual_id)

        individual = df.loc[individual_id]

        # Condition the event on women being alive and log the birth in a tracker
        if individual.is_alive:
            logger.debug('person %d has is now going to deliver at home', individual_id)
            self.module.labour_tracker['home_birth'] += 1

            # ===================================  APPLICATION OF COMPLICATIONS ====================
            # Using the complication_application function we loop through each complication and determine if a woman
            # will experience any of these if she has delivered at home

            for complication in ['obstructed_labour', 'antepartum_haem', 'sepsis', 'eclampsia', 'uterine_rupture']:
                self.module.set_intrapartum_complications(individual_id, complication=complication)

            # ==============================  CARE SEEKING FOLLOWING COMPLICATIONS =================
            # We use a logistic regression equation, stored in the linear model, to determine if women will seek care
            # for delivery if they have developed a complication at a home birth

            # Women who couldn't get care for delivery due to reduced capacity WILL NOT seek care for complications
            if not mni[individual_id]['squeeze_to_high_for_hsi']:
                if (individual.la_obstructed_labour or
                        individual.la_antepartum_haem or
                        individual.la_sepsis or
                        individual.la_eclampsia or
                        individual.la_uterine_rupture):
                    if self.module.predict(params['la_labour_equations']['care_seeking_for_complication'],
                                           individual_id):
                        mni[individual_id]['sought_care_for_complication'] = True
                        mni[individual_id]['sought_care_labour_phase'] = 'intrapartum'

                        from tlo.methods.hsi_generic_first_appts import (
                            HSI_GenericEmergencyFirstApptAtFacilityLevel1,
                        )
                        event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                                module=self.module,
                                person_id=individual_id
                                )

                        logger.debug('mother %d will now seek care for a complication that has developed during labour '
                                     'on date %s', individual_id, self.sim.date)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))


class BirthEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by LabourOnsetEvent. For women who survived labour, the appropriate
    variables are reset/updated and the function do_birth is executed. This event schedules PostPartumLabourEvent for
    those women who have survived"""

    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        df = self.sim.population.props
        person = df.loc[mother_id]

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.info('mother %d at birth event on date %s', mother_id, self.sim.date)

        # Check the correct amount of time has passed between labour onset and birth event and that women at the event
        # have the right characteristics present
        assert (self.sim.date - df.at[mother_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.labour_characteristics_checker(mother_id)

        # If the mother is alive and still pregnant we generate a  child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications (intrapartum stillbirths till
        # trigger births for monitoring purposes)
        if person.is_alive and person.is_pregnant:
            logger.info('@@@@ A Birth is now occuring, to mother %d', mother_id)
            self.sim.do_birth(mother_id)
            logger.debug('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourSchedulerEvent(self.module, mother_id),
                                    self.sim.date)

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~person.is_alive and ~person.la_intrapartum_still_birth and person.la_maternal_death_in_labour:
            logger.debug('@@@@ A Birth is now occuring, to mother %d who died in childbirth but her child survived',
                         mother_id)
            self.sim.do_birth(mother_id)


class PostpartumLabourSchedulerEvent(Event, IndividualScopeEventMixin):
    """ This is PostpartumLabourSchedulerEvent. It is scheduled by the BirthEvent. This event schedule additional care
    through  HSI_Labour_ReceivesCareForPostpartumPeriod or PostpartumLabourAtHomeEvent for women who delivered at home.
    It also schedules the PostPartumDeathEvent and DisabilityResetEvent for all women."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.postpartum_characteristics_checker(individual_id)

        # Event should only run if woman is still alive
        if not df.at[individual_id, 'is_alive']:
            return

        # If a woman has delivered in a facility we schedule her to now receive additional care following birth
        health_centre_care = HSI_Labour_ReceivesCareForPostpartumPeriod(
            self.module, person_id=individual_id, facility_level_of_this_hsi=1)

        all_facility_care = HSI_Labour_ReceivesCareForPostpartumPeriod(
            self.module, person_id=individual_id, facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))

        if mni[individual_id]['delivery_setting'] == 'home_birth':
            self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, individual_id), self.sim.date)
            logger.info('This is PostPartumEvent scheduling PostpartumLabourAtHomeEvent for person %d on date %s',
                        individual_id, self.sim.date)

        elif mni[individual_id]['delivery_setting'] == 'health_centre':
            logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod '
                        'for person %d on date %s', individual_id, self.sim.date)
            self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_care,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        elif mni[individual_id]['delivery_setting'] == 'hospital':
            logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod '
                        'for person %d on date %s', individual_id, self.sim.date)
            self.sim.modules['HealthSystem'].schedule_hsi_event(all_facility_care,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
        # complications may experience death
        self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id), self.sim.date + DateOffset(days=5))
        logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s',
                    individual_id, self.sim.date + DateOffset(days=4))  # Date offset to allow for interventions

        # Here we schedule women to an event which resets 'daly' disability associated with delivery
        # complications
        self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id), self.sim.date + DateOffset(weeks=4))

        # TODO: lots of scheduling here, conditioned on ifs, would be good to check a womans event queue


class PostpartumLabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """This is PostpartumLabourAtHomeEvent. This event is scheduled by PostpartumLabourSchedulerEvent for women whose
    whole delivery has taken place at home OR HSI_Labour_ReceivesCareForPostpartumPeriod for women who couldnt receive
    post-partum care due to high squeeze factor. """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.postpartum_characteristics_checker(individual_id)

        # Event should only run if woman is still alive
        if not df.at[individual_id, 'is_alive']:
            return

        # Here labour_stage 'pp' means postpartum
        self.module.set_postpartum_complications(individual_id, complication='postpartum_haem')

        if ~df.at[individual_id, 'la_sepsis']:
            # Women who already have sepsis cannot develop it again immediately after birth
            self.module.set_postpartum_complications(individual_id, complication='sepsis')
            # Women who already have already developed eclampsia cannot develop it again immediately after birth
        if ~df.at[individual_id, 'la_eclampsia']:
            self.module.set_postpartum_complications(individual_id, complication='eclampsia')

        # Women who have come home, following a facility delivery, due to high squeeze will not try and seek care
        # for any complications
        if ~mni[individual_id]['squeeze_to_high_for_hsi_pp'] and (df.at[individual_id, 'la_sepsis_postpartum'] or
                                                                  df.at[individual_id, 'la_eclampsia_postpartum'] or
                                                                  df.at[individual_id, 'la_postpartum_haem']):

            if self.module.predict(params['la_labour_equations']['care_seeking_for_complication'], individual_id):
                mni[individual_id]['sought_care_for_complication'] = True
                mni[individual_id]['sought_care_labour_phase'] = 'postpartum'

                from tlo.methods.hsi_generic_first_appts import (
                    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
                )
                event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                     module=self.module, person_id=individual_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    event,
                    priority=0,
                    topen=self.sim.date,
                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug('mother %d will now seek care for a complication that has developed following labour '
                             'on date %s', individual_id, self.sim.date)


class LabourDeathEvent (Event, IndividualScopeEventMixin):
    """This is the LabourDeathEvent. It is scheduled by the LabourOnsetEvent for all women who go through labour. This
    event determines if women who have experienced complications in labour will die or experience an intrapartum
    stillbirth."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')
        self.module.labour_characteristics_checker(individual_id)

        if df.at[individual_id, 'ps_severe_pre_eclamp']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='severe_pre_eclamp')

        if df.at[individual_id, 'la_eclampsia']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='eclampsia')

        if df.at[individual_id, 'la_antepartum_haem']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='antepartum_haem')

        if df.at[individual_id, 'la_sepsis']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='sepsis')

        if df.at[individual_id, 'la_uterine_rupture']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='uterine_rupture')

        # Schedule death for women who die in labour
        if mni[individual_id]['death_in_labour']:
            self.module.labour_tracker['maternal_death'] += 1
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)

            # Log the maternal death
            logger.info('This is LabourDeathEvent scheduling a death for person %d on date %s who died due to '
                        'intrapartum complications', individual_id, self.sim.date)

            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': mni[individual_id]})

            if mni[individual_id]['death_in_labour'] and df.at[individual_id, 'la_intrapartum_still_birth']:
                # We delete the mni dictionary if both mother and baby have died in labour, if the mother has died but
                # the baby has survived we delete the dictionary following the on_birth function of NewbornOutcomes
                del mni[individual_id]

        if df.at[individual_id, 'la_intrapartum_still_birth']:
            self.module.labour_tracker['ip_stillbirth'] += 1
            logger.info('@@@@ A Still Birth has occurred, to mother %s', individual_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'mother_id': individual_id})


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):
    """This is the PostPartumDeathEvent. It is scheduled by the PostpartumLabourEvent. This event determines if women
    who have experienced complications following labour will die. This event schedules the DiseaseResetEvent for
    surviving women"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        logger.debug('mother %d at PPD at date %s', individual_id, self.sim.date)

        # Check the correct amount of time has passed between labour onset and PostPartumDeathEvent
        assert pd.to_timedelta(10, unit='D') >= (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'])\
                                             >= pd.to_timedelta(9, unit='D')
        self.module.postpartum_characteristics_checker(individual_id)

        if df.at[individual_id, 'is_alive']:
            if df.at[individual_id, 'la_eclampsia_postpartum']:
                self.module.set_maternal_death_status_postpartum(individual_id, cause='eclampsia')

            if df.at[individual_id, 'la_postpartum_haem']:
                self.module.set_maternal_death_status_postpartum(individual_id, cause='postpartum_haem')

            if df.at[individual_id, 'la_sepsis_postpartum']:
                self.module.set_maternal_death_status_postpartum(individual_id, cause='sepsis')

            if mni[individual_id]['death_postpartum']:
                self.module.labour_tracker['maternal_death'] += 1
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='postpartum labour'), self.sim.date)

                logger.debug('This is PostPartumDeathEvent scheduling a death for person %d on date %s who died due to '
                             'postpartum complications', individual_id,
                             self.sim.date)

                logger.debug('%s|labour_complications|%s', self.sim.date,
                             {'person_id': individual_id,
                              'labour_profile': mni[individual_id]})
                del mni[individual_id]

            else:
                # Surviving women pass through the DiseaseResetEvent to ensure all complication variable are set to
                # false
                self.sim.schedule_event(DiseaseResetEvent(self.module, individual_id),
                                        self.sim.date + DateOffset(weeks=1))

                logger.debug('%s|labour_complications|%s', self.sim.date,
                             {'person_id': individual_id,
                              'labour_profile': mni[individual_id]})

            # End the period of current labour
            # should this be for everyone?
                df.at[individual_id, 'la_currently_in_labour'] = False

        # Here we remove all women (dead and alive) who have passed through the labour events
        self.module.women_in_labour.remove(individual_id)

        # todo: reset la_due_date_current_pregnancy ?


class DisabilityResetEvent (Event, IndividualScopeEventMixin):
    """This is the DisabilityResetEvent. It is scheduled by the PostPartumLabourEvent. This event resets a woman's
    disability properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        # TODO:========================= MOVE TO PREGNANCY SUPERVISOR ===============================
        # todo: discuss with Tim H when resubmitting - seems strange to change a lot of variables that are
        #  applied in this module, in another module

        # Check the correct amount of time has passed between labour onset and DisabilityResetEvent and that only women
        # post-pregnancy are coming to this event

        assert pd.to_timedelta(34, unit='D') > (self.sim.date - df.at[
            individual_id, 'la_due_date_current_pregnancy']) >= pd.to_timedelta(32, unit='D')
        assert ~df.at[individual_id, 'is_pregnant']
        assert individual_id not in self.module.women_in_labour

        # Here we turn off all the properties which are used to count DALYs
        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their disability status reset on date %s', individual_id, self.sim.date)

            df.at[individual_id, 'la_sepsis_disab'] = False
            df.at[individual_id, 'la_obstructed_labour_disab'] = False
            df.at[individual_id, 'la_uterine_rupture_disab'] = False
            df.at[individual_id, 'la_eclampsia_disab'] = False
            df.at[individual_id, 'la_maternal_haem_severe_disab'] = False
            df.at[individual_id, 'la_maternal_haem_non_severe_disab'] = False


class DiseaseResetEvent (Event, IndividualScopeEventMixin):
    """This is the DiseaseResetEvent. It is scheduled by the PostPartumDeathEvent. This event resets a woman's
    disease and treatment properties within the data frame """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        # TODO:================================================ SEE ABOVE    ==========================================

        # Check the correct amount of time has passed between labour onset and DiseaseResetEvent and that only women
        # post-pregnancy are coming to this event
        assert pd.to_timedelta(17, unit='D') >= (self.sim.date - df.at[
            individual_id, 'la_due_date_current_pregnancy']) >= pd.to_timedelta(16, unit='D')
        assert ~df.at[individual_id, 'is_pregnant']
        assert individual_id not in self.module.women_in_labour

        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their maternal disease status reset', individual_id)

            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'la_sepsis_postpartum'] = False
            df.at[individual_id, 'la_obstructed_labour'] = False
            df.at[individual_id, 'la_antepartum_haem'] = False
            df.at[individual_id, 'la_uterine_rupture'] = False
            df.at[individual_id, 'la_eclampsia'] = False
            df.at[individual_id, 'la_eclampsia_postpartum'] = False
            df.at[individual_id, 'la_postpartum_haem'] = False

            df.at[individual_id, 'la_sepsis_treatment'] = False
            df.at[individual_id, 'la_obstructed_labour_treatment'] = False
            df.at[individual_id, 'la_antepartum_haem_treatment'] = False
            df.at[individual_id, 'la_uterine_rupture_treatment'] = False
            df.at[individual_id, 'la_eclampsia_treatment'] = False
            df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] = False
            df.at[individual_id, 'la_maternal_hypertension_treatment'] = False
            df.at[individual_id, 'la_postpartum_haem_treatment'] = False

            # This function resets the antenatal diseases a woman will have had during pregnancy
            self.sim.modules['PregnancySupervisor'].antenatal_disease_reset(individual_id)

            del mni[individual_id]


class HSI_Labour_PresentsForSkilledBirthAttendanceInLabour(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PresentsForSkilledAttendanceInLabour. This event is scheduled by the LabourOnset Event.
    This event manages initial care around the time of delivery including prophylactic interventions (i.e. clean
    birth practices) for women presenting to the health system for delivery care. This event uses a womans
    stored risk of complications, which may be manipulated by treatment effects to determines if they will experience
    a complication during their labour in hospital. It is responsible for delivering interventions or scheduling
    treatment HSIs for certain complications."""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'NormalDelivery': 1})
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.parameters

        if not df.at[person_id, 'is_alive']:
            return

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Mother %d has presented to a health '
                    'facility on date %s following the onset of her labour', person_id, self.sim.date)

        # Women who developed a complication at home, then presented to a facility for delivery, are counted as
        # facility deliveries
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'
        # TODO: currently dont have choice of facility level in complication care-seeking equation

        # Delivery setting is captured in the labour_tracker, processed by the logger and reset yearly
        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.labour_tracker['health_centre_birth'] += 1

        elif mni[person_id]['delivery_setting'] == 'hospital':
            self.module.labour_tracker['hospital_birth'] += 1

        # Next we check this woman has the right characteristics to be at this event
        self.module.labour_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'
        #    assert self.sim.date == df.at[person_id, 'la_due_date_current_pregnancy'] or \
        #       self.sim.date == (df.at[person_id, 'la_due_date_current_pregnancy'] + pd.to_timedelta(1, unit='D'))

        # ============================== AVAILABILITY SKILLED BIRTH ATTENDANTS =====================================
        # On presentation to the facility, we use the squeeze factor to determine if this woman will receive
        # delivery care from a health care professional, or if she will delivered unassisted in a facility
        if squeeze_factor > params['squeeze_factor_threshold_delivery_attendance']:
            mni[person_id]['delivery_attended'] = False
            logger.debug('mother %d is delivering without assistance at a health facility', person_id)
        else:
            mni[person_id]['delivery_attended'] = True
            logger.debug('mother %d is delivering with assistance at a level 1 health facility', person_id)

        # Run checks that key facility properties in the mni have been set
        assert mni[person_id]['delivery_attended'] is not None

        # ===================================== PROPHYLACTIC CARE ===================================================
        # The following function manages the consumables and administration of prophylactic interventions in labour
        # TODO: Confirm how to ensure that only the consumables NEEDED are those which are logged (whether or not they
        #  are available) as required consumables vary from woman to woman based on presenting history

        # If this womans delivery is attended, the function will run. However women presenting for delivery
        # following a
        # complication do not benefit from prophylaxis as the risk of complications has already been applied
        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            self.module.prophylactic_labour_interventions(self)
        else:
            # Otherwise she receives no benefit of prophylaxis
            logger.debug('mother %d received no prophylaxis as she is delivering unattended', person_id)

        # ================================= PROPHYLACTIC MANAGEMENT PRE-ECLAMPSIA  ==============================
        # As women with severe pre-eclampsia are the most at risk group of women to develop eclampsia, they should
        # receive magnesium to reduce the probability of seizures- hence this intervention is applied before we apply
        # risk of complications

        # We assume that only women who are having an attended delivery will have their severe pre-eclampsia
        # identified
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_of_severe_pre_eclampsia(self, 'hc')
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_severe_pre_eclampsia(self, 'hp')

        # ===================================== APPLYING COMPLICATION INCIDENCE ====================
        # Following administration of prophylaxis (for attended deliveries) we assess if this woman will develop any
        # complications (effect of prophylaxis is included in the linear model for relevant complications)

        if not mni[person_id]['sought_care_for_complication']:
            [self.module.set_intrapartum_complications(person_id, complication=complication)
                for complication in
                ['obstructed_labour', 'eclampsia', 'antepartum_haem', 'sepsis']]

        # n.b. we do not apply the risk of uterine rupture due to the causal link between obstructed labour and
        # uterine rupture. We want interventions for obstructed labour to reduce the risk of uterine rupture

        # ======================================= COMPLICATION MANAGEMENT ==========================
        # Next, women in labour are assessed for complications and treatment delivered if a need is identified and
        # consumables are available

        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_of_obstructed_labour(self, 'hc')
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_obstructed_labour(self, 'hp')

        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hc', 'ip')
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hp', 'ip')

        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_of_hypertension(self, 'hc')
        if mni[person_id]['delivery_attended'] and mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_hypertension(self, 'hp')

        # Because eclampsia presents and easily recognisable seizures we do not use the dx_test function and
        # assume all women have equal likelihood of treatment irrespective of facility level or delivery attendance
        if df.at[person_id, 'la_eclampsia']:
            self.module.assessment_and_treatment_of_eclampsia(self)

        # Antepartum haemorrhage can only be treated by delivery in our model. Here the dx_test in a health centre
        # has its sensitivity set to reflect likelihood this woman will be correctly referred to a higher level
        # facility.The dx_test in a hospital has its sensitivity set to reflect likelihood of treatment within the
        # same facility
        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_plan_for_referral_antepartum_haemorrhage(self, 'hc', 'referral')
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_plan_for_referral_antepartum_haemorrhage(self, 'hp', 'treatment')

        # We now calculate and apply the risk of uterine rupture
        # Women whose obstructed labour has been successfully treated will not have risk of uterine rupture applied,
        # women who will be referred for caesarean also wont have the risk applied
        if ~df.at[person_id, 'la_obstructed_labour_treatment'] or mni[person_id]['referred_for_cs'] == 'none':
            self.module.set_intrapartum_complications(
                person_id, complication='uterine_rupture')

        # Uterine rupture follows the same pattern as antepartum haemorrhage
        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_plan_for_referral_uterine_rupture(self, 'hc', 'referral')
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_plan_for_referral_uterine_rupture(self, 'hp', 'treatment')

        # ============================================== REFERRAL ==================================
        # Finally we send any women who require additional treatment to the following HSIs
        if mni[person_id]['referred_for_cs'] != 'none':
            caesarean_section = HSI_Labour_CaesareanSection(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)
            self.sim.modules['HealthSystem'].schedule_hsi_event(caesarean_section,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplications(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusion(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # TODO: Women are only being sent to the same facility LEVEL (this doesnt reflect HC/DH referall to
        #  national hospitals)

        # If a this woman has experienced a complication the appointment footprint is changed from normal to
        # complicated
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        if (df.at[person_id, 'la_sepsis'] or
                df.at[person_id, 'la_antepartum_haem'] or
                df.at[person_id, 'la_uterine_rupture'] or
                df.at[person_id, 'la_eclampsia']):
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery']  # todo: is this right?

        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info

        # If a woman has chosen to deliver in a facility from the onset of labour, but the squeeze factor is too high,
        # she will be forced to return home to deliver
        if not mni[person_id]['sought_care_for_complication']:
            logger.debug('squeeze factor is too high for this event to run for mother %d on date %s and she will now '
                         'deliver at home', person_id, self.sim.date)
            mni[person_id]['delivery_setting'] = 'home_birth'
            mni[person_id]['squeeze_to_high_for_hsi'] = True
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)

        # If a woman has presented to this event during labour due to a complication, she will not receive any treatment
        if mni[person_id]['sought_care_for_complication']:
            logger.debug('squeeze factor is too high for this event to run for mother %d on date %s and she could not '
                         'receive care for the complications developed during her homebirth', person_id, self.sim.date)

        return False

    def not_available(self):
        """This is called when the HSI is passed to the healthsystem scheduler but not scheduled as the TREATMENT_ID
        is not allowed under the 'services_available' parameter of the health system.
        Note that this called at the time of the event being passed to the Health System at schedule_hsi_event(...) and
        not at the time when the HSI is intended to be run (as specified by the 'topen' parameter in that call)"""
        person_id = self.target
        mni = self.module.mother_and_newborn_info
        # If a woman has chosen to deliver in a facility but this event isnt allowed with the set service configuration
        # then she will deliver at home
        logger.debug('This event is not in the allowed service availability and therefore cannot  run for mother %d on '
                     'date %s, she will now deliver at home', person_id, self.sim.date)

        if not mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'home_birth'
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)


class HSI_Labour_ReceivesCareForPostpartumPeriod(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI HSI_Labour_ReceivesCareForPostpartumPeriod. This event is scheduled by the PostpartumLabourEvent
    Event. This event manages care immediately following delivery including prophylactic interventions (i.e. active
    management of the third stage) for women who have delivered in a facility, or presetn for postpartum care. This
    event uses a womans stored risk of complications, which may be manipulated by treatment effects to determines if
    they will experience a complication during their postpartum period in hospital. It is responsible for delivering
    interventions or scheduling treatment HSIs for certain complications."""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1: Providing skilled attendance '
                    'following birth for person %d', person_id)

        if not df.at[person_id, 'is_alive']:
            return

        # Although we change the delivery setting variable to 'facility_delivery' we do not include women who present
        # for care following birth, due to complications, as facility deliveries
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # We run similar checks as the labour HSI
        self.module.postpartum_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        # -------------------------- Active Management of the third stage of labour ------------
        # Prophlyactic treatment to prevent postpartum bleeding is applied
        if mni[person_id]['delivery_attended'] and ~mni[person_id]['sought_care_for_complication']:
            self.module.active_management_of_the_third_stage_of_labour(self)
        else:
            logger.debug('mother %d did not receive active management of the third stage of labour due to resource '
                         'constraints')

        # ===================================== APPLYING COMPLICATION INCIDENCE ================
        # Again we use the mothers individual risk of each complication to determine if she will experience any
        # complications using the set_complications_during_facility_birth function.
        if not mni[person_id]['sought_care_for_complication']:
            self.module.set_postpartum_complications(
                person_id, complication='postpartum_haem')

            # Only women who haven't already developed eclampsia are able to become septic at this point
            if ~df.at[person_id, 'la_eclampsia']:
                self.module.set_postpartum_complications(
                    person_id, complication='eclampsia')

            # Only women who haven't already developed sepsis are able to become septic at this point
            if ~df.at[person_id, 'la_sepsis']:
                self.module.set_postpartum_complications(person_id, complication='sepsis')

        # ======================================= COMPLICATION MANAGEMENT ======================
        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hc', 'pp')
        if mni[person_id]['delivery_setting'] == 'hospital':
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hp', 'pp')

        if df.at[person_id, 'la_eclampsia_postpartum']:
            self.module.assessment_and_treatment_of_eclampsia(self)

        # We apply a delayed treatment effect to women whose delivery was not attended by staff for treatment
        # of both causes of PPH
        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'retained_placenta':
            self.module.assessment_and_treatment_of_pph_retained_placenta(self)
        elif df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'uterine_atony':
            self.module.assessment_and_treatment_of_pph_uterine_atony(self)

        # ============================================== REFERRAL ==============================
        if mni[person_id]['referred_for_surgery'] != 'none':
            surgery = HSI_Labour_SurgeryForLabourComplications(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgery,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        if mni[person_id]['referred_for_blood'] != 'none':
            blood_transfusion = HSI_Labour_ReceivesBloodTransfusion(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)
            self.sim.modules['HealthSystem'].schedule_hsi_event(blood_transfusion,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        # TODO: same issue as with the labour event

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.module.mother_and_newborn_info

        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run as the squeeze factor is too high, '
                     'mother %d will return home on date %s', person_id, self.sim.date)

        # Women who delivered at a facility, but can receive no more care due to high squeeze, will go home for the
        # immediate period after birth- where there risk of complications is applied
        if mni[person_id]['delivery_setting'] != 'home_birth':
            mni[person_id]['squeeze_to_high_for_hsi_pp'] = True
            self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, person_id), self.sim.date)

        return False

    def not_available(self):
        person_id = self.target
        logger.debug('This event is not in the allowed service availability and therefore cannot  run for mother %d on '
                     'date %s, she will now deliver at home', person_id, self.sim.date)


class HSI_Labour_CaesareanSection(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_Labour_CaesareanSection. It is scheduled by HSI_Labour_PresentsForSkilledAttendanceInLabour.
    This event manages caesarean section delivery."""
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_CaesareanSection'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info
        df = self.sim.population.props

        # We check the right women have been sent to this event
        self.module.labour_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'
        assert mni[person_id]['referred_for_cs'] != 'none'

        logger.info('This is HSI_Labour_CaesareanSection: Person %d will now undergo delivery via Caesarean Section',
                    person_id)
        self.module.labour_tracker['caesarean_section'] += 1

        # We define the consumables needed for this event
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Cesearian Section with indication (with complication)',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_cs = {'Intervention_Package_Code': {pkg_code_cs: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_cs)

        # Currently we do not stop the event from running if consumables are unavailble
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_cs]:
            logger.debug('All the required consumables are available and will can be used for this caesarean delivery.')
        else:
            logger.debug('The required consumables are not available for this delivery')

        # Set the variables to indicate this woman has undergone a caesarean delivery...
        mni[person_id]['mode_of_delivery'] = 'caesarean_section'
        df.at[person_id, 'la_previous_cs_delivery'] = True

        # We set the treatment variables for those complications for which caesarean section is part of the treatment...
        if df.at[person_id, 'la_obstructed_labour']:
            df.at[person_id, 'la_obstructed_labour_treatment'] = True

        if df.at[person_id, 'la_antepartum_haem']:
            df.at[person_id, 'la_antepartum_haem_treatment'] = True

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'deliver via caesarean section', person_id, self.sim.date)

        # Here we apply the risk that for women who were referred for caesarean, but it wasnt performed, develop uterine
        # rupture
        self.module.set_intrapartum_complications(
            person_id, complication='uterine_rupture')
        return False

    def not_available(self):
        person_id = self.target
        # This function would only run if a simulation is ran in which skilled birth attendance is allowed, but
        # caesareans are not. Therefore we need to apply a risk of uterine rupture to women who needed a caesarean but
        # didnt get one
        logger.debug('this event is not allowed in the current service availability, for mother %s on date %d and '
                     'she is unable to deliver via caesarean section', person_id, self.sim.date)

        self.module.set_intrapartum_complications(
            person_id, complication='uterine_rupture')
        return False


class HSI_Labour_ReceivesBloodTransfusion(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel1. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabour or HSI_Labour_ReceivesCareForPostpartumPeriod
    This event manages blood transfusion for women who have experienced significant blood loss in labour"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesBloodTransfusion'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        mni = self.module.mother_and_newborn_info

        assert mni[person_id]['delivery_setting'] != 'home_birth'
        assert mni[person_id]['referred_for_blood'] != 'none'

        logger.info('This is HSI_Labour_ReceivesBloodTransfusionFacilityLevel1: Person %d will now receive a blood '
                    'transfusion following haemorrhage/blood loss delivery', person_id)

        # Required consumables are defined
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_bt2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                  'Item_Code'])[0]
        consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1,
                                                                                item_code_bt3: 1, item_code_bt4: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_bt)

        # If they're availble, the event happens
        if outcome_of_request_for_consumables:
            mni[person_id]['received_blood_transfusion'] = True
            logger.debug('Mother %s has received a blood transfusion due following a maternal haemorrhage',
                         person_id)
        else:
            logger.debug('Mother %s was unable to receive a blood transfusion due to insufficient consumables',
                         person_id)

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'receive a blood transfusion', person_id, self.sim.date)

        return False

    def not_available(self):
        pass


class HSI_Labour_SurgeryForLabourComplications(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_Labour_SurgeryForLabourComplications. It can be scheduled by HSI_Labour_
    PresentsForSkilledAttendanceInLabour or HSI_Labour_ReceivesCareForPostpartumPeriod.This event manages surgery for
    women who have developed complication in labour where medical management has failed. This includes uterine rupture
    and postpartum haemorrhage."""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_SurgeryForLabourComplicationsFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info
        params = self.module.parameters

        assert mni[person_id]['delivery_setting'] != 'home_birth'
        assert mni[person_id]['referred_for_surgery'] != 'none'

        logger.info('This is HSI_Labour_SurgeryForLabourComplications: Person %d will now undergo surgery'
                    'for complications developed in labour', person_id)

        # TODO: Consumable aren't currently correct

        # Consumables are defined
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_surg_pkg_code = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                        'Cesearian Section with indication (with complication)',
                                                        'Intervention_Pkg_Code'])[0]

        consumables_needed_surgery = {'Intervention_Package_Code': {dummy_surg_pkg_code: 1}, 'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed_surgery)

        if outcome_of_request_for_consumables:
            logger.debug('Consumables required for surgery are available and therefore have been used')
        else:
            logger.debug('Consumables required for surgery are unavailable and therefore have not been used')

        # Interventions...
        treatment_success_ur = params['success_rate_uterine_repair'] < self.module.rng.random_sample()

        # Uterine Repair...

        if df.at[person_id, 'la_uterine_rupture'] and ~df.at[person_id, 'la_uterine_rupture_treatment']:
            if treatment_success_ur:
                df.at[person_id, 'la_uterine_rupture_treatment'] = True
            elif ~treatment_success_ur:
                df.at[person_id, 'la_uterine_rupture_treatment'] = True
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        treatment_success_pph = params['success_rate_pph_surgery'] < self.module.rng.random_sample()
        treatment_success_surgical_removal = params['success_rate_surgical_removal_placenta'] < \
            self.module.rng.random_sample()

        # Surgery for refractory atonic uterus...
        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'uterine_atony':
            if treatment_success_pph:
                df.at[person_id, 'la_postpartum_haem_treatment'] = True
            elif ~treatment_success_pph:
                # nb. evidence suggests uterine preserving surgery vs hysterectomy have comparable outcomes in LMICs
                df.at[person_id, 'la_postpartum_haem_treatment'] = True
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        # Surgery for retained placenta...
        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['source_pph'] == 'retained_placenta':
            if treatment_success_surgical_removal:
                df.at[person_id, 'la_postpartum_haem_treatment'] = True
            elif treatment_success_pph:
                df.at[person_id, 'la_postpartum_haem_treatment'] = True
            elif ~treatment_success_pph:
                logger.debug('Surgical intervention for mothers %d postpartum haemorrhage has been unsuccessful',
                             person_id)

    def did_not_run(self):
        person_id = self.target
        logger.debug('squeeze factor is too high for this event to run for mother %s on date %d and she is unable to '
                     'receive surgical care for her complication in labour', person_id, self.sim.date)
        return False

    def not_available(self):
        pass


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is the LabourLoggingEvent. It uses the data frame and the labour_tracker to produce summary statistics which
    are processed and presented by different analysis scripts """
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # TODO: Wont be outputting all these stats/ numbers obviously, just for debugging at the moment

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        # Denominators...
        # todo this is currently all births
        total_ip_maternal_deaths_last_year = len(df.index[df.la_maternal_death_in_labour & (
            df.la_maternal_death_in_labour_date > one_year_prior) & (df.la_maternal_death_in_labour_date <
                                                                     self.sim.date)])
        # todo: this is just to stop code crashing on small runs
        if total_ip_maternal_deaths_last_year == 0:
            total_ip_maternal_deaths_last_year = 1

        if total_births_last_year == 0:
            total_births_last_year = 1

        # yearly number of complications
        deaths = self.module.labour_tracker['maternal_death']
        still_births = self.module.labour_tracker['ip_stillbirth']
        ol = self.module.labour_tracker['obstructed_labour']
        aph = self.module.labour_tracker['antepartum_haem']
        ur = self.module.labour_tracker['uterine_rupture']
        ec = self.module.labour_tracker['eclampsia']
        ec_pp = self.module.labour_tracker['eclampsia_postpartum']
        pph = self.module.labour_tracker['postpartum_haem']
        sep = self.module.labour_tracker['sepsis']
        sep_pp = self.module.labour_tracker['sepsis_postpartum']

        # yearly number deliveries by setting
        home = self.module.labour_tracker['home_birth']
        hospital = self.module.labour_tracker['health_centre_birth']
        health_centre = self.module.labour_tracker['hospital_birth']
        cs_deliveries = self.module.labour_tracker['caesarean_section']

        ept = self.module.labour_tracker['early_preterm']
        lpt = self.module.labour_tracker['late_preterm']
        pt = self.module.labour_tracker['post_term']
        t = self.module.labour_tracker['term']

        # TODO: division by zero crashes code on small runs

        dict_for_output = {
                           'total_births_last_year': total_births_last_year,
                           'maternal_deaths_checker': deaths,
                           'maternal_deaths_df': total_ip_maternal_deaths_last_year,
                           'still_births': still_births,
                           'sbr': still_births / total_births_last_year * 100,
                           'intrapartum_mmr': total_ip_maternal_deaths_last_year / total_births_last_year * 100000,
                           'home_births_crude': home,
                           'home_births_prop': home / total_births_last_year * 100,
                           'health_centre_births': health_centre / total_births_last_year * 100,
                           'hospital_births': hospital / total_births_last_year * 100,
                           'cs_delivery_rate': cs_deliveries/total_births_last_year * 100,
                           'ol_incidence': ol / total_births_last_year * 100,
                           'aph_incidence': aph / total_births_last_year * 100,
                           'ur_incidence': ur / total_births_last_year * 100,
                           'ec_incidence': ec + ec_pp / total_births_last_year * 100,
                           'ec_incidence_pp': ec_pp / total_births_last_year * 100,
                           'sep_incidence': sep + sep_pp / total_births_last_year * 100,
                           'sep_incidence_pp': sep_pp / total_births_last_year * 100,
                           'pph_incidence': pph / total_births_last_year * 100,
                           }

        dict_crude_cases = {'intrapartum_mmr': total_ip_maternal_deaths_last_year,
                            'ol_cases': ol,
                            'aph_cases': aph,
                            'ur_cases': ur,
                            'ec_cases': ec,
                            'ec_cases_pp': ec_pp,
                            'sep_cases': sep,
                            'sep_cases_pp': sep_pp,
                            'pph_cases': pph}

        deliveries = {'ept': ept/total_births_last_year * 100,
                      'lpt': lpt/total_births_last_year * 100,
                      'term': t/total_births_last_year * 100,
                      'post_term': pt/total_births_last_year * 100}

        # deaths
        sep_d = self.module.labour_tracker['sepsis_death']
        ur_d = self.module.labour_tracker['uterine_rupture_death']
        aph_d = self.module.labour_tracker['antepartum_haem_death']
        ec_d = self.module.labour_tracker['eclampsia_death']
        pph_d = self.module.labour_tracker['postpartum_haem_death']

        deaths = {'sepsis': sep_d,
                  'uterine_rupture': ur_d,
                  'aph': aph_d,
                  'eclampsia': ec_d,
                  'postpartum_haem': pph_d,
                  }

        # TODO: SBR, health system outputs, check denominators

        logger.info('%s|summary_stats_incidence|%s', self.sim.date, dict_for_output)
        logger.info('%s|summary_stats_crude_cases|%s', self.sim.date, dict_crude_cases)
        logger.info('%s|summary_stats_deliveries|%s', self.sim.date, deliveries)
        logger.info('%s|summary_stats_death|%s', self.sim.date, deaths)

        # Reset the EventTracker
        self.module.labour_tracker = {
            'ip_stillbirth': 0, 'maternal_death': 0, 'obstructed_labour': 0,
            'antepartum_haem': 0, 'antepartum_haem_death': 0, 'sepsis': 0, 'sepsis_death': 0,
            'eclampsia': 0, 'eclampsia_death': 0, 'uterine_rupture': 0,
            'uterine_rupture_death': 0, 'postpartum_haem': 0, 'postpartum_haem_death': 0,
            'sepsis_postpartum': 0, 'eclampsia_postpartum': 0, 'home_birth': 0,
            'health_centre_birth': 0, 'hospital_birth': 0, 'caesarean_section': 0,
            'early_preterm': 0, 'late_preterm': 0, 'post_term': 0, 'term': 0
        }
